"""Training, calibration, and evaluation for the DR classifier."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from src.data import (
    _cfg,
    _save_json,
    _load_json,
    _write_alias_copy,
    _set_seed,
    _resolve_device,
    prepare_data_manifests,
    _FundusDataset,
    _manifest_path,
    _backbone_name,
    _model_image_size,
    _split_policy_tag,
    _table_path,
)


class DRClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        use_pretrained: bool = True,
        dropout: float = 0.0,
        backbone: str = "efficientnet_b4",
    ) -> None:
        super().__init__()
        if str(backbone).strip().lower() != "efficientnet_b4":
            raise ValueError(f"Unsupported backbone: {backbone}. Only efficientnet_b4 is supported.")
        p_drop = float(np.clip(dropout, 0.0, 0.9))
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.net = efficientnet_b4(weights=weights)
        in_features = int(self.net.classifier[-1].in_features)
        if p_drop > 0.0:
            self.net.classifier = nn.Sequential(nn.Dropout(p=p_drop), nn.Linear(in_features, num_classes))
        else:
            self.net.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LogitWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        # Keep backward hooks safe for attribution libraries that fail on view/in-place interactions.
        if isinstance(logits, torch.Tensor):
            return logits.clone()
        return logits


def _classification_metrics(df: pd.DataFrame, num_classes: int) -> tuple[dict[str, float], pd.DataFrame]:
    if len(df) == 0:
        empty_cm = pd.DataFrame(np.zeros((num_classes, num_classes), dtype=int))
        return {"n": 0.0}, empty_cm

    y_true = df["true_class"].astype(int).to_numpy()
    y_pred = df["pred_class"].astype(int).to_numpy()

    metrics = {
        "n": float(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }

    prob_cols = [f"prob_{i}" for i in range(num_classes)]
    if set(prob_cols).issubset(df.columns):
        y_prob = df[prob_cols].to_numpy(dtype=float)
        y_onehot = np.eye(num_classes)[y_true]
        try:
            auc = float(roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
        metrics["roc_auc_ovr_macro"] = auc

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)

    metrics["sensitivity_macro"] = float(np.mean(sensitivity))
    metrics["specificity_macro"] = float(np.mean(specificity))
    return metrics, pd.DataFrame(cm.astype(int))


def _per_class_metrics_from_confusion(cm_df: pd.DataFrame, label_order: list[str]) -> pd.DataFrame:
    cm = cm_df.to_numpy(dtype=np.float64)
    n_classes = cm.shape[0]
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0)
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    support = cm.sum(axis=1)

    rows: list[dict[str, Any]] = []
    for i in range(n_classes):
        label = label_order[i] if i < len(label_order) else str(i)
        rows.append(
            {
                "class_id": int(i),
                "class_name": label,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "sensitivity": float(recall[i]),
                "specificity": float(specificity[i]),
                "f1": float(f1[i]),
            }
        )
    return pd.DataFrame(rows)


def _build_reliability_table(df: pd.DataFrame, n_bins: int, interpolate_empty_bins: bool = False) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("Cannot build reliability table from empty dataframe")

    use = df.copy()
    use["correct"] = (use["pred_class"].astype(int) == use["true_class"].astype(int)).astype(float)
    use["confidence"] = use["confidence"].astype(float).clip(0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        low, high = float(bins[i]), float(bins[i + 1])
        if i == n_bins - 1:
            mask = (use["confidence"] >= low) & (use["confidence"] <= high)
        else:
            mask = (use["confidence"] >= low) & (use["confidence"] < high)

        sub = use[mask]
        if len(sub) == 0:
            acc = np.nan
            conf = np.nan
            gap = np.nan
        else:
            acc = float(sub["correct"].mean())
            conf = float(sub["confidence"].mean())
            gap = abs(acc - conf)

        rows.append(
            {
                "bin_index": i,
                "low": low,
                "high": high,
                "n": int(len(sub)),
                "bin_acc": float(acc) if not np.isnan(acc) else np.nan,
                "bin_conf": float(conf) if not np.isnan(conf) else np.nan,
                "calibration_gap": float(gap) if not np.isnan(gap) else np.nan,
            }
        )

    rel = pd.DataFrame(rows)
    if interpolate_empty_bins:
        if rel["calibration_gap"].notna().any():
            rel["calibration_gap"] = rel["calibration_gap"].interpolate(limit_direction="both")
        else:
            rel["calibration_gap"] = 0.0
        rel["calibration_gap"] = rel["calibration_gap"].fillna(0.0)
    return rel


def _expected_calibration_error(df: pd.DataFrame, n_bins: int) -> float:
    if len(df) == 0:
        return float("nan")
    use = df.copy()
    use["correct"] = (use["pred_class"].astype(int) == use["true_class"].astype(int)).astype(float)
    use["confidence"] = use["confidence"].astype(float).clip(0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    n_total = float(len(use))
    ece = 0.0
    for i in range(int(n_bins)):
        low, high = float(bins[i]), float(bins[i + 1])
        if i == int(n_bins) - 1:
            mask = (use["confidence"] >= low) & (use["confidence"] <= high)
        else:
            mask = (use["confidence"] >= low) & (use["confidence"] < high)
        sub = use[mask]
        if len(sub) == 0:
            continue
        acc = float(sub["correct"].mean())
        conf = float(sub["confidence"].mean())
        ece += (len(sub) / n_total) * abs(acc - conf)
    return float(ece)


def _latest_run_record_path(conf: dict[str, Any], seed: int) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"latest_run_seed{seed}.json"


def _new_run_id(conf: dict[str, Any], seed: int) -> str:
    backbone = _backbone_name(conf)
    split_policy = _split_policy_tag(conf)
    timestamp_visible = datetime.now().strftime("%Y%m%dT%H%M")
    timestamp_entropy = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    signature_payload = {
        "seed": int(seed),
        "backbone": backbone,
        "split_policy": split_policy,
        "config_signature": _checkpoint_config_signature(conf),
    }
    signature_text = json.dumps(signature_payload, sort_keys=True)
    short_hash = hashlib.sha1(f"{signature_text}|{timestamp_entropy}".encode("utf-8")).hexdigest()[:6]
    return f"dr_{backbone}_{split_policy}_seed{seed}_{timestamp_visible}_{short_hash}"


def _run_id_from_checkpoint_path(conf: dict[str, Any], seed: int, checkpoint_path: str | Path) -> str:
    ckpt = Path(checkpoint_path)
    stem = ckpt.stem
    if stem.startswith("dr_"):
        return stem
    backbone = _backbone_name(conf)
    split_policy = _split_policy_tag(conf)
    if ckpt.exists():
        ts = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y%m%dT%H%M")
    else:
        ts = datetime.now().strftime("%Y%m%dT%H%M")
    short_hash = hashlib.sha1(f"{stem}|{seed}|{backbone}|{split_policy}".encode("utf-8")).hexdigest()[:6]
    return f"dr_{backbone}_{split_policy}_seed{seed}_{ts}_{short_hash}"


def _checkpoint_path_for_run_id(conf: dict[str, Any], run_id: str) -> Path:
    return Path(conf["paths"]["checkpoints_dir"]) / f"{run_id}.pt"


def _calibration_path_for_run_id(conf: dict[str, Any], run_id: str) -> Path:
    return Path(conf["paths"]["checkpoints_dir"]) / f"{run_id}_calibration.json"


def _predictions_path_for_run_id(conf: dict[str, Any], run_id: str, split: str) -> Path:
    return Path(conf["paths"]["predictions_dir"]) / f"{run_id}_{split}_predictions.csv"


def _train_history_log_path(conf: dict[str, Any], run_id: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"{run_id}_train_history.json"


def _gradcam_status_log_path(conf: dict[str, Any], run_id: str, split: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"{run_id}_{split}_gradcam_status.json"


def _shap_status_log_path(conf: dict[str, Any], run_id: str, split: str) -> Path:
    return Path(conf["paths"]["logs_dir"]) / f"{run_id}_{split}_shap_status.json"


def _save_latest_run_record(
    conf: dict[str, Any],
    seed: int,
    run_id: str,
    checkpoint_path: str | Path,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    base_payload = {
        "seed": int(seed),
        "run_id": run_id,
        "backbone": _backbone_name(conf),
        "split_policy": _split_policy_tag(conf),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    existing_payload: dict[str, Any] = {}
    latest_path = _latest_run_record_path(conf, seed)
    if latest_path.exists():
        try:
            loaded = _load_json(latest_path)
            if isinstance(loaded, dict):
                existing_payload = loaded
        except Exception:
            existing_payload = {}

    if str(existing_payload.get("run_id", "")) == str(run_id):
        payload = dict(existing_payload)
        payload.update(base_payload)
    else:
        payload = base_payload

    if extra_fields:
        payload.update(extra_fields)
    _save_json(latest_path, payload)


def _load_latest_run_record(conf: dict[str, Any], seed: int) -> dict[str, Any] | None:
    p = _latest_run_record_path(conf, seed)
    if not p.exists():
        return None
    try:
        payload = _load_json(p)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    ckpt = payload.get("checkpoint_path")
    run_id = payload.get("run_id")
    if not ckpt or not run_id:
        return None
    return payload


def _latest_named_checkpoint(conf: dict[str, Any], seed: int) -> Path | None:
    ckpt_dir = Path(conf["paths"]["checkpoints_dir"])
    pattern = f"dr_{_backbone_name(conf)}_{_split_policy_tag(conf)}_seed{seed}_*.pt"
    matches = [p for p in ckpt_dir.glob(pattern) if p.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _resolve_checkpoint_and_run_id(
    conf: dict[str, Any],
    seed: int,
    checkpoint: str | Path | None = None,
    require_existing: bool = True,
) -> tuple[Path, str]:
    if checkpoint is not None:
        ckpt = Path(checkpoint).expanduser().resolve()
        if require_existing and not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        run_id = _run_id_from_checkpoint_path(conf, seed, ckpt)
        _save_latest_run_record(conf, seed, run_id, ckpt)
        return ckpt, run_id

    record = _load_latest_run_record(conf, seed)
    if record is not None:
        ckpt = Path(str(record["checkpoint_path"]))
        run_id = str(record["run_id"])
        if ckpt.exists():
            if not ckpt.stem.startswith("dr_"):
                migrated_ckpt = _checkpoint_path_for_run_id(conf, run_id)
                _write_alias_copy(ckpt, migrated_ckpt)
                _save_latest_run_record(conf, seed, run_id, migrated_ckpt)
                return migrated_ckpt, run_id
            return ckpt, run_id

    ckpt = _latest_named_checkpoint(conf, seed)
    if ckpt is not None:
        run_id = _run_id_from_checkpoint_path(conf, seed, ckpt)
        _save_latest_run_record(conf, seed, run_id, ckpt)
        return ckpt, run_id

    if require_existing:
        raise FileNotFoundError(
            f"No checkpoint found for seed={seed} under {Path(conf['paths']['checkpoints_dir'])}"
        )

    run_id = _new_run_id(conf, seed)
    return _checkpoint_path_for_run_id(conf, run_id), run_id


def _signature_subset_match(saved: Any, query: Any) -> bool:
    """Return True if every key/value in query is present in saved with matching value.
    Compares dicts recursively so a slimmer current signature still matches an older,
    fuller saved signature provided every key in current is in saved with the same value.
    """
    if isinstance(query, dict):
        if not isinstance(saved, dict):
            return False
        return all(k in saved and _signature_subset_match(saved[k], v) for k, v in query.items())
    return saved == query


def _find_matching_checkpoint_by_signature(
    conf: dict[str, Any],
    seed: int,
    config_signature: dict[str, Any],
) -> tuple[Path, str] | None:
    candidates: list[Path] = []
    latest_named = _latest_named_checkpoint(conf, seed)
    if latest_named is not None:
        ckpt_dir = Path(conf["paths"]["checkpoints_dir"])
        pattern = f"dr_{_backbone_name(conf)}_{_split_policy_tag(conf)}_seed{seed}_*.pt"
        named = [p for p in ckpt_dir.glob(pattern) if p.is_file()]
        named.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(named)

    seen: set[str] = set()
    for ckpt in candidates:
        resolved = str(ckpt.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            payload = torch.load(ckpt, map_location="cpu")
            saved_sig = payload.get("config_signature")
        except Exception:
            continue
        if _signature_subset_match(saved_sig, config_signature):
            run_id = _run_id_from_checkpoint_path(conf, seed, ckpt)
            return ckpt, run_id
    return None


def _run_id_from_predictions_path(predictions_path: str | Path, split: str) -> str:
    stem = Path(predictions_path).stem
    suffix = f"_{split}_predictions"
    if stem.startswith("dr_") and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return ""


_USED_AUG_KEYS = (
    "profile_horizontal_flip",
    "profile_vertical_flip",
    "profile_rotation_degrees",
    "profile_translate",
    "profile_scale_min",
    "profile_scale_max",
    "profile_shear_degrees",
    "profile_brightness",
    "profile_contrast",
)


def _checkpoint_config_signature(conf: dict[str, Any]) -> dict[str, Any]:
    aug = conf.get("augmentation", {})
    return {
        "data_protocol": "benchmark",
        "data_source": str(conf.get("data", {}).get("source", "aptos_only")),
        "profile_test_ratio": float(conf.get("data", {}).get("profile_test_ratio", 0.15)),
        "profile_val_ratio_within_train": float(conf.get("data", {}).get("profile_val_ratio_within_train", 0.10)),
        "backbone": _backbone_name(conf),
        "image_size": int(conf["data"]["image_size"]),
        "effective_image_size": _model_image_size(conf),
        "num_classes": int(conf["data"]["num_classes"]),
        "use_pretrained": bool(conf["training"].get("use_pretrained", True)),
        "dropout": float(conf["training"].get("dropout", 0.0)),
        "label_smoothing": float(conf["training"].get("label_smoothing", 0.0)),
        "optimizer_name": str(conf["training"].get("optimizer_name", "adamw")),
        "lr": float(conf["training"]["lr"]),
        "weight_decay": float(conf["training"]["weight_decay"]),
        "use_class_weights": bool(conf["training"].get("use_class_weights", True)),
        "loss_name": str(conf["training"].get("loss_name", "focal")),
        "focal_gamma": float(conf["training"].get("focal_gamma", 2.0)),
        "grad_accum_steps": int(conf["training"].get("grad_accum_steps", 1)),
        "scheduler_name": str(conf["training"].get("scheduler_name", "cosine_annealing")),
        "use_scheduler": bool(conf["training"].get("use_scheduler", True)),
        "scheduler_min_lr": float(conf["training"].get("scheduler_min_lr", 1e-8)),
        "augmentation": {k: aug[k] for k in _USED_AUG_KEYS if k in aug},
        "preprocessing": dict(conf.get("preprocessing", {})),
    }


def _class_weights(class_ids: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.array(class_ids), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    return loss.mean()


def _predict_manifest(
    model: DRClassifier,
    manifest_path: str | Path,
    conf: dict[str, Any],
    split: str,
    device: torch.device,
    temperature: float = 1.0,
) -> pd.DataFrame:
    image_size = _model_image_size(conf)
    ds = _FundusDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=False,
    )
    cfg_workers = int(conf["training"].get("num_workers", 0))
    loader_workers = 0 if getattr(device, "type", "") == "mps" else cfg_workers
    loader = DataLoader(
        ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=False,
    )

    model.eval()
    rows: list[dict[str, Any]] = []
    temp = max(1e-4, float(temperature))

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy().tolist()
            sample_ids = list(batch["sample_id"])
            laterality = list(batch["laterality"])
            image_paths = list(batch["image_path"])

            logits = model(images)
            logits = logits / temp
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            probs_np = probs.detach().cpu().numpy()
            conf_np = confs.detach().cpu().numpy()
            pred_np = preds.detach().cpu().numpy()

            for i, sid in enumerate(sample_ids):
                row = {
                    "sample_id": sid,
                    "true_class": int(labels[i]),
                    "pred_class": int(pred_np[i]),
                    "confidence": float(conf_np[i]),
                    "laterality": laterality[i],
                    "split": split,
                    "image_path": image_paths[i],
                }
                for c in range(probs_np.shape[1]):
                    row[f"prob_{c}"] = float(probs_np[i, c])
                rows.append(row)

    return pd.DataFrame(rows)


def _collect_logits_and_labels(
    model: DRClassifier,
    manifest_path: str | Path,
    conf: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = _model_image_size(conf)
    ds = _FundusDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=False,
    )
    cfg_workers = int(conf["training"].get("num_workers", 0))
    loader_workers = 0 if getattr(device, "type", "") == "mps" else cfg_workers
    loader = DataLoader(
        ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=False,
    )

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            logits_list.append(logits.detach().cpu())
            labels_list.append(y.detach().cpu())

    if not logits_list:
        return torch.empty((0, int(conf["data"]["num_classes"])), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] == 0:
        return 1.0

    logits_f = logits.float()
    labels_f = labels.long()
    criterion = torch.nn.CrossEntropyLoss()
    temperature = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def _closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        t = torch.clamp(temperature, min=0.5, max=10.0)
        loss = criterion(logits_f / t, labels_f)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(torch.clamp(temperature.detach(), min=0.5, max=10.0).item())


def _load_model(
    conf: dict[str, Any],
    seed: int,
    device: torch.device,
    checkpoint: str | Path | None = None,
) -> DRClassifier:
    ckpt, _ = _resolve_checkpoint_and_run_id(
        conf,
        seed=seed,
        checkpoint=checkpoint,
        require_existing=True,
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = DRClassifier(
        num_classes=int(conf["data"]["num_classes"]),
        use_pretrained=False,
        dropout=float(conf["training"].get("dropout", 0.0)),
        backbone=_backbone_name(conf),
    ).to(device)

    payload = torch.load(ckpt, map_location=device)
    state_dict = dict(payload["state_dict"])
    target_state = model.state_dict()

    for plain, dropout in [("net.classifier.weight", "net.classifier.1.weight"),
                           ("net.classifier.bias",   "net.classifier.1.bias")]:
        if plain in state_dict and dropout in target_state:
            state_dict[dropout] = state_dict.pop(plain)
        elif dropout in state_dict and plain in target_state:
            state_dict[plain] = state_dict.pop(dropout)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def train_dr_classifier(
    cfg: str | Path | dict[str, Any],
    seed: int = 1988,
    manifests: dict[str, str] | None = None,
    reuse_if_exists: bool = True,
) -> str:
    conf = _cfg(cfg)
    _set_seed(int(seed))
    current_sig = _checkpoint_config_signature(conf)

    if reuse_if_exists:
        matched = _find_matching_checkpoint_by_signature(conf, seed=seed, config_signature=current_sig)
        if matched is not None:
            existing_ckpt, existing_run_id = matched
            if not existing_ckpt.stem.startswith("dr_"):
                migrated_ckpt = _checkpoint_path_for_run_id(conf, existing_run_id)
                _write_alias_copy(existing_ckpt, migrated_ckpt)
                existing_ckpt = migrated_ckpt
            _save_latest_run_record(conf, seed=seed, run_id=existing_run_id, checkpoint_path=existing_ckpt)
            print(f"Reusing existing checkpoint: {existing_ckpt}")
            return str(existing_ckpt)

    # No exact signature match: always start a fresh run ID/path to avoid
    # overwriting a previous checkpoint from a different config signature.
    run_id = _new_run_id(conf, seed=seed)
    ckpt = _checkpoint_path_for_run_id(conf, run_id)
    train_started_at = datetime.now().isoformat(timespec="seconds")
    train_start_perf = time.perf_counter()
    _save_latest_run_record(
        conf,
        seed=seed,
        run_id=run_id,
        checkpoint_path=ckpt,
        extra_fields={
            "status": "training_started",
            "train_started_at": train_started_at,
        },
    )

    if manifests is None:
        train_manifest = _manifest_path(conf, "train", seed=seed)
        val_manifest = _manifest_path(conf, "val", seed=seed)
        if not train_manifest.exists() or not val_manifest.exists():
            manifests = prepare_data_manifests(conf, seed=seed)
            train_manifest = Path(manifests["train_split"])
            val_manifest = Path(manifests["val_split"])
    else:
        train_manifest = Path(manifests["train_split"])
        val_manifest = Path(manifests["val_split"])

    device = _resolve_device(conf["training"].get("device", "mps"))
    image_size = _model_image_size(conf)

    train_df = pd.read_csv(train_manifest)
    class_w = _class_weights(
        class_ids=train_df["class_id"].astype(int).tolist(),
        num_classes=int(conf["data"]["num_classes"]),
        device=device,
    )

    train_ds = _FundusDataset(
        manifest_path=train_manifest,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=True,
    )
    val_ds = _FundusDataset(
        manifest_path=val_manifest,
        image_size=image_size,
        aug_cfg=conf.get("augmentation", {}),
        preprocessing_cfg=conf.get("preprocessing", {}),
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(conf["training"].get("num_workers", 0)),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(conf["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(conf["training"].get("num_workers", 0)),
        pin_memory=False,
    )

    model = DRClassifier(
        num_classes=int(conf["data"]["num_classes"]),
        use_pretrained=bool(conf["training"].get("use_pretrained", True)),
        dropout=float(conf["training"].get("dropout", 0.0)),
        backbone=_backbone_name(conf),
    ).to(device)

    base_lr = float(conf["training"]["lr"])
    wd = float(conf["training"]["weight_decay"])
    optimizer_name = str(conf["training"].get("optimizer_name", "adamw")).strip().lower()
    if optimizer_name not in {"adamw", "adam"}:
        raise ValueError(f"Unsupported optimizer_name={optimizer_name}. Use 'adamw' or 'adam'.")
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=base_lr, weight_decay=wd)

    use_class_weights = bool(conf["training"].get("use_class_weights", True))
    class_weights_for_loss = class_w if use_class_weights else None
    label_smoothing = float(conf["training"].get("label_smoothing", 0.0))
    loss_name = str(conf["training"].get("loss_name", "focal")).strip().lower()
    use_focal_loss = (loss_name == "focal")
    focal_gamma = float(conf["training"].get("focal_gamma", 2.0))
    grad_accum_steps = int(max(1, conf["training"].get("grad_accum_steps", 1)))
    max_epochs = int(conf["training"]["epochs"])
    patience_limit = int(conf["training"]["early_stopping_patience"])

    def _loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if use_focal_loss:
            return _focal_loss(
                logits=logits,
                targets=targets,
                gamma=focal_gamma,
                class_weights=class_weights_for_loss,
                label_smoothing=label_smoothing,
            )
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights_for_loss,
            label_smoothing=label_smoothing,
        )

    use_scheduler = bool(conf["training"].get("use_scheduler", True))
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs),
            eta_min=float(conf["training"].get("scheduler_min_lr", 1e-8)),
        )

    best_f1 = -1.0
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "lr": [],
        "lr_groups": [],
        "epoch_duration_seconds": [],
        "lr_group_names": [f"group_{i}" for i in range(len(optimizer.param_groups))],
        "optimizer_name": optimizer_name,
        "scheduler_name": "cosine_annealing" if use_scheduler else "none",
        "loss_name": "focal" if use_focal_loss else "cross_entropy",
        "grad_accum_steps": int(grad_accum_steps),
        "train_started_at": train_started_at,
    }

    log_every_batches = max(0, int(conf["training"].get("log_every_batches", 100)))

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)
        num_train_batches = max(1, len(train_loader))
        epoch_start = time.perf_counter()

        if log_every_batches > 0:
            print(
                f"seed={seed} epoch={epoch:02d}/{max_epochs} started "
                f"batches={num_train_batches}",
                flush=True,
            )

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            logits = model(x)
            loss = _loss_fn(logits, y)
            (loss / grad_accum_steps).backward()
            if (batch_idx % grad_accum_steps == 0) or (batch_idx == num_train_batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_sum += loss.item() * x.size(0)
            train_count += int(x.size(0))

            if log_every_batches > 0 and (
                batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == num_train_batches
            ):
                elapsed_s = max(1e-6, time.perf_counter() - epoch_start)
                batches_per_s = batch_idx / elapsed_s
                eta_s = max(0.0, (num_train_batches - batch_idx) / max(1e-6, batches_per_s))
                running_train_loss = train_sum / max(1, train_count)
                print(
                    f"seed={seed} epoch={epoch:02d} "
                    f"batch={batch_idx:04d}/{num_train_batches} "
                    f"train_loss_running={running_train_loss:.4f} "
                    f"elapsed={elapsed_s/60.0:.1f}m eta={eta_s/60.0:.1f}m",
                    flush=True,
                )

        train_loss = train_sum / max(1, train_count)

        model.eval()
        val_sum = 0.0
        val_count = 0
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                loss = _loss_fn(logits, y)
                val_sum += loss.item() * x.size(0)
                val_count += int(x.size(0))

                pred = torch.argmax(logits, dim=1)
                y_true.extend(y.detach().cpu().numpy().tolist())
                y_pred.extend(pred.detach().cpu().numpy().tolist())

        val_loss = val_sum / max(1, val_count)
        val_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if scheduler is not None:
            scheduler.step()
        current_lr = float(max(pg["lr"] for pg in optimizer.param_groups))

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_macro_f1"].append(float(val_f1))
        history["lr"].append(current_lr)
        history["lr_groups"].append([float(pg["lr"]) for pg in optimizer.param_groups])
        epoch_elapsed_s = max(1e-6, time.perf_counter() - epoch_start)
        history["epoch_duration_seconds"].append(float(epoch_elapsed_s))

        print(
            f"seed={seed} epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} lr={current_lr:.6f} "
            f"epoch_time={epoch_elapsed_s/60.0:.1f}m",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": int(seed),
                    "run_id": run_id,
                    "best_val_macro_f1": float(best_f1),
                    "config_signature": _checkpoint_config_signature(conf),
                },
                ckpt,
            )
        else:
            wait += 1

        if wait >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break

    if not ckpt.exists():
        raise RuntimeError(f"Training completed but no checkpoint was saved: {ckpt}")

    train_finished_at = datetime.now().isoformat(timespec="seconds")
    train_duration_seconds = max(1e-6, time.perf_counter() - train_start_perf)
    finish_fields = {
        "train_finished_at": train_finished_at,
        "train_duration_seconds": float(train_duration_seconds),
        "train_duration_minutes": float(train_duration_seconds / 60.0),
        "epochs_ran": int(len(history["train_loss"])),
        "best_val_macro_f1": float(best_f1) if np.isfinite(best_f1) else None,
    }
    history.update(finish_fields)

    train_history_path = _train_history_log_path(conf, run_id)
    _save_json(train_history_path, history)
    _save_latest_run_record(
        conf, seed=seed, run_id=run_id, checkpoint_path=ckpt,
        extra_fields={
            "status": "trained",
            "train_started_at": train_started_at,
            **finish_fields,
            "train_history_path": str(train_history_path.resolve()),
        },
    )

    return str(ckpt)


def build_validation_calibration_table(cfg: str | Path | dict[str, Any], seed: int = 1988, manifests: dict[str, str] | None = None, checkpoint: str | Path | None = None) -> str:
    conf = _cfg(cfg)
    calib_cfg = conf.get("calibration", {})
    n_bins = int(calib_cfg.get("n_bins", 15))
    temperature_max_iter = int(calib_cfg.get("temperature_max_iter", 50))
    interpolate_empty_bins = bool(calib_cfg.get("interpolate_empty_bins", False))

    if manifests is None:
        val_manifest = _manifest_path(conf, "val", seed=seed)
        if not val_manifest.exists():
            manifests = prepare_data_manifests(conf, seed=seed)
            val_manifest = Path(manifests["val_split"])
    else:
        val_manifest = Path(manifests["val_split"])

    device = _resolve_device(conf["training"].get("device", "mps"))

    if checkpoint is None:
        try:
            ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
        except FileNotFoundError:
            trained_ckpt = train_dr_classifier(conf, seed=seed, manifests=manifests)
            ckpt_path, run_id = _resolve_checkpoint_and_run_id(
                conf,
                seed=seed,
                checkpoint=trained_ckpt,
                require_existing=True,
            )
    else:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(
            conf,
            seed=seed,
            checkpoint=checkpoint,
            require_existing=True,
        )

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    val_logits, val_labels = _collect_logits_and_labels(model, val_manifest, conf, device=device)
    temperature = _fit_temperature(
        logits=val_logits,
        labels=val_labels,
        max_iter=temperature_max_iter,
    )
    val_df_uncal = _predict_manifest(
        model,
        val_manifest,
        conf,
        split="val",
        device=device,
        temperature=1.0,
    )
    val_df = _predict_manifest(
        model,
        val_manifest,
        conf,
        split="val",
        device=device,
        temperature=temperature,
    )
    val_predictions_path = _predictions_path_for_run_id(conf, run_id, "val")
    val_df.to_csv(val_predictions_path, index=False)

    rel = _build_reliability_table(val_df, n_bins=n_bins, interpolate_empty_bins=interpolate_empty_bins)
    rel_path = _table_path(conf, "calibration_bins", seed=seed)
    rel.to_csv(rel_path, index=False)
    ece_before = _expected_calibration_error(val_df_uncal, n_bins=n_bins)
    ece_after = _expected_calibration_error(val_df, n_bins=n_bins)
    calib_summary_path = _table_path(conf, "calibration_summary", seed=seed)
    pd.DataFrame(
        [
            {
                "seed": int(seed),
                "run_id": run_id,
                "n_bins": int(n_bins),
                "interpolate_empty_bins": int(interpolate_empty_bins),
                "temperature": float(temperature),
                "ece_before": float(ece_before),
                "ece_after": float(ece_after),
                "ece_delta": float(ece_after - ece_before) if (not np.isnan(ece_before) and not np.isnan(ece_after)) else float("nan"),
            }
        ]
    ).to_csv(calib_summary_path, index=False)

    payload = {
        "seed": int(seed),
        "run_id": run_id,
        "n_bins": n_bins,
        "temperature": float(temperature),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "rows": rel.to_dict(orient="records"),
        "table_path": str(rel_path),
        "calibration_bins_path": str(rel_path),
        "calibration_summary_path": str(calib_summary_path),
        "checkpoint_path": str(Path(ckpt_path).resolve()),
    }
    out = _calibration_path_for_run_id(conf, run_id)
    _save_json(out, payload)
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)
    return str(out)


def run_split_inference(cfg: str | Path | dict[str, Any], seed: int = 1988, split: str = "test") -> str:
    conf = _cfg(cfg)
    manifest = _manifest_path(conf, split, seed=seed)
    if not manifest.exists():
        prepare_data_manifests(conf, seed=seed)

    device = _resolve_device(conf["training"].get("device", "mps"))

    try:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    except FileNotFoundError:
        trained_ckpt = train_dr_classifier(conf, seed=seed)
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(
            conf,
            seed=seed,
            checkpoint=trained_ckpt,
            require_existing=True,
        )

    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        build_validation_calibration_table(conf, seed=seed, checkpoint=ckpt_path)
        calibration_path = _calibration_path_for_run_id(conf, run_id)

    calibration_payload = _load_json(calibration_path)
    temperature = float(calibration_payload.get("temperature", 1.0))
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    pred_df = _predict_manifest(
        model,
        manifest,
        conf,
        split=split,
        device=device,
        temperature=temperature,
    )

    ordered_cols = [
        "sample_id",
        "true_class",
        "pred_class",
        "confidence",
        "laterality",
        "split",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
        "image_path",
    ]
    pred_df = pred_df[ordered_cols]
    out = _predictions_path_for_run_id(conf, run_id, split)
    pred_df.to_csv(out, index=False)
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)
    return str(out)


def evaluate_pipeline_outputs(cfg: str | Path | dict[str, Any], seed: int = 1988, split: str = "test", predictions_csv: str | Path | None = None) -> dict[str, str]:
    conf = _cfg(cfg)

    if predictions_csv is None:
        pred_path = Path(run_split_inference(conf, seed=seed, split=split))
    else:
        pred_path = Path(predictions_csv)

    df = pd.read_csv(pred_path)
    num_classes = int(conf["data"]["num_classes"])
    high_conf_thr = float(conf["evaluation"].get("high_conf_threshold", 0.80))
    run_id = _run_id_from_predictions_path(pred_path, split=split)
    if not run_id:
        try:
            _, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
        except FileNotFoundError:
            run_id = ""

    m_all, cm_all = _classification_metrics(df, num_classes=num_classes)
    m_high, _ = _classification_metrics(df[df["confidence"] >= high_conf_thr], num_classes=num_classes)

    metrics_rows = [
        {"run_id": run_id, "scope": "all", **m_all},
        {"run_id": run_id, "scope": "high_conf", **m_high},
    ]

    metrics_path = _table_path(conf, "metrics", seed=seed, split=split)
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    headline_path = Path(export_final_headline_metrics(conf, seed=seed, split=split, metrics_csv=metrics_path))

    cm_path = _table_path(conf, "confusion", seed=seed, split=split)
    cm_all.to_csv(cm_path, index=False)

    per_class_path = _table_path(conf, "per_class", seed=seed, split=split)
    label_order = [str(x) for x in conf["data"].get("label_order", [])]
    per_class_df = _per_class_metrics_from_confusion(cm_all, label_order=label_order)
    per_class_df.to_csv(per_class_path, index=False)

    run_summary = {
        "seed": int(seed),
        "run_id": run_id,
        "split": split,
        "n_all": int(m_all.get("n", 0.0)),
        "accuracy_all": float(m_all.get("accuracy", np.nan)),
        "qwk_all": float(m_all.get("qwk", np.nan)),
        "recall_macro_all": float(m_all.get("recall_macro", np.nan)),
        "sensitivity_macro_all": float(m_all.get("sensitivity_macro", np.nan)),
        "specificity_macro_all": float(m_all.get("specificity_macro", np.nan)),
        "f1_macro_all": float(m_all.get("f1_macro", np.nan)),
        "n_high_conf": int(m_high.get("n", 0.0)),
        "accuracy_high_conf": float(m_high.get("accuracy", np.nan)),
        "qwk_high_conf": float(m_high.get("qwk", np.nan)),
        "recall_macro_high_conf": float(m_high.get("recall_macro", np.nan)),
        "sensitivity_macro_high_conf": float(m_high.get("sensitivity_macro", np.nan)),
        "specificity_macro_high_conf": float(m_high.get("specificity_macro", np.nan)),
        "f1_macro_high_conf": float(m_high.get("f1_macro", np.nan)),
    }
    run_summary_path = _table_path(conf, "run_summary", seed=seed, split=split)
    pd.DataFrame([run_summary]).to_csv(run_summary_path, index=False)

    return {
        "metrics": str(metrics_path),
        "final_headline_metrics": str(headline_path),
        "confusion": str(cm_path),
        "per_class": str(per_class_path),
        "run_summary": str(run_summary_path),
    }


def export_final_headline_metrics(
    cfg: str | Path | dict[str, Any] = "configs/base.yaml",
    seed: int = 1988,
    split: str = "test",
    metrics_csv: str | Path | None = None,
) -> str:
    conf = _cfg(cfg)
    if metrics_csv is None:
        metrics_path = _table_path(conf, "metrics", seed=seed, split=split)
    else:
        metrics_path = Path(metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}. Run evaluate_pipeline_outputs first.")

    metrics_df = pd.read_csv(metrics_path)
    required = ["run_id", "scope", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing required metrics columns: {missing}")

    all_df = metrics_df[metrics_df["scope"].astype(str).str.lower() == "all"].copy()
    if len(all_df) == 0:
        raise ValueError("No row found with scope='all' in metrics table.")

    row = all_df.iloc[[0]][["run_id", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].copy()
    out_path = _table_path(conf, "final_headline_metrics", seed=seed, split=split)
    row.to_csv(out_path, index=False)
    return str(out_path)


def notebook_run_training(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    manifests: dict[str, str] | None = None,
    force_retrain: bool = False,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    reuse_if_exists = not bool(force_retrain)
    current_sig = _checkpoint_config_signature(conf)
    pre_match = _find_matching_checkpoint_by_signature(conf, seed=int(seed), config_signature=current_sig) if reuse_if_exists else None
    had_any_checkpoint = _latest_named_checkpoint(conf, int(seed)) is not None

    checkpoint_path = train_dr_classifier(
        conf,
        seed=int(seed),
        manifests=manifests,
        reuse_if_exists=reuse_if_exists,
    )

    latest_path = _latest_run_record_path(conf, int(seed))
    latest_payload = _load_json(latest_path) if latest_path.exists() else {}
    run_id = str(latest_payload.get("run_id", Path(checkpoint_path).stem))

    history_path = _train_history_log_path(conf, run_id)
    history = _load_json(history_path) if history_path.exists() else {}

    history_fig: plt.Figure | None = None
    train_loss = list(history.get("train_loss", []))
    val_loss = list(history.get("val_loss", []))
    val_macro_f1 = list(history.get("val_macro_f1", []))
    if len(train_loss) > 0 and len(val_loss) > 0:
        epochs = np.arange(1, min(len(train_loss), len(val_loss)) + 1)
        history_fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss[: len(epochs)], marker="o", label="train_loss")
        axes[0].plot(epochs, val_loss[: len(epochs)], marker="o", label="val_loss")
        axes[0].set_title("Training vs Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        f1_epochs = np.arange(1, len(val_macro_f1) + 1)
        axes[1].plot(f1_epochs, val_macro_f1, marker="o", color="tab:green")
        axes[1].set_title("Validation Macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        history_fig.tight_layout()

    reused_checkpoint = bool(pre_match is not None and Path(pre_match[0]).resolve() == Path(checkpoint_path).resolve())
    if force_retrain:
        reuse_reason = "force_retrain"
    elif reused_checkpoint:
        reuse_reason = "matched_signature"
    elif had_any_checkpoint:
        reuse_reason = "signature_mismatch"
    else:
        reuse_reason = "no_checkpoint_found"

    return {
        "cfg": conf,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "run_id": run_id,
        "latest_run_record_path": str(latest_path),
        "latest_run_record": latest_payload,
        "history_path": str(history_path),
        "history": history,
        "history_fig": history_fig,
        "reused_checkpoint": bool(reused_checkpoint),
        "reuse_reason": str(reuse_reason),
    }


def notebook_run_core_evaluation(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    predictions_csv: str | Path | None = None,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    split_key = str(split).strip().lower() or "test"
    predictions_path = str(predictions_csv) if predictions_csv else run_split_inference(conf, seed=int(seed), split=split_key)

    eval_outputs = evaluate_pipeline_outputs(
        conf,
        seed=int(seed),
        split=split_key,
        predictions_csv=predictions_path,
    )

    headline_path = Path(eval_outputs["final_headline_metrics"])
    metrics_path = Path(eval_outputs["metrics"])
    summary_path = Path(eval_outputs["run_summary"])
    per_class_path = Path(eval_outputs["per_class"])
    confusion_path = Path(eval_outputs["confusion"])

    headline_df = pd.read_csv(headline_path)
    metrics_df = pd.read_csv(metrics_path)
    summary_df = pd.read_csv(summary_path)
    per_class_df = pd.read_csv(per_class_path)
    confusion_df = pd.read_csv(confusion_path)

    headline_cols = [
        c
        for c in ["run_id", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
        if c in headline_df.columns
    ]
    summary_cols = [
        c
        for c in ["split", "n_all", "qwk_all", "n_high_conf", "accuracy_high_conf", "qwk_high_conf"]
        if c in summary_df.columns
    ]
    overall_summary_df = pd.concat(
        [
            headline_df[headline_cols].reset_index(drop=True),
            summary_df[summary_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    output_paths_df = pd.DataFrame([eval_outputs]).T.rename(columns={0: "path"})

    headline_fig: plt.Figure | None = None
    if len(headline_df):
        vals = headline_df.iloc[0]
        metric_labels = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]
        metric_values = [
            float(vals["accuracy"]),
            float(vals["precision_macro"]),
            float(vals["recall_macro"]),
            float(vals["f1_macro"]),
        ]
        headline_fig, ax = plt.subplots(figsize=(7, 3.8))
        bars = ax.bar(metric_labels, metric_values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Headline Metrics")
        ax.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, metric_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value) + 0.01,
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        headline_fig.tight_layout()

    confusion_fig: plt.Figure | None = None
    cm = confusion_df.to_numpy()
    if cm.ndim == 2 and cm.size > 0:
        label_order = list(conf["data"]["label_order"])
        confusion_fig, ax = plt.subplots(figsize=(6, 5))
        heat = ax.imshow(cm, cmap="Blues")
        confusion_fig.colorbar(heat, ax=ax)
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels(label_order, rotation=25)
        ax.set_yticks(range(len(label_order)))
        ax.set_yticklabels(label_order)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        confusion_fig.tight_layout()

    return {
        "cfg": conf,
        "seed": int(seed),
        "split": split_key,
        "predictions_path": str(predictions_path),
        "eval_outputs": eval_outputs,
        "eval_output_paths_df": output_paths_df,
        "headline_df": headline_df,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "overall_summary_df": overall_summary_df,
        "per_class_df": per_class_df,
        "confusion_df": confusion_df,
        "headline_fig": headline_fig,
        "confusion_fig": confusion_fig,
    }


def clean_generated_outputs(cfg_path: str | Path = "configs/base.yaml") -> None:
    conf = _cfg(cfg_path)
    targets = [
        Path(conf["paths"]["manifests_dir"]),
        Path(conf["paths"]["checkpoints_dir"]),
        Path(conf["paths"]["predictions_dir"]),
        Path(conf["paths"]["logs_dir"]),
        Path(conf["paths"]["figures_dir"]),
        Path(conf["paths"]["tables_dir"]),
    ]

    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    project_root = Path(conf["project_root"])
    for cache_path in project_root.rglob("__pycache__"):
        if cache_path.is_dir():
            shutil.rmtree(cache_path, ignore_errors=True)

    pytest_cache = project_root / ".pytest_cache"
    if pytest_cache.exists() and pytest_cache.is_dir():
        shutil.rmtree(pytest_cache, ignore_errors=True)
