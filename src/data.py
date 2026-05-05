"""APTOS 2019 data loading, manifest creation, fundus preprocessing, and overview figures."""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import yaml


_REQUIRED_PATH_KEYS = [
    "dataset_root",
    "aptos_root",
    "aptos_train_csv",
    "aptos_valid_csv",
    "aptos_test_csv",
    "aptos_train_image_dir",
    "aptos_valid_image_dir",
    "aptos_test_image_dir",
    "artifacts_root",
    "manifests_dir",
    "checkpoints_dir",
    "predictions_dir",
    "logs_dir",
    "reports_root",
    "figures_dir",
    "tables_dir",
]


def _infer_project_root(cfg_path: Path) -> Path:
    if cfg_path.parent.name == "configs":
        return cfg_path.parent.parent.resolve()
    return Path.cwd().resolve()


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")

    project_root = _infer_project_root(cfg_path)

    paths = dict(cfg.get("paths", {}))
    missing = [k for k in _REQUIRED_PATH_KEYS if k not in paths]
    if missing:
        raise ValueError(f"Missing required config paths: {missing}")

    for key, value in paths.items():
        p = Path(str(value))
        if not p.is_absolute():
            p = (project_root / p).resolve()
        paths[key] = str(p)

    cfg["paths"] = paths
    cfg["project_root"] = str(project_root)

    for key in [
        "dataset_root",
        "aptos_root",
        "aptos_train_image_dir",
        "aptos_valid_image_dir",
        "aptos_test_image_dir",
        "artifacts_root",
        "manifests_dir",
        "checkpoints_dir",
        "predictions_dir",
        "logs_dir",
        "reports_root",
        "figures_dir",
        "tables_dir",
    ]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    return cfg


def _cfg(cfg_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg_or_path, dict):
        cfg = dict(cfg_or_path)
        if "project_root" not in cfg:
            raise ValueError("Config dict must contain project_root. Use load_project_config(path).")
        for key in [
            "manifests_dir",
            "checkpoints_dir",
            "predictions_dir",
            "logs_dir",
            "figures_dir",
            "tables_dir",
        ]:
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
        return cfg
    return load_project_config(cfg_or_path)


def _save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_alias_copy(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    if not src_path.exists():
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() == dst_path.resolve():
        return
    shutil.copy2(src_path, dst_path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _resolve_device(requested: str = "mps") -> torch.device:
    req = str(requested).lower().strip()
    mps_available = torch.backends.mps.is_available()

    if req in {"", "auto", "default"}:
        if mps_available:
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if req == "mps":
        if mps_available:
            return torch.device("mps")
        raise RuntimeError("Requested device=mps but MPS is not available")

    if req in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device=cuda but CUDA is not available")

    if req == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device request: {requested}. Use one of: auto, mps, cuda, cpu.")


def _infer_laterality(filename: str) -> str:
    name = filename.lower()
    if "_left_" in name:
        return "left"
    if "_right_" in name:
        return "right"
    return "unknown"


def _class_name_from_id(class_id: int, label_order: list[str]) -> str:
    cid = int(class_id)
    if cid < 0 or cid >= len(label_order):
        raise ValueError(f"class_id out of range: {cid}")
    return str(label_order[cid])


def _parse_aptos_csv(
    csv_path: str | Path,
    image_dir: str | Path,
    label_order: list[str],
    source_name: str,
    source_dataset: str = "aptos2019",
) -> pd.DataFrame:
    csv_file = Path(csv_path)
    image_root = Path(image_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image dir not found: {image_root}")

    df = pd.read_csv(csv_file)
    required_cols = ["id_code", "diagnosis"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")

    rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        id_code = str(row["id_code"]).strip()
        filename = f"{id_code}.png"
        class_id = int(row["diagnosis"])
        class_name = _class_name_from_id(class_id, label_order)
        image_path = image_root / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        rows.append(
            {
                "sample_id": f"{source_name}_{id_code}_{int(row_idx)}",
                "patient_id": id_code,
                "split": source_name,
                "filename": filename,
                "image_path": str(image_path),
                "class_id": int(class_id),
                "class_name": class_name,
                "laterality": _infer_laterality(filename),
                "source_dataset": str(source_dataset),
            }
        )

    return pd.DataFrame(rows)


def _load_aptos_full_pool(conf: dict[str, Any], label_order: list[str]) -> pd.DataFrame:
    required_aptos_keys = [
        "aptos_train_csv",
        "aptos_valid_csv",
        "aptos_test_csv",
        "aptos_train_image_dir",
        "aptos_valid_image_dir",
        "aptos_test_image_dir",
    ]
    missing_aptos = [k for k in required_aptos_keys if k not in conf["paths"]]
    if missing_aptos:
        raise KeyError(f"Missing APTOS path keys in config: {missing_aptos}")

    aptos_train = _parse_aptos_csv(
        csv_path=conf["paths"]["aptos_train_csv"],
        image_dir=conf["paths"]["aptos_train_image_dir"],
        label_order=label_order,
        source_name="source_aptos_train",
        source_dataset="aptos2019",
    )
    aptos_valid = _parse_aptos_csv(
        csv_path=conf["paths"]["aptos_valid_csv"],
        image_dir=conf["paths"]["aptos_valid_image_dir"],
        label_order=label_order,
        source_name="source_aptos_valid",
        source_dataset="aptos2019",
    )
    aptos_test = _parse_aptos_csv(
        csv_path=conf["paths"]["aptos_test_csv"],
        image_dir=conf["paths"]["aptos_test_image_dir"],
        label_order=label_order,
        source_name="source_aptos_test",
        source_dataset="aptos2019",
    )
    aptos_full = pd.concat([aptos_train, aptos_valid, aptos_test], ignore_index=True)
    if aptos_full["patient_id"].astype(str).duplicated().any():
        dup = int(aptos_full["patient_id"].astype(str).duplicated().sum())
        raise RuntimeError(f"Duplicate patient_id entries detected in APTOS pool: {dup}")
    return aptos_full


def _data_source(conf: dict[str, Any]) -> str:
    return str(conf.get("data", {}).get("source", "aptos_only")).strip().lower()


def _load_dataset_pool(conf: dict[str, Any], label_order: list[str]) -> pd.DataFrame:
    source = _data_source(conf)
    if source in {"aptos_only", "aptos", "aptos2019"}:
        return _load_aptos_full_pool(conf, label_order=label_order)
    raise ValueError(f"Unsupported data.source={source}. Only aptos_only is supported.")


def _slug_token(raw: Any, fallback: str = "value") -> str:
    token = "".join(ch.lower() if str(ch).isalnum() else "_" for ch in str(raw))
    token = "_".join([part for part in token.split("_") if part])
    return token or fallback


def _profile_dataset_tag(conf: dict[str, Any]) -> str:
    source_raw = _data_source(conf)
    if source_raw in {"aptos_only", "aptos", "aptos2019"} or "aptos" in source_raw:
        return "aptos2019"
    return _slug_token(source_raw, fallback="dataset")


def _profile_split_tag(conf: dict[str, Any]) -> str:
    data_cfg = conf.get("data", {})
    test_ratio = _to_ratio_fraction(data_cfg.get("profile_test_ratio", data_cfg.get("test_ratio", 0.15)), "profile_test_ratio")
    train_ratio = max(0.0, 1.0 - test_ratio)
    val_ratio = _to_ratio_fraction(
        data_cfg.get("profile_val_ratio_within_train", data_cfg.get("val_ratio", 0.10)),
        "profile_val_ratio_within_train",
    )
    return f"{int(round(train_ratio * 100))}-{int(round(test_ratio * 100))}-v{int(round(val_ratio * 100))}"


def _profile_profile_tag(conf: dict[str, Any]) -> str:
    return f"{_profile_dataset_tag(conf)}_{_profile_split_tag(conf)}"


def _manifest_suffix(conf: dict[str, Any], seed: int | None = None) -> str:
    use_seed = int(seed if seed is not None else conf.get("data", {}).get("stratify_seed", 1988))
    return f"_{_profile_profile_tag(conf)}_seed{use_seed}"


def _manifest_filename_map(conf: dict[str, Any], seed: int | None = None) -> dict[str, str]:
    suffix = _manifest_suffix(conf, seed=seed)
    return {
        "full_data": f"full_data{suffix}.csv",
        "test_full": f"test_full{suffix}.csv",
        "train_split": f"train_split{suffix}.csv",
        "val_split": f"val_split{suffix}.csv",
        "summary": f"summary{suffix}.json",
    }


def _manifest_outputs(conf: dict[str, Any], seed: int | None = None) -> dict[str, str]:
    manifests_dir = Path(conf["paths"]["manifests_dir"])
    names = _manifest_filename_map(conf, seed=seed)
    return {
        "full_data": str((manifests_dir / names["full_data"]).resolve()),
        "test_full": str((manifests_dir / names["test_full"]).resolve()),
        "train_split": str((manifests_dir / names["train_split"]).resolve()),
        "val_split": str((manifests_dir / names["val_split"]).resolve()),
        "summary": str((manifests_dir / names["summary"]).resolve()),
    }


def _split_two_stage_stratified_pool(
    full_df: pd.DataFrame,
    test_ratio: float,
    val_ratio_within_train: float,
    stratify_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "patient_id" not in full_df.columns:
        raise ValueError("benchmark split requires 'patient_id' column.")

    train_pool, test_split = train_test_split(
        full_df,
        test_size=float(test_ratio),
        random_state=int(stratify_seed),
        stratify=full_df["class_id"],
    )

    train_pool = train_pool.reset_index(drop=True)
    test_split = test_split.reset_index(drop=True)

    # Keep deterministic train/val split from train_pool for early stopping + calibration.
    val_ratio = float(max(0.01, min(0.5, val_ratio_within_train)))
    train_split, val_split = train_test_split(
        train_pool,
        test_size=val_ratio,
        random_state=int(stratify_seed),
        stratify=train_pool["class_id"],
    )

    train_split = train_split.reset_index(drop=True)
    val_split = val_split.reset_index(drop=True)

    train_ids = set(train_split["patient_id"].astype(str))
    val_ids = set(val_split["patient_id"].astype(str))
    test_ids = set(test_split["patient_id"].astype(str))
    if train_ids & test_ids:
        raise RuntimeError("train/test overlap detected in patient_id for two-stage stratified split.")
    if val_ids & test_ids:
        raise RuntimeError("val/test overlap detected in patient_id for two-stage stratified split.")

    return train_split, val_split, test_split, train_pool


def prepare_data_manifests(cfg: str | Path | dict[str, Any], seed: int = 1988) -> dict[str, str]:
    conf = _cfg(cfg)
    label_order = list(conf["data"]["label_order"])
    manifests = _manifest_outputs(conf, seed=seed)
    full_df = _load_dataset_pool(conf, label_order=label_order)

    test_ratio = float(conf.get("data", {}).get("profile_test_ratio", conf["data"].get("test_ratio", 0.15)))
    val_ratio_within_train = float(conf.get("data", {}).get("profile_val_ratio_within_train", conf["data"].get("val_ratio", 0.10)))
    split_seed = int(seed)

    train_split, val_split, test_split, train_pool = _split_two_stage_stratified_pool(
        full_df=full_df,
        test_ratio=test_ratio,
        val_ratio_within_train=val_ratio_within_train,
        stratify_seed=split_seed,
    )

    train_split = train_split.assign(split="train")
    val_split = val_split.assign(split="val")
    test_split = test_split.assign(split="test")
    all_full = pd.concat([train_pool.assign(split="train_pool"), test_split], ignore_index=True)

    all_full.to_csv(manifests["full_data"], index=False)
    test_split.to_csv(manifests["test_full"], index=False)
    train_split.to_csv(manifests["train_split"], index=False)
    val_split.to_csv(manifests["val_split"], index=False)

    n_total = len(all_full)
    summary = {
        "protocol": "benchmark",
        "seed": int(split_seed),
        "full_data_rows": int(n_total),
        "train_pool_rows": int(len(train_pool)),
        "train_split_rows": int(len(train_split)),
        "val_split_rows": int(len(val_split)),
        "test_full_rows": int(len(test_split)),
        "profile_test_ratio": float(test_ratio),
        "profile_val_ratio_within_train": float(val_ratio_within_train),
        "realized_train_ratio": float(len(train_split) / n_total) if n_total else 0.0,
        "realized_val_ratio": float(len(val_split) / n_total) if n_total else 0.0,
        "realized_test_ratio": float(len(test_split) / n_total) if n_total else 0.0,
        "train_val_disjoint": bool(set(train_split["patient_id"].astype(str)).isdisjoint(set(val_split["patient_id"].astype(str)))),
        "train_test_disjoint": bool(set(train_split["patient_id"].astype(str)).isdisjoint(set(test_split["patient_id"].astype(str)))),
        "val_test_disjoint": bool(set(val_split["patient_id"].astype(str)).isdisjoint(set(test_split["patient_id"].astype(str)))),
        "source_counts": {str(k): int(v) for k, v in all_full["source_dataset"].astype(str).value_counts().to_dict().items()},
        "class_counts_train": {str(k): int(v) for k, v in train_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
        "class_counts_val": {str(k): int(v) for k, v in val_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
        "class_counts_test": {str(k): int(v) for k, v in test_split["class_id"].astype(int).value_counts().sort_index().to_dict().items()},
    }
    _save_json(manifests["summary"], summary)
    return {
        "full_data": manifests["full_data"],
        "test_full": manifests["test_full"],
        "train_split": manifests["train_split"],
        "val_split": manifests["val_split"],
    }


def _build_transform(
    image_size: int,
    train: bool,
    aug_cfg: dict[str, float],
    preprocessing_cfg: dict[str, Any] | None = None,
    skip_resize: bool = False,
) -> T.Compose:
    prep = preprocessing_cfg or {}
    ops: list[Any] = []
    if train:
        ops.append(T.RandomHorizontalFlip(p=float(aug_cfg.get("profile_horizontal_flip", 0.5))))
        ops.append(T.RandomVerticalFlip(p=float(aug_cfg.get("profile_vertical_flip", 0.0))))
        profile_rot = float(aug_cfg.get("profile_rotation_degrees", 20.0))
        if profile_rot > 0.0:
            ops.append(T.RandomRotation(degrees=profile_rot))
        profile_translate = float(aug_cfg.get("profile_translate", 0.03))
        profile_scale_min = float(aug_cfg.get("profile_scale_min", 0.95))
        profile_scale_max = float(aug_cfg.get("profile_scale_max", 1.05))
        profile_shear = float(aug_cfg.get("profile_shear_degrees", 8.0))
        ops.append(
            T.RandomAffine(
                degrees=0.0,
                translate=(profile_translate, profile_translate),
                scale=(profile_scale_min, profile_scale_max),
                shear=(-profile_shear, profile_shear),
            )
        )
        ops.append(
            T.ColorJitter(
                brightness=float(aug_cfg.get("profile_brightness", 0.15)),
                contrast=float(aug_cfg.get("profile_contrast", 0.15)),
                saturation=float(aug_cfg.get("profile_saturation", 0.0)),
                hue=float(aug_cfg.get("profile_hue", 0.0)),
            )
        )

    if not skip_resize:
        ops.append(T.Resize((image_size, image_size)))
    norm_mean = prep.get("norm_mean", [0.485, 0.456, 0.406])
    norm_std = prep.get("norm_std", [0.229, 0.224, 0.225])
    if not (isinstance(norm_mean, list) and len(norm_mean) == 3 and isinstance(norm_std, list) and len(norm_std) == 3):
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=[float(x) for x in norm_mean], std=[float(x) for x in norm_std]),
        ]
    )
    return T.Compose(ops)


def _apply_fundus_preprocessing(image: Image.Image, image_size: int, preprocessing_cfg: dict[str, Any] | None = None) -> Image.Image:
    cfg = preprocessing_cfg or {}
    if not bool(cfg.get("enabled", True)):
        return image

    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # CLAHE on luminance for local contrast boost.
    clip_limit = float(cfg.get("clahe_clip_limit", 2.0))
    grid = max(2, int(cfg.get("clahe_tile_grid", 8)))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    l_chan = clahe.apply(l_chan)
    arr = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2RGB)

    # Ben Graham style local-mean subtraction.
    sigma = float(cfg.get("gaussian_sigma", max(1.0, image_size / 30.0)))
    weight = float(cfg.get("ben_graham_weight", 4.0))
    bias = float(cfg.get("ben_graham_bias", 128.0))
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    arr = cv2.addWeighted(arr, weight, blur, -weight, bias)

    # Circular crop to suppress outer border artifacts.
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    crop_ratio = float(cfg.get("circle_crop_ratio", 0.90))
    radius = max(1, int(min(h, w) * crop_ratio * 0.5))
    cv2.circle(mask, (w // 2, h // 2), radius, color=255, thickness=-1)
    arr_masked = np.zeros_like(arr)
    arr_masked[mask > 0] = arr[mask > 0]
    arr = arr_masked.astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


class _FundusDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int,
        aug_cfg: dict[str, float],
        preprocessing_cfg: dict[str, Any] | None = None,
        train: bool = False,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.image_size = int(image_size)
        self.preprocessing_cfg = preprocessing_cfg or {}
        self.preprocessing_enabled = bool(self.preprocessing_cfg.get("enabled", True))
        self.transform = _build_transform(
            image_size=image_size,
            train=train,
            aug_cfg=aug_cfg,
            preprocessing_cfg=self.preprocessing_cfg,
            skip_resize=self.preprocessing_enabled,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = _apply_fundus_preprocessing(image, image_size=self.image_size, preprocessing_cfg=self.preprocessing_cfg)
        x = self.transform(image)
        return {
            "image": x,
            "label": torch.tensor(int(row["class_id"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "laterality": str(row["laterality"]),
            "image_path": str(row["image_path"]),
        }


def _load_image_for_inference(
    image_path: str | Path,
    image_size: int,
    preprocessing_cfg: dict[str, Any] | None = None,
) -> tuple[Image.Image, torch.Tensor]:
    cfg = preprocessing_cfg or {}
    preprocessing_enabled = bool(cfg.get("enabled", True))
    image = Image.open(image_path).convert("RGB")
    image = _apply_fundus_preprocessing(image, image_size=image_size, preprocessing_cfg=cfg)
    transform = _build_transform(
        image_size=image_size,
        train=False,
        aug_cfg={},
        preprocessing_cfg=cfg,
        skip_resize=preprocessing_enabled,
    )
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def _manifest_path(conf: dict[str, Any], split: str, seed: int | None = None) -> Path:
    paths = _manifest_outputs(conf, seed=seed)
    mapping = {
        "train": Path(paths["train_split"]),
        "val": Path(paths["val_split"]),
        "test": Path(paths["test_full"]),
    }
    if split not in mapping:
        raise ValueError(f"Unsupported split={split}")
    return mapping[split]


def _backbone_name(conf: dict[str, Any]) -> str:
    raw = str(conf.get("training", {}).get("backbone", "resnet50")).strip().lower()
    aliases = {
        "resnet50": "resnet50",
        "resnet-50": "resnet50",
        "efficientnet_b4": "efficientnet_b4",
        "efficientnet-b4": "efficientnet_b4",
        "efficientnetb4": "efficientnet_b4",
        "vit_b16": "vit_b16",
        "vit_b_16": "vit_b16",
        "vit-base": "vit_b16",
        "vit_base": "vit_b16",
        "vitb16": "vit_b16",
    }
    if raw in aliases:
        return aliases[raw]
    raise ValueError(f"Unsupported backbone: {raw}. Supported: resnet50, efficientnet_b4, vit_b16")


def _model_image_size(conf: dict[str, Any]) -> int:
    requested = int(conf["data"]["image_size"])
    backbone = _backbone_name(conf)
    if backbone == "vit_b16":
        return 224
    return requested


def _to_ratio_fraction(value: Any, key: str) -> float:
    ratio = float(value)
    if ratio > 1.0:
        ratio = ratio / 100.0
    if ratio <= 0.0:
        raise ValueError(f"{key} must be > 0, got {value}")
    return ratio


def _split_policy_tag(conf: dict[str, Any]) -> str:
    return _profile_profile_tag(conf)


def _table_path(conf: dict[str, Any], stem: str, seed: int | None = None, split: str | None = None, ext: str = "csv") -> Path:
    base = Path(conf["paths"]["tables_dir"])
    parts = [str(stem)]
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    if split:
        parts.append(str(split))
    parts.append(_profile_profile_tag(conf))
    return base / f"{'_'.join(parts)}.{ext}"


def notebook_prepare_data_overview(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    samples_per_split: int = 3,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    manifests = prepare_data_manifests(conf, seed=int(seed))

    train_df = pd.read_csv(manifests["train_split"])
    val_df = pd.read_csv(manifests["val_split"])
    test_df = pd.read_csv(manifests["test_full"])

    total_rows = int(len(train_df) + len(val_df) + len(test_df))
    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": int(len(train_df))},
            {"split": "val", "rows": int(len(val_df))},
            {"split": "test", "rows": int(len(test_df))},
        ]
    )
    if total_rows > 0:
        split_summary["percentage"] = (split_summary["rows"] / float(total_rows)) * 100.0
    else:
        split_summary["percentage"] = np.nan
    split_summary["percentage"] = split_summary["percentage"].map(
        lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
    )

    label_order = list(conf["data"]["label_order"])
    class_map = {i: name for i, name in enumerate(label_order)}
    all_df = pd.concat(
        [
            train_df.assign(view="train"),
            val_df.assign(view="val"),
            test_df.assign(view="test"),
        ],
        ignore_index=True,
    )

    counts = (
        all_df.groupby(["view", "class_id"])
        .size()
        .reset_index(name="count")
    )
    counts["class_name"] = counts["class_id"].map(class_map)
    split_totals = counts.groupby("view")["count"].transform("sum")
    counts["split_pct"] = np.where(
        split_totals > 0,
        (counts["count"] / split_totals) * 100.0,
        np.nan,
    )
    counts["split_pct"] = counts["split_pct"].map(
        lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
    )

    overall = (
        all_df.groupby("class_id")
        .size()
        .reset_index(name="count")
    )
    overall["class_name"] = overall["class_id"].map(class_map)
    if len(all_df) > 0:
        overall["overall_pct"] = (overall["count"] / float(len(all_df))) * 100.0
    else:
        overall["overall_pct"] = np.nan
    overall["overall_pct"] = overall["overall_pct"].map(
        lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
    )

    source_summary = pd.DataFrame()
    if "source_dataset" in all_df.columns:
        source_summary = (
            all_df.groupby(["view", "source_dataset"])
            .size()
            .reset_index(name="rows")
        )
        source_summary["view_total"] = source_summary.groupby("view")["rows"].transform("sum")
        source_summary["view_pct"] = np.where(
            source_summary["view_total"] > 0,
            (source_summary["rows"] / source_summary["view_total"]) * 100.0,
            np.nan,
        )
        source_summary["view_pct"] = source_summary["view_pct"].map(
            lambda x: f"{float(x):.2f}%" if np.isfinite(x) else "N/A"
        )
        source_summary = source_summary.sort_values(["view", "source_dataset"]).reset_index(drop=True)

    pivot_counts = (
        counts.pivot(index="class_name", columns="view", values="count")
        .fillna(0)
        .reindex(label_order)
    )
    fig_counts, ax_counts = plt.subplots(figsize=(9, 4))
    pivot_counts.plot(kind="bar", ax=ax_counts)
    ax_counts.set_title("Class Distribution by Split (Counts)")
    ax_counts.set_ylabel("Count")
    ax_counts.tick_params(axis="x", rotation=20)
    fig_counts.tight_layout()

    pivot_prop = (
        counts.assign(split_pct_num=pd.to_numeric(counts["split_pct"].str.rstrip("%"), errors="coerce"))
        .pivot(index="class_name", columns="view", values="split_pct_num")
        .fillna(0.0)
        .reindex(label_order)
    )
    fig_prop, ax_prop = plt.subplots(figsize=(9, 4))
    pivot_prop.plot(kind="bar", ax=ax_prop)
    ax_prop.set_title("Class Distribution by Split (Within-Split %)")
    ax_prop.set_ylabel("Percentage (%)")
    ax_prop.tick_params(axis="x", rotation=20)
    fig_prop.tight_layout()

    samples_n = max(1, int(samples_per_split))
    split_frames: list[tuple[str, pd.DataFrame]] = [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]
    fig_samples, axes = plt.subplots(
        len(split_frames),
        samples_n,
        figsize=(4.0 * samples_n, 3.6 * len(split_frames)),
    )
    axes = np.array(axes, dtype=object)
    if axes.ndim == 1:
        if len(split_frames) == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(-1, 1)

    for r, (split_name, split_df) in enumerate(split_frames):
        available = int(len(split_df))
        n = min(samples_n, available)
        sample_df = split_df.sample(n=n, random_state=int(seed)).reset_index(drop=True) if n > 0 else pd.DataFrame()
        for c in range(samples_n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= n:
                continue
            row = sample_df.iloc[c]
            img = plt.imread(str(row["image_path"]))
            class_name = class_map.get(int(row["class_id"]), str(row["class_id"]))
            ax.imshow(img)
            ax.set_title(f"{split_name} | {class_name}", fontsize=9)
    fig_samples.suptitle("Sample Fundus Images", y=1.02)
    fig_samples.tight_layout()

    return {
        "cfg": conf,
        "seed": int(seed),
        "manifest_paths": manifests,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "split_summary": split_summary,
        "class_distribution": counts,
        "overall_class_proportions": overall.sort_values("class_id").reset_index(drop=True),
        "source_dataset_mix": source_summary,
        "class_count_fig": fig_counts,
        "class_proportion_fig": fig_prop,
        "sample_images_fig": fig_samples,
    }
