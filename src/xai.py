"""XAI audit module.

Explainability pipeline for the APTOS 2019 DR grading project: Grad-CAM and
SHAP DeepExplainer generation, per-sample metrics (border ratio, retina ratio,
faithfulness deltas, AOPC), the retinal-disc attribution-mask correction, and
all aggregate tables used by the report (method stats, pairwise McNemar,
continuous paired tests, per-class breakdown).

Main entry points used by the notebook:
    notebook_run_xai()                    -- full audit on the class-balanced subset
    notebook_load_xai_committee_summary() -- loads result tables for display
    notebook_run_single_case_report()     -- single-image Grad-CAM + SHAP demo
"""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
import types
import warnings
import hashlib
import math
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import EfficientNet_B4_Weights, ResNet50_Weights, ViT_B_16_Weights, efficientnet_b4, resnet50, vit_b_16
from torchvision.models.efficientnet import FusedMBConv, MBConv
from torchvision.models.resnet import BasicBlock, Bottleneck



import yaml


# -----------------------------
# Config + common utils
# -----------------------------


try:
    from captum.attr import LayerAttribution, LayerGradCam
except Exception as exc:  # pragma: no cover
    LayerAttribution = None
    LayerGradCam = None
    _CAPTUM_IMPORT_ERROR = exc
else:
    _CAPTUM_IMPORT_ERROR = None

try:
    import shap
except Exception as exc:  # pragma: no cover
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None

from src.data import (  # noqa: F401
    _REQUIRED_PATH_KEYS,
    _infer_project_root,
    load_project_config,
    _cfg,
    _save_json,
    _load_json,
    _write_alias_copy,
    _set_seed,
    _resolve_device,
    _infer_laterality,
    _class_name_from_id,
    _normalize_label_token,
    _parse_aptos_csv,
    _parse_roboflow_classes_csv,
    _load_aptos_full_pool,
    _load_roboflow_full_pool,
    _data_source,
    _load_dataset_pool,
    _split_train_val_test,
    _split_train_val_only,
    _data_protocol,
    _is_benchmark,
    _use_legacy_aliases,
    _slug_token,
    _profile_dataset_tag,
    _profile_split_tag,
    _profile_profile_tag,
    _manifest_suffix,
    _manifest_filename_map,
    _manifest_outputs,
    _split_two_stage_stratified_pool,
    freeze_current_test_manifest,
    prepare_data_manifests,
    _build_transform,
    _apply_fundus_preprocessing,
    _FundusDataset,
    _load_image_for_inference,
    _manifest_path,
    _backbone_name,
    _model_image_size,
    _to_ratio_fraction,
    _split_policy_tag,
    _table_path,
    notebook_prepare_data_overview,
)
from src.train import (  # noqa: F401
    DRClassifier,
    _LogitWrapper,
    _classification_metrics,
    _per_class_metrics_from_confusion,
    _build_reliability_table,
    _expected_calibration_error,
    _latest_run_record_path,
    _legacy_checkpoint_alias_path,
    _legacy_calibration_alias_path,
    _legacy_predictions_alias_path,
    _legacy_train_history_alias_path,
    _legacy_gradcam_status_alias_path,
    _legacy_shap_status_alias_path,
    _legacy_run_log_alias_path,
    _new_run_id,
    _run_id_from_checkpoint_path,
    _checkpoint_path_for_run_id,
    _calibration_path_for_run_id,
    _predictions_path_for_run_id,
    _train_history_log_path,
    _gradcam_status_log_path,
    _shap_status_log_path,
    _workflow_run_log_path,
    _save_latest_run_record,
    _load_latest_run_record,
    _latest_named_checkpoint,
    _resolve_checkpoint_and_run_id,
    _find_matching_checkpoint_by_signature,
    _checkpoint_path,
    _calibration_path,
    _predictions_path,
    _run_id_from_predictions_path,
    _checkpoint_config_signature,
    _class_weights,
    _ordinal_ce_loss,
    _focal_loss,
    _predict_manifest,
    _collect_logits_and_labels,
    _fit_temperature,
    _load_model,
    train_dr_classifier,
    build_validation_calibration_table,
    run_split_inference,
    evaluate_pipeline_outputs,
    export_final_headline_metrics,
    _ci95_summary,
    _profile_seed_list,
    export_benchmark_scoreboard,
    run_complete_workflow,
    run_benchmark_experiments,
    notebook_run_training,
    notebook_run_core_evaluation,
    clean_generated_outputs,
)


def _resolve_xai_device(conf: dict[str, Any]) -> torch.device:
    xai_cfg = conf.get("xai", {}) if isinstance(conf, dict) else {}
    requested = xai_cfg.get("device", None)
    if requested is None or str(requested).strip() == "":
        requested = conf.get("training", {}).get("device", "mps")
    return _resolve_device(str(requested))


def _border_mask(shape: tuple[int, int], border_ratio: float = 0.10) -> np.ndarray:
    h, w = shape
    bh = max(1, int(round(h * border_ratio)))
    bw = max(1, int(round(w * border_ratio)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bh, :] = True
    mask[-bh:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    return mask


def _retina_circle_mask(shape: tuple[int, int], radius_ratio: float = 0.45) -> np.ndarray:
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    r = min(h, w) * radius_ratio
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _attribution_retina_mask(shape: tuple[int, int], conf: dict) -> np.ndarray:
    override = conf.get("xai", {}).get("attribution_mask_radius_ratio")
    if override is None:
        crop_ratio = float(conf.get("preprocessing", {}).get("circle_crop_ratio", 1.00))
        radius_ratio = crop_ratio * 0.5
    else:
        radius_ratio = float(override)
    return _retina_circle_mask(shape, radius_ratio=radius_ratio)


def _attribution_mass_ratios(attr_map: np.ndarray) -> dict[str, float]:
    arr = np.abs(np.asarray(attr_map, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError("attr_map must be 2D")

    total = float(arr.sum())
    if total <= 1e-12:
        return {"border_ratio": 0.0, "retina_ratio": 0.0}

    border = _border_mask(arr.shape, border_ratio=0.10)
    retina = _retina_circle_mask(arr.shape, radius_ratio=0.45)

    return {
        "border_ratio": float(arr[border].sum() / total),
        "retina_ratio": float(arr[retina].sum() / total),
    }


def _mask_by_score_map(image_tensor: torch.Tensor, score_map: np.ndarray, top_k_ratio: float, random_seed: int = 1988) -> tuple[torch.Tensor, torch.Tensor]:
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape [1, C, H, W]")

    c, h, w = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
    arr = np.asarray(score_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"score_map must be 2D, got {arr.shape}")

    if arr.shape != (h, w):
        arr = np.array(
            Image.fromarray(arr.astype(np.float32), mode="F").resize((w, h), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

    flat = arr.reshape(-1)
    k = max(1, int(round(flat.size * top_k_ratio)))

    top_idx = np.argsort(flat)[-k:]
    rng = np.random.default_rng(random_seed)
    rand_idx = rng.choice(flat.size, size=k, replace=False)

    top_mask = np.zeros(flat.size, dtype=bool)
    top_mask[top_idx] = True
    rand_mask = np.zeros(flat.size, dtype=bool)
    rand_mask[rand_idx] = True

    top_mask_2d = top_mask.reshape(h, w)
    rand_mask_2d = rand_mask.reshape(h, w)

    x_top = image_tensor.clone()
    x_rand = image_tensor.clone()

    top_mask_t = torch.tensor(top_mask_2d, dtype=torch.bool, device=image_tensor.device)
    rand_mask_t = torch.tensor(rand_mask_2d, dtype=torch.bool, device=image_tensor.device)

    for ch in range(c):
        x_top[0, ch][top_mask_t] = 0.0
        x_rand[0, ch][rand_mask_t] = 0.0

    return x_top, x_rand


@torch.inference_mode()
def _faithfulness_delta(model: nn.Module, image_tensor: torch.Tensor, score_map: np.ndarray, pred_class: int, top_k_ratio: float = 0.20, random_seed: int = 1988) -> float:
    logits_base = model(image_tensor)
    probs_base = torch.softmax(logits_base, dim=1)
    base_prob = float(probs_base[0, pred_class].item())

    x_top, x_rand = _mask_by_score_map(image_tensor=image_tensor, score_map=score_map, top_k_ratio=top_k_ratio, random_seed=random_seed)
    probs_top = torch.softmax(model(x_top), dim=1)
    probs_rand = torch.softmax(model(x_rand), dim=1)

    drop_top = base_prob - float(probs_top[0, pred_class].item())
    drop_rand = base_prob - float(probs_rand[0, pred_class].item())
    return float(drop_top - drop_rand)


def _parse_faithfulness_k_list(raw_value: Any) -> list[float]:
    default_k = [0.05, 0.10, 0.20, 0.30]
    values: list[float] = []
    if isinstance(raw_value, (list, tuple)):
        for v in raw_value:
            try:
                k = float(v)
            except (TypeError, ValueError):
                continue
            if 0.0 < k < 1.0:
                values.append(k)
    if not values:
        values = default_k
    values = sorted({round(v, 4) for v in values})
    return values


def _k_to_col_name(k_ratio: float) -> str:
    return f"faith_delta_k{int(round(k_ratio * 100)):02d}"


def _faithfulness_multi_k(
    model: nn.Module,
    image_tensor: torch.Tensor,
    score_map: np.ndarray,
    pred_class: int,
    k_list: list[float],
    random_seed: int = 1988,
) -> tuple[dict[str, float], float]:
    out: dict[str, float] = {}
    for k in k_list:
        col = _k_to_col_name(k)
        out[col] = float(
            _faithfulness_delta(
                model=model,
                image_tensor=image_tensor,
                score_map=score_map,
                pred_class=pred_class,
                top_k_ratio=float(k),
                random_seed=random_seed,
            )
        )
    if out:
        aopc_delta = float(np.mean(list(out.values())))
    else:
        aopc_delta = float("nan")
    return out, aopc_delta


def _bootstrap_pass_rate_ci(pass_values: np.ndarray, n_boot: int = 1000, seed: int = 1988) -> tuple[float, float]:
    vals = np.asarray(pass_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan")
    n_boot = max(1, int(n_boot))
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _xai_pass_rule_thresholds(conf: dict[str, Any]) -> tuple[float, float]:
    xai_cfg = conf.get("xai", {}) if isinstance(conf, dict) else {}
    border_ratio_max = float(xai_cfg.get("pass_border_ratio_max", 0.35))
    faith_delta_min = float(xai_cfg.get("pass_faith_delta_min", 0.0))
    return border_ratio_max, faith_delta_min


def _xai_pass_flag(border_ratio: float, faith_delta: float, border_ratio_max: float, faith_delta_min: float) -> int:
    if np.isnan(border_ratio) or np.isnan(faith_delta):
        return 0
    return int((float(border_ratio) <= float(border_ratio_max)) and (float(faith_delta) > float(faith_delta_min)))


def _mcnemar_exact_pvalue(n01: int, n10: int) -> float:
    n01 = int(n01)
    n10 = int(n10)
    n = n01 + n10
    if n <= 0:
        return float("nan")
    k = min(n01, n10)
    p_tail = 0.0
    for i in range(0, k + 1):
        p_tail += math.comb(n, i)
    p_two_sided = min(1.0, 2.0 * p_tail * (0.5**n))
    return float(p_two_sided)


def _mcnemar_chi2_approx(n01: int, n10: int, yates: bool = True) -> tuple[float, float]:
    n01 = int(n01)
    n10 = int(n10)
    n = n01 + n10
    if n <= 0:
        return float("nan"), float("nan")

    diff = abs(n01 - n10)
    if yates:
        chi2 = ((max(diff - 1, 0)) ** 2) / n
    else:
        chi2 = (diff**2) / n

    # For chi-square(df=1), survival function is erfc(sqrt(x/2)).
    pval = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return float(chi2), float(pval)


def _build_xai_method_stats_row(
    method: str,
    df: pd.DataFrame,
    pass_col: str,
    n_boot: int,
    seed: int,
    border_ratio_max: float,
    faith_delta_min: float,
    evaluation_scope: str = "",
    status_note: str = "",
) -> dict[str, Any]:
    if len(df) == 0:
        return {
            "method": method,
            "n": 0,
            "pass_rate": float("nan"),
            "pass_rate_ci95_low": float("nan"),
            "pass_rate_ci95_high": float("nan"),
            "mean_border_ratio": float("nan"),
            "mean_retina_ratio": float("nan"),
            "mean_faith_delta_k20": float("nan"),
            "mean_aopc_delta": float("nan"),
            "aopc_pass_rate": float("nan"),
            "pass_rule_border_ratio_max": float(border_ratio_max),
            "pass_rule_faith_delta_k20_min": float(faith_delta_min),
            "evaluation_scope": str(evaluation_scope),
            "status_note": status_note or "no_rows",
        }

    pass_vals = df[pass_col].astype(float).to_numpy()
    ci_low, ci_high = _bootstrap_pass_rate_ci(pass_vals, n_boot=n_boot, seed=seed)
    if "faith_delta_k20" in df.columns:
        faith_k20 = pd.to_numeric(df["faith_delta_k20"], errors="coerce")
    else:
        faith_k20 = pd.to_numeric(df.get("faithfulness_delta", np.nan), errors="coerce")

    if "aopc_pass" in df.columns:
        aopc_pass_rate = float(pd.to_numeric(df["aopc_pass"], errors="coerce").mean())
    else:
        aopc_pass_rate = float("nan")

    pass_rate_value = float(np.mean(pass_vals))
    return {
        "method": method,
        "n": int(len(df)),
        "pass_rate": float(pass_rate_value),
        "pass_rate_ci95_low": float(ci_low),
        "pass_rate_ci95_high": float(ci_high),
        "mean_border_ratio": float(pd.to_numeric(df["border_ratio"], errors="coerce").mean()),
        "mean_retina_ratio": float(pd.to_numeric(df["retina_ratio"], errors="coerce").mean()),
        "mean_faith_delta_k20": float(faith_k20.mean()),
        "mean_aopc_delta": float(pd.to_numeric(df.get("aopc_delta", np.nan), errors="coerce").mean()),
        "aopc_pass_rate": float(aopc_pass_rate),
        "pass_rule_border_ratio_max": float(border_ratio_max),
        "pass_rule_faith_delta_k20_min": float(faith_delta_min),
        "evaluation_scope": str(evaluation_scope),
        "status_note": status_note,
    }


def _build_xai_pass_by_correctness_table(method: str, df: pd.DataFrame, pass_col: str) -> pd.DataFrame:
    out_cols = [
        "method",
        "group",
        "n",
        "pass_rate",
        "mean_border_ratio",
        "mean_faith_delta_k20",
    ]
    if len(df) == 0 or pass_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    work = df.copy()
    if "faith_delta_k20" in work.columns:
        work["_faith_k20"] = pd.to_numeric(work["faith_delta_k20"], errors="coerce")
    else:
        work["_faith_k20"] = pd.to_numeric(work.get("faithfulness_delta", np.nan), errors="coerce")
    work["_pass"] = pd.to_numeric(work[pass_col], errors="coerce")
    work["_border"] = pd.to_numeric(work.get("border_ratio", np.nan), errors="coerce")

    groups: list[tuple[str, pd.DataFrame]] = [("all", work)]
    if {"target_class", "pred_class"}.issubset(work.columns):
        truth = pd.to_numeric(work["target_class"], errors="coerce")
        pred = pd.to_numeric(work["pred_class"], errors="coerce")
        is_correct = truth == pred
        groups.append(("correct", work[is_correct]))
        groups.append(("wrong", work[~is_correct]))

    rows: list[dict[str, Any]] = []
    for group_name, part in groups:
        if len(part) == 0:
            rows.append(
                {
                    "method": method,
                    "group": group_name,
                    "n": 0,
                    "pass_rate": float("nan"),
                    "mean_border_ratio": float("nan"),
                    "mean_faith_delta_k20": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "method": method,
                "group": group_name,
                "n": int(len(part)),
                "pass_rate": float(part["_pass"].mean()),
                "mean_border_ratio": float(part["_border"].mean()),
                "mean_faith_delta_k20": float(part["_faith_k20"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


def _build_xai_pass_by_class_table(
    method: str,
    df: pd.DataFrame,
    pass_col: str,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    out_cols = ["method", "class_id", "class_name", "n", "pass_rate"]
    if len(df) == 0 or pass_col not in df.columns or "target_class" not in df.columns:
        return pd.DataFrame(columns=out_cols)

    work = df.copy()
    work["target_class"] = pd.to_numeric(work["target_class"], errors="coerce")
    work["_pass"] = pd.to_numeric(work[pass_col], errors="coerce")
    grouped = (
        work.dropna(subset=["target_class"])
        .groupby("target_class", dropna=False)["_pass"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"target_class": "class_id", "mean": "pass_rate", "count": "n"})
    )
    grouped["class_id"] = grouped["class_id"].astype(int)
    if class_names:
        grouped["class_name"] = grouped["class_id"].apply(
            lambda cid: class_names[cid] if 0 <= int(cid) < len(class_names) else str(cid)
        )
    else:
        grouped["class_name"] = grouped["class_id"].astype(str)
    grouped["method"] = method
    return grouped[out_cols]


_CONTINUOUS_METRIC_DISPLAY: dict[str, tuple[str, str]] = {
    "border_ratio":    ("Border ratio (lower = on-retina)",        "↓"),
    "retina_ratio":    ("Retina ratio (higher = on-retina)",       "↑"),
    "faith_delta_k10": ("Faithfulness Δ at k=10% (higher better)", "↑"),
    "faith_delta_k20": ("Faithfulness Δ at k=20% (higher better)", "↑"),
    "faith_delta_k30": ("Faithfulness Δ at k=30% (higher better)", "↑"),
    "aopc_delta":      ("AOPC (Samek 2017, higher better)",         "↑"),
}


def _format_continuous_table_for_display(continuous_df: pd.DataFrame) -> pd.DataFrame:
    if continuous_df is None or continuous_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, rec in continuous_df.iterrows():
        metric_key = str(rec.get("metric", ""))
        label, arrow = _CONTINUOUS_METRIC_DISPLAY.get(metric_key, (metric_key, ""))
        g_mean = float(rec.get("gradcam_mean", float("nan")))
        s_mean = float(rec.get("shap_mean", float("nan")))
        diff = float(rec.get("paired_mean_diff_gradcam_minus_shap", float("nan")))
        w_p = float(rec.get("wilcoxon_pvalue", float("nan")))
        cohen = float(rec.get("cohen_dz", float("nan")))
        winner_raw = str(rec.get("winner", ""))
        winner_display = {"gradcam": "Grad-CAM", "shap": "SHAP", "tie": "tie"}.get(winner_raw, winner_raw)
        if not np.isnan(w_p):
            if w_p < 0.001:
                p_text = "< 0.001"
            elif w_p < 0.01:
                p_text = f"{w_p:.3f}"
            else:
                p_text = f"{w_p:.3f}"
        else:
            p_text = "nan"
        rows.append(
            {
                "Metric": label,
                "Direction": arrow,
                "Grad-CAM (mean)": round(g_mean, 3),
                "SHAP (mean)": round(s_mean, 3),
                "Paired Δ (Grad−SHAP)": round(diff, 3),
                "Wilcoxon p": p_text,
                "Cohen's dz": round(cohen, 2) if not np.isnan(cohen) else float("nan"),
                "Winner": winner_display,
            }
        )
    return pd.DataFrame(rows)


def _format_continuous_bottom_line(continuous_df: pd.DataFrame) -> str:
    if continuous_df is None or continuous_df.empty:
        return (
            "> **Continuous analysis unavailable** (rq_xai_continuous_*.csv not found). "
            "Re-run cell 15 to generate it."
        )
    rec_border = continuous_df[continuous_df["metric"] == "border_ratio"]
    rec_aopc = continuous_df[continuous_df["metric"] == "aopc_delta"]
    rec_k20 = continuous_df[continuous_df["metric"] == "faith_delta_k20"]

    def _fmt_p(val: float) -> str:
        if np.isnan(val):
            return "nan"
        if val < 0.001:
            return "< 0.001"
        return f"{val:.3f}"

    parts: list[str] = [
        "> **Continuous analysis (research-standard, no arbitrary thresholds):** "
        "paired Wilcoxon signed-rank tests on raw scores.",
    ]
    if len(rec_border):
        r = rec_border.iloc[0]
        p = _fmt_p(float(r["wilcoxon_pvalue"]))
        parts.append(
            f"> **Localization (border_ratio):** Grad-CAM={float(r['gradcam_mean']):.3f}, "
            f"SHAP={float(r['shap_mean']):.3f}, winner=**{str(r['winner']).upper()}**, "
            f"Wilcoxon p={p}, Cohen's dz={float(r['cohen_dz']):+.2f}."
        )
    if len(rec_k20):
        r = rec_k20.iloc[0]
        p = _fmt_p(float(r["wilcoxon_pvalue"]))
        parts.append(
            f"> **Faithfulness (Δ_k20):** Grad-CAM={float(r['gradcam_mean']):.3f}, "
            f"SHAP={float(r['shap_mean']):.3f}, winner=**{str(r['winner']).upper()}**, "
            f"Wilcoxon p={p}, Cohen's dz={float(r['cohen_dz']):+.2f}."
        )
    if len(rec_aopc):
        r = rec_aopc.iloc[0]
        p = _fmt_p(float(r["wilcoxon_pvalue"]))
        parts.append(
            f"> **AOPC (Samek 2017):** Grad-CAM={float(r['gradcam_mean']):.3f}, "
            f"SHAP={float(r['shap_mean']):.3f}, winner=**{str(r['winner']).upper()}**, "
            f"Wilcoxon p={p}, Cohen's dz={float(r['cohen_dz']):+.2f}."
        )
    parts.append(
        "> **Interpretation:** the thresholded pass-rate is a descriptive summary "
        "under a project-specific rule; this continuous analysis is the primary finding."
    )
    return "\n".join(parts)


def _build_xai_continuous_stats(rq1_df: pd.DataFrame, rq2_df: pd.DataFrame) -> pd.DataFrame:
    """Research-standard paired continuous-score comparison.

    Computes Wilcoxon signed-rank test, paired t-test, and Cohen's dz on each
    continuous XAI metric (border_ratio, retina_ratio, faith_delta_k10/20/30,
    aopc_delta) between Grad-CAM and SHAP. This is the primary comparison used
    in the XAI literature (Samek et al. 2017 AOPC; Petsiuk et al. 2018 RISE
    Insertion/Deletion; Yeh et al. 2019 Infidelity) and avoids the arbitrary
    thresholds required by a binary pass/fail rule.
    """
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError:
        return pd.DataFrame()

    if rq1_df.empty or rq2_df.empty:
        return pd.DataFrame()
    if "sample_id" not in rq1_df.columns or "sample_id" not in rq2_df.columns:
        return pd.DataFrame()

    pair = rq1_df.merge(rq2_df, on="sample_id", suffixes=("_grad", "_shap"))
    if pair.empty:
        return pd.DataFrame()

    metrics: list[tuple[str, str]] = [
        ("border_ratio", "lower_is_better"),
        ("retina_ratio", "higher_is_better"),
        ("faith_delta_k10", "higher_is_better"),
        ("faith_delta_k20", "higher_is_better"),
        ("faith_delta_k30", "higher_is_better"),
        ("aopc_delta", "higher_is_better"),
    ]

    rows: list[dict[str, Any]] = []
    for col, direction in metrics:
        g_col = f"{col}_grad"
        s_col = f"{col}_shap"
        if g_col not in pair.columns or s_col not in pair.columns:
            continue
        g = pd.to_numeric(pair[g_col], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(pair[s_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(g) & np.isfinite(s)
        g = g[mask]
        s = s[mask]
        if g.size < 3:
            continue

        diff = g - s
        mean_g = float(np.mean(g))
        mean_s = float(np.mean(s))
        median_g = float(np.median(g))
        median_s = float(np.median(s))
        mean_diff = float(np.mean(diff))
        median_diff = float(np.median(diff))
        std_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0

        try:
            w_stat, w_p = wilcoxon(g, s, zero_method="pratt", alternative="two-sided")
            wilcoxon_stat = float(w_stat)
            wilcoxon_pvalue = float(w_p)
        except Exception:
            wilcoxon_stat = float("nan")
            wilcoxon_pvalue = float("nan")

        try:
            t_stat, t_p = ttest_rel(g, s)
            ttest_stat = float(t_stat)
            ttest_pvalue = float(t_p)
        except Exception:
            ttest_stat = float("nan")
            ttest_pvalue = float("nan")

        cohen_dz = float(mean_diff / (std_diff + 1e-12)) if std_diff > 0 else float("nan")

        if direction == "lower_is_better":
            winner = "shap" if mean_g > mean_s else "gradcam"
        else:
            winner = "gradcam" if mean_g > mean_s else "shap"
        if mean_g == mean_s:
            winner = "tie"

        rows.append(
            {
                "metric": col,
                "direction": direction,
                "n_paired": int(g.size),
                "gradcam_mean": mean_g,
                "gradcam_median": median_g,
                "shap_mean": mean_s,
                "shap_median": median_s,
                "paired_mean_diff_gradcam_minus_shap": mean_diff,
                "paired_median_diff_gradcam_minus_shap": median_diff,
                "paired_std_diff": std_diff,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_pvalue": wilcoxon_pvalue,
                "ttest_stat": ttest_stat,
                "ttest_pvalue": ttest_pvalue,
                "cohen_dz": cohen_dz,
                "winner": winner,
            }
        )

    return pd.DataFrame(rows)


def _build_xai_pairwise_stats(rq1_df: pd.DataFrame, rq2_df: pd.DataFrame) -> pd.DataFrame:
    left_cols = ["sample_id", "gradcam_pass"]
    right_cols = ["sample_id", "shap_pass"]
    if not set(left_cols).issubset(rq1_df.columns) or not set(right_cols).issubset(rq2_df.columns):
        return pd.DataFrame(
            [
                {
                    "n_paired": 0,
                    "gradcam_pass_rate": float("nan"),
                    "shap_pass_rate": float("nan"),
                    "delta_pass_rate": float("nan"),
                    "n00": 0,
                    "n01": 0,
                    "n10": 0,
                    "n11": 0,
                    "mcnemar_chi2": float("nan"),
                    "mcnemar_pvalue_chi2": float("nan"),
                    "mcnemar_pvalue_exact": float("nan"),
                    "status_note": "missing_pass_columns",
                }
            ]
        )

    pair = rq1_df[left_cols].merge(rq2_df[right_cols], on="sample_id", how="inner")
    if len(pair) == 0:
        return pd.DataFrame(
            [
                {
                    "n_paired": 0,
                    "gradcam_pass_rate": float("nan"),
                    "shap_pass_rate": float("nan"),
                    "delta_pass_rate": float("nan"),
                    "n00": 0,
                    "n01": 0,
                    "n10": 0,
                    "n11": 0,
                    "mcnemar_chi2": float("nan"),
                    "mcnemar_pvalue_chi2": float("nan"),
                    "mcnemar_pvalue_exact": float("nan"),
                    "status_note": "no_paired_samples",
                }
            ]
        )

    g = pair["gradcam_pass"].astype(int).to_numpy()
    s = pair["shap_pass"].astype(int).to_numpy()
    n00 = int(np.sum((g == 0) & (s == 0)))
    n01 = int(np.sum((g == 0) & (s == 1)))
    n10 = int(np.sum((g == 1) & (s == 0)))
    n11 = int(np.sum((g == 1) & (s == 1)))

    status_note = "ok"
    pvalue = float("nan")
    chi2 = float("nan")
    pvalue_chi2 = float("nan")
    if (n01 + n10) == 0:
        status_note = "no_discordant_pairs"
    else:
        pvalue = _mcnemar_exact_pvalue(n01=n01, n10=n10)
        chi2, pvalue_chi2 = _mcnemar_chi2_approx(n01=n01, n10=n10, yates=True)

    grad_rate = float(np.mean(g))
    shap_rate = float(np.mean(s))
    return pd.DataFrame(
        [
            {
                "n_paired": int(len(pair)),
                "gradcam_pass_rate": grad_rate,
                "shap_pass_rate": shap_rate,
                "delta_pass_rate": float(shap_rate - grad_rate),
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
                "mcnemar_chi2": chi2,
                "mcnemar_pvalue_chi2": pvalue_chi2,
                "mcnemar_pvalue_exact": pvalue,
                "status_note": status_note,
            }
        ]
    )


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = np.maximum(a, 0)
    d = float(a.max() - a.min())
    if d <= 1e-8:
        return np.zeros_like(a)
    return (a - a.min()) / d


def _overlay(base_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, cmap_name: str = "jet") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(heatmap)[..., :3]
    overlay = (1 - alpha) * (base_image / 255.0) + alpha * colored
    return np.clip(overlay, 0, 1)


def _save_overlay_image(overlay: np.ndarray, out_path: str | Path, dpi: int = 180) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(overlay)
    plt.tight_layout()
    plt.savefig(out, dpi=int(dpi), bbox_inches="tight", pad_inches=0)
    plt.close()
    return str(out)


def _infer_backbone_from_model(model: nn.Module) -> str:
    net = getattr(model, "net", model)
    if all(hasattr(net, name) for name in ["layer2", "layer3", "layer4"]):
        return "resnet50"
    if hasattr(net, "features") and isinstance(getattr(net, "features"), nn.Sequential):
        return "efficientnet_b4"
    if hasattr(net, "heads"):
        return "vit_b16"
    return net.__class__.__name__.strip().lower()


def _resolve_gradcam_target_layer(
    model: DRClassifier,
    layer_name: str,
    backbone_hint: str = "",
) -> tuple[nn.Module | None, str, str]:
    layer_key = str(layer_name).strip().lower()
    allowed = {"layer2", "layer3", "layer4"}
    if layer_key not in allowed:
        return None, "", f"Unsupported gradcam layer '{layer_name}'. Supported layers: layer2, layer3, layer4"

    net = model.net
    backbone = str(backbone_hint).strip().lower() or _infer_backbone_from_model(model)
    if backbone.startswith("vit"):
        return None, "", f"Grad-CAM requires CNN feature maps; backbone={backbone} is not supported."

    # ResNet-family mapping (explicit named stages).
    if all(hasattr(net, name) for name in ["layer2", "layer3", "layer4"]):
        target = getattr(net, layer_key, None)
        if target is not None:
            return target, f"net.{layer_key}", ""

    # EfficientNet family mapping (explicit feature stages requested by design).
    if hasattr(net, "features") and isinstance(getattr(net, "features"), nn.Sequential):
        stage_map = {"layer2": 4, "layer3": 6, "layer4": 8}
        features = net.features
        req_idx = int(stage_map[layer_key])
        if len(features) > 0:
            idx = req_idx if req_idx < len(features) else int(round((req_idx / 8.0) * (len(features) - 1)))
            idx = int(min(max(idx, 0), len(features) - 1))
            return features[idx], f"net.features.{idx}", ""

    # Generic CNN fallback: map layer2/3/4 to early/mid/late Conv2d stages.
    conv_layers = [(name, module) for name, module in net.named_modules() if isinstance(module, nn.Conv2d)]
    if len(conv_layers) == 0:
        return None, "", "Grad-CAM requires at least one Conv2d spatial layer."

    n_conv = len(conv_layers)
    idx_map = {
        "layer2": int(min(max(round(n_conv * 0.33) - 1, 0), n_conv - 1)),
        "layer3": int(min(max(round(n_conv * 0.66) - 1, 0), n_conv - 1)),
        "layer4": n_conv - 1,
    }
    sel_idx = idx_map[layer_key]
    sel_name, sel_module = conv_layers[sel_idx]
    resolved_name = f"net.{sel_name}" if sel_name else "net"
    return sel_module, resolved_name, ""


def _is_shap_inplace_view_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "backwardhookfunctionbackward" in msg
        or ("modified inplace" in msg and "view" in msg)
        or ("custom function" in msg and "inplace" in msg)
    )


def _is_mps_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("mps" in msg and "out of memory" in msg) or "mps backend out of memory" in msg


def _should_retry_shap_on_cpu(exc: Exception, device: torch.device) -> bool:
    dev_type = getattr(device, "type", "")
    if dev_type == "mps":
        return _is_shap_inplace_view_error(exc) or _is_mps_oom_error(exc)
    if dev_type == "cuda":
        return "out of memory" in str(exc).lower() or _is_shap_inplace_view_error(exc)
    return False


def _empty_mps_cache_if_available() -> None:
    try:
        if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _generate_gradcam(
    model: DRClassifier,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_id: int,
    layer_name: str,
    output_path: str | Path,
    device: torch.device,
    conf: dict,
    overlay_dpi: int = 180,
    backbone_hint: str = "",
) -> tuple[str, np.ndarray, np.ndarray]:
    if _CAPTUM_IMPORT_ERROR is not None:
        raise RuntimeError(f"captum import failed: {_CAPTUM_IMPORT_ERROR}")

    model.eval()
    target_layer, resolved_layer_name, reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=layer_name,
        backbone_hint=backbone_hint,
    )
    if target_layer is None:
        raise ValueError(reason or f"Unable to resolve Grad-CAM layer '{layer_name}'.")

    gradcam = LayerGradCam(lambda x: model(x), target_layer)
    attr = gradcam.attribute(input_tensor.to(device), target=class_id)

    input_h, input_w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
    attr_input = LayerAttribution.interpolate(attr, (input_h, input_w))

    heat_input = attr_input.squeeze().detach().cpu().numpy()
    if heat_input.ndim == 3:
        heat_input = np.mean(heat_input, axis=0)
    heat_input = _normalize_map(heat_input)
    heat_raw = heat_input.copy()
    heat_input = heat_input * _attribution_retina_mask(heat_input.shape, conf)

    base_h, base_w = original_image.shape[0], original_image.shape[1]
    heat_display = np.array(
        Image.fromarray(heat_input.astype(np.float32), mode="F").resize((base_w, base_h), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    heat_display = _normalize_map(heat_display)

    overlay = _overlay(original_image, heat_display, alpha=0.4, cmap_name="jet")
    artifact = _save_overlay_image(overlay, output_path, dpi=overlay_dpi)
    return artifact, heat_input, heat_raw


def _bottleneck_forward_shap_safe(self: Bottleneck, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=False)

    out = self.conv2(out)
    out = self.bn2(out)
    out = F.relu(out, inplace=False)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = F.relu(out, inplace=False)
    return out


def _basicblock_forward_shap_safe(self: BasicBlock, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=False)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity
    out = F.relu(out, inplace=False)
    return out


def _mbconv_forward_shap_safe(self: MBConv, x: torch.Tensor) -> torch.Tensor:
    result = self.block(x)
    if self.use_res_connect:
        result = self.stochastic_depth(result)
        result = result + x
    return result


def _fused_mbconv_forward_shap_safe(self: FusedMBConv, x: torch.Tensor) -> torch.Tensor:
    result = self.block(x)
    if self.use_res_connect:
        result = self.stochastic_depth(result)
        result = result + x
    return result


def _make_shap_compatible(model: nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "inplace"):
            try:
                module.inplace = False
            except Exception:
                pass
        if isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish)):
            module.inplace = False
        if isinstance(module, Bottleneck):
            module.forward = types.MethodType(_bottleneck_forward_shap_safe, module)
        elif isinstance(module, BasicBlock):
            module.forward = types.MethodType(_basicblock_forward_shap_safe, module)
        elif isinstance(module, MBConv):
            module.forward = types.MethodType(_mbconv_forward_shap_safe, module)
        elif isinstance(module, FusedMBConv):
            module.forward = types.MethodType(_fused_mbconv_forward_shap_safe, module)


def _build_shap_explainer_with_known_warning_filter(
    wrapper: nn.Module,
    background: torch.Tensor,
) -> Any:
    # SHAP DeepExplainer logs "unrecognized nn.Module" for several harmless
    # modules in torchvision EfficientNet (e.g., SiLU, StochasticDepth).
    # Filter only these known warnings to keep notebook logs clean.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: SiLU",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: StochasticDepth",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        return shap.DeepExplainer(wrapper, background)


def _shap_values_with_known_warning_filter(explainer: Any, inputs: torch.Tensor) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: SiLU",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"unrecognized nn\.Module: StochasticDepth",
            category=UserWarning,
            module=r"shap\.explainers\._deep\.deep_pytorch",
        )
        return explainer.shap_values(inputs, check_additivity=False)


def _pick_shap_map(shap_values: object, class_id: int, sample_index: int = 0) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_id][sample_index])

    arr = np.asarray(shap_values)
    if arr.ndim == 5:
        return arr[sample_index, :, :, :, class_id]
    if arr.ndim == 4:
        return arr[sample_index]
    raise ValueError(f"Unsupported SHAP array shape: {arr.shape}")


def _shap_to_2d(shap_map: np.ndarray, mode: str = "positive") -> np.ndarray:
    arr = np.asarray(shap_map, dtype=np.float32)
    if arr.ndim == 3:
        if mode == "signed":
            arr = np.mean(arr, axis=0)
        elif mode == "abs":
            arr = np.mean(np.abs(arr), axis=0)
        else:
            arr = np.mean(np.maximum(arr, 0.0), axis=0)
    elif arr.ndim != 2:
        raise ValueError(f"SHAP map must be 2D or 3D, got ndim={arr.ndim}")

    mode_key = str(mode).strip().lower()
    if mode_key == "signed":
        max_abs = float(np.max(np.abs(arr)))
        if max_abs <= 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr / max_abs).astype(np.float32)
    if mode_key == "abs":
        return _normalize_map(np.abs(arr))
    # default: positive evidence for predicted class
    return _normalize_map(np.maximum(arr, 0.0))


def _save_map_overlay(attr_map: np.ndarray, base_image: np.ndarray, output_path: str | Path, overlay_dpi: int = 180) -> str:
    arr = np.asarray(attr_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("attr_map must be 2D")

    if base_image.ndim == 2:
        base_image = np.stack([base_image] * 3, axis=-1)

    if base_image.shape[:2] != arr.shape[:2]:
        base_image = np.array(
            Image.fromarray(base_image.astype(np.uint8)).resize((arr.shape[1], arr.shape[0]), Image.Resampling.BILINEAR)
        )

    overlay = _overlay(base_image, arr, alpha=0.4, cmap_name="coolwarm")
    return _save_overlay_image(overlay, output_path, dpi=overlay_dpi)


def _choose_xai_targets(
    df: pd.DataFrame,
    high_conf_threshold: float,
    max_targets: int | None = None,
    num_classes: int = 5,
    class_col: str = "true_class",
    balance_by_class: bool = True,
    fill_missing_from_all: bool = True,
) -> pd.DataFrame:
    target = df[df["confidence"] >= high_conf_threshold].copy()
    if len(target) == 0:
        target = df.sort_values("confidence", ascending=False).head(min(32, len(df))).copy()

    if not balance_by_class or len(target) == 0:
        if max_targets is not None and int(max_targets) > 0:
            target = target.head(int(max_targets)).copy()
        return target.reset_index(drop=True)

    target = target.sort_values("confidence", ascending=False).copy()
    available_class_col = class_col if class_col in target.columns else ("pred_class" if "pred_class" in target.columns else class_col)
    if available_class_col not in target.columns:
        if max_targets is not None and int(max_targets) > 0:
            target = target.head(int(max_targets)).copy()
        return target.reset_index(drop=True)

    limit = int(max_targets) if (max_targets is not None and int(max_targets) > 0) else len(target)
    limit = max(1, limit)
    class_ids = list(range(int(num_classes)))
    per_class = max(1, limit // max(1, len(class_ids)))

    selected_parts: list[pd.DataFrame] = []
    for cid in class_ids:
        part = target[target[available_class_col].astype(int) == int(cid)].head(per_class)
        if len(part):
            selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else target.head(0).copy()

    if fill_missing_from_all and len(selected) < limit:
        missing_classes = {
            int(cid)
            for cid in class_ids
            if len(selected[selected[available_class_col].astype(int) == int(cid)]) == 0
        }
        if missing_classes and available_class_col in df.columns:
            pool = df.sort_values("confidence", ascending=False)
            for cid in sorted(missing_classes):
                add = pool[pool[available_class_col].astype(int) == int(cid)].head(1)
                if len(add):
                    selected = pd.concat([selected, add], ignore_index=True)

    if len(selected) < limit:
        used_ids = set(selected["sample_id"].astype(str).tolist()) if "sample_id" in selected.columns else set()
        filler = target[~target["sample_id"].astype(str).isin(used_ids)] if "sample_id" in target.columns else target
        need = max(0, limit - len(selected))
        if need > 0:
            selected = pd.concat([selected, filler.head(need)], ignore_index=True)

    selected = selected.head(limit).copy()
    return selected.reset_index(drop=True)


def _parse_gradcam_layers(raw_layers: Any, default_layer: str = "layer4") -> list[str]:
    allowed = {"layer2", "layer3", "layer4"}
    values: list[str] = []
    if isinstance(raw_layers, (list, tuple)):
        for v in raw_layers:
            key = str(v).strip().lower()
            if key in allowed and key not in values:
                values.append(key)
    elif isinstance(raw_layers, str):
        key = raw_layers.strip().lower()
        if key in allowed:
            values.append(key)
    default_key = str(default_layer).strip().lower()
    if default_key in allowed and default_key not in values:
        values.insert(0, default_key)
    if not values:
        values = [default_key if default_key in allowed else "layer4"]
    return values


def run_xai_analysis(
    cfg: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    checkpoint: str | Path | None = None,
    predictions_csv: str | Path | None = None,
    shap_mode: str | None = None,
    shap_max_samples: int | None = None,
) -> dict[str, str]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    _empty_mps_cache_if_available()
    if _CAPTUM_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"captum is required for Grad-CAM audit but failed to import: "
            f"{_CAPTUM_IMPORT_ERROR}. Install with `pip install captum==0.7.0`."
        )
    conf = _cfg(cfg)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    xai_cfg = conf.get("xai", {})
    pass_border_ratio_max, pass_faith_delta_min = _xai_pass_rule_thresholds(conf)
    k_list = _parse_faithfulness_k_list(xai_cfg.get("faithfulness_k_list", [0.05, 0.10, 0.20, 0.30]))
    k_cols = [_k_to_col_name(k) for k in k_list]
    max_targets = int(xai_cfg.get("max_targets", 0))
    if max_targets <= 0:
        max_targets = None
    fig_dpi = int(xai_cfg.get("figure_dpi", 180))
    bootstrap_iters = int(xai_cfg.get("stats_bootstrap_iters", 1000))
    bootstrap_seed = int(xai_cfg.get("stats_bootstrap_seed", 1988))
    xai_log_every_samples = int(xai_cfg.get("log_every_samples", 16))
    xai_start_time = time.time()

    if checkpoint is None:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    else:
        ckpt_path, run_id = _resolve_checkpoint_and_run_id(
            conf,
            seed=seed,
            checkpoint=checkpoint,
            require_existing=True,
        )
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)

    if predictions_csv is None:
        pred_path = Path(run_split_inference(conf, seed=seed, split=split))
    else:
        pred_path = Path(predictions_csv)

    pred_df = pd.read_csv(pred_path)
    high_conf_thr = float(conf["evaluation"].get("high_conf_threshold", 0.80))
    balance_targets = bool(xai_cfg.get("balance_targets_by_class", True))
    fill_missing_class = bool(xai_cfg.get("balance_fill_from_all", True))
    target_class_col = str(xai_cfg.get("target_balance_class_col", "true_class"))
    target_df = _choose_xai_targets(
        pred_df,
        high_conf_threshold=high_conf_thr,
        max_targets=max_targets,
        num_classes=int(conf["data"]["num_classes"]),
        class_col=target_class_col,
        balance_by_class=balance_targets,
        fill_missing_from_all=fill_missing_class,
    )

    use_mode = str(shap_mode if shap_mode is not None else xai_cfg.get("shap_mode", "subset")).strip().lower()
    if use_mode not in {"subset", "full"}:
        raise ValueError(f"Unsupported shap_mode={use_mode}")
    max_samples = int(shap_max_samples if shap_max_samples is not None else xai_cfg.get("shap_max_samples", 128))
    enforce_consistent_targets = bool(xai_cfg.get("enforce_consistent_targets", True))
    xai_target_df = target_df.copy()
    if enforce_consistent_targets and use_mode == "subset" and max_samples > 0:
        xai_target_df = xai_target_df.head(max_samples).copy()
    xai_target_df = xai_target_df.reset_index(drop=True)

    xai_targets_path = Path(conf["paths"]["tables_dir"]) / f"xai_targets_seed{seed}_{split}.csv"
    xai_target_df.to_csv(xai_targets_path, index=False)

    if high_conf_thr > 0.0:
        evaluation_scope = f"high_confidence_subset(confidence>={high_conf_thr:.2f})"
    else:
        evaluation_scope = "all_confidence"
    if max_targets is not None and max_targets > 0:
        evaluation_scope = f"{evaluation_scope}; max_targets={int(max_targets)}"

    print(
        f"[XAI] run_id={run_id} split={split} device={device} "
        f"targets={len(xai_target_df)} backbone={backbone} "
        f"shared_targets={enforce_consistent_targets} shap_mode={use_mode} "
        f"pass_rule=(border<={pass_border_ratio_max:.2f}, faith_k20>{pass_faith_delta_min:.2f})"
    )

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)

    # Grad-CAM (RQ1)
    grad_rows: list[dict[str, Any]] = []
    grad_dir = Path(conf["paths"]["figures_dir"]) / "gradcam"
    grad_dir.mkdir(parents=True, exist_ok=True)
    for _stale in grad_dir.glob("*.png"):
        _stale.unlink()
    gradcam_status_path = _gradcam_status_log_path(conf, run_id, split)
    gradcam_status_note = ""
    layer_summary_path: Path | None = None

    default_gradcam_layer = str(conf["xai"].get("gradcam_layer", "layer4"))
    gradcam_layers_eval = _parse_gradcam_layers(
        raw_layers=conf["xai"].get("gradcam_layers_eval", [default_gradcam_layer]),
        default_layer=default_gradcam_layer,
    )
    layer_rows: dict[str, list[dict[str, Any]]] = {layer: [] for layer in gradcam_layers_eval}
    unsupported_layers: dict[str, str] = {}
    supported_layers: list[str] = []
    layer_errors: dict[str, str] = {}

    for layer_name in gradcam_layers_eval:
        _, _, reason = _resolve_gradcam_target_layer(
            model=model,
            layer_name=layer_name,
            backbone_hint=backbone,
        )
        if reason:
            unsupported_layers[layer_name] = reason
        else:
            supported_layers.append(layer_name)

    if not supported_layers:
        reason = "; ".join(sorted(set(unsupported_layers.values()))) or "No Grad-CAM compatible feature layer was found."
        _save_json(
            gradcam_status_path,
            {
                "status": "skipped",
                "seed": int(seed),
                "run_id": run_id,
                "split": split,
                "backbone": backbone,
                "requested_layers": gradcam_layers_eval,
                "supported_layers": [],
                "unsupported_layers": unsupported_layers,
                "reason": reason,
                "shared_targets_enforced": bool(enforce_consistent_targets),
                "target_rows": int(len(xai_target_df)),
            },
        )
        gradcam_status_note = f"skipped: {reason}"
    else:
        total_grad_jobs = int(len(xai_target_df) * len(supported_layers))
        grad_job_idx = 0
        for _, row in xai_target_df.iterrows():
            image_pil, image_tensor = _load_image_for_inference(
                row["image_path"],
                image_size=image_size,
                preprocessing_cfg=conf.get("preprocessing", {}),
            )
            image_tensor = image_tensor.to(device)
            target_class_value = int(row[target_class_col]) if target_class_col in row.index else int(row["pred_class"])

            for layer_name in supported_layers:
                grad_job_idx += 1
                out_path = grad_dir / f"{row['sample_id']}_cls{int(row['pred_class'])}_{layer_name}.png"
                try:
                    artifact, heat_input, heat_raw = _generate_gradcam(
                        model=model,
                        input_tensor=image_tensor,
                        original_image=np.array(image_pil),
                        class_id=int(row["pred_class"]),
                        layer_name=layer_name,
                        output_path=out_path,
                        device=device,
                        conf=conf,
                        overlay_dpi=fig_dpi,
                        backbone_hint=backbone,
                    )
                except Exception as exc:
                    layer_errors[layer_name] = str(exc)
                    if xai_log_every_samples > 0 and (
                        grad_job_idx == 1
                        or grad_job_idx % xai_log_every_samples == 0
                        or grad_job_idx == total_grad_jobs
                    ):
                        elapsed_m = (time.time() - xai_start_time) / 60.0
                        print(
                            f"[XAI][GradCAM] job={grad_job_idx}/{total_grad_jobs} "
                            f"layer={layer_name} status=error elapsed={elapsed_m:.1f}m"
                        )
                    continue

                ratios = _attribution_mass_ratios(heat_input)
                raw_ratios = _attribution_mass_ratios(heat_raw)
                faith_by_k, aopc_delta = _faithfulness_multi_k(
                    model=model,
                    image_tensor=image_tensor,
                    score_map=heat_input,
                    pred_class=int(row["pred_class"]),
                    k_list=k_list,
                    random_seed=bootstrap_seed,
                )
                faith_legacy = float(faith_by_k.get("faith_delta_k20", np.nan))
                if np.isnan(faith_legacy):
                    faith_legacy = _faithfulness_delta(
                        model=model,
                        image_tensor=image_tensor,
                        score_map=heat_input,
                        pred_class=int(row["pred_class"]),
                        top_k_ratio=0.20,
                        random_seed=bootstrap_seed,
                    )
                pass_flag = _xai_pass_flag(
                    border_ratio=float(ratios["border_ratio"]),
                    faith_delta=float(faith_legacy),
                    border_ratio_max=pass_border_ratio_max,
                    faith_delta_min=pass_faith_delta_min,
                )
                aopc_pass = int(aopc_delta > 0.0) if not np.isnan(aopc_delta) else 0

                layer_rows[layer_name].append(
                    {
                        "sample_id": row["sample_id"],
                        "target_class": target_class_value,
                        "pred_class": int(row["pred_class"]),
                        "confidence": float(row["confidence"]),
                        "gradcam_layer": layer_name,
                        "artifact_path": artifact,
                        "border_ratio": float(ratios["border_ratio"]),
                        "retina_ratio": float(ratios["retina_ratio"]),
                        "border_ratio_raw": float(raw_ratios["border_ratio"]),
                        "retina_ratio_raw": float(raw_ratios["retina_ratio"]),
                        "faithfulness_delta": float(faith_legacy),
                        "gradcam_pass": pass_flag,
                        **faith_by_k,
                        "aopc_delta": float(aopc_delta),
                        "aopc_pass": int(aopc_pass),
                    }
                )
                if xai_log_every_samples > 0 and (
                    grad_job_idx == 1
                    or grad_job_idx % xai_log_every_samples == 0
                    or grad_job_idx == total_grad_jobs
                ):
                    elapsed_m = (time.time() - xai_start_time) / 60.0
                    print(
                        f"[XAI][GradCAM] job={grad_job_idx}/{total_grad_jobs} "
                        f"layer={layer_name} status=ok elapsed={elapsed_m:.1f}m"
                    )

        layer_scores: list[tuple[str, float, float, float, int]] = []
        for layer_name in supported_layers:
            rows = layer_rows.get(layer_name, [])
            if len(rows) == 0:
                layer_scores.append((layer_name, -np.inf, float("nan"), float("nan"), 0))
                continue
            layer_df = pd.DataFrame(rows)
            aopc_mean = float(pd.to_numeric(layer_df["aopc_delta"], errors="coerce").mean())
            border_mean = float(pd.to_numeric(layer_df["border_ratio"], errors="coerce").mean())
            if np.isnan(aopc_mean) or np.isnan(border_mean):
                score = -np.inf
            else:
                score = aopc_mean * (1.0 - float(np.clip(border_mean, 0.0, 1.0)))
            layer_scores.append((layer_name, score, aopc_mean, border_mean, int(len(rows))))

        selected_layer = default_gradcam_layer if default_gradcam_layer in supported_layers else supported_layers[0]
        if layer_scores:
            ranked_scores = sorted(layer_scores, key=lambda x: x[1], reverse=True)
            selected_layer = ranked_scores[0][0]
        grad_rows = layer_rows.get(selected_layer, [])

        layer_summary_path = Path(conf["paths"]["tables_dir"]) / f"gradcam_layer_selection_seed{seed}_{split}.csv"
        pd.DataFrame(
            layer_scores,
            columns=["gradcam_layer", "composite_score", "mean_aopc_delta", "mean_border_ratio", "n_rows"],
        ).to_csv(layer_summary_path, index=False)

        if len(grad_rows) == 0:
            reason = "No Grad-CAM rows were produced from supported layers."
            if layer_errors:
                reason = f"{reason} Errors: {layer_errors}"
            _save_json(
                gradcam_status_path,
                {
                    "status": "failed",
                    "seed": int(seed),
                    "run_id": run_id,
                    "split": split,
                    "backbone": backbone,
                    "requested_layers": gradcam_layers_eval,
                    "supported_layers": supported_layers,
                    "unsupported_layers": unsupported_layers,
                    "errors": layer_errors,
                    "selected_layer": selected_layer,
                    "reason": reason,
                    "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                    "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                    "evaluation_scope": evaluation_scope,
                    "shared_targets_enforced": bool(enforce_consistent_targets),
                    "target_rows": int(len(xai_target_df)),
                },
            )
            gradcam_status_note = f"failed: {reason}"
        else:
            status = "success" if len(layer_errors) == 0 and len(unsupported_layers) == 0 else "partial_success"
            _save_json(
                gradcam_status_path,
                {
                    "status": status,
                    "seed": int(seed),
                    "run_id": run_id,
                    "split": split,
                    "backbone": backbone,
                    "requested_layers": gradcam_layers_eval,
                    "supported_layers": supported_layers,
                    "unsupported_layers": unsupported_layers,
                    "errors": layer_errors,
                    "selected_layer": selected_layer,
                    "rows": int(len(grad_rows)),
                    "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                    "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                    "evaluation_scope": evaluation_scope,
                    "shared_targets_enforced": bool(enforce_consistent_targets),
                    "target_rows": int(len(xai_target_df)),
                },
            )
            gradcam_status_note = status

    if _use_legacy_aliases(conf):
        _write_alias_copy(gradcam_status_path, _legacy_gradcam_status_alias_path(conf, seed, split))

    rq1_df = pd.DataFrame(
        grad_rows,
        columns=[
            "sample_id",
            "target_class",
            "pred_class",
            "confidence",
            "gradcam_layer",
            "artifact_path",
            "border_ratio",
            "retina_ratio",
            "border_ratio_raw",
            "retina_ratio_raw",
            "faithfulness_delta",
            "gradcam_pass",
            *k_cols,
            "aopc_delta",
            "aopc_pass",
        ],
    )
    rq1_path = Path(conf["paths"]["tables_dir"]) / f"rq1_gradcam_seed{seed}_{split}.csv"
    rq1_df.to_csv(rq1_path, index=False)

    # SHAP (RQ2)
    shap_rows: list[dict[str, Any]] = []
    shap_status_path = _shap_status_log_path(conf, run_id, split)
    shap_status_note = ""
    shap_attempted_device = str(device)
    shap_final_device = str(device)
    shap_fallback_used = False
    shap_error_primary = ""
    shap_error = ""

    try:
        if _SHAP_IMPORT_ERROR is not None:
            raise RuntimeError(f"shap import failed: {_SHAP_IMPORT_ERROR}")

        bg_size = int(xai_cfg.get("shap_background_size", 64))
        shap_score_mode = str(xai_cfg.get("shap_score_mode", "positive")).strip().lower()
        if enforce_consistent_targets:
            shap_target = xai_target_df.copy()
        elif use_mode == "full":
            shap_target = target_df.copy()
        else:
            shap_target = target_df.head(max_samples).copy()

        if len(shap_target) == 0:
            raise RuntimeError("No target samples selected for SHAP")

        train_manifest = _manifest_path(conf, "train", seed=seed)
        train_df = pd.read_csv(train_manifest)
        if len(train_df) == 0:
            raise RuntimeError("Train manifest is empty; cannot build SHAP background")

        _per_class_bg = max(1, bg_size // int(conf["data"]["num_classes"]))
        bg_df = train_df.groupby("class_id", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), _per_class_bg), random_state=int(seed))
        ).reset_index(drop=True)
        if len(bg_df) < bg_size:
            extra = train_df.sample(n=bg_size - len(bg_df), random_state=int(seed))
            bg_df = pd.concat([bg_df, extra], ignore_index=True)
        bg_df = bg_df.head(bg_size)
        shap_dir = Path(conf["paths"]["figures_dir"]) / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)
        for _stale in shap_dir.glob("*.png"):
            _stale.unlink()

        def _run_shap_for_device(shap_device: torch.device) -> list[dict[str, Any]]:
            if str(shap_device) == str(device):
                shap_model = model
            else:
                shap_model = _load_model(conf, seed, shap_device, checkpoint=ckpt_path)
            _make_shap_compatible(shap_model)
            wrapper = _LogitWrapper(shap_model).to(shap_device)

            bg_tensors = []
            for _, b in bg_df.iterrows():
                _, bt = _load_image_for_inference(
                    b["image_path"],
                    image_size=image_size,
                    preprocessing_cfg=conf.get("preprocessing", {}),
                )
                bg_tensors.append(bt.squeeze(0))
            background = torch.stack(bg_tensors, dim=0).to(shap_device)
            explainer = _build_shap_explainer_with_known_warning_filter(wrapper, background)

            rows: list[dict[str, Any]] = []
            total_shap = int(len(shap_target))
            shap_start = time.time()
            for shap_idx, (_, row) in enumerate(shap_target.iterrows(), start=1):
                base_image, t = _load_image_for_inference(
                    row["image_path"],
                    image_size=image_size,
                    preprocessing_cfg=conf.get("preprocessing", {}),
                )
                t = t.to(shap_device)

                shap_values = _shap_values_with_known_warning_filter(explainer, t)
                smap = _pick_shap_map(shap_values, class_id=int(row["pred_class"]), sample_index=0)
                s2d = _shap_to_2d(smap, mode=shap_score_mode)
                raw_ratios = _attribution_mass_ratios(s2d)
                s2d = s2d * _attribution_retina_mask(s2d.shape, conf)
                target_class_value = int(row[target_class_col]) if target_class_col in row.index else int(row["pred_class"])

                out_path = shap_dir / f"{row['sample_id']}_cls{int(row['pred_class'])}.png"
                artifact = _save_map_overlay(s2d, np.array(base_image), out_path, overlay_dpi=fig_dpi)

                ratios = _attribution_mass_ratios(s2d)
                faith_by_k, aopc_delta = _faithfulness_multi_k(
                    model=shap_model,
                    image_tensor=t,
                    score_map=s2d,
                    pred_class=int(row["pred_class"]),
                    k_list=k_list,
                    random_seed=bootstrap_seed,
                )
                faith_legacy = float(faith_by_k.get("faith_delta_k20", np.nan))
                if np.isnan(faith_legacy):
                    faith_legacy = _faithfulness_delta(
                        model=shap_model,
                        image_tensor=t,
                        score_map=s2d,
                        pred_class=int(row["pred_class"]),
                        top_k_ratio=0.20,
                        random_seed=bootstrap_seed,
                    )
                pass_flag = _xai_pass_flag(
                    border_ratio=float(ratios["border_ratio"]),
                    faith_delta=float(faith_legacy),
                    border_ratio_max=pass_border_ratio_max,
                    faith_delta_min=pass_faith_delta_min,
                )
                aopc_pass = int(aopc_delta > 0.0) if not np.isnan(aopc_delta) else 0

                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "target_class": target_class_value,
                        "pred_class": int(row["pred_class"]),
                        "confidence": float(row["confidence"]),
                        "shap_score_mode": shap_score_mode,
                        "artifact_path": artifact,
                        "border_ratio": float(ratios["border_ratio"]),
                        "retina_ratio": float(ratios["retina_ratio"]),
                        "border_ratio_raw": float(raw_ratios["border_ratio"]),
                        "retina_ratio_raw": float(raw_ratios["retina_ratio"]),
                        "faithfulness_delta": float(faith_legacy),
                        "shap_pass": pass_flag,
                        **faith_by_k,
                        "aopc_delta": float(aopc_delta),
                        "aopc_pass": int(aopc_pass),
                    }
                )
                del shap_values, smap, s2d, t, base_image
                if getattr(shap_device, "type", "") == "mps":
                    _empty_mps_cache_if_available()
                    gc.collect()
                if xai_log_every_samples > 0 and (
                    shap_idx == 1
                    or shap_idx % xai_log_every_samples == 0
                    or shap_idx == total_shap
                ):
                    elapsed_m = (time.time() - shap_start) / 60.0
                    print(
                        f"[XAI][SHAP][{shap_device}] sample={shap_idx}/{total_shap} "
                        f"elapsed={elapsed_m:.1f}m"
                    )
            return rows

        try:
            shap_rows = _run_shap_for_device(device)
        except Exception as primary_exc:
            shap_error_primary = str(primary_exc)
            if _should_retry_shap_on_cpu(primary_exc, device):
                _empty_mps_cache_if_available()
                cpu_device = torch.device("cpu")
                print(f"[XAI][SHAP] retry on CPU after MPS failure: {shap_error_primary}")
                try:
                    shap_rows = _run_shap_for_device(cpu_device)
                    shap_final_device = str(cpu_device)
                    shap_fallback_used = True
                except Exception as fallback_exc:
                    shap_error = f"{fallback_exc} (primary_mps_error={shap_error_primary})"
            else:
                shap_error = shap_error_primary

    except Exception as exc:
        shap_error = str(exc)

    if shap_error:
        _save_json(
            shap_status_path,
            {
                "status": "failed",
                "seed": int(seed),
                "run_id": run_id,
                "split": split,
                "mode": str(shap_mode),
                "error": str(shap_error),
                "attempted_device": shap_attempted_device,
                "final_device": shap_final_device,
                "fallback_used": bool(shap_fallback_used),
                "error_primary": str(shap_error_primary),
                "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                "evaluation_scope": evaluation_scope,
                "shared_targets_enforced": bool(enforce_consistent_targets),
                "target_rows": int(len(xai_target_df)),
            },
        )
        shap_status_note = f"failed: {shap_error}"
    else:
        _save_json(
            shap_status_path,
            {
                "status": "success",
                "seed": int(seed),
                "run_id": run_id,
                "split": split,
                "mode": use_mode,
                "samples": int(len(shap_rows)),
                "attempted_device": shap_attempted_device,
                "final_device": shap_final_device,
                "fallback_used": bool(shap_fallback_used),
                "error_primary": str(shap_error_primary),
                "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
                "evaluation_scope": evaluation_scope,
                "shared_targets_enforced": bool(enforce_consistent_targets),
                "target_rows": int(len(xai_target_df)),
            },
        )
        shap_status_note = "success_cpu_fallback" if shap_fallback_used else "success"

    if _use_legacy_aliases(conf):
        _write_alias_copy(shap_status_path, _legacy_shap_status_alias_path(conf, seed, split))

    total_elapsed_m = (time.time() - xai_start_time) / 60.0
    print(
        f"[XAI] completed run_id={run_id} split={split} "
        f"gradcam_rows={len(grad_rows)} shap_rows={len(shap_rows)} "
        f"elapsed={total_elapsed_m:.1f}m"
    )

    rq2_df = pd.DataFrame(
        shap_rows,
        columns=[
            "sample_id",
            "target_class",
            "pred_class",
            "confidence",
            "shap_score_mode",
            "artifact_path",
            "border_ratio",
            "retina_ratio",
            "border_ratio_raw",
            "retina_ratio_raw",
            "faithfulness_delta",
            "shap_pass",
            *k_cols,
            "aopc_delta",
            "aopc_pass",
        ],
    )
    rq2_path = Path(conf["paths"]["tables_dir"]) / f"rq2_shap_seed{seed}_{split}.csv"
    rq2_df.to_csv(rq2_path, index=False)

    coverage_rows: list[dict[str, Any]] = []
    selected_ids = set(xai_target_df["sample_id"].astype(str).tolist()) if "sample_id" in xai_target_df.columns else set()
    grad_ids = set(rq1_df["sample_id"].astype(str).tolist()) if "sample_id" in rq1_df.columns else set()
    shap_ids = set(rq2_df["sample_id"].astype(str).tolist()) if "sample_id" in rq2_df.columns else set()
    for sid in sorted(selected_ids):
        coverage_rows.append(
            {
                "sample_id": sid,
                "selected_for_xai": 1,
                "gradcam_done": int(sid in grad_ids),
                "shap_done": int(sid in shap_ids),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows, columns=["sample_id", "selected_for_xai", "gradcam_done", "shap_done"])
    coverage_path = Path(conf["paths"]["tables_dir"]) / f"xai_target_coverage_seed{seed}_{split}.csv"
    coverage_df.to_csv(coverage_path, index=False)
    if len(coverage_df):
        grad_missing = int((coverage_df["gradcam_done"] == 0).sum())
        shap_missing = int((coverage_df["shap_done"] == 0).sum())
        if grad_missing > 0 or shap_missing > 0:
            print(
                f"[XAI][WARN] target coverage gap: "
                f"gradcam_missing={grad_missing}, shap_missing={shap_missing}"
            )

    method_stats_rows = [
        _build_xai_method_stats_row(
            method="gradcam",
            df=rq1_df,
            pass_col="gradcam_pass",
            n_boot=bootstrap_iters,
            seed=bootstrap_seed,
            border_ratio_max=pass_border_ratio_max,
            faith_delta_min=pass_faith_delta_min,
            evaluation_scope=evaluation_scope,
            status_note=gradcam_status_note or ("success" if len(rq1_df) else "no_rows"),
        ),
        _build_xai_method_stats_row(
            method="shap",
            df=rq2_df,
            pass_col="shap_pass",
            n_boot=bootstrap_iters,
            seed=bootstrap_seed,
            border_ratio_max=pass_border_ratio_max,
            faith_delta_min=pass_faith_delta_min,
            evaluation_scope=evaluation_scope,
            status_note=shap_status_note or ("success" if len(rq2_df) else "no_rows"),
        ),
    ]
    method_stats_df = pd.DataFrame(method_stats_rows)
    method_stats_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_method_stats_seed{seed}_{split}.csv"
    method_stats_df.to_csv(method_stats_path, index=False)

    class_names = [str(x) for x in conf.get("data", {}).get("label_order", [])]
    pass_by_correctness_df = pd.concat(
        [
            _build_xai_pass_by_correctness_table("gradcam", rq1_df, "gradcam_pass"),
            _build_xai_pass_by_correctness_table("shap", rq2_df, "shap_pass"),
        ],
        ignore_index=True,
    )
    pass_by_correctness_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_pass_by_correctness_seed{seed}_{split}.csv"
    pass_by_correctness_df.to_csv(pass_by_correctness_path, index=False)

    pass_by_class_df = pd.concat(
        [
            _build_xai_pass_by_class_table("gradcam", rq1_df, "gradcam_pass", class_names=class_names),
            _build_xai_pass_by_class_table("shap", rq2_df, "shap_pass", class_names=class_names),
        ],
        ignore_index=True,
    )
    pass_by_class_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_pass_by_class_seed{seed}_{split}.csv"
    pass_by_class_df.to_csv(pass_by_class_path, index=False)

    pairwise_df = _build_xai_pairwise_stats(rq1_df=rq1_df, rq2_df=rq2_df)
    pairwise_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_pairwise_seed{seed}_{split}.csv"
    pairwise_df.to_csv(pairwise_path, index=False)

    continuous_df = _build_xai_continuous_stats(rq1_df=rq1_df, rq2_df=rq2_df)
    continuous_path = Path(conf["paths"]["tables_dir"]) / f"rq_xai_continuous_seed{seed}_{split}.csv"
    continuous_df.to_csv(continuous_path, index=False)

    # ProtoPNet future-phase stub
    proto_stub = pd.DataFrame(
        [
            {
                "method": "ProtoPNetLite",
                "status": "planned_future_phase",
                "reason": "Planned future extension outside current implementation scope",
            }
        ]
    )
    proto_stub_path = Path(conf["paths"]["tables_dir"]) / f"protopnet_stub_seed{seed}_{split}.csv"
    proto_stub.to_csv(proto_stub_path, index=False)

    return {
        "run_id": run_id,
        "rq1_table": str(rq1_path),
        "rq2_table": str(rq2_path),
        "gradcam_layer_selection_table": str(layer_summary_path) if layer_summary_path is not None else "",
        "gradcam_status": str(gradcam_status_path),
        "rq_method_stats_table": str(method_stats_path),
        "rq_pass_by_correctness_table": str(pass_by_correctness_path),
        "rq_pass_by_class_table": str(pass_by_class_path),
        "rq_pairwise_table": str(pairwise_path),
        "rq_continuous_table": str(continuous_path),
        "shap_status": str(shap_status_path),
        "protopnet_stub": str(proto_stub_path),
        "xai_targets_table": str(xai_targets_path),
        "xai_target_coverage_table": str(coverage_path),
    }


def predict_single_image_with_explanations(cfg_path: str | Path = "configs/base.yaml", seed: int = 1988, image_path: str = "") -> dict[str, Any]:
    conf = _cfg(cfg_path)
    if not image_path:
        raise ValueError("image_path is required")

    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    fig_dpi = int(conf.get("xai", {}).get("figure_dpi", 180))
    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration table not found: {calibration_path}. Run calibration first.")

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    calibration_payload = _load_json(calibration_path)
    temperature = float(calibration_payload.get("temperature", 1.0))

    image, tensor = _load_image_for_inference(
        image_path,
        image_size=image_size,
        preprocessing_cfg=conf.get("preprocessing", {}),
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        logits = logits / max(1e-4, temperature)
        probs = torch.softmax(logits, dim=1)
        conf_score, pred = torch.max(probs, dim=1)

    confidence = float(conf_score.item())
    pred_class = int(pred.item())

    artifact = ""
    warning_message = ""
    try:
        out_path = Path(conf["paths"]["figures_dir"]) / "single" / f"{Path(image_path).stem}_gradcam.png"
        artifact, _, _ = _generate_gradcam(
            model=model,
            input_tensor=tensor,
            original_image=np.array(image),
            class_id=pred_class,
            layer_name=str(conf["xai"].get("gradcam_layer", "layer4")),
            output_path=out_path,
            device=device,
            conf=conf,
            overlay_dpi=fig_dpi,
            backbone_hint=backbone,
        )
    except Exception as exc:
        warning_message = f"Grad-CAM unavailable for backbone={backbone}: {exc}"

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "seed": int(seed),
        "run_id": run_id,
        "device": str(device),
        "pred_class": pred_class,
        "confidence": confidence,
        "artifact_path": artifact,
        "warning_message": warning_message,
    }

    probs_list = probs.squeeze(0).detach().cpu().numpy().tolist()
    for i, p in enumerate(probs_list):
        result[f"prob_{i}"] = float(p)

    return result


def explain_single_image_detailed(
    cfg_path: str | Path | dict[str, Any] = "configs/base.yaml",
    seed: int = 1988,
    image_path: str = "",
    gradcam_layers: list[str] | None = None,
    shap_background_size: int | None = None,
) -> dict[str, Any]:
    conf = _cfg(cfg_path)
    if not image_path:
        raise ValueError("image_path is required")

    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    fig_dpi = int(conf.get("xai", {}).get("figure_dpi", 180))
    pass_border_ratio_max, pass_faith_delta_min = _xai_pass_rule_thresholds(conf)
    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration table not found: {calibration_path}. Run calibration first.")

    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    calibration_payload = _load_json(calibration_path)
    temperature = float(calibration_payload.get("temperature", 1.0))

    image, tensor = _load_image_for_inference(
        image_path,
        image_size=image_size,
        preprocessing_cfg=conf.get("preprocessing", {}),
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        logits = logits / max(1e-4, temperature)
        probs = torch.softmax(logits, dim=1)
        conf_score, pred = torch.max(probs, dim=1)

    confidence = float(conf_score.item())
    pred_class = int(pred.item())

    warnings: list[str] = []
    gradcam_artifacts: dict[str, str] = {}
    gradcam_details: list[dict[str, Any]] = []
    default_gradcam_artifact = ""

    if gradcam_layers is None or len(gradcam_layers) == 0:
        gradcam_layers = ["layer2", "layer3", str(conf["xai"].get("gradcam_layer", "layer4"))]
    # preserve order while removing duplicates
    seen_layers: set[str] = set()
    unique_layers: list[str] = []
    for layer in gradcam_layers:
        key = str(layer).strip().lower()
        if not key or key in seen_layers:
            continue
        seen_layers.add(key)
        unique_layers.append(key)

    for layer in unique_layers:
        out_path = Path(conf["paths"]["figures_dir"]) / "single" / f"{Path(image_path).stem}_gradcam_{layer}.png"
        try:
            artifact, heat_input, _ = _generate_gradcam(
                model=model,
                input_tensor=tensor,
                original_image=np.array(image),
                class_id=pred_class,
                layer_name=layer,
                output_path=out_path,
                device=device,
                conf=conf,
                overlay_dpi=fig_dpi,
                backbone_hint=backbone,
            )
            if not default_gradcam_artifact:
                default_gradcam_artifact = artifact
            ratios = _attribution_mass_ratios(heat_input)
            faith_by_k, aopc_val = _faithfulness_multi_k(
                model=model,
                image_tensor=tensor,
                score_map=heat_input,
                pred_class=pred_class,
                k_list=(0.10, 0.20, 0.30),
            )
            faith = float(faith_by_k.get("faith_delta_k20", float("nan")))
            pass_flag = _xai_pass_flag(
                border_ratio=float(ratios["border_ratio"]),
                faith_delta=faith,
                border_ratio_max=pass_border_ratio_max,
                faith_delta_min=pass_faith_delta_min,
            )
            gradcam_artifacts[layer] = artifact
            gradcam_details.append(
                {
                    "layer": layer,
                    "artifact_path": artifact,
                    "border_ratio": float(ratios["border_ratio"]),
                    "retina_ratio": float(ratios["retina_ratio"]),
                    "faithfulness_delta": faith,
                    "faith_delta_k20": faith,
                    "aopc_delta": float(aopc_val) if aopc_val == aopc_val else float("nan"),
                    "gradcam_pass": pass_flag,
                }
            )
        except Exception as exc:
            warnings.append(f"Grad-CAM failed for layer={layer}: {exc}")

    shap_artifact_path = ""
    shap_details: dict[str, Any] = {
        "status": "skipped",
        "artifact_path": "",
        "border_ratio": np.nan,
        "retina_ratio": np.nan,
        "faithfulness_delta": np.nan,
        "shap_pass": np.nan,
        "error": "",
        "attempted_device": str(device),
        "final_device": str(device),
        "fallback_used": False,
        "error_primary": "",
    }

    try:
        if _SHAP_IMPORT_ERROR is not None:
            raise RuntimeError(f"shap import failed: {_SHAP_IMPORT_ERROR}")

        bg_size = int(shap_background_size or conf["xai"].get("shap_background_size", 64))
        bg_size = max(1, bg_size)

        train_manifest = _manifest_path(conf, "train", seed=seed)
        if not train_manifest.exists():
            prepare_data_manifests(conf, seed=seed)
        train_df = pd.read_csv(train_manifest)
        if len(train_df) == 0:
            raise RuntimeError("Train manifest is empty; cannot build SHAP background")

        per_class = max(1, bg_size // int(conf["data"]["num_classes"]))
        bg_df = train_df.groupby("class_id", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), per_class), random_state=int(seed))
        ).reset_index(drop=True)
        if len(bg_df) < bg_size:
            extra = train_df.sample(n=bg_size - len(bg_df), random_state=int(seed))
            bg_df = pd.concat([bg_df, extra], ignore_index=True)
        bg_df = bg_df.head(bg_size)

        bg_tensors: list[torch.Tensor] = []
        for _, b in bg_df.iterrows():
            _, bt = _load_image_for_inference(
                b["image_path"],
                image_size=image_size,
                preprocessing_cfg=conf.get("preprocessing", {}),
            )
            bg_tensors.append(bt.squeeze(0))
        background_cpu = torch.stack(bg_tensors, dim=0)

        def _run_single_shap(shap_device: torch.device) -> dict[str, Any]:
            if str(shap_device) == str(device):
                shap_model = model
            else:
                shap_model = _load_model(conf, seed, shap_device, checkpoint=ckpt_path)
            _make_shap_compatible(shap_model)
            wrapper = _LogitWrapper(shap_model).to(shap_device)
            background = background_cpu.to(shap_device)
            explainer = _build_shap_explainer_with_known_warning_filter(wrapper, background)
            shap_input = tensor.to(shap_device)
            shap_values = _shap_values_with_known_warning_filter(explainer, shap_input)
            smap = _pick_shap_map(shap_values, class_id=pred_class, sample_index=0)
            s2d = _shap_to_2d(smap)
            s2d = s2d * _attribution_retina_mask(s2d.shape, conf)

            shap_out_path = Path(conf["paths"]["figures_dir"]) / "single" / f"{Path(image_path).stem}_shap_cls{pred_class}.png"
            artifact_path = _save_map_overlay(s2d, np.array(image), shap_out_path, overlay_dpi=fig_dpi)
            shap_ratios = _attribution_mass_ratios(s2d)
            shap_faith_by_k, shap_aopc = _faithfulness_multi_k(
                model=shap_model,
                image_tensor=shap_input,
                score_map=s2d,
                pred_class=pred_class,
                k_list=(0.10, 0.20, 0.30),
            )
            shap_faith = float(shap_faith_by_k.get("faith_delta_k20", float("nan")))
            shap_pass = _xai_pass_flag(
                border_ratio=float(shap_ratios["border_ratio"]),
                faith_delta=shap_faith,
                border_ratio_max=pass_border_ratio_max,
                faith_delta_min=pass_faith_delta_min,
            )
            return {
                "status": "success",
                "artifact_path": artifact_path,
                "border_ratio": float(shap_ratios["border_ratio"]),
                "retina_ratio": float(shap_ratios["retina_ratio"]),
                "faithfulness_delta": shap_faith,
                "faith_delta_k20": shap_faith,
                "aopc_delta": float(shap_aopc) if shap_aopc == shap_aopc else float("nan"),
                "shap_pass": shap_pass,
                "error": "",
                "attempted_device": str(device),
                "final_device": str(shap_device),
                "fallback_used": bool(str(shap_device) != str(device)),
                "error_primary": "",
                "pass_rule_border_ratio_max": float(pass_border_ratio_max),
                "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
            }

        primary_error = ""
        try:
            shap_details = _run_single_shap(device)
        except Exception as primary_exc:
            primary_error = str(primary_exc)
            if _should_retry_shap_on_cpu(primary_exc, device):
                _empty_mps_cache_if_available()
                try:
                    shap_details = _run_single_shap(torch.device("cpu"))
                    shap_details["fallback_used"] = True
                    shap_details["error_primary"] = primary_error
                    warnings.append(f"SHAP fallback used CPU after MPS failure: {primary_error}")
                except Exception as fallback_exc:
                    raise RuntimeError(f"{fallback_exc} (primary_mps_error={primary_error})") from fallback_exc
            else:
                raise
        shap_artifact_path = str(shap_details.get("artifact_path", ""))
        if str(shap_details.get("final_device", "")) == "mps":
            _empty_mps_cache_if_available()
            gc.collect()
    except Exception as exc:
        shap_details = {
            "status": "failed",
            "artifact_path": "",
            "border_ratio": np.nan,
            "retina_ratio": np.nan,
            "faithfulness_delta": np.nan,
            "shap_pass": np.nan,
            "error": str(exc),
            "attempted_device": str(device),
            "final_device": str(device),
            "fallback_used": False,
            "error_primary": "",
            "pass_rule_border_ratio_max": float(pass_border_ratio_max),
            "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
        }
        warnings.append(f"SHAP failed: {exc}")

    result: dict[str, Any] = {
        "image_path": str(image_path),
        "seed": int(seed),
        "run_id": run_id,
        "device": str(device),
        "pred_class": pred_class,
        "confidence": confidence,
        "artifact_path": default_gradcam_artifact,
        "warning_message": "; ".join(warnings),
        "gradcam_artifacts": gradcam_artifacts,
        "gradcam_details": gradcam_details,
        "shap_artifact_path": shap_artifact_path,
        "shap_details": shap_details,
        "xai_warnings": warnings,
    }

    probs_list = probs.squeeze(0).detach().cpu().numpy().tolist()
    for i, p in enumerate(probs_list):
        result[f"prob_{i}"] = float(p)

    return result


def run_single_case_demo(
    cfg_path: str | Path | dict[str, Any] = "configs/base.yaml",
    seed: int = 1988,
    image_path: str = "",
    split: str = "test",
    gradcam_layers: list[str] | None = None,
    shap_background_size: int | None = None,
    shap_panel_size: tuple[float, float] = (3.2, 3.6),
    shap_dpi: int | None = None,
) -> dict[str, Any]:
    """
    Run a compact single-case demo flow for notebooks:
    1) pick one image (or use provided image_path),
    2) run detailed Grad-CAM + SHAP,
    3) export Grad-CAM panel and SHAP class grid,
    4) return a concise summary payload.
    """
    conf = _cfg(cfg_path)
    split_key = str(split).strip().lower() or "test"
    if split_key not in {"train", "val", "test"}:
        raise ValueError(f"split must be one of train/val/test, got: {split}")

    manifest_path = _manifest_path(conf, split_key, seed=seed)
    if not manifest_path.exists():
        prepare_data_manifests(conf, seed=seed)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for split={split_key}: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if len(manifest_df) == 0:
        raise RuntimeError(f"Empty manifest for split={split_key}: {manifest_path}")

    sampled_row: pd.Series | None = None
    if not image_path:
        sampled_row = manifest_df.sample(n=1, random_state=int(seed)).iloc[0]
        image_path = str(sampled_row["image_path"])

    img_path = Path(str(image_path))
    if not img_path.is_absolute():
        img_path = (Path(conf["project_root"]) / img_path).resolve()
    else:
        img_path = img_path.resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    true_class: int | None = None
    if sampled_row is not None:
        true_class = int(sampled_row["class_id"])
    else:
        # Best-effort lookup of true class from manifest for correctness border/captions.
        target = str(img_path)
        for _, row in manifest_df.iterrows():
            row_path = Path(str(row["image_path"]))
            if not row_path.is_absolute():
                row_path = (Path(conf["project_root"]) / row_path).resolve()
            else:
                row_path = row_path.resolve()
            if str(row_path) == target:
                true_class = int(row["class_id"])
                break

    requested_layers = gradcam_layers if gradcam_layers is not None else ["layer2", "layer3", str(conf["xai"].get("gradcam_layer", "layer4"))]
    seen_layers: set[str] = set()
    gradcam_layer_order: list[str] = []
    for layer in requested_layers:
        layer_key = str(layer).strip().lower()
        if not layer_key or layer_key in seen_layers:
            continue
        seen_layers.add(layer_key)
        gradcam_layer_order.append(layer_key)
    if not gradcam_layer_order:
        gradcam_layer_order = ["layer4"]

    single_result = explain_single_image_detailed(
        cfg_path=conf,
        seed=seed,
        image_path=str(img_path),
        gradcam_layers=gradcam_layer_order,
        shap_background_size=shap_background_size,
    )

    num_classes = int(conf["data"]["num_classes"])
    label_order = list(conf["data"]["label_order"])
    prob_table = pd.DataFrame(
        {
            "class_id": list(range(num_classes)),
            "class_name": label_order,
            "probability": [float(single_result.get(f"prob_{i}", 0.0)) for i in range(num_classes)],
        }
    ).sort_values("probability", ascending=False).reset_index(drop=True)

    pred_class = int(single_result.get("pred_class", -1))
    pred_label = label_order[pred_class] if 0 <= pred_class < len(label_order) else str(pred_class)
    true_label = label_order[true_class] if (true_class is not None and 0 <= true_class < len(label_order)) else None

    fig_dpi = int(conf.get("xai", {}).get("figure_dpi", 180))
    single_dir = Path(conf["paths"]["figures_dir"]) / "single"
    single_dir.mkdir(parents=True, exist_ok=True)
    for _stale in single_dir.glob(f"{img_path.stem}_*.png"):
        _stale.unlink()

    # Build one compact Grad-CAM panel: [Input | layer2 | layer3 | layer4].
    gradcam_panel_path = single_dir / f"{img_path.stem}_gradcam_panel.png"
    raw = Image.open(img_path).convert("RGB")
    model_view = _apply_fundus_preprocessing(
        raw,
        image_size=int(_model_image_size(conf)),
        preprocessing_cfg=conf.get("preprocessing", {}),
    )
    panels: list[tuple[str, np.ndarray]] = [("Input (Model View)", np.array(model_view))]
    for layer in gradcam_layer_order:
        artifact = str((single_result.get("gradcam_artifacts") or {}).get(layer, ""))
        if artifact and Path(artifact).exists():
            panels.append((f"Grad-CAM {layer}", np.asarray(plt.imread(artifact))))

    fig_g, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.0), facecolor="white")
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, image_arr) in zip(axes, panels):
        ax.imshow(image_arr)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig_g.tight_layout()
    fig_g.savefig(gradcam_panel_path, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig_g)

    bg_size = int(max(1, shap_background_size if shap_background_size is not None else conf.get("xai", {}).get("shap_background_size", 64)))
    out_dpi = int(shap_dpi if shap_dpi is not None else fig_dpi)
    shap_grid_path = single_dir / f"{img_path.stem}_shap_grid.png"
    fig_s = plot_shap_grid(
        cfg_path=conf,
        seed=seed,
        image_paths=[str(img_path)],
        background_size=bg_size,
        vmax_percentile=99.5,
        save_path=shap_grid_path,
        dpi=out_dpi,
        true_classes=[int(true_class)] if true_class is not None else None,
        show_correctness_border=bool(true_class is not None),
        panel_size=shap_panel_size,
    )
    plt.close(fig_s)

    return {
        "manifest_path": str(manifest_path),
        "image_path": str(img_path),
        "split": split_key,
        "seed": int(seed),
        "run_id": str(single_result.get("run_id", "")),
        "device": str(single_result.get("device", "")),
        "true_class": int(true_class) if true_class is not None else None,
        "true_label": true_label,
        "pred_class": pred_class,
        "pred_label": pred_label,
        "confidence": float(single_result.get("confidence", float("nan"))),
        "prob_table": prob_table,
        "gradcam_panel_path": str(gradcam_panel_path),
        "shap_grid_path": str(shap_grid_path),
        "single_result": single_result,
        "xai_warnings": list(single_result.get("xai_warnings") or []),
        "shap_background_size": bg_size,
    }


@torch.inference_mode()
def _predict_one_with_temperature(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> tuple[int, float, np.ndarray]:
    logits = model(input_tensor.to(device))
    logits = logits / max(1e-4, float(temperature))
    probs = torch.softmax(logits, dim=1)
    conf_score, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf_score.item()), probs.squeeze(0).detach().cpu().numpy()


def _temperature_for_run(conf: dict[str, Any], run_id: str) -> float:
    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        return 1.0
    payload = _load_json(calibration_path)
    return float(payload.get("temperature", 1.0))


def _gradcam_heatmap_for_display(
    model: DRClassifier,
    input_tensor: torch.Tensor,
    class_id: int,
    layer_name: str,
    output_size: tuple[int, int],
    device: torch.device,
    backbone_hint: str = "",
) -> np.ndarray:
    if _CAPTUM_IMPORT_ERROR is not None:
        raise RuntimeError(f"captum import failed: {_CAPTUM_IMPORT_ERROR}")

    target_layer, _, reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=layer_name,
        backbone_hint=backbone_hint,
    )
    if target_layer is None:
        raise ValueError(reason or f"Unable to resolve Grad-CAM layer '{layer_name}'.")

    gradcam = LayerGradCam(lambda x: model(x), target_layer)
    attr = gradcam.attribute(input_tensor.to(device), target=int(class_id))
    attr_input = LayerAttribution.interpolate(attr, (int(input_tensor.shape[-2]), int(input_tensor.shape[-1])))
    heat = attr_input.squeeze().detach().cpu().numpy()
    if heat.ndim == 3:
        heat = np.mean(heat, axis=0)
    heat = _normalize_map(heat)
    out_w, out_h = int(output_size[0]), int(output_size[1])
    heat = np.array(
        Image.fromarray(heat.astype(np.float32), mode="F").resize((out_w, out_h), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return _normalize_map(heat)


def plot_gradcam_grid(
    cfg_path: str | Path | dict[str, Any],
    seed: int,
    samples: list[tuple[str, int]],
    ncols: int = 2,
    gradcam_layer: str = "layer4",
    cmap_name: str = "jet",
    alpha: float = 0.45,
    figsize_per_pair: tuple[float, float] = (5.0, 2.8),
    save_path: str | Path | None = None,
    dpi: int | None = None,
) -> plt.Figure:
    """
    Publication-style Grad-CAM grid:
    [Original | Grad-CAM] pairs with green/red border by correctness.
    """
    if len(samples) == 0:
        raise ValueError("samples must not be empty")

    conf = _cfg(cfg_path)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    backbone = _backbone_name(conf)
    label_order = list(conf["data"]["label_order"])
    out_dpi = int(dpi if dpi is not None else conf.get("xai", {}).get("figure_dpi", 180))

    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    temperature = _temperature_for_run(conf, run_id)
    _, _, gradcam_reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=gradcam_layer,
        backbone_hint=backbone,
    )
    if gradcam_reason:
        raise ValueError(f"plot_gradcam_grid cannot run Grad-CAM for backbone={backbone}: {gradcam_reason}")

    n = len(samples)
    cols = max(1, int(ncols))
    nrows = int(np.ceil(n / cols))

    fig_w = float(figsize_per_pair[0]) * cols * 2
    fig_h = float(figsize_per_pair[1]) * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    outer = gridspec.GridSpec(nrows, cols, figure=fig, hspace=0.45, wspace=0.08)

    for idx, (image_path, true_class) in enumerate(samples):
        row, col = divmod(idx, cols)
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col], wspace=0.03)
        ax_orig = fig.add_subplot(inner[0])
        ax_cam = fig.add_subplot(inner[1])

        image_pil, tensor = _load_image_for_inference(
            image_path,
            image_size=image_size,
            preprocessing_cfg=conf.get("preprocessing", {}),
        )
        pred_class, confidence, _ = _predict_one_with_temperature(model, tensor, device, temperature)
        base = np.array(image_pil)
        heat = _gradcam_heatmap_for_display(
            model=model,
            input_tensor=tensor,
            class_id=pred_class,
            layer_name=gradcam_layer,
            output_size=(base.shape[1], base.shape[0]),
            device=device,
            backbone_hint=backbone,
        )
        overlay = _overlay(base, heat, alpha=float(alpha), cmap_name=str(cmap_name))

        ax_orig.imshow(base)
        ax_orig.set_title("Original Image", fontsize=7.5, pad=3)
        ax_orig.axis("off")

        ax_cam.imshow(overlay)
        ax_cam.set_title("Grad-CAM", fontsize=7.5, pad=3)
        ax_cam.axis("off")

        correct = int(pred_class) == int(true_class)
        border_color = "#27ae60" if correct else "#e74c3c"
        p1 = ax_orig.get_position()
        p2 = ax_cam.get_position()
        x0 = min(p1.x0, p2.x0)
        y0 = min(p1.y0, p2.y0)
        x1 = max(p1.x1, p2.x1)
        y1 = max(p1.y1, p2.y1)
        pad = 0.004
        rect = mpatches.Rectangle(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2 * pad,
            (y1 - y0) + 2 * pad,
            fill=False,
            edgecolor=border_color,
            linewidth=1,
            transform=fig.transFigure,
            clip_on=False,
            zorder=20,
        )
        fig.add_artist(rect)

        true_label = label_order[int(true_class)] if 0 <= int(true_class) < len(label_order) else str(true_class)
        pred_label = label_order[int(pred_class)] if 0 <= int(pred_class) < len(label_order) else str(pred_class)
        inner_ax = fig.add_subplot(outer[row, col])
        inner_ax.set_axis_off()
        inner_ax.text(
            0.5,
            -0.10,
            (
                f"True label: {true_label} ({int(true_class)}),  "
                f"Predicted label: {pred_label} ({int(pred_class)}),  "
                f"Pred prob: {float(confidence):.3f}"
            ),
            transform=inner_ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color="black",
            fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(facecolor="#27ae60", label="Correct prediction"),
        mpatches.Patch(facecolor="#e74c3c", label="Wrong prediction"),
    ]
    fig.subplots_adjust(bottom=0.16)
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(
        0.5,
        0.005,
        "Figure: Grad-CAM Visualization. Green border = correct prediction, Red border = wrong prediction.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#444",
    )

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=out_dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")
    return fig


def plot_shap_grid(
    cfg_path: str | Path | dict[str, Any],
    seed: int,
    image_paths: list[str],
    background_size: int | None = None,
    vmax_percentile: float = 99.5,
    shap_cmap: str = "RdBu_r",
    bg_cmap: str = "gray",
    bg_alpha: float = 0.35,
    shap_alpha: float = 0.85,
    save_path: str | Path | None = None,
    dpi: int | None = None,
    true_classes: list[int] | None = None,
    show_correctness_border: bool = False,
    panel_size: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot SHAP per class for each input image.
    Layout: one row per image, columns [original, class0, class1, ...].
    """
    if _SHAP_IMPORT_ERROR is not None or shap is None:
        raise RuntimeError(f"shap import failed: {_SHAP_IMPORT_ERROR}")
    if len(image_paths) == 0:
        raise ValueError("image_paths must not be empty")
    if true_classes is not None and len(true_classes) != len(image_paths):
        raise ValueError(
            f"true_classes length ({len(true_classes)}) must match image_paths length ({len(image_paths)})."
        )

    from matplotlib.colors import TwoSlopeNorm

    conf = _cfg(cfg_path)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    num_classes = int(conf["data"]["num_classes"])
    label_order = list(conf["data"]["label_order"])
    out_dpi = int(dpi if dpi is not None else conf.get("xai", {}).get("figure_dpi", 180))

    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    temperature = _temperature_for_run(conf, run_id)

    train_df = pd.read_csv(_manifest_path(conf, "train", seed=seed))
    bg_default = int(conf.get("xai", {}).get("shap_background_size", 64))
    bg_size = int(max(1, background_size if background_size is not None else bg_default))
    per_class = max(1, bg_size // max(1, num_classes))
    bg_df = train_df.groupby("class_id", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), per_class), random_state=int(seed))
    ).reset_index(drop=True)
    if len(bg_df) < bg_size:
        extra = train_df.sample(n=bg_size - len(bg_df), random_state=int(seed))
        bg_df = pd.concat([bg_df, extra], ignore_index=True)
    bg_df = bg_df.head(bg_size)

    bg_tensors: list[torch.Tensor] = []
    for _, row in bg_df.iterrows():
        _, bt = _load_image_for_inference(
            row["image_path"],
            image_size=image_size,
            preprocessing_cfg=conf.get("preprocessing", {}),
        )
        bg_tensors.append(bt.squeeze(0))
    background_cpu = torch.stack(bg_tensors, dim=0)

    def _collect_rows_for_device(shap_device: torch.device) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        if str(shap_device) == str(device):
            shap_model = model
        else:
            shap_model = _load_model(conf, seed, shap_device, checkpoint=ckpt_path)
        _make_shap_compatible(shap_model)
        wrapper = _LogitWrapper(shap_model).to(shap_device)
        explainer = _build_shap_explainer_with_known_warning_filter(wrapper, background_cpu.to(shap_device))

        rows: list[dict[str, Any]] = []
        vals: list[np.ndarray] = []
        for img_idx, image_path in enumerate(image_paths):
            image_pil, tensor = _load_image_for_inference(
                image_path,
                image_size=image_size,
                preprocessing_cfg=conf.get("preprocessing", {}),
            )
            pred_class, confidence, _ = _predict_one_with_temperature(model, tensor, device, temperature)
            shap_values = _shap_values_with_known_warning_filter(explainer, tensor.to(shap_device))

            class_maps: list[np.ndarray] = []
            for cls_idx in range(num_classes):
                smap = _pick_shap_map(shap_values, class_id=cls_idx, sample_index=0)
                arr = np.asarray(smap, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(0)
                class_maps.append(arr)
                vals.append(np.abs(arr.ravel()))

            rows.append(
                {
                    "image_path": str(image_path),
                    "image_pil": image_pil,
                    "pred_class": int(pred_class),
                    "true_class": int(true_classes[img_idx]) if true_classes is not None else None,
                    "confidence": float(confidence),
                    "class_maps": class_maps,
                }
            )
            del shap_values, class_maps
            if getattr(shap_device, "type", "") == "mps":
                _empty_mps_cache_if_available()
                gc.collect()
        return rows, vals

    shap_final_device = str(device)
    shap_fallback_used = False
    shap_primary_error = ""
    try:
        rows_data, all_vals = _collect_rows_for_device(device)
    except Exception as primary_exc:
        shap_primary_error = str(primary_exc)
        if _should_retry_shap_on_cpu(primary_exc, device):
            _empty_mps_cache_if_available()
            try:
                rows_data, all_vals = _collect_rows_for_device(torch.device("cpu"))
                shap_final_device = "cpu"
                shap_fallback_used = True
            except Exception as fallback_exc:
                raise RuntimeError(f"{fallback_exc} (primary_mps_error={shap_primary_error})") from fallback_exc
        else:
            raise

    vmax = 1e-6
    if all_vals:
        vmax = max(float(np.percentile(np.concatenate(all_vals), float(vmax_percentile))), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap(str(shap_cmap))

    nrows = len(rows_data)
    ncols = 1 + num_classes
    if panel_size is None:
        panel_w, panel_h = 2.2, 2.8
    else:
        panel_w = float(panel_size[0])
        panel_h = float(panel_size[1])
        panel_w = max(1.5, panel_w)
        panel_h = max(2.0, panel_h)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), facecolor="white")
    axes = np.array(axes).reshape(nrows, ncols)

    im = None
    row_border_specs: list[tuple[int, str]] = []
    for r, row in enumerate(rows_data):
        image_pil = row["image_pil"]
        pred_class = int(row["pred_class"])
        true_class_raw = row.get("true_class")
        true_class = int(true_class_raw) if true_class_raw is not None else None
        confidence = float(row["confidence"])
        class_maps = row["class_maps"]

        axes[r, 0].imshow(np.array(image_pil.convert("L")), cmap=str(bg_cmap))
        axes[r, 0].axis("off")
        axes[r, 0].set_title("Original", fontsize=8)
        if true_class is None:
            caption = f"pred={label_order[pred_class]} conf={confidence:.3f}"
        else:
            true_label = label_order[int(true_class)] if 0 <= int(true_class) < len(label_order) else str(true_class)
            pred_label = label_order[int(pred_class)] if 0 <= int(pred_class) < len(label_order) else str(pred_class)
            is_correct = int(pred_class) == int(true_class)
            verdict = "correct" if is_correct else "wrong"
            caption = f"true={true_label} pred={pred_label} ({verdict}) conf={confidence:.3f}"
            if show_correctness_border:
                row_border_specs.append((r, "#27ae60" if is_correct else "#e74c3c"))
        axes[r, 0].text(
            0.5,
            -0.08,
            caption,
            transform=axes[r, 0].transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )

        for cls_idx, smap in enumerate(class_maps):
            ax = axes[r, cls_idx + 1]
            bg = np.array(
                image_pil.convert("L").resize((smap.shape[1], smap.shape[0]), Image.Resampling.BILINEAR)
            )
            ax.imshow(bg, cmap=str(bg_cmap), alpha=float(bg_alpha), vmin=0, vmax=255, interpolation="bilinear")
            im = ax.imshow(smap, cmap=cmap, norm=norm, alpha=float(shap_alpha), interpolation="nearest")
            ax.axis("off")
            score = 1.0 if cls_idx == pred_class else 0.0
            ax.set_title(f"{score:.1f}", fontsize=8, color="#e74c3c" if cls_idx == pred_class else "#333")

    if im is not None:
        cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.02])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, orientation="horizontal")
        cb.set_label("SHAP value", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # Avoid tight_layout() here because we manually add a colorbar axes.
    # tight_layout triggers a noisy warning and can produce unstable geometry.
    bottom_margin = 0.09 if im is not None else 0.05
    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=bottom_margin, wspace=0.02, hspace=0.32)
    if show_correctness_border and row_border_specs:
        for row_idx, border_color in row_border_specs:
            p_first = axes[row_idx, 0].get_position()
            p_last = axes[row_idx, ncols - 1].get_position()
            x0 = min(p_first.x0, p_last.x0)
            y0 = min(p_first.y0, p_last.y0)
            x1 = max(p_first.x1, p_last.x1)
            y1 = max(p_first.y1, p_last.y1)
            pad = 0.0025
            rect = mpatches.Rectangle(
                (x0 - pad, y0 - pad),
                (x1 - x0) + 2 * pad,
                (y1 - y0) + 2 * pad,
                fill=False,
                edgecolor=border_color,
                linewidth=1,
                transform=fig.transFigure,
                clip_on=False,
                zorder=20,
            )
            fig.add_artist(rect)
        fig.text(
            0.5,
            0.005,
            "Green border = correct prediction, Red border = wrong prediction.",
            ha="center",
            fontsize=8,
            style="italic",
            color="#444",
        )
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=out_dpi, bbox_inches="tight", facecolor="white")
    if shap_fallback_used:
        print(f"SHAP fallback used: attempted_device={device}, final_device={shap_final_device}, primary_error={shap_primary_error}")
    return fig


def notebook_run_xai(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    manifests: dict[str, str] | None = None,
    checkpoint: str | Path | None = None,
    safe_mode: bool = False,
) -> dict[str, Any]:
    conf = copy.deepcopy(_cfg(cfg_or_path))
    split_key = str(split).strip().lower() or "test"
    xai_cfg = conf.setdefault("xai", {})

    if safe_mode:
        xai_cfg["device"] = "cpu"
        xai_cfg["max_targets"] = min(int(xai_cfg.get("max_targets", 128)), 24)
        xai_cfg["shap_max_samples"] = min(int(xai_cfg.get("shap_max_samples", 128)), 24)
        xai_cfg["shap_background_size"] = min(int(xai_cfg.get("shap_background_size", 64)), 12)
    else:
        xai_cfg.setdefault("device", conf.get("training", {}).get("device", "mps"))
    xai_cfg["log_every_samples"] = max(1, int(xai_cfg.get("log_every_samples", 16)))

    calibration_path = build_validation_calibration_table(
        conf,
        seed=int(seed),
        manifests=manifests,
        checkpoint=checkpoint,
    )
    xai_outputs = run_xai_analysis(
        conf,
        seed=int(seed),
        split=split_key,
        checkpoint=checkpoint,
        predictions_csv=None,
        shap_mode=str(xai_cfg.get("shap_mode", "subset")),
        shap_max_samples=int(xai_cfg.get("shap_max_samples", 64)),
    )

    run_id = str(xai_outputs.get("run_id", ""))
    if not run_id:
        latest_payload = _load_latest_run_record(conf, int(seed))
        if latest_payload:
            run_id = str(latest_payload.get("run_id", ""))

    pred_primary = (
        Path(conf["paths"]["predictions_dir"]) / f"{run_id}_{split_key}_predictions.csv"
        if run_id
        else Path("")
    )
    pred_alias = Path(conf["paths"]["predictions_dir"]) / f"predictions_seed{int(seed)}_{split_key}.csv"
    predictions_path = pred_primary if pred_primary.exists() else pred_alias
    predictions_df = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

    return {
        "cfg_xai": conf,
        "seed": int(seed),
        "split": split_key,
        "safe_mode": bool(safe_mode),
        "xai_runtime_settings": {
            "device": str(xai_cfg.get("device")),
            "max_targets": int(xai_cfg.get("max_targets", 0)),
            "shap_max_samples": int(xai_cfg.get("shap_max_samples", 0)),
            "shap_background_size": int(xai_cfg.get("shap_background_size", 0)),
            "log_every_samples": int(xai_cfg.get("log_every_samples", 1)),
        },
        "calibration_path": str(calibration_path),
        "xai_outputs": xai_outputs,
        "run_id": run_id,
        "predictions_path": str(predictions_path),
        "predictions_df": predictions_df,
    }


def notebook_load_xai_committee_summary(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    split_key = str(split).strip().lower() or "test"
    tables_dir = Path(conf["paths"]["tables_dir"])

    method_path = tables_dir / f"rq_xai_method_stats_seed{int(seed)}_{split_key}.csv"
    pairwise_path = tables_dir / f"rq_xai_pairwise_seed{int(seed)}_{split_key}.csv"
    continuous_path = tables_dir / f"rq_xai_continuous_seed{int(seed)}_{split_key}.csv"
    coverage_path = tables_dir / f"xai_target_coverage_seed{int(seed)}_{split_key}.csv"
    targets_path = tables_dir / f"xai_targets_seed{int(seed)}_{split_key}.csv"
    correctness_path = tables_dir / f"rq_xai_pass_by_correctness_seed{int(seed)}_{split_key}.csv"

    method_stats_df = pd.read_csv(method_path) if method_path.exists() else pd.DataFrame()
    pairwise_df = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    continuous_df = pd.read_csv(continuous_path) if continuous_path.exists() else pd.DataFrame()
    coverage_df = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    targets_df = pd.read_csv(targets_path) if targets_path.exists() else pd.DataFrame()
    correctness_df = pd.read_csv(correctness_path) if correctness_path.exists() else pd.DataFrame()

    if method_stats_df.empty:
        raise RuntimeError(f"Missing or empty method stats table: {method_path}")

    plot_df = method_stats_df[method_stats_df["method"].isin(["gradcam", "shap"])].copy()
    plot_df = plot_df.set_index("method").reindex(["gradcam", "shap"]).reset_index()
    plot_df["method"] = plot_df["method"].replace({"gradcam": "Grad-CAM", "shap": "SHAP"})

    for col in [
        "n",
        "pass_rate",
        "pass_rate_ci95_low",
        "pass_rate_ci95_high",
        "mean_border_ratio",
        "mean_faith_delta_k20",
        "pass_rule_border_ratio_max",
        "pass_rule_faith_delta_k20_min",
    ]:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    summary_tbl = plot_df[
        [
            "method",
            "n",
            "pass_rate",
            "pass_rate_ci95_low",
            "pass_rate_ci95_high",
            "mean_border_ratio",
            "mean_faith_delta_k20",
        ]
    ].copy()
    summary_tbl.columns = [
        "Method",
        "N",
        "Rule-Satisfied (descriptive)",
        "CI95 Low",
        "CI95 High",
        "Mean Border Ratio",
        "Mean Faithfulness (k=20%)",
    ]
    summary_tbl["Rule-Satisfied (descriptive)"] = (
        pd.to_numeric(summary_tbl["Rule-Satisfied (descriptive)"], errors="coerce") * 100.0
    ).round(1).astype(str) + "%"
    summary_tbl["CI95 Low"] = (
        pd.to_numeric(summary_tbl["CI95 Low"], errors="coerce") * 100.0
    ).round(1).astype(str) + "%"
    summary_tbl["CI95 High"] = (
        pd.to_numeric(summary_tbl["CI95 High"], errors="coerce") * 100.0
    ).round(1).astype(str) + "%"
    summary_tbl["Mean Border Ratio"] = pd.to_numeric(summary_tbl["Mean Border Ratio"], errors="coerce").round(3)
    summary_tbl["Mean Faithfulness (k=20%)"] = pd.to_numeric(
        summary_tbl["Mean Faithfulness (k=20%)"], errors="coerce"
    ).round(3)

    rate = pd.to_numeric(plot_df["pass_rate"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(plot_df["pass_rate_ci95_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(plot_df["pass_rate_ci95_high"], errors="coerce").to_numpy(dtype=float)
    err_low = np.clip(rate - low, 0.0, None)
    err_high = np.clip(high - rate, 0.0, None)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(plot_df["method"], rate, yerr=[err_low, err_high], capsize=6, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion satisfying threshold rule")
    ax.set_title("Operational Threshold Summary (Descriptive Only)")
    for i, value in enumerate(rate):
        if np.isfinite(value):
            ax.text(i, min(float(value) + 0.03, 0.98), f"{float(value) * 100.0:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()

    delta_text = "N/A"
    p_text = "N/A"
    if len(pairwise_df):
        row = pairwise_df.iloc[0]
        delta = pd.to_numeric(row.get("delta_pass_rate", np.nan), errors="coerce")
        pval = pd.to_numeric(row.get("mcnemar_pvalue_exact", np.nan), errors="coerce")
        if np.isfinite(delta):
            delta_text = f"{float(delta) * 100.0:.1f}%"
        if np.isfinite(pval):
            p_text = f"{float(pval):.4f}"

    coverage_text = "coverage file not found"
    if len(coverage_df):
        g_miss = int((pd.to_numeric(coverage_df.get("gradcam_done", pd.Series(dtype=float)), errors="coerce") == 0).sum()) if "gradcam_done" in coverage_df.columns else -1
        s_miss = int((pd.to_numeric(coverage_df.get("shap_done", pd.Series(dtype=float)), errors="coerce") == 0).sum()) if "shap_done" in coverage_df.columns else -1
        coverage_text = f"coverage check: gradcam_missing={g_miss}, shap_missing={s_miss}"

    scope_text = "scope unavailable"
    if "evaluation_scope" in method_stats_df.columns and len(method_stats_df):
        non_null_scope = method_stats_df["evaluation_scope"].dropna()
        if len(non_null_scope):
            scope_text = str(non_null_scope.iloc[0])

    rule_border = 0.35
    rule_faith = 0.0
    if "pass_rule_border_ratio_max" in method_stats_df.columns and len(method_stats_df):
        vals = pd.to_numeric(method_stats_df["pass_rule_border_ratio_max"], errors="coerce").dropna()
        if len(vals):
            rule_border = float(vals.iloc[0])
    if "pass_rule_faith_delta_k20_min" in method_stats_df.columns and len(method_stats_df):
        vals = pd.to_numeric(method_stats_df["pass_rule_faith_delta_k20_min"], errors="coerce").dropna()
        if len(vals):
            rule_faith = float(vals.iloc[0])

    audit_n = int(len(targets_df))
    correctness_note = "correct/wrong split table unavailable"
    if len(correctness_df):
        correctness_note = "correct/wrong split is provided in Section 7 advanced table"

    # Determine pass-rate direction live from the pairwise rates. `delta_pass_rate`
    # in the CSV is `shap_pass_rate - gradcam_pass_rate`, but the column order can
    # shift across runs, so read the per-method rates directly and compute locally.
    if len(pairwise_df):
        g_rate = pd.to_numeric(pairwise_df.iloc[0].get("gradcam_pass_rate", np.nan), errors="coerce")
        s_rate = pd.to_numeric(pairwise_df.iloc[0].get("shap_pass_rate", np.nan), errors="coerce")
    else:
        g_rate = s_rate = np.nan
    if np.isfinite(g_rate) and np.isfinite(s_rate) and abs(float(g_rate) - float(s_rate)) > 1e-6:
        if float(g_rate) > float(s_rate):
            direction_text = "Grad-CAM > SHAP"
        else:
            direction_text = "SHAP > Grad-CAM"
    else:
        direction_text = "methods equivalent"

    # Use absolute magnitude in the narrative so the direction word and the
    # number don't appear to contradict each other (the CSV signs delta as
    # shap − gradcam; direction is carried by `direction_text`).
    delta_mag_text = "N/A"
    if np.isfinite(delta_raw := (pd.to_numeric(pairwise_df.iloc[0].get("delta_pass_rate", np.nan), errors="coerce")
                                  if len(pairwise_df) else np.nan)):
        delta_mag_text = f"{abs(float(delta_raw)) * 100.0:.1f} pp"
    bottom_line_markdown = (
        f"> **Operational threshold summary (descriptive only):** under the project-specific rule "
        f"`border_ratio \u2264 {rule_border:.2f}` AND `\u0394_k20 > {rule_faith:.2f}`, "
        f"{direction_text} on pass rate (gap **{delta_mag_text}**, McNemar exact p=**{p_text}**).\n"
        f"> **Caution:** the thresholded pass rate compresses continuous evidence, discards "
        f"magnitude information, and can introduce boundary effects around arbitrary cut-offs; "
        f"refer to the primary continuous analysis for the main finding.\n"
        f"> **Audit scope:** `{scope_text}` with `N={audit_n}` targets.\n"
        f"> **Consistency check:** {coverage_text}; {correctness_note}."
    )

    continuous_table = _format_continuous_table_for_display(continuous_df)
    continuous_bottom_line = _format_continuous_bottom_line(continuous_df)

    return {
        "cfg": conf,
        "seed": int(seed),
        "split": split_key,
        "method_stats_path": str(method_path),
        "pairwise_path": str(pairwise_path),
        "continuous_path": str(continuous_path),
        "coverage_path": str(coverage_path),
        "targets_path": str(targets_path),
        "correctness_path": str(correctness_path),
        "method_stats_df": method_stats_df,
        "pairwise_df": pairwise_df,
        "continuous_df": continuous_df,
        "continuous_table": continuous_table,
        "continuous_bottom_line_markdown": continuous_bottom_line,
        "coverage_df": coverage_df,
        "targets_df": targets_df,
        "correctness_df": correctness_df,
        "summary_table": summary_tbl,
        "pass_rate_fig": fig,
        "bottom_line_markdown": bottom_line_markdown,
    }


def notebook_load_xai_advanced_audit(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    xai_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    conf = _cfg(cfg_or_path)
    split_key = str(split).strip().lower() or "test"
    tables_dir = Path(conf["paths"]["tables_dir"])

    rq1_path = tables_dir / f"rq1_gradcam_seed{int(seed)}_{split_key}.csv"
    rq2_path = tables_dir / f"rq2_shap_seed{int(seed)}_{split_key}.csv"
    pair_path = tables_dir / f"rq_xai_pairwise_seed{int(seed)}_{split_key}.csv"
    pass_by_class_path = tables_dir / f"rq_xai_pass_by_class_seed{int(seed)}_{split_key}.csv"
    pass_by_correctness_path = tables_dir / f"rq_xai_pass_by_correctness_seed{int(seed)}_{split_key}.csv"

    rq1_df = pd.read_csv(rq1_path) if rq1_path.exists() else pd.DataFrame()
    rq2_df = pd.read_csv(rq2_path) if rq2_path.exists() else pd.DataFrame()
    pair_df = pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    classwise_df = pd.read_csv(pass_by_class_path) if pass_by_class_path.exists() else pd.DataFrame()
    correctness_df = pd.read_csv(pass_by_correctness_path) if pass_by_correctness_path.exists() else pd.DataFrame()

    correctness_view = pd.DataFrame()
    if len(correctness_df):
        correctness_view = correctness_df.copy()
        correctness_view["pass_rate"] = pd.to_numeric(correctness_view["pass_rate"], errors="coerce").round(3)
        correctness_view["mean_border_ratio"] = pd.to_numeric(correctness_view["mean_border_ratio"], errors="coerce").round(3)
        correctness_view["mean_faith_delta_k20"] = pd.to_numeric(correctness_view["mean_faith_delta_k20"], errors="coerce").round(3)
        correctness_view = correctness_view.sort_values(["method", "group"]).reset_index(drop=True)

    classwise_view = pd.DataFrame()
    if len(classwise_df):
        classwise_view = classwise_df.copy()
        classwise_view["pass_rate"] = pd.to_numeric(classwise_view["pass_rate"], errors="coerce").round(3)
        classwise_view = classwise_view.sort_values(["method", "class_id"]).reset_index(drop=True)

    discord_df = pd.DataFrame()
    if (
        len(rq1_df)
        and len(rq2_df)
        and {"sample_id", "gradcam_pass"}.issubset(rq1_df.columns)
        and {"sample_id", "shap_pass"}.issubset(rq2_df.columns)
    ):
        discord_df = rq1_df[
            ["sample_id", "target_class", "pred_class", "confidence", "gradcam_pass"]
        ].merge(
            rq2_df[["sample_id", "shap_pass"]],
            on="sample_id",
            how="inner",
        )
        discord_df = discord_df[
            discord_df["gradcam_pass"].astype(int) != discord_df["shap_pass"].astype(int)
        ].sort_values("sample_id").reset_index(drop=True)

    def _load_status_payload(path_like: str | Path | None) -> dict[str, Any]:
        if path_like is None:
            return {}
        p = Path(str(path_like))
        if not p.exists():
            return {}
        try:
            return _load_json(p)
        except Exception:
            return {}

    run_id = ""
    if isinstance(xai_outputs, dict):
        run_id = str(xai_outputs.get("run_id", "") or "")
    if not run_id:
        latest_payload = _load_latest_run_record(conf, int(seed))
        if latest_payload:
            run_id = str(latest_payload.get("run_id", "") or "")

    grad_status_path: Path | None = None
    shap_status_path: Path | None = None
    if isinstance(xai_outputs, dict):
        grad_raw = str(xai_outputs.get("gradcam_status", "") or "").strip()
        shap_raw = str(xai_outputs.get("shap_status", "") or "").strip()
        grad_status_path = Path(grad_raw) if grad_raw else None
        shap_status_path = Path(shap_raw) if shap_raw else None
    if grad_status_path is None and run_id:
        grad_status_path = _gradcam_status_log_path(conf, run_id, split_key)
    if shap_status_path is None and run_id:
        shap_status_path = _shap_status_log_path(conf, run_id, split_key)

    grad_status_payload = _load_status_payload(grad_status_path)
    shap_status_payload = _load_status_payload(shap_status_path)

    status_df = pd.DataFrame(
        [
            {
                "method": "gradcam",
                "status": grad_status_payload.get("status", "unknown"),
                "reason_or_error": grad_status_payload.get("reason", ""),
                "supported_layers": ", ".join(grad_status_payload.get("supported_layers", []) or []),
                "requested_layers": ", ".join(grad_status_payload.get("requested_layers", []) or []),
                "target_rows": grad_status_payload.get("target_rows", ""),
            },
            {
                "method": "shap",
                "status": shap_status_payload.get("status", "unknown"),
                "reason_or_error": shap_status_payload.get("error", ""),
                "attempted_device": shap_status_payload.get("attempted_device", ""),
                "final_device": shap_status_payload.get("final_device", ""),
                "fallback_used": shap_status_payload.get("fallback_used", ""),
                "target_rows": shap_status_payload.get("target_rows", ""),
            },
        ]
    )

    return {
        "cfg": conf,
        "seed": int(seed),
        "split": split_key,
        "rq1_path": str(rq1_path),
        "rq2_path": str(rq2_path),
        "pair_path": str(pair_path),
        "pass_by_class_path": str(pass_by_class_path),
        "pass_by_correctness_path": str(pass_by_correctness_path),
        "rq1_df": rq1_df,
        "rq2_df": rq2_df,
        "pair_df": pair_df,
        "classwise_df": classwise_df,
        "correctness_df": correctness_df,
        "correctness_view": correctness_view,
        "classwise_view": classwise_view,
        "discord_df": discord_df,
        "status_df": status_df,
        "gradcam_status_path": str(grad_status_path) if grad_status_path is not None else "",
        "shap_status_path": str(shap_status_path) if shap_status_path is not None else "",
    }


def notebook_run_visual_review(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    split: str = "test",
    wanted_classes: tuple[int, ...] = (0, 2, 3, 4),
    safe_mode: bool = False,
) -> dict[str, Any]:
    conf = copy.deepcopy(_cfg(cfg_or_path))
    split_key = str(split).strip().lower() or "test"
    split_key_to_manifest_key = {
        "train": "train_split",
        "val": "val_split",
        "test": "test_full",
    }
    if split_key not in split_key_to_manifest_key:
        raise ValueError(f"split must be one of train/val/test, got: {split}")

    xai_cfg = conf.setdefault("xai", {})
    if safe_mode:
        xai_cfg["device"] = "cpu"
        xai_cfg["shap_background_size"] = min(int(xai_cfg.get("shap_background_size", 64)), 8)
        max_shap_images = 2
    else:
        xai_cfg.setdefault("device", conf.get("training", {}).get("device", "mps"))
        max_shap_images = 4

    xai_shap_bg = int(xai_cfg.get("shap_background_size", 64))
    xai_dpi = int(xai_cfg.get("figure_dpi", 180))
    export_dpi = min(600, max(240, xai_dpi))
    shap_panel_size = (3.0, 3.4)
    gradcam_pair_size = (6.0, 3.4)

    manifests = prepare_data_manifests(conf, seed=int(seed))
    manifest_path = Path(manifests[split_key_to_manifest_key[split_key]])
    manifest_df = pd.read_csv(manifest_path)
    if len(manifest_df) == 0:
        raise RuntimeError(f"Empty {split_key} manifest: {manifest_path}")

    class_targets = [int(c) for c in wanted_classes]
    samples: list[tuple[str, int]] = []
    for cls in class_targets:
        sub = manifest_df[manifest_df["class_id"].astype(int) == int(cls)]
        if len(sub) == 0:
            continue
        row = sub.sample(n=1, random_state=int(seed)).iloc[0]
        samples.append((str(row["image_path"]), int(row["class_id"])))

    if len(samples) == 0:
        row = manifest_df.sample(n=1, random_state=int(seed)).iloc[0]
        samples = [(str(row["image_path"]), int(row["class_id"]))]

    gradcam_save_path = Path(conf["paths"]["figures_dir"]) / "gradcam_demo_grid.png"
    shap_save_path = Path(conf["paths"]["figures_dir"]) / "shap_demo_grid.png"

    gradcam_fig: plt.Figure | None = None
    shap_fig: plt.Figure | None = None
    warnings_list: list[str] = []
    try:
        gradcam_fig = plot_gradcam_grid(
            cfg_path=conf,
            seed=int(seed),
            samples=samples,
            ncols=2,
            gradcam_layer=str(xai_cfg.get("gradcam_layer", "layer4")),
            alpha=0.45,
            figsize_per_pair=gradcam_pair_size,
            save_path=gradcam_save_path,
            dpi=export_dpi,
        )
    except Exception as exc:
        warnings_list.append(f"Grad-CAM grid unavailable: {exc}")

    shap_subset = samples[:max_shap_images]
    shap_paths = [p for p, _ in shap_subset]
    shap_true_classes = [int(c) for _, c in shap_subset]
    try:
        shap_fig = plot_shap_grid(
            cfg_path=conf,
            seed=int(seed),
            image_paths=shap_paths,
            background_size=xai_shap_bg,
            vmax_percentile=99.5,
            save_path=shap_save_path,
            dpi=export_dpi,
            true_classes=shap_true_classes,
            show_correctness_border=True,
            panel_size=shap_panel_size,
        )
    except Exception as exc:
        warnings_list.append(f"SHAP grid unavailable: {exc}")

    return {
        "cfg_xai": conf,
        "seed": int(seed),
        "split": split_key,
        "safe_mode": bool(safe_mode),
        "manifest_path": str(manifest_path),
        "manifest_df": manifest_df,
        "samples": samples,
        "gradcam_fig": gradcam_fig,
        "shap_fig": shap_fig,
        "gradcam_path": str(gradcam_save_path),
        "shap_path": str(shap_save_path),
        "settings": {
            "device": str(xai_cfg.get("device")),
            "shap_background_size": int(xai_shap_bg),
            "shap_images": int(max_shap_images),
            "export_dpi": int(export_dpi),
            "shap_panel_size": shap_panel_size,
            "gradcam_pair_size": gradcam_pair_size,
        },
        "warnings": warnings_list,
    }


def notebook_run_single_case_report(
    cfg_or_path: str | Path | dict[str, Any],
    seed: int = 1988,
    image_path: str = "",
    split: str = "test",
    safe_mode: bool = False,
) -> dict[str, Any]:
    conf = copy.deepcopy(_cfg(cfg_or_path))
    xai_cfg = conf.setdefault("xai", {})

    if safe_mode:
        xai_cfg["device"] = "cpu"
        xai_cfg["shap_background_size"] = min(int(xai_cfg.get("shap_background_size", 64)), 8)
    else:
        xai_cfg.setdefault("device", conf.get("training", {}).get("device", "mps"))

    default_gradcam_layer = str(xai_cfg.get("gradcam_layer", "layer4"))
    requested_gradcam_layers = _parse_gradcam_layers(
        raw_layers=xai_cfg.get("gradcam_layers_eval", ["layer2", "layer3", default_gradcam_layer]),
        default_layer=default_gradcam_layer,
    )

    demo_image_path = str(image_path).strip()
    single_demo = run_single_case_demo(
        cfg_path=conf,
        seed=int(seed),
        image_path=demo_image_path,
        split=str(split).strip().lower() or "test",
        gradcam_layers=requested_gradcam_layers,
        shap_background_size=int(xai_cfg.get("shap_background_size", 64)),
        shap_panel_size=(3.2, 3.6),
        shap_dpi=min(600, max(240, int(xai_cfg.get("figure_dpi", 180)))),
    )

    prob_table = single_demo["prob_table"].copy()
    top1 = prob_table.iloc[0] if len(prob_table) else None
    top2 = prob_table.iloc[1] if len(prob_table) > 1 else None

    pred_name = str(top1["class_name"]) if top1 is not None else "N/A"
    pred_prob = float(top1["probability"]) if top1 is not None else float("nan")
    alt_txt = "N/A" if top2 is None else f"{top2['class_name']} ({float(top2['probability']):.3f})"

    triage_map = {
        0: "No DR pattern dominant. Continue routine follow-up if clinical exam is consistent.",
        1: "Mild DR pattern dominant. Consider short-interval follow-up and risk-factor optimization.",
        2: "Moderate DR pattern dominant. Recommend retina referral and closer follow-up planning.",
        3: "Severe DR pattern dominant. Escalate referral urgency to retina specialist.",
        4: "Proliferative DR pattern dominant. Treat as high-urgency retinal review candidate.",
    }
    pred_class = int(single_demo["pred_class"])
    triage_line = triage_map.get(pred_class, "Prediction outside expected range; use specialist review.")

    if np.isfinite(pred_prob) and pred_prob >= 0.80:
        conf_line = "Model confidence is high for this case."
    elif np.isfinite(pred_prob) and pred_prob >= 0.60:
        conf_line = "Model confidence is moderate; confirm with full clinical context."
    else:
        conf_line = "Model confidence is low; treat as uncertain and prioritize manual review."

    summary_markdown = (
        f"**Manifest used:** `{single_demo['manifest_path']}`\n\n"
        f"**Image:** `{single_demo['image_path']}`\n\n"
        f"**Prediction:** {single_demo['pred_label']} ({single_demo['pred_class']}) | "
        f"**Confidence:** {single_demo['confidence']:.4f}\n\n"
        f"**True label:** {single_demo.get('true_label', 'N/A')} ({single_demo.get('true_class', 'N/A')})\n\n"
        f"**Run ID:** `{single_demo['run_id']}` | **Device:** `{single_demo['device']}`"
    )
    story_markdown = "\n".join(
        [
            "### Clinical Decision Support Narrative",
            f"- Predicted DR grade: **{pred_name} ({pred_class})** with probability **{pred_prob:.3f}**.",
            f"- Next most likely alternative: **{alt_txt}**.",
            f"- Suggested triage framing: {triage_line}",
            f"- Confidence note: {conf_line}",
            "- How to use maps: highlighted regions are decision-support cues, not standalone diagnostic proof.",
            "- Clinical safeguard: final diagnosis and treatment decisions remain clinician-led.",
        ]
    )

    figure_sections = [
        {
            "title": "### Grad-CAM (Input + Requested Layers)",
            "path": str(single_demo["gradcam_panel_path"]),
        },
        {
            "title": "### SHAP (Per-Class Grid)",
            "path": str(single_demo["shap_grid_path"]),
        },
    ]

    # Explanation quality profile: surface per-case continuous metrics so the
    # notebook can display the same 4 numbers the aggregate analysis uses.
    single_result = single_demo.get("single_result", {}) or {}
    gradcam_details_list = single_result.get("gradcam_details", []) or []
    shap_details = single_result.get("shap_details", {}) or {}
    # Pick the Grad-CAM layer the audit treats as primary (the config default).
    gcam_row = None
    for entry in gradcam_details_list:
        if str(entry.get("layer", "")) == default_gradcam_layer:
            gcam_row = entry
            break
    if gcam_row is None and gradcam_details_list:
        gcam_row = gradcam_details_list[0]
    gcam_row = gcam_row or {}

    def _fmt(v: Any) -> str:
        try:
            f = float(v)
            if not np.isfinite(f):
                return "N/A"
            return f"{f:.3f}"
        except (TypeError, ValueError):
            return "N/A"

    quality_profile_df = pd.DataFrame(
        [
            {
                "method": "Grad-CAM",
                "border_ratio": float(gcam_row.get("border_ratio", float("nan"))),
                "retina_ratio": float(gcam_row.get("retina_ratio", float("nan"))),
                "faith_delta_k20": float(gcam_row.get("faith_delta_k20", gcam_row.get("faithfulness_delta", float("nan")))),
                "aopc_delta": float(gcam_row.get("aopc_delta", float("nan"))),
            },
            {
                "method": "SHAP",
                "border_ratio": float(shap_details.get("border_ratio", float("nan"))),
                "retina_ratio": float(shap_details.get("retina_ratio", float("nan"))),
                "faith_delta_k20": float(shap_details.get("faith_delta_k20", shap_details.get("faithfulness_delta", float("nan")))),
                "aopc_delta": float(shap_details.get("aopc_delta", float("nan"))),
            },
        ]
    )

    quality_profile_markdown = "\n".join(
        [
            "### Explanation Quality Profile",
            "",
            "| Method | Border ratio (\u2193) | Retina ratio (\u2191) | \u0394$_{k20}$ (\u2191) | AOPC (\u2191) |",
            "|---|---|---|---|---|",
            f"| Grad-CAM | {_fmt(gcam_row.get('border_ratio'))} | {_fmt(gcam_row.get('retina_ratio'))} | "
            f"{_fmt(gcam_row.get('faith_delta_k20', gcam_row.get('faithfulness_delta')))} | "
            f"{_fmt(gcam_row.get('aopc_delta'))} |",
            f"| SHAP | {_fmt(shap_details.get('border_ratio'))} | {_fmt(shap_details.get('retina_ratio'))} | "
            f"{_fmt(shap_details.get('faith_delta_k20', shap_details.get('faithfulness_delta')))} | "
            f"{_fmt(shap_details.get('aopc_delta'))} |",
            "",
            "**Interpretation:**",
            "",
            "- **Border ratio** (\u2193 lower is better) \u2014 share of the heatmap falling on the dark corners outside the eye.",
            "- **Retina ratio** (\u2191 higher is better) \u2014 share landing inside the retinal disc. Low border plus high retina means attention stays on the eye.",
            "- **\u0394$_{k20}$** (\u2191 higher is better) \u2014 drop in the model's confidence when the top 20\\% most-important pixels are removed. A bigger drop means those pixels really mattered to the prediction.",
            "- **AOPC** (\u2191 higher is better) \u2014 the same idea as \u0394$_{k20}$ averaged across several removal sizes (a smoother version).",
            "",
            "These are per-case decision-support cues, not standalone diagnostic evidence.",
        ]
    )

    return {
        "cfg_xai": conf,
        "seed": int(seed),
        "split": str(split).strip().lower() or "test",
        "safe_mode": bool(safe_mode),
        "single_demo": single_demo,
        "prob_table": prob_table,
        "summary_markdown": summary_markdown,
        "story_markdown": story_markdown,
        "figure_sections": figure_sections,
        "quality_profile_df": quality_profile_df,
        "quality_profile_markdown": quality_profile_markdown,
        "xai_warnings": list(single_demo.get("xai_warnings", []) or []),
    }
