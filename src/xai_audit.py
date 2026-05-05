"""XAI audit pipeline: per-sample Grad-CAM and SHAP loops plus aggregate tables."""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import shap
except ImportError:
    shap = None

try:
    from captum.attr import LayerGradCam
except ImportError:
    LayerGradCam = None

from src.data import (
    _backbone_name,
    _cfg,
    _load_image_for_inference,
    _model_image_size,
    _save_json,
)
from src.train import (
    _gradcam_status_log_path,
    _load_model,
    _resolve_checkpoint_and_run_id,
    _save_latest_run_record,
    _shap_status_log_path,
    run_split_inference,
)
from src.xai_common import (
    _resolve_xai_device,
)
from src.xai_gradcam import (
    _generate_gradcam,
    _resolve_gradcam_target_layer,
)
from src.xai_metrics import (
    _attribution_mass_ratios,
    _attribution_retina_mask,
    _faithfulness_delta,
    _faithfulness_multi_k,
    _k_to_col_name,
    _parse_faithfulness_k_list,
)
from src.xai_shap import (
    _build_shap_explainer_with_known_warning_filter,
    _empty_mps_cache_if_available,
    _make_shap_compatible,
    _pick_shap_map,
    _shap_to_2d,
    _shap_values_with_known_warning_filter,
    _should_retry_shap_on_cpu,
)
from src.xai_stats import (
    _bootstrap_pass_rate_ci,
    _build_xai_continuous_stats,
    _build_xai_pairwise_stats,
    _xai_pass_flag,
)
from src.xai_viz import (
    _save_map_overlay,
)


_CONTINUOUS_METRIC_DISPLAY: dict[str, tuple[str, str]] = {
    "border_ratio":    ("Border ratio (lower = on-retina)",        "↓"),
    "retina_ratio":    ("Retina ratio (higher = on-retina)",       "↑"),
    "faith_delta_k10": ("Faithfulness Δ at k=10% (higher better)", "↑"),
    "faith_delta_k20": ("Faithfulness Δ at k=20% (higher better)", "↑"),
    "faith_delta_k30": ("Faithfulness Δ at k=30% (higher better)", "↑"),
    "aopc_delta":      ("AOPC (Samek 2017, higher better)",         "↑"),
}


def _xai_pass_rule_thresholds(conf: dict[str, Any]) -> tuple[float, float]:
    xai_cfg = conf.get("xai", {}) if isinstance(conf, dict) else {}
    border_ratio_max = float(xai_cfg.get("pass_border_ratio_max", 0.35))
    faith_delta_min = float(xai_cfg.get("pass_faith_delta_min", 0.0))
    return border_ratio_max, faith_delta_min


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
    """Pass rate broken down by class. Groups by the ``target_class`` column,
    which stores the value of whatever column ``target_balance_class_col`` in
    config points at — defaults to ``true_class`` in ``configs/base.yaml``.
    So the emitted ``class_id``/``class_name`` reflects the TRUE class, not
    the predicted class, unless the config is changed.
    """
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
        if np.isnan(w_p):
            p_text = "nan"
        elif w_p < 0.001:
            p_text = "< 0.001"
        else:
            p_text = f"{w_p:.3f}"
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

    def _fmt_p(val: float) -> str:
        if np.isnan(val):
            return "nan"
        return "< 0.001" if val < 0.001 else f"{val:.3f}"

    metric_lines = [
        ("border_ratio", "Localization (border_ratio)"),
        ("faith_delta_k20", "Faithfulness (Δ_k20)"),
        ("aopc_delta", "AOPC (Samek 2017)"),
    ]
    parts: list[str] = [
        "> **Continuous analysis (research-standard, no arbitrary thresholds):** "
        "paired Wilcoxon signed-rank tests on raw scores.",
    ]
    for metric_key, label in metric_lines:
        rec = continuous_df[continuous_df["metric"] == metric_key]
        if len(rec):
            r = rec.iloc[0]
            parts.append(
                f"> **{label}:** Grad-CAM={float(r['gradcam_mean']):.3f}, "
                f"SHAP={float(r['shap_mean']):.3f}, winner=**{str(r['winner']).upper()}**, "
                f"Wilcoxon p={_fmt_p(float(r['wilcoxon_pvalue']))}, Cohen's dz={float(r['cohen_dz']):+.2f}."
            )
    parts.append(
        "> **Interpretation:** the thresholded pass-rate is a descriptive summary "
        "under a project-specific rule; this continuous analysis is the primary finding."
    )
    return "\n".join(parts)


def _choose_xai_targets(
    df: pd.DataFrame,
    high_conf_threshold: float,
    max_targets: int | None = None,
    num_classes: int = 5,
    class_col: str = "true_class",
    balance_by_class: bool = True,
    fill_missing_from_all: bool = True,
) -> pd.DataFrame:
    """Pick the XAI audit subset, aiming for ``max_targets / num_classes`` per
    ``class_col``. Perfect balance is a target, not a guarantee: if a rare
    class has fewer high-confidence samples than the per-class quota, the
    filler pass tops up from remaining classes to reach ``max_targets`` total.
    Actual per-class Ns will therefore drift from the nominal quota on
    imbalanced datasets (e.g. APTOS 2019 Severe/Proliferative).
    """
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
    if LayerGradCam is None:
        raise RuntimeError("captum is required for Grad-CAM audit. Install with `pip install captum==0.7.0`.")
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

    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=checkpoint, require_existing=True)
    _save_latest_run_record(conf, seed=seed, run_id=run_id, checkpoint_path=ckpt_path)

    pred_path = Path(predictions_csv) if predictions_csv else Path(run_split_inference(conf, seed=seed, split=split))

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

    tables_dir = Path(conf["paths"]["tables_dir"])
    def _xai_csv(stem: str) -> Path:
        return tables_dir / f"{stem}_seed{seed}_{split}.csv"

    xai_targets_path = _xai_csv("xai_targets")
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

        layer_summary_path = _xai_csv("gradcam_layer_selection")
        pd.DataFrame(
            layer_scores,
            columns=["gradcam_layer", "composite_score", "mean_aopc_delta", "mean_border_ratio", "n_rows"],
        ).to_csv(layer_summary_path, index=False)

        gradcam_status_payload = {
            "seed": int(seed),
            "run_id": run_id,
            "split": split,
            "backbone": backbone,
            "requested_layers": gradcam_layers_eval,
            "supported_layers": supported_layers,
            "unsupported_layers": unsupported_layers,
            "errors": layer_errors,
            "selected_layer": selected_layer,
            "pass_rule_border_ratio_max": float(pass_border_ratio_max),
            "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
            "evaluation_scope": evaluation_scope,
            "shared_targets_enforced": bool(enforce_consistent_targets),
            "target_rows": int(len(xai_target_df)),
        }
        if len(grad_rows) == 0:
            reason = "No Grad-CAM rows were produced from supported layers."
            if layer_errors:
                reason = f"{reason} Errors: {layer_errors}"
            gradcam_status_payload.update({"status": "failed", "reason": reason})
            _save_json(gradcam_status_path, gradcam_status_payload)
            gradcam_status_note = f"failed: {reason}"
        else:
            status = "success" if len(layer_errors) == 0 and len(unsupported_layers) == 0 else "partial_success"
            gradcam_status_payload.update({"status": status, "rows": int(len(grad_rows))})
            _save_json(gradcam_status_path, gradcam_status_payload)
            gradcam_status_note = status

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
    rq1_path = _xai_csv("rq1_gradcam")
    rq1_df.to_csv(rq1_path, index=False)

    shap_rows: list[dict[str, Any]] = []
    shap_status_path = _shap_status_log_path(conf, run_id, split)
    shap_status_note = ""
    shap_attempted_device = str(device)
    shap_final_device = str(device)
    shap_fallback_used = False
    shap_error_primary = ""
    shap_error = ""

    try:
        if shap is None:
            raise RuntimeError("shap is required for SHAP audit. Install with `pip install shap==0.47.2`.")

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

    shap_status_payload = {
        "seed": int(seed),
        "run_id": run_id,
        "split": split,
        "mode": str(shap_mode) if shap_error else use_mode,
        "attempted_device": shap_attempted_device,
        "final_device": shap_final_device,
        "fallback_used": bool(shap_fallback_used),
        "error_primary": str(shap_error_primary),
        "pass_rule_border_ratio_max": float(pass_border_ratio_max),
        "pass_rule_faith_delta_k20_min": float(pass_faith_delta_min),
        "evaluation_scope": evaluation_scope,
        "shared_targets_enforced": bool(enforce_consistent_targets),
        "target_rows": int(len(xai_target_df)),
    }
    if shap_error:
        shap_status_payload["status"] = "failed"
        shap_status_payload["error"] = str(shap_error)
        _save_json(shap_status_path, shap_status_payload)
        shap_status_note = f"failed: {shap_error}"
    else:
        shap_status_payload["status"] = "success"
        shap_status_payload["samples"] = int(len(shap_rows))
        _save_json(shap_status_path, shap_status_payload)
        shap_status_note = "success_cpu_fallback" if shap_fallback_used else "success"

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
    rq2_path = _xai_csv("rq2_shap")
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
    coverage_path = _xai_csv("xai_target_coverage")
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
    method_stats_path = _xai_csv("rq_xai_method_stats")
    method_stats_df.to_csv(method_stats_path, index=False)

    class_names = [str(x) for x in conf.get("data", {}).get("label_order", [])]
    pass_by_correctness_df = pd.concat(
        [
            _build_xai_pass_by_correctness_table("gradcam", rq1_df, "gradcam_pass"),
            _build_xai_pass_by_correctness_table("shap", rq2_df, "shap_pass"),
        ],
        ignore_index=True,
    )
    pass_by_correctness_path = _xai_csv("rq_xai_pass_by_correctness")
    pass_by_correctness_df.to_csv(pass_by_correctness_path, index=False)

    pass_by_class_df = pd.concat(
        [
            _build_xai_pass_by_class_table("gradcam", rq1_df, "gradcam_pass", class_names=class_names),
            _build_xai_pass_by_class_table("shap", rq2_df, "shap_pass", class_names=class_names),
        ],
        ignore_index=True,
    )
    pass_by_class_path = _xai_csv("rq_xai_pass_by_class")
    pass_by_class_df.to_csv(pass_by_class_path, index=False)

    pairwise_df = _build_xai_pairwise_stats(rq1_df=rq1_df, rq2_df=rq2_df)
    pairwise_path = _xai_csv("rq_xai_pairwise")
    pairwise_df.to_csv(pairwise_path, index=False)

    continuous_df = _build_xai_continuous_stats(rq1_df=rq1_df, rq2_df=rq2_df)
    continuous_path = _xai_csv("rq_xai_continuous")
    continuous_df.to_csv(continuous_path, index=False)

    proto_stub_path = _xai_csv("protopnet_stub")
    pd.DataFrame([{"method": "ProtoPNetLite", "status": "planned_future_phase", "reason": "Planned future extension"}]).to_csv(proto_stub_path, index=False)

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

