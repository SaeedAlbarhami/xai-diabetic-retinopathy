"""Thin wrappers the project demo notebook calls for each XAI section.

Each ``notebook_*`` function composes the audit, single-case, or visual
review pipelines and returns display-ready markdown, DataFrames, and
matplotlib figures that the notebook just hands to ``display(...)``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import (
    _cfg,
    _class_name_from_id,
    _load_json,
    _manifest_path,
    _table_path,
)
from src.train import _predictions_path_for_run_id
from src.xai_audit import (
    _format_continuous_bottom_line,
    _format_continuous_table_for_display,
    _xai_pass_rule_thresholds,
    run_xai_analysis,
)
from src.xai_gradcam import plot_gradcam_grid
from src.xai_shap import plot_shap_grid
from src.xai_single import run_single_case_demo


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
            "title": f"### Grad-CAM (Per-Class Grid, layer={single_demo.get('gradcam_layer', 'layer4')})",
            "path": str(single_demo["gradcam_grid_path"]),
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

