"""Build the masked-vs-unmasked ablation table from the dual-metric audit CSVs.

Produces two deliverables that the LaTeX report will cite:

1. ``rq_xai_mask_ablation_seed{seed}_{split}.csv`` — long-format per-metric
   paired comparison between Grad-CAM and SHAP, reported BOTH on the raw
   (pre-mask) attribution and on the masked attribution. Six metrics:
   border_ratio, retina_ratio, faith_delta_k{10,20,30}, aopc_delta.
   Columns: metric, masked (bool), n_paired, gradcam_mean, shap_mean,
   paired_mean_diff, wilcoxon_p, cohen_dz, winner.

2. ``rq_xai_per_class_seed{seed}_{split}.csv`` — per-class pass rate for both
   methods, on the masked metrics only (since that is the primary audit).

The existing ``rq_xai_method_stats_*`` and ``rq_xai_continuous_*`` files stay
authoritative for aggregate numbers; this script is additive, not a
replacement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TABLES_DIR = PROJECT_ROOT / "artifacts" / "reports" / "tables"


def _paired_stats(
    g_vals: np.ndarray,
    s_vals: np.ndarray,
    *,
    direction: str,
) -> dict[str, float]:
    """Compute paired mean diff, Wilcoxon p, Cohen's dz, and winner label."""
    diff = g_vals - s_vals
    paired_mean = float(np.mean(diff))
    paired_std = float(np.std(diff, ddof=1))
    dz = paired_mean / paired_std if paired_std > 1e-12 else 0.0

    try:
        stat, p = wilcoxon(g_vals, s_vals, alternative="two-sided")
        p = float(p)
    except ValueError:
        p = float("nan")

    if direction == "lower_is_better":
        winner = "gradcam" if paired_mean < 0 else "shap"
    else:
        winner = "gradcam" if paired_mean > 0 else "shap"
    return {
        "paired_mean_diff": paired_mean,
        "wilcoxon_p": p,
        "cohen_dz": float(dz),
        "winner": winner,
    }


def main(seed: int = 1988, split: str = "test") -> int:
    rq1 = pd.read_csv(TABLES_DIR / f"rq1_gradcam_seed{seed}_{split}.csv")
    rq2 = pd.read_csv(TABLES_DIR / f"rq2_shap_seed{seed}_{split}.csv")

    required = {"border_ratio_raw", "retina_ratio_raw"}
    if not required.issubset(rq1.columns) or not required.issubset(rq2.columns):
        print(
            "error: missing *_raw columns — rerun the dual-metric pipeline first",
            file=sys.stderr,
        )
        return 2

    merged = rq1.merge(rq2, on="sample_id", suffixes=("_g", "_s"))
    n = len(merged)

    # --- Ablation table (masked vs raw)
    metric_specs = [
        ("border_ratio", "lower_is_better"),
        ("retina_ratio", "higher_is_better"),
        ("faith_delta_k10", "higher_is_better"),
        ("faith_delta_k20", "higher_is_better"),
        ("faith_delta_k30", "higher_is_better"),
        ("aopc_delta", "higher_is_better"),
    ]
    rows = []
    for base, direction in metric_specs:
        # Masked (the current primary metric)
        g = merged[f"{base}_g"].to_numpy()
        s = merged[f"{base}_s"].to_numpy()
        stats = _paired_stats(g, s, direction=direction)
        rows.append({
            "metric": base,
            "masked": True,
            "n_paired": n,
            "gradcam_mean": float(np.mean(g)),
            "gradcam_median": float(np.median(g)),
            "shap_mean": float(np.mean(s)),
            "shap_median": float(np.median(s)),
            **stats,
            "direction": direction,
        })
        # Raw (only available for border/retina — faith/aopc are inherently
        # computed on the masked heatmap; see report Methodology).
        raw_g = f"{base}_raw_g"
        raw_s = f"{base}_raw_s"
        if raw_g in merged.columns and raw_s in merged.columns:
            g_raw = merged[raw_g].to_numpy()
            s_raw = merged[raw_s].to_numpy()
            stats = _paired_stats(g_raw, s_raw, direction=direction)
            rows.append({
                "metric": base,
                "masked": False,
                "n_paired": n,
                "gradcam_mean": float(np.mean(g_raw)),
                "gradcam_median": float(np.median(g_raw)),
                "shap_mean": float(np.mean(s_raw)),
                "shap_median": float(np.median(s_raw)),
                **stats,
                "direction": direction,
            })

    ablation_df = pd.DataFrame(rows)
    ablation_path = TABLES_DIR / f"rq_xai_mask_ablation_seed{seed}_{split}.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"wrote {ablation_path}")
    print(ablation_df.to_string(index=False))

    # --- Per-class pass rate (masked, primary audit)
    rq1c = rq1.copy()
    rq2c = rq2.copy()
    rq1c["pass"] = (rq1c["border_ratio"] <= 0.25) & (rq1c["faith_delta_k20"] > 0.10)
    rq2c["pass"] = (rq2c["border_ratio"] <= 0.25) & (rq2c["faith_delta_k20"] > 0.10)

    class_names = {
        0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferate_DR",
    }
    per_class_rows = []
    for cls in sorted(set(rq1c["pred_class"]) | set(rq2c["pred_class"])):
        g_sub = rq1c[rq1c["pred_class"] == cls]
        s_sub = rq2c[rq2c["pred_class"] == cls]
        per_class_rows.append({
            "pred_class": cls,
            "class_name": class_names.get(cls, str(cls)),
            "n": len(g_sub),
            "gradcam_pass_rate": float(g_sub["pass"].mean()) if len(g_sub) else float("nan"),
            "shap_pass_rate": float(s_sub["pass"].mean()) if len(s_sub) else float("nan"),
            "gradcam_mean_border": float(g_sub["border_ratio"].mean()) if len(g_sub) else float("nan"),
            "shap_mean_border": float(s_sub["border_ratio"].mean()) if len(s_sub) else float("nan"),
            "gradcam_mean_faith_k20": float(g_sub["faith_delta_k20"].mean()) if len(g_sub) else float("nan"),
            "shap_mean_faith_k20": float(s_sub["faith_delta_k20"].mean()) if len(s_sub) else float("nan"),
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = TABLES_DIR / f"rq_xai_per_class_seed{seed}_{split}.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print()
    print(f"wrote {per_class_path}")
    print(per_class_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
