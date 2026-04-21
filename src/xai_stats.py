"""Pure statistics helpers for the XAI audit.

Extracted from ``src/xai.py`` as part of the architecture-via-relocation
refactor. Every function here is a pure math / pandas transformation: no
filesystem I/O, no model calls, no config reads. Used by the audit
aggregation code in ``src/xai.py`` (and later ``src/xai_audit.py``).

This module is a leaf: it must not import from any ``src.xai_*`` sibling.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


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
