"""Render the overall APTOS 2019 class distribution bar chart.

One-shot generator for ``src/report/assets/figure08_class_distribution_overall.png``.
Reads counts from the manifest summary JSON (the benchmark split is fully
deterministic at seed=1988, so the totals never change between runs) and the
class label order from ``configs/base.yaml``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifests" / "summary_aptos2019_85-15-v10_seed1988.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"
OUT_PATH = PROJECT_ROOT / "src" / "report" / "assets" / "figure08_class_distribution_overall.png"


def main() -> int:
    summary = json.loads(MANIFEST_PATH.read_text())
    labels = yaml.safe_load(CONFIG_PATH.read_text())["data"]["label_order"]

    counts = [
        summary["class_counts_train"][str(i)]
        + summary["class_counts_val"][str(i)]
        + summary["class_counts_test"][str(i)]
        for i in range(len(labels))
    ]
    total = sum(counts)
    pcts = [c / total * 100.0 for c in counts]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, pcts, color="#1f77b4", edgecolor="black", linewidth=0.5)

    ymax = max(pcts) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("class_name")
    ax.set_ylabel("Percentage (%)")

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            f"{pct:.1f}%\n({cnt})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {OUT_PATH}  (total={total}, counts={counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
