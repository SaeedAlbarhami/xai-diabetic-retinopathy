#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    load_project_config,
    notebook_prepare_data_overview,
)
from src.train import (  # noqa: E402
    notebook_run_core_evaluation,
    notebook_run_training,
)
from src.xai import (  # noqa: E402
    notebook_load_xai_committee_summary,
)


@dataclass(frozen=True)
class FigureRecord:
    number: int
    title: str
    path: Path
    source: str


def _save_figure(fig: plt.Figure | None, out_path: Path, dpi: int) -> None:
    if fig is None:
        raise RuntimeError(f"No figure available for {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _copy_image(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _extract_notebook_png(
    notebook_path: Path,
    cell_index: int,
    image_index_within_cell: int,
    out_path: Path,
) -> None:
    nb = json.loads(notebook_path.read_text())
    images: list[str] = []
    for output in nb["cells"][cell_index].get("outputs", []):
        data = output.get("data", {})
        png = data.get("image/png")
        if png is None:
            continue
        if isinstance(png, list):
            png = "".join(png)
        images.append(png)

    if image_index_within_cell >= len(images):
        raise IndexError(
            f"Notebook cell {cell_index} has {len(images)} image outputs; "
            f"cannot extract image index {image_index_within_cell}."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(images[image_index_within_cell])
    Image.open(BytesIO(raw)).save(out_path)


def export_assets(
    config_path: Path,
    notebook_path: Path,
    output_dir: Path,
    seed: int,
    split: str,
    dpi: int,
    zip_path: Path | None = None,
) -> list[FigureRecord]:
    cfg = load_project_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[FigureRecord] = []

    # Section 4: intended figures are the two distribution charts.
    data_overview = notebook_prepare_data_overview(cfg, seed=int(seed), samples_per_split=3)
    try:
        fig1 = output_dir / "figure1.png"
        _save_figure(data_overview["class_count_fig"], fig1, dpi=dpi)
        records.append(FigureRecord(1, "Class Distribution by Split (Counts)", fig1, "regenerated"))

        fig2 = output_dir / "figure2.png"
        _save_figure(data_overview["class_proportion_fig"], fig2, dpi=dpi)
        records.append(FigureRecord(2, "Class Distribution by Split (Within-Split %)", fig2, "regenerated"))
    finally:
        if data_overview.get("sample_images_fig") is not None:
            plt.close(data_overview["sample_images_fig"])

    # Section 5: training history plot.
    training_out = notebook_run_training(cfg, seed=int(seed), manifests=data_overview["manifest_paths"], force_retrain=False)
    fig3 = output_dir / "figure3.png"
    history_fig = training_out.get("history_fig")
    if history_fig is not None:
        _save_figure(history_fig, fig3, dpi=dpi)
        records.append(FigureRecord(3, "Training History", fig3, "regenerated"))
    else:
        _extract_notebook_png(notebook_path, cell_index=11, image_index_within_cell=0, out_path=fig3)
        records.append(FigureRecord(3, "Training History", fig3, "notebook fallback"))

    # Section 6: evaluation charts, reusing existing predictions file when available.
    predictions_dir = PROJECT_ROOT / "artifacts" / "predictions"
    latest_prediction = _latest_matching_file(predictions_dir, f"*_{split}_predictions.csv")
    predictions_path = latest_prediction if latest_prediction is not None else None
    core_eval = notebook_run_core_evaluation(cfg, seed=int(seed), split=split, predictions_csv=predictions_path)

    fig4 = output_dir / "figure4.png"
    headline_fig = core_eval.get("headline_fig")
    if headline_fig is not None:
        _save_figure(headline_fig, fig4, dpi=dpi)
        records.append(FigureRecord(4, "Headline Metrics", fig4, "regenerated"))
    else:
        _extract_notebook_png(notebook_path, cell_index=13, image_index_within_cell=0, out_path=fig4)
        records.append(FigureRecord(4, "Headline Metrics", fig4, "notebook fallback"))

    fig5 = output_dir / "figure5.png"
    confusion_fig = core_eval.get("confusion_fig")
    if confusion_fig is not None:
        _save_figure(confusion_fig, fig5, dpi=dpi)
        records.append(FigureRecord(5, "Confusion Matrix", fig5, "regenerated"))
    else:
        _extract_notebook_png(notebook_path, cell_index=13, image_index_within_cell=1, out_path=fig5)
        records.append(FigureRecord(5, "Confusion Matrix", fig5, "notebook fallback"))

    # Section 7: XAI summary pass-rate chart.
    xai_summary = notebook_load_xai_committee_summary(cfg, seed=int(seed), split=split)
    fig6 = output_dir / "figure6.png"
    pass_rate_fig = xai_summary.get("pass_rate_fig")
    if pass_rate_fig is not None:
        _save_figure(pass_rate_fig, fig6, dpi=dpi)
        records.append(FigureRecord(6, "Operational Threshold Summary (Descriptive)", fig6, "regenerated"))
    else:
        _extract_notebook_png(notebook_path, cell_index=16, image_index_within_cell=0, out_path=fig6)
        records.append(FigureRecord(6, "Operational Threshold Summary (Descriptive)", fig6, "notebook fallback"))

    # Sections 8-9: prefer the already exported high-resolution XAI PNGs.
    figures_dir = PROJECT_ROOT / "artifacts" / "reports" / "figures"
    single_dir = figures_dir / "single"

    fig7 = output_dir / "figure7.png"
    gradcam_demo = figures_dir / "gradcam_demo_grid.png"
    if gradcam_demo.exists():
        _copy_image(gradcam_demo, fig7)
        records.append(FigureRecord(7, "Grad-CAM Demo Grid", fig7, "copied"))
    else:
        _extract_notebook_png(notebook_path, cell_index=19, image_index_within_cell=0, out_path=fig7)
        records.append(FigureRecord(7, "Grad-CAM Demo Grid", fig7, "notebook fallback"))

    fig8 = output_dir / "figure8.png"
    shap_demo = figures_dir / "shap_demo_grid.png"
    if shap_demo.exists():
        _copy_image(shap_demo, fig8)
        records.append(FigureRecord(8, "SHAP Demo Grid", fig8, "copied"))
    else:
        _extract_notebook_png(notebook_path, cell_index=19, image_index_within_cell=1, out_path=fig8)
        records.append(FigureRecord(8, "SHAP Demo Grid", fig8, "notebook fallback"))

    fig9 = output_dir / "figure9.png"
    single_gradcam = _latest_matching_file(single_dir, "*_gradcam_grid.png")
    if single_gradcam is not None:
        _copy_image(single_gradcam, fig9)
        records.append(FigureRecord(9, "Single-Case Grad-CAM Grid", fig9, "copied"))
    else:
        _extract_notebook_png(notebook_path, cell_index=21, image_index_within_cell=0, out_path=fig9)
        records.append(FigureRecord(9, "Single-Case Grad-CAM Grid", fig9, "notebook fallback"))

    fig10 = output_dir / "figure10.png"
    single_shap = _latest_matching_file(single_dir, "*_shap_grid.png")
    if single_shap is not None:
        _copy_image(single_shap, fig10)
        records.append(FigureRecord(10, "Single-Case SHAP Grid", fig10, "copied"))
    else:
        _extract_notebook_png(notebook_path, cell_index=21, image_index_within_cell=1, out_path=fig10)
        records.append(FigureRecord(10, "Single-Case SHAP Grid", fig10, "notebook fallback"))

    manifest_path = output_dir / "figure_manifest.csv"
    pd.DataFrame(
        [
            {
                "figure_number": item.number,
                "filename": item.path.name,
                "title": item.title,
                "source": item.source,
            }
            for item in records
        ]
    ).to_csv(manifest_path, index=False)

    if zip_path is not None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        archive_base = zip_path.with_suffix("")
        created = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
        if Path(created) != zip_path:
            shutil.move(created, zip_path)

    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export numbered, high-resolution figures from notebooks/project_demo.ipynb into an assets folder."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="Project config path.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "project_demo.ipynb",
        help="Notebook path for fallback image extraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "assets",
        help="Directory where figure1.png ... figure10.png will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1988,
        help="Seed used by the project notebook.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Evaluation split used by the notebook.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for figures regenerated from matplotlib objects.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Optional zip archive path, for example assets.zip.",
    )
    args = parser.parse_args()

    records = export_assets(
        config_path=args.config.resolve(),
        notebook_path=args.notebook.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=int(args.seed),
        split=str(args.split).strip().lower() or "test",
        dpi=int(args.dpi),
        zip_path=args.zip_path.resolve() if args.zip_path is not None else None,
    )

    for item in records:
        print(f"figure{item.number}: {item.path} [{item.source}]")
    print(f"manifest: {args.output_dir.resolve() / 'figure_manifest.csv'}")
    if args.zip_path is not None:
        print(f"zip: {args.zip_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
