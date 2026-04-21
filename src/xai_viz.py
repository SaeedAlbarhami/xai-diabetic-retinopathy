"""Pure rendering helpers for the XAI audit.

Extracted from ``src/xai.py`` as part of the architecture-via-relocation
refactor. Every function here is pure image math / matplotlib rendering:
no model calls, no config reads, no SHAP/Grad-CAM compute. Used by the
method-compute and orchestration modules in ``src/xai.py`` (and later
``src/xai_gradcam.py``, ``src/xai_shap.py``).

This module is a leaf: it must not import from any ``src.xai_*`` sibling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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


def _plot_attribution_grid(
    rows_data: list[dict[str, Any]],
    num_classes: int,
    label_order: list[str],
    cmap: Any,
    norm: Any,
    colorbar_label: str,
    panel_size: tuple[float, float],
    bg_cmap: str = "gray",
    bg_alpha: float = 0.35,
    fg_alpha: float = 0.85,
    show_correctness_border: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 180,
) -> plt.Figure:
    """Render a shared (nrows x (1+num_classes)) attribution grid.

    Each row in ``rows_data`` must carry: ``image_pil``, ``pred_class``,
    ``confidence``, ``class_maps`` (length ``num_classes``), and optionally
    ``true_class``. Used by both :func:`plot_shap_grid` and
    :func:`plot_gradcam_class_grid` so the two methods render identically.
    """
    nrows = len(rows_data)
    ncols = 1 + int(num_classes)
    panel_w = max(1.5, float(panel_size[0]))
    panel_h = max(2.0, float(panel_size[1]))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), facecolor="white")
    axes = np.array(axes).reshape(nrows, ncols)

    im = None
    row_border_specs: list[tuple[int, str]] = []
    for r, row in enumerate(rows_data):
        image_pil = row["image_pil"]
        pred_class = int(row["pred_class"])
        true_class_raw = row.get("true_class")
        true_class = int(true_class_raw) if true_class_raw is not None else None
        confidence = float(row.get("confidence", float("nan")))
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
            im = ax.imshow(smap, cmap=cmap, norm=norm, alpha=float(fg_alpha), interpolation="nearest")
            ax.axis("off")
            score = 1.0 if cls_idx == pred_class else 0.0
            ax.set_title(f"{score:.1f}", fontsize=8, color="#e74c3c" if cls_idx == pred_class else "#333")

    if im is not None:
        cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.02])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, orientation="horizontal")
        cb.set_label(str(colorbar_label), fontsize=8)
        cb.ax.tick_params(labelsize=7)

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
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    return fig
