"""Grad-CAM computation and the report figure grids."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

try:
    from captum.attr import LayerAttribution, LayerGradCam
except ImportError:
    LayerAttribution = None
    LayerGradCam = None

from src.data import (
    _cfg,
    _load_image_for_inference,
    _model_image_size,
)
from src.train import (
    DRClassifier,
    _load_model,
    _resolve_checkpoint_and_run_id,
)
from src.xai_common import (
    _predict_one_with_temperature,
    _resolve_xai_device,
    _temperature_for_run,
)
from src.xai_metrics import _attribution_retina_mask
from src.xai_viz import (
    _normalize_map,
    _overlay,
    _plot_attribution_grid,
    _save_overlay_image,
)


def _resolve_gradcam_target_layer(
    model: DRClassifier,
    layer_name: str,
) -> tuple[nn.Module | None, str, str]:
    layer_key = str(layer_name).strip().lower()
    if layer_key not in {"layer2", "layer3", "layer4"}:
        return None, "", f"Unsupported gradcam layer '{layer_name}'. Supported layers: layer2, layer3, layer4"

    features = model.net.features
    stage_map = {"layer2": 4, "layer3": 6, "layer4": 8}
    req_idx = int(stage_map[layer_key])
    idx = req_idx if req_idx < len(features) else int(round((req_idx / 8.0) * (len(features) - 1)))
    idx = int(min(max(idx, 0), len(features) - 1))
    return features[idx], f"net.features.{idx}", ""


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
) -> tuple[str, np.ndarray, np.ndarray]:
    if LayerGradCam is None:
        raise RuntimeError("captum is required for Grad-CAM. Install with `pip install captum==0.7.0`.")

    model.eval()
    target_layer, resolved_layer_name, reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=layer_name,
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


def _gradcam_heatmap_for_display(
    model: DRClassifier,
    input_tensor: torch.Tensor,
    class_id: int,
    layer_name: str,
    output_size: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    if LayerGradCam is None:
        raise RuntimeError("captum is required for Grad-CAM. Install with `pip install captum==0.7.0`.")

    target_layer, _, reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=layer_name,
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
    if len(samples) == 0:
        raise ValueError("samples must not be empty")

    conf = _cfg(cfg_path)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    label_order = list(conf["data"]["label_order"])
    out_dpi = int(dpi if dpi is not None else conf.get("xai", {}).get("figure_dpi", 180))

    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    temperature = _temperature_for_run(conf, run_id)
    _, _, gradcam_reason = _resolve_gradcam_target_layer(
        model=model,
        layer_name=gradcam_layer,
    )
    if gradcam_reason:
        raise ValueError(f"plot_gradcam_grid cannot run Grad-CAM: {gradcam_reason}")

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


def plot_gradcam_class_grid(
    cfg_path: str | Path | dict[str, Any],
    seed: int,
    image_paths: list[str],
    gradcam_layer: str | None = None,
    cmap_name: str = "inferno",
    bg_cmap: str = "gray",
    bg_alpha: float = 0.35,
    fg_alpha: float = 0.85,
    save_path: str | Path | None = None,
    dpi: int | None = None,
    true_classes: list[int] | None = None,
    show_correctness_border: bool = False,
    panel_size: tuple[float, float] | None = None,
) -> plt.Figure:
    if LayerGradCam is None:
        raise RuntimeError("captum is required for Grad-CAM. Install with `pip install captum==0.7.0`.")
    if len(image_paths) == 0:
        raise ValueError("image_paths must not be empty")
    if true_classes is not None and len(true_classes) != len(image_paths):
        raise ValueError(
            f"true_classes length ({len(true_classes)}) must match image_paths length ({len(image_paths)})."
        )

    from matplotlib.colors import Normalize

    conf = _cfg(cfg_path)
    device = _resolve_xai_device(conf)
    image_size = _model_image_size(conf)
    num_classes = int(conf["data"]["num_classes"])
    label_order = list(conf["data"]["label_order"])
    out_dpi = int(dpi if dpi is not None else conf.get("xai", {}).get("figure_dpi", 180))
    layer_name = str(gradcam_layer or conf.get("xai", {}).get("gradcam_layer", "layer4")).strip().lower()

    ckpt_path, run_id = _resolve_checkpoint_and_run_id(conf, seed=seed, checkpoint=None, require_existing=True)
    model = _load_model(conf, seed, device, checkpoint=ckpt_path)
    temperature = _temperature_for_run(conf, run_id)

    rows_data: list[dict[str, Any]] = []
    for img_idx, image_path in enumerate(image_paths):
        image_pil, tensor = _load_image_for_inference(
            image_path,
            image_size=image_size,
            preprocessing_cfg=conf.get("preprocessing", {}),
        )
        tensor = tensor.to(device)
        pred_class, confidence, _ = _predict_one_with_temperature(model, tensor, device, temperature)

        class_maps: list[np.ndarray] = []
        out_h, out_w = int(tensor.shape[-2]), int(tensor.shape[-1])
        for cls_idx in range(num_classes):
            heat = _gradcam_heatmap_for_display(
                model=model,
                input_tensor=tensor,
                class_id=int(cls_idx),
                layer_name=layer_name,
                output_size=(out_w, out_h),
                device=device,
            )
            heat = heat * _attribution_retina_mask(heat.shape, conf)
            class_maps.append(heat.astype(np.float32))

        rows_data.append(
            {
                "image_pil": image_pil,
                "pred_class": int(pred_class),
                "true_class": int(true_classes[img_idx]) if true_classes is not None else None,
                "confidence": float(confidence),
                "class_maps": class_maps,
            }
        )

    cmap = plt.get_cmap(str(cmap_name))
    norm = Normalize(vmin=0.0, vmax=1.0)

    if panel_size is None:
        effective_panel_size = (2.2, 2.8)
    else:
        effective_panel_size = (float(panel_size[0]), float(panel_size[1]))

    return _plot_attribution_grid(
        rows_data=rows_data,
        num_classes=num_classes,
        label_order=label_order,
        cmap=cmap,
        norm=norm,
        colorbar_label=f"Grad-CAM intensity ({layer_name})",
        panel_size=effective_panel_size,
        bg_cmap=bg_cmap,
        bg_alpha=bg_alpha,
        fg_alpha=fg_alpha,
        show_correctness_border=show_correctness_border,
        save_path=save_path,
        dpi=out_dpi,
    )
