"""SHAP DeepExplainer with backbone forward-pass patches and the per-class attribution grid."""
from __future__ import annotations

import gc
import types
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.efficientnet import FusedMBConv, MBConv

try:
    import shap
except ImportError:
    shap = None

from src.data import (
    _cfg,
    _load_image_for_inference,
    _manifest_path,
    _model_image_size,
)
from src.train import (
    _LogitWrapper,
    _load_model,
    _resolve_checkpoint_and_run_id,
)
from src.xai_common import (
    _predict_one_with_temperature,
    _resolve_xai_device,
    _temperature_for_run,
)
from src.xai_viz import _normalize_map, _plot_attribution_grid


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
        if isinstance(module, MBConv):
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
    if shap is None:
        raise RuntimeError("shap is required. Install with `pip install shap==0.47.2`.")
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

    if panel_size is None:
        effective_panel_size = (2.2, 2.8)
    else:
        effective_panel_size = (float(panel_size[0]), float(panel_size[1]))

    fig = _plot_attribution_grid(
        rows_data=rows_data,
        num_classes=num_classes,
        label_order=label_order,
        cmap=cmap,
        norm=norm,
        colorbar_label="SHAP value",
        panel_size=effective_panel_size,
        bg_cmap=bg_cmap,
        bg_alpha=bg_alpha,
        fg_alpha=shap_alpha,
        show_correctness_border=show_correctness_border,
        save_path=save_path,
        dpi=out_dpi,
    )
    if shap_fallback_used:
        print(f"SHAP fallback used: attempted_device={device}, final_device={shap_final_device}, primary_error={shap_primary_error}")
    return fig

