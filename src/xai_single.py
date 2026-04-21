"""Single-case explanation flow.

Extracted from ``src/xai.py`` as part of the architecture-via-relocation
refactor. Contains the three single-image orchestrators:

* ``predict_single_image_with_explanations``: quick-start prediction + XAI
  for a single fundus image.
* ``explain_single_image_detailed``: full per-layer Grad-CAM + SHAP detail
  with per-case metrics.
* ``run_single_case_demo``: end-to-end single-case demo producing both
  figure-11a (class-conditional Grad-CAM) and figure-11b (SHAP per-class
  grid) plus the supporting metrics table.

Dependency layer: L4 (orchestration). Imports from L1/L2/L3. Never
imports from ``src.xai``.
"""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data import (
    _apply_fundus_preprocessing,
    _backbone_name,
    _cfg,
    _class_name_from_id,
    _load_image_for_inference,
    _load_json,
    _manifest_path,
    _model_image_size,
    _save_json,
    load_project_config,
    prepare_data_manifests,
)
from src.train import (
    _LogitWrapper,
    _calibration_path_for_run_id,
    _load_model,
    _resolve_checkpoint_and_run_id,
)
from src.xai_common import (
    _predict_one_with_temperature,
    _resolve_xai_device,
    _temperature_for_run,
)
from src.xai_gradcam import (
    _generate_gradcam,
    _gradcam_heatmap_for_display,
    plot_gradcam_class_grid,
)
from src.xai_metrics import (
    _attribution_mass_ratios,
    _attribution_retina_mask,
    _faithfulness_multi_k,
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
    plot_shap_grid,
)
from src.xai_stats import _xai_pass_flag
from src.xai_viz import _save_map_overlay

try:
    import shap
except Exception as exc:  # pragma: no cover
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None

try:
    from captum.attr import LayerAttribution, LayerGradCam
except Exception as exc:  # pragma: no cover
    LayerAttribution = None
    LayerGradCam = None
    _CAPTUM_IMPORT_ERROR = exc
else:
    _CAPTUM_IMPORT_ERROR = None

from src.xai_audit import _xai_pass_rule_thresholds, _parse_gradcam_layers


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

    # Build a class-conditional Grad-CAM grid at the config's primary layer
    # so Grad-CAM and SHAP share an identical layout: [Original | per-class...].
    out_dpi = int(shap_dpi if shap_dpi is not None else fig_dpi)
    primary_gradcam_layer = str(conf.get("xai", {}).get("gradcam_layer", gradcam_layer_order[0] if gradcam_layer_order else "layer4"))
    gradcam_grid_path = single_dir / f"{img_path.stem}_gradcam_grid.png"
    fig_g = plot_gradcam_class_grid(
        cfg_path=conf,
        seed=seed,
        image_paths=[str(img_path)],
        gradcam_layer=primary_gradcam_layer,
        save_path=gradcam_grid_path,
        dpi=out_dpi,
        true_classes=[int(true_class)] if true_class is not None else None,
        show_correctness_border=bool(true_class is not None),
        panel_size=shap_panel_size,
    )
    plt.close(fig_g)

    bg_size = int(max(1, shap_background_size if shap_background_size is not None else conf.get("xai", {}).get("shap_background_size", 64)))
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
        "gradcam_grid_path": str(gradcam_grid_path),
        "gradcam_layer": primary_gradcam_layer,
        "shap_grid_path": str(shap_grid_path),
        "single_result": single_result,
        "xai_warnings": list(single_result.get("xai_warnings") or []),
        "shap_background_size": bg_size,
    }

