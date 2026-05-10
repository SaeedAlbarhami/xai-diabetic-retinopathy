"""Single-image XAI demo: quick prediction, detailed per-layer explanations, and report-figure rendering."""
from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import (
    _cfg,
    _load_image_for_inference,
    _load_json,
    _manifest_path,
    _model_image_size,
    prepare_data_manifests,
)
from src.train import (
    _calibration_path_for_run_id,
    _load_model,
    _resolve_checkpoint_and_run_id,
)
from src.xai_common import (
    _resolve_xai_device,
)
from src.xai_gradcam import (
    _generate_gradcam,
    plot_gradcam_class_grid,
)
from src.xai_metrics import (
    _attribution_mass_ratios,
    _attribution_retina_mask,
    _faithfulness_multi_k,
)
from src.xai_shap import (
    _empty_mps_cache_if_available,
    _pick_shap_map,
    _shap_to_2d,
    _shap_values_with_known_warning_filter,
    _should_retry_shap_on_cpu,
    build_shap_session,
    plot_shap_grid,
)
from src.xai_stats import _xai_pass_flag
from src.xai_viz import _save_map_overlay

try:
    import shap
except ImportError:
    shap = None

try:
    from captum.attr import LayerGradCam
except ImportError:
    LayerGradCam = None

from src.xai_audit import _xai_pass_rule_thresholds


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
        if shap is None:
            raise RuntimeError("shap is required for SHAP attribution. Install with `pip install shap==0.47.2`.")

        bg_size = int(shap_background_size or conf["xai"].get("shap_background_size", 64))
        bg_size = max(1, bg_size)

        train_manifest = _manifest_path(conf, "train", seed=seed)
        if not train_manifest.exists():
            prepare_data_manifests(conf, seed=seed)

        def _run_single_shap(shap_device: torch.device) -> dict[str, Any]:
            shap_model, explainer = build_shap_session(
                conf=conf,
                seed=seed,
                ckpt_path=ckpt_path,
                primary_device=device,
                shap_device=shap_device,
                source_model=model,
                bg_size=bg_size,
            )
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
        # Look up true class from manifest when available, for correctness display.
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

    demo_image_path = str(image_path).strip()
    single_demo = run_single_case_demo(
        cfg_path=conf,
        seed=int(seed),
        image_path=demo_image_path,
        split=str(split).strip().lower() or "test",
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

    single_result = single_demo.get("single_result", {}) or {}
    gradcam_details_list = single_result.get("gradcam_details", []) or []
    shap_details = single_result.get("shap_details", {}) or {}
    primary_gradcam_layer = str(xai_cfg.get("gradcam_layer", "layer4"))
    gcam_row = None
    for entry in gradcam_details_list:
        if str(entry.get("layer", "")) == primary_gradcam_layer:
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
            "| Method | Border ratio (↓) | Retina ratio (↑) | Δ$_{k20}$ (↑) | AOPC (↑) |",
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
            "- **Border ratio** (↓ lower is better) — share of the heatmap falling on the dark corners outside the eye.",
            "- **Retina ratio** (↑ higher is better) — share landing inside the retinal disc. Low border plus high retina means attention stays on the eye.",
            "- **Δ$_{k20}$** (↑ higher is better) — drop in the model's confidence when the top 20\\% most-important pixels are removed. A bigger drop means those pixels really mattered to the prediction.",
            "- **AOPC** (↑ higher is better) — the same idea as Δ$_{k20}$ averaged across several removal sizes (a smoother version).",
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

