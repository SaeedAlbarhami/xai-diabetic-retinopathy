"""Runtime-glue helpers shared across the XAI modules.

Small utilities that both ``xai_gradcam`` and ``xai_shap`` need (device
resolution, temperature-scaled prediction, calibration lookup). Living in
this leaf keeps ``xai_gradcam`` and ``xai_shap`` from having to import
upward into ``xai.py``, which would violate the strict-DAG invariant.

This module is a leaf within the ``src.xai_*`` family: it must not import
from any other ``src.xai_*`` sibling. It does reach across into
``src.data`` and ``src.train``, which is allowed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data import _load_json, _resolve_device
from src.train import _calibration_path_for_run_id


def _resolve_xai_device(conf: dict[str, Any]) -> torch.device:
    xai_cfg = conf.get("xai", {}) if isinstance(conf, dict) else {}
    requested = xai_cfg.get("device", None)
    if requested is None or str(requested).strip() == "":
        requested = conf.get("training", {}).get("device", "mps")
    return _resolve_device(str(requested))


@torch.inference_mode()
def _predict_one_with_temperature(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> tuple[int, float, np.ndarray]:
    logits = model(input_tensor.to(device))
    logits = logits / max(1e-4, float(temperature))
    probs = torch.softmax(logits, dim=1)
    conf_score, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf_score.item()), probs.squeeze(0).detach().cpu().numpy()


def _temperature_for_run(conf: dict[str, Any], run_id: str) -> float:
    calibration_path = _calibration_path_for_run_id(conf, run_id)
    if not calibration_path.exists():
        return 1.0
    payload = _load_json(calibration_path)
    return float(payload.get("temperature", 1.0))
