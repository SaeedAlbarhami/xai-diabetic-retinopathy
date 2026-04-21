"""Small runtime helpers shared by the Grad-CAM and SHAP modules.

Device resolution, a temperature-scaled single-image predict, and a
calibration-temperature lookup. Kept in one leaf module so the compute
modules can share them without importing each other.
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
