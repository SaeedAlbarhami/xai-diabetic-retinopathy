"""Per-sample attribution-map metrics: retinal-disc mask, border ratio, retina ratio, faithfulness delta, AOPC."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _border_mask(shape: tuple[int, int], border_ratio: float = 0.10) -> np.ndarray:
    h, w = shape
    bh = max(1, int(round(h * border_ratio)))
    bw = max(1, int(round(w * border_ratio)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bh, :] = True
    mask[-bh:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    return mask


def _retina_circle_mask(shape: tuple[int, int], radius_ratio: float = 0.45) -> np.ndarray:
    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    r = min(h, w) * radius_ratio
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def _attribution_retina_mask(shape: tuple[int, int], conf: dict) -> np.ndarray:
    override = conf.get("xai", {}).get("attribution_mask_radius_ratio")
    if override is None:
        crop_ratio = float(conf.get("preprocessing", {}).get("circle_crop_ratio", 1.00))
        radius_ratio = crop_ratio * 0.5
    else:
        radius_ratio = float(override)
    return _retina_circle_mask(shape, radius_ratio=radius_ratio)


def _attribution_mass_ratios(attr_map: np.ndarray) -> dict[str, float]:
    arr = np.abs(np.asarray(attr_map, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError("attr_map must be 2D")

    total = float(arr.sum())
    if total <= 1e-12:
        return {"border_ratio": 0.0, "retina_ratio": 0.0}

    border = _border_mask(arr.shape, border_ratio=0.10)
    retina = _retina_circle_mask(arr.shape, radius_ratio=0.45)

    return {
        "border_ratio": float(arr[border].sum() / total),
        "retina_ratio": float(arr[retina].sum() / total),
    }


def _mask_by_score_map(image_tensor: torch.Tensor, score_map: np.ndarray, top_k_ratio: float, random_seed: int = 1988) -> tuple[torch.Tensor, torch.Tensor]:
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("image_tensor must have shape [1, C, H, W]")

    c, h, w = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
    arr = np.asarray(score_map, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"score_map must be 2D, got {arr.shape}")

    if arr.shape != (h, w):
        arr = np.array(
            Image.fromarray(arr.astype(np.float32), mode="F").resize((w, h), resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

    flat = arr.reshape(-1)
    k = max(1, int(round(flat.size * top_k_ratio)))

    top_idx = np.argsort(flat)[-k:]
    rng = np.random.default_rng(random_seed)
    rand_idx = rng.choice(flat.size, size=k, replace=False)

    top_mask = np.zeros(flat.size, dtype=bool)
    top_mask[top_idx] = True
    rand_mask = np.zeros(flat.size, dtype=bool)
    rand_mask[rand_idx] = True

    top_mask_2d = top_mask.reshape(h, w)
    rand_mask_2d = rand_mask.reshape(h, w)

    x_top = image_tensor.clone()
    x_rand = image_tensor.clone()

    top_mask_t = torch.tensor(top_mask_2d, dtype=torch.bool, device=image_tensor.device)
    rand_mask_t = torch.tensor(rand_mask_2d, dtype=torch.bool, device=image_tensor.device)

    for ch in range(c):
        x_top[0, ch][top_mask_t] = 0.0
        x_rand[0, ch][rand_mask_t] = 0.0

    return x_top, x_rand


@torch.inference_mode()
def _faithfulness_delta(model: nn.Module, image_tensor: torch.Tensor, score_map: np.ndarray, pred_class: int, top_k_ratio: float = 0.20, random_seed: int = 1988) -> float:
    """Random-baseline-adjusted faithfulness delta at one k.

    Returns Δ_k = p̂(I_rand_k) − p̂(I_top_k), i.e. (drop after zeroing the
    top-k% most-attributed pixels) minus (drop after zeroing a matched-size
    random subset). Positive ⇒ the attribution's top-k matters more than
    random pixels of the same count. Single random draw per call (R=1) with
    a fixed seed — not an average over R draws.
    """
    logits_base = model(image_tensor)
    probs_base = torch.softmax(logits_base, dim=1)
    base_prob = float(probs_base[0, pred_class].item())

    x_top, x_rand = _mask_by_score_map(image_tensor=image_tensor, score_map=score_map, top_k_ratio=top_k_ratio, random_seed=random_seed)
    probs_top = torch.softmax(model(x_top), dim=1)
    probs_rand = torch.softmax(model(x_rand), dim=1)

    drop_top = base_prob - float(probs_top[0, pred_class].item())
    drop_rand = base_prob - float(probs_rand[0, pred_class].item())
    return float(drop_top - drop_rand)


def _parse_faithfulness_k_list(raw_value: Any) -> list[float]:
    default_k = [0.05, 0.10, 0.20, 0.30]
    values: list[float] = []
    if isinstance(raw_value, (list, tuple)):
        for v in raw_value:
            try:
                k = float(v)
            except (TypeError, ValueError):
                continue
            if 0.0 < k < 1.0:
                values.append(k)
    if not values:
        values = default_k
    values = sorted({round(v, 4) for v in values})
    return values


def _k_to_col_name(k_ratio: float) -> str:
    return f"faith_delta_k{int(round(k_ratio * 100)):02d}"


def _faithfulness_multi_k(
    model: nn.Module,
    image_tensor: torch.Tensor,
    score_map: np.ndarray,
    pred_class: int,
    k_list: list[float],
    random_seed: int = 1988,
) -> tuple[dict[str, float], float]:
    out: dict[str, float] = {}
    for k in k_list:
        col = _k_to_col_name(k)
        out[col] = float(
            _faithfulness_delta(
                model=model,
                image_tensor=image_tensor,
                score_map=score_map,
                pred_class=pred_class,
                top_k_ratio=float(k),
                random_seed=random_seed,
            )
        )
    if out:
        aopc_delta = float(np.mean(list(out.values())))
    else:
        aopc_delta = float("nan")
    return out, aopc_delta
