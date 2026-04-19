#!/usr/bin/env python3
"""Export a 5-panel worked example of the fundus preprocessing pipeline.

Loads a single representative Moderate-DR fundus image from the APTOS 2019
training set and replays every preprocessing step in the exact order and with
the exact parameters used by `src.data._apply_fundus_preprocessing` (see
src/data.py:856-890). Each intermediate state is rendered in a horizontal
strip and saved as `src/report/assets/figure12_preprocessing_example.png`.

The composed figure is the visual companion to the TikZ preprocessing flow
chart and the parameter table in Section 6 of the report. Rerunning this
script is deterministic — the same image and parameters always produce the
same output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_project_config  # noqa: E402


DEMO_SAMPLE_ID = "source_aptos_train_86baef833ae0_1542"
DEMO_IMAGE_PATH = PROJECT_ROOT / "dataset" / "aptos2019" / "train_images" / "86baef833ae0.png"
OUTPUT_PATH = PROJECT_ROOT / "src" / "report" / "assets" / "figure12_preprocessing_example.png"


def _step_resize(img_rgb: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _step_clahe(img_rgb: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    l_chan = clahe.apply(l_chan)
    merged = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _step_ben_graham(img_rgb: np.ndarray, sigma: float, weight: float, bias: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(img_rgb, weight, blur, -weight, bias)


def _step_circle_crop(img_rgb: np.ndarray, crop_ratio: float) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = max(1, int(min(h, w) * crop_ratio * 0.5))
    cv2.circle(mask, (w // 2, h // 2), radius, color=255, thickness=-1)
    out = np.zeros_like(img_rgb)
    out[mask > 0] = img_rgb[mask > 0]
    return out.astype(np.uint8)


def main() -> int:
    if not DEMO_IMAGE_PATH.exists():
        print(f"error: demo image not found at {DEMO_IMAGE_PATH}", file=sys.stderr)
        return 2

    cfg = load_project_config(str(PROJECT_ROOT / "configs" / "base.yaml"))
    prep_cfg = cfg.get("preprocessing", {})
    image_size = int(cfg["data"]["image_size"])

    # Load raw
    raw_img = np.array(Image.open(DEMO_IMAGE_PATH).convert("RGB"), dtype=np.uint8)

    # Apply each stage in the order defined by _apply_fundus_preprocessing
    resized = _step_resize(raw_img, image_size)
    clahed = _step_clahe(
        resized,
        clip_limit=float(prep_cfg.get("clahe_clip_limit", 2.0)),
        grid=max(2, int(prep_cfg.get("clahe_tile_grid", 8))),
    )
    benned = _step_ben_graham(
        clahed,
        sigma=float(prep_cfg.get("gaussian_sigma", max(1.0, image_size / 30.0))),
        weight=float(prep_cfg.get("ben_graham_weight", 4.0)),
        bias=float(prep_cfg.get("ben_graham_bias", 128.0)),
    )
    cropped = _step_circle_crop(
        benned,
        crop_ratio=float(prep_cfg.get("circle_crop_ratio", 1.00)),
    )

    # Compose 5-panel figure
    panels = [
        ("Raw fundus", raw_img),
        (f"Resize {image_size}x{image_size}", resized),
        ("CLAHE (LAB L-channel)", clahed),
        ("Ben-Graham subtraction", benned),
        ("Circle crop (ratio=1.0)", cropped),
    ]

    fig_dpi = int(cfg.get("xai", {}).get("figure_dpi", 600))
    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 3.2, 3.4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {OUTPUT_PATH}")
    print(f"sample_id={DEMO_SAMPLE_ID}, true_class=2 (Moderate DR)")
    print(
        f"parameters: image_size={image_size}, "
        f"clahe_clip={prep_cfg.get('clahe_clip_limit', 2.0)}, "
        f"clahe_tile={prep_cfg.get('clahe_tile_grid', 8)}, "
        f"gaussian_sigma={prep_cfg.get('gaussian_sigma', 10.0)}, "
        f"ben_graham_weight={prep_cfg.get('ben_graham_weight', 4.0)}, "
        f"ben_graham_bias={prep_cfg.get('ben_graham_bias', 128.0)}, "
        f"circle_crop_ratio={prep_cfg.get('circle_crop_ratio', 1.0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
