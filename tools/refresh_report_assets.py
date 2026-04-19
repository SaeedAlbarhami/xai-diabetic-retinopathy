"""Refresh stale figures under ``src/report/assets/`` from the latest run.

Regenerates or copies the four figures that can drift between runs:

* ``figure08_xai_explanation_pass_rate.png`` — rebuilt via the XAI summary
  helper so the softened chart title ("Operational Threshold Summary") matches
  the current ``src/xai.py`` implementation, not the pre-reframe title.
* ``figure09_gradcam_demo_grid.png`` — copied from the latest
  ``artifacts/reports/figures/gradcam_demo_grid.png``.
* ``figure10_shap_demo_grid.png`` — copied from the latest
  ``artifacts/reports/figures/shap_demo_grid.png``.
* ``figure11_single_case_combined.png`` — composed from the latest
  single-case Grad-CAM panel + SHAP grid under
  ``artifacts/reports/figures/single/``.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_project_config  # noqa: E402
from src.xai import notebook_load_xai_committee_summary  # noqa: E402

ASSETS = PROJECT_ROOT / "src" / "report" / "assets"
FIGS_DIR = PROJECT_ROOT / "artifacts" / "reports" / "figures"
SINGLE_DIR = FIGS_DIR / "single"
SEED = 1988
SPLIT = "test"
DPI = 300


def _save_figure(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _latest(pattern: str) -> Path:
    matches = sorted(SINGLE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"no matches for {pattern} in {SINGLE_DIR}")
    return matches[0]


def refresh_figure08(cfg) -> None:
    """Regenerate the pass-rate chart with the current softened title."""
    summary = notebook_load_xai_committee_summary(cfg, seed=SEED, split=SPLIT)
    fig = summary.get("pass_rate_fig")
    if fig is None:
        raise RuntimeError("pass_rate_fig not produced by notebook_load_xai_committee_summary")
    _save_figure(fig, ASSETS / "figure08_xai_explanation_pass_rate.png")
    print(f"wrote {ASSETS / 'figure08_xai_explanation_pass_rate.png'}")


def refresh_demo_grids() -> None:
    """Copy the latest demo-grid PNGs into the report assets."""
    for src_name, dst_name in [
        ("gradcam_demo_grid.png", "figure09_gradcam_demo_grid.png"),
        ("shap_demo_grid.png", "figure10_shap_demo_grid.png"),
    ]:
        src = FIGS_DIR / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = ASSETS / dst_name
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


def refresh_figure11() -> None:
    """Compose the single-case combined figure: Grad-CAM panel + SHAP grid stacked.

    The SHAP grid is downscaled to the Grad-CAM panel's width to keep the
    composed PNG at a reasonable file size (~2 MB) while preserving visual
    layout.
    """
    gradcam_path = _latest("*_gradcam_panel.png")
    shap_path = _latest("*_shap_grid.png")

    top = Image.open(gradcam_path).convert("RGB")
    bottom = Image.open(shap_path).convert("RGB")

    target_w = top.width
    if bottom.width != target_w:
        new_h = int(round(bottom.height * (target_w / bottom.width)))
        bottom = bottom.resize((target_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, top.height + bottom.height), "white")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height))

    out_path = ASSETS / "figure11_single_case_combined.png"
    canvas.save(out_path, format="PNG", optimize=True)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.2f} MB, from {gradcam_path.name} + {shap_path.name})")


def main() -> int:
    cfg = load_project_config(PROJECT_ROOT / "configs" / "base.yaml")
    refresh_figure08(cfg)
    refresh_demo_grids()
    refresh_figure11()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
