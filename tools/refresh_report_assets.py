"""Refresh stale figures under ``src/report/assets/`` from the latest run.

Regenerates or copies the four figures that can drift between runs:

* ``figure11_xai_explanation_pass_rate.png`` — rebuilt via the XAI summary
  helper so the softened chart title ("Operational Threshold Summary") matches
  the current ``src/xai.py`` implementation, not the pre-reframe title.
* ``figure12_gradcam_demo_grid.png`` — copied from the latest
  ``artifacts/reports/figures/gradcam_demo_grid.png``.
* ``figure13_shap_demo_grid.png`` — copied from the latest
  ``artifacts/reports/figures/shap_demo_grid.png``.
* ``figure14_gradcam_class_grid.png`` and ``figure15_shap_class_grid.png``
  — copied from the latest single-case class-conditional Grad-CAM grid and
  SHAP per-class grid under ``artifacts/reports/figures/single/``. Emitted as
  two separate assets at the same target width so the report can stack them
  as two symmetric subfigures without rescaling.
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


def refresh_figure11(cfg) -> None:
    """Regenerate the pass-rate chart with the current softened title."""
    summary = notebook_load_xai_committee_summary(cfg, seed=SEED, split=SPLIT)
    fig = summary.get("pass_rate_fig")
    if fig is None:
        raise RuntimeError("pass_rate_fig not produced by notebook_load_xai_committee_summary")
    _save_figure(fig, ASSETS / "figure11_xai_explanation_pass_rate.png")
    print(f"wrote {ASSETS / 'figure11_xai_explanation_pass_rate.png'}")


def refresh_demo_grids() -> None:
    """Copy the latest demo-grid PNGs into the report assets."""
    for src_name, dst_name in [
        ("gradcam_demo_grid.png", "figure12_gradcam_demo_grid.png"),
        ("shap_demo_grid.png", "figure13_shap_demo_grid.png"),
    ]:
        src = FIGS_DIR / src_name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = ASSETS / dst_name
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")


def refresh_single_case_grids() -> None:
    """Emit the single-case figure as two separate, equal-width assets.

    Writes ``figure14_gradcam_class_grid.png`` and ``figure15_shap_class_grid.png``
    to ``src/report/assets/``, resized so both PNGs share the same width. Each
    input PNG already renders a (1 x (1+num_classes)) grid from the shared
    renderer, so equal width yields equal per-cell pixel budget.

    The old combined asset (``figure11_single_case_combined.png``) is removed
    if present so the report cannot accidentally pick up a stale stacked
    version.
    """
    gradcam_path = _latest("*_gradcam_grid.png")
    shap_path = _latest("*_shap_grid.png")

    top = Image.open(gradcam_path).convert("RGB")
    bottom = Image.open(shap_path).convert("RGB")

    target_w = max(top.width, bottom.width)
    if top.width != target_w:
        new_h = int(round(top.height * (target_w / top.width)))
        top = top.resize((target_w, new_h), Image.LANCZOS)
    if bottom.width != target_w:
        new_h = int(round(bottom.height * (target_w / bottom.width)))
        bottom = bottom.resize((target_w, new_h), Image.LANCZOS)

    out_gradcam = ASSETS / "figure14_gradcam_class_grid.png"
    out_shap = ASSETS / "figure15_shap_class_grid.png"
    top.save(out_gradcam, format="PNG", optimize=True)
    bottom.save(out_shap, format="PNG", optimize=True)
    print(
        f"wrote {out_gradcam} ({out_gradcam.stat().st_size / (1024 * 1024):.2f} MB, from {gradcam_path.name})"
    )
    print(
        f"wrote {out_shap} ({out_shap.stat().st_size / (1024 * 1024):.2f} MB, from {shap_path.name})"
    )

    stale_combined = ASSETS / "figure11_single_case_combined.png"
    if stale_combined.exists():
        stale_combined.unlink()
        print(f"removed stale {stale_combined}")


def main() -> int:
    cfg = load_project_config(PROJECT_ROOT / "configs" / "base.yaml")
    refresh_figure11(cfg)
    refresh_demo_grids()
    refresh_single_case_grids()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
