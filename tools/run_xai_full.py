"""Full N=120 XAI rerun driver for autonomous operation.

Mirrors the notebook flow (sections 4 / 5 / 6 / 7) so this can be invoked
from the shell. Uses the config as-is from disk.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_project_config, notebook_prepare_data_overview  # noqa: E402
from src.train import notebook_run_training  # noqa: E402
from src.xai import notebook_run_xai  # noqa: E402


def main() -> int:
    t0 = time.time()
    cfg = load_project_config(str(PROJECT_ROOT / "configs" / "base.yaml"))
    print(
        f"[run] xai cfg: max_targets={cfg['xai']['max_targets']} "
        f"shap_max_samples={cfg['xai']['shap_max_samples']} "
        f"shap_background_size={cfg['xai']['shap_background_size']} "
        f"mask_radius={cfg['xai'].get('attribution_mask_radius_ratio')}"
    )

    data_overview = notebook_prepare_data_overview(cfg, seed=1988, samples_per_split=1)
    manifests = data_overview["manifest_paths"]

    training_out = notebook_run_training(cfg, seed=1988, manifests=manifests, force_retrain=False)
    ckpt = training_out["checkpoint_path"]
    print(
        f"[run] checkpoint reused={training_out['reused_checkpoint']} "
        f"reason={training_out['reuse_reason']}"
    )

    xai_run = notebook_run_xai(
        cfg, seed=1988, split="test",
        manifests=manifests, checkpoint=ckpt, safe_mode=False,
    )

    elapsed_m = (time.time() - t0) / 60.0
    print(f"[run] XAI pipeline finished in {elapsed_m:.1f} min")
    print(f"[run] run_id={xai_run['run_id']}")

    # Quick sanity
    import pandas as pd
    tables = Path(cfg["paths"]["tables_dir"])
    rq1 = pd.read_csv(tables / "rq1_gradcam_seed1988_test.csv")
    rq2 = pd.read_csv(tables / "rq2_shap_seed1988_test.csv")
    print(f"[run] rq1_gradcam rows={len(rq1)} cols={list(rq1.columns)}")
    print(f"[run] rq2_shap   rows={len(rq2)} cols={list(rq2.columns)}")
    assert "border_ratio_raw" in rq1.columns and "retina_ratio_raw" in rq1.columns, "missing raw cols in rq1"
    assert "border_ratio_raw" in rq2.columns and "retina_ratio_raw" in rq2.columns, "missing raw cols in rq2"
    print("[run] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
