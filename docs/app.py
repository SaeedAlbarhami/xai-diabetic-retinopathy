"""Local demo web app for the Explainable AI Diabetic Retinopathy pipeline.

Single-script flow:
1. Validate prerequisites (trained checkpoint, calibration, source images).
2. For each of the five DR classes, pick the highest-confidence correctly
   predicted test image and run ``explain_single_image_detailed`` to produce
   the prediction probabilities, Grad-CAM overlay, and SHAP overlay.
3. Cache everything under ``docs/static/cases/<class_id>/``.
4. Render ``docs/index.html`` from the cached data.
5. Serve the demo on http://127.0.0.1:5050 and open the browser.

Usage::

    python docs/app.py            # full run (generates cases if missing)
    python docs/app.py --rebuild  # force regeneration even if cached
    python docs/app.py --no-open  # do not open browser
    python docs/app.py --port 5050

Caches are deterministic; rerunning without ``--rebuild`` reuses outputs.
"""
from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from copy import deepcopy  # noqa: E402

from src.data import load_project_config  # noqa: E402
from src.xai_single import run_single_case_demo  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEMO_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = DEMO_ROOT / "static"
CASES_ROOT = STATIC_ROOT / "cases"
INDEX_PATH = DEMO_ROOT / "index.html"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
LABEL_ORDER = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]  # canonical project labels

# Headline test-set metrics from the committed seed-1988 run (artifacts/reports/tables/final_headline_metrics_*.csv).
# These are corpus-level — they describe the trained model on the full 550-image held-out split.
MODEL_PERFORMANCE = {
    "test_size": 550,
    "accuracy": 0.8218,
    "precision_macro": 0.7104,
    "recall_macro": 0.7408,
    "f1_macro": 0.7105,
    "qwk": 0.8958,
    "ece_before": 0.079,
    "ece_after": 0.044,
    "audit_size": 120,
    "gradcam_pass_rate": 0.5583,
    "shap_pass_rate": 0.3917,
}

# Five demo cases — highest-confidence test images per class where BOTH
# Grad-CAM and SHAP pass the operational rule (border ≤ 0.25 AND Δₖ₂₀ > 0.10)
# in the committed XAI audit (rq1_gradcam_seed1988_test.csv + rq2_shap_seed1988_test.csv,
# Grad-CAM at layer2). Severe is structurally low-confidence even among passing cases.
DEMO_CASES = [
    {"class_id": 0, "image_id": "4704dbb59536"},  # No DR,            conf 1.000
    {"class_id": 1, "image_id": "78bcdffb8785"},  # Mild,             conf 0.857
    {"class_id": 2, "image_id": "3c53198519f7"},  # Moderate,         conf 0.886
    {"class_id": 3, "image_id": "4fecf87184e6"},  # Severe,           conf 0.639 (best both-pass)
    {"class_id": 4, "image_id": "a3fcf42ff56d"},  # Proliferative DR, conf 0.837
]

TRIAGE_FRAMING = {
    0: "No DR pattern dominant. Continue routine follow-up if clinical exam is consistent.",
    1: "Mild non-proliferative DR pattern. Annual screening with documentation of progression markers.",
    2: "Moderate non-proliferative DR pattern. Refer for ophthalmology evaluation; consider 6-month follow-up imaging.",
    3: "Severe non-proliferative DR pattern. Prompt retinal-specialist referral indicated.",
    4: "Proliferative DR pattern. URGENT referral for treatment evaluation (anti-VEGF, panretinal photocoagulation, or vitrectomy).",
}


# --------------------------------------------------------------------------------------
# Case generation
# --------------------------------------------------------------------------------------


def _source_image_path(image_id: str) -> Path:
    return PROJECT_ROOT / "dataset" / "aptos2019" / "train_images" / f"{image_id}.png"


def _case_dir(class_id: int) -> Path:
    return CASES_ROOT / f"case_{class_id}"


def _case_is_cached(class_id: int) -> bool:
    d = _case_dir(class_id)
    return all(
        (d / name).exists()
        for name in ("fundus.png", "gradcam.png", "shap.png", "data.json")
    )


def _build_clinical_narrative(result: dict) -> list[str]:
    """Build the clinical decision-support narrative as an ordered list of bullets.

    Format mirrors the project rubric:
      - Predicted DR grade
      - Next most likely alternative
      - Suggested triage framing
      - Confidence note
      - How-to-use disclaimer
      - Clinical safeguard
    """
    pred = int(result.get("pred_class", 0))
    conf = float(result.get("confidence", 0.0))
    probs = [float(result.get(f"prob_{i}", 0.0)) for i in range(len(LABEL_ORDER))]

    # Find the second-highest probability class (deterministic tie break by class index).
    ranked = sorted(enumerate(probs), key=lambda x: (-x[1], x[0]))
    second_idx, second_prob = ranked[1] if len(ranked) > 1 else (pred, 0.0)

    return [
        f"Predicted DR grade: {LABEL_ORDER[pred]} ({pred}) with probability {conf:.3f}.",
        f"Next most likely alternative: {LABEL_ORDER[second_idx]} ({second_prob:.3f}).",
        f"Suggested triage framing: {TRIAGE_FRAMING.get(pred, 'Review with an ophthalmology specialist.')}",
        "Confidence note: probabilities support triage review and should be interpreted with image quality and clinical context.",
        "How to use maps: compare Grad-CAM and SHAP panels to verify that highlighted regions fall on retinal tissue.",
        "Clinical safeguard: explanations are decision-support evidence only and do not replace clinician grading.",
    ]


def _generate_case(case: dict) -> dict:
    """Run the per-class single-case demo on one image and save artefacts under docs/static/cases/.

    Uses ``run_single_case_demo`` so the Grad-CAM and SHAP visualisations match the
    per-class grids in the report (Figures 14 and 15): one row showing the original
    fundus followed by attribution panels targeting each of the five DR classes.
    """
    class_id = int(case["class_id"])
    image_id = str(case["image_id"])
    image_path = _source_image_path(image_id)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Source image missing: {image_path}. Make sure the APTOS 2019 dataset is in place."
        )

    case_dir = _case_dir(class_id)
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"  -> generating case {class_id} ({CLASS_NAMES[class_id]}) from {image_id}.png ...")
    t0 = time.time()
    # Override the project's default Grad-CAM layer (layer4) so the per-class grid
    # is rendered at the same layer as the audited metrics (layer2). This keeps the
    # visualisation, the colourbar label, and the per-image pass-rule numbers all
    # consistent — the audit subset shows all 5 chosen cases pass at layer2.
    cfg_override = deepcopy(load_project_config(PROJECT_ROOT / "configs" / "base.yaml"))
    cfg_override.setdefault("xai", {})["gradcam_layer"] = "layer2"
    demo_result = run_single_case_demo(
        cfg_path=cfg_override,
        seed=1988,
        image_path=str(image_path),
        split="test",
        gradcam_layers=["layer2"],
        shap_background_size=16,
    )
    single = demo_result.get("single_result", {}) or {}
    elapsed = time.time() - t0
    pred_class = int(demo_result.get("pred_class", -1))
    confidence = float(demo_result.get("confidence", float("nan")))
    print(f"     done in {elapsed:.1f}s; pred_class={pred_class} confidence={confidence:.3f}")

    # Copy fundus thumbnail (raw image) plus the Grad-CAM and SHAP per-class grids.
    shutil.copy2(image_path, case_dir / "fundus.png")

    gc_grid = demo_result.get("gradcam_grid_path") or ""
    if gc_grid and Path(gc_grid).exists():
        shutil.copy2(gc_grid, case_dir / "gradcam.png")
    else:
        raise RuntimeError(f"Grad-CAM grid missing for case {class_id}: {gc_grid}")

    shap_grid = demo_result.get("shap_grid_path") or ""
    if shap_grid and Path(shap_grid).exists():
        shutil.copy2(shap_grid, case_dir / "shap.png")
    else:
        (case_dir / "shap.png").unlink(missing_ok=True)

    probs = [float(single.get(f"prob_{i}", 0.0)) for i in range(len(CLASS_NAMES))]
    # Append a cache-busting query string keyed on the file's mtime so the
    # browser always fetches the freshly-generated PNG even when the URL
    # path is unchanged across reruns.
    def _bust(p: Path) -> str:
        return f"static/cases/case_{class_id}/{p.name}?v={int(p.stat().st_mtime)}" if p.exists() else ""

    # Per-case XAI comparison metrics (Grad-CAM at the audit layer vs SHAP), defined in src/xai_metrics.py.
    gc_details = single.get("gradcam_details", []) or []
    gc_primary = gc_details[0] if gc_details else {}
    shap_details = single.get("shap_details", {}) or {}

    def _safe(d: dict, key: str) -> float:
        v = d.get(key, float("nan"))
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return v

    def _to_bool(v) -> bool | None:
        if v is True or v == 1 or v == 1.0:
            return True
        if v is False or v == 0 or v == 0.0:
            return False
        return None

    xai_metrics = {
        "gradcam": {
            "border_ratio": _safe(gc_primary, "border_ratio"),
            "retina_ratio": _safe(gc_primary, "retina_ratio"),
            "faith_delta_k20": _safe(gc_primary, "faith_delta_k20"),
            "aopc_delta": _safe(gc_primary, "aopc_delta"),
            "passes_rule": _to_bool(gc_primary.get("gradcam_pass")),
        },
        "shap": {
            "border_ratio": _safe(shap_details, "border_ratio"),
            "retina_ratio": _safe(shap_details, "retina_ratio"),
            "faith_delta_k20": _safe(shap_details, "faith_delta_k20"),
            "aopc_delta": _safe(shap_details, "aopc_delta"),
            "passes_rule": _to_bool(shap_details.get("shap_pass")),
        },
        "rule": {
            "border_ratio_max": 0.25,
            "faith_delta_k20_min": 0.10,
        },
    }

    metadata = {
        "class_id": class_id,
        "image_id": image_id,
        "true_class": class_id,
        "true_class_name": CLASS_NAMES[class_id],
        "true_label": LABEL_ORDER[class_id],
        "pred_class": pred_class,
        "pred_class_name": CLASS_NAMES[pred_class] if 0 <= pred_class < len(CLASS_NAMES) else str(pred_class),
        "pred_label": LABEL_ORDER[pred_class] if 0 <= pred_class < len(LABEL_ORDER) else str(pred_class),
        "confidence": confidence,
        "probabilities": probs,
        "narrative": _build_clinical_narrative(single),
        "shap_available": (case_dir / "shap.png").exists(),
        "fundus_url": _bust(case_dir / "fundus.png"),
        "gradcam_url": _bust(case_dir / "gradcam.png"),
        "shap_url": _bust(case_dir / "shap.png"),
        "xai_metrics": xai_metrics,
    }
    (case_dir / "data.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def ensure_cases_ready(rebuild: bool = False) -> list[dict]:
    """Generate any missing cases. Returns the list of case metadata dicts."""
    CASES_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_list: list[dict] = []
    for case in DEMO_CASES:
        class_id = int(case["class_id"])
        if not rebuild and _case_is_cached(class_id):
            md = json.loads((_case_dir(class_id) / "data.json").read_text())
            print(f"  ✓ case {class_id} ({CLASS_NAMES[class_id]}) cached, reusing")
            metadata_list.append(md)
            continue
        md = _generate_case(case)
        metadata_list.append(md)
    return metadata_list


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Explainable AI for Diabetic Retinopathy — Live Demo</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-2: #273449;
    --border: #334155;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --accent-2: #34d399;
    --warn: #fbbf24;
    --danger: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    padding: 24px 32px 16px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  }
  header h1 { margin: 0 0 12px; font-size: 22px; font-weight: 600; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  .model-strip {
    display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px;
    font-size: 12px;
  }
  .stat {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    min-width: 120px;
  }
  .stat .lbl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat .v { font-size: 16px; font-weight: 600; font-family: ui-monospace, monospace; }
  .stat .sub { color: var(--muted); font-size: 10px; margin-top: 2px; }
  .thumbs {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    padding: 20px 32px;
    border-bottom: 1px solid var(--border);
  }
  .thumb {
    cursor: pointer;
    border: 2px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    background: var(--panel);
    transition: transform 0.12s ease, border-color 0.12s ease;
  }
  .thumb:hover { transform: translateY(-2px); border-color: var(--accent); }
  .thumb.active { border-color: var(--accent-2); box-shadow: 0 0 0 2px rgba(52,211,153,0.25); }
  .thumb img { width: 100%; height: 130px; object-fit: cover; display: block; }
  .thumb-label { padding: 8px 10px; font-size: 12px; }
  .thumb-label .cls { font-weight: 600; }
  .thumb-label .id { color: var(--muted); font-family: ui-monospace, monospace; font-size: 11px; }
  main { padding: 24px 32px 48px; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }
  .panel h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; }
  .pred-row { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
  .pred-badge {
    background: var(--panel-2);
    border-radius: 10px;
    padding: 12px 18px;
    border-left: 4px solid var(--accent-2);
  }
  .pred-badge .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .pred-badge .value { font-size: 22px; font-weight: 600; }
  .pred-badge.warn { border-left-color: var(--warn); }
  .pred-badge.danger { border-left-color: var(--danger); }
  table.probs { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.probs th, table.probs td { padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left; }
  table.probs th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  .bar-wrap { background: #0b1220; border-radius: 6px; height: 16px; overflow: hidden; min-width: 200px; }
  .bar { height: 100%; background: var(--accent); transition: width 0.3s ease; }
  .bar.pred { background: var(--accent-2); }
  .grids { display: grid; grid-template-columns: 1fr; gap: 16px; }
  .grids .panel { margin-bottom: 0; }
  .grids img { width: 100%; border-radius: 8px; display: block; background: #0b1220; }
  .grids p { color: var(--muted); font-size: 12px; margin: 8px 0 0; }
  .narrative { font-size: 14px; line-height: 1.65; }
  .narrative ul { list-style: none; padding: 0; margin: 0; }
  .narrative li { padding: 10px 12px; margin: 0 0 8px; background: var(--panel-2); border-left: 3px solid var(--accent); border-radius: 6px; }
  .narrative li:last-child { margin: 0; }
  .narrative li b { color: var(--accent); }
  table.xai { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.xai th, table.xai td { padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: left; }
  table.xai th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
  table.xai td.num { font-family: ui-monospace, monospace; text-align: right; }
  table.xai td.win { color: var(--accent-2); font-weight: 600; }
  table.xai td.tie { color: var(--muted); }
  table.xai td.fail { color: var(--danger); }
  table.xai td.pass { color: var(--accent-2); }
  .winner-pill {
    display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
    font-weight: 600; margin-left: 6px;
  }
  .winner-pill.gc { background: rgba(56,189,248,0.15); color: var(--accent); }
  .winner-pill.sh { background: rgba(52,211,153,0.15); color: var(--accent-2); }
  .winner-pill.tie { background: rgba(148,163,184,0.15); color: var(--muted); }
  .help { color: var(--muted); font-size: 11px; margin-top: 8px; }
  .col-legend {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr;
    gap: 6px;
    margin: 4px 0 12px;
  }
  .col-cell {
    background: var(--panel-2);
    border-radius: 6px;
    padding: 6px 8px;
    border: 1px solid var(--border);
    text-align: center;
  }
  .col-cell .c-name { font-size: 12px; font-weight: 600; }
  .col-cell .c-prob { font-size: 11px; color: var(--muted); font-family: ui-monospace, monospace; }
  .col-cell.original { background: rgba(56,189,248,0.10); border-color: var(--accent); }
  .col-cell.predicted { background: rgba(52,211,153,0.18); border-color: var(--accent-2); }
  .col-cell.predicted .c-name { color: var(--accent-2); }
  .col-cell .c-tag { font-size: 9px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--accent-2); margin-top: 2px; font-weight: 700; }
  .empty { text-align: center; color: var(--muted); padding: 60px 0; }
  .footer { color: var(--muted); font-size: 11px; text-align: center; padding: 24px; }
  @media (max-width: 1000px) {
    .thumbs { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<header>
  <h1>Explainable AI for Diabetic Retinopathy — Demo</h1>
  <div class="model-strip" id="modelStrip"></div>
</header>

<section class="thumbs" id="thumbs"></section>

<main id="detail">
  <div class="empty">Select a case above to view the analysis.</div>
</main>

<div class="footer">
  Committed model: EfficientNet-B4 (seed 1988) on APTOS 2019 · Cases pre-rendered from the held-out test split.
</div>

<script>
const CASES = __CASES_JSON__;
const CLASS_NAMES = __CLASS_NAMES_JSON__;
const MODEL_PERF = __MODEL_PERF_JSON__;

function renderModelStrip() {
  const m = MODEL_PERF;
  document.getElementById('modelStrip').innerHTML = `
    <div class="stat"><div class="lbl">Test set</div><div class="v">${m.test_size}</div><div class="sub">held-out images</div></div>
    <div class="stat"><div class="lbl">Accuracy</div><div class="v">${(m.accuracy*100).toFixed(1)}%</div><div class="sub">all classes, full test</div></div>
    <div class="stat"><div class="lbl">QWK</div><div class="v">${m.qwk.toFixed(3)}</div><div class="sub">ordinal-weighted κ</div></div>
    <div class="stat"><div class="lbl">F1 (macro)</div><div class="v">${m.f1_macro.toFixed(3)}</div><div class="sub">unweighted across 5 classes</div></div>
    <div class="stat"><div class="lbl">ECE</div><div class="v">${m.ece_after.toFixed(3)}</div><div class="sub">post temperature scaling</div></div>
    <div class="stat"><div class="lbl">XAI audit</div><div class="v">${m.audit_size}</div><div class="sub">cases (24 / class target)</div></div>
    <div class="stat"><div class="lbl">Grad-CAM pass</div><div class="v">${(m.gradcam_pass_rate*100).toFixed(1)}%</div><div class="sub">audit subset</div></div>
    <div class="stat"><div class="lbl">SHAP pass</div><div class="v">${(m.shap_pass_rate*100).toFixed(1)}%</div><div class="sub">audit subset</div></div>
  `;
}

function renderThumbs() {
  const root = document.getElementById('thumbs');
  root.innerHTML = CASES.map((c, idx) => `
    <div class="thumb" data-idx="${idx}">
      <img src="${c.fundus_url}" alt="case ${c.class_id}">
      <div class="thumb-label">
        <div class="cls">${c.true_class_name}</div>
        <div class="id">${c.image_id}</div>
      </div>
    </div>`).join('');
  root.querySelectorAll('.thumb').forEach(el => {
    el.addEventListener('click', () => selectCase(parseInt(el.dataset.idx, 10)));
  });
}

function selectCase(idx) {
  const c = CASES[idx];
  document.querySelectorAll('.thumb').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });
  const correct = c.pred_class === c.true_class;
  const badgeCls = correct ? '' : (Math.abs(c.pred_class - c.true_class) > 1 ? 'danger' : 'warn');
  const probsRows = c.probabilities.map((p, i) => {
    const isPred = i === c.pred_class;
    return `
      <tr>
        <td>${CLASS_NAMES[i]}${isPred ? ' <span style="color:var(--accent-2)">●</span>' : ''}</td>
        <td><div class="bar-wrap"><div class="bar ${isPred ? 'pred' : ''}" style="width:${(p * 100).toFixed(1)}%"></div></div></td>
        <td style="text-align:right; font-family: ui-monospace, monospace;">${(p * 100).toFixed(1)}%</td>
      </tr>`;
  }).join('');
  const legend = renderColumnLegend(c);
  const shapBlock = c.shap_available
    ? `<div class="panel">
         <h2>SHAP — per-class attribution grid</h2>
         <p class="help" style="margin-top:0;margin-bottom:6px">How to read this row: each column below is one DR class. Below each column header you'll see the same 5 grades and the model's probability for each.</p>
         ${legend}
         <img src="${c.shap_url}" alt="SHAP per-class grid">
         <p>Each panel shows the signed SHAP attribution targeting that specific DR class on this image. <span style="color:#ef4444">Red</span> pixels pushed the prediction <i>toward</i> that class; <span style="color:#3b82f6">blue</span> pixels pushed it <i>away</i>; near-white pixels were neutral. The prediction itself is the highlighted column.</p>
       </div>`
    : `<div class="panel">
         <h2>SHAP — per-class attribution grid</h2>
         <p style="color:var(--warn)">SHAP output unavailable for this case (see logs).</p>
       </div>`;
  document.getElementById('detail').innerHTML = `
    <div class="panel">
      <h2>Prediction</h2>
      <div class="pred-row">
        <div class="pred-badge ${badgeCls}">
          <div class="label">Model prediction</div>
          <div class="value">${c.pred_class_name}</div>
        </div>
        <div class="pred-badge" title="Per-image predicted-class probability after temperature scaling: max(softmax(logits / T*)). Not the same as model accuracy (corpus-level metric shown in the header).">
          <div class="label">Confidence (this image)</div>
          <div class="value">${(c.confidence * 100).toFixed(1)}%</div>
        </div>
        <div class="pred-badge">
          <div class="label">Ground truth</div>
          <div class="value">${c.true_class_name}</div>
        </div>
      </div>
      <p class="help">"Confidence" is the per-image predicted-class probability — the model's belief about <i>this</i> case. Corpus-level accuracy (${(MODEL_PERF.accuracy*100).toFixed(1)}%) and QWK (${MODEL_PERF.qwk.toFixed(3)}) are shown in the header bar.</p>
    </div>

    <div class="panel">
      <h2>Probability distribution</h2>
      <table class="probs">
        <thead><tr><th>Class</th><th>Bar</th><th style="text-align:right">Probability</th></tr></thead>
        <tbody>${probsRows}</tbody>
      </table>
    </div>

    <div class="grids">
      <div class="panel">
        <h2>Grad-CAM — per-class attention grid (layer 2)</h2>
        ${legend}
        <img src="${c.gradcam_url}" alt="Grad-CAM per-class grid">
        <p>Each heatmap is the model's class-conditional attention: <i>"if asked to predict <b>this class</b>, where does the model look?"</i> The <b>predicted class</b> column is the actual answer the model gave for this image; the other 4 are counterfactual views (e.g. "what would convince me of Severe?"). Brighter/yellower regions = stronger evidence, darker/purple = weaker. Masked to the retinal disc to remove the corner upsampling artefact.</p>
      </div>
      ${shapBlock}
    </div>

    ${renderXaiComparison(c.xai_metrics)}

    <div class="panel">
      <h2>Clinical Decision Support Narrative</h2>
      <div class="narrative">
        <ul>${c.narrative.map(line => {
          const idx = line.indexOf(':');
          if (idx > 0 && idx < 60) {
            return `<li><b>${line.slice(0, idx)}:</b>${line.slice(idx + 1)}</li>`;
          }
          return `<li>${line}</li>`;
        }).join('')}</ul>
      </div>
    </div>
  `;
}

function fmt(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return 'n/a';
  return Number(v).toFixed(digits);
}

const LABEL_ORDER_JS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"];

function renderColumnLegend(c) {
  // The 6 columns of every grid image, in left-to-right order:
  //   1: original fundus, 2..6: per-class attribution at class indexes 0..4.
  const cells = [];
  cells.push(`<div class="col-cell original"><div class="c-name">Original</div><div class="c-prob">fundus</div></div>`);
  for (let i = 0; i < 5; i++) {
    const isPred = i === c.pred_class;
    cells.push(`
      <div class="col-cell ${isPred ? 'predicted' : ''}">
        <div class="c-name">${LABEL_ORDER_JS[i]}</div>
        <div class="c-prob">p = ${c.probabilities[i].toFixed(3)}</div>
        ${isPred ? '<div class="c-tag">predicted</div>' : ''}
      </div>`);
  }
  return `<div class="col-legend">${cells.join('')}</div>`;
}

function renderXaiComparison(m) {
  if (!m) return '';
  const rule = m.rule || { border_ratio_max: 0.25, faith_delta_k20_min: 0.10 };

  // Direction: lower-is-better for border_ratio; higher-is-better for the rest.
  const rows = [
    { key: 'border_ratio',     label: 'Border ratio',          better: 'lower',  rule: `≤ ${rule.border_ratio_max.toFixed(2)}`,         help: 'Fraction of attribution mass on the 10% border band — lower means the explanation stays inside the retina.' },
    { key: 'retina_ratio',     label: 'Retina ratio',          better: 'higher', rule: `(complementary)`,                                help: 'Fraction of attribution mass inside the retinal disc — higher is better.' },
    { key: 'faith_delta_k20',  label: 'Faithfulness Δₖ₂₀',     better: 'higher', rule: `> ${rule.faith_delta_k20_min.toFixed(2)}`,      help: 'Drop in predicted-class probability when the top 20% most-attributed pixels are masked, vs a random 20% — higher means the highlighted pixels actually drive the prediction.' },
    { key: 'aopc_delta',       label: 'AOPC',                  better: 'higher', rule: `(higher better)`,                                help: 'Area Over the Perturbation Curve — averaged faithfulness across multiple top-k removals.' },
  ];

  const winnerCount = { gradcam: 0, shap: 0 };
  const rowsHtml = rows.map(r => {
    const gv = m.gradcam[r.key];
    const sv = m.shap[r.key];
    let gcCls = 'num', shCls = 'num';
    if (Number.isFinite(gv) && Number.isFinite(sv)) {
      const gWins = (r.better === 'lower') ? gv < sv : gv > sv;
      const sWins = (r.better === 'lower') ? sv < gv : sv > gv;
      if (gWins) { gcCls = 'num win'; winnerCount.gradcam++; }
      else if (sWins) { shCls = 'num win'; winnerCount.shap++; }
    }
    return `
      <tr>
        <td title="${r.help}">${r.label} <span style="color:var(--muted);font-size:11px">(${r.better === 'lower' ? '↓' : '↑'})</span></td>
        <td class="${gcCls}">${fmt(gv, 3)}</td>
        <td class="${shCls}">${fmt(sv, 3)}</td>
        <td style="color:var(--muted);font-size:11px">${r.rule}</td>
      </tr>`;
  }).join('');

  // Operational rule pass / fail row
  const gPass = m.gradcam.passes_rule;
  const sPass = m.shap.passes_rule;
  const passLabel = (b) => b === true ? '<span class="pass">passes</span>' : (b === false ? '<span class="fail">fails</span>' : 'n/a');
  const passRow = `
    <tr>
      <td title="border ratio ≤ 0.25 AND faithfulness Δₖ₂₀ > 0.10 on masked attribution maps">Operational pass rule</td>
      <td class="num">${passLabel(gPass)}</td>
      <td class="num">${passLabel(sPass)}</td>
      <td style="color:var(--muted);font-size:11px">both must hold</td>
    </tr>`;

  let summary = '';
  if (winnerCount.gradcam > winnerCount.shap) {
    summary = `<span class="winner-pill gc">Grad-CAM wins ${winnerCount.gradcam} / 4 metrics on this image</span>`;
  } else if (winnerCount.shap > winnerCount.gradcam) {
    summary = `<span class="winner-pill sh">SHAP wins ${winnerCount.shap} / 4 metrics on this image</span>`;
  } else {
    summary = `<span class="winner-pill tie">Tied ${winnerCount.gradcam} / 4 metrics on this image</span>`;
  }

  return `
    <div class="panel">
      <h2>XAI Comparison — Grad-CAM vs SHAP on this image ${summary}</h2>
      <table class="xai">
        <thead><tr><th>Metric</th><th style="text-align:right">Grad-CAM</th><th style="text-align:right">SHAP</th><th>Operational rule</th></tr></thead>
        <tbody>${rowsHtml}${passRow}</tbody>
      </table>
      <p class="help">Metrics defined in <code>src/xai_metrics.py</code>. Lower border ratio = explanation stays inside the retina. Higher Δₖ₂₀ / AOPC = the highlighted pixels actually drive the prediction. The operational pass rule (border ≤ 0.25 AND Δₖ₂₀ > 0.10) is descriptive — primary comparison is the continuous metrics.</p>
    </div>
  `;
}

renderModelStrip();
renderThumbs();
selectCase(0);
</script>
</body>
</html>
"""


def write_index_html(cases: list[dict]) -> None:
    html = (
        HTML_TEMPLATE
        .replace("__CASES_JSON__", json.dumps(cases))
        .replace("__CLASS_NAMES_JSON__", json.dumps(CLASS_NAMES))
        .replace("__MODEL_PERF_JSON__", json.dumps(MODEL_PERFORMANCE))
    )
    INDEX_PATH.write_text(html)
    print(f"  ✓ wrote {INDEX_PATH}")


# --------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------


class _DemoHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - keep stdlib signature
        # Quieter than the default per-request logger.
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def end_headers(self):
        # Force every browser to revalidate so a regenerated case never gets
        # served from the user's local cache mid-demo.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def start_server(port: int, open_browser: bool) -> None:
    handler = lambda *a, **kw: _DemoHandler(*a, directory=str(DEMO_ROOT), **kw)  # noqa: E731
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/index.html"
        print(f"\nServing demo at {url}")
        print("Press Ctrl+C to stop.\n")
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate cases even if cached.")
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab.")
    parser.add_argument("--port", type=int, default=5050, help="Port to listen on (default 5050).")
    parser.add_argument("--prepare-only", action="store_true", help="Generate cases and exit (do not start server).")
    args = parser.parse_args()

    print("Explainable AI for DR — demo")
    print("=" * 60)
    print("Step 1: ensure case studies are generated")
    cases = ensure_cases_ready(rebuild=args.rebuild)
    print(f"  ✓ {len(cases)} cases ready")

    print("Step 2: render index.html")
    write_index_html(cases)

    if args.prepare_only:
        print("\n--prepare-only: skipping server start.")
        return 0

    print("Step 3: start local web server")
    start_server(port=args.port, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
