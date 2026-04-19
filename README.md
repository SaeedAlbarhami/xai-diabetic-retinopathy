# Explainable Diabetic Retinopathy Classification Pipeline

An end-to-end reproducible pipeline for five-class diabetic retinopathy (DR) severity grading on the APTOS 2019 dataset, with post-hoc explainability via Grad-CAM and SHAP and a research-standard XAI audit. Built as the ITPG708 final project.

**At a glance:**

- **Model**: EfficientNet-B4 fine-tuned from ImageNet weights with focal loss (γ=2.0) + temperature scaling for post-hoc calibration.
- **Test metrics (N=550, seed 1988)**: accuracy 82.2%, macro-F1 0.711, QWK 0.896, ECE 0.044 after calibration.
- **Explanation-quality comparison** on an audit subset of 120 class-balanced test targets (24 per DR grade, chosen to keep perturbation-based evaluation computationally tractable while preserving full class coverage); predictive metrics above use the full 550-image test set. Attribution maps are post-processed with the retinal-disc mask correction described in Section 6.2 of the report.
  - **Primary continuous comparison** shows more favourable values for Grad-CAM on all six metrics (paired Wilcoxon p < 10⁻³ each): border 0.206 vs 0.234 (dz=0.32), retina 0.816 vs 0.791 (dz=0.27), AOPC 0.287 vs 0.163 (dz=0.58), Δ_k20 0.328 vs 0.187 (dz=0.54).
  - **Mask ablation** (see [rq_xai_mask_ablation CSV](artifacts/reports/tables/rq_xai_mask_ablation_seed1988_test.csv)): on the raw (pre-mask) maps, SHAP has the lower border ratio (the Grad-CAM bilinear-upsample artefact inflates corner mass); after the symmetric mask, the localisation result is Grad-CAM-favoured. SHAP numbers are bit-identical across conditions (DeepExplainer attributes zero to zero-valued corner pixels), so the correction is Grad-CAM-specific.
  - **Per-class complementarity**: SHAP has more favourable values on No_DR (76.9% vs 53.8%) and Severe (38.1% vs 14.3%); Grad-CAM has more favourable values on Mild/Moderate/Proliferative. Both methods should be reported together in clinical workflows.
  - **Secondary descriptive threshold summary** under the operational rule (border ≤ 0.25 AND Δ_k20 > 0.10): Grad-CAM 55.8% vs SHAP 39.2%, McNemar exact p = 0.012. This is a descriptive companion only; the threshold gap narrows to ~2 pp at a relaxed 0.30 border cut-off, which is why the continuous analysis is treated as primary.

The primary entry point is the notebook [notebooks/project_demo.ipynb](notebooks/project_demo.ipynb). Every stage is idempotent: once a checkpoint, predictions file, or XAI table exists, subsequent runs reuse it unless you explicitly force a rebuild.

---

## Project Layout

```
ITPG708Project/
├── configs/
│   └── base.yaml                    # single source of truth for hyperparameters
├── dataset/
│   └── aptos2019/                   # APTOS 2019 labels and image folders
├── notebooks/
│   └── project_demo.ipynb           # primary entry point — runs the full pipeline
├── src/
│   ├── data.py                      # config, seeds, paths, CSV parsing, splits,
│   │                                # manifests, preprocessing, Dataset classes
│   ├── train.py                     # DRClassifier, focal loss, training loop,
│   │                                # temperature calibration, evaluation,
│   │                                # checkpoint/run plumbing
│   ├── xai.py                       # Grad-CAM, SHAP, audit (border ratio,
│   │                                # faithfulness, McNemar, Wilcoxon), notebook
│   │                                # helpers
├── tools/
│   ├── export_project_demo_assets.py     # exports figure1..figure10 to assets/
│   └── export_preprocessing_steps.py     # produces figure12 preprocessing example
├── artifacts/
│   ├── checkpoints/                 # trained .pt + calibration .json
│   ├── predictions/                 # per-split prediction CSVs
│   ├── manifests/                   # frozen train/val/test splits
│   ├── logs/                        # per-run metadata, training history, XAI status
│   └── reports/
│       ├── tables/                  # evaluation + XAI audit CSVs (see below)
│       └── figures/                 # Grad-CAM / SHAP / single-case overlay PNGs
└── requirements.txt
```

### Module dependency chain

`src/` is a flat three-file layout (no package, no `__init__.py`) with a strict linear dependency chain:

```
data.py  →  train.py  →  xai.py
```

- `data.py` has no upstream dependencies. It can run without PyTorch imports failing. Owns config loading, seeding, path utilities, CSV parsing, stratified splits, manifests, the preprocessing pipeline, and the `_FundusDataset` class.
- `train.py` imports explicit names from `src.data`. Owns the `DRClassifier` model, focal loss, the training loop, temperature calibration, all evaluation metrics, and checkpoint/run plumbing.
- `xai.py` imports from `src.data` and `src.train`. Owns Grad-CAM (via captum), SHAP (via DeepExplainer), the XAI audit (border ratio, multi-k faithfulness, AOPC, McNemar, Wilcoxon, Cohen's dz), explanation figures, and all notebook helpers. **Only `xai.py` imports `captum` and `shap`** — the other two modules run fine without them.

This flat structure means the codebase can be opened, navigated, and edited without any module indirection.

---

## Setup

### 1. Python environment

Developed and tested against Python 3.11 in a conda env named `ceng709`. From a fresh env:

```bash
conda create -n ceng709 python=3.11
conda activate ceng709
pip install -r requirements.txt
```

Dependencies (pinned in [requirements.txt](requirements.txt)):

- `torch==2.9.1`, `torchvision==0.24.1` — backbone + data loading
- `captum==0.7.0` — Grad-CAM via `LayerGradCam`
- `shap==0.47.2` — SHAP via `DeepExplainer`
- `scipy>=1.11` — Wilcoxon signed-rank and paired t-tests for the continuous XAI analysis
- `scikit-learn==1.6.1` — confusion matrix, macro/weighted metrics, QWK
- `pandas==2.2.3`, `numpy==2.1.3`, `matplotlib==3.10.0`, `Pillow==11.1.0`, `PyYAML==6.0.2`

### 2. Dataset

See [dataset/README.md](dataset/README.md) for the dataset used and expected local folder layout.

Download APTOS 2019 from Kaggle and place it under `dataset/aptos2019/` with paths matching [configs/base.yaml](configs/base.yaml):

```
dataset/aptos2019/
├── train_1.csv              # training labels
├── valid.csv                # validation labels
├── test.csv                 # test labels
├── train_images/            # training image PNGs
├── val_images/              # validation image PNGs
└── test_images/             # test image PNGs
```

### 3. Device

The pipeline auto-detects CUDA, MPS (Apple Silicon), or CPU via `_resolve_device` / `_resolve_xai_device` helpers in [src/data.py](src/data.py). No manual device selection is needed.

**Known device-specific behaviour:**

- On **MPS**, `DataLoader` workers are forced to `num_workers=0` inside inference helpers ([src/train.py:689](src/train.py#L689), [src/train.py:751](src/train.py#L751)) because multi-process DataLoaders with MPS tensors can deadlock.
- On **CUDA**, SHAP DeepExplainer has a CPU fallback path (`_should_retry_shap_on_cpu` in [src/xai.py](src/xai.py)) that triggers automatically on CUDA OOM.
- On **CPU**, everything works but is ~5x slower.

---

## How to run

Open the notebook and execute top-to-bottom:

```bash
jupyter notebook notebooks/project_demo.ipynb
```

The notebook has two setup cells (Colab drive mount + imports) followed by numbered sections 1 through 9.

### Control flags (Section 2 "Run Configuration")

The "Run Configuration" cell is the single control panel. All other cells read from these variables:

| Flag | Default | Purpose |
|---|---|---|
| `SEED` | `1988` | Run seed. The committed checkpoint was trained with this seed. |
| `EVAL_SPLIT` | `'test'` | Which split to evaluate and audit on. |
| `RUN_CLEAN_BEFORE_START` | `False` | If `True`, wipes all artifacts before starting. **Leave False** unless you want a full rebuild. |
| `FORCE_RETRAIN` | `False` | If `False`, the training cell reuses an existing checkpoint whose config signature matches. **Leave False** to avoid a 58-minute retrain. |
| `XAI_SAFE_MODE` | `False` | If `True`, the XAI audit uses smaller SHAP background / sample sizes and forces CPU execution. Flip to `True` only if you hit MPS/CUDA memory pressure. |
| `SHOW_ADVANCED_XAI_AUDIT` | `False` | If `True`, the audit displays extra tables (correctness split, class-wise pass rate, discordant cases). |
| `XAI_VIS_SAFE_MODE` | `False` | Controls the visual review grids. Flip on for smaller memory budget. |
| `XAI_SINGLE_SAFE_MODE` | `False` | Controls the single-case demo. |

### What each section does

| Section | What it does | Runtime with reuse | What it produces |
|---|---|---|---|
| 1. Environment Setup | adds project root to `sys.path` | <1s | — |
| 2. Run Configuration | sets the control flags above | <1s | — |
| 3. Optional Output Reset | wipes artifacts if `RUN_CLEAN_BEFORE_START=True` | <1s | — |
| 4. Data Preparation | generates / reuses train/val/test manifests | ~5–10s | [artifacts/manifests/](artifacts/manifests/) |
| 5. Model Fine-Tuning | reuses checkpoint if config signature matches; else trains from scratch | ~5s reused / ~60min fresh | [artifacts/checkpoints/](artifacts/checkpoints/) + calibration JSON |
| 6. Core Evaluation | inference on the eval split, confusion matrix, per-class metrics, headline table, calibration | ~1–2min on MPS | [artifacts/predictions/](artifacts/predictions/), [artifacts/reports/tables/](artifacts/reports/tables/) |
| **7. Explainability Analysis** | **Grad-CAM + SHAP audit at N=120 (24 per class)** | **~2 hours on MPS** | `rq_xai_method_stats`, `rq_xai_pairwise`, `rq_xai_continuous`, `rq1_gradcam`, `rq2_shap`, per-sample Grad-CAM / SHAP overlay PNGs |
| 7. Summary display | reads the audit CSVs and renders a descriptive pass-rate table + bar chart + continuous-analysis table | <5s | — |
| 7. Advanced audit (gated) | correctness split, class-wise pass rate, discordant cases | <5s | — |
| 8. Visual Review | Grad-CAM + SHAP demo grids across target classes | ~3–5min | `gradcam_demo_grid.png`, `shap_demo_grid.png` |
| 9. Single-Case Demo | detailed XAI panel for one fundus image | ~30s | `artifacts/reports/figures/single/` |

### Reuse semantics

- **Manifests** are regenerated deterministically from the stratification seed every time. Fast but idempotent.
- **Checkpoint reuse** is driven by `_checkpoint_config_signature(cfg)` in [src/train.py](src/train.py). If the current config hashes to the same signature as an existing checkpoint for the same seed, that checkpoint is reused and training returns in seconds. Change any training hyperparameter (loss, optimizer, epochs, lr, batch size, etc.) and the signature changes, triggering a fresh run.
- **Predictions CSVs** are regenerated on every evaluation run to stay consistent with whatever checkpoint is loaded.
- **XAI tables and figures** are always regenerated by Section 7. The pipeline does not auto-wipe the `gradcam/`, `shap/`, or `single/` figure directories between runs — manually clean them if you change `max_targets` or `seed` to avoid stale per-sample PNGs from a previous configuration.

### If Section 7 fails with out-of-memory

1. In the run configuration cell, set `XAI_SAFE_MODE = True`. This forces the audit onto CPU and caps SHAP at 24 background / 24 samples.
2. Re-run from the run configuration cell. All upstream work (manifests, training, core evaluation) is cached.
3. If it still fails, lower `shap_background_size` and `shap_max_samples` in [configs/base.yaml](configs/base.yaml). Current defaults: `shap_max_samples: 120`, `shap_background_size: 16`, `max_targets: 120`, `attribution_mask_radius_ratio: 0.50`.

The pipeline has an automatic CPU fallback (`_should_retry_shap_on_cpu` in [src/xai.py](src/xai.py)) that catches CUDA OOM, MPS OOM, and SHAP in-place-view errors and retries the affected sample on CPU — slower but reliable.

---

## Preprocessing pipeline

Every fundus image goes through a four-stage preprocessing pipeline before the model sees it, implemented in [src/data.py:856-890](src/data.py#L856-L890) (`_apply_fundus_preprocessing`):

```
raw fundus  →  Resize 380²  →  CLAHE  →  Ben-Graham  →  Circle crop  →  ImageNet norm  →  model
```

| Stage | What it does | Parameters |
|---|---|---|
| Resize | Standardises spatial resolution for the backbone | `image_size: 380` |
| CLAHE | Contrast-Limited Adaptive Histogram Equalisation on the LAB L-channel. Boosts local contrast so subtle lesions become visible. | `clahe_clip_limit: 2.0`, `clahe_tile_grid: 8×8` |
| Ben-Graham | Subtracts a Gaussian-blurred copy of the image. Removes slowly-varying illumination and emphasises vessels and lesions. Standard APTOS 2019 preprocessing. | `gaussian_sigma: 10.0`, `ben_graham_weight: 4.0`, `ben_graham_bias: 128.0` |
| Circle crop | Masks the fundus disc region, suppressing outer-frame artefacts from the camera aperture. | `circle_crop_ratio: 1.00` |
| ImageNet norm | Standard mean/std normalisation expected by the ImageNet-pretrained backbone. | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

A 5-panel worked example is generated by [tools/export_preprocessing_steps.py](tools/export_preprocessing_steps.py) and embedded in the report as `figure12_preprocessing_example.png`.

---

## Current checked-in configuration

Committed [configs/base.yaml](configs/base.yaml) uses:

- **Data**: `aptos_only`, benchmark protocol, 5 classes (`No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`), image size 380
- **Backbone**: EfficientNet-B4, ImageNet-pretrained
- **Training**: AdamW, lr=7e-5, 15 epochs, batch size 32, weight decay 1e-4, label smoothing 0.03, early stopping patience 3, cosine annealing scheduler
- **Loss**: focal loss with γ=2.0 and inverse-frequency class weights
- **Calibration**: temperature scaling fitted post-hoc on the validation split
- **XAI audit**: `max_targets: 120` (24 per class, class-balanced), `shap_max_samples: 120`, `shap_background_size: 16` (class-stratified random sample, seeded with `stratify_seed=1988`), `attribution_mask_radius_ratio: 0.50` (retinal-disc mask applied to both Grad-CAM and SHAP attribution maps before metrics; corrects Grad-CAM's bilinear-upsample corner artefact, no effect on SHAP), Grad-CAM evaluated at layers 2 / 3 / 4 with joint-criterion selection (`aopc × (1 − border_ratio)`)
- **Statistics**: Wilcoxon signed-rank + paired t-test + Cohen's dz + McNemar exact test, 1000 bootstrap iterations for CI
- **Figure DPI**: 600

## Current checked-in run

- **Seed**: 1988
- **Run ID**: `dr_efficientnet_b4_aptos2019_85-15-v10_seed1988_20260316T0607_9eee8e`
- **Training duration**: 58.40 minutes (one-time)
- **Best validation macro-F1**: 0.6801
- **Test accuracy**: 0.8218
- **Test macro precision**: 0.7104
- **Test macro recall**: 0.7408
- **Test macro F1**: 0.7105
- **Test QWK**: 0.8958
- **Calibration temperature**: 0.7435
- **ECE before / after calibration**: 0.0790 → 0.0440
- **XAI audit N**: 120 (24 per class, all 5 grades covered)
- **XAI audit device**: MPS (no CPU fallback triggered)

Detailed per-class and confusion tables are under [artifacts/reports/tables/](artifacts/reports/tables/).

---

## XAI audit artifacts

After Section 7 completes, the following CSVs land in [artifacts/reports/tables/](artifacts/reports/tables/):

| File | What it contains |
|---|---|
| `rq_xai_method_stats_seed1988_test.csv` | Descriptive pass rate per method (N, pass rate, 95% CI, mean border, mean faithfulness, AOPC) |
| `rq_xai_pairwise_seed1988_test.csv` | McNemar contingency (n00, n01, n10, n11), chi² p, exact p for the paired pass/fail comparison |
| **`rq_xai_continuous_seed1988_test.csv`** | **Primary research-standard analysis: paired Wilcoxon signed-rank + paired t-test + Cohen's dz on raw per-sample scores. One row per metric (border_ratio, retina_ratio, faith_delta_k10/20/30, aopc_delta).** |
| `rq_xai_pass_by_class_seed1988_test.csv` | Per-class pass-rate breakdown |
| `rq_xai_pass_by_correctness_seed1988_test.csv` | Pass rate split by correct vs wrong predictions |
| `rq1_gradcam_seed1988_test.csv` | Per-sample Grad-CAM scores (selected layer × 120 targets; `border_ratio`/`retina_ratio` are masked, `border_ratio_raw`/`retina_ratio_raw` are pre-mask — see `rq_xai_mask_ablation_*.csv`) |
| `rq2_shap_seed1988_test.csv` | Per-sample SHAP scores (120 rows; raw and masked columns are bit-identical since DeepExplainer attributes zero to zero-valued corner pixels) |
| `xai_targets_seed1988_test.csv` | The 120 audit targets with sample_id, true class, predicted class, confidence, image path |
| `xai_target_coverage_seed1988_test.csv` | Per-target flag for gradcam_done / shap_done (used for consistency checking) |
| `gradcam_layer_selection_seed1988_test.csv` | Per-layer composite_score = mean_aopc_delta × (1 − mean_border_ratio), alongside the individual means and row count. Winner is selected by the composite. |
| **`rq_xai_mask_ablation_seed1988_test.csv`** | **Raw-vs-masked ablation on border and retina ratios. Demonstrates the retinal-disc mask only affects Grad-CAM (symmetric correction of a bilinear-upsample artefact).** |
| `rq_xai_per_class_seed1988_test.csv` | Per-class pass rate + means, showing SHAP dominates No_DR/Severe while Grad-CAM dominates Mild/Moderate/Proliferative. |

The **primary** XAI analysis is the continuous-score comparison in `rq_xai_continuous_seed1988_test.csv`. This follows standard XAI benchmarking practice (Samek et al. 2017 AOPC, Lundberg & Lee 2017 SHAP) and avoids arbitrary thresholds. The descriptive pass rate in `rq_xai_method_stats_seed1988_test.csv` uses a project-specific operational rule (border ≤ 0.25 AND Δ_k20 > 0.10) and is reported as a companion, not as the primary finding.

---

## Reproducibility

This project aims for bit-for-bit determinism where possible. The guarantees are:

- **Data splits**: stratified with `stratify_seed: 1988`. Deterministic manifests across runs.
- **Model weights**: training uses `torch.manual_seed`, `numpy.random.seed`, `random.seed` set in [src/data.py](src/data.py) `_set_seed`. Given the same seed and config, trained weights are identical modulo non-deterministic backend kernels (MPS, CuDNN).
- **Temperature calibration**: LBFGS optimisation on validation logits. Deterministic given the validation predictions.
- **XAI audit target selection**: deterministic. The same 120 targets are selected every time for a given seed and predictions CSV.
- **SHAP background sampling**: seeded with `random_state=int(seed)` across all three SHAP call sites ([src/xai.py:1600](src/xai.py#L1600), [src/xai.py:2116](src/xai.py#L2116), [src/xai.py:2648](src/xai.py#L2648)). Same seed, same background images across runs.
- **Bootstrap CI**: seeded with `stats_bootstrap_seed: 1988`.

**What's not deterministic:** backend-level floating-point non-determinism in convolutions on MPS and CuDNN can cause small numerical differences (~1e-4 level) in the forward pass and in gradient-based attribution maps. These do not affect the direction of any audit finding but can shift a sample near a faithfulness threshold between pass and fail. If you need bit-for-bit identical XAI outputs, run on CPU.

---

## Exporting figures

After a successful run:

```bash
python tools/export_project_demo_assets.py --output-dir assets --zip-path assets.zip
```

This writes `figure1.png` through `figure10.png` to `assets/` (plus a `figure_manifest.csv` that maps each one to its title and source). Note: the tool uses `figure1` / `figure10` naming without leading zeros.

Separately, generate the preprocessing worked-example figure:

```bash
python tools/export_preprocessing_steps.py
```

This writes `figure12_preprocessing_example.png`.

---

## Output folders

| Path | Contents |
|---|---|
| [artifacts/manifests/](artifacts/manifests/) | Frozen train/val/test splits (per-seed CSVs) |
| [artifacts/checkpoints/](artifacts/checkpoints/) | Trained `.pt` and calibration `.json` keyed by run_id |
| [artifacts/predictions/](artifacts/predictions/) | Per-split prediction CSVs with logits, probabilities, and labels |
| [artifacts/logs/](artifacts/logs/) | Run records, training history, XAI runtime status logs (`*_gradcam_status.json`, `*_shap_status.json`) |
| [artifacts/reports/tables/](artifacts/reports/tables/) | Evaluation and XAI audit CSVs (see XAI audit artifacts table above) |
| [artifacts/reports/figures/gradcam/](artifacts/reports/figures/) | One Grad-CAM overlay PNG per (target, layer) — 360 files for N=120 × 3 layers |
| [artifacts/reports/figures/shap/](artifacts/reports/figures/) | One SHAP overlay PNG per target — 120 files for N=120 |
| [artifacts/reports/figures/single/](artifacts/reports/figures/) | Single-case demo outputs |
| `assets/` | Numbered figures for the report (produced by the export tool) |

---

## Development notes

- **Changing hyperparameters**: edit [configs/base.yaml](configs/base.yaml). Any change that affects the checkpoint signature (`training` section) will trigger a fresh training run next time the training cell executes.
- **Running without captum/shap**: `data.py` and `train.py` have no dependency on explainability libraries. You can run sections 1–6 (setup through core evaluation) in an environment that has only the core ML stack. Only Section 7 requires captum and shap.
- **Adding a new seed**: add it to `project.seed_list` in [configs/base.yaml](configs/base.yaml), set `SEED` in the run configuration cell, and re-run. The manifest stratification honours `stratify_seed` separately from the training seed.
- **Multi-seed benchmarking**: `run_benchmark_experiments` in [src/train.py](src/train.py) loops over `seed_list` and produces a cross-seed scoreboard via `export_benchmark_scoreboard`.
- **Smaller / faster audit for development**: in [configs/base.yaml](configs/base.yaml), set `max_targets` and `shap_max_samples` to a small value like 20 or 40.
- **Cross-dataset evaluation** (future work): the pipeline supports a Roboflow source via `data.source: roboflow` in the config, but it has not been exercised for the current committed run.
