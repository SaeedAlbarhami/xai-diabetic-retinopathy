This repository contains an end-to-end pipeline for five-class diabetic retinopathy (DR) severity grading on the APTOS 2019 dataset, with post-hoc explainability via Grad-CAM and SHAP.

- Primary entry point: [notebooks/project_demo.ipynb](notebooks/project_demo.ipynb)
- Configuration source of truth: [configs/base.yaml](configs/base.yaml)
- Final Report PDF: [src/report/ITPG_708_FinalReport.pdf](src/report/ITPG_708_FinalReport.pdf)
- Repository: `https://github.com/SaeedAlbarhami/xai-diabetic-retinopathy.git`
---

## Project Summary

The pipeline fine-tunes an ImageNet-pretrained EfficientNet-B4 on APTOS 2019 and then audits two explainers:

- **Grad-CAM** for spatial heatmaps over retinal regions
- **SHAP** (DeepExplainer) for pixel-level attributions

Both methods are evaluated after applying the same retinal-disc attribution mask, which keeps the comparison symmetric.

The current committed configuration uses:

- APTOS-only benchmark protocol
- image size `380 x 380`
- focal loss with `gamma=2.0`
- AdamW with `lr=7e-5`, `weight_decay=1e-4`, `epochs=15`, `batch_size=32`
- post-hoc temperature scaling on the validation split
- XAI audit settings `max_targets=120`, `shap_max_samples=120`, `shap_background_size=16`, `attribution_mask_radius_ratio=0.50`

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/SaeedAlbarhami/xai-diabetic-retinopathy.git
cd xai-diabetic-retinopathy
```

### 2. Create the verified Python environment

Developed and tested against **Python 3.11** in a conda environment named `itpg708`.

```bash
conda create -n itpg708 python=3.11
conda activate itpg708
pip install -r requirements.txt
```

### 3. Verify the core stack

```bash
python - <<'EOF'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())
EOF
```

Pinned dependencies live in [requirements.txt](requirements.txt). Core versions used by the committed run include:

- `torch==2.9.1`
- `torchvision==0.24.1`
- `captum==0.7.0`
- `shap==0.47.2`
- `scikit-learn==1.6.1`
- `pandas==2.2.3`
- `numpy==2.1.3`
- `matplotlib==3.10.0`
- `Pillow==11.1.0`
- `PyYAML==6.0.2`

### 4. Open the notebook

```bash
jupyter notebook notebooks/project_demo.ipynb
```

The notebook is the single end-to-end entry point for data preparation, training, evaluation, XAI audit, visual review, and the single-case demo.

---

## Hardware and Runtime Notes

The committed run was produced on the following machine:

| Component | Specification |
|---|---|
| Machine | MacBook Pro (Mac15,8) |
| Chip | Apple M3 Max |
| CPU cores | 16 (12 performance + 4 efficiency) |
| GPU | Apple M3 Max integrated GPU via PyTorch MPS |
| Unified memory | 48 GB |
| OS | macOS 26.4.1 (Build 25E253) |

Device selection is automatic in both training and XAI (`training.device: auto`, `xai.device: auto` in [configs/base.yaml](configs/base.yaml)). MPS, CUDA, and CPU backends are all supported; CPU is much slower.

---

## Dataset and Preprocessing

### Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset from Kaggle:

- Source: <https://www.kaggle.com/c/aptos2019-blindness-detection>
- Committed data source: `aptos_only`
- Label order: `No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`

Expected layout:

```text
dataset/
└── aptos2019/
    ├── train_1.csv
    ├── valid.csv
    ├── test.csv
    ├── train_images/
    ├── val_images/
    └── test_images/
```

The repo may include lightweight CSV metadata, but the large image folders are not committed. Generated outputs under `artifacts/` are also ignored by git.

### Preprocessing

Every fundus image goes through the same deterministic preprocessing pipeline implemented in `src/data.py`:

1. Resize to `380 x 380`
2. CLAHE on the LAB L-channel
3. Ben-Graham background subtraction
4. Circle crop
5. ImageNet normalization

Key parameters from [configs/base.yaml](configs/base.yaml):

| Parameter | Value |
|---|---|
| `image_size` | `380` |
| `circle_crop_ratio` | `1.00` |
| `clahe_clip_limit` | `2.0` |
| `clahe_tile_grid` | `8` |
| `gaussian_sigma` | `10.0` |
| `ben_graham_weight` | `4.0` |
| `ben_graham_bias` | `128.0` |
| `norm_mean` | `[0.485, 0.456, 0.406]` |
| `norm_std` | `[0.229, 0.224, 0.225]` |

The report-ready preprocessing example is stored in `src/report/assets/`.

---

## Notebook Workflow

The notebook is organized into Sections 1 through 9. The key behavior is:

| Section | Purpose | Main outputs |
|---|---|---|
| 1. Audit Setup and Configuration | Adds the project root to `sys.path` and sets seed, split, config, checkpoint reuse, and safe-mode controls | runtime context and config |
| 2. Dataset and Preprocessing Pipeline | Builds or refreshes manifests and dataset/preprocessing overview tables/figures | `artifacts/manifests/` and report figures |
| 3. Classification Model Training | Reuses a compatible checkpoint or trains a new one | `artifacts/checkpoints/` |
| 4. Classification Model Evaluation | Runs inference and evaluation on `EVAL_SPLIT` | `artifacts/predictions/`, metrics tables, confusion matrix, calibration outputs |
| 5. XAI Methodology | Runs the paired Grad-CAM / SHAP audit and applies mask-aware attribution correction | XAI CSVs and overlays |
| 6. XAI Metrics and Statistical Analysis | Loads paired comparison, mask-ablation, pass-rate, and optional descriptive audit tables | `artifacts/reports/tables/` |
| 7. Report Tables and Figures Traceability | Renders and records the report visual review grids | demo PNGs |
| 8. Qualitative Case Study | Builds one detailed case report on the test split | `artifacts/reports/figures/single/` |
| 9. Audit Checklist | Lists the source parsing, import, notebook JSON, and CSV sanity checks used before audit | verification commands |

### Section 2 control flags

These notebook flags are the main user-facing controls:

| Flag | Default | Purpose |
|---|---|---|
| `SEED` | `1988` | Reproducibility seed |
| `EVAL_SPLIT` | `'test'` | Split used by Sections 6 and 7 |
| `RUN_CLEAN_BEFORE_START` | `False` | Deletes generated outputs before the run |
| `FORCE_RETRAIN` | `False` | Forces a fresh training run instead of checkpoint reuse |

### Reuse semantics

- **Manifests** are rebuilt deterministically from the same split seed.
- **Checkpoints** are reused when the current training configuration matches the stored checkpoint signature.
- **Predictions** are rebuilt on each evaluation run.
- **XAI tables and figures** are refreshed in Section 7, but stale per-sample images can remain if you change the seed or target count without cleaning the output folders first.

---

## Repository Layout

```text
xai-diabetic-retinopathy/
├── configs/
│   └── base.yaml
├── notebooks/
│   └── project_demo.ipynb
├── src/
│   ├── data.py
│   ├── train.py
│   ├── xai.py
│   ├── xai_common.py
│   ├── xai_metrics.py
│   ├── xai_stats.py
│   ├── xai_viz.py
│   ├── xai_gradcam.py
│   ├── xai_shap.py
│   ├── xai_audit.py
│   ├── xai_single.py
│   ├── xai_notebook.py
│   └── report/
├── dataset/
├── artifacts/
└── requirements.txt
```

Module ownership is intentionally split:

- `src/data.py`: config loading, manifests, preprocessing, dataset classes
- `src/train.py`: model training, checkpointing, evaluation, calibration
- `src/xai.py`: public facade used by the notebook and scripts
- `src/xai_*.py`: implementation modules for visualization, statistics, metrics, Grad-CAM, SHAP, audit orchestration, notebook adapters, and single-case reporting

## Outputs and Key Artifacts

Generated outputs land under `artifacts/`:

| Path | Contents |
|---|---|
| `artifacts/manifests/` | train/val/test manifests |
| `artifacts/checkpoints/` | `.pt` checkpoints and calibration sidecars |
| `artifacts/predictions/` | inference CSVs |
| `artifacts/logs/` | run metadata, training history, XAI status |
| `artifacts/reports/tables/` | evaluation and XAI audit CSVs |
| `artifacts/reports/figures/` | Grad-CAM, SHAP, and single-case PNGs |

Most important evaluation and XAI tables from the committed run:

- `final_headline_metrics_seed1988_test_aptos2019_85-15-v10.csv`
- `per_class_seed1988_test_aptos2019_85-15-v10.csv`
- `run_summary_seed1988_test_aptos2019_85-15-v10.csv`
- `rq_xai_continuous_seed1988_test.csv`
- `rq_xai_method_stats_seed1988_test.csv`
- `rq_xai_pairwise_seed1988_test.csv`
- `rq_xai_pass_by_class_seed1988_test.csv`
- `rq_xai_pass_by_correctness_seed1988_test.csv`
- `rq_xai_mask_ablation_seed1988_test.csv`
- `rq1_gradcam_seed1988_test.csv`
- `rq2_shap_seed1988_test.csv`
- `xai_targets_seed1988_test.csv`
- `gradcam_layer_selection_seed1988_test.csv`

Committed headline numbers:

| Metric | Value |
|---|---|
| Accuracy | `0.8218` |
| Precision (macro) | `0.7104` |
| Recall (macro) | `0.7408` |
| F1 (macro) | `0.7105` |
| QWK | `0.8958` |

Committed XAI method summary on the 120-target audit:

| Method | Pass rate | Mean border ratio | Mean retina ratio | Mean `Delta_{k=20}` | Mean AOPC |
|---|---|---|---|---|---|
| Grad-CAM | `0.5583` | `0.2064` | `0.8164` | `0.3275` | `0.2872` |
| SHAP | `0.3917` | `0.2342` | `0.7910` | `0.1868` | `0.1635` |

The primary method-comparison table is `rq_xai_continuous_seed1988_test.csv`. The thresholded pass-rate summary is descriptive and secondary.

---

## Reproducibility Notes

The repo is configured for deterministic reruns wherever the backend allows it.

- Split seed: `1988`
- Benchmark split protocol: 85/15 outer split with validation carved from training
- XAI target balancing: enabled by true class (`target_balance_class_col: true_class`)
- SHAP background sampling: deterministic, size `16`
- Bootstrap statistics: `1000` iterations, seed `1988`

What is deterministic:

- manifest generation
- checkpoint selection logic
- audit target selection
- SHAP background sampling
- bootstrap confidence intervals


For the closest reproduction of the committed result, keep:

- `SEED = 1988`
- `FORCE_RETRAIN = False` if the committed checkpoint is already present
- the default XAI settings from [configs/base.yaml](configs/base.yaml)
