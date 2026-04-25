# Dataset

This project uses the **APTOS 2019 Blindness Detection** dataset (Asia Pacific Tele-Ophthalmology Society / Kaggle), for five-class diabetic retinopathy severity grading on the International Clinical Diabetic Retinopathy (ICDR) scale.

**Source.** Kaggle competition: <https://www.kaggle.com/c/aptos2019-blindness-detection>. 

**Classes.** The five ICDR grades used throughout this project are: `0 = No_DR`, `1 = Mild`, `2 = Moderate`, `3 = Severe`, `4 = Proliferate_DR`.

**File format.**

| File | Format | Contents |
|---|---|---|
| `train_1.csv`, `valid.csv`, `test.csv` | CSV, UTF-8 | Two columns: `id_code` (string, matches PNG filename without extension) and `diagnosis` (integer, 0–4 per ICDR grade above). |
| `train_images/*.png`, `val_images/*.png`, `test_images/*.png` | 8-bit RGB PNG | One image per row of the corresponding CSV, named `<id_code>.png`. Native resolutions vary (typically ~3000×2000); the pipeline resizes every image to 380×380 during the preprocessing stage. |

**Preprocessing applied by this project** (implemented in `src/data.py` via `_apply_fundus_preprocessing`, details in the project README's "Preprocessing pipeline" section): resize to 380×380 → CLAHE on the LAB L-channel → Ben-Graham background subtraction → circular crop to the retinal disc → ImageNet mean/std normalisation.

**Layout on disk.** Place the dataset under:

```text
dataset/aptos2019/
├── train_1.csv
├── valid.csv
├── test.csv
├── train_images/
├── val_images/
└── test_images/
```

The current project configuration expects this layout; it is read from `configs/base.yaml` (`paths.dataset_root`, `paths.train_csv`, `paths.val_csv`, `paths.test_csv`).
