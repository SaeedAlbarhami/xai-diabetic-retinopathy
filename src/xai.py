"""XAI audit module.

Explainability pipeline for the APTOS 2019 DR grading project: Grad-CAM and
SHAP DeepExplainer generation, per-sample metrics (border ratio, retina ratio,
faithfulness deltas, AOPC), the retinal-disc attribution-mask correction, and
all aggregate tables used by the report (method stats, pairwise McNemar,
continuous paired tests, per-class breakdown).

Main entry points used by the notebook:
    notebook_run_xai()                    -- full audit on the class-balanced subset
    notebook_load_xai_committee_summary() -- loads result tables for display
    notebook_run_single_case_report()     -- single-image Grad-CAM + SHAP demo
"""
from __future__ import annotations

import copy
import json
import os
import random
import shutil
import types
import warnings
import hashlib
import math
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import EfficientNet_B4_Weights, ResNet50_Weights, ViT_B_16_Weights, efficientnet_b4, resnet50, vit_b_16
from torchvision.models.efficientnet import FusedMBConv, MBConv
from torchvision.models.resnet import BasicBlock, Bottleneck



import yaml


# -----------------------------
# Config + common utils
# -----------------------------


try:
    from captum.attr import LayerAttribution, LayerGradCam
except Exception as exc:  # pragma: no cover
    LayerAttribution = None
    LayerGradCam = None
    _CAPTUM_IMPORT_ERROR = exc
else:
    _CAPTUM_IMPORT_ERROR = None

try:
    import shap
except Exception as exc:  # pragma: no cover
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None

from src.data import (  # noqa: F401
    _REQUIRED_PATH_KEYS,
    _infer_project_root,
    load_project_config,
    _cfg,
    _save_json,
    _load_json,
    _write_alias_copy,
    _set_seed,
    _resolve_device,
    _infer_laterality,
    _class_name_from_id,
    _normalize_label_token,
    _parse_aptos_csv,
    _parse_roboflow_classes_csv,
    _load_aptos_full_pool,
    _load_roboflow_full_pool,
    _data_source,
    _load_dataset_pool,
    _split_train_val_test,
    _split_train_val_only,
    _data_protocol,
    _is_benchmark,
    _use_legacy_aliases,
    _slug_token,
    _profile_dataset_tag,
    _profile_split_tag,
    _profile_profile_tag,
    _manifest_suffix,
    _manifest_filename_map,
    _manifest_outputs,
    _split_two_stage_stratified_pool,
    freeze_current_test_manifest,
    prepare_data_manifests,
    _build_transform,
    _apply_fundus_preprocessing,
    _FundusDataset,
    _load_image_for_inference,
    _manifest_path,
    _backbone_name,
    _model_image_size,
    _to_ratio_fraction,
    _split_policy_tag,
    _table_path,
    notebook_prepare_data_overview,
)
from src.train import (  # noqa: F401
    DRClassifier,
    _LogitWrapper,
    _classification_metrics,
    _per_class_metrics_from_confusion,
    _build_reliability_table,
    _expected_calibration_error,
    _latest_run_record_path,
    _legacy_checkpoint_alias_path,
    _legacy_calibration_alias_path,
    _legacy_predictions_alias_path,
    _legacy_train_history_alias_path,
    _legacy_gradcam_status_alias_path,
    _legacy_shap_status_alias_path,
    _legacy_run_log_alias_path,
    _new_run_id,
    _run_id_from_checkpoint_path,
    _checkpoint_path_for_run_id,
    _calibration_path_for_run_id,
    _predictions_path_for_run_id,
    _train_history_log_path,
    _gradcam_status_log_path,
    _shap_status_log_path,
    _workflow_run_log_path,
    _save_latest_run_record,
    _load_latest_run_record,
    _latest_named_checkpoint,
    _resolve_checkpoint_and_run_id,
    _find_matching_checkpoint_by_signature,
    _checkpoint_path,
    _calibration_path,
    _predictions_path,
    _run_id_from_predictions_path,
    _checkpoint_config_signature,
    _class_weights,
    _ordinal_ce_loss,
    _focal_loss,
    _predict_manifest,
    _collect_logits_and_labels,
    _fit_temperature,
    _load_model,
    train_dr_classifier,
    build_validation_calibration_table,
    run_split_inference,
    evaluate_pipeline_outputs,
    export_final_headline_metrics,
    _ci95_summary,
    _profile_seed_list,
    export_benchmark_scoreboard,
    run_complete_workflow,
    run_benchmark_experiments,
    notebook_run_training,
    notebook_run_core_evaluation,
    clean_generated_outputs,
)

# Re-export leaf helpers from sibling modules so ``src.xai.<name>`` continues
# to resolve for every name that resolved before the refactor. These siblings
# are pure leaves (no imports from src.xai), so no cycle risk.
from src.xai_stats import (  # noqa: F401
    _bootstrap_pass_rate_ci,
    _xai_pass_flag,
    _mcnemar_exact_pvalue,
    _mcnemar_chi2_approx,
    _build_xai_continuous_stats,
    _build_xai_pairwise_stats,
)
from src.xai_viz import (  # noqa: F401
    _normalize_map,
    _overlay,
    _save_overlay_image,
    _save_map_overlay,
    _plot_attribution_grid,
)
from src.xai_metrics import (  # noqa: F401
    _border_mask,
    _retina_circle_mask,
    _attribution_retina_mask,
    _attribution_mass_ratios,
    _mask_by_score_map,
    _faithfulness_delta,
    _parse_faithfulness_k_list,
    _k_to_col_name,
    _faithfulness_multi_k,
)
from src.xai_common import (  # noqa: F401
    _resolve_xai_device,
    _predict_one_with_temperature,
    _temperature_for_run,
)
from src.xai_gradcam import (  # noqa: F401
    _infer_backbone_from_model,
    _resolve_gradcam_target_layer,
    _generate_gradcam,
    _gradcam_heatmap_for_display,
    plot_gradcam_grid,
    plot_gradcam_class_grid,
)
from src.xai_shap import (  # noqa: F401
    _is_shap_inplace_view_error,
    _is_mps_oom_error,
    _should_retry_shap_on_cpu,
    _empty_mps_cache_if_available,
    _bottleneck_forward_shap_safe,
    _basicblock_forward_shap_safe,
    _mbconv_forward_shap_safe,
    _fused_mbconv_forward_shap_safe,
    _make_shap_compatible,
    _build_shap_explainer_with_known_warning_filter,
    _shap_values_with_known_warning_filter,
    _pick_shap_map,
    _shap_to_2d,
    plot_shap_grid,
)
from src.xai_audit import (  # noqa: F401
    _CONTINUOUS_METRIC_DISPLAY,
    _xai_pass_rule_thresholds,
    _build_xai_method_stats_row,
    _build_xai_pass_by_correctness_table,
    _build_xai_pass_by_class_table,
    _format_continuous_table_for_display,
    _format_continuous_bottom_line,
    _choose_xai_targets,
    _parse_gradcam_layers,
    run_xai_analysis,
)
from src.xai_single import (  # noqa: F401
    predict_single_image_with_explanations,
    explain_single_image_detailed,
    run_single_case_demo,
)
from src.xai_notebook import (  # noqa: F401
    notebook_run_xai,
    notebook_load_xai_committee_summary,
    notebook_load_xai_advanced_audit,
    notebook_run_visual_review,
    notebook_run_single_case_report,
)

# Annotate the re-exported display map to restore the ``__annotations__``
# module attribute that existed pre-refactor (backwards-compat parity).
_CONTINUOUS_METRIC_DISPLAY: dict[str, tuple[str, str]]


