"""Filesystem paths used by the Solar Potential app.

Keeping these in one module avoids path arithmetic in individual views and
keeps the app working regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_PATH = DATA_DIR / "Pasion et al dataset.csv"
BATCH_EXAMPLE_PATH = DATA_DIR / "batch_example_solar.csv"

RF_MODEL_PATH = MODELS_DIR / "tuned_rf_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "tuned_xgb_model.joblib"
