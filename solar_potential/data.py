"""Dataset loading and feature assembly for the Solar Potential app.

Pure pandas functions with no Streamlit dependency. Caching belongs to the
entry point, so these stay cheap to call from tests and scripts.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import DATASET_PATH
from .schema import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN


def initial_data_transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the timestamp and normalise the temporal columns.

    The source stores ``Date`` as ``20171203`` and ``Time`` as ``1145`` (meaning
    11:45). Both are zero-padded and parsed together, then ``Month`` and
    ``Hour`` are re-derived so they can never disagree with the timestamp.

    Frames without both date and time columns are returned with their existing
    ``Month``/``Hour`` untouched, which is what batch uploads need.
    """
    data = raw.copy()

    if {"Date", "Time"}.issubset(data.columns):
        stamps = pd.to_datetime(
            data["Date"].astype(str) + " " + data["Time"].astype(str).str.zfill(4),
            format="%Y%m%d %H%M",
        )
        data["Datetime"] = stamps
        data["Month"] = stamps.dt.month
        data["Hour"] = stamps.dt.hour

    return data.drop(columns=["YRMODAHRMI"], errors="ignore")


def load_dataset() -> pd.DataFrame:
    """Load the raw Kaggle-style CSV and apply the timestamp transform."""
    return initial_data_transform(pd.read_csv(DATASET_PATH))


def site_presets(dataset: pd.DataFrame) -> pd.DataFrame:
    """Median weather per site, month, and hour.

    Used to prefill the prediction form with conditions a site actually saw,
    rather than with a neutral default.
    """
    grouped = dataset.groupby(["Location", "Month", "Hour"], as_index=False)[
        NUMERIC_FEATURES
    ].median()
    return grouped


def preset_for(
    presets: pd.DataFrame,
    dataset: pd.DataFrame,
    site: str,
    month: int,
    hour: int,
) -> Dict[str, float]:
    """Resolve weather for a site/month/hour, with graceful degradation.

    Order: the exact cell, then the site's overall median, then the dataset
    median. The returned mapping always carries all six model features, so the
    caller can hand it straight to the model.
    """
    match = presets[
        (presets["Location"] == site)
        & (presets["Month"] == month)
        & (presets["Hour"] == hour)
    ]

    if not match.empty:
        weather = match.iloc[0][NUMERIC_FEATURES]
    else:
        site_rows = dataset[dataset["Location"] == site]
        source = site_rows if not site_rows.empty else dataset
        weather = source[NUMERIC_FEATURES].median()

    resolved = {feature: float(weather[feature]) for feature in NUMERIC_FEATURES}
    resolved["Month"] = float(month)
    resolved["Hour"] = float(hour)
    return {feature: resolved[feature] for feature in FEATURE_COLUMNS}


def to_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and coerce the six columns the persisted models expect."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("missing required feature columns: " + ", ".join(missing))

    return frame[FEATURE_COLUMNS].astype(float)


def prepare_batch(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn an uploaded CSV into a scoring matrix.

    Exports often carry an index column and may include the target or the
    original composite timestamp; all are tolerated and dropped rather than
    treated as errors.
    """
    data = raw.loc[
        :, [column for column in raw.columns if not str(column).startswith("Unnamed:")]
    ].copy()

    return to_model_matrix(
        data.drop(columns=[TARGET_COLUMN, "YRMODAHRMI"], errors="ignore")
    )
