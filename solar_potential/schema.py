"""Dataset schema and feature metadata for the Solar Potential app.

This module is the single source of truth for column names, model inputs, and
the data dictionary. It has no Streamlit or modelling dependencies, so the
data, analytics, and view modules can all import it without creating a cycle.

The deployed model consumes six numeric features. ``Location`` and ``Season``
exist in the source data but are deliberately not model inputs: the notebook
shows that ``Season`` is a recode of ``Month`` and that site dummies encode the
same elevation signal as ``Pressure``.
"""

from __future__ import annotations

from typing import Dict, Tuple

TARGET_COLUMN = "PolyPwr"

# Exact order the persisted models expect.
FEATURE_COLUMNS = [
    "AmbientTemp",
    "Pressure",
    "Cloud.Ceiling",
    "Humidity",
    "Month",
    "Hour",
]

NUMERIC_FEATURES = ["AmbientTemp", "Pressure", "Cloud.Ceiling", "Humidity"]

HOURS = [10, 11, 12, 13, 14, 15]

SITE_NAMES = [
    "Camp Murray",
    "Grissom",
    "Hill Weber",
    "JDMT",
    "Kahului",
    "MNANG",
    "Malmstrom",
    "March AFB",
    "Offutt",
    "Peterson",
    "Travis",
    "USAFA",
]

# Mirrors the source dataset exactly: Season is a deterministic recode of Month.
MONTH_SEASON = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
    12: "Winter",
}

DISPLAY_NAMES = {
    "AmbientTemp": "Ambient temperature (°C)",
    "Pressure": "Station pressure (hPa)",
    "Cloud.Ceiling": "Cloud ceiling (m, 722 = clear)",
    "Humidity": "Humidity (%)",
    "Month": "Month",
    "Hour": "Hour of day",
}

# (minimum, maximum, step) used by the number inputs, taken from the training range.
INPUT_BOUNDS: Dict[str, Tuple[float, float, float]] = {
    "AmbientTemp": (-20.0, 66.0, 0.5),
    "Pressure": (780.0, 1030.0, 1.0),
    "Cloud.Ceiling": (0.0, 722.0, 10.0),
    "Humidity": (0.0, 100.0, 1.0),
    "Month": (1.0, 12.0, 1.0),
    "Hour": (10.0, 15.0, 1.0),
}


def get_feature_definitions() -> Dict[str, Dict[str, str]]:
    """Data dictionary rendered by the About tab."""
    return {
        "Location": {
            "description": "Site identifier (12 northern-hemisphere sites)",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Season": {
            "description": "Season label; a recode of Month",
            "data_type": "Categorical",
            "specific_type": "Nominal",
        },
        "Month": {
            "description": "Month of year; controls sun height and day length",
            "data_type": "Numerical",
            "specific_type": "Cyclical",
        },
        "Hour": {
            "description": "Hour of day; controls sun position (recorded 10:00-15:00)",
            "data_type": "Numerical",
            "specific_type": "Cyclical",
        },
        "Humidity": {
            "description": "Relative humidity (%); refraction, diffraction, panel dust",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "AmbientTemp": {
            "description": "Ambient temperature (°C); panel efficiency falls as it rises",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Pressure": {
            "description": "Station pressure (hPa); encodes elevation and weather systems",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Cloud.Ceiling": {
            "description": "Cloud ceiling height (m); 722 is the recorder's clear-sky cap",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Altitude": {
            "description": "Site elevation; dropped for collinearity with Pressure",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Wind.Speed": {
            "description": "Wind speed; dropped for weak signal",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        "Visibility": {
            "description": "Visibility distance; dropped for weak signal",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
        TARGET_COLUMN: {
            "description": "Panel power output (Watts) — the target variable",
            "data_type": "Numerical",
            "specific_type": "Continuous",
        },
    }
