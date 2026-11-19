"""Smoke checks for the Solar Potential app.

Plain asserts, no test framework: the project vendors no test dependency and
this script is the gate the implementation plan names. Run with
``python scripts/smoke_check.py``; exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from solar_potential.data import (  # noqa: E402
    initial_data_transform,
    preset_for,
    site_presets,
    to_model_matrix,
)
from solar_potential.schema import (  # noqa: E402
    FEATURE_COLUMNS,
    HOURS,
    MONTH_SEASON,
    SITE_NAMES,
    TARGET_COLUMN,
    get_feature_definitions,
)

# --- schema -----------------------------------------------------------------
assert TARGET_COLUMN == "PolyPwr"
assert FEATURE_COLUMNS == [
    "AmbientTemp",
    "Pressure",
    "Cloud.Ceiling",
    "Humidity",
    "Month",
    "Hour",
]
assert HOURS == [10, 11, 12, 13, 14, 15]
assert len(SITE_NAMES) == 12 and "Camp Murray" in SITE_NAMES
assert MONTH_SEASON[1] == "Winter" and MONTH_SEASON[7] == "Summer"
assert "AmbientTemp" in get_feature_definitions()
print("schema ok")

# --- data -------------------------------------------------------------------
raw = pd.DataFrame(
    {"Date": [20171203, 20180701], "Time": [1145, 900], "YRMODAHRMI": [1.0, 2.0]}
)
parsed = initial_data_transform(raw)
assert parsed["Datetime"].tolist() == [
    pd.Timestamp("2017-12-03 11:45"),
    pd.Timestamp("2018-07-01 09:00"),
]
assert "YRMODAHRMI" not in parsed.columns

dataset = pd.DataFrame(
    {
        "Location": ["A", "A", "B"],
        "Month": [1, 1, 7],
        "Hour": [10, 11, 12],
        "AmbientTemp": [10.0, 20.0, 30.0],
        "Pressure": [900.0, 910.0, 920.0],
        "Cloud.Ceiling": [100.0, 200.0, 722.0],
        "Humidity": [40.0, 50.0, 60.0],
    }
)
presets = site_presets(dataset)

exact = preset_for(presets, dataset, "A", 1, 10)
assert exact["AmbientTemp"] == 10.0
assert list(exact) == FEATURE_COLUMNS
assert exact["Month"] == 1.0 and exact["Hour"] == 10.0

# No rows for March 14: fall back to the site median (10 and 20 -> 15).
fallback_site = preset_for(presets, dataset, "A", 3, 14)
assert fallback_site["AmbientTemp"] == 15.0
assert fallback_site["Month"] == 3.0 and fallback_site["Hour"] == 14.0

# Unknown site: fall back to the global median (10, 20, 30 -> 20).
fallback_global = preset_for(presets, dataset, "Z", 3, 14)
assert fallback_global["AmbientTemp"] == 20.0

matrix = to_model_matrix(pd.DataFrame([exact]))
assert list(matrix.columns) == FEATURE_COLUMNS
assert matrix.shape == (1, 6)

try:
    to_model_matrix(pd.DataFrame({"AmbientTemp": [1.0]}))
except ValueError as exc:
    assert "missing required feature columns" in str(exc).lower()
else:
    raise AssertionError("expected ValueError for missing columns")
print("data ok")

# --- analytics --------------------------------------------------------------
import tempfile  # noqa: E402

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from solar_potential.analytics import (  # noqa: E402
    load_artifact,
    model_comparison,
    predict_power,
    sensitivity,
)

with tempfile.TemporaryDirectory() as tmp:
    stale = Path(tmp) / "stale.joblib"
    joblib.dump(RandomForestRegressor(), stale)
    try:
        load_artifact(stale)
    except ValueError as exc:
        assert "bundle" in str(exc).lower(), exc
    else:
        raise AssertionError("a bare estimator must be rejected")

sample = pd.DataFrame(
    [[25.0, 950.0, 300.0, 40.0, 7, 12]] * 50, columns=FEATURE_COLUMNS
)
fitted = RandomForestRegressor(n_estimators=5, random_state=42).fit(
    sample, [1.0] * 50
)
bundle = {
    "model": fitted,
    "feature_columns": list(FEATURE_COLUMNS),
    "metrics": {"cv": {}, "test": {}},
    "label": "Test Forest",
    "target": TARGET_COLUMN,
}

value = predict_power(bundle, sample.iloc[[0]])
assert isinstance(value, float) and value == value  # finite, not NaN

table = sensitivity(bundle, sample.iloc[0], sample.median())
assert len(table) == 6
assert set(table.columns) == {"feature", "display", "input", "baseline", "delta"}

comparison = model_comparison({"Test Forest": bundle})
assert list(comparison.columns) == ["Model", "CV R2", "CV MAE", "CV RMSE", "Test R2"]
assert comparison.loc[0, "Model"] == "Test Forest"

with tempfile.TemporaryDirectory() as tmp:
    corrupt = Path(tmp) / "corrupt.joblib"
    corrupt.write_bytes(b"not a joblib file")
    try:
        load_artifact(corrupt)
    except ValueError as exc:
        assert "regenerate" in str(exc).lower(), exc
    else:
        raise AssertionError("a corrupt artifact must raise ValueError")

    reordered = dict(bundle)
    reordered["feature_columns"] = list(reversed(FEATURE_COLUMNS))
    mismatched = Path(tmp) / "mismatched.joblib"
    joblib.dump(reordered, mismatched)
    try:
        load_artifact(mismatched)
    except ValueError as exc:
        assert "feature" in str(exc).lower(), exc
    else:
        raise AssertionError("a bundle with a different feature order must be rejected")
print("analytics ok")

# --- batch ------------------------------------------------------------------
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from solar_potential.data import prepare_batch  # noqa: E402

messy = pd.DataFrame(
    {
        "Unnamed: 0": [0, 1],
        "Location": ["A", "B"],
        "AmbientTemp": [25.0, 30.0],
        "Pressure": [950.0, 900.0],
        "Cloud.Ceiling": [300.0, 722.0],
        "Humidity": [40.0, 60.0],
        "Month": [7, 7],
        "Hour": [12, 13],
        "TARGET": [10.0, 20.0],
    }
).rename(columns={"TARGET": TARGET_COLUMN})

clean = prepare_batch(messy)
assert list(clean.columns) == FEATURE_COLUMNS
assert len(clean) == 2
assert clean["AmbientTemp"].tolist() == [25.0, 30.0]
print("batch ok")

# --- end to end -------------------------------------------------------------
from solar_potential.config import RF_MODEL_PATH, XGB_MODEL_PATH  # noqa: E402
from solar_potential.data import load_dataset, site_presets, preset_for  # noqa: E402

dataset = load_dataset()
presets = site_presets(dataset)
assert len(dataset) == 21045

for label, path in [("Random Forest", RF_MODEL_PATH), ("XGBoost", XGB_MODEL_PATH)]:
    artifact = load_artifact(path)
    assert artifact["feature_columns"] == FEATURE_COLUMNS, artifact["feature_columns"]

    preset = preset_for(presets, dataset, "Travis", 7, 12)
    row = pd.Series(preset)
    value = predict_power(artifact, row.to_frame().T)
    assert 0.0 <= value <= 40.0, (label, value)

    table = sensitivity(artifact, row, dataset[list(preset)].median())
    assert len(table) == len(artifact["feature_columns"])

print("end-to-end ok")



