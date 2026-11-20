"""Model and dataset analytics for the Solar Potential app.

Pure functions with no Streamlit dependency so they stay testable and reusable
by both the views and any script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import numpy as np
import pandas as pd

from .schema import DISPLAY_NAMES, FEATURE_COLUMNS

ARTIFACT_KEYS = ("model", "feature_columns", "metrics", "label", "target")


def load_artifact(path: Path) -> Dict[str, Any]:
    """Load a notebook bundle, refusing anything that is not one.

    A bare estimator pickled by an older notebook has no feature order and no
    metrics, so scoring with it would silently use the wrong columns. The
    schema check turns that into an actionable message instead.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"{Path(path).name} could not be read ({type(exc).__name__}). "
            "Re-run notebook.ipynb to regenerate it."
        ) from exc

    if not isinstance(bundle, dict):
        raise ValueError(
            f"{Path(path).name} is not a notebook bundle artifact "
            "(loaded object is not a dictionary). "
            "Re-run notebook.ipynb to regenerate it."
        )

    missing = [key for key in ARTIFACT_KEYS if key not in bundle]
    if missing:
        raise ValueError(
            f"{Path(path).name} is not a notebook bundle artifact "
            f"(missing: {', '.join(missing)}). "
            "Re-run notebook.ipynb to regenerate it."
        )

    features = list(bundle["feature_columns"])
    if features != FEATURE_COLUMNS:
        raise ValueError(
            f"{Path(path).name} was trained on features {features}, but this app "
            f"scores {FEATURE_COLUMNS}. Re-run notebook.ipynb to regenerate it."
        )

    return bundle


def predict_power(artifact: Mapping[str, Any], frame: pd.DataFrame) -> float:
    """Predict panel output in Watts for one row."""
    model_input = frame[list(artifact["feature_columns"])]
    return float(artifact["model"].predict(model_input)[0])


def contributions(
    artifact: Mapping[str, Any],
    row: pd.Series,
    baseline: pd.Series,
) -> pd.DataFrame:
    """Attribute the gap between this prediction and a reference prediction.

    Walks the features in the model's own order: start from the baseline row,
    set one feature to this row's value, and record how the prediction moved.
    The deltas therefore sum exactly to ``predict(row) - predict(baseline)``,
    which is the question a reader actually asks — "why is this score above or
    below typical?".

    The decomposition is order-dependent: correlated features share credit
    depending on which is applied first. That is inherent to any additive
    attribution of a non-additive model, and it is why the caption says so.
    """
    features = list(artifact["feature_columns"])
    current = baseline.copy()
    previous = predict_power(artifact, current.to_frame().T)

    records = []
    for feature in features:
        current[feature] = row[feature]
        updated = predict_power(artifact, current.to_frame().T)
        records.append(
            {
                "feature": feature,
                "display": DISPLAY_NAMES.get(feature, feature),
                "input": float(row[feature]),
                "baseline": float(baseline[feature]),
                "delta": updated - previous,
            }
        )
        previous = updated

    table = pd.DataFrame(records)
    order = table["delta"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def _metric(metrics: Mapping[str, Any], group: str, name: str) -> float:
    """Read one stored metric, tolerating artifacts that predate a metric."""
    try:
        if group == "cv":
            return float(metrics["cv"][name]["test_mean"])
        return float(metrics["test"][name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def model_comparison(artifacts: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Side-by-side CV and held-out metrics for every loaded bundle.

    Values are returned as floats; the view is responsible for formatting.
    """
    rows = []
    for name, artifact in artifacts.items():
        metrics = artifact.get("metrics", {})
        rows.append(
            {
                "Model": name,
                "CV R2": _metric(metrics, "cv", "r2"),
                "CV MAE": _metric(metrics, "cv", "mae"),
                "CV RMSE": _metric(metrics, "cv", "rmse"),
                "Test R2": _metric(metrics, "test", "r2"),
            }
        )

    return pd.DataFrame(rows)


def feature_importance(artifact: Mapping[str, Any]) -> pd.DataFrame:
    """Impurity-based importance for a tree bundle, ready for charting."""
    model = artifact["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    features = list(artifact["feature_columns"])
    return (
        pd.DataFrame(
            {"feature": features, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def extrapolation_flags(row: pd.Series, bounds: Mapping[str, tuple]) -> list[str]:
    """Name the features whose value sits outside the value used for training."""
    flagged = []
    for feature, (low, high, _step) in bounds.items():
        if feature in row and not (low <= float(row[feature]) <= high):
            flagged.append(feature)
    return flagged


def target_band(value: float) -> str:
    """Coarse label for a predicted output, used in result summaries."""
    if value >= 20.0:
        return "High"
    if value >= 10.0:
        return "Moderate"
    return "Low"


__all__ = [
    "ARTIFACT_KEYS",
    "FEATURE_COLUMNS",
    "contributions",
    "extrapolation_flags",
    "feature_importance",
    "load_artifact",
    "model_comparison",
    "predict_power",
    "target_band",
]
