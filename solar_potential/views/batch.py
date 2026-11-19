"""Batch scoring view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from solar_potential.analytics import target_band
from solar_potential.config import BATCH_EXAMPLE_PATH
from solar_potential.data import initial_data_transform, prepare_batch
from solar_potential.schema import FEATURE_COLUMNS, INPUT_BOUNDS

OUTPUT_COLUMN = "Predicted output (W)"
BAND_COLUMN = "Band"


def _out_of_range(frame: pd.DataFrame) -> dict[str, int]:
    """Count values outside the range the models were trained on."""
    counts: dict[str, int] = {}
    for feature, (low, high, _step) in INPUT_BOUNDS.items():
        if feature not in frame.columns:
            continue
        outside = frame[feature].map(
            lambda value: pd.isna(value) or value < low or value > high
        )
        if int(outside.sum()):
            counts[feature] = int(outside.sum())
    return counts


def _render_results(display: pd.DataFrame, predictions: list[float]) -> None:
    results = display.copy()
    results[OUTPUT_COLUMN] = predictions
    results[BAND_COLUMN] = [target_band(value) for value in predictions]

    with st.container(horizontal=True):
        st.metric("Rows scored", f"{len(results):,}", border=True)
        st.metric("Mean output", f"{sum(predictions) / len(predictions):.2f} W", border=True)
        st.metric("Peak output", f"{max(predictions):.2f} W", border=True)

    st.dataframe(
        results,
        hide_index=True,
        column_config={
            OUTPUT_COLUMN: st.column_config.ProgressColumn(
                "Predicted output",
                format="%.1f W",
                min_value=0,
                max_value=35,
            )
        },
    )

    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="solar_output_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
    )


def render_batch_tab(artifacts: dict, dataset: pd.DataFrame, presets: pd.DataFrame) -> None:
    st.markdown(
        "Upload weather observations and score every row in one pass. "
        "The file needs the six model features; any other columns are ignored."
    )

    source = st.segmented_control(
        "Data source",
        ["Upload CSV", "Example data"],
        default="Upload CSV",
        key="batch_source",
    )

    raw: pd.DataFrame | None = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Weather CSV",
            type="csv",
            help="Same columns as the source dataset. The target is optional and ignored.",
        )
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read the CSV: {exc}")
                return
    elif source == "Example data":
        raw = pd.read_csv(BATCH_EXAMPLE_PATH)
        st.caption("Using the five-row example from data/batch_example_solar.csv.")

    if raw is None:
        with st.expander("Expected columns"):
            st.markdown(
                "- AmbientTemp\n- Pressure\n- Cloud.Ceiling\n- Humidity\n"
                "- Month\n- Hour\n\n"
                "Date, Time, Location, Season, and the target are accepted but not "
                "required by the model."
            )
        return

    if raw.empty:
        st.error("The uploaded file has no rows.")
        return

    try:
        prepared = prepare_batch(initial_data_transform(raw))
    except ValueError as exc:
        st.error(f"Could not score this file: {exc}")
        return
    except Exception as exc:
        st.error(f"Could not prepare the input: {exc}")
        return

    outside = _out_of_range(prepared)
    if outside:
        details = ", ".join(f"{name} ({count} row(s))" for name, count in outside.items())
        st.warning(
            "Values outside the training range were found: "
            + details
            + ". Those predictions are extrapolations and should be treated as "
            "indicative only."
        )

    model_name = st.selectbox("Model", list(artifacts), key="batch_model")
    artifact = artifacts[model_name]

    try:
        predictions = [float(value) for value in artifact["model"].predict(prepared)]
    except Exception as exc:
        st.error(f"Could not score the batch: {exc}")
        return

    display_columns = [column for column in FEATURE_COLUMNS if column in raw.columns]
    display = raw.loc[prepared.index, display_columns]

    _render_results(display, predictions)
