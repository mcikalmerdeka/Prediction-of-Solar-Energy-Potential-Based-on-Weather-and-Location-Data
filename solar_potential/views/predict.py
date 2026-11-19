"""Individual prediction view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from solar_potential.analytics import predict_power, target_band, sensitivity
from solar_potential.data import preset_for
from solar_potential.schema import (
    DISPLAY_NAMES,
    FEATURE_COLUMNS,
    HOURS,
    INPUT_BOUNDS,
    MONTH_SEASON,
    NUMERIC_FEATURES,
    SITE_NAMES,
)
from solar_potential.ui import gauge_chart, sensitivity_chart

SITE_KEY = "predict_site"
MONTH_KEY = "predict_month"
HOUR_KEY = "predict_hour"
MODEL_KEY = "predict_model"
RESULT_KEY = "predict_result"
SIGNATURE_KEY = "predict_signature"


def _input_key(feature: str) -> str:
    return f"predict_input_{feature}"


def _sync_preset(site: str, month: int, hour: int, preset: dict) -> None:
    """Refill the weather inputs when the site/month/hour selection changes."""
    signature = (site, month, hour)
    if st.session_state.get(SIGNATURE_KEY) == signature:
        return

    for feature in NUMERIC_FEATURES:
        st.session_state[_input_key(feature)] = float(preset[feature])
    st.session_state[SIGNATURE_KEY] = signature
    st.session_state.pop(RESULT_KEY, None)


def _render_result(result: dict, artifacts: dict) -> None:
    predictions = result["predictions"]
    selected = result["model"]

    st.markdown("### Prediction")
    with st.container(horizontal=True):
        st.metric(
            f"Predicted output · {selected}",
            f"{predictions[selected]:.2f} W",
            border=True,
        )
        st.metric("Band", target_band(predictions[selected]), border=True)
        st.metric("Site", result["site"], border=True)
        st.metric("Season", result["season"], border=True)

    st.altair_chart(
        gauge_chart(predictions[selected]), width="stretch"
    )
    st.caption(
        "Track spans the observed dataset range (0–35 W). "
        "The model was trained on horizontal panels at 12 northern-hemisphere sites."
    )

    with st.container(border=True):
        st.markdown("**What moved this prediction**")
        st.altair_chart(
            sensitivity_chart(result["sensitivity"]), width="stretch"
        )
        st.caption(
            "Each bar swaps one feature for this site's typical value at the chosen "
            "month and hour, then re-scores. Positive bars add output relative to "
            "typical conditions; negative bars reduce it."
        )

    if len(predictions) > 1:
        table = pd.DataFrame(
            {
                "Model": list(predictions),
                "Predicted output (W)": [round(value, 3) for value in predictions.values()],
            }
        )
        st.dataframe(table, hide_index=True)

    with st.expander("Model input (6 features)"):
        st.dataframe(result["model_input"], hide_index=True)

    st.caption(
        "Location is not a model input. It selects this site's typical weather; the "
        "score is a function of the weather and time values shown above."
    )


def render_predict_tab(artifacts: dict, dataset: pd.DataFrame, presets: pd.DataFrame) -> None:
    st.markdown(
        "Score the expected panel output for a site and moment. Pick a site, month, "
        "and hour to prefill that site's median weather, then adjust the conditions."
    )

    left, middle, right = st.columns(3, gap="large")
    with left:
        site = st.selectbox("Site", SITE_NAMES, key=SITE_KEY)
    with middle:
        month = st.slider("Month", 1, 12, value=7, key=MONTH_KEY)
    with right:
        hour = st.selectbox("Hour of day", HOURS, index=2, key=HOUR_KEY)

    preset = preset_for(presets, dataset, site, month, hour)
    _sync_preset(site, month, hour, preset)

    st.caption(
        f"{site} in {MONTH_SEASON[month]} at {hour:02d}:00 — inputs prefilled with "
        "that site's median weather for this month and hour."
    )

    with st.form("predict_form", border=True):
        st.markdown("**Weather conditions**")
        st.caption(
            "Values are bounded to the range observed in the training data, so the "
            "model never extrapolates from this form."
        )

        columns = st.columns(2, gap="large")
        for index, feature in enumerate(NUMERIC_FEATURES):
            low, high, step = INPUT_BOUNDS[feature]
            with columns[index % 2]:
                st.number_input(
                    DISPLAY_NAMES[feature],
                    min_value=float(low),
                    max_value=float(high),
                    step=float(step),
                    format="%.1f",
                    key=_input_key(feature),
                )

        model = st.segmented_control(
            "Model",
            list(artifacts),
            default=list(artifacts)[0],
            key=MODEL_KEY,
        )
        submitted = st.form_submit_button(
            "Predict output", type="primary", icon=":material/bolt:"
        )

    if submitted:
        model = model or list(artifacts)[0]
        inputs = {feature: float(st.session_state[_input_key(feature)]) for feature in NUMERIC_FEATURES}
        inputs["Month"] = float(month)
        inputs["Hour"] = float(hour)
        row = pd.Series(inputs)[FEATURE_COLUMNS]

        try:
            model_input = row.to_frame().T
            predictions = {
                name: predict_power(artifact, model_input)
                for name, artifact in artifacts.items()
            }
            table = sensitivity(
                artifacts[model], row, pd.Series(preset)[FEATURE_COLUMNS]
            )
        except Exception as exc:  # surface the reason instead of blanking the page
            st.error(f"Could not score these conditions: {exc}")
        else:
            st.session_state[RESULT_KEY] = {
                "predictions": predictions,
                "model": model,
                "sensitivity": table,
                "model_input": model_input,
                "site": site,
                "season": MONTH_SEASON[month],
            }

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.caption("Submit the form to score these conditions.")
        return

    _render_result(result, artifacts)
