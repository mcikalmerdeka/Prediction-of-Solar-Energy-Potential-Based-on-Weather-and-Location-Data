"""Solar Potential Studio — Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from solar_potential.analytics import load_artifact
from solar_potential.config import RF_MODEL_PATH, XGB_MODEL_PATH
from solar_potential.data import load_dataset, site_presets
from solar_potential.schema import FEATURE_COLUMNS
from solar_potential.views import (
    render_about_tab,
    render_batch_tab,
    render_insights_tab,
    render_predict_tab,
)

st.set_page_config(
    page_title="Solar Potential Studio",
    page_icon=":material/solar_power:",
    layout="wide",
)


@st.cache_resource
def load_artifacts() -> dict:
    """Load both model bundles once per session."""
    return {
        "Random Forest": load_artifact(RF_MODEL_PATH),
        "XGBoost": load_artifact(XGB_MODEL_PATH),
    }


@st.cache_data
def load_reference_data():
    """Load the source dataset and its site presets once per session."""
    dataset = load_dataset()
    return dataset, site_presets(dataset)


missing = [path for path in (RF_MODEL_PATH, XGB_MODEL_PATH) if not path.exists()]
if missing:
    st.error(
        "Model artifact not found: "
        + ", ".join(str(path) for path in missing)
        + ". Run `notebook.ipynb` end to end to regenerate `models/`, then reload."
    )
    st.stop()

try:
    artifacts = load_artifacts()
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

dataset, presets = load_reference_data()

st.title("Solar Potential Studio")
st.caption(
    "Estimate photovoltaic output from weather and time, compare sites, and see "
    "what drives each prediction."
)

with st.container(horizontal=True):
    for name, artifact in artifacts.items():
        st.badge(artifact["label"], icon=":material/model_training:", color="orange")
    st.badge(
        f"{len(FEATURE_COLUMNS)} features", icon=":material/tune:", color="gray"
    )
    st.badge(
        f"{len(dataset):,} observations", icon=":material/database:", color="gray"
    )

tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["Predict", "Batch", "Insights", "About"]
)

with tab_predict:
    render_predict_tab(artifacts, dataset, presets)

with tab_batch:
    render_batch_tab(artifacts, dataset, presets)

with tab_insights:
    render_insights_tab(artifacts, dataset, presets)

with tab_about:
    render_about_tab(artifacts, dataset, presets)
