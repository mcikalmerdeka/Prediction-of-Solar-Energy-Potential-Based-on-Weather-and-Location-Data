"""About and model card view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from solar_potential.analytics import model_comparison
from solar_potential.schema import FEATURE_COLUMNS, TARGET_COLUMN, get_feature_definitions


def _feature_table() -> pd.DataFrame:
    used = set(FEATURE_COLUMNS)
    rows = []

    for name, meta in get_feature_definitions().items():
        if name == TARGET_COLUMN:
            status = "Target"
        elif name in used:
            status = "Yes"
        else:
            status = "No"

        rows.append(
            {
                "Feature": name,
                "Description": meta["description"],
                "Type": f"{meta['data_type']} · {meta['specific_type']}",
                "Used by model": status,
            }
        )

    return pd.DataFrame(rows)


def render_about_tab(artifacts: dict, dataset: pd.DataFrame, presets: pd.DataFrame) -> None:
    st.markdown(
        "Solar output is intermittent: it depends on cloud cover, temperature, "
        "humidity, air mass, the season, and the hour. This project models that "
        "relationship directly, so a site planner can estimate output from weather "
        "instead of metering every location."
    )

    st.markdown("**Why this matters**")
    st.markdown(
        "US electricity demand is projected to grow by roughly 30% by 2035, while "
        "solar's share of supply is projected to rise from about 3% to as much as "
        "40%. Deciding *where* to build, and what output to expect under given "
        "conditions, is the practical bottleneck — not panel technology."
    )

    comparison = model_comparison(artifacts)
    best = comparison.sort_values("CV R2", ascending=False).iloc[0]

    with st.container(horizontal=True):
        st.metric("Best CV R²", f"{best['CV R2']:.4f}", border=True)
        st.metric("Best CV MAE", f"{best['CV MAE']:.2f} W", border=True)
        st.metric("Observations", f"{len(dataset):,}", border=True)
        st.metric("Sites", f"{dataset['Location'].nunique()}", border=True)

    with st.container(border=True):
        st.markdown("**Model card**")
        st.markdown(
            f"- **Algorithms** · {' and '.join(artifact['label'] for artifact in artifacts.values())}"
            f" (regression ensembles)\n"
            f"- **Target** · {TARGET_COLUMN} in Watts\n"
            f"- **Features** · {len(FEATURE_COLUMNS)} numeric: "
            + ", ".join(f"`{feature}`" for feature in FEATURE_COLUMNS)
            + "\n"
            f"- **Training data** · {len(dataset):,} observations from 12 US sites, "
            "December 2017 – February 2019\n"
            "- **Artifacts** · `models/tuned_rf_model.joblib`, "
            "`models/tuned_xgb_model.joblib`\n"
            "- **Evaluation** · 5-fold, 3-repeat cross-validation on the training "
            "split, plus a held-out 25% test split"
        )
        st.caption(
            "Each artifact is a bundle: model, feature order, stored metrics, label, "
            "and target name. Re-running the notebook regenerates both."
        )

    st.subheader("Data dictionary")
    st.caption(
        "Location and Season exist in the source data but are not model inputs — "
        "Season is a recode of Month, and site dummies duplicate the elevation "
        "signal already carried by Pressure."
    )
    st.dataframe(
        _feature_table(),
        hide_index=True,
        column_config={
            "Description": st.column_config.TextColumn(width="large"),
            "Used by model": st.column_config.TextColumn(width="small"),
        },
    )

    with st.expander("Source data preview"):
        st.dataframe(dataset.head(10), hide_index=True)

    st.subheader("Limitations")
    st.markdown(
        "- **Random split, not temporal.** The test split is random over rows, so the "
        "same site and timestamp can appear in training and testing. Reported R² is "
        "optimistic for a genuinely new day.\n"
        "- **Daylight only.** Observations cover 10:00–15:00. The model says nothing "
        "about night or early-morning output.\n"
        "- **Twelve sites, fourteen months, northern hemisphere.** Nothing here "
        "transfers to other latitudes without re-validation.\n"
        "- **Location is not a feature.** Two sites with identical weather get the "
        "same prediction, even if their panels or wiring differ.\n"
        "- **Cloud ceiling is capped at 722.** Above that, the feature is effectively "
        "binary (clear or not).\n"
        "- **MAPE is unstable near zero.** With output as low as 0.26 W, quote MAE and "
        "RMSE instead of percentage error."
    )

    st.subheader("References")
    st.markdown(
        "- Pasion C, Wagner T, Koschnick C, Schuldt S, Williams J, Hallinan K. "
        "*Machine Learning Modeling of Horizontal Photovoltaics Using Weather and "
        "Location Data.* Energies. 2020; 13(10):2570. "
        "https://doi.org/10.3390/en13102570\n"
        "- Dataset: "
        "https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic"
    )

    st.subheader("Author")
    st.markdown(
        "Developed by **Muhammad Cikal Merdeka** · Data Analyst / Data Scientist\n\n"
        "[GitHub](https://github.com/mcikalmerdeka) · "
        "[LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)"
    )
    st.caption(
        "Light and dark themes are available from the app menu (⋮ → Settings → Theme)."
    )
