"""Model and dataset insights view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from solar_potential.analytics import feature_importance, model_comparison
from solar_potential.schema import NUMERIC_FEATURES, TARGET_COLUMN
from solar_potential.ui import (
    importance_chart,
    power_by_hour_chart,
    power_by_site_chart,
    season_hour_heatmap,
)


def _takeaway(text: str) -> None:
    st.caption(f"**Takeaway** · {text}")


def render_insights_tab(artifacts: dict, dataset: pd.DataFrame, presets: pd.DataFrame) -> None:
    st.markdown(
        "How the models score, which conditions produce the most output, and what "
        "the models rely on."
    )

    st.subheader("Model comparison")
    comparison = model_comparison(artifacts)
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "CV R2": st.column_config.NumberColumn("CV R²", format="%.4f"),
            "CV MAE": st.column_config.NumberColumn("CV MAE (W)", format="%.3f"),
            "CV RMSE": st.column_config.NumberColumn("CV RMSE (W)", format="%.3f"),
            "Test R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
        },
    )

    best = comparison.sort_values("CV R2", ascending=False).iloc[0]
    runner_up = comparison.sort_values("CV R2", ascending=False).iloc[1]
    _takeaway(
        f"{best['Model']} leads on cross-validated R² ({best['CV R2']:.4f}) by "
        f"{abs(best['CV R2'] - runner_up['CV R2']):.4f} over {runner_up['Model']}. "
        "Both were tuned with 5-fold cross-validation on the training split; the "
        "test R² column is the held-out score."
    )

    st.subheader("Where output comes from")
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Mean output by site**")
            site_means = (
                dataset.groupby("Location")[TARGET_COLUMN].mean().reset_index()
            )
            st.altair_chart(power_by_site_chart(site_means), width="stretch")

    with right:
        with st.container(border=True):
            st.markdown("**Mean output by hour**")
            hour_means = dataset.groupby("Hour")[TARGET_COLUMN].mean().reset_index()
            st.altair_chart(power_by_hour_chart(hour_means), width="stretch")

    _takeaway(
        "Site choice is worth about 2× in mean output (JDMT at the top, Grissom and "
        "Malmstrom at the bottom), and midday hours outperform the edges of the "
        "10:00–15:00 window."
    )

    with st.container(border=True):
        st.markdown("**Season and hour together**")
        season_hour = (
            dataset.groupby(["Season", "Hour"])[TARGET_COLUMN]
            .mean()
            .reset_index()
        )
        st.altair_chart(season_hour_heatmap(season_hour), width="stretch")

    _takeaway(
        "Summer middays are the strongest cell and winter afternoons the weakest. "
        "The gap between them is larger than the gap between most sites, which is "
        "why Month and Hour sit in the model alongside the weather features."
    )

    st.subheader("What the models rely on")
    columns = st.columns(len(artifacts), gap="large")
    for column, (name, artifact) in zip(columns, artifacts.items()):
        with column:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.altair_chart(
                    importance_chart(feature_importance(artifact)), width="stretch"
                )
    _takeaway(
        "Both models put ambient temperature first by a wide margin, then cloud "
        "cover and month. Where they disagree — humidity and hour — the cause is "
        "correlated inputs: either feature can satisfy the same split, so "
        "importance ranks influence rather than causation."
    )

    st.subheader("Correlation with output")
    correlation_columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    correlation = dataset[correlation_columns].corr(method="spearman")
    st.dataframe(
        correlation.round(3),
        column_config={
            column: st.column_config.NumberColumn(column, format="%.3f")
            for column in correlation.columns
        },
    )
    _takeaway(
        "Spearman correlation captures monotone relationships only, which is why "
        "Month looks weak here despite mattering to the model: its effect is an "
        "inverted-U, and trees can represent that where a rank correlation cannot."
    )
