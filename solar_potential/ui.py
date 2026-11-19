"""Altair chart builders for the Solar Potential app.

No Streamlit import here on purpose: the charts are plain Altair objects, so
they can be built and inspected from a script or a test without a server.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

ACCENT = "#D97706"
POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
INK = "#1C1917"
TRACK = "#E7E5E4"

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]


def power_by_site_chart(frame: pd.DataFrame) -> alt.Chart:
    """Mean output per site, highest first."""
    order = frame.sort_values("PolyPwr", ascending=False)["Location"].tolist()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("PolyPwr:Q", title="Mean output (W)"),
            y=alt.Y("Location:N", title=None, sort=order),
            tooltip=[
                alt.Tooltip("Location:N", title="Site"),
                alt.Tooltip("PolyPwr:Q", title="Mean output (W)", format=".2f"),
            ],
        )
        .properties(height=alt.Step(26))
    )


def power_by_hour_chart(frame: pd.DataFrame) -> alt.Chart:
    """Mean output per hour of day, in clock order."""
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("Hour:O", title="Hour of day"),
            y=alt.Y("PolyPwr:Q", title="Mean output (W)"),
            tooltip=[
                alt.Tooltip("Hour:O", title="Hour"),
                alt.Tooltip("PolyPwr:Q", title="Mean output (W)", format=".2f"),
            ],
        )
        .properties(height=260)
    )


def season_hour_heatmap(frame: pd.DataFrame) -> alt.Chart:
    """Mean output by season and hour, as a labelled grid."""
    base = alt.Chart(frame).mark_rect(cornerRadius=2).encode(
        x=alt.X("Hour:O", title="Hour of day"),
        y=alt.Y("Season:N", title=None, sort=SEASON_ORDER),
        color=alt.Color(
            "PolyPwr:Q",
            scale=alt.Scale(scheme="orangered"),
            title="Mean output (W)",
        ),
        tooltip=[
            alt.Tooltip("Season:N", title="Season"),
            alt.Tooltip("Hour:O", title="Hour"),
            alt.Tooltip("PolyPwr:Q", title="Mean output (W)", format=".2f"),
        ],
    )
    labels = base.mark_text(fontSize=12, fontWeight="bold").encode(
        text=alt.Text("PolyPwr:Q", format=".1f"),
        color=alt.condition(
            "datum.PolyPwr > 12", alt.value("#FFFFFF"), alt.value(INK)
        ),
    )
    return (base + labels).properties(height=200)


def importance_chart(frame: pd.DataFrame) -> alt.Chart:
    """Impurity-based feature importance, largest first."""
    order = frame["feature"].tolist()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", title=None, sort=order),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".3f"),
            ],
        )
        .properties(height=alt.Step(30))
    )


def sensitivity_chart(frame: pd.DataFrame) -> alt.Chart:
    """Per-prediction deltas: what this weather does versus typical weather."""
    frame = frame.assign(
        direction=frame["delta"].map(
            lambda delta: "Adds output" if delta >= 0 else "Reduces output"
        )
    )
    order = frame["display"].tolist()
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("delta:Q", title="Change vs typical conditions (W)"),
            y=alt.Y("display:N", title=None, sort=order),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Adds output", "Reduces output"],
                    range=[POSITIVE, NEGATIVE],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("display:N", title="Feature"),
                alt.Tooltip("input:Q", title="This value", format=".1f"),
                alt.Tooltip("baseline:Q", title="Typical value", format=".1f"),
                alt.Tooltip("delta:Q", title="Change (W)", format="+.2f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(color="#A8A29E", strokeWidth=1)
        .encode(x="zero:Q")
    )
    return (bars + rule).properties(height=alt.Step(30))


def gauge_chart(value: float, low: float = 0.0, high: float = 35.0) -> alt.Chart:
    """Bullet gauge: a full-range track with the prediction marked on it.

    Deliberately a horizontal bullet rather than an arc: a bar encodes length,
    which is read accurately, and it carries a text label in the same view.
    """
    clamped = min(max(float(value), low), high)
    track = (
        alt.Chart(pd.DataFrame({"low": [low], "high": [high]}))
        .mark_bar(color=TRACK, cornerRadius=4, height=34)
        .encode(x=alt.X("low:Q", title=None), x2="high:Q")
    )
    fill = (
        alt.Chart(pd.DataFrame({"low": [low], "value": [clamped]}))
        .mark_bar(color=ACCENT, cornerRadius=4, height=34)
        .encode(x=alt.X("low:Q", title=None), x2="value:Q")
    )
    label = (
        alt.Chart(pd.DataFrame({"x": [(low + high) / 2], "label": [f"{value:.1f} W"]}))
        .mark_text(fontSize=18, fontWeight="bold", color=INK)
        .encode(x=alt.X("x:Q", title=None), text="label:N")
    )
    return (
        (track + fill + label)
        .encode(x=alt.X(scale=alt.Scale(domain=[low, high], nice=False)))
        .properties(height=60)
    )
