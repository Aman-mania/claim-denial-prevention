"""
Dashboard — Chart Components
==============================
Pure chart functions: data in → plotly Figure out.
No st.* calls here — charts are rendered by tab modules.
This keeps charts testable and reusable outside Streamlit.

All functions handle empty DataFrames gracefully (return an empty figure
with a descriptive title rather than crashing).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Brand palette — consistent across all charts
_COLORS = {
    "primary":  "#4F46E5",   # indigo
    "success":  "#10B981",   # green
    "warning":  "#F59E0B",   # amber
    "danger":   "#EF4444",   # red
    "neutral":  "#6B7280",   # grey
    "light":    "#E5E7EB",   # light grey
}
_SEQ_PALETTE = ["#4F46E5", "#7C3AED", "#10B981", "#F59E0B", "#EF4444", "#06B6D4"]


def _empty_figure(message: str = "No data available") -> go.Figure:
    """Standard empty-state figure shown when data is missing."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 14, "color": _COLORS["neutral"]})
    fig.update_layout(height=250, template="plotly_white",
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig


# ── Null rate bar chart ────────────────────────────────────────────────────────

def null_rate_chart(null_df: pd.DataFrame, title: str = "Null Rate by Column") -> go.Figure:
    """
    Horizontal bar chart showing null % per column.
    null_df must have columns: [column, null_pct].
    Columns with 0% null are greyed out for readability.
    """
    if null_df.empty or "column" not in null_df.columns:
        return _empty_figure("No null profile data")

    df = null_df[null_df["null_pct"] > 0].sort_values("null_pct", ascending=True)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="All columns fully populated", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font={"size": 14, "color": _COLORS["success"]})
        fig.update_layout(height=200, template="plotly_white",
                          title=title, xaxis={"visible": False}, yaxis={"visible": False})
        return fig

    # Colour-code by severity
    colors = [
        _COLORS["danger"]  if p > 25 else
        _COLORS["warning"] if p > 10 else
        _COLORS["neutral"]
        for p in df["null_pct"]
    ]

    fig = go.Figure(go.Bar(
        x=df["null_pct"],
        y=df["column"],
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in df["null_pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Null %",
        xaxis={"range": [0, max(df["null_pct"]) * 1.3]},
        template="plotly_white",
        height=max(200, len(df) * 45 + 80),
        margin={"l": 160, "r": 60, "t": 50, "b": 40},
    )
    return fig


# ── Claims timeline ────────────────────────────────────────────────────────────

def claims_timeline_chart(timeline_df: pd.DataFrame) -> go.Figure:
    """Line chart: daily claim count over time."""
    if timeline_df.empty or "claim_count" not in timeline_df.columns:
        return _empty_figure("No timeline data")

    date_col = "date_str" if "date_str" in timeline_df.columns else timeline_df.columns[0]

    fig = go.Figure(go.Scatter(
        x=timeline_df[date_col],
        y=timeline_df["claim_count"],
        mode="lines+markers",
        line={"color": _COLORS["primary"], "width": 2},
        marker={"size": 5},
        fill="tozeroy",
        fillcolor="rgba(79,70,229,0.1)",
    ))
    fig.update_layout(
        title="Claims per Day",
        xaxis_title="Date",
        yaxis_title="Claims",
        template="plotly_white",
        height=300,
        margin={"t": 50, "b": 60},
    )
    return fig


# ── Categorical bar chart ──────────────────────────────────────────────────────

def bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color: str = "primary",
    top_n: int = 15,
    horizontal: bool = True,
) -> go.Figure:
    """
    Generic bar chart for categorical breakdowns (provider, diagnosis, specialty).
    x_col = category labels, y_col = numeric values.
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return _empty_figure(f"No data for {title}")

    plot_df = df[[x_col, y_col]].dropna(subset=[y_col]).head(top_n)

    # Fill NaN category labels with "Unknown"
    plot_df[x_col] = plot_df[x_col].fillna("Unknown").astype(str)

    if horizontal:
        plot_df = plot_df.sort_values(y_col, ascending=True)
        fig = go.Figure(go.Bar(
            x=plot_df[y_col], y=plot_df[x_col],
            orientation="h",
            marker_color=_COLORS.get(color, _COLORS["primary"]),
            text=plot_df[y_col].round(1),
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title=y_col.replace("_", " ").title(),
            height=max(250, len(plot_df) * 40 + 80),
            margin={"l": 160, "r": 60},
        )
    else:
        fig = go.Figure(go.Bar(
            x=plot_df[x_col], y=plot_df[y_col],
            marker_color=_COLORS.get(color, _COLORS["primary"]),
        ))
        fig.update_layout(yaxis_title=y_col.replace("_", " ").title(), height=300)

    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    return fig


# ── Cost comparison chart ──────────────────────────────────────────────────────

def cost_comparison_chart(cost_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: avg_billed vs expected_cost per procedure code.
    Visually shows which procedures have the largest billing deviations.
    """
    if cost_df.empty or "procedure_code" not in cost_df.columns:
        return _empty_figure("No cost analysis data")

    df = cost_df.dropna(subset=["avg_billed", "expected_cost"]).copy()
    df["procedure_code"] = df["procedure_code"].fillna("Unknown")

    fig = go.Figure([
        go.Bar(
            name="Avg Billed",
            x=df["procedure_code"],
            y=df["avg_billed"],
            marker_color=_COLORS["danger"],
        ),
        go.Bar(
            name="Expected Cost",
            x=df["procedure_code"],
            y=df["expected_cost"],
            marker_color=_COLORS["success"],
        ),
    ])
    fig.update_layout(
        title="Billed vs Expected Cost by Procedure",
        xaxis_title="Procedure Code",
        yaxis_title="Amount (₹)",
        barmode="group",
        template="plotly_white",
        height=350,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


# ── Null before/after comparison ───────────────────────────────────────────────

def null_comparison_chart(impact: dict) -> go.Figure:
    """
    Grouped bar chart comparing null rates Before (Bronze) and After (Silver).
    impact dict must have bronze_* and silver_* null_pct keys.
    """
    required = [
        "bronze_diag_null_pct", "silver_diag_null_pct",
        "bronze_proc_null_pct", "silver_proc_null_pct",
        "bronze_amount_null_pct", "silver_amount_null_pct",
    ]
    if not all(k in impact for k in required):
        return _empty_figure("Cleaning impact data incomplete")

    labels = ["Diagnosis Code", "Procedure Code", "Billed Amount"]
    before = [impact["bronze_diag_null_pct"], impact["bronze_proc_null_pct"], impact["bronze_amount_null_pct"]]
    after  = [impact["silver_diag_null_pct"], impact["silver_proc_null_pct"], impact["silver_amount_null_pct"]]

    fig = go.Figure([
        go.Bar(name="Before (Bronze)", x=labels, y=before,
               marker_color=_COLORS["warning"], opacity=0.8),
        go.Bar(name="After (Silver)",  x=labels, y=after,
               marker_color=_COLORS["success"], opacity=0.9),
    ])
    fig.update_layout(
        title="Null Rate: Before vs After Cleaning",
        yaxis_title="Null %",
        barmode="group",
        template="plotly_white",
        height=320,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


# ── Missing flags breakdown ────────────────────────────────────────────────────

def missing_flags_chart(silver_claims: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing how many claims are flagged for each missing field.
    Uses Silver flag columns: *_missing boolean columns.
    """
    flag_cols = {
        "diagnosis_code_missing": "Missing Diagnosis Code",
        "procedure_code_missing": "Missing Procedure Code",
        "billed_amount_missing":  "Missing Billed Amount",
    }

    available = {v: int(silver_claims[k].sum())
                 for k, v in flag_cols.items()
                 if k in silver_claims.columns}

    if not available:
        return _empty_figure("No missing-field flag data found")

    labels = list(available.keys())
    values = list(available.values())

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=[_COLORS["danger"], _COLORS["warning"], _COLORS["primary"]],
        text=values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Claims Flagged by Missing Field (Silver Layer)",
        yaxis_title="Number of Claims",
        template="plotly_white",
        height=320,
        showlegend=False,
    )
    return fig


# ── Completeness funnel ────────────────────────────────────────────────────────

def completeness_chart(completeness_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal stacked bar showing claim completeness breakdown.
    completeness_df: [completeness_level, claim_count, pct]
    """
    if completeness_df.empty:
        return _empty_figure("No completeness data")

    colors = [_COLORS["success"], _COLORS["warning"], _COLORS["danger"], "#1A1A2E"]
    fig = go.Figure(go.Bar(
        x=completeness_df["claim_count"],
        y=completeness_df["completeness_level"],
        orientation="h",
        marker_color=colors[:len(completeness_df)],
        text=[f"{row['claim_count']} ({row['pct']}%)"
              for _, row in completeness_df.iterrows()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Claim Completeness Breakdown",
        xaxis_title="Number of Claims",
        template="plotly_white",
        height=280,
        margin={"l": 200, "r": 80, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig


# ── Violation summary chart ───────────────────────────────────────────────────

def violation_chart(violations_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of business logic violation counts.
    Color-coded by severity.
    """
    if violations_df.empty:
        return _empty_figure("No violation data")

    df = violations_df.sort_values("claim_count", ascending=True)
    colors = [_COLORS["danger"] if s == "High" else _COLORS["warning"]
              for s in df["severity"]]

    fig = go.Figure(go.Bar(
        x=df["claim_count"],
        y=df["violation"],
        orientation="h",
        marker_color=colors,
        text=[f"{row['claim_count']} ({row['pct_of_claims']}%)"
              for _, row in df.iterrows()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Business Logic Violations Detected",
        xaxis_title="Claims Affected",
        template="plotly_white",
        height=300,
        margin={"l": 220, "r": 80, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig


# ── Billed amount histogram ───────────────────────────────────────────────────

def billed_histogram(dist_df: pd.DataFrame) -> go.Figure:
    """Bar chart representing billed_amount distribution bins."""
    if dist_df.empty:
        return _empty_figure("No billed amount data")

    fig = go.Figure(go.Bar(
        x=dist_df["bin_label"],
        y=dist_df["claim_count"],
        marker_color=_COLORS["primary"],
        text=dist_df["claim_count"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Billed Amount Distribution",
        xaxis_title="Amount Range",
        yaxis_title="Claims",
        template="plotly_white",
        height=320,
        xaxis_tickangle=-35,
        margin={"t": 50, "b": 80},
    )
    return fig


# ── Provider violation rate chart ─────────────────────────────────────────────

def provider_violation_chart(risk_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Bar chart: providers sorted by violation rate (%).
    Highlights high-risk providers for the rule engine.
    """
    if risk_df.empty or "violation_rate_pct" not in risk_df.columns:
        return _empty_figure("No provider risk data")

    df = risk_df.head(top_n).sort_values("violation_rate_pct", ascending=True)

    # Color: red if >50% violation rate, amber if >25%, green otherwise
    colors = [
        _COLORS["danger"]  if v > 50 else
        _COLORS["warning"] if v > 25 else
        _COLORS["success"]
        for v in df["violation_rate_pct"]
    ]

    fig = go.Figure(go.Bar(
        x=df["violation_rate_pct"],
        y=df["provider_id"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in df["violation_rate_pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Provider Violation Rate (% of claims with a flag)",
        xaxis_title="Violation Rate %",
        xaxis={"range": [0, df["violation_rate_pct"].max() * 1.3]},
        template="plotly_white",
        height=max(280, len(df) * 32 + 80),
        margin={"l": 80, "r": 80, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig
