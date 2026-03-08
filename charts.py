"""
charts.py - Plotly chart generation functions for the AI Fitness App dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",     # indigo
    "success": "#10B981",     # emerald
    "warning": "#F59E0B",     # amber
    "danger": "#EF4444",      # red
    "info": "#3B82F6",        # blue
    "purple": "#8B5CF6",
    "pink": "#EC4899",
    "teal": "#14B8A6",
    "bg": "#0F172A",
    "card": "#1E293B",
    "text": "#E2E8F0",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


# ── Weight Chart ───────────────────────────────────────────────────────────────

def weight_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart with actual weight + 7-day rolling average."""
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No weight data yet", **CHART_LAYOUT)
        return fig

    # Actual weight scatter
    fig.add_trace(go.Scatter(
        x=df["logged_date"], y=df["weight_kg"],
        mode="markers+lines",
        name="Weight",
        marker=dict(color=COLORS["primary"], size=7),
        line=dict(color=COLORS["primary"], width=2, dash="dot"),
    ))

    # Rolling average
    if "rolling_avg" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["logged_date"], y=df["rolling_avg"],
            mode="lines",
            name="7-Day Avg",
            line=dict(color=COLORS["success"], width=3),
        ))

    fig.update_layout(
        title="⚖️ Weight Trend",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        hovermode="x unified",
        **CHART_LAYOUT,
    )
    return fig


# ── Calorie Balance Chart ──────────────────────────────────────────────────────

def calorie_balance_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: intake vs burned per day."""
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No calorie data yet", **CHART_LAYOUT)
        return fig

    fig.add_trace(go.Bar(
        x=df["date"], y=df["intake"],
        name="Calories In",
        marker_color=COLORS["warning"],
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["burned"],
        name="Calories Burned",
        marker_color=COLORS["info"],
    ))

    # Net calorie line
    if "net" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["net"],
            mode="lines+markers",
            name="Net",
            line=dict(color=COLORS["danger"], width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title="🔥 Calorie Balance (7 Days)",
        barmode="group",
        xaxis_title="Date",
        yaxis_title="Calories (kcal)",
        hovermode="x unified",
        **CHART_LAYOUT,
    )
    return fig


# ── Workout Frequency Chart ────────────────────────────────────────────────────

def workout_frequency_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing workout sessions over time."""
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No workout data yet", **CHART_LAYOUT)
        return fig

    daily = df.groupby(df["logged_date"].dt.date).size().reset_index()
    daily.columns = ["date", "sessions"]

    fig.add_trace(go.Bar(
        x=daily["date"], y=daily["sessions"],
        name="Workouts",
        marker=dict(
            color=daily["sessions"],
            colorscale=[[0, COLORS["purple"]], [1, COLORS["primary"]]],
        ),
    ))

    fig.update_layout(
        title="🏋️ Workout Sessions",
        xaxis_title="Date",
        yaxis_title="Sessions",
        **CHART_LAYOUT,
    )
    return fig


# ── Workout Category Pie ───────────────────────────────────────────────────────

def workout_category_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart of workouts by category."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No workout data yet", **CHART_LAYOUT)
        return fig

    cat_data = df.groupby("category")["calories_burned"].sum().reset_index()

    fig = go.Figure(go.Pie(
        labels=cat_data["category"],
        values=cat_data["calories_burned"],
        hole=0.55,
        textinfo="label+percent",
        marker=dict(colors=[COLORS["primary"], COLORS["success"], COLORS["warning"],
                             COLORS["info"], COLORS["purple"]]),
    ))
    fig.update_layout(title="🏃 Calories by Category", **CHART_LAYOUT)
    return fig


# ── Macro Distribution Chart ───────────────────────────────────────────────────

def macro_donut_chart(protein_g: float, carbs_g: float, fat_g: float,
                      target_protein: float, target_carbs: float, target_fat: float) -> go.Figure:
    """Donut chart comparing actual vs target macros."""
    labels = ["Protein", "Carbs", "Fat"]
    actual = [protein_g, carbs_g, fat_g]
    targets = [target_protein, target_carbs, target_fat]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels,
        values=actual,
        hole=0.5,
        name="Actual",
        marker=dict(colors=[COLORS["primary"], COLORS["warning"], COLORS["success"]]),
        textinfo="label+value",
        title=dict(text="Today", font=dict(size=14)),
    ))
    fig.update_layout(title="🥗 Macro Breakdown (Today)", **CHART_LAYOUT)
    return fig


# ── BMI Gauge ─────────────────────────────────────────────────────────────────

def bmi_gauge(bmi: float) -> go.Figure:
    """Gauge chart for BMI."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bmi,
        number={"suffix": " BMI"},
        gauge={
            "axis": {"range": [10, 40], "tickcolor": COLORS["text"]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [10, 18.5], "color": COLORS["info"]},
                {"range": [18.5, 25], "color": COLORS["success"]},
                {"range": [25, 30], "color": COLORS["warning"]},
                {"range": [30, 40], "color": COLORS["danger"]},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": bmi,
            },
        },
        title={"text": "BMI", "font": {"size": 20}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]),
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ── Weight Prediction Chart ────────────────────────────────────────────────────

def weight_prediction_chart(actual_df: pd.DataFrame,
                             pred_dates: list, pred_weights: list) -> go.Figure:
    """Overlay predicted weight on top of actual weight history."""
    fig = go.Figure()

    if not actual_df.empty:
        fig.add_trace(go.Scatter(
            x=actual_df["logged_date"], y=actual_df["weight_kg"],
            mode="markers+lines",
            name="Actual Weight",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=6),
        ))

    if pred_dates and pred_weights:
        fig.add_trace(go.Scatter(
            x=pred_dates, y=pred_weights,
            mode="lines",
            name="Predicted Weight",
            line=dict(color=COLORS["warning"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title="🔮 Weight Prediction (Next 30 Days)",
        xaxis_title="Date",
        yaxis_title="Weight (kg)",
        hovermode="x unified",
        **CHART_LAYOUT,
    )
    return fig


# ── Calorie Intake Trend ───────────────────────────────────────────────────────

def calorie_intake_trend_chart(df: pd.DataFrame, tdee: float) -> go.Figure:
    """Area chart of daily calorie intake with TDEE reference line."""
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No calorie data yet", **CHART_LAYOUT)
        return fig

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["calories"],
        fill="tozeroy",
        mode="lines",
        name="Daily Intake",
        line=dict(color=COLORS["warning"], width=2),
        fillcolor="rgba(245,158,11,0.15)",
    ))

    if "rolling_avg" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rolling_avg"],
            mode="lines",
            name="7-Day Avg",
            line=dict(color=COLORS["info"], width=2, dash="dash"),
        ))

    # TDEE reference line
    fig.add_hline(y=tdee, line_dash="dot", line_color=COLORS["success"],
                  annotation_text=f"TDEE: {tdee} kcal", annotation_position="top right")

    fig.update_layout(
        title="📊 Calorie Intake Trend",
        xaxis_title="Date",
        yaxis_title="Calories (kcal)",
        hovermode="x unified",
        **CHART_LAYOUT,
    )
    return fig
