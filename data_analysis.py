"""
data_analysis.py - Data analysis for weight trends, workout frequency, calorie trends
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta


# ── Weight Analysis ────────────────────────────────────────────────────────────

def weight_trend_analysis(weight_log: list) -> dict:
    """
    Analyse weight log data.
    Returns: current_weight, start_weight, change, trend_direction, rolling_avg_df
    """
    if not weight_log:
        return {}

    df = pd.DataFrame(weight_log)
    df["logged_date"] = pd.to_datetime(df["logged_date"])
    df = df.sort_values("logged_date")

    current_weight = df["weight_kg"].iloc[-1]
    start_weight = df["weight_kg"].iloc[0]
    change = round(current_weight - start_weight, 2)

    # 7-day rolling average to smooth noise
    df["rolling_avg"] = df["weight_kg"].rolling(window=7, min_periods=1).mean().round(2)

    # Linear trend (slope)
    if len(df) >= 2:
        x = np.arange(len(df))
        slope, _ = np.polyfit(x, df["weight_kg"], 1)
        trend_direction = "decreasing" if slope < -0.01 else "increasing" if slope > 0.01 else "stable"
        weekly_rate = round(slope * 7, 2)  # kg per week
    else:
        trend_direction = "stable"
        weekly_rate = 0.0

    return {
        "current_weight": current_weight,
        "start_weight": start_weight,
        "change": change,
        "trend_direction": trend_direction,
        "weekly_rate_kg": weekly_rate,
        "total_entries": len(df),
        "dataframe": df,
    }


def weight_log_to_dataframe(weight_log: list) -> pd.DataFrame:
    if not weight_log:
        return pd.DataFrame(columns=["id", "weight_kg", "logged_date", "notes"])
    df = pd.DataFrame(weight_log)
    df["logged_date"] = pd.to_datetime(df["logged_date"])
    return df.sort_values("logged_date")


# ── Workout Frequency Analysis ─────────────────────────────────────────────────

def workout_frequency_analysis(workout_log: list, days: int = 30) -> dict:
    """
    Analyse workout frequency over the past N days.
    """
    if not workout_log:
        return {"sessions_per_week": 0, "most_common_exercise": "N/A", "total_minutes": 0}

    df = pd.DataFrame(workout_log)
    df["logged_date"] = pd.to_datetime(df["logged_date"])
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    df = df[df["logged_date"] >= cutoff]

    if df.empty:
        return {"sessions_per_week": 0, "most_common_exercise": "N/A", "total_minutes": 0}

    sessions_per_week = round(len(df) / (days / 7), 1)
    most_common = df["exercise_name"].mode().iloc[0] if not df.empty else "N/A"
    total_minutes = int(df["duration_minutes"].sum())
    total_calories = round(df["calories_burned"].sum(), 1)

    # Daily workout counts
    daily_counts = df.groupby(df["logged_date"].dt.date).size().reset_index()
    daily_counts.columns = ["date", "sessions"]

    return {
        "sessions_per_week": sessions_per_week,
        "most_common_exercise": most_common,
        "total_minutes": total_minutes,
        "total_calories_burned": total_calories,
        "daily_counts": daily_counts,
        "dataframe": df,
    }


# ── Calorie Trend Analysis ─────────────────────────────────────────────────────

def calorie_trend_analysis(calorie_log: list, days: int = 30) -> dict:
    """
    Analyse calorie intake trends over the past N days.
    """
    if not calorie_log:
        return {"avg_daily_intake": 0, "max_day": 0, "min_day": 0, "dataframe": pd.DataFrame()}

    df = pd.DataFrame(calorie_log)
    df["logged_date"] = pd.to_datetime(df["logged_date"])
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    df = df[df["logged_date"] >= cutoff]

    if df.empty:
        return {"avg_daily_intake": 0, "max_day": 0, "min_day": 0, "dataframe": pd.DataFrame()}

    daily = df.groupby(df["logged_date"].dt.date)["calories"].sum().reset_index()
    daily.columns = ["date", "calories"]
    daily["rolling_avg"] = daily["calories"].rolling(7, min_periods=1).mean().round(1)

    return {
        "avg_daily_intake": round(daily["calories"].mean(), 1),
        "max_day": round(daily["calories"].max(), 1),
        "min_day": round(daily["calories"].min(), 1),
        "dataframe": daily,
    }


# ── Combined Progress Summary ──────────────────────────────────────────────────

def generate_progress_summary(weight_log, workout_log, calorie_log, tdee) -> dict:
    """
    Produce a single summary dict for the dashboard metrics strip.
    """
    weight_info = weight_trend_analysis(weight_log)
    workout_info = workout_frequency_analysis(workout_log)
    calorie_info = calorie_trend_analysis(calorie_log)

    return {
        "current_weight": weight_info.get("current_weight", "—"),
        "weight_change": weight_info.get("change", 0),
        "weight_trend": weight_info.get("trend_direction", "stable"),
        "workouts_per_week": workout_info.get("sessions_per_week", 0),
        "avg_daily_calories": calorie_info.get("avg_daily_intake", 0),
        "calorie_vs_tdee": round(calorie_info.get("avg_daily_intake", 0) - tdee, 1),
    }
