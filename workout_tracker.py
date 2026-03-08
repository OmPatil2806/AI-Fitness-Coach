"""
workout_tracker.py - Workout logging utilities and calorie burn estimation
"""

import pandas as pd

# MET (Metabolic Equivalent of Task) values for common exercises
# Calories burned = MET × weight_kg × duration_hours
EXERCISE_MET = {
    # Cardio
    "Running (6 mph)": 9.8,
    "Running (8 mph)": 11.8,
    "Cycling (moderate)": 8.0,
    "Cycling (vigorous)": 10.0,
    "Swimming (moderate)": 7.0,
    "Jump rope": 11.0,
    "Rowing (moderate)": 7.0,
    "Elliptical": 5.0,
    "Walking (brisk)": 4.3,
    "HIIT": 10.0,
    "Aerobics": 6.5,
    "Stair climbing": 9.0,
    # Strength
    "Weight training (light)": 3.5,
    "Weight training (moderate)": 5.0,
    "Weight training (vigorous)": 6.0,
    "Bodyweight exercises": 4.0,
    "Powerlifting": 6.0,
    "CrossFit": 8.0,
    # Flexibility & Mind-Body
    "Yoga": 3.0,
    "Pilates": 3.5,
    "Stretching": 2.3,
    # Sports
    "Basketball": 8.0,
    "Soccer": 7.0,
    "Tennis": 7.3,
    "Volleyball": 4.0,
    "Hiking": 6.0,
    # Custom
    "Other": 5.0,
}

WORKOUT_CATEGORIES = {
    "Cardio": ["Running (6 mph)", "Running (8 mph)", "Cycling (moderate)", "Cycling (vigorous)",
                "Swimming (moderate)", "Jump rope", "Rowing (moderate)", "Elliptical",
                "Walking (brisk)", "HIIT", "Aerobics", "Stair climbing"],
    "Strength": ["Weight training (light)", "Weight training (moderate)", "Weight training (vigorous)",
                 "Bodyweight exercises", "Powerlifting", "CrossFit"],
    "Flexibility": ["Yoga", "Pilates", "Stretching"],
    "Sports": ["Basketball", "Soccer", "Tennis", "Volleyball", "Hiking"],
    "Other": ["Other"],
}


def get_category_for_exercise(exercise_name: str) -> str:
    """Look up which category an exercise belongs to."""
    for category, exercises in WORKOUT_CATEGORIES.items():
        if exercise_name in exercises:
            return category
    return "Other"


def estimate_calories_burned(exercise_name: str, duration_minutes: float, weight_kg: float) -> float:
    """
    Estimate calories burned using MET formula.
    Calories = MET × weight_kg × (duration_minutes / 60)
    """
    met = EXERCISE_MET.get(exercise_name, 5.0)
    calories = met * weight_kg * (duration_minutes / 60)
    return round(calories, 1)


def workout_log_to_dataframe(workout_log: list) -> pd.DataFrame:
    """Convert raw workout log list to a Pandas DataFrame."""
    if not workout_log:
        return pd.DataFrame(columns=["id", "exercise_name", "category", "duration_minutes",
                                     "calories_burned", "intensity", "logged_date", "notes"])
    df = pd.DataFrame(workout_log)
    df["logged_date"] = pd.to_datetime(df["logged_date"])
    return df


def weekly_workout_summary(workout_log: list) -> dict:
    """
    Summarise workout data for the past 7 days.
    Returns: total_sessions, total_minutes, total_calories, category_breakdown
    """
    df = workout_log_to_dataframe(workout_log)
    if df.empty:
        return {"total_sessions": 0, "total_minutes": 0, "total_calories": 0, "category_breakdown": {}}

    from datetime import date, timedelta
    cutoff = pd.Timestamp(date.today() - timedelta(days=7))
    recent = df[df["logged_date"] >= cutoff]

    category_breakdown = (
        recent.groupby("category")["calories_burned"].sum().to_dict()
        if not recent.empty else {}
    )

    return {
        "total_sessions": len(recent),
        "total_minutes": int(recent["duration_minutes"].sum()),
        "total_calories": round(recent["calories_burned"].sum(), 1),
        "category_breakdown": category_breakdown,
    }


def get_all_exercises() -> list:
    """Return a flat list of all exercise names."""
    exercises = []
    for ex_list in WORKOUT_CATEGORIES.values():
        exercises.extend(ex_list)
    return sorted(exercises)


def estimate_calories_with_intensity(exercise_name: str, duration_minutes: float,
                                      weight_kg: float, intensity: str) -> float:
    """
    Estimate calories burned with MET adjusted for intensity.
    Low = 0.85x, Moderate = 1.0x, High = 1.20x
    """
    met = EXERCISE_MET.get(exercise_name, 5.0)
    multiplier = {"Low": 0.85, "Moderate": 1.0, "High": 1.20}.get(intensity, 1.0)
    calories = met * multiplier * weight_kg * (duration_minutes / 60)
    return round(calories, 1)


def get_personal_records_by_exercise(workout_log: list) -> dict:
    """
    For each exercise, return best: calories_burned, duration_minutes, sets (if logged).
    Returns dict keyed by exercise_name.
    """
    if not workout_log:
        return {}
    df = pd.DataFrame(workout_log)
    prs = {}
    for ex, grp in df.groupby("exercise_name"):
        best_cal  = grp.loc[grp["calories_burned"].idxmax()]
        best_dur  = grp.loc[grp["duration_minutes"].idxmax()]
        sets_col  = "sets" if "sets" in grp.columns else None
        best_sets = int(grp[sets_col].max()) if sets_col and grp[sets_col].notna().any() else None
        prs[ex] = {
            "sessions":       len(grp),
            "best_cal":       round(float(best_cal["calories_burned"]), 1),
            "best_cal_date":  str(best_cal["logged_date"])[:10],
            "best_dur":       int(best_dur["duration_minutes"]),
            "best_dur_date":  str(best_dur["logged_date"])[:10],
            "best_sets":      best_sets,
            "total_minutes":  int(grp["duration_minutes"].sum()),
        }
    return prs


def category_streak(workout_log: list) -> dict:
    """
    Calculate current consecutive-day streak per workout category.
    Returns dict: {category: streak_days}
    """
    from datetime import date, timedelta
    if not workout_log:
        return {"Cardio": 0, "Strength": 0, "Flexibility": 0, "Sports": 0}

    result = {}
    for cat in ["Cardio", "Strength", "Flexibility", "Sports"]:
        cat_dates = sorted(
            {entry["logged_date"] for entry in workout_log if entry.get("category") == cat},
            reverse=True,
        )
        streak = 0
        check  = date.today()
        for d in cat_dates:
            if str(check) == d:
                streak += 1
                check -= timedelta(days=1)
            elif str(check - timedelta(days=1)) == d:
                check -= timedelta(days=1)
                streak += 1
                check -= timedelta(days=1)
            else:
                break
        result[cat] = streak
    return result


def last_session_per_category(workout_log: list) -> dict:
    """Return most recent logged_date per category."""
    result = {}
    for entry in workout_log:
        cat = entry.get("category", "Other")
        d   = entry["logged_date"]
        if cat not in result or d > result[cat]:
            result[cat] = d
    return result