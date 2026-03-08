"""
calorie_calculator.py - Daily calorie needs, calorie balance, net calorie tracking
"""

import pandas as pd
from datetime import date, timedelta


def daily_calorie_balance(calorie_log: list, workout_log: list, tdee: float, target_date: date = None) -> dict:
    """
    Compute calorie intake, burned, and net balance for a specific date.
    
    Args:
        calorie_log:  list of dicts from database.get_calorie_log()
        workout_log:  list of dicts from database.get_workout_log()
        tdee:         Total Daily Energy Expenditure (kcal)
        target_date:  date to compute for (defaults to today)
    
    Returns dict with keys: intake, burned_exercise, burned_bmr, total_burned, net, surplus_deficit
    """
    if target_date is None:
        target_date = date.today()
    target_str = str(target_date)

    # Sum calories consumed on target date
    intake = sum(
        entry["calories"]
        for entry in calorie_log
        if entry["logged_date"] == target_str
    )

    # Sum exercise calories burned on target date
    burned_exercise = sum(
        entry["calories_burned"]
        for entry in workout_log
        if entry["logged_date"] == target_str
    )

    # BMR represents baseline calories burned just by existing (already in TDEE)
    # Net = calories eaten - total calories burned (TDEE includes activity)
    total_burned = tdee + burned_exercise
    net = intake - total_burned

    return {
        "intake": round(intake, 1),
        "burned_exercise": round(burned_exercise, 1),
        "burned_bmr_and_activity": round(tdee, 1),
        "total_burned": round(total_burned, 1),
        "net": round(net, 1),
        "surplus_deficit": "Surplus" if net > 0 else "Deficit",
    }


def weekly_calorie_summary(calorie_log: list, workout_log: list, tdee: float) -> pd.DataFrame:
    """
    Return a DataFrame with daily calorie balance for the past 7 days.
    Columns: date, intake, burned, net
    """
    today = date.today()
    records = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        balance = daily_calorie_balance(calorie_log, workout_log, tdee, target_date=d)
        records.append({
            "date": d,
            "intake": balance["intake"],
            "burned": balance["total_burned"],
            "net": balance["net"],
        })
    return pd.DataFrame(records)


def calorie_log_to_dataframe(calorie_log: list) -> pd.DataFrame:
    """Convert raw calorie log list to a Pandas DataFrame."""
    if not calorie_log:
        return pd.DataFrame(columns=["id", "meal_name", "meal_type", "calories",
                                     "protein_g", "carbs_g", "fat_g", "logged_date"])
    df = pd.DataFrame(calorie_log)
    df["logged_date"] = pd.to_datetime(df["logged_date"])
    return df


def macro_totals_for_date(calorie_log: list, target_date: date = None) -> dict:
    """Sum macronutrients for a given date."""
    if target_date is None:
        target_date = date.today()
    target_str = str(target_date)

    entries = [e for e in calorie_log if e["logged_date"] == target_str]
    return {
        "calories": round(sum(e["calories"] for e in entries), 1),
        "protein_g": round(sum(e.get("protein_g", 0) for e in entries), 1),
        "carbs_g": round(sum(e.get("carbs_g", 0) for e in entries), 1),
        "fat_g": round(sum(e.get("fat_g", 0) for e in entries), 1),
    }


def estimate_calories_to_goal(current_weight_kg: float, goal_weight_kg: float,
                               weeks: int = 12) -> dict:
    """
    Estimate required daily calorie adjustment to reach goal weight in given weeks.
    1 kg of fat ≈ 7700 kcal.
    """
    weight_diff_kg = current_weight_kg - goal_weight_kg
    total_kcal_needed = weight_diff_kg * 7700  # positive = deficit needed
    days = weeks * 7
    daily_adjustment = total_kcal_needed / days if days > 0 else 0

    return {
        "weight_diff_kg": round(weight_diff_kg, 2),
        "total_kcal": round(total_kcal_needed),
        "days": days,
        "daily_adjustment_kcal": round(daily_adjustment),
        "direction": "deficit" if weight_diff_kg > 0 else "surplus",
    }
