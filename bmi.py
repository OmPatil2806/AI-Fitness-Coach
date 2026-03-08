"""
bmi.py - BMI, BMR, body fat %, lean body mass, ideal weight, macro calculations
"""


# ── BMI ────────────────────────────────────────────────────────────────────────

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Body Mass Index = weight(kg) / height(m)^2"""
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal weight"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


def bmi_color(bmi: float) -> str:
    """Return a hex color for visual display."""
    if bmi < 18.5:
        return "#3B82F6"   # blue
    elif bmi < 25.0:
        return "#10B981"   # green
    elif bmi < 30.0:
        return "#F59E0B"   # amber
    else:
        return "#EF4444"   # red


# ── BMR ────────────────────────────────────────────────────────────────────────

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """
    Mifflin-St Jeor equation (most accurate for general population).
    Male:   10*weight + 6.25*height - 5*age + 5
    Female: 10*weight + 6.25*height - 5*age - 161
    """
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return round(base + 5 if gender.lower() == "male" else base - 161, 2)


# ── TDEE ───────────────────────────────────────────────────────────────────────

ACTIVITY_MULTIPLIERS = {
    "Sedentary (little or no exercise)": 1.2,
    "Lightly active (1-3 days/week)": 1.375,
    "Moderately active (3-5 days/week)": 1.55,
    "Very active (6-7 days/week)": 1.725,
    "Super active (physical job + training)": 1.9,
}


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Total Daily Energy Expenditure = BMR × activity multiplier."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)
    return round(bmr * multiplier, 2)


# ── Body Fat % ─────────────────────────────────────────────────────────────────

def calculate_body_fat(bmi: float, age: int, gender: str) -> float:
    """
    Deurenberg formula: Body fat% = 1.20*BMI + 0.23*age - 10.8*sex - 5.4
    sex: male=1, female=0
    """
    sex = 1 if gender.lower() == "male" else 0
    bf = 1.20 * bmi + 0.23 * age - 10.8 * sex - 5.4
    return round(max(bf, 0), 2)


# ── Lean Body Mass ─────────────────────────────────────────────────────────────

def calculate_lean_body_mass(weight_kg: float, body_fat_pct: float) -> float:
    """LBM = weight × (1 - body_fat%)"""
    return round(weight_kg * (1 - body_fat_pct / 100), 2)


# ── Ideal Weight ───────────────────────────────────────────────────────────────

def calculate_ideal_weight(height_cm: float, gender: str) -> dict:
    """
    Three common formulas:
      - Devine (1974)
      - Robinson (1983)
      - Miller (1983)
    Returns dict with each formula's result in kg.
    """
    height_in = height_cm / 2.54
    inches_over_5ft = max(height_in - 60, 0)

    if gender.lower() == "male":
        devine = 50 + 2.3 * inches_over_5ft
        robinson = 52 + 1.9 * inches_over_5ft
        miller = 56.2 + 1.41 * inches_over_5ft
    else:
        devine = 45.5 + 2.3 * inches_over_5ft
        robinson = 49 + 1.7 * inches_over_5ft
        miller = 53.1 + 1.36 * inches_over_5ft

    return {
        "Devine": round(devine, 1),
        "Robinson": round(robinson, 1),
        "Miller": round(miller, 1),
        "Average": round((devine + robinson + miller) / 3, 1),
    }


# ── Macronutrients ─────────────────────────────────────────────────────────────

def calculate_macros(tdee: float, goal: str) -> dict:
    """
    Adjust TDEE for goal, then split into macros.
    Returns: calories target, protein_g, carbs_g, fat_g
    """
    goal_adjustments = {
        "Lose weight": -500,
        "Maintain weight": 0,
        "Gain muscle": +300,
        "Improve endurance": +200,
    }
    calorie_target = tdee + goal_adjustments.get(goal, 0)

    # Macro splits (protein 30%, carbs 40%, fat 30%) — standard balanced split
    protein_cal = calorie_target * 0.30
    carbs_cal = calorie_target * 0.40
    fat_cal = calorie_target * 0.30

    return {
        "calorie_target": round(calorie_target),
        "protein_g": round(protein_cal / 4),   # 4 kcal/g
        "carbs_g": round(carbs_cal / 4),         # 4 kcal/g
        "fat_g": round(fat_cal / 9),             # 9 kcal/g
    }


# ── Water Intake ───────────────────────────────────────────────────────────────

def calculate_water_intake(weight_kg: float, activity_level: str) -> float:
    """
    Base: 35 ml per kg of body weight.
    Add extra for active individuals.
    Returns litres/day.
    """
    base_ml = weight_kg * 35
    activity_bonus = {
        "Sedentary (little or no exercise)": 0,
        "Lightly active (1-3 days/week)": 350,
        "Moderately active (3-5 days/week)": 500,
        "Very active (6-7 days/week)": 700,
        "Super active (physical job + training)": 1000,
    }
    bonus = activity_bonus.get(activity_level, 0)
    return round((base_ml + bonus) / 1000, 2)
