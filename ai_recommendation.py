"""
ai_recommendation.py - AI/ML-based workout and diet recommendations.

Combines rule-based logic with ML predictions to generate personalised advice.
This module is designed to be extended with more sophisticated LLM or ML models later.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Recommendation:
    category: str        # "Workout" | "Diet" | "Lifestyle" | "Health Alert"
    priority: str        # "High" | "Medium" | "Low"
    title: str
    description: str
    action_items: List[str] = field(default_factory=list)
    icon: str = "💡"


# ── Rule-Based Recommendation Engine ──────────────────────────────────────────

def generate_recommendations(
    bmi: float,
    bmi_category: str,
    goal: str,
    activity_level: str,
    weekly_workouts: float,
    avg_daily_calories: float,
    calorie_target: float,
    weight_trend: str,          # "increasing" | "decreasing" | "stable"
    weight_change_kg: float,
    body_fat_pct: float,
    gender: str,
    age: int,
    predicted_weight_30d: float = None,
    current_weight: float = None,
) -> List[Recommendation]:
    """
    Generate a ranked list of personalised recommendations based on health metrics.
    
    Returns a list of Recommendation objects sorted by priority.
    """
    recs: List[Recommendation] = []
    calorie_diff = avg_daily_calories - calorie_target

    # ── BMI-based alerts ──────────────────────────────────────────────────────
    if bmi < 18.5:
        recs.append(Recommendation(
            category="Health Alert",
            priority="High",
            title="You are Underweight",
            description=(
                f"Your BMI of {bmi} indicates you are underweight. "
                "This may increase risk of nutrient deficiencies and bone density loss."
            ),
            action_items=[
                "Increase daily calorie intake by 300–500 kcal",
                "Focus on nutrient-dense foods: nuts, avocados, whole grains, lean protein",
                "Add strength training to build muscle mass",
                "Consult a dietitian or physician",
            ],
            icon="⚠️",
        ))
    elif bmi >= 30:
        recs.append(Recommendation(
            category="Health Alert",
            priority="High",
            title="BMI Indicates Obesity",
            description=(
                f"Your BMI of {bmi} is in the obese range. "
                "This increases risk of cardiovascular disease, diabetes, and other conditions."
            ),
            action_items=[
                "Create a daily calorie deficit of 500–750 kcal",
                "Aim for 150+ minutes of moderate cardio per week",
                "Prioritise whole foods and reduce ultra-processed food intake",
                "Consider consulting a physician or registered dietitian",
            ],
            icon="🚨",
        ))
    elif bmi >= 25:
        recs.append(Recommendation(
            category="Health Alert",
            priority="Medium",
            title="BMI Indicates Overweight",
            description=f"Your BMI of {bmi} is slightly above the healthy range (18.5–24.9).",
            action_items=[
                "Aim for a moderate daily calorie deficit of 300–500 kcal",
                "Add 2–3 cardio sessions per week",
                "Track food intake to improve awareness",
            ],
            icon="⚡",
        ))

    # ── Goal-specific workout recommendations ─────────────────────────────────
    if goal == "Lose weight":
        if weekly_workouts < 3:
            recs.append(Recommendation(
                category="Workout",
                priority="High",
                title="Increase Workout Frequency",
                description=f"You're averaging {weekly_workouts} workouts/week. Aim for 4–5 for effective fat loss.",
                action_items=[
                    "Add 2 cardio sessions (e.g., 30-min brisk walk or cycling)",
                    "Include 1–2 HIIT sessions per week",
                    "Try: Running, Cycling, Jump Rope, Elliptical",
                ],
                icon="🏃",
            ))
        else:
            recs.append(Recommendation(
                category="Workout",
                priority="Low",
                title="Great Workout Consistency!",
                description=f"You're hitting {weekly_workouts} workouts/week — keep it up!",
                action_items=[
                    "Introduce progressive overload in strength training",
                    "Add a weekly active recovery session (yoga or stretching)",
                ],
                icon="✅",
            ))

    elif goal == "Gain muscle":
        recs.append(Recommendation(
            category="Workout",
            priority="High",
            title="Muscle Building Protocol",
            description="Focus on progressive strength training with sufficient recovery.",
            action_items=[
                "Train each muscle group 2× per week",
                "Compound lifts first: squats, deadlifts, bench press, overhead press",
                "Aim for 3–4 sets of 6–12 reps per exercise",
                "Rest 48–72 hours between sessions for the same muscle group",
            ],
            icon="💪",
        ))

    elif goal == "Improve endurance":
        recs.append(Recommendation(
            category="Workout",
            priority="Medium",
            title="Endurance Training Plan",
            description="Progressively increase cardiovascular workload.",
            action_items=[
                "80% of training at low intensity (conversational pace)",
                "20% at high intensity (intervals or tempo runs)",
                "Increase weekly distance/duration by no more than 10% per week",
                "Include cross-training (cycling, swimming) to prevent injury",
            ],
            icon="🚴",
        ))

    # ── Calorie / diet recommendations ────────────────────────────────────────
    if calorie_diff > 300:
        recs.append(Recommendation(
            category="Diet",
            priority="Medium",
            title="Calorie Intake Too High",
            description=f"You're eating ~{abs(int(calorie_diff))} kcal/day above your target.",
            action_items=[
                "Reduce portion sizes — use a food scale",
                "Limit liquid calories (sodas, alcohol, juices)",
                "Prioritise high-volume, low-calorie foods (vegetables, soups)",
                "Eat protein-rich foods to stay satiated longer",
            ],
            icon="🍽️",
        ))
    elif calorie_diff < -300 and goal != "Lose weight":
        recs.append(Recommendation(
            category="Diet",
            priority="Medium",
            title="Calorie Intake Too Low",
            description=f"You're eating ~{abs(int(calorie_diff))} kcal/day below your target.",
            action_items=[
                "Add nutrient-dense snacks: Greek yogurt, nuts, protein shakes",
                "Include healthy fats: avocado, olive oil, fatty fish",
                "Do not skip meals — aim for 3 main meals + 2 snacks",
            ],
            icon="🥗",
        ))

    # ── Protein recommendation ─────────────────────────────────────────────────
    protein_target = 1.6 if goal == "Gain muscle" else 1.2  # g per kg body weight
    if current_weight:
        recs.append(Recommendation(
            category="Diet",
            priority="Medium" if goal == "Gain muscle" else "Low",
            title=f"Protein Target: {round(protein_target * current_weight)} g/day",
            description=f"Recommended: {protein_target} g of protein per kg of body weight for your goal.",
            action_items=[
                "Best sources: chicken breast, tuna, eggs, Greek yogurt, lentils, tofu",
                "Spread protein intake across all meals",
                "Consider a protein supplement if dietary intake is insufficient",
            ],
            icon="🥩",
        ))

    # ── Weight trend-based recommendations ────────────────────────────────────
    if weight_trend == "increasing" and goal == "Lose weight":
        recs.append(Recommendation(
            category="Lifestyle",
            priority="High",
            title="Weight is Trending Up — Adjust Strategy",
            description="Your weight has been increasing despite a weight-loss goal.",
            action_items=[
                "Review calorie log for hidden calories (sauces, oils, drinks)",
                "Ensure you're tracking all meals accurately",
                "Increase NEAT (non-exercise activity): take stairs, walk more",
                "Check sleep quality — poor sleep increases hunger hormones",
            ],
            icon="📈",
        ))
    elif weight_trend == "decreasing" and goal == "Gain muscle":
        recs.append(Recommendation(
            category="Lifestyle",
            priority="Medium",
            title="Weight Decreasing — Eat More",
            description="Your weight is trending down while your goal is muscle gain.",
            action_items=[
                "Add 200–300 kcal to daily intake, primarily from carbs and protein",
                "Time carbohydrates around your workouts for better performance",
            ],
            icon="📉",
        ))

    # ── Sleep and recovery ─────────────────────────────────────────────────────
    if weekly_workouts >= 5:
        recs.append(Recommendation(
            category="Lifestyle",
            priority="Low",
            title="Recovery is Key",
            description="With a high workout frequency, recovery becomes critical.",
            action_items=[
                "Sleep 7–9 hours per night for hormonal balance and muscle repair",
                "Include 1–2 rest days or active recovery days",
                "Consider foam rolling, stretching, or a cold shower post-workout",
                "Monitor heart rate variability (HRV) if possible",
            ],
            icon="😴",
        ))

    # ── Hydration reminder (always shown) ─────────────────────────────────────
    recs.append(Recommendation(
        category="Lifestyle",
        priority="Low",
        title="Stay Hydrated",
        description="Proper hydration supports metabolism, performance, and recovery.",
        action_items=[
            "Drink water first thing in the morning",
            "Carry a reusable water bottle throughout the day",
            "Drink 250–500 ml before each workout and replace sweat losses after",
        ],
        icon="💧",
    ))

    # ── ML-based prediction insight ───────────────────────────────────────────
    if predicted_weight_30d is not None and current_weight is not None:
        diff = round(predicted_weight_30d - current_weight, 2)
        direction = "gain" if diff > 0 else "lose"
        recs.append(Recommendation(
            category="AI Insight",
            priority="Medium",
            title=f"30-Day Weight Forecast: {predicted_weight_30d} kg",
            description=(
                f"Based on your current trajectory, our model predicts you will "
                f"{direction} {abs(diff)} kg over the next 30 days."
            ),
            action_items=[
                "Adjust calorie intake or workout intensity to align with your goal",
                "Log weight daily for more accurate predictions",
                "Review your progress weekly and adjust the plan",
            ],
            icon="🔮",
        ))

    # ── Sort by priority ───────────────────────────────────────────────────────
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: priority_order.get(r.priority, 3))

    return recs


def priority_badge_color(priority: str) -> str:
    return {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}.get(priority, "#6366F1")
