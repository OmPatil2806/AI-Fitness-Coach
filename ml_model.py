"""
ml_model.py - Machine learning models for weight prediction and calorie burn prediction.

Models:
  1. WeightPredictor   - predicts future weight from historical weight + calorie data
  2. CalorieBurnPredictor - predicts calories burned from exercise, duration, weight
"""

import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

import joblib
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

WEIGHT_MODEL_PATH = os.path.join(MODELS_DIR, "weight_predictor.joblib")
CALORIE_MODEL_PATH = os.path.join(MODELS_DIR, "calorie_burn_predictor.joblib")


# ── Synthetic Data Generation (for demo / cold-start) ─────────────────────────

def generate_synthetic_weight_data(start_weight=82.0, days=90) -> pd.DataFrame:
    """
    Generate realistic synthetic weight data with a gradual downward trend,
    noise, and weekly fluctuations.
    """
    np.random.seed(42)
    dates = [date.today() - timedelta(days=days - i) for i in range(days)]
    trend = np.linspace(0, -3.5, days)            # lose ~3.5 kg over 90 days
    noise = np.random.normal(0, 0.3, days)        # daily noise
    weekly = 0.4 * np.sin(np.arange(days) * 2 * np.pi / 7)  # water retention cycle

    weights = start_weight + trend + noise + weekly
    avg_calories = 2100 + np.random.normal(0, 150, days)
    workouts = np.random.randint(0, 2, days)
    calories_burned = workouts * np.random.randint(200, 450, days)

    df = pd.DataFrame({
        "date": dates,
        "weight_kg": np.round(weights, 2),
        "avg_calories_eaten": np.round(avg_calories, 0),
        "calories_burned": np.round(calories_burned, 0),
        "workouts_today": workouts,
        "day_index": np.arange(days),
    })
    return df


def generate_synthetic_calorie_burn_data(n=500) -> pd.DataFrame:
    """
    Generate synthetic calorie burn dataset for model training.
    Features: duration_minutes, weight_kg, met_value
    Target: calories_burned
    """
    np.random.seed(0)
    duration = np.random.randint(15, 90, n)
    weight = np.random.uniform(55, 120, n)
    met = np.random.choice([3.0, 4.0, 5.0, 6.5, 7.0, 8.0, 9.8, 11.0], n)
    # True formula with small noise
    calories = met * weight * (duration / 60) + np.random.normal(0, 10, n)

    return pd.DataFrame({
        "duration_minutes": duration,
        "weight_kg": weight,
        "met_value": met,
        "calories_burned": np.round(calories, 1),
    })


# ── Weight Predictor ───────────────────────────────────────────────────────────

class WeightPredictor:
    """
    Predicts future weight using:
      - day index (time feature)
      - cumulative calorie surplus/deficit
      - workout frequency
    Uses Ridge regression (simple, interpretable, good for small datasets).
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ])
        self.trained = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        return df[["day_index", "avg_calories_eaten", "calories_burned", "workouts_today"]].values

    def train(self, df: pd.DataFrame) -> dict:
        """Train on historical weight data. Returns evaluation metrics."""
        X = self._build_features(df)
        y = df["weight_kg"].values

        if len(X) < 10:
            # Not enough data — fit on all
            self.model.fit(X, y)
            self.trained = True
            return {"mae": None, "r2": None, "note": "Trained on all data (< 10 samples)"}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = round(mean_absolute_error(y_test, preds), 3)
        r2 = round(r2_score(y_test, preds), 3)
        self.trained = True
        return {"mae": mae, "r2": r2}

    def predict_future(self, df: pd.DataFrame, days_ahead: int = 30,
                       daily_calories: float = 2000, daily_workout_cal: float = 300) -> pd.DataFrame:
        """Predict weight for the next N days."""
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")

        last_day_idx = df["day_index"].max()
        future_rows = []
        for i in range(1, days_ahead + 1):
            future_rows.append({
                "day_index": last_day_idx + i,
                "avg_calories_eaten": daily_calories,
                "calories_burned": daily_workout_cal,
                "workouts_today": 1 if i % 2 == 0 else 0,
            })

        future_df = pd.DataFrame(future_rows)
        X_future = future_df[["day_index", "avg_calories_eaten", "calories_burned", "workouts_today"]].values
        preds = self.model.predict(X_future)

        future_dates = [date.today() + timedelta(days=i) for i in range(1, days_ahead + 1)]
        return pd.DataFrame({"date": future_dates, "predicted_weight_kg": np.round(preds, 2)})

    def save(self):
        joblib.dump(self, WEIGHT_MODEL_PATH)

    @classmethod
    def load(cls):
        if os.path.exists(WEIGHT_MODEL_PATH):
            return joblib.load(WEIGHT_MODEL_PATH)
        return None


# ── Calorie Burn Predictor ─────────────────────────────────────────────────────

class CalorieBurnPredictor:
    """
    Predicts calories burned from:
      - duration_minutes
      - weight_kg
      - met_value (exercise intensity proxy)
    Uses GradientBoostingRegressor for better accuracy.
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ])
        self.trained = False

    def train(self, df: pd.DataFrame) -> dict:
        X = df[["duration_minutes", "weight_kg", "met_value"]].values
        y = df["calories_burned"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = round(mean_absolute_error(y_test, preds), 2)
        r2 = round(r2_score(y_test, preds), 3)
        self.trained = True
        return {"mae": mae, "r2": r2}

    def predict(self, duration_minutes: float, weight_kg: float, met_value: float) -> float:
        """Predict calories burned for a single workout."""
        if not self.trained:
            raise RuntimeError("Model not trained.")
        X = np.array([[duration_minutes, weight_kg, met_value]])
        return round(float(self.model.predict(X)[0]), 1)

    def save(self):
        joblib.dump(self, CALORIE_MODEL_PATH)

    @classmethod
    def load(cls):
        if os.path.exists(CALORIE_MODEL_PATH):
            return joblib.load(CALORIE_MODEL_PATH)
        return None


# ── Convenience: train & save both models ─────────────────────────────────────

def train_and_save_all_models(weight_df: pd.DataFrame = None) -> dict:
    """
    Train both models (using provided or synthetic data) and save to disk.
    Returns a dict with training metrics.
    """
    results = {}

    # Weight predictor
    wp = WeightPredictor()
    if weight_df is None or len(weight_df) < 5:
        weight_df = generate_synthetic_weight_data()
    metrics_w = wp.train(weight_df)
    wp.save()
    results["weight_predictor"] = metrics_w

    # Calorie burn predictor
    cb = CalorieBurnPredictor()
    cal_df = generate_synthetic_calorie_burn_data()
    metrics_c = cb.train(cal_df)
    cb.save()
    results["calorie_burn_predictor"] = metrics_c

    return results


def load_or_train_models():
    """Load models from disk; train fresh if not found."""
    wp = WeightPredictor.load()
    cb = CalorieBurnPredictor.load()

    if wp is None or cb is None:
        train_and_save_all_models()
        wp = WeightPredictor.load()
        cb = CalorieBurnPredictor.load()

    return wp, cb
