"""
database.py - SQLite connection and table creation for AI Fitness App
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "fitness.db")


def get_connection():
    """Return a SQLite connection to the fitness database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _get_conn():
    """Internal: return open connection (caller must commit/close)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all database tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # User profile table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            weight_kg REAL,
            height_cm REAL,
            activity_level TEXT,
            goal TEXT,
            fitness_level TEXT DEFAULT 'Beginner',
            target_weight REAL DEFAULT 0,
            dob TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrate existing DB: add new columns if missing
    for col, typedef in [
        ("fitness_level", "TEXT DEFAULT 'Beginner'"),
        ("target_weight", "REAL DEFAULT 0"),
        ("dob",           "TEXT DEFAULT ''"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE user_profile ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # column already exists

    # Weight log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            weight_kg REAL NOT NULL,
            logged_date DATE NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Workout log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workout_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            exercise_name TEXT NOT NULL,
            category TEXT,
            duration_minutes INTEGER,
            calories_burned REAL,
            intensity TEXT,
            sets INTEGER DEFAULT NULL,
            reps INTEGER DEFAULT NULL,
            logged_date DATE NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrate existing workout_log: add sets/reps if missing
    for col, typedef in [("sets", "INTEGER DEFAULT NULL"), ("reps", "INTEGER DEFAULT NULL")]:
        try:
            cursor.execute(f"ALTER TABLE workout_log ADD COLUMN {col} {typedef}")
        except Exception:
            pass

    # Calorie log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calorie_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            meal_name TEXT,
            meal_type TEXT,
            calories REAL NOT NULL,
            protein_g REAL DEFAULT 0,
            carbs_g REAL DEFAULT 0,
            fat_g REAL DEFAULT 0,
            logged_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ── User Profile CRUD ──────────────────────────────────────────────────────────

def save_user_profile(name, age, gender, weight_kg, height_cm, activity_level, goal,
                      fitness_level="Beginner", target_weight=0.0, dob=""):
    """Insert or update the single user profile (id=1)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM user_profile WHERE id = 1")
    existing = cursor.fetchone()
    if existing:
        cursor.execute("""
            UPDATE user_profile
            SET name=?, age=?, gender=?, weight_kg=?, height_cm=?,
                activity_level=?, goal=?, fitness_level=?, target_weight=?,
                dob=?, updated_at=CURRENT_TIMESTAMP
            WHERE id = 1
        """, (name, age, gender, weight_kg, height_cm, activity_level, goal,
              fitness_level, target_weight, dob))
    else:
        cursor.execute("""
            INSERT INTO user_profile
                (id, name, age, gender, weight_kg, height_cm, activity_level, goal,
                 fitness_level, target_weight, dob)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, age, gender, weight_kg, height_cm, activity_level, goal,
              fitness_level, target_weight, dob))
    conn.commit()
    conn.close()


def get_user_profile():
    """Fetch the user profile (id=1). Returns dict or None."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profile WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ── Weight Log CRUD ────────────────────────────────────────────────────────────

def log_weight(weight_kg, logged_date, notes=""):
    conn = get_connection()
    conn.execute("""
        INSERT INTO weight_log (weight_kg, logged_date, notes)
        VALUES (?, ?, ?)
    """, (weight_kg, logged_date, notes))
    conn.commit()
    conn.close()


def get_weight_log():
    """Return all weight entries as a list of dicts, newest first."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM weight_log ORDER BY logged_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_weight_entry(entry_id):
    conn = get_connection()
    conn.execute("DELETE FROM weight_log WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()


# ── Workout Log CRUD ───────────────────────────────────────────────────────────

def log_workout(exercise_name, category, duration_minutes, calories_burned,
                intensity, logged_date, notes="", sets=None, reps=None):
    conn = get_connection()
    conn.execute("""
        INSERT INTO workout_log
            (exercise_name, category, duration_minutes, calories_burned, intensity,
             sets, reps, logged_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (exercise_name, category, duration_minutes, calories_burned, intensity,
          sets, reps, logged_date, notes))
    conn.commit()
    conn.close()


def get_workout_log():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM workout_log ORDER BY logged_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_workout_entry(entry_id):
    conn = get_connection()
    conn.execute("DELETE FROM workout_log WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()


# ── Calorie Log CRUD ───────────────────────────────────────────────────────────

def log_calories(meal_name, meal_type, calories, protein_g, carbs_g, fat_g, logged_date):
    conn = get_connection()
    conn.execute("""
        INSERT INTO calorie_log
            (meal_name, meal_type, calories, protein_g, carbs_g, fat_g, logged_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (meal_name, meal_type, calories, protein_g, carbs_g, fat_g, logged_date))
    conn.commit()
    conn.close()


def get_calorie_log():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM calorie_log ORDER BY logged_date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_calorie_entry(entry_id):
    conn = get_connection()
    conn.execute("DELETE FROM calorie_log WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()