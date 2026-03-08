"""
app.py - Main Streamlit app for AI Fitness Coach & Health Analytics App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

# ── Local modules ──────────────────────────────────────────────────────────────
import database as db
import bmi as bmi_module
import calorie_calculator as cc
import workout_tracker as wt
import data_analysis as da
import charts
import ml_model
import ai_recommendation as ai_rec

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark gradient background */
  .stApp { background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%); }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    border-right: 1px solid #334155;
  }
  section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
  }
  [data-testid="stMetricValue"] { color: #6366F1 !important; font-size: 2rem !important; }
  [data-testid="stMetricLabel"] { color: #94A3B8 !important; }
  [data-testid="stMetricDelta"] { font-weight: 600; }

  /* Section headers */
  h1, h2, h3 { color: #E2E8F0 !important; }

  /* Cards */
  .fitness-card {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px;
    margin: 8px 0;
  }

  /* Rec card */
  .rec-card {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border-left: 4px solid #6366F1;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
  }

  /* Inputs */
  .stTextInput input, .stNumberInput input, .stSelectbox select {
    background: #1E293B !important;
    border: 1px solid #475569 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s;
  }
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(99,102,241,0.4); }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #94A3B8 !important; }
  .stTabs [aria-selected="true"] { color: #6366F1 !important; border-bottom-color: #6366F1 !important; }

  /* Divider */
  hr { border-color: #334155; }
</style>
""", unsafe_allow_html=True)


# ── Initialise database ────────────────────────────────────────────────────────
db.init_db()


# ── Load / cache ML models ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def get_models():
    return ml_model.load_or_train_models()

weight_predictor, calorie_predictor = get_models()


# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:

    # ── App branding ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
      <div style='font-size:3.5rem; filter: drop-shadow(0 0 12px #6366F1);'>🏋️</div>
      <h2 style='margin:6px 0 2px; background: linear-gradient(135deg,#6366F1,#EC4899);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 font-size:1.4rem; letter-spacing:0.5px;'>
        AI Fitness Coach
      </h2>
      <p style='color:#64748B; font-size:0.75rem; margin:0; letter-spacing:1px;
                text-transform:uppercase;'>Health Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ── App feature highlights ──────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                border:1px solid #334155; border-radius:12px;
                padding:14px 16px; margin: 8px 0;'>
      <p style='color:#6366F1; font-size:0.72rem; font-weight:700;
                letter-spacing:1px; text-transform:uppercase; margin:0 0 10px;'>
        ✦ What This App Does
      </p>
      <div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>
        <span style='font-size:1rem'>📊</span>
        <span style='color:#CBD5E1; font-size:0.8rem;'>Real-time Health Metrics & BMI</span>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>
        <span style='font-size:1rem'>🏋️</span>
        <span style='color:#CBD5E1; font-size:0.8rem;'>Smart Workout & Calorie Tracker</span>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>
        <span style='font-size:1rem'>🔮</span>
        <span style='color:#CBD5E1; font-size:0.8rem;'>ML-Powered Weight Predictions</span>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>
        <span style='font-size:1rem'>🤖</span>
        <span style='color:#CBD5E1; font-size:0.8rem;'>AI Personalised Recommendations</span>
      </div>
      <div style='display:flex; align-items:center; gap:8px; margin:6px 0;'>
        <span style='font-size:1rem'>📈</span>
        <span style='color:#CBD5E1; font-size:0.8rem;'>Visual Progress Dashboard</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Navigation ──────────────────────────────────────────────────────────────
    st.markdown("<p style='color:#6366F1; font-size:0.72rem; font-weight:700; "
                "letter-spacing:1px; text-transform:uppercase; margin:0 0 6px;'>"
                "✦ Navigation</p>", unsafe_allow_html=True)

    PAGE = st.radio(
        "Navigate",
        ["🏠 Dashboard", "👤 My Profile", "⚖️ Health Metrics",
         "🏋️ Workout Tracker", "🍽️ Calorie Tracker",
         "🧬 Body Intelligence", "🔮 Predictions & AI",
         "💬 AI Coach"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Motivation quote ────────────────────────────────────────────────────────
    import random
    quotes = [
        ("The body achieves what the mind believes.", "Napoleon Hill"),
        ("Take care of your body. It's the only place you have to live.", "Jim Rohn"),
        ("Strength does not come from the body. It comes from the will of the soul.", "Gandhi"),
        ("The pain you feel today will be the strength you feel tomorrow.", "Arnold Schwarzenegger"),
        ("Success is the sum of small efforts repeated day in and day out.", "Robert Collier"),
        ("Don't wish for it. Work for it.", "Unknown"),
        ("Your only limit is you.", "Unknown"),
        ("Push yourself because no one else is going to do it for you.", "Unknown"),
    ]
    # Use day of year to keep quote consistent per day (not random on every rerun)
    quote_text, quote_author = quotes[date.today().timetuple().tm_yday % len(quotes)]

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1E1B4B, #0F172A);
                border: 1px solid #4338CA;
                border-left: 4px solid #6366F1;
                border-radius: 10px; padding: 14px 16px; margin: 4px 0;'>
      <p style='color:#A5B4FC; font-size:0.72rem; font-weight:700;
                letter-spacing:1px; text-transform:uppercase; margin:0 0 8px;'>
        💡 Daily Motivation
      </p>
      <p style='color:#E2E8F0; font-size:0.85rem; font-style:italic;
                line-height:1.5; margin:0 0 6px;'>
        "{quote_text}"
      </p>
      <p style='color:#6366F1; font-size:0.75rem; font-weight:600; margin:0;'>
        — {quote_author}
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Welcome / Profile summary ───────────────────────────────────────────────
    profile = db.get_user_profile()
    if profile:
        # Compute quick stats for sidebar
        _bmi = bmi_module.calculate_bmi(profile["weight_kg"], profile["height_cm"])
        _bmi_cat = bmi_module.bmi_category(_bmi)
        _bmi_color = bmi_module.bmi_color(_bmi)
        _bmr = bmi_module.calculate_bmr(profile["weight_kg"], profile["height_cm"],
                                         profile["age"], profile["gender"])
        _tdee = bmi_module.calculate_tdee(_bmr, profile["activity_level"])

        # Goal emoji map
        goal_icons = {
            "Lose weight": "🔥",
            "Gain muscle": "💪",
            "Maintain weight": "⚖️",
            "Improve endurance": "🚴",
        }
        goal_icon = goal_icons.get(profile["goal"], "🎯")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-radius:12px; padding:16px;'>

          <!-- Avatar + Name -->
          <div style='text-align:center; margin-bottom:12px;'>
            <div style='font-size:2.5rem; margin-bottom:4px;'>
              {"👨" if profile["gender"] == "Male" else "👩"}‍💪
            </div>
            <p style='color:#E2E8F0; font-weight:700; font-size:1rem; margin:0;'>
              {profile["name"]}
            </p>
            <p style='color:#64748B; font-size:0.75rem; margin:2px 0 0;'>
              {profile["age"]} yrs • {profile["gender"]} • {profile["weight_kg"]} kg
            </p>
          </div>

          <!-- Stats row -->
          <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px;'>
            <div style='background:#0F172A; border-radius:8px; padding:8px; text-align:center;'>
              <div style='color:{_bmi_color}; font-size:1.1rem; font-weight:700;'>{_bmi}</div>
              <div style='color:#64748B; font-size:0.68rem;'>BMI</div>
            </div>
            <div style='background:#0F172A; border-radius:8px; padding:8px; text-align:center;'>
              <div style='color:#10B981; font-size:1.1rem; font-weight:700;'>{int(_tdee)}</div>
              <div style='color:#64748B; font-size:0.68rem;'>TDEE kcal</div>
            </div>
          </div>

          <!-- BMI badge -->
          <div style='text-align:center; margin-bottom:10px;'>
            <span style='background:{_bmi_color}22; color:{_bmi_color};
                         border:1px solid {_bmi_color}55;
                         font-size:0.72rem; font-weight:600;
                         padding:3px 10px; border-radius:20px;'>
              {_bmi_cat}
            </span>
          </div>

          <!-- Goal -->
          <div style='background:#0F172A; border-radius:8px; padding:8px 12px;
                      display:flex; align-items:center; gap:8px;'>
            <span style='font-size:1.1rem;'>{goal_icon}</span>
            <div>
              <div style='color:#64748B; font-size:0.65rem; text-transform:uppercase;
                          letter-spacing:0.5px;'>Current Goal</div>
              <div style='color:#E2E8F0; font-size:0.82rem; font-weight:600;'>
                {profile["goal"]}
              </div>
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px dashed #475569; border-radius:12px;
                    padding:20px; text-align:center;'>
          <div style='font-size:2rem; margin-bottom:8px;'>👤</div>
          <p style='color:#E2E8F0; font-weight:600; font-size:0.9rem; margin:0 0 4px;'>
            Welcome, New User!
          </p>
          <p style='color:#64748B; font-size:0.78rem; margin:0 0 12px; line-height:1.4;'>
            Set up your profile to unlock personalised health metrics, AI recommendations,
            and your fitness dashboard.
          </p>
          <p style='color:#6366F1; font-size:0.78rem; font-weight:600; margin:0;'>
            👆 Go to "My Profile" to get started!
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── App version footer ──────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 4px;'>
      <p style='color:#1E293B; font-size:0.7rem; margin:0;
                background:linear-gradient(135deg,#6366F1,#EC4899);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                font-weight:600;'>
        AI Fitness Coach v1.0
      </p>
      <p style='color:#334155; font-size:0.65rem; margin:2px 0 0;'>
        Powered by ML & Streamlit
      </p>
    </div>
    """, unsafe_allow_html=True)


# ── Helper: compute all health metrics ────────────────────────────────────────
def get_health_metrics(profile):
    """Return dict of computed health metrics from profile."""
    if not profile:
        return None
    bmi_val = bmi_module.calculate_bmi(profile["weight_kg"], profile["height_cm"])
    bmr = bmi_module.calculate_bmr(profile["weight_kg"], profile["height_cm"],
                                    profile["age"], profile["gender"])
    tdee = bmi_module.calculate_tdee(bmr, profile["activity_level"])
    body_fat = bmi_module.calculate_body_fat(bmi_val, profile["age"], profile["gender"])
    lbm = bmi_module.calculate_lean_body_mass(profile["weight_kg"], body_fat)
    ideal = bmi_module.calculate_ideal_weight(profile["height_cm"], profile["gender"])
    macros = bmi_module.calculate_macros(tdee, profile["goal"])
    water = bmi_module.calculate_water_intake(profile["weight_kg"], profile["activity_level"])
    return dict(
        bmi=bmi_val, bmi_cat=bmi_module.bmi_category(bmi_val),
        bmi_color=bmi_module.bmi_color(bmi_val),
        bmr=bmr, tdee=tdee, body_fat=body_fat, lbm=lbm,
        ideal=ideal, macros=macros, water=water,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠 Dashboard":

    profile = db.get_user_profile()
    if not profile:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px;'>
          <div style='font-size:4rem'>🏋️</div>
          <h2 style='color:#E2E8F0'>Welcome to AI Fitness Coach!</h2>
          <p style='color:#64748B; font-size:1.1rem'>Set up your profile in the sidebar to unlock your personal dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    metrics        = get_health_metrics(profile)
    weight_log     = db.get_weight_log()
    workout_log    = db.get_workout_log()
    calorie_log    = db.get_calorie_log()
    today_str      = str(date.today())
    today_balance  = cc.daily_calorie_balance(calorie_log, workout_log, metrics["tdee"])
    today_macros   = cc.macro_totals_for_date(calorie_log)
    wk_summary     = wt.weekly_workout_summary(workout_log)
    weight_info    = da.weight_trend_analysis(weight_log)

    # ── Helper functions ───────────────────────────────────────────────────────
    def calc_streak(workout_log):
        if not workout_log:
            return 0
        dates = sorted({entry["logged_date"] for entry in workout_log}, reverse=True)
        streak = 0
        check = date.today()
        for d in dates:
            if str(check) == d:
                streak += 1
                check -= timedelta(days=1)
            elif str(check - timedelta(days=1)) == d:
                check -= timedelta(days=1)
                streak += 1
                check -= timedelta(days=1)
            else:
                break
        return streak

    def weekly_goal_progress(workout_log, goal_sessions=5):
        cutoff = str(date.today() - timedelta(days=7))
        done = len([w for w in workout_log if w["logged_date"] >= cutoff])
        return min(done, goal_sessions), goal_sessions

    def get_personal_records(workout_log):
        if not workout_log:
            return {}
        df = pd.DataFrame(workout_log)
        pr_cal  = df.loc[df["calories_burned"].idxmax()] if not df.empty else None
        pr_dur  = df.loc[df["duration_minutes"].idxmax()] if not df.empty else None
        return {"best_cal": pr_cal, "best_dur": pr_dur}

    streak          = calc_streak(workout_log)
    done_sessions, goal_sessions = weekly_goal_progress(workout_log)
    prs             = get_personal_records(workout_log)
    net_cal         = today_balance["net"]
    calorie_target  = metrics["macros"]["calorie_target"]
    intake_today    = today_balance["intake"]
    tdee            = metrics["tdee"]

    # ── MOTIVATIONAL BANNER ────────────────────────────────────────────────────
    hour = date.today().timetuple().tm_hour if hasattr(date.today(), 'timetuple') else 10
    import datetime as _dt
    _hour = _dt.datetime.now().hour
    if _hour < 12:
        greeting = "Good Morning"
        greet_icon = "🌅"
    elif _hour < 17:
        greeting = "Good Afternoon"
        greet_icon = "☀️"
    else:
        greeting = "Good Evening"
        greet_icon = "🌙"

    # Dynamic message based on performance
    if streak >= 5:
        banner_msg = f"🔥 You're on a {streak}-day streak! Unstoppable!"
        banner_color = "#10B981"
    elif done_sessions >= goal_sessions:
        banner_msg = "🏆 Weekly workout goal CRUSHED! You're a champion!"
        banner_color = "#6366F1"
    elif intake_today == 0:
        banner_msg = "📝 Don't forget to log your meals today!"
        banner_color = "#F59E0B"
    elif net_cal > 300:
        banner_msg = "⚠️ You're in a calorie surplus today. Consider a workout!"
        banner_color = "#EF4444"
    else:
        banner_msg = "💪 Keep going! Every rep, every meal — it all counts."
        banner_color = "#6366F1"

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {banner_color}22, #0F172A);
                border: 1px solid {banner_color}55;
                border-left: 5px solid {banner_color};
                border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;
                display:flex; align-items:center; justify-content:space-between;'>
      <div>
        <p style='color:#94A3B8; font-size:0.8rem; margin:0 0 4px;
                  text-transform:uppercase; letter-spacing:1px;'>
          {greet_icon} {greeting}, {profile["name"]}!
        </p>
        <p style='color:#E2E8F0; font-size:1.1rem; font-weight:700; margin:0;'>
          {banner_msg}
        </p>
      </div>
      <div style='text-align:right;'>
        <p style='color:#64748B; font-size:0.8rem; margin:0;'>
          {date.today().strftime("%A, %B %d %Y")}
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ROW 1: TOP KPI STRIP ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("⚖️ Weight", f"{profile['weight_kg']} kg",
                  delta=f"{weight_info.get('change', 0):+.1f} kg" if weight_info else None)
    with c2:
        st.metric("📊 BMI", metrics["bmi"], metrics["bmi_cat"])
    with c3:
        st.metric("🔥 TDEE", f"{int(tdee)} kcal")
    with c4:
        st.metric("🍽️ Net Calories", f"{int(net_cal)} kcal",
                  delta="Surplus" if net_cal > 0 else "Deficit")
    with c5:
        st.metric("🏋️ Workouts (7d)", wk_summary["total_sessions"])
    with c6:
        st.metric("🔥 Streak", f"{streak} days", delta="Keep it up! 💪" if streak > 0 else "Start today!")

    st.divider()

    # ── ROW 2: STREAK + WEEKLY GOAL + CALORIE METER + BODY FAT ────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:12px;'>⚡ Live Stats</h3>", unsafe_allow_html=True)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    # 1. Streak Counter card
    streak_pct = min(streak / 30 * 100, 100)
    with r2c1:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-radius:14px; padding:18px; text-align:center;'>
          <div style='font-size:2.2rem; margin-bottom:6px;'>🔥</div>
          <div style='color:#F59E0B; font-size:2rem; font-weight:800; line-height:1;'>{streak}</div>
          <div style='color:#94A3B8; font-size:0.8rem; margin:4px 0 10px;'>Day Streak</div>
          <div style='background:#0F172A; border-radius:20px; height:6px; overflow:hidden;'>
            <div style='background:linear-gradient(90deg,#F59E0B,#EF4444);
                        width:{streak_pct}%; height:100%; border-radius:20px;'></div>
          </div>
          <div style='color:#64748B; font-size:0.7rem; margin-top:6px;'>
            {"🏆 30-day goal!" if streak >= 30 else f"{30 - streak} days to 30-day goal"}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Weekly Goal Progress
    goal_pct = round(done_sessions / goal_sessions * 100)
    circles  = "".join(["🟢" if i < done_sessions else "⚪" for i in range(goal_sessions)])
    with r2c2:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-radius:14px; padding:18px; text-align:center;'>
          <div style='font-size:2.2rem; margin-bottom:6px;'>🎯</div>
          <div style='color:#6366F1; font-size:2rem; font-weight:800; line-height:1;'>{goal_pct}%</div>
          <div style='color:#94A3B8; font-size:0.8rem; margin:4px 0 10px;'>Weekly Goal</div>
          <div style='background:#0F172A; border-radius:20px; height:6px; overflow:hidden;'>
            <div style='background:linear-gradient(90deg,#6366F1,#8B5CF6);
                        width:{goal_pct}%; height:100%; border-radius:20px;'></div>
          </div>
          <div style='color:#64748B; font-size:0.75rem; margin-top:8px; letter-spacing:2px;'>
            {circles}
          </div>
          <div style='color:#64748B; font-size:0.7rem; margin-top:4px;'>
            {done_sessions}/{goal_sessions} sessions
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Calorie Deficit/Surplus Meter
    cal_pct  = min(round(intake_today / calorie_target * 100) if calorie_target > 0 else 0, 120)
    cal_color = "#10B981" if cal_pct <= 100 else "#EF4444"
    with r2c3:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-radius:14px; padding:18px; text-align:center;'>
          <div style='font-size:2.2rem; margin-bottom:6px;'>🍽️</div>
          <div style='color:{cal_color}; font-size:2rem; font-weight:800; line-height:1;'>{int(intake_today)}</div>
          <div style='color:#94A3B8; font-size:0.8rem; margin:4px 0 10px;'>kcal eaten today</div>
          <div style='background:#0F172A; border-radius:20px; height:6px; overflow:hidden;'>
            <div style='background:linear-gradient(90deg,#10B981,{cal_color});
                        width:{min(cal_pct,100)}%; height:100%; border-radius:20px;'></div>
          </div>
          <div style='color:#64748B; font-size:0.7rem; margin-top:6px;'>
            {cal_pct}% of {calorie_target} kcal target
          </div>
          <div style='margin-top:6px;'>
            <span style='background:{"#10B98122" if net_cal <= 0 else "#EF444422"};
                         color:{"#10B981" if net_cal <= 0 else "#EF4444"};
                         border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:600;'>
              {"✅ Deficit" if net_cal <= 0 else f"⚠️ +{int(net_cal)} surplus"}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Body Fat % Card
    bf = metrics["body_fat"]
    gender = profile["gender"]
    if gender == "Male":
        bf_ranges = [(6, 13, "Athletic", "#3B82F6"), (14, 17, "Fit", "#10B981"),
                     (18, 24, "Average", "#F59E0B"), (25, 100, "Above Avg", "#EF4444")]
    else:
        bf_ranges = [(14, 20, "Athletic", "#3B82F6"), (21, 24, "Fit", "#10B981"),
                     (25, 31, "Average", "#F59E0B"), (32, 100, "Above Avg", "#EF4444")]
    bf_label, bf_color = "High", "#EF4444"
    for lo, hi, label, color in bf_ranges:
        if lo <= bf <= hi:
            bf_label, bf_color = label, color
            break
    bf_pct_bar = min(bf / 50 * 100, 100)
    with r2c4:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-radius:14px; padding:18px; text-align:center;'>
          <div style='font-size:2.2rem; margin-bottom:6px;'>💪</div>
          <div style='color:{bf_color}; font-size:2rem; font-weight:800; line-height:1;'>{bf}%</div>
          <div style='color:#94A3B8; font-size:0.8rem; margin:4px 0 10px;'>Body Fat</div>
          <div style='background:#0F172A; border-radius:20px; height:6px; overflow:hidden;'>
            <div style='background:linear-gradient(90deg,#3B82F6,{bf_color});
                        width:{bf_pct_bar}%; height:100%; border-radius:20px;'></div>
          </div>
          <div style='margin-top:8px;'>
            <span style='background:{bf_color}22; color:{bf_color};
                         border:1px solid {bf_color}55;
                         border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:600;'>
              {bf_label}
            </span>
          </div>
          <div style='color:#64748B; font-size:0.7rem; margin-top:6px;'>
            LBM: {metrics["lbm"]} kg
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── ROW 3: PROGRESS RING + WEIGHT BANNER + MACRO BARS ─────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:12px;'>📊 Today's Progress</h3>", unsafe_allow_html=True)
    pr1, pr2 = st.columns([1, 2])

    with pr1:
        # Circular progress ring (SVG-based)
        ring_pct  = min(cal_pct, 100)
        radius    = 54
        circ      = 2 * 3.14159 * radius
        dash_val  = round(circ * ring_pct / 100, 1)
        ring_color = "#10B981" if ring_pct <= 85 else "#F59E0B" if ring_pct <= 100 else "#EF4444"
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-radius:14px; padding:20px; text-align:center;'>
          <p style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase;
                    letter-spacing:1px; margin:0 0 12px;'>🔄 Calorie Ring</p>
          <svg width="140" height="140" viewBox="0 0 140 140">
            <circle cx="70" cy="70" r="{radius}" fill="none" stroke="#1E293B" stroke-width="14"/>
            <circle cx="70" cy="70" r="{radius}" fill="none" stroke="{ring_color}" stroke-width="14"
              stroke-dasharray="{dash_val} {circ}" stroke-dashoffset="{circ * 0.25}"
              stroke-linecap="round" transform="rotate(-90 70 70)"/>
            <text x="70" y="65" text-anchor="middle" fill="{ring_color}"
                  font-size="20" font-weight="bold">{int(intake_today)}</text>
            <text x="70" y="82" text-anchor="middle" fill="#64748B" font-size="11">kcal</text>
            <text x="70" y="96" text-anchor="middle" fill="#475569" font-size="10">of {calorie_target}</text>
          </svg>
          <div style='margin-top:8px;'>
            <span style='color:{ring_color}; font-size:1.2rem; font-weight:700;'>{ring_pct}%</span>
            <span style='color:#64748B; font-size:0.8rem;'> of daily goal</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with pr2:
        # Macro progress bars — build HTML first, then pass to st.markdown
        mac = metrics["macros"]

        def macro_bar(label, icon, current, target, color):
            pct = min(round(current / target * 100) if target > 0 else 0, 100)
            html = (
                "<div style='margin-bottom:14px;'>"
                "<div style='display:flex; justify-content:space-between; margin-bottom:5px;'>"
                "<span style='color:#CBD5E1; font-size:0.85rem;'>" + icon + " " + label + "</span>"
                "<span style='color:" + color + "; font-size:0.85rem; font-weight:600;'>"
                + str(current) + "g / " + str(target) + "g (" + str(pct) + "%)"
                "</span></div>"
                "<div style='background:#0F172A; border-radius:20px; height:8px; overflow:hidden;'>"
                "<div style='background:linear-gradient(90deg," + color + "88," + color + ");"
                " width:" + str(pct) + "%; height:100%; border-radius:20px;'></div>"
                "</div></div>"
            )
            return html

        # Pre-build all macro bars as plain strings
        protein_bar = macro_bar("Protein",       "🥩", today_macros["protein_g"], mac["protein_g"], "#6366F1")
        carbs_bar   = macro_bar("Carbohydrates", "🍚", today_macros["carbs_g"],   mac["carbs_g"],   "#F59E0B")
        fat_bar     = macro_bar("Fat",           "🧈", today_macros["fat_g"],     mac["fat_g"],     "#10B981")
        water_val   = metrics["water"]

        macro_html = (
            "<div style='background:linear-gradient(135deg,#1E293B,#0F172A);"
            "border:1px solid #334155; border-radius:14px; padding:20px;'>"
            "<p style='color:#94A3B8; font-size:0.75rem; text-transform:uppercase;"
            "letter-spacing:1px; margin:0 0 16px;'>🥗 Macro Progress Bars</p>"
            + protein_bar
            + carbs_bar
            + fat_bar +
            "<div style='margin-top:6px; padding-top:12px; border-top:1px solid #334155;"
            "display:flex; justify-content:space-between;'>"
            "<span style='color:#64748B; font-size:0.78rem;'>💧 Water Target</span>"
            "<span style='color:#3B82F6; font-size:0.78rem; font-weight:600;'>"
            + str(water_val) + " L/day</span></div></div>"
        )

        st.markdown(macro_html, unsafe_allow_html=True)

    st.divider()

    # ── ROW 4: BEFORE/AFTER WEIGHT BANNER ─────────────────────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:12px;'>🎯 Weight Journey</h3>", unsafe_allow_html=True)

    start_w   = weight_info.get("start_weight", profile["weight_kg"]) if weight_info else profile["weight_kg"]
    current_w = weight_info.get("current_weight", profile["weight_kg"]) if weight_info else profile["weight_kg"]
    ideal_w   = metrics["ideal"]["Average"]
    total_change = round(current_w - start_w, 2)
    to_go     = round(current_w - ideal_w, 2)
    if start_w != ideal_w:
        journey_pct = max(0, min(round((start_w - current_w) / (start_w - ideal_w) * 100) if start_w != ideal_w else 0, 100))
    else:
        journey_pct = 100

    change_color = "#10B981" if total_change <= 0 else "#EF4444"
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1E1B4B,#0F172A);
                border:1px solid #4338CA; border-radius:14px; padding:22px;'>
      <div style='display:grid; grid-template-columns:1fr auto 1fr auto 1fr; gap:10px;
                  align-items:center; text-align:center; margin-bottom:18px;'>
        <div>
          <div style='color:#64748B; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;'>
            🏁 Start Weight
          </div>
          <div style='color:#94A3B8; font-size:1.6rem; font-weight:800;'>{start_w} kg</div>
        </div>
        <div style='color:#334155; font-size:1.5rem;'>→</div>
        <div>
          <div style='color:#64748B; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;'>
            📍 Current
          </div>
          <div style='color:#6366F1; font-size:1.6rem; font-weight:800;'>{current_w} kg</div>
          <div style='color:{change_color}; font-size:0.8rem; font-weight:600;'>
            {total_change:+.2f} kg
          </div>
        </div>
        <div style='color:#334155; font-size:1.5rem;'>→</div>
        <div>
          <div style='color:#64748B; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;'>
            🏆 Ideal Weight
          </div>
          <div style='color:#10B981; font-size:1.6rem; font-weight:800;'>{ideal_w} kg</div>
          <div style='color:#64748B; font-size:0.75rem;'>{abs(to_go):.1f} kg to go</div>
        </div>
      </div>
      <div style='background:#0F172A; border-radius:20px; height:10px; overflow:hidden;'>
        <div style='background:linear-gradient(90deg,#6366F1,#10B981);
                    width:{journey_pct}%; height:100%; border-radius:20px;'></div>
      </div>
      <div style='display:flex; justify-content:space-between; margin-top:6px;'>
        <span style='color:#64748B; font-size:0.7rem;'>Journey Start</span>
        <span style='color:#6366F1; font-size:0.75rem; font-weight:600;'>{journey_pct}% to ideal weight</span>
        <span style='color:#10B981; font-size:0.7rem;'>Goal Reached 🏆</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── ROW 5: MONTHLY WORKOUT HEATMAP ────────────────────────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:12px;'>📅 Monthly Workout Heatmap</h3>", unsafe_allow_html=True)

    # Build 35-day grid (5 weeks)
    today_d = date.today()
    workout_dates = {entry["logged_date"] for entry in workout_log}
    calorie_dates = {entry["logged_date"] for entry in calorie_log}
    weight_dates  = {entry["logged_date"] for entry in weight_log}

    # Start from 34 days ago (5 weeks)
    grid_start = today_d - timedelta(days=34)
    weeks = [[] for _ in range(5)]
    for i in range(35):
        d = grid_start + timedelta(days=i)
        ds = str(d)
        week_idx = i // 7
        if ds in workout_dates:
            intensity = sum(1 for w in workout_log if w["logged_date"] == ds)
            color = "#10B981" if intensity >= 2 else "#34D399"
            label = f"{d.strftime('%b %d')}: {intensity} workout(s)"
        elif ds == today_str:
            color = "#6366F1"
            label = "Today"
        else:
            color = "#1E293B"
            label = d.strftime('%b %d')
        weeks[week_idx].append((d, color, label))

    day_labels_html = "".join([
        f"<div style='color:#64748B;font-size:0.65rem;width:32px;text-align:center;"
        f"margin:2px;'>{['M','T','W','T','F','S','S'][i]}</div>"
        for i in range(7)
    ])
    grid_html = ""
    for week in weeks:
        grid_html += "<div style='display:flex; gap:4px; margin-bottom:4px;'>"
        for d, color, label in week:
            grid_html += f"""
            <div title="{label}" style='width:32px; height:32px; background:{color};
                 border-radius:6px; border:1px solid #334155;
                 {"box-shadow: 0 0 8px " + color + "88;" if color != "#1E293B" else ""}'></div>"""
        grid_html += "</div>"

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                border:1px solid #334155; border-radius:14px; padding:20px;'>
      <div style='display:flex; gap:4px; margin-bottom:6px;'>
        {day_labels_html}
      </div>
      {grid_html}
      <div style='display:flex; gap:16px; margin-top:12px; align-items:center;'>
        <span style='color:#64748B; font-size:0.72rem;'>Legend:</span>
        <span style='display:flex;align-items:center;gap:4px;'>
          <span style='width:12px;height:12px;background:#1E293B;border:1px solid #334155;
                       border-radius:3px;display:inline-block;'></span>
          <span style='color:#64748B;font-size:0.72rem;'>No workout</span>
        </span>
        <span style='display:flex;align-items:center;gap:4px;'>
          <span style='width:12px;height:12px;background:#34D399;border-radius:3px;display:inline-block;'></span>
          <span style='color:#64748B;font-size:0.72rem;'>1 session</span>
        </span>
        <span style='display:flex;align-items:center;gap:4px;'>
          <span style='width:12px;height:12px;background:#10B981;border-radius:3px;display:inline-block;'></span>
          <span style='color:#64748B;font-size:0.72rem;'>2+ sessions</span>
        </span>
        <span style='display:flex;align-items:center;gap:4px;'>
          <span style='width:12px;height:12px;background:#6366F1;border-radius:3px;display:inline-block;'></span>
          <span style='color:#64748B;font-size:0.72rem;'>Today</span>
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── ROW 6: SMART INSIGHTS PANEL ───────────────────────────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:12px;'>🧠 Smart Insights</h3>", unsafe_allow_html=True)
    ins1, ins2, ins3, ins4 = st.columns(4)

    # 1. Today's Recommended Workout
    recent_workouts = sorted(workout_log, key=lambda x: x["logged_date"], reverse=True)
    last_categories = [w["category"] for w in recent_workouts[:3]]
    consecutive_same = len(set(last_categories)) == 1 and len(last_categories) >= 2
    if "Strength" in last_categories[:2]:
        rec_workout = "🧘 Yoga or Stretching"
        rec_reason  = "Recovery after strength days"
    elif "Cardio" in last_categories[:2]:
        rec_workout = "🏋️ Weight Training"
        rec_reason  = "Balance cardio with strength"
    elif not workout_log or recent_workouts[0]["logged_date"] < str(date.today() - timedelta(days=2)):
        rec_workout = "🏃 30-min Run or HIIT"
        rec_reason  = "You haven't trained recently!"
    else:
        rec_workout = "🚴 Cycling (moderate)"
        rec_reason  = "Active recovery day"

    with ins1:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-left:4px solid #6366F1;
                    border-radius:12px; padding:16px; height:140px;'>
          <p style='color:#6366F1; font-size:0.7rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px; margin:0 0 8px;'>
            💡 Today's Workout
          </p>
          <p style='color:#E2E8F0; font-size:1rem; font-weight:700; margin:0 0 6px;'>
            {rec_workout}
          </p>
          <p style='color:#64748B; font-size:0.78rem; margin:0;'>{rec_reason}</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Nutrition Alert
    if intake_today == 0:
        nut_icon, nut_msg, nut_color = "📝", "No meals logged today!", "#F59E0B"
    elif intake_today < calorie_target * 0.7:
        nut_icon, nut_msg, nut_color = "⬇️", f"Eating too little! {int(calorie_target - intake_today)} kcal remaining", "#3B82F6"
    elif intake_today > calorie_target * 1.15:
        nut_icon, nut_msg, nut_color = "⚠️", f"Over target by {int(intake_today - calorie_target)} kcal", "#EF4444"
    else:
        nut_icon, nut_msg, nut_color = "✅", "Nutrition on track today!", "#10B981"

    with ins2:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-left:4px solid {nut_color};
                    border-radius:12px; padding:16px; height:140px;'>
          <p style='color:{nut_color}; font-size:0.7rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px; margin:0 0 8px;'>
            🍽️ Nutrition Alert
          </p>
          <p style='font-size:1.8rem; margin:0 0 4px;'>{nut_icon}</p>
          <p style='color:#E2E8F0; font-size:0.85rem; font-weight:600; margin:0;'>{nut_msg}</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. Rest Day Detector
    recent_3 = [w["logged_date"] for w in recent_workouts[:3]]
    last_3_days = [str(date.today() - timedelta(days=i)) for i in range(3)]
    trained_3_in_row = all(d in workout_dates for d in last_3_days)
    if trained_3_in_row:
        rest_icon, rest_msg, rest_color = "😴", "3 days in a row! Rest day recommended.", "#F59E0B"
    elif streak >= 6:
        rest_icon, rest_msg, rest_color = "🛌", f"{streak}-day streak! Schedule a rest day soon.", "#EF4444"
    else:
        rest_icon, rest_msg, rest_color = "💪", "Recovery is good. Keep training!", "#10B981"

    with ins3:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-left:4px solid {rest_color};
                    border-radius:12px; padding:16px; height:140px;'>
          <p style='color:{rest_color}; font-size:0.7rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px; margin:0 0 8px;'>
            🔄 Recovery Status
          </p>
          <p style='font-size:1.8rem; margin:0 0 4px;'>{rest_icon}</p>
          <p style='color:#E2E8F0; font-size:0.85rem; font-weight:600; margin:0;'>{rest_msg}</p>
        </div>
        """, unsafe_allow_html=True)

    # 4. Hydration Tracker
    water_target = metrics["water"]
    # Estimate glasses (assume 250ml per glass)
    glasses_target = round(water_target * 1000 / 250)
    water_drops = "💧" * min(glasses_target, 12)
    with ins4:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-left:4px solid #3B82F6;
                    border-radius:12px; padding:16px; height:140px;'>
          <p style='color:#3B82F6; font-size:0.7rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px; margin:0 0 8px;'>
            💧 Hydration Goal
          </p>
          <p style='color:#E2E8F0; font-size:1rem; font-weight:700; margin:0 0 4px;'>
            {water_target} L / day
          </p>
          <p style='color:#64748B; font-size:0.72rem; margin:0 0 8px;'>
            ≈ {glasses_target} glasses of water
          </p>
          <p style='font-size:0.9rem; margin:0; letter-spacing:1px;'>{water_drops}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── ROW 7: GOAL PROGRESS + WEEKLY CHECK-IN + PERSONAL RECORDS ─────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:12px;'>🏆 Goals & Records</h3>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)

    # Goal Progress Card
    with g1:
        goal_to_ideal = round(abs(current_w - ideal_w), 2)
        total_journey = round(abs(start_w - ideal_w), 2) if abs(start_w - ideal_w) > 0 else 1
        progress_done = round(abs(start_w - current_w), 2)
        goal_pct_val  = min(round(progress_done / total_journey * 100), 100)

        # Estimated days to goal (500 kcal deficit = 0.5 kg/week)
        weekly_loss_rate = abs(weight_info.get("weekly_rate_kg", 0.3)) if weight_info else 0.3
        est_weeks = round(goal_to_ideal / weekly_loss_rate) if weekly_loss_rate > 0 else "—"

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px solid #334155; border-radius:14px; padding:20px;'>
          <p style='color:#10B981; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px; margin:0 0 12px;'>
            🎯 Goal Progress
          </p>
          <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
            <span style='color:#94A3B8; font-size:0.85rem;'>To ideal weight ({ideal_w} kg)</span>
            <span style='color:#10B981; font-size:0.85rem; font-weight:700;'>{goal_pct_val}%</span>
          </div>
          <div style='background:#0F172A; border-radius:20px; height:10px; overflow:hidden; margin-bottom:12px;'>
            <div style='background:linear-gradient(90deg,#6366F1,#10B981);
                        width:{goal_pct_val}%; height:100%; border-radius:20px;'></div>
          </div>
          <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;'>
            <div style='background:#0F172A; border-radius:8px; padding:10px; text-align:center;'>
              <div style='color:#6366F1; font-size:1.1rem; font-weight:700;'>{goal_to_ideal} kg</div>
              <div style='color:#64748B; font-size:0.7rem;'>Still to lose</div>
            </div>
            <div style='background:#0F172A; border-radius:8px; padding:10px; text-align:center;'>
              <div style='color:#F59E0B; font-size:1.1rem; font-weight:700;'>
                {est_weeks if isinstance(est_weeks, str) else f"~{est_weeks}w"}
              </div>
              <div style='color:#64748B; font-size:0.7rem;'>Est. time</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Weekly Check-in Summary
    with g2:
        logged_weight_today  = today_str in weight_dates
        logged_meal_today    = today_str in calorie_dates
        logged_workout_today = today_str in workout_dates
        checklist = [
            ("⚖️ Logged weight",      logged_weight_today),
            ("🍽️ Logged meals",       logged_meal_today),
            ("🏋️ Completed workout",  logged_workout_today),
            ("💧 Hydration tracked",  intake_today > 0),
        ]
        check_score = sum(1 for _, v in checklist if v)

        st.markdown(f"**✅ Daily Checklist** — {check_score}/4 done")
        for label, done in checklist:
            tick = "✅" if done else "❌"
            st.write(tick + "  " + label)

    # Personal Records
    with g3:
        st.markdown("**🏅 Personal Records**")
        if prs.get("best_cal") is not None:
            bc = prs["best_cal"]
            bd = prs["best_dur"]

            st.markdown("🔥 **Most Calories Burned**")
            st.metric(
                label=str(bc["exercise_name"]) + "  •  " + str(bc["logged_date"]),
                value=str(int(bc["calories_burned"])) + " kcal"
            )

            st.markdown("⏱️ **Longest Workout**")
            st.metric(
                label=str(bd["exercise_name"]) + "  •  " + str(bd["logged_date"]),
                value=str(int(bd["duration_minutes"])) + " min"
            )
        else:
            st.info("No workouts logged yet. Start training to set PRs! 💪")

    st.divider()

    # ── ROW 8: TODAY'S SCHEDULE ────────────────────────────────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:12px;'>📅 Today's Schedule</h3>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)

    # ── Meal plan summary (pure Streamlit) ────────────────────────────────────
    with sc1:
        meal_types  = ["Breakfast", "Lunch", "Dinner", "Snack"]
        meal_icons  = {"Breakfast": "🌅", "Lunch": "☀️", "Dinner": "🌙", "Snack": "🍎"}
        today_meals = [e for e in calorie_log if e["logged_date"] == today_str]
        meal_summary = {mt: 0.0 for mt in meal_types}
        for entry in today_meals:
            mt = entry.get("meal_type", "Snack")
            if mt in meal_summary:
                meal_summary[mt] += entry["calories"]
        total_logged = sum(meal_summary.values())

        st.markdown("**🍽️ Today's Meal Summary**")
        for mt in meal_types:
            cal = meal_summary[mt]
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(f"{meal_icons[mt]} {mt}")
            with col_b:
                if cal > 0:
                    st.success(f"{int(cal)} kcal")
                else:
                    st.caption("Not logged")

        st.divider()
        st.metric("📊 Total Today", f"{int(total_logged)} kcal",
                  delta=f"Target: {calorie_target} kcal")

    # ── 7-Day Calories Burned (Plotly bar) ────────────────────────────────────
    with sc2:
        days_labels = [(date.today() - timedelta(days=6-i)).strftime("%a") for i in range(7)]
        days_vals   = []
        today_label = date.today().strftime("%a")
        for i in range(7):
            d     = str(date.today() - timedelta(days=6-i))
            total = sum(w["calories_burned"] for w in workout_log if w["logged_date"] == d)
            days_vals.append(round(total, 1))

        bar_colors = ["#6366F1" if l == today_label else
                      "#334155" if v == 0 else "#10B981"
                      for l, v in zip(days_labels, days_vals)]

        import plotly.graph_objects as go
        fig_bar = go.Figure(go.Bar(
            x=days_labels,
            y=days_vals,
            marker_color=bar_colors,
            text=[str(int(v)) + " kcal" if v > 0 else "" for v in days_vals],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=11),
        ))
        fig_bar.update_layout(
            title="📊 7-Day Calories Burned",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(showgrid=False, color="#64748B"),
            yaxis=dict(showgrid=False, visible=False),
            showlegend=False,
            height=220,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="dash_7day_bar")

        st.markdown(f"""
        <div style='background:#1E293B; border-radius:10px; padding:12px 16px; margin-top:4px;'>
          <div style='color:#64748B; font-size:0.7rem; text-transform:uppercase;
                      letter-spacing:1px; margin-bottom:4px;'>⏭️ Next Suggested Workout</div>
          <div style='color:#E2E8F0; font-size:0.95rem; font-weight:600;'>{rec_workout}</div>
          <div style='color:#64748B; font-size:0.78rem; margin-top:2px;'>{rec_reason}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "👤 My Profile":
    import datetime as _dt
    import plotly.graph_objects as go

    profile = db.get_user_profile() or {}

    # ── FEATURE 1: Profile Avatar Card (pure Streamlit — no dynamic HTML) ──────
    if profile.get("name"):
        p_metrics  = get_health_metrics(profile)
        gender_val = profile.get("gender", "Male")
        goal_val   = profile.get("goal", "")

        avatar = "🧔" if (gender_val == "Male" and goal_val == "Gain muscle") else \
                 "👨" if gender_val == "Male" else "👩"

        fitness_level = profile.get("fitness_level", "Beginner")
        lv_icon  = {"Beginner": "🌱", "Intermediate": "⚡", "Advanced": "🔥"}.get(fitness_level, "🌱")
        lv_color = {"Beginner": "#10B981", "Intermediate": "#F59E0B", "Advanced": "#EF4444"}.get(fitness_level, "#10B981")

        dob_stored = profile.get("dob", "")
        if dob_stored:
            try:
                import datetime as _dt2
                dob_obj     = _dt2.date.fromisoformat(dob_stored)
                age_display = str((_dt2.date.today() - dob_obj).days // 365) + " yrs"
            except Exception:
                age_display = str(profile.get("age", "—")) + " yrs"
        else:
            age_display = str(profile.get("age", "—")) + " yrs"

        bmi_val   = p_metrics["bmi"]
        bmi_cat   = p_metrics["bmi_cat"]
        tdee_val  = int(p_metrics["tdee"])
        bf_val    = p_metrics["body_fat"]
        raw_tw    = profile.get("target_weight", 0)
        target_w  = raw_tw if raw_tw and float(raw_tw) >= 30 else p_metrics["ideal"]["Average"]

        goal_icon_disp = {"Lose weight":"🔥","Gain muscle":"💪",
                          "Maintain weight":"⚖️","Improve endurance":"🚴"}.get(goal_val, "🎯")

        gender_display = "👨 Male" if gender_val == "Male" else "👩 Female"
        activity_display = profile.get("activity_level", "—")

        # Hero banner header
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E1B4B,#0F172A);
                    border:1px solid #4338CA; border-radius:18px;
                    padding:20px 24px 16px; margin-bottom:16px;'>
          <p style='color:#A5B4FC; font-size:0.72rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:1px; margin:0 0 4px;'>
            👤 Profile
          </p>
          <h2 style='color:#E2E8F0; margin:0; font-size:1.8rem;'>
            {avatar} {profile["name"]}
          </h2>
          <p style='color:#64748B; font-size:0.88rem; margin:4px 0 0;'>
            🎂 {age_display} &nbsp;•&nbsp;
            {gender_display} &nbsp;•&nbsp;
            {lv_icon} {fitness_level} &nbsp;•&nbsp;
            🎯 {goal_val}
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Stats row using st.columns + st.metric (always renders correctly)
        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("📊 BMI", bmi_val, bmi_cat)
        sc2.metric("🔥 TDEE", f"{tdee_val} kcal")
        sc3.metric("💪 Body Fat", f"{bf_val}%")
        sc4.metric("⚖️ Weight", f"{profile.get('weight_kg','—')} kg")
        sc5.metric("🎯 Target", f"{target_w} kg")
        sc6.metric("📏 Height", f"{profile.get('height_cm','—')} cm")

        st.markdown(f"""
        <div style='background:{lv_color}11; border:1px solid {lv_color}44;
                    border-radius:10px; padding:8px 16px; margin:8px 0 4px;
                    display:inline-block;'>
          <span style='color:{lv_color}; font-weight:700; font-size:0.85rem;'>
            {lv_icon} Fitness Level: {fitness_level}
          </span>
          &nbsp;&nbsp;
          <span style='color:#64748B; font-size:0.8rem;'>
            📍 {activity_display}
          </span>
          &nbsp;&nbsp;
          <span style='font-size:1.2rem;'>{goal_icon_disp}</span>
          <span style='color:#94A3B8; font-size:0.8rem;'> {goal_val}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1E293B,#0F172A);
                    border:1px dashed #475569; border-radius:14px;
                    padding:30px; text-align:center; margin-bottom:20px;'>
          <div style='font-size:4rem;'>👤</div>
          <h3 style='color:#E2E8F0;'>Welcome! Fill in your details below to get started 🚀</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin:16px 0 8px;'>✏️ Edit Profile</h3>", unsafe_allow_html=True)

    # ── PROFILE FORM (with new fields) ────────────────────────────────────────
    with st.form("profile_form"):
        st.markdown("##### 👤 Personal Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            name   = st.text_input("👤 Full Name", value=profile.get("name", ""),
                                    placeholder="e.g. John Doe")
            gender = st.selectbox("⚥ Gender", ["Male", "Female"],
                                   index=0 if profile.get("gender", "Male") == "Male" else 1)

        with col2:
            import datetime as _dt
            default_dob = _dt.date(2000, 1, 1)
            if profile.get("dob"):
                try:
                    default_dob = _dt.date.fromisoformat(profile["dob"])
                except Exception:
                    pass
            dob      = st.date_input("🎂 Date of Birth", value=default_dob,
                                      min_value=_dt.date(1930, 1, 1),
                                      max_value=_dt.date.today())
            auto_age = (_dt.date.today() - dob).days // 365
            st.caption(f"🔢 Auto-calculated age: **{auto_age} years**")
            weight_kg = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=300.0,
                                         value=float(profile.get("weight_kg", 70.0)), step=0.1)

        with col3:
            height_cm = st.number_input("📏 Height (cm)", min_value=100.0, max_value=250.0,
                                         value=float(profile.get("height_cm", 170.0)), step=0.5)
            # FEATURE 2: Target weight — fix default so it's never below min_value
            raw_tw_stored = profile.get("target_weight", 0)
            tw_default    = float(raw_tw_stored) if raw_tw_stored and float(raw_tw_stored) >= 30 \
                            else float(profile.get("weight_kg", 70.0))
            target_weight = st.number_input("🎯 Target Weight (kg)", min_value=30.0, max_value=300.0,
                                             value=tw_default, step=0.1,
                                             help="Your goal weight you want to reach")

        st.markdown("##### 🏃 Fitness Info")
        f1, f2, f3 = st.columns(3)
        with f1:
            activity_opts = list(bmi_module.ACTIVITY_MULTIPLIERS.keys())
            act_idx       = activity_opts.index(profile["activity_level"]) \
                            if profile.get("activity_level") in activity_opts else 1
            activity_level = st.selectbox("🏃 Activity Level", activity_opts, index=act_idx)
        with f2:
            goal_opts = ["Lose weight", "Maintain weight", "Gain muscle", "Improve endurance"]
            goal_idx  = goal_opts.index(profile["goal"]) if profile.get("goal") in goal_opts else 0
            goal      = st.selectbox("🎯 Primary Goal", goal_opts, index=goal_idx)
        with f3:
            level_opts        = ["Beginner", "Intermediate", "Advanced"]
            level_idx         = level_opts.index(profile["fitness_level"]) \
                                if profile.get("fitness_level") in level_opts else 0
            fitness_level_inp = st.selectbox("💪 Fitness Level", level_opts, index=level_idx,
                                              help="🌱 Beginner = <6 months  ⚡ Intermediate = 6m–2yr  🔥 Advanced = 2yr+")

        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        if submitted:
            if not name:
                st.error("⚠️ Please enter your name.")
            else:
                db.save_user_profile(name, auto_age, gender, weight_kg, height_cm,
                                     activity_level, goal,
                                     fitness_level=fitness_level_inp,
                                     target_weight=target_weight,
                                     dob=str(dob))
                st.success("✅ Profile saved successfully!")
                st.balloons()
                st.rerun()

    st.divider()

    # ── WEIGHT LOG SECTION ─────────────────────────────────────────────────────
    st.markdown("### 📝 Log Today's Weight")
    with st.form("weight_log_form"):
        wc1, wc2, wc3 = st.columns(3)
        log_w    = wc1.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=300.0,
                                     value=float(profile.get("weight_kg", 70.0)), step=0.1)
        log_date = wc2.date_input("📅 Date", value=date.today())
        log_notes = wc3.text_input("📝 Notes (optional)",
                                    placeholder="e.g. Morning, post-workout…")
        if st.form_submit_button("➕ Log Weight", use_container_width=True):
            db.log_weight(log_w, str(log_date), log_notes)
            st.success(f"✅ Logged {log_w} kg for {log_date}")
            st.rerun()

    # ── FEATURE 5: Weight Trend Mini Chart ────────────────────────────────────
    weight_log = db.get_weight_log()
    if weight_log and len(weight_log) >= 2:
        wl_df = pd.DataFrame(weight_log).sort_values("logged_date")
        wl_df["logged_date"] = pd.to_datetime(wl_df["logged_date"])

        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            x=wl_df["logged_date"],
            y=wl_df["weight_kg"],
            mode="lines+markers",
            line=dict(color="#6366F1", width=2.5),
            marker=dict(size=6, color="#8B5CF6"),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            name="Weight",
        ))
        # Rolling avg
        if len(wl_df) >= 3:
            wl_df["roll"] = wl_df["weight_kg"].rolling(3, min_periods=1).mean()
            fig_spark.add_trace(go.Scatter(
                x=wl_df["logged_date"], y=wl_df["roll"],
                mode="lines", line=dict(color="#10B981", width=1.5, dash="dot"),
                name="3-day avg",
            ))
        fig_spark.update_layout(
            title="📈 Your Weight Trend",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            height=200,
            margin=dict(l=10, r=10, t=36, b=10),
            xaxis=dict(showgrid=False, color="#64748B"),
            yaxis=dict(showgrid=True, gridcolor="#1E293B", color="#64748B"),
            legend=dict(orientation="h", y=1.15, x=0),
            showlegend=True,
        )
        st.plotly_chart(fig_spark, use_container_width=True, key="profile_weight_spark")

    st.divider()

    # ── FEATURE 6: Weight Log Table + Delete ──────────────────────────────────
    st.markdown("### 📋 Weight History")
    if weight_log:
        df_wl = pd.DataFrame(weight_log)

        # Show last entry change
        if len(df_wl) >= 2:
            latest  = df_wl.iloc[0]["weight_kg"]
            prev    = df_wl.iloc[1]["weight_kg"]
            delta_w = round(latest - prev, 2)
            sign    = "📈 +" if delta_w > 0 else "📉 "
            color_d = "#EF4444" if delta_w > 0 else "#10B981"
            a1, a2, a3 = st.columns(3)
            a1.metric("📍 Latest", f"{latest} kg")
            a2.metric("📅 Previous", f"{prev} kg")
            a3.metric("📊 Change", f"{sign}{abs(delta_w)} kg")

        st.markdown("**Select entries to delete ↓**")
        df_display = df_wl[["id", "logged_date", "weight_kg", "notes"]].copy()
        df_display.columns = ["ID", "Date", "Weight (kg)", "Notes"]

        # Editable table with selection for delete
        selected_ids = []
        for _, row in df_display.head(15).iterrows():
            col_chk, col_date, col_w, col_note = st.columns([0.5, 2, 1.5, 3])
            with col_chk:
                checked = st.checkbox("", key=f"del_{row['ID']}")
            with col_date:
                st.write(str(row["Date"]))
            with col_w:
                st.write(f"⚖️ {row['Weight (kg)']} kg")
            with col_note:
                st.write(str(row["Notes"]) if row["Notes"] else "—")
            if checked:
                selected_ids.append(int(row["ID"]))

        if selected_ids:
            if st.button(f"🗑️ Delete {len(selected_ids)} selected entr{'y' if len(selected_ids)==1 else 'ies'}",
                         type="primary"):
                conn = db._get_conn()
                for sid in selected_ids:
                    conn.execute("DELETE FROM weight_log WHERE id=?", (sid,))
                conn.commit()
                st.success(f"✅ Deleted {len(selected_ids)} weight entr{'y' if len(selected_ids)==1 else 'ies'}!")
                st.rerun()
        else:
            st.caption("☝️ Tick checkboxes above to select entries for deletion")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HEALTH METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "⚖️ Health Metrics":
    st.markdown("<h1>⚖️ Health Metrics</h1>", unsafe_allow_html=True)
    profile = db.get_user_profile()
    if not profile:
        st.warning("Please set up your profile first.")
        st.stop()

    m = get_health_metrics(profile)

    # BMI Section
    st.subheader("📊 Body Mass Index (BMI)")
    b1, b2, b3 = st.columns([1, 2, 1])
    with b1:
        st.markdown(f"""
        <div class='fitness-card' style='text-align:center'>
          <div style='font-size:3.5rem;font-weight:800;color:{m["bmi_color"]}'>{m["bmi"]}</div>
          <div style='color:#94A3B8;font-size:1rem'>BMI Score</div>
          <div style='font-size:1.2rem;font-weight:600;color:{m["bmi_color"]};margin-top:8px'>{m["bmi_cat"]}</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.plotly_chart(charts.bmi_gauge(m["bmi"]), use_container_width=True, key="metrics_bmi_gauge")
    with b3:
        ideal = m["ideal"]
        st.markdown(f"""
        <div class='fitness-card'>
          <p style='color:#94A3B8;margin:0 0 8px'>Ideal Weight Range</p>
          <p style='color:#10B981;font-size:1.4rem;font-weight:700;margin:0'>{ideal['Average']} kg</p>
          <p style='color:#64748B;font-size:0.8rem;margin:4px 0 0'>Average of 3 formulas</p>
          <hr style='border-color:#334155;margin:12px 0'>
          <p style='color:#94A3B8;font-size:0.8rem;margin:2px 0'>Devine: {ideal['Devine']} kg</p>
          <p style='color:#94A3B8;font-size:0.8rem;margin:2px 0'>Robinson: {ideal['Robinson']} kg</p>
          <p style='color:#94A3B8;font-size:0.8rem;margin:2px 0'>Miller: {ideal['Miller']} kg</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Metabolic rates
    st.subheader("🔥 Metabolic Rates")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("BMR (Basal)", f"{int(m['bmr'])} kcal/day", "Calories at rest")
    mc2.metric("TDEE (Total)", f"{int(m['tdee'])} kcal/day", "Maintenance calories")
    mc3.metric("Calorie Target", f"{m['macros']['calorie_target']} kcal/day", f"Goal: {profile['goal']}")
    mc4.metric("💧 Water Intake", f"{m['water']} L/day")

    st.divider()

    # Body composition
    st.subheader("💪 Body Composition")
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Body Fat %", f"{m['body_fat']}%")
    bc2.metric("Lean Body Mass", f"{m['lbm']} kg")
    bc3.metric("Fat Mass", f"{round(profile['weight_kg'] - m['lbm'], 1)} kg")

    st.divider()

    # Macronutrients
    st.subheader("🥗 Daily Macronutrient Targets")
    mac = m["macros"]
    mac1, mac2, mac3, mac4 = st.columns(4)
    mac1.metric("🍽️ Daily Calories", f"{mac['calorie_target']} kcal")
    mac2.metric("🥩 Protein", f"{mac['protein_g']} g", "30% of calories")
    mac3.metric("🍚 Carbohydrates", f"{mac['carbs_g']} g", "40% of calories")
    mac4.metric("🧈 Fat", f"{mac['fat_g']} g", "30% of calories")

    calorie_log = db.get_calorie_log()
    today_macros = cc.macro_totals_for_date(calorie_log)
    st.plotly_chart(
        charts.macro_donut_chart(today_macros["protein_g"], today_macros["carbs_g"],
                                  today_macros["fat_g"], mac["protein_g"], mac["carbs_g"], mac["fat_g"]),
        use_container_width=True,
        key="metrics_macro_donut",
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION: FITNESS SCORE + HEALTH AGE + WHtR
    # ══════════════════════════════════════════════════════════════════════════

    workout_log_hm  = db.get_workout_log()
    calorie_log_hm  = db.get_calorie_log()
    weight_log_hm   = db.get_weight_log()

    # ── Pre-compute shared values ─────────────────────────────────────────────
    age        = profile["age"]
    gender     = profile["gender"]
    weight_kg  = profile["weight_kg"]
    height_cm  = profile["height_cm"]
    bmi_val    = m["bmi"]
    bf_val     = m["body_fat"]
    tdee_val   = m["tdee"]

    # Workouts last 7 days
    cutoff_7d  = str(date.today() - timedelta(days=7))
    wk_count   = len([w for w in workout_log_hm if w["logged_date"] >= cutoff_7d])

    # Avg daily calorie intake last 7 days
    cutoff_7d_cal = str(date.today() - timedelta(days=6))
    daily_cals = {}
    for e in calorie_log_hm:
        if e["logged_date"] >= cutoff_7d_cal:
            daily_cals[e["logged_date"]] = daily_cals.get(e["logged_date"], 0) + e["calories"]
    avg_intake = sum(daily_cals.values()) / max(len(daily_cals), 1)
    cal_target = m["macros"]["calorie_target"]
    cal_diff   = abs(avg_intake - cal_target)

    # ── 1. FITNESS SCORE (0–100) ──────────────────────────────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:4px;'>🏅 Fitness Score</h3>", unsafe_allow_html=True)
    st.caption("A composite 0–100 score based on BMI, body fat, workout frequency, and calorie balance.")

    # Component scores (each 0–25)
    # BMI score: 25 if in normal range (18.5–24.9), scales down outside
    if 18.5 <= bmi_val <= 24.9:
        bmi_score = 25
    elif bmi_val < 18.5:
        bmi_score = max(0, round(25 - (18.5 - bmi_val) * 3))
    else:
        bmi_score = max(0, round(25 - (bmi_val - 24.9) * 2))

    # Body fat score: 25 if athletic/fit, scales down
    if gender == "Male":
        bf_ideal_lo, bf_ideal_hi = 10, 17
    else:
        bf_ideal_lo, bf_ideal_hi = 18, 24
    if bf_ideal_lo <= bf_val <= bf_ideal_hi:
        bf_score = 25
    elif bf_val < bf_ideal_lo:
        bf_score = max(0, round(25 - (bf_ideal_lo - bf_val) * 1.5))
    else:
        bf_score = max(0, round(25 - (bf_val - bf_ideal_hi) * 1.5))

    # Workout frequency score: 25 if 5+ sessions/week
    wk_score = min(25, round(wk_count / 5 * 25))

    # Calorie balance score: 25 if within 10% of target
    if cal_target > 0:
        cal_pct_off = cal_diff / cal_target
        cal_score   = max(0, round(25 - cal_pct_off * 50))
    else:
        cal_score = 12  # neutral if no data

    fitness_score = bmi_score + bf_score + wk_score + cal_score
    fitness_score = max(0, min(100, fitness_score))

    # Grade and color
    if fitness_score >= 85:
        fs_grade, fs_color, fs_label, fs_icon = "A+", "#10B981", "Excellent", "🏆"
    elif fitness_score >= 70:
        fs_grade, fs_color, fs_label, fs_icon = "A",  "#6366F1", "Very Good", "💪"
    elif fitness_score >= 55:
        fs_grade, fs_color, fs_label, fs_icon = "B",  "#F59E0B", "Good",      "👍"
    elif fitness_score >= 40:
        fs_grade, fs_color, fs_label, fs_icon = "C",  "#F97316", "Fair",      "📈"
    else:
        fs_grade, fs_color, fs_label, fs_icon = "D",  "#EF4444", "Needs Work","⚠️"

    fs_bar_pct = fitness_score

    fs1, fs2, fs3, fs4, fs5 = st.columns(5)
    with fs1:
        st.markdown(
            "<div style='background:linear-gradient(135deg,#1E293B,#0F172A);"
            "border:1px solid #334155; border-radius:14px; padding:20px; text-align:center;'>"
            "<div style='color:#64748B; font-size:0.7rem; text-transform:uppercase;"
            "letter-spacing:1px; margin-bottom:6px;'>🏅 Overall Score</div>"
            "<div style='color:" + fs_color + "; font-size:3rem; font-weight:900; line-height:1;'>"
            + str(fitness_score) + "</div>"
            "<div style='color:#64748B; font-size:0.75rem; margin:2px 0 10px;'>/100</div>"
            "<div style='background:#0F172A; border-radius:20px; height:8px; overflow:hidden;'>"
            "<div style='background:linear-gradient(90deg," + fs_color + "88," + fs_color + ");"
            "width:" + str(fs_bar_pct) + "%; height:100%; border-radius:20px;'></div></div>"
            "<div style='margin-top:8px;'>"
            "<span style='background:" + fs_color + "22; color:" + fs_color + ";"
            "border:1px solid " + fs_color + "55; border-radius:20px;"
            "padding:3px 12px; font-size:0.78rem; font-weight:700;'>"
            + fs_icon + " " + fs_label + " · Grade " + fs_grade + "</span></div></div>",
            unsafe_allow_html=True
        )

    # Component breakdown using st.metric (safe, no HTML bugs)
    with fs2:
        st.metric("📊 BMI Score", f"{bmi_score}/25",
                  delta="Optimal" if bmi_score == 25 else f"{25 - bmi_score} pts to max")
    with fs3:
        st.metric("💪 Body Fat Score", f"{bf_score}/25",
                  delta="Optimal" if bf_score == 25 else f"{25 - bf_score} pts to max")
    with fs4:
        st.metric("🏋️ Workout Score", f"{wk_score}/25",
                  delta=f"{wk_count} sessions this week")
    with fs5:
        st.metric("🍽️ Calorie Score", f"{cal_score}/25",
                  delta="On target" if cal_score >= 20 else f"{int(cal_diff)} kcal off target")

    st.divider()

    # ── 2. HEALTH AGE ─────────────────────────────────────────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:4px;'>🧬 Health Age</h3>", unsafe_allow_html=True)
    st.caption("Your estimated biological age based on BMI, body fat %, and activity level — compared to your actual age.")

    # Activity multiplier map to a numeric score 1–5
    activity_score_map = {
        "Sedentary (little or no exercise)":       1,
        "Lightly active (1-3 days/week)":           2,
        "Moderately active (3-5 days/week)":        3,
        "Very active (6-7 days/week)":              4,
        "Extra active (physical job or 2x/day)":   5,
    }
    act_score = activity_score_map.get(profile.get("activity_level", ""), 2)

    # Health age algorithm
    # Start with actual age, then adjust based on each factor
    health_age = float(age)

    # BMI adjustment: ideal 21.5, +0.5yr per BMI point away
    bmi_diff = abs(bmi_val - 21.5)
    health_age += bmi_diff * 0.5

    # Body fat adjustment: ideal midpoint of fit range
    bf_ideal = 13.5 if gender == "Male" else 21.5
    bf_diff  = abs(bf_val - bf_ideal)
    health_age += bf_diff * 0.3

    # Activity adjustment: -1yr per activity level above sedentary
    health_age -= (act_score - 1) * 1.2

    health_age = round(max(10, min(age + 20, health_age)), 1)
    age_diff   = round(health_age - age, 1)

    if age_diff <= -3:
        ha_msg, ha_color, ha_icon = f"Your body is {abs(age_diff)} years YOUNGER than your age!", "#10B981", "🌟"
    elif age_diff <= 0:
        ha_msg, ha_color, ha_icon = "Your health age is on par with your actual age.", "#6366F1", "✅"
    elif age_diff <= 5:
        ha_msg, ha_color, ha_icon = f"Your body is aging slightly faster. Focus on fitness & diet.", "#F59E0B", "⚠️"
    else:
        ha_msg, ha_color, ha_icon = f"Your body is {age_diff} years older than your actual age. Act now!", "#EF4444", "🚨"

    ha1, ha2, ha3 = st.columns(3)
    ha1.metric("🎂 Actual Age",  f"{age} yrs")
    ha2.metric("🧬 Health Age",  f"{health_age} yrs",
               delta=f"{age_diff:+.1f} yrs vs actual")
    ha3.metric("📊 Difference",
               f"{'Younger' if age_diff < 0 else 'Older'} by {abs(age_diff)} yrs")

    ha_diff_display = str(abs(age_diff))
    ha_younger_older = "younger" if age_diff < 0 else "older"

    st.markdown(
        "<div style='background:" + ha_color + "11; border:1px solid " + ha_color + "44;"
        "border-left:4px solid " + ha_color + "; border-radius:12px;"
        "padding:14px 20px; margin-top:8px;'>"
        "<span style='font-size:1.3rem;'>" + ha_icon + "</span>"
        "&nbsp;&nbsp;<span style='color:" + ha_color + "; font-weight:700; font-size:0.95rem;'>"
        + ha_msg + "</span>"
        "<div style='color:#64748B; font-size:0.78rem; margin-top:6px;'>"
        "Health Age is estimated from your BMI, body fat %, and activity level. "
        "Improve any of these to lower your biological age.</div></div>",
        unsafe_allow_html=True
    )

    # Factor breakdown table using pure st.write (no HTML risks)
    st.markdown("**🔍 What's Affecting Your Health Age:**")
    hf1, hf2, hf3 = st.columns(3)
    with hf1:
        bmi_impact = round(bmi_diff * 0.5, 1)
        bmi_sign   = "+" if bmi_val != 21.5 else "="
        st.metric("📊 BMI Impact", f"{bmi_sign}{bmi_impact} yrs",
                  delta="Ideal BMI = 21.5" if bmi_val != 21.5 else "Perfect!")
    with hf2:
        bf_impact = round(bf_diff * 0.3, 1)
        bf_sign   = "+" if bf_val != bf_ideal else "="
        st.metric("💪 Body Fat Impact", f"{bf_sign}{bf_impact} yrs",
                  delta=f"Ideal = {bf_ideal}%" if bf_val != bf_ideal else "Perfect!")
    with hf3:
        act_impact = round((act_score - 1) * 1.2, 1)
        st.metric("🏃 Activity Benefit", f"-{act_impact} yrs",
                  delta=f"Level {act_score}/5 activity")

    st.divider()

    # ── 3. WAIST-TO-HEIGHT RATIO (WHtR) ───────────────────────────────────────
    st.markdown("<h3 style='color:#E2E8F0; margin-bottom:4px;'>📐 Waist-to-Height Ratio (WHtR)</h3>", unsafe_allow_html=True)
    st.caption("A stronger predictor of cardiometabolic risk than BMI. Enter your waist circumference to calculate.")

    whtr_col1, whtr_col2 = st.columns([1, 2])
    with whtr_col1:
        waist_cm = st.number_input("📏 Waist Circumference (cm)",
                                    min_value=40.0, max_value=200.0,
                                    value=float(st.session_state.get("waist_cm", round(height_cm * 0.5, 1))),
                                    step=0.5, key="waist_input",
                                    help="Measure at navel level, relaxed breath")
        st.session_state["waist_cm"] = waist_cm

    whtr = round(waist_cm / height_cm, 3)

    # Risk categories (universal thresholds)
    if whtr < 0.40:
        whtr_label, whtr_color, whtr_icon, whtr_desc = (
            "Slim / Underweight", "#3B82F6", "🔵",
            "Waist may be too slim. Ensure adequate nutrition."
        )
    elif whtr < 0.50:
        whtr_label, whtr_color, whtr_icon, whtr_desc = (
            "Healthy", "#10B981", "🟢",
            "Excellent! Your waist-to-height ratio is in the optimal healthy range."
        )
    elif whtr < 0.60:
        whtr_label, whtr_color, whtr_icon, whtr_desc = (
            "Overweight / Increased Risk", "#F59E0B", "🟡",
            "Slightly elevated abdominal fat. Consider cardio and dietary adjustments."
        )
    else:
        whtr_label, whtr_color, whtr_icon, whtr_desc = (
            "Obese / High Risk", "#EF4444", "🔴",
            "High cardiovascular and metabolic risk. Prioritise weight management."
        )

    with whtr_col2:
        # WHtR value + risk badge using st.metric + safe HTML for badge only
        wh1, wh2, wh3 = st.columns(3)
        wh1.metric("📐 WHtR Value",     str(whtr),
                   delta="Optimal < 0.50")
        wh2.metric("📏 Waist",          f"{waist_cm} cm")
        wh3.metric("📏 Height",         f"{height_cm} cm")

        # Risk scale bar (all pre-computed, no inline conditionals)
        scale_pct = min(round(whtr / 0.8 * 100), 100)

        st.markdown(
            "<div style='background:linear-gradient(135deg,#1E293B,#0F172A);"
            "border:1px solid #334155; border-radius:12px; padding:16px; margin-top:8px;'>"
            "<div style='display:flex; justify-content:space-between; margin-bottom:6px;'>"
            "<span style='color:#94A3B8; font-size:0.8rem;'>WHtR Scale</span>"
            "<span style='color:" + whtr_color + "; font-weight:700; font-size:0.9rem;'>"
            + whtr_icon + " " + whtr_label + "</span></div>"
            "<div style='background:#0F172A; border-radius:20px; height:10px; overflow:hidden; margin-bottom:10px;'>"
            "<div style='background:linear-gradient(90deg,#10B981,#F59E0B,#EF4444);"
            "width:100%; height:100%; border-radius:20px; position:relative;'></div></div>"
            "<div style='background:#0F172A; border-radius:20px; height:10px; overflow:hidden; margin-bottom:10px;'>"
            "<div style='background:" + whtr_color + "; width:" + str(scale_pct) + "%;"
            "height:100%; border-radius:20px;'></div></div>"
            "<div style='color:#64748B; font-size:0.75rem; display:flex; justify-content:space-between;'>"
            "<span>0.40</span><span style='color:#10B981;'>0.50 ✓ Healthy</span>"
            "<span style='color:#F59E0B;'>0.60</span><span style='color:#EF4444;'>0.80+</span></div>"
            "<div style='margin-top:10px; padding-top:10px; border-top:1px solid #334155;'>"
            "<span style='color:" + whtr_color + "; font-size:0.85rem; font-weight:600;'>"
            + whtr_desc + "</span></div></div>",
            unsafe_allow_html=True
        )

    # Reference table using st.dataframe — completely safe
    st.markdown("**📋 WHtR Risk Category Reference:**")
    import pandas as pd
    whtr_ref = pd.DataFrame({
        "WHtR Range":    ["< 0.40", "0.40 – 0.49", "0.50 – 0.59", "≥ 0.60"],
        "Category":      ["Slim / Underweight", "Healthy ✅", "Overweight ⚠️", "Obese / High Risk 🚨"],
        "Health Risk":   ["Possible under-nutrition", "Low risk — optimal range",
                          "Increased cardiovascular risk", "High cardiometabolic risk"],
        "Action":        ["Ensure adequate diet", "Maintain current lifestyle",
                          "Reduce abdominal fat", "Seek medical advice"],
    })
    st.dataframe(whtr_ref, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WORKOUT TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🏋️ Workout Tracker":
    import datetime as _dt
    st.markdown("<h1>🏋️ Workout Tracker</h1>", unsafe_allow_html=True)
    profile    = db.get_user_profile()
    user_weight = profile["weight_kg"] if profile else 70.0

    # ── LOG WORKOUT FORM ──────────────────────────────────────────────────────
    st.markdown("### ➕ Log a Workout")
    with st.form("workout_form"):
        wf1, wf2, wf3 = st.columns(3)
        exercise  = wf1.selectbox("🏃 Exercise", wt.get_all_exercises())
        duration  = wf2.number_input("⏱️ Duration (min)", min_value=1, max_value=300, value=30)
        intensity = wf3.selectbox("⚡ Intensity", ["Low", "Moderate", "High"])

        # FEATURE 1: Sets & Reps — shown for Strength category
        ex_category = wt.get_category_for_exercise(exercise)
        is_strength = ex_category == "Strength"

        wf4, wf5, wf6, wf7 = st.columns(4)
        w_date  = wf4.date_input("📅 Date", value=date.today())
        w_notes = wf5.text_input("📝 Notes", placeholder="Optional notes…")
        w_sets  = wf6.number_input("🔢 Sets",
                                    min_value=0, max_value=100, value=0,
                                    help="For strength exercises. Leave 0 if not applicable.")
        w_reps  = wf7.number_input("🔁 Reps per Set",
                                    min_value=0, max_value=200, value=0,
                                    help="For strength exercises. Leave 0 if not applicable.")

        # FEATURE: MET intensity auto-adjust (High = +20%, Low = -15%)
        est_cal = wt.estimate_calories_with_intensity(exercise, duration, user_weight, intensity)
        mult_label = {"Low": "−15% Low intensity", "Moderate": "Standard MET", "High": "+20% High intensity"}
        st.info(f"🔥 Estimated calories burned: **{est_cal} kcal** — {mult_label[intensity]}")

        if st.form_submit_button("➕ Log Workout", use_container_width=True):
            sets_val = int(w_sets) if w_sets > 0 else None
            reps_val = int(w_reps) if w_reps > 0 else None
            db.log_workout(exercise, ex_category, duration, est_cal, intensity,
                           str(w_date), w_notes, sets=sets_val, reps=reps_val)
            sets_str = f" · {sets_val}×{reps_val}" if sets_val else ""
            st.success(f"✅ Logged: {exercise} — {duration} min — {est_cal} kcal{sets_str}")
            st.rerun()

    st.divider()

    workout_log = db.get_workout_log()

    # ── WEEKLY KPI STRIP ──────────────────────────────────────────────────────
    wk_summary = wt.weekly_workout_summary(workout_log)
    ws1, ws2, ws3, ws4 = st.columns(4)
    ws1.metric("🏋️ Sessions (7d)",      wk_summary["total_sessions"])
    ws2.metric("⏱️ Minutes (7d)",        wk_summary["total_minutes"])
    ws3.metric("🔥 Calories (7d)",       f"{wk_summary['total_calories']} kcal")
    ws4.metric("📊 Avg Duration",
               f"{round(wk_summary['total_minutes'] / max(wk_summary['total_sessions'], 1))} min")

    st.divider()

    # ── FEATURE: CATEGORY STREAKS ─────────────────────────────────────────────
    st.markdown("### 🔥 Streak per Category")
    cat_streaks   = wt.category_streak(workout_log)
    last_sessions = wt.last_session_per_category(workout_log)
    cat_icons     = {"Cardio": "🏃", "Strength": "💪", "Flexibility": "🧘", "Sports": "⚽"}
    cat_colors    = {"Cardio": "#EF4444", "Strength": "#6366F1", "Flexibility": "#10B981", "Sports": "#F59E0B"}

    csc1, csc2, csc3, csc4 = st.columns(4)
    for col, cat in zip([csc1, csc2, csc3, csc4], ["Cardio", "Strength", "Flexibility", "Sports"]):
        streak_val  = cat_streaks.get(cat, 0)
        last_d      = last_sessions.get(cat, None)
        days_ago    = (date.today() - _dt.date.fromisoformat(last_d)).days if last_d else None
        days_label  = f"Last: {days_ago}d ago" if days_ago is not None else "Never logged"
        col.metric(
            label=cat_icons[cat] + " " + cat,
            value=f"{streak_val} day streak",
            delta=days_label,
        )

    st.divider()

    # ── FEATURE: WEEKLY WORKOUT PLANNER ──────────────────────────────────────
    st.markdown("### 📅 Weekly Workout Planner")
    st.caption("Plan your week and track what you've completed.")

    days_of_week   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    today_weekday  = date.today().weekday()  # 0=Mon

    # Build set of days this week that have workouts logged
    week_start = date.today() - timedelta(days=today_weekday)
    done_days  = set()
    for entry in workout_log:
        try:
            ld = _dt.date.fromisoformat(entry["logged_date"])
            if week_start <= ld <= date.today():
                done_days.add(ld.weekday())
        except Exception:
            pass

    # Planner session state — planned days
    if "planned_days" not in st.session_state:
        st.session_state["planned_days"] = set()

    plan_cols = st.columns(7)
    for i, (col, day) in enumerate(zip(plan_cols, days_of_week)):
        with col:
            is_done    = i in done_days
            is_today   = i == today_weekday
            is_future  = i > today_weekday
            is_planned = i in st.session_state["planned_days"]

            if is_done:
                bg, border, label = "#10B98122", "#10B981", "✅ Done"
                txt_color = "#10B981"
            elif is_today:
                bg, border, label = "#6366F122", "#6366F1", "📍 Today"
                txt_color = "#6366F1"
            elif is_planned and is_future:
                bg, border, label = "#F59E0B22", "#F59E0B", "📌 Planned"
                txt_color = "#F59E0B"
            else:
                bg, border, label = "#1E293B", "#334155", "—"
                txt_color = "#64748B"

            st.markdown(
                "<div style='background:" + bg + "; border:2px solid " + border + ";"
                "border-radius:10px; padding:10px 6px; text-align:center;'>"
                "<div style='color:" + txt_color + "; font-weight:700; font-size:0.9rem;'>" + day + "</div>"
                "<div style='color:" + txt_color + "; font-size:0.7rem; margin-top:4px;'>" + label + "</div>"
                "</div>",
                unsafe_allow_html=True
            )
            # Toggle plan button for future days
            if is_future:
                if st.button("Plan" if i not in st.session_state["planned_days"] else "Unplan",
                             key=f"plan_{i}", use_container_width=True):
                    if i in st.session_state["planned_days"]:
                        st.session_state["planned_days"].discard(i)
                    else:
                        st.session_state["planned_days"].add(i)
                    st.rerun()

    st.divider()

    # ── CHARTS ────────────────────────────────────────────────────────────────
    wdf = wt.workout_log_to_dataframe(workout_log)
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(charts.workout_frequency_chart(wdf),
                        use_container_width=True, key="workout_freq_chart")
    with ch2:
        st.plotly_chart(charts.workout_category_pie(wdf),
                        use_container_width=True, key="workout_pie_chart")

    st.divider()

    # ── FEATURE: PERSONAL RECORDS PER EXERCISE ───────────────────────────────
    st.markdown("### 🏅 Personal Records by Exercise")
    prs_by_ex = wt.get_personal_records_by_exercise(workout_log)
    if prs_by_ex:
        # Filter selector
        ex_with_prs = sorted(prs_by_ex.keys())
        selected_ex = st.selectbox("🔍 Select Exercise to view PRs",
                                    ex_with_prs, key="pr_ex_select")
        pr = prs_by_ex[selected_ex]

        pr1, pr2, pr3, pr4 = st.columns(4)
        pr1.metric("🔥 Best Calories",   f"{pr['best_cal']} kcal",
                   delta=pr["best_cal_date"])
        pr2.metric("⏱️ Longest Session", f"{pr['best_dur']} min",
                   delta=pr["best_dur_date"])
        pr3.metric("📋 Total Sessions",  str(pr["sessions"]))
        pr4.metric("⏳ Total Minutes",   str(pr["total_minutes"]))

        if pr["best_sets"]:
            st.metric("🔢 Most Sets Logged", str(pr["best_sets"]) + " sets")

        # Mini history table for selected exercise
        ex_entries = [e for e in workout_log if e["exercise_name"] == selected_ex]
        if ex_entries:
            df_ex = pd.DataFrame(ex_entries)[["logged_date", "duration_minutes",
                                               "calories_burned", "intensity",
                                               "sets", "reps", "notes"]]
            df_ex.columns = ["Date", "Duration (min)", "Calories", "Intensity",
                              "Sets", "Reps", "Notes"]
            st.dataframe(df_ex.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No workouts logged yet. Start training to see your PRs! 💪")

    st.divider()

    # ── FEATURE: DELETE WORKOUT ENTRIES ──────────────────────────────────────
    st.markdown("### 📋 Recent Workouts & Delete")
    if workout_log:
        # FEATURE: Edit last entry expander
        with st.expander("✏️ Edit Last Entry"):
            last = workout_log[0]
            st.caption(f"Editing: **{last['exercise_name']}** on {last['logged_date']}")
            with st.form("edit_last_form"):
                el1, el2, el3 = st.columns(3)
                edit_dur = el1.number_input("Duration (min)", min_value=1, max_value=300,
                                             value=int(last["duration_minutes"]))
                edit_cal = el2.number_input("Calories", min_value=0.0,
                                             value=float(last["calories_burned"]), step=1.0)
                edit_int = el3.selectbox("Intensity", ["Low", "Moderate", "High"],
                                          index=["Low","Moderate","High"].index(
                                              last.get("intensity","Moderate")))
                el4, el5 = st.columns(2)
                edit_sets = el4.number_input("Sets", min_value=0, max_value=100,
                                              value=int(last["sets"]) if last.get("sets") else 0)
                edit_reps = el5.number_input("Reps", min_value=0, max_value=200,
                                              value=int(last["reps"]) if last.get("reps") else 0)
                edit_notes = st.text_input("Notes", value=last.get("notes","") or "")

                if st.form_submit_button("💾 Save Changes", use_container_width=True):
                    conn = db._get_conn()
                    conn.execute("""UPDATE workout_log SET
                        duration_minutes=?, calories_burned=?, intensity=?,
                        sets=?, reps=?, notes=? WHERE id=?""",
                        (edit_dur, edit_cal, edit_int,
                         int(edit_sets) if edit_sets > 0 else None,
                         int(edit_reps) if edit_reps > 0 else None,
                         edit_notes, last["id"]))
                    conn.commit()
                    st.success("✅ Entry updated!")
                    st.rerun()

        # Delete entries with checkboxes
        st.markdown("**Select entries to delete ↓**")
        del_ids = []
        for entry in workout_log[:15]:
            sets_reps_str = ""
            if entry.get("sets"):
                sets_reps_str = f" · {entry['sets']}×{entry.get('reps','?')}"
            c1, c2, c3, c4, c5 = st.columns([0.4, 1.8, 2, 1.5, 1.5])
            with c1:
                if st.checkbox("", key=f"wdel_{entry['id']}"):
                    del_ids.append(entry["id"])
            with c2:
                st.write(str(entry["logged_date"]))
            with c3:
                st.write(f"🏋️ {entry['exercise_name']}")
            with c4:
                st.write(f"⏱️ {entry['duration_minutes']} min{sets_reps_str}")
            with c5:
                st.write(f"🔥 {entry['calories_burned']} kcal")

        if del_ids:
            n = len(del_ids)
            if st.button(f"🗑️ Delete {n} entr{'y' if n==1 else 'ies'}",
                         type="primary", key="wdel_btn"):
                conn = db._get_conn()
                for wid in del_ids:
                    conn.execute("DELETE FROM workout_log WHERE id=?", (wid,))
                conn.commit()
                st.success(f"✅ Deleted {n} workout entr{'y' if n==1 else 'ies'}!")
                st.rerun()
        else:
            st.caption("☝️ Tick checkboxes to select entries for deletion")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CALORIE TRACKER
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🍽️ Calorie Tracker":
    st.markdown("<h1>🍽️ Calorie Tracker</h1>", unsafe_allow_html=True)
    profile = db.get_user_profile()
    metrics = get_health_metrics(profile) if profile else None
    cal_target = metrics["macros"]["calorie_target"] if metrics else 2000

    # ── FEATURE 1: QUICK-ADD COMMON FOODS ────────────────────────────────────
    st.markdown("### ⚡ Quick-Add Common Foods")
    st.caption("One click to instantly log a common food item.")

    # Food database: name → (calories, protein_g, carbs_g, fat_g, meal_type)
    QUICK_FOODS = {
        "🥚 Egg":            ("Egg (1 whole)",       78,  6.0,  0.6,  5.0, "Breakfast"),
        "🍚 Rice":           ("Rice (1 cup cooked)", 206, 4.3, 44.5,  0.4, "Lunch"),
        "🍌 Banana":         ("Banana (1 medium)",    89, 1.1, 23.0,  0.3, "Snack"),
        "🥛 Milk":           ("Milk (1 cup 250ml)",  149, 8.0, 11.7,  8.0, "Breakfast"),
        "🍗 Chicken Breast": ("Chicken Breast 100g", 165,31.0,  0.0,  3.6, "Lunch"),
        "🥜 Peanut Butter":  ("Peanut Butter 2tbsp", 188, 8.0,  6.0, 16.0, "Snack"),
        "🍞 Bread":          ("Bread (1 slice)",      79, 2.7, 15.0,  1.0, "Breakfast"),
        "🍎 Apple":          ("Apple (1 medium)",     95, 0.5, 25.0,  0.3, "Snack"),
        "🥦 Broccoli":       ("Broccoli (1 cup)",     55, 3.7, 11.0,  0.6, "Dinner"),
        "🐟 Tuna":           ("Tuna canned 100g",    116,25.5,  0.0,  0.8, "Lunch"),
    }

    qa_cols = st.columns(5)
    for idx, (btn_label, food_data) in enumerate(QUICK_FOODS.items()):
        fname, fcal, fpro, fcarb, ffat, fmeal = food_data
        with qa_cols[idx % 5]:
            if st.button(btn_label, use_container_width=True, key=f"qa_{idx}"):
                db.log_calories(fname, fmeal, fcal, fpro, fcarb, ffat, str(date.today()))
                st.success(f"✅ Logged {fname} — {fcal} kcal")
                st.rerun()

    st.divider()

    # ── MANUAL LOG FORM ───────────────────────────────────────────────────────
    st.markdown("### ✍️ Log a Meal Manually")

    # Pre-fill state from quick-add selection
    with st.form("calorie_form"):
        cf1, cf2, cf3 = st.columns(3)
        meal_name = cf1.text_input("🍽️ Meal / Food", placeholder="e.g. Grilled Chicken")
        meal_type = cf2.selectbox("🕐 Meal Type",
                                   ["Breakfast", "Lunch", "Dinner", "Snack", "Drink"])
        c_date    = cf3.date_input("📅 Date", value=date.today())

        cf4, cf5, cf6, cf7 = st.columns(4)
        calories  = cf4.number_input("🔥 Calories (kcal)", min_value=0.0, value=0.0, step=10.0)
        protein_g = cf5.number_input("🥩 Protein (g)",     min_value=0.0, value=0.0, step=1.0)
        carbs_g   = cf6.number_input("🍚 Carbs (g)",       min_value=0.0, value=0.0, step=1.0)
        fat_g     = cf7.number_input("🧈 Fat (g)",         min_value=0.0, value=0.0, step=1.0)

        if st.form_submit_button("➕ Log Meal", use_container_width=True):
            db.log_calories(meal_name, meal_type, calories,
                            protein_g, carbs_g, fat_g, str(c_date))
            st.success(f"✅ Logged: {meal_name} — {calories} kcal")
            st.rerun()

    st.divider()

    # ── DATA ──────────────────────────────────────────────────────────────────
    calorie_log = db.get_calorie_log()
    workout_log = db.get_workout_log()
    tdee        = metrics["tdee"] if metrics else 2000
    today_str   = str(date.today())

    today_bal    = cc.daily_calorie_balance(calorie_log, workout_log, tdee)
    today_macros = cc.macro_totals_for_date(calorie_log)

    # ── TODAY'S CALORIE BALANCE ───────────────────────────────────────────────
    st.markdown("### 📅 Today's Calorie Balance")
    tb1, tb2, tb3, tb4 = st.columns(4)
    tb1.metric("🍽️ Consumed",         f"{int(today_bal['intake'])} kcal")
    tb2.metric("🔥 TDEE + Exercise",   f"{int(today_bal['total_burned'])} kcal")
    tb3.metric("⚡ Net",               f"{int(today_bal['net'])} kcal",
               delta=today_bal["surplus_deficit"])
    tb4.metric("🎯 Target",            f"{cal_target} kcal")

    # Weekly charts
    weekly_df = cc.weekly_calorie_summary(calorie_log, workout_log, tdee)
    cal_trend = da.calorie_trend_analysis(calorie_log)
    ch1, ch2  = st.columns(2)
    with ch1:
        st.plotly_chart(charts.calorie_balance_chart(weekly_df),
                        use_container_width=True, key="cal_balance_chart")
    with ch2:
        st.plotly_chart(charts.calorie_intake_trend_chart(cal_trend["dataframe"], tdee),
                        use_container_width=True, key="cal_trend_chart")

    st.divider()

    # ── FEATURE 2: MEAL TIMING BREAKDOWN ─────────────────────────────────────
    st.markdown("### 🕐 Meal Timing Breakdown — Today")
    st.caption("How your calories are distributed across each meal type today.")

    meal_types  = ["Breakfast", "Lunch", "Dinner", "Snack", "Drink"]
    meal_icons  = {"Breakfast":"🌅","Lunch":"☀️","Dinner":"🌙","Snack":"🍎","Drink":"💧"}
    today_meals = [e for e in calorie_log if e["logged_date"] == today_str]

    meal_totals = {mt: {"calories":0.0,"protein_g":0.0,"carbs_g":0.0,"fat_g":0.0,"items":0}
                   for mt in meal_types}
    for e in today_meals:
        mt = e.get("meal_type","Snack")
        if mt not in meal_totals:
            mt = "Snack"
        meal_totals[mt]["calories"]  += e["calories"]
        meal_totals[mt]["protein_g"] += e.get("protein_g", 0)
        meal_totals[mt]["carbs_g"]   += e.get("carbs_g", 0)
        meal_totals[mt]["fat_g"]     += e.get("fat_g", 0)
        meal_totals[mt]["items"]     += 1

    total_today_cal = sum(v["calories"] for v in meal_totals.values())

    # Build breakdown table as a clean DataFrame
    breakdown_rows = []
    for mt in meal_types:
        d   = meal_totals[mt]
        pct = round(d["calories"] / total_today_cal * 100) if total_today_cal > 0 else 0
        breakdown_rows.append({
            "Meal":         meal_icons[mt] + " " + mt,
            "Items":        d["items"],
            "Calories":     int(d["calories"]),
            "% of Day":     str(pct) + "%",
            "Protein (g)":  round(d["protein_g"], 1),
            "Carbs (g)":    round(d["carbs_g"], 1),
            "Fat (g)":      round(d["fat_g"], 1),
        })
    breakdown_df = pd.DataFrame(breakdown_rows)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # Visual bar per meal type (pure st.metric + progress via st.markdown string concat)
    if total_today_cal > 0:
        mt_cols = st.columns(5)
        for col, mt in zip(mt_cols, meal_types):
            d   = meal_totals[mt]
            pct = round(d["calories"] / total_today_cal * 100) if total_today_cal > 0 else 0
            col.metric(
                label=meal_icons[mt] + " " + mt,
                value=f"{int(d['calories'])} kcal",
                delta=f"{pct}% of today",
            )

    st.divider()

    # ── FEATURE 3: CALORIE-TO-GOAL CALCULATOR ────────────────────────────────
    st.markdown("### 🎯 Calorie-to-Goal Calculator")
    st.caption("Based on your current average intake — when will you reach your target weight?")

    if profile:
        current_w = profile["weight_kg"]
        raw_tw    = profile.get("target_weight", 0)
        goal_w    = float(raw_tw) if raw_tw and float(raw_tw) >= 30 else metrics["ideal"]["Average"]

        # Avg daily intake last 7 days
        last7_cals = {}
        cutoff7    = str(date.today() - timedelta(days=6))
        for e in calorie_log:
            if e["logged_date"] >= cutoff7:
                last7_cals[e["logged_date"]] = \
                    last7_cals.get(e["logged_date"], 0) + e["calories"]
        avg_intake_7d = round(sum(last7_cals.values()) / max(len(last7_cals), 1), 0)

        # Net daily surplus/deficit vs TDEE
        daily_net   = avg_intake_7d - tdee
        # 1 kg fat = 7700 kcal
        weight_diff = current_w - goal_w

        cg1, cg2, cg3 = st.columns(3)
        cg1.metric("⚖️ Current Weight",  f"{current_w} kg")
        cg2.metric("🎯 Target Weight",   f"{goal_w} kg")
        cg3.metric("📏 Difference",      f"{round(abs(weight_diff), 1)} kg to go")

        if abs(daily_net) > 10:
            kcal_to_shift   = abs(weight_diff) * 7700
            days_to_goal    = round(kcal_to_shift / abs(daily_net))
            weeks_to_goal   = round(days_to_goal / 7)
            goal_date       = date.today() + timedelta(days=days_to_goal)
            direction       = "deficit" if daily_net < 0 else "surplus"
            dir_color       = "#10B981" if direction == "deficit" and weight_diff > 0 else \
                              "#EF4444" if direction == "surplus" and weight_diff > 0 else "#6366F1"

            cg4, cg5, cg6 = st.columns(3)
            cg4.metric("📊 Avg Intake (7d)",   f"{int(avg_intake_7d)} kcal/day",
                       delta=f"{int(daily_net):+d} vs TDEE")
            cg5.metric("📅 Est. Weeks to Goal", f"~{weeks_to_goal} weeks")
            cg6.metric("🏁 Est. Goal Date",     goal_date.strftime("%b %d, %Y"))

            dir_sign = "−" if direction == "deficit" else "+"
            abs_net  = str(int(abs(daily_net)))
            st.markdown(
                "<div style='background:" + dir_color + "11; border:1px solid " + dir_color + "44;"
                "border-left:4px solid " + dir_color + "; border-radius:12px;"
                "padding:14px 20px; margin-top:8px;'>"
                "<span style='color:" + dir_color + "; font-weight:700;'>📈 Projection: </span>"
                "<span style='color:#E2E8F0;'>At your current average of "
                + str(int(avg_intake_7d)) + " kcal/day ("
                + dir_sign + abs_net + " kcal vs TDEE), "
                "you are projected to reach <strong>" + str(goal_w) + " kg</strong> "
                "in approximately <strong>~" + str(weeks_to_goal) + " weeks</strong> "
                "(" + goal_date.strftime("%b %d, %Y") + ").</span></div>",
                unsafe_allow_html=True
            )

            # Suggested adjustment
            ideal_weekly_loss = 0.5  # kg per week safe rate
            ideal_daily_deficit = round(ideal_weekly_loss * 7700 / 7)
            suggested_intake = round(tdee - ideal_daily_deficit)
            st.markdown("**💡 Recommended daily intake for healthy 0.5 kg/week loss:**")
            sr1, sr2 = st.columns(2)
            sr1.metric("🍽️ Suggested Intake",   f"{suggested_intake} kcal/day")
            sr2.metric("📉 Required Deficit",    f"{ideal_daily_deficit} kcal/day")
        else:
            st.info("Log at least a few days of meals to see your personalised projection.")
    else:
        st.warning("Set up your profile to use the goal calculator.")

    st.divider()

    # ── FEATURE 4: MEAL PATTERN INSIGHTS ─────────────────────────────────────
    st.markdown("### 🧠 Meal Pattern Insights")
    st.caption("Smart analysis of your eating habits over the last 7 days.")

    last7_entries = [e for e in calorie_log if e["logged_date"] >= str(date.today() - timedelta(days=6))]

    if last7_entries:
        # Group by date and meal_type
        from collections import defaultdict
        date_meals = defaultdict(set)
        date_cal_by_type = defaultdict(lambda: defaultdict(float))
        for e in last7_entries:
            date_meals[e["logged_date"]].add(e.get("meal_type","Snack"))
            date_cal_by_type[e["logged_date"]][e.get("meal_type","Snack")] += e["calories"]

        days_logged      = len(date_meals)
        days_no_breakfast = sum(1 for d, meals in date_meals.items() if "Breakfast" not in meals)
        days_no_lunch     = sum(1 for d, meals in date_meals.items() if "Lunch" not in meals)
        days_no_dinner    = sum(1 for d, meals in date_meals.items() if "Dinner" not in meals)

        # Front-loading: breakfast + lunch > 60% of daily cals
        front_loaded_days = 0
        back_loaded_days  = 0
        for d, by_type in date_cal_by_type.items():
            total_d = sum(by_type.values())
            if total_d > 0:
                morning_pct = (by_type.get("Breakfast",0) + by_type.get("Lunch",0)) / total_d
                dinner_pct  = by_type.get("Dinner",0) / total_d
                if morning_pct > 0.60:
                    front_loaded_days += 1
                if dinner_pct > 0.50:
                    back_loaded_days += 1

        insights = []

        # Breakfast skipping
        if days_no_breakfast >= 3:
            insights.append(("⚠️", "#F59E0B",
                "Skipping Breakfast Often",
                f"You skipped breakfast {days_no_breakfast}/7 days. "
                "Breakfast helps regulate metabolism and reduces overeating later."))
        elif days_no_breakfast == 0:
            insights.append(("✅", "#10B981",
                "Consistent Breakfast",
                "Great! You logged breakfast every day this week."))

        # Lunch skipping
        if days_no_lunch >= 3:
            insights.append(("⚠️", "#F59E0B",
                "Skipping Lunch Often",
                f"Lunch was skipped {days_no_lunch}/7 days. "
                "This may lead to energy crashes and evening overeating."))

        # Back-loading (dinner heavy)
        if back_loaded_days >= 3:
            insights.append(("🌙", "#EF4444",
                "Heavy Evening Eating",
                f"{back_loaded_days}/7 days had dinner as 50%+ of total calories. "
                "Front-loading calories earlier improves metabolic health."))
        elif back_loaded_days == 0 and days_logged >= 3:
            insights.append(("🌟", "#10B981",
                "Good Calorie Distribution",
                "Your calories are well spread across the day — great habit!"))

        # Front-loading (breakfast + lunch heavy)
        if front_loaded_days >= 4:
            insights.append(("🌅", "#6366F1",
                "Front-Loading Calories",
                f"{front_loaded_days}/7 days had 60%+ of calories before dinner. "
                "This is actually beneficial for metabolism and weight management!"))

        # Overall consistency
        if days_logged >= 6:
            insights.append(("📋", "#10B981",
                "Excellent Logging Consistency",
                f"You logged meals on {days_logged}/7 days. "
                "Consistent tracking is one of the strongest predictors of success."))
        elif days_logged <= 3:
            insights.append(("📝", "#F59E0B",
                "Improve Logging Consistency",
                f"Only {days_logged}/7 days logged. "
                "Try to log every meal for accurate insights and better progress."))

        # Show insight cards using safe string concat (no inline conditionals in f-strings)
        for icon, color, title, desc in insights:
            st.markdown(
                "<div style='background:" + color + "11; border:1px solid " + color + "33;"
                "border-left:4px solid " + color + "; border-radius:12px;"
                "padding:14px 18px; margin-bottom:10px;'>"
                "<div style='display:flex; align-items:center; gap:10px; margin-bottom:4px;'>"
                "<span style='font-size:1.3rem;'>" + icon + "</span>"
                "<span style='color:" + color + "; font-weight:700; font-size:0.95rem;'>"
                + title + "</span></div>"
                "<span style='color:#94A3B8; font-size:0.85rem;'>" + desc + "</span>"
                "</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Log meals for at least a few days to see pattern insights.")

    st.divider()

    # ── RECENT MEALS TABLE + DELETE ──────────────────────────────────────────
    st.markdown("### 📋 Recent Meals & Delete")
    if calorie_log:
        del_meal_ids = []
        meal_type_icons = {"Breakfast":"🌅","Lunch":"☀️","Dinner":"🌙","Snack":"🍎","Drink":"💧"}

        for entry in calorie_log[:15]:
            mt_icon = meal_type_icons.get(entry.get("meal_type","Snack"), "🍽️")
            c1, c2, c3, c4, c5, c6 = st.columns([0.4, 1.5, 2.5, 1.2, 1.2, 1.2])
            with c1:
                if st.checkbox("", key=f"mdel_{entry['id']}"):
                    del_meal_ids.append(entry["id"])
            with c2:
                st.write(str(entry["logged_date"]))
            with c3:
                st.write(mt_icon + " " + str(entry["meal_name"]))
            with c4:
                st.write(f"🔥 {int(entry['calories'])} kcal")
            with c5:
                st.write(f"🥩 {entry.get('protein_g', 0)}g")
            with c6:
                st.write(entry.get("meal_type", "—"))

        if del_meal_ids:
            n = len(del_meal_ids)
            if st.button(f"🗑️ Delete {n} meal entr{'y' if n==1 else 'ies'}",
                         type="primary", key="mdel_btn"):
                conn = db._get_conn()
                for mid in del_meal_ids:
                    conn.execute("DELETE FROM calorie_log WHERE id=?", (mid,))
                conn.commit()
                st.success(f"✅ Deleted {n} meal entr{'y' if n==1 else 'ies'}!")
                st.rerun()
        else:
            st.caption("☝️ Tick checkboxes to select meals for deletion")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BODY INTELLIGENCE (formerly Data Analysis)
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🧬 Body Intelligence":
    import datetime as _dt
    import plotly.graph_objects as go
    from collections import defaultdict

    st.markdown("<h1>🧬 Body Intelligence</h1>", unsafe_allow_html=True)
    st.caption("Deep analysis of your weight, workouts, calories, and habit consistency.")

    profile     = db.get_user_profile()
    metrics     = get_health_metrics(profile) if profile else None
    tdee        = metrics["tdee"] if metrics else 2000

    weight_log  = db.get_weight_log()
    workout_log = db.get_workout_log()
    calorie_log = db.get_calorie_log()

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚖️ Weight Analysis",
        "🏋️ Workout Analysis",
        "🍽️ Calorie Analysis",
        "📅 Consistency Score",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: WEIGHT ANALYSIS  (+ Trend Velocity + 30/60/90 day projection)
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("⚖️ Weight Trend Analysis")
        weight_info = da.weight_trend_analysis(weight_log)
        if weight_info:
            # ── KPI strip ────────────────────────────────────────────────────
            da1, da2, da3, da4 = st.columns(4)
            da1.metric("📍 Current Weight",  f"{weight_info['current_weight']} kg")
            da2.metric("🏁 Starting Weight", f"{weight_info['start_weight']} kg")
            da3.metric("📊 Total Change",    f"{weight_info['change']:+.2f} kg")
            da4.metric("📈 Weekly Rate",     f"{weight_info['weekly_rate_kg']:+.2f} kg/wk",
                       delta=weight_info["trend_direction"])

            st.divider()

            # ── FEATURE 1: TREND VELOCITY ─────────────────────────────────
            st.markdown("#### 🚀 Trend Velocity")
            st.caption("How fast are you progressing — and is it speeding up or slowing down?")

            wl_df = weight_info["dataframe"].copy()
            wl_df = wl_df.sort_values("logged_date").reset_index(drop=True)

            if len(wl_df) >= 4:
                # Split into first half and second half to detect acceleration
                mid     = len(wl_df) // 2
                first_h = wl_df.iloc[:mid]
                second_h= wl_df.iloc[mid:]

                days_first  = max((first_h["logged_date"].iloc[-1] - first_h["logged_date"].iloc[0]).days, 1)
                days_second = max((second_h["logged_date"].iloc[-1] - second_h["logged_date"].iloc[0]).days, 1)

                rate_first  = round((first_h["weight_kg"].iloc[-1] - first_h["weight_kg"].iloc[0]) / days_first * 7, 3)
                rate_second = round((second_h["weight_kg"].iloc[-1] - second_h["weight_kg"].iloc[0]) / days_second * 7, 3)
                acceleration = round(rate_second - rate_first, 3)

                # Overall rate per week
                total_days  = max((wl_df["logged_date"].iloc[-1] - wl_df["logged_date"].iloc[0]).days, 1)
                overall_rate = round((wl_df["weight_kg"].iloc[-1] - wl_df["weight_kg"].iloc[0]) / total_days * 7, 3)

                # Determine trend direction label
                if abs(overall_rate) < 0.05:
                    vel_label, vel_color, vel_icon = "Stable / Maintaining", "#6366F1", "⚖️"
                elif overall_rate < -0.5:
                    vel_label, vel_color, vel_icon = "Losing Fast", "#10B981", "📉"
                elif overall_rate < 0:
                    vel_label, vel_color, vel_icon = "Losing Steadily", "#34D399", "📉"
                elif overall_rate > 0.5:
                    vel_label, vel_color, vel_icon = "Gaining Fast", "#EF4444", "📈"
                else:
                    vel_label, vel_color, vel_icon = "Gaining Slowly", "#F59E0B", "📈"

                # Acceleration label
                if abs(acceleration) < 0.05:
                    acc_label, acc_color = "Steady pace", "#6366F1"
                elif acceleration < 0:
                    acc_label, acc_color = "Accelerating loss 🔽", "#10B981"
                else:
                    acc_label, acc_color = "Decelerating / slowing 🔼", "#F59E0B"

                tv1, tv2, tv3, tv4 = st.columns(4)
                tv1.metric("⚡ Current Velocity",    f"{overall_rate:+.2f} kg/wk", delta=vel_label)
                tv2.metric("🔰 Early Period Rate",   f"{rate_first:+.3f} kg/wk")
                tv3.metric("🔰 Recent Period Rate",  f"{rate_second:+.3f} kg/wk")
                tv4.metric("📐 Acceleration",        f"{acceleration:+.3f} kg/wk²", delta=acc_label)

                # Safe HTML insight card
                st.markdown(
                    "<div style='background:" + vel_color + "11; border-left:4px solid " + vel_color + ";"
                    "border:1px solid " + vel_color + "33; border-radius:12px; padding:14px 18px; margin-top:8px;'>"
                    "<span style='font-size:1.2rem;'>" + vel_icon + "</span>"
                    "&nbsp;<span style='color:" + vel_color + "; font-weight:700;'>" + vel_label + "</span>"
                    "&nbsp;&nbsp;<span style='color:#94A3B8; font-size:0.85rem;'>"
                    "You are losing/gaining at <strong>" + str(abs(overall_rate)) + " kg/week</strong>. "
                    "Trend momentum: <strong style='color:" + acc_color + ";'>" + acc_label + "</strong>."
                    "</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.info("Log at least 4 weight entries to see trend velocity.")

            st.divider()

            # ── FEATURE 2: 30 / 60 / 90-DAY WEIGHT PROJECTION ─────────────
            st.markdown("#### 🔮 Weight Projection — 30 / 60 / 90 Days")
            st.caption("Linear projection based on your actual rate of change.")

            if len(wl_df) >= 2:
                rate_per_day = (wl_df["weight_kg"].iloc[-1] - wl_df["weight_kg"].iloc[0]) / max(total_days, 1)
                current_w    = wl_df["weight_kg"].iloc[-1]
                last_date    = wl_df["logged_date"].iloc[-1]

                proj_30  = round(current_w + rate_per_day * 30, 2)
                proj_60  = round(current_w + rate_per_day * 60, 2)
                proj_90  = round(current_w + rate_per_day * 90, 2)

                ideal_w  = metrics["ideal"]["Average"] if metrics else current_w

                pj1, pj2, pj3, pj4 = st.columns(4)
                pj1.metric("📍 Today",      f"{current_w} kg")
                pj2.metric("📅 In 30 days", f"{proj_30} kg",
                           delta=f"{proj_30 - current_w:+.2f} kg")
                pj3.metric("📅 In 60 days", f"{proj_60} kg",
                           delta=f"{proj_60 - current_w:+.2f} kg")
                pj4.metric("📅 In 90 days", f"{proj_90} kg",
                           delta=f"{proj_90 - current_w:+.2f} kg")

                # Build chart: actual history + projection line
                proj_dates  = [last_date + timedelta(days=d) for d in range(0, 91)]
                proj_weights = [round(current_w + rate_per_day * d, 2) for d in range(0, 91)]

                fig_proj = go.Figure()
                # Actual data
                fig_proj.add_trace(go.Scatter(
                    x=wl_df["logged_date"], y=wl_df["weight_kg"],
                    mode="lines+markers",
                    name="Actual Weight",
                    line=dict(color="#6366F1", width=2.5),
                    marker=dict(size=7, color="#8B5CF6"),
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
                ))
                # Projection
                fig_proj.add_trace(go.Scatter(
                    x=proj_dates, y=proj_weights,
                    mode="lines",
                    name="Projection",
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                ))
                # Ideal weight reference line
                fig_proj.add_hline(
                    y=ideal_w, line_dash="dot",
                    line_color="#10B981", opacity=0.6,
                    annotation_text=f"Ideal: {ideal_w} kg",
                    annotation_font_color="#10B981",
                )
                # Milestone markers at 30/60/90
                for days_out, proj_w, label in [(30, proj_30, "30d"), (60, proj_60, "60d"), (90, proj_90, "90d")]:
                    fig_proj.add_trace(go.Scatter(
                        x=[last_date + timedelta(days=days_out)],
                        y=[proj_w],
                        mode="markers+text",
                        marker=dict(size=12, color="#F59E0B", symbol="diamond"),
                        text=[f"{proj_w} kg"],
                        textposition="top center",
                        textfont=dict(color="#F59E0B", size=11),
                        showlegend=False,
                    ))
                fig_proj.update_layout(
                    title="📈 Weight History + 90-Day Projection",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"),
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=10),
                    xaxis=dict(showgrid=False, color="#64748B"),
                    yaxis=dict(showgrid=True, gridcolor="#1E293B", color="#64748B"),
                    legend=dict(orientation="h", y=1.12, x=0),
                )
                st.plotly_chart(fig_proj, use_container_width=True, key="bi_weight_proj_chart")

                # Days to ideal weight
                if abs(rate_per_day) > 0.001:
                    days_to_ideal = round(abs((ideal_w - current_w) / rate_per_day))
                    eta_date      = date.today() + timedelta(days=days_to_ideal)
                    going_right   = (rate_per_day < 0 and ideal_w < current_w) or \
                                    (rate_per_day > 0 and ideal_w > current_w)
                    if going_right:
                        eta_color = "#10B981"
                        eta_msg   = "At your current rate you'll reach your ideal weight in "
                    else:
                        eta_color = "#EF4444"
                        eta_msg   = "⚠️ At current pace you're moving away from ideal. Time to adjust! "
                        days_to_ideal = None

                    if days_to_ideal:
                        st.markdown(
                            "<div style='background:" + eta_color + "11; border-left:4px solid " + eta_color + ";"
                            "border:1px solid " + eta_color + "33; border-radius:10px; padding:12px 18px;'>"
                            "<span style='color:" + eta_color + "; font-weight:700;'>🏁 ETA to Ideal Weight: </span>"
                            "<span style='color:#E2E8F0;'>" + eta_msg
                            + "<strong>" + str(days_to_ideal) + " days</strong>"
                            + " (" + eta_date.strftime("%b %d, %Y") + ")</span></div>",
                            unsafe_allow_html=True
                        )
            else:
                st.info("Log at least 2 weight entries to see projections.")

            st.divider()
            # Original weight chart
            st.markdown("#### 📊 Full Weight Trend Chart")
            st.plotly_chart(charts.weight_trend_chart(wl_df),
                            use_container_width=True, key="analysis_weight_chart")
        else:
            st.info("No weight data yet. Log your weight in My Profile.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: WORKOUT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🏋️ Workout Analysis (Last 30 Days)")
        wk_info = da.workout_frequency_analysis(workout_log, days=30)
        if wk_info.get("total_sessions", 0) > 0:
            wa1, wa2, wa3, wa4 = st.columns(4)
            wa1.metric("📅 Sessions/Week",   wk_info["sessions_per_week"])
            wa2.metric("⏱️ Total Minutes",   wk_info["total_minutes"])
            wa3.metric("🔥 Calories Burned", f"{wk_info['total_calories_burned']} kcal")
            wa4.metric("🏆 Top Exercise",    wk_info["most_common_exercise"])

            wdf_chart = wt.workout_log_to_dataframe(workout_log)
            wch1, wch2 = st.columns(2)
            with wch1:
                st.plotly_chart(charts.workout_frequency_chart(wdf_chart),
                                use_container_width=True, key="analysis_workout_freq")
            with wch2:
                st.plotly_chart(charts.workout_category_pie(wdf_chart),
                                use_container_width=True, key="analysis_workout_pie")

            st.divider()

            # ── Volume progression (weekly minutes over last 8 weeks) ──────
            st.markdown("#### 📈 Weekly Volume Progression")
            st.caption("Total workout minutes per week — are you doing more or less over time?")

            weekly_vol = {}
            for entry in workout_log:
                try:
                    ld   = _dt.date.fromisoformat(entry["logged_date"])
                    # ISO week key
                    wkey = ld.strftime("%Y-W%W")
                    weekly_vol[wkey] = weekly_vol.get(wkey, 0) + entry["duration_minutes"]
                except Exception:
                    pass

            if len(weekly_vol) >= 2:
                sorted_weeks = sorted(weekly_vol.keys())[-8:]
                vol_vals     = [weekly_vol[w] for w in sorted_weeks]
                week_labels  = [w.split("-")[1] for w in sorted_weeks]

                fig_vol = go.Figure(go.Bar(
                    x=week_labels, y=vol_vals,
                    marker_color=["#6366F1" if i == len(vol_vals)-1 else "#334155"
                                  for i in range(len(vol_vals))],
                    text=[str(v) + " min" for v in vol_vals],
                    textposition="outside",
                    textfont=dict(color="#94A3B8", size=11),
                ))
                fig_vol.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E2E8F0"), height=260,
                    margin=dict(l=10, r=10, t=20, b=10),
                    xaxis=dict(showgrid=False, color="#64748B", title="Week"),
                    yaxis=dict(showgrid=False, visible=False),
                )
                st.plotly_chart(fig_vol, use_container_width=True, key="bi_vol_chart")
            else:
                st.info("Log workouts across multiple weeks to see volume progression.")

            st.divider()

            # ── Top 5 Exercises ───────────────────────────────────────────
            st.markdown("#### 🏅 Favourite Exercises Ranking")
            ex_counts = defaultdict(lambda: {"sessions": 0, "minutes": 0, "calories": 0})
            for entry in workout_log:
                ex = entry["exercise_name"]
                ex_counts[ex]["sessions"] += 1
                ex_counts[ex]["minutes"]  += entry["duration_minutes"]
                ex_counts[ex]["calories"] += entry["calories_burned"]

            top5 = sorted(ex_counts.items(), key=lambda x: x[1]["sessions"], reverse=True)[:5]
            if top5:
                rank_rows = []
                for rank, (ex_name, stats) in enumerate(top5, 1):
                    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                    rank_rows.append({
                        "Rank":          medals.get(rank, f"#{rank}"),
                        "Exercise":      ex_name,
                        "Sessions":      stats["sessions"],
                        "Total Minutes": stats["minutes"],
                        "Total Calories":f"{round(stats['calories'])} kcal",
                    })
                st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)

        else:
            st.info("No workout data yet. Log workouts in the Workout Tracker.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: CALORIE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("🍽️ Calorie Trend Analysis (Last 30 Days)")
        cal_info = da.calorie_trend_analysis(calorie_log, days=30)
        if not cal_info["dataframe"].empty:
            ca1, ca2, ca3, ca4 = st.columns(4)
            ca1.metric("📊 Avg Daily Intake", f"{cal_info['avg_daily_intake']} kcal")
            ca2.metric("⬆️ Highest Day",      f"{cal_info['max_day']} kcal")
            ca3.metric("⬇️ Lowest Day",       f"{cal_info['min_day']} kcal")
            ca4.metric("⚡ vs TDEE",          f"{round(cal_info['avg_daily_intake'] - tdee):+d} kcal")
            st.plotly_chart(charts.calorie_intake_trend_chart(cal_info["dataframe"], tdee),
                            use_container_width=True, key="analysis_cal_trend")

            st.divider()

            # ── Best & worst calorie day ──────────────────────────────────
            st.markdown("#### 🏆 Best & Worst Calorie Days (Last 30 Days)")
            daily_totals = defaultdict(float)
            for e in calorie_log:
                cutoff_30 = str(date.today() - timedelta(days=30))
                if e["logged_date"] >= cutoff_30:
                    daily_totals[e["logged_date"]] += e["calories"]

            if daily_totals:
                best_day  = min(daily_totals.items(), key=lambda x: abs(x[1] - tdee))
                worst_day = max(daily_totals.items(), key=lambda x: abs(x[1] - tdee))

                bd1, bd2 = st.columns(2)
                with bd1:
                    diff_best = round(best_day[1] - tdee)
                    st.markdown(
                        "<div style='background:#10B98111; border-left:4px solid #10B981;"
                        "border:1px solid #10B98133; border-radius:12px; padding:16px;'>"
                        "<div style='color:#10B981; font-weight:700; font-size:0.8rem;"
                        "text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>"
                        "✅ Best Day — Closest to Target</div>"
                        "<div style='color:#E2E8F0; font-size:1.4rem; font-weight:800;'>"
                        + best_day[0] + "</div>"
                        "<div style='color:#10B981; font-size:1rem; font-weight:700;'>"
                        + str(int(best_day[1])) + " kcal"
                        + " (" + str(diff_best) + " vs TDEE)</div></div>",
                        unsafe_allow_html=True
                    )
                with bd2:
                    diff_worst = round(worst_day[1] - tdee)
                    st.markdown(
                        "<div style='background:#EF444411; border-left:4px solid #EF4444;"
                        "border:1px solid #EF444433; border-radius:12px; padding:16px;'>"
                        "<div style='color:#EF4444; font-weight:700; font-size:0.8rem;"
                        "text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>"
                        "⚠️ Worst Day — Furthest from Target</div>"
                        "<div style='color:#E2E8F0; font-size:1.4rem; font-weight:800;'>"
                        + worst_day[0] + "</div>"
                        "<div style='color:#EF4444; font-size:1rem; font-weight:700;'>"
                        + str(int(worst_day[1])) + " kcal"
                        + " (" + str(diff_worst) + " vs TDEE)</div></div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("No calorie data yet. Log meals in the Calorie Tracker.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: CONSISTENCY SCORE
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("📅 Consistency Score")
        st.caption("Did you log weight, meals, and a workout every day? Your weekly habit score.")

        # Build last 4 weeks of data
        weight_dates  = {e["logged_date"] for e in weight_log}
        calorie_dates = {e["logged_date"] for e in calorie_log}
        workout_dates = {e["logged_date"] for e in workout_log}

        NUM_WEEKS = 4
        weeks_data = []
        for w in range(NUM_WEEKS - 1, -1, -1):
            week_days = []
            week_start_d = date.today() - timedelta(days=date.today().weekday() + w * 7)
            week_scores  = []
            for d_offset in range(7):
                d        = week_start_d + timedelta(days=d_offset)
                ds       = str(d)
                is_future = d > date.today()
                logged_w  = ds in weight_dates
                logged_c  = ds in calorie_dates
                logged_wo = ds in workout_dates
                day_score = sum([logged_w, logged_c, logged_wo])  # 0–3
                week_days.append({
                    "date": d, "ds": ds,
                    "weight": logged_w, "calories": logged_c, "workout": logged_wo,
                    "score": day_score, "future": is_future,
                })
                if not is_future:
                    week_scores.append(day_score)

            week_total  = sum(week_scores)
            week_max    = len(week_scores) * 3
            week_pct    = round(week_total / week_max * 100) if week_max > 0 else 0
            weeks_data.append({
                "label":   f"Week of {week_start_d.strftime('%b %d')}",
                "days":    week_days,
                "score":   week_total,
                "max":     week_max,
                "pct":     week_pct,
            })

        # ── 4-week score summary ─────────────────────────────────────────────
        wk_cols = st.columns(NUM_WEEKS)
        for col, wk in zip(wk_cols, weeks_data):
            pct   = wk["pct"]
            color = "#10B981" if pct >= 80 else "#F59E0B" if pct >= 50 else "#EF4444"
            grade = "A" if pct >= 80 else "B" if pct >= 60 else "C" if pct >= 40 else "D"
            col.metric(
                label=wk["label"],
                value=f"{pct}% — Grade {grade}",
                delta=f"{wk['score']}/{wk['max']} points",
            )

        st.divider()

        # ── Day-by-day heatmap grid ──────────────────────────────────────────
        st.markdown("#### 📊 Day-by-Day Habit Tracker")
        st.caption("⚖️ Weight &nbsp;|&nbsp; 🍽️ Meals &nbsp;|&nbsp; 🏋️ Workout logged per day")

        legend_cols = st.columns(4)
        legend_cols[0].markdown("🟢 All 3 logged")
        legend_cols[1].markdown("🟡 2 of 3 logged")
        legend_cols[2].markdown("🔴 0–1 logged")
        legend_cols[3].markdown("⬜ Future day")

        # Day-of-week header
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        header_cols = st.columns([0.8] + [1]*7)
        header_cols[0].markdown("**Week**")
        for i, dl in enumerate(dow_labels):
            header_cols[i+1].markdown(f"**{dl}**")

        for wk in weeks_data:
            row_cols = st.columns([0.8] + [1]*7)
            row_cols[0].caption(wk["label"].split(" of ")[1])
            for i, day in enumerate(wk["days"]):
                with row_cols[i+1]:
                    if day["future"]:
                        bg, border = "#1E293B", "#334155"
                        dot = "⬜"
                    elif day["score"] == 3:
                        bg, border, dot = "#10B98122", "#10B981", "🟢"
                    elif day["score"] == 2:
                        bg, border, dot = "#F59E0B22", "#F59E0B", "🟡"
                    elif day["score"] == 1:
                        bg, border, dot = "#EF444422", "#EF4444", "🔴"
                    else:
                        bg, border, dot = "#EF444422", "#EF4444", "🔴"

                    icons = ""
                    if not day["future"]:
                        icons = ("⚖️" if day["weight"] else "·") + \
                                ("🍽️" if day["calories"] else "·") + \
                                ("🏋️" if day["workout"] else "·")

                    st.markdown(
                        "<div style='background:" + bg + "; border:1px solid " + border + ";"
                        "border-radius:8px; padding:6px 4px; text-align:center;"
                        "margin:2px 0;'>"
                        "<div style='font-size:1rem;'>" + dot + "</div>"
                        "<div style='color:#64748B; font-size:0.6rem; margin-top:2px;'>"
                        + day["date"].strftime("%d") + "</div>"
                        "<div style='font-size:0.55rem; color:#475569; margin-top:2px;'>"
                        + icons + "</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )

        st.divider()

        # ── Overall 4-week consistency stats ────────────────────────────────
        st.markdown("#### 🏅 4-Week Consistency Summary")
        all_past_days = [d for wk in weeks_data for d in wk["days"] if not d["future"]]
        total_days_n  = len(all_past_days)

        if total_days_n > 0:
            perfect_days = sum(1 for d in all_past_days if d["score"] == 3)
            weight_days  = sum(1 for d in all_past_days if d["weight"])
            cal_days     = sum(1 for d in all_past_days if d["calories"])
            wo_days      = sum(1 for d in all_past_days if d["workout"])
            overall_pct  = round(sum(d["score"] for d in all_past_days) / (total_days_n * 3) * 100)

            cs1, cs2, cs3, cs4, cs5 = st.columns(5)
            cs1.metric("🌟 Overall Score",     f"{overall_pct}%",
                       delta=f"{perfect_days}/{total_days_n} perfect days")
            cs2.metric("⚖️ Weight Logged",     f"{weight_days}/{total_days_n} days",
                       delta=f"{round(weight_days/total_days_n*100)}%")
            cs3.metric("🍽️ Meals Logged",      f"{cal_days}/{total_days_n} days",
                       delta=f"{round(cal_days/total_days_n*100)}%")
            cs4.metric("🏋️ Workouts Logged",   f"{wo_days}/{total_days_n} days",
                       delta=f"{round(wo_days/total_days_n*100)}%")

            # Overall grade
            grade_label = "A+" if overall_pct >= 90 else "A" if overall_pct >= 80 else \
                          "B"  if overall_pct >= 60 else "C" if overall_pct >= 40 else "D"
            grade_color = "#10B981" if overall_pct >= 80 else \
                          "#F59E0B" if overall_pct >= 50 else "#EF4444"
            with cs5:
                st.markdown(
                    "<div style='background:" + grade_color + "11; border:1px solid " + grade_color + "44;"
                    "border-radius:12px; padding:16px; text-align:center;'>"
                    "<div style='color:" + grade_color + "; font-size:2.5rem; font-weight:900;'>"
                    + grade_label + "</div>"
                    "<div style='color:#94A3B8; font-size:0.75rem;'>4-Week Grade</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

            # Personalised tip
            weakest = min(
                [("weight logging", weight_days), ("meal logging", cal_days), ("workout logging", wo_days)],
                key=lambda x: x[1]
            )
            tip_color = "#6366F1"
            st.markdown(
                "<div style='background:#6366F111; border-left:4px solid #6366F1;"
                "border:1px solid #6366F133; border-radius:10px; padding:12px 16px; margin-top:8px;'>"
                "<span style='color:#6366F1; font-weight:700;'>💡 Tip: </span>"
                "<span style='color:#CBD5E1;'>Your weakest habit is <strong>"
                + weakest[0] + "</strong> ("
                + str(weakest[1]) + "/" + str(total_days_n) + " days). "
                "Focus on this one habit first — consistency compounds!</span></div>",
                unsafe_allow_html=True
            )

    # ── Overall Progress Summary ─────────────────────────────────────────────
    if profile:
        st.divider()
        st.subheader("🎯 Overall Progress Summary")
        summary = da.generate_progress_summary(weight_log, workout_log, calorie_log, tdee)
        ps1, ps2, ps3 = st.columns(3)
        ps1.metric("⚖️ Current Weight",       f"{summary['current_weight']} kg",
                   delta=f"{summary['weight_change']:+.2f} kg overall")
        ps2.metric("🏋️ Workouts/Week",        summary["workouts_per_week"])
        ps3.metric("🍽️ Avg Calories vs TDEE", f"{summary['calorie_vs_tdee']:+.0f} kcal")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS & AI
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔮 Predictions & AI":
    import plotly.graph_objects as go
    import datetime as _dt

    st.markdown("<h1>🔮 Predictions & AI</h1>", unsafe_allow_html=True)
    profile = db.get_user_profile()
    if not profile:
        st.warning("Please set up your profile first.")
        st.stop()

    metrics     = get_health_metrics(profile)
    weight_log  = db.get_weight_log()
    workout_log = db.get_workout_log()
    calorie_log = db.get_calorie_log()
    weight_info = da.weight_trend_analysis(weight_log)
    cal_info    = da.calorie_trend_analysis(calorie_log)
    wk_info     = da.workout_frequency_analysis(workout_log)
    tdee        = metrics["tdee"]
    cal_target  = metrics["macros"]["calorie_target"]
    current_w   = profile["weight_kg"]
    raw_tw      = profile.get("target_weight", 0)
    goal_w      = float(raw_tw) if raw_tw and float(raw_tw) >= 30 else metrics["ideal"]["Average"]

    tab_pred, tab_cal, tab_recs = st.tabs([
        "⚖️ Weight Predictor",
        "🔥 Calorie Predictor",
        "💡 Recommendations",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: WEIGHT PREDICTOR — simple & visual
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pred:
        st.markdown("### ⚖️ What Will My Weight Be?")
        st.caption("Move the sliders below and instantly see where your weight is headed in 30 days.")

        # ── Plain-language sliders ────────────────────────────────────────────
        avg_intake_default = int(cal_info.get("avg_daily_intake", cal_target))
        sl1, sl2 = st.columns(2)
        with sl1:
            st.markdown("**🍽️ How much will you eat per day?**")
            p_daily_cal = st.slider(
                "Daily Calories",
                min_value=1000, max_value=4000,
                value=avg_intake_default, step=50,
                label_visibility="collapsed",
                help="Your planned average daily food intake",
            )
            # Plain-language label
            cal_vs_tdee = p_daily_cal - tdee
            if abs(cal_vs_tdee) <= 100:
                intake_label, intake_color = "Maintenance level — weight stays stable", "#6366F1"
            elif cal_vs_tdee < -400:
                intake_label, intake_color = "Big deficit — faster weight loss", "#10B981"
            elif cal_vs_tdee < 0:
                intake_label, intake_color = "Small deficit — gradual weight loss", "#34D399"
            elif cal_vs_tdee > 400:
                intake_label, intake_color = "Big surplus — weight will increase", "#EF4444"
            else:
                intake_label, intake_color = "Small surplus — slight weight gain", "#F59E0B"
            st.markdown(
                "<div style='background:" + intake_color + "11; border-left:3px solid " + intake_color + ";"
                "border-radius:8px; padding:8px 12px; margin-top:4px;'>"
                "<span style='color:" + intake_color + "; font-size:0.82rem; font-weight:600;'>"
                + str(p_daily_cal) + " kcal/day &nbsp;·&nbsp; "
                + ("+" if cal_vs_tdee >= 0 else "") + str(int(cal_vs_tdee)) + " vs your TDEE"
                + "<br>" + intake_label + "</span></div>",
                unsafe_allow_html=True
            )

        with sl2:
            st.markdown("**🏋️ How much will you exercise per day?**")
            p_workout_cal = st.slider(
                "Exercise Calories",
                min_value=0, max_value=1000,
                value=200, step=25,
                label_visibility="collapsed",
                help="Average calories you'll burn through exercise each day",
            )
            if p_workout_cal == 0:
                ex_label, ex_color = "No exercise", "#64748B"
            elif p_workout_cal < 200:
                ex_label, ex_color = "Light activity (walk, yoga)", "#F59E0B"
            elif p_workout_cal < 400:
                ex_label, ex_color = "Moderate exercise (30–45 min)", "#6366F1"
            elif p_workout_cal < 600:
                ex_label, ex_color = "Active training (45–60 min)", "#10B981"
            else:
                ex_label, ex_color = "Intense training (60+ min)", "#EF4444"
            st.markdown(
                "<div style='background:" + ex_color + "11; border-left:3px solid " + ex_color + ";"
                "border-radius:8px; padding:8px 12px; margin-top:4px;'>"
                "<span style='color:" + ex_color + "; font-size:0.82rem; font-weight:600;'>"
                + str(p_workout_cal) + " kcal burned &nbsp;·&nbsp; " + ex_label
                + "</span></div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ── Run prediction ────────────────────────────────────────────────────
        try:
            if weight_log:
                hist_df = da.weight_log_to_dataframe(weight_log).reset_index(drop=True)
                hist_df["day_index"]          = range(len(hist_df))
                hist_df["avg_calories_eaten"] = p_daily_cal
                hist_df["calories_burned"]    = p_workout_cal
                hist_df["workouts_today"]     = 1
            else:
                hist_df = ml_model.generate_synthetic_weight_data(start_weight=current_w)

            future_df   = weight_predictor.predict_future(
                hist_df, days_ahead=30,
                daily_calories=p_daily_cal,
                daily_workout_cal=p_workout_cal,
            )
            pred_30d    = round(float(future_df["predicted_weight_kg"].iloc[-1]), 2)
            pred_change = round(pred_30d - current_w, 2)
            going_right = (pred_change < 0 and goal_w < current_w) or \
                          (pred_change > 0 and goal_w > current_w) or \
                          (abs(pred_change) < 0.3)

            # ── BIG RESULT CARD ───────────────────────────────────────────────
            res_color = "#10B981" if going_right else "#EF4444"
            res_arrow = "📉" if pred_change < 0 else "📈" if pred_change > 0 else "➡️"
            res_verb  = "lose" if pred_change < 0 else "gain" if pred_change > 0 else "maintain"
            abs_change = abs(pred_change)

            st.markdown("### 🎯 Your 30-Day Prediction")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("⚖️ Today",           f"{current_w} kg")
            r2.metric("📅 In 30 Days",      f"{pred_30d} kg",
                      delta=f"{pred_change:+.2f} kg")
            r3.metric("🎯 Your Goal",       f"{goal_w} kg")
            kg_left = round(abs(pred_30d - goal_w), 2)
            r4.metric("📏 Still to Goal",   f"{kg_left} kg")

            # Plain-language verdict
            st.markdown(
                "<div style='background:" + res_color + "11; border:2px solid " + res_color + "44;"
                "border-left:6px solid " + res_color + "; border-radius:14px;"
                "padding:20px 24px; margin:12px 0;'>"
                "<div style='font-size:2rem; margin-bottom:8px;'>" + res_arrow + "</div>"
                "<div style='color:#E2E8F0; font-size:1.1rem; font-weight:700; margin-bottom:6px;'>"
                "If you keep eating <strong>" + str(p_daily_cal) + " kcal/day</strong> and burning "
                "<strong>" + str(p_workout_cal) + " kcal/day</strong> from exercise:</div>"
                "<div style='color:" + res_color + "; font-size:1.4rem; font-weight:900;'>"
                "You will " + res_verb + " <strong>" + str(abs_change) + " kg</strong> in 30 days"
                " → reaching <strong>" + str(pred_30d) + " kg</strong></div>"
                "<div style='color:#94A3B8; font-size:0.82rem; margin-top:8px;'>"
                + ("✅ You're on track toward your goal!" if going_right
                   else "⚠️ Try adjusting your intake or exercise to get back on track.")
                + "</div></div>",
                unsafe_allow_html=True
            )

            # ── Prediction chart ──────────────────────────────────────────────
            weight_df = da.weight_log_to_dataframe(weight_log)
            if not weight_df.empty:
                weight_df["rolling_avg"] = weight_df["weight_kg"].rolling(7, min_periods=1).mean()

            fig = charts.weight_prediction_chart(
                weight_df,
                list(future_df["date"]),
                list(future_df["predicted_weight_kg"]),
            )
            # Add goal weight line
            fig.add_hline(
                y=goal_w, line_dash="dot", line_color="#10B981", opacity=0.7,
                annotation_text=f"Goal: {goal_w} kg",
                annotation_font_color="#10B981",
            )
            fig.update_layout(
                title="📈 Your Weight History + 30-Day ML Prediction",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                height=340,
                margin=dict(l=10, r=10, t=44, b=10),
                xaxis=dict(showgrid=False, color="#64748B"),
                yaxis=dict(showgrid=True, gridcolor="#1E293B", color="#64748B"),
            )
            st.plotly_chart(fig, use_container_width=True, key="pred_weight_chart")

            # ── What-if scenarios table ───────────────────────────────────────
            st.markdown("### 🤔 What If I Changed My Plan?")
            st.caption("See how different calorie levels would affect your weight in 30 days.")

            scenarios = [
                ("🥗 Strict Diet",      p_daily_cal - 500, p_workout_cal),
                ("✅ Current Plan",     p_daily_cal,       p_workout_cal),
                ("🍔 Eat 200 More",     p_daily_cal + 200, p_workout_cal),
                ("🏃 + 200 kcal Exercise", p_daily_cal,    p_workout_cal + 200),
                ("💪 Best Case",        p_daily_cal - 300, p_workout_cal + 200),
            ]

            scenario_rows = []
            for label, s_cal, s_ex in scenarios:
                try:
                    hdf = hist_df.copy()
                    hdf["avg_calories_eaten"] = s_cal
                    hdf["calories_burned"]    = s_ex
                    fdf = weight_predictor.predict_future(
                        hdf, days_ahead=30,
                        daily_calories=s_cal, daily_workout_cal=s_ex,
                    )
                    s_pred   = round(float(fdf["predicted_weight_kg"].iloc[-1]), 2)
                    s_change = round(s_pred - current_w, 2)
                    s_sign   = "+" if s_change > 0 else ""
                    scenario_rows.append({
                        "Scenario":      label,
                        "Daily Intake":  f"{s_cal} kcal",
                        "Exercise":      f"{s_ex} kcal",
                        "Predicted Wt":  f"{s_pred} kg",
                        "Change":        f"{s_sign}{s_change} kg",
                    })
                except Exception:
                    pass

            if scenario_rows:
                st.dataframe(pd.DataFrame(scenario_rows),
                             use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

        # ── Retrain button (tucked at bottom) ────────────────────────────────
        with st.expander("⚙️ Advanced — Retrain ML Model"):
            st.caption("Retrain the model with your actual logged data for better accuracy.")
            if st.button("🔄 Retrain ML Models with My Data", key="retrain_btn"):
                with st.spinner("Training…"):
                    try:
                        if weight_log and len(weight_log) >= 5:
                            hist_df2 = da.weight_log_to_dataframe(weight_log).reset_index(drop=True)
                            hist_df2["day_index"]          = range(len(hist_df2))
                            hist_df2["avg_calories_eaten"] = cal_info.get("avg_daily_intake", 2000)
                            hist_df2["calories_burned"]    = wk_info.get("total_calories_burned", 0) / 7
                            hist_df2["workouts_today"]     = 1
                            results = ml_model.train_and_save_all_models(hist_df2)
                        else:
                            results = ml_model.train_and_save_all_models()
                        st.success(
                            "✅ Done! Weight model accuracy: "
                            + str(results['weight_predictor'].get('mae')) + " kg error | "
                            + "Calorie model: "
                            + str(results['calorie_burn_predictor'].get('mae')) + " kcal error"
                        )
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retraining failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: CALORIE BURN PREDICTOR — simple & visual
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cal:
        st.markdown("### 🔥 How Many Calories Will I Burn?")
        st.caption("Pick an exercise, set the duration, and instantly see your estimated calorie burn.")

        cb1, cb2, cb3 = st.columns(3)
        pred_ex  = cb1.selectbox("🏃 Choose Exercise", wt.get_all_exercises(), key="pred_ex")
        pred_dur = cb2.number_input("⏱️ Duration (minutes)",
                                     min_value=5, max_value=300, value=30, step=5, key="pred_dur")
        pred_int = cb3.selectbox("⚡ Intensity", ["Low", "Moderate", "High"], index=1, key="pred_int")

        ex_category = wt.get_category_for_exercise(pred_ex)
        met_val     = wt.EXERCISE_MET.get(pred_ex, 5.0)

        try:
            formula_cal = wt.estimate_calories_with_intensity(pred_ex, pred_dur, current_w, pred_int)
            ml_cal      = calorie_predictor.predict(pred_dur, current_w, met_val)

            # ── Big result ────────────────────────────────────────────────────
            st.markdown("### 🎯 Your Estimated Burn")
            cb_r1, cb_r2, cb_r3 = st.columns(3)
            cb_r1.metric("🔥 Calories Burned",    f"{formula_cal} kcal",
                         delta=f"{pred_int} intensity")
            cb_r2.metric("⏱️ Duration",           f"{pred_dur} min")
            cb_r3.metric("🤖 ML Estimate",         f"{ml_cal} kcal",
                         delta="AI model")

            # Plain-language equivalent
            pizza_slices  = round(formula_cal / 285)
            choc_bars     = round(formula_cal / 230)
            minutes_walk  = round(formula_cal / 4.5)

            st.markdown("#### 🍕 What does that burn equal?")
            eq1, eq2, eq3 = st.columns(3)
            eq1.metric("🍕 Pizza Slices",     str(pizza_slices) + " slices",  delta="≈285 kcal each")
            eq2.metric("🍫 Chocolate Bars",   str(choc_bars) + " bars",       delta="≈230 kcal each")
            eq3.metric("🚶 Equiv. Walking",   str(minutes_walk) + " min",     delta="at brisk pace")

            st.divider()

            # ── Duration vs Calories chart ────────────────────────────────────
            st.markdown("#### 📊 How Burn Changes With Duration")
            durations   = list(range(10, 121, 10))
            cals_by_dur = [wt.estimate_calories_with_intensity(pred_ex, d, current_w, pred_int)
                           for d in durations]

            fig_dur = go.Figure()
            fig_dur.add_trace(go.Scatter(
                x=durations, y=cals_by_dur,
                mode="lines+markers",
                line=dict(color="#6366F1", width=3),
                marker=dict(size=8, color="#8B5CF6"),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
                name="Calories Burned",
            ))
            # Mark current selection
            fig_dur.add_trace(go.Scatter(
                x=[pred_dur], y=[formula_cal],
                mode="markers+text",
                marker=dict(size=14, color="#F59E0B", symbol="star"),
                text=[f"You: {formula_cal} kcal"],
                textposition="top center",
                textfont=dict(color="#F59E0B", size=12),
                showlegend=False,
            ))
            fig_dur.update_layout(
                title=pred_ex + " — Calories Burned vs Duration",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E2E8F0"),
                height=280,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(title="Duration (min)", showgrid=False, color="#64748B"),
                yaxis=dict(title="Calories", showgrid=True, gridcolor="#1E293B", color="#64748B"),
                showlegend=False,
            )
            st.plotly_chart(fig_dur, use_container_width=True, key="pred_cal_dur_chart")

            st.divider()

            # ── Exercise comparison table ─────────────────────────────────────
            st.markdown("#### 🏅 Compare with Similar Exercises")
            cat_exercises = [e for e in wt.get_all_exercises()
                             if wt.get_category_for_exercise(e) == ex_category][:8]
            comp_rows = []
            for ex in cat_exercises:
                c = wt.estimate_calories_with_intensity(ex, pred_dur, current_w, pred_int)
                comp_rows.append({
                    "Exercise":             ex,
                    f"Calories ({pred_dur} min)": f"{c} kcal",
                    "vs Your Pick":         ("⭐ Selected" if ex == pred_ex
                                             else ("🔥 Burns More" if c > formula_cal
                                                   else "💧 Burns Less")),
                })
            comp_rows.sort(key=lambda x: float(x[f"Calories ({pred_dur} min)"].split()[0]), reverse=True)
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: AI RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_recs:
        st.markdown("### 💡 Your Personalised Recommendations")
        st.caption("Based on your profile, goals, and recent activity — here's exactly what to do next.")

        predicted_w30 = None
        try:
            if weight_log:
                hist_df_r = da.weight_log_to_dataframe(weight_log).reset_index(drop=True)
                hist_df_r["day_index"]          = range(len(hist_df_r))
                hist_df_r["avg_calories_eaten"] = cal_info.get("avg_daily_intake", cal_target)
                hist_df_r["calories_burned"]    = 300
                hist_df_r["workouts_today"]     = 1
                fut = weight_predictor.predict_future(hist_df_r, days_ahead=30)
                predicted_w30 = float(fut["predicted_weight_kg"].iloc[-1])
        except Exception:
            pass

        recommendations = ai_rec.generate_recommendations(
            bmi=metrics["bmi"],
            bmi_category=metrics["bmi_cat"],
            goal=profile["goal"],
            activity_level=profile["activity_level"],
            weekly_workouts=wk_info.get("sessions_per_week", 0),
            avg_daily_calories=cal_info.get("avg_daily_intake", 0),
            calorie_target=cal_target,
            weight_trend=weight_info.get("trend_direction", "stable") if weight_info else "stable",
            weight_change_kg=weight_info.get("change", 0) if weight_info else 0,
            body_fat_pct=metrics["body_fat"],
            gender=profile["gender"],
            age=profile["age"],
            predicted_weight_30d=predicted_w30,
            current_weight=current_w,
        )

        for rec in recommendations:
            badge_color = ai_rec.priority_badge_color(rec.priority)
            # Build action items HTML safely with string concat (no f-string loops)
            items_html = ""
            for item in rec.action_items:
                items_html += "<li style='margin:5px 0; color:#CBD5E1; font-size:0.87rem;'>" + item + "</li>"

            st.markdown(
                "<div style='background:linear-gradient(135deg,#1E293B,#0F172A);"
                "border:1px solid #334155; border-left:5px solid " + badge_color + ";"
                "border-radius:14px; padding:20px 22px; margin-bottom:14px;'>"
                "<div style='display:flex; align-items:flex-start; gap:14px; margin-bottom:10px;'>"
                "<span style='font-size:2rem; line-height:1;'>" + rec.icon + "</span>"
                "<div style='flex:1;'>"
                "<div style='color:#E2E8F0; font-weight:700; font-size:1.05rem; margin-bottom:6px;'>"
                + rec.title + "</div>"
                "<span style='background:" + badge_color + "; color:white; font-size:0.7rem;"
                "padding:3px 10px; border-radius:20px; font-weight:700; margin-right:6px;'>"
                + rec.priority + " Priority</span>"
                "<span style='background:#1E293B; color:#94A3B8; font-size:0.7rem;"
                "padding:3px 10px; border-radius:20px; border:1px solid #334155;'>"
                + rec.category + "</span>"
                "</div></div>"
                "<p style='color:#94A3B8; margin:0 0 12px; font-size:0.9rem; line-height:1.5;'>"
                + rec.description + "</p>"
                "<ul style='margin:0; padding-left:18px;'>" + items_html + "</ul>"
                "</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI COACH (Rule-Based Chatbot)
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "💬 AI Coach":
    import re as _re
    import datetime as _dt

    st.markdown("<h1>💬 AI Coach</h1>", unsafe_allow_html=True)
    st.caption("Your personal fitness assistant — ask anything about workouts, nutrition, and your health.")

    # ── Load user context ─────────────────────────────────────────────────────
    profile     = db.get_user_profile()
    metrics     = get_health_metrics(profile) if profile else None
    weight_log  = db.get_weight_log()
    workout_log = db.get_workout_log()
    calorie_log = db.get_calorie_log()

    # ── Context summary for bot ───────────────────────────────────────────────
    if profile and metrics:
        wk_sum   = wt.weekly_workout_summary(workout_log)
        today_bal = cc.daily_calorie_balance(
            calorie_log, workout_log, metrics["tdee"])
        ctx = {
            "name":         profile["name"],
            "age":          profile["age"],
            "gender":       profile["gender"],
            "weight":       profile["weight_kg"],
            "height":       profile["height_cm"],
            "goal":         profile["goal"],
            "activity":     profile["activity_level"],
            "bmi":          metrics["bmi"],
            "bmi_cat":      metrics["bmi_cat"],
            "tdee":         int(metrics["tdee"]),
            "cal_target":   metrics["macros"]["calorie_target"],
            "protein_g":    metrics["macros"]["protein_g"],
            "carbs_g":      metrics["macros"]["carbs_g"],
            "fat_g":        metrics["macros"]["fat_g"],
            "body_fat":     metrics["body_fat"],
            "water":        metrics["water"],
            "sessions_7d":  wk_sum["total_sessions"],
            "cal_7d":       wk_sum["total_calories"],
            "intake_today": int(today_bal["intake"]),
            "net_today":    int(today_bal["net"]),
        }
    else:
        ctx = None

    # ══════════════════════════════════════════════════════════════════════════
    # RULE-BASED RESPONSE ENGINE
    # ══════════════════════════════════════════════════════════════════════════
    def bot_response(msg: str, ctx: dict) -> str:
        m = msg.lower().strip()

        # ── Greeting ──────────────────────────────────────────────────────────
        if _re.search(r"\b(hi|hello|hey|howdy|sup|good morning|good evening)\b", m):
            name = ctx["name"] if ctx else "there"
            return (
                f"Hey {name}! 👋 I'm your AI Fitness Coach.\n\n"
                "I can help you with:\n"
                "• 🏋️ Workout suggestions\n"
                "• 🍽️ Nutrition & calorie advice\n"
                "• ⚖️ Your BMI, TDEE & body metrics\n"
                "• 💧 Hydration & recovery tips\n"
                "• 🎯 Goal-based recommendations\n\n"
                "What would you like to know?"
            )

        # ── BMI ───────────────────────────────────────────────────────────────
        if _re.search(r"\bbmi\b", m):
            if ctx:
                color_word = {"Underweight": "low", "Normal weight": "healthy",
                              "Overweight": "slightly high", "Obese": "high"}.get(ctx["bmi_cat"], "")
                return (
                    f"📊 **Your BMI is {ctx['bmi']} — {ctx['bmi_cat']}**\n\n"
                    f"That's {color_word} compared to the healthy range of 18.5–24.9.\n\n"
                    + ("✅ Great job maintaining a healthy weight!" if ctx["bmi_cat"] == "Normal weight"
                       else "💡 Focus on your nutrition and workout plan to move toward the healthy range.")
                )
            return "📊 BMI (Body Mass Index) measures body fat based on height and weight. Healthy range is 18.5–24.9. Set up your profile to see yours!"

        # ── Weight / goal ─────────────────────────────────────────────────────
        if _re.search(r"\b(my weight|current weight|how much do i weigh)\b", m):
            if ctx:
                return f"⚖️ Your current logged weight is **{ctx['weight']} kg**."
            return "⚖️ Set up your profile to track your weight!"

        if _re.search(r"\b(goal|target|lose weight|gain muscle|weight loss|fat loss)\b", m):
            if ctx:
                goal = ctx["goal"]
                tdee = ctx["tdee"]
                if goal == "Lose weight":
                    return (
                        f"🔥 **Goal: Lose Weight**\n\n"
                        f"Your TDEE is **{tdee} kcal/day**. To lose ~0.5 kg/week, eat around "
                        f"**{tdee - 550} kcal/day** (a 550 kcal deficit).\n\n"
                        "✅ Tips:\n"
                        "• Prioritise protein (keeps you full)\n"
                        "• Add 3–4 cardio sessions per week\n"
                        "• Track every meal — logging = awareness\n"
                        "• Sleep 7–8 hours (poor sleep increases hunger hormones)"
                    )
                elif goal == "Gain muscle":
                    return (
                        f"💪 **Goal: Gain Muscle**\n\n"
                        f"Eat **{tdee + 300} kcal/day** (a 300 kcal surplus) and hit "
                        f"**{ctx['protein_g']} g of protein/day**.\n\n"
                        "✅ Tips:\n"
                        "• Lift heavy — progressive overload is key\n"
                        "• Train each muscle group 2× per week\n"
                        "• Sleep is when muscles actually grow\n"
                        "• Creatine monohydrate is the most evidence-backed supplement"
                    )
                elif goal == "Maintain weight":
                    return (
                        f"⚖️ **Goal: Maintain Weight**\n\n"
                        f"Eat around your TDEE: **{tdee} kcal/day**.\n\n"
                        "✅ Tips:\n"
                        "• Mix cardio + strength training\n"
                        "• Stay consistent with meal timing\n"
                        "• Weigh yourself weekly to catch drift early"
                    )
                else:
                    return f"🎯 Your goal is **{goal}**. Stay consistent and track your progress daily!"
            return "🎯 Set up your profile first to get goal-specific advice!"

        # ── Calories ──────────────────────────────────────────────────────────
        if _re.search(r"\b(calorie|calories|kcal|how much (should i|to) eat|daily intake)\b", m):
            if ctx:
                surplus = ctx["net_today"]
                status  = "deficit ✅" if surplus < 0 else "surplus ⚠️"
                return (
                    f"🍽️ **Your Calorie Summary**\n\n"
                    f"• Daily target: **{ctx['cal_target']} kcal**\n"
                    f"• TDEE (maintenance): **{ctx['tdee']} kcal**\n"
                    f"• Today's intake: **{ctx['intake_today']} kcal** ({status} of {abs(surplus)} kcal)\n\n"
                    f"💡 Aim to stay within ±100 kcal of your target for best results."
                )
            return "🍽️ Your calorie target is based on your TDEE × goal adjustment. Set up your profile to get your personal number!"

        # ── Macros ────────────────────────────────────────────────────────────
        if _re.search(r"\b(macro|protein|carb|fat|nutrition|nutrients)\b", m):
            if ctx:
                return (
                    f"🥗 **Your Daily Macro Targets**\n\n"
                    f"• 🥩 Protein: **{ctx['protein_g']} g** (builds & preserves muscle)\n"
                    f"• 🍚 Carbs: **{ctx['carbs_g']} g** (primary energy source)\n"
                    f"• 🧈 Fat: **{ctx['fat_g']} g** (hormones & absorption)\n\n"
                    f"💡 Protein is the most important macro — hit it every day.\n"
                    f"Best sources: chicken, eggs, tuna, Greek yogurt, lentils, tofu."
                )
            return "🥗 Macros are protein, carbs, and fat. Set up your profile to see your personalised targets!"

        # ── TDEE ──────────────────────────────────────────────────────────────
        if _re.search(r"\b(tdee|maintenance calories|how many calories (do i|should i) burn)\b", m):
            if ctx:
                return (
                    f"🔥 **Your TDEE is {ctx['tdee']} kcal/day**\n\n"
                    "TDEE = Total Daily Energy Expenditure — the total calories your body burns each day including all activity.\n\n"
                    f"• Eat **less** than {ctx['tdee']} → lose weight\n"
                    f"• Eat **equal** to {ctx['tdee']} → maintain weight\n"
                    f"• Eat **more** than {ctx['tdee']} → gain weight"
                )
            return "🔥 TDEE is your total daily calorie burn. Set up your profile to see yours!"

        # ── Water / hydration ─────────────────────────────────────────────────
        if _re.search(r"\b(water|hydration|drink|hydrate)\b", m):
            if ctx:
                glasses = round(ctx["water"] * 1000 / 250)
                return (
                    f"💧 **Your Daily Water Target: {ctx['water']} L ({glasses} glasses)**\n\n"
                    "✅ Hydration tips:\n"
                    "• Drink a full glass first thing in the morning\n"
                    "• Drink 500 ml 30–60 min before exercise\n"
                    "• Carry a water bottle everywhere\n"
                    "• Check urine colour — pale yellow = good hydration\n"
                    "• Increase by 500 ml for every 30 min of intense exercise"
                )
            return "💧 A general rule: drink 35 ml per kg of body weight per day. More if you exercise!"

        # ── Workout suggestions ───────────────────────────────────────────────
        if _re.search(r"\b(workout|exercise|training|should i (do|train)|what (exercise|workout))\b", m):
            if ctx:
                sessions = ctx["sessions_7d"]
                goal     = ctx["goal"]
                if goal == "Lose weight":
                    plan = (
                        "🏃 **Workout Plan for Fat Loss**\n\n"
                        "• Mon: HIIT (20–30 min)\n"
                        "• Tue: Strength training (full body)\n"
                        "• Wed: Brisk walk or cycling (45 min)\n"
                        "• Thu: Rest or yoga\n"
                        "• Fri: HIIT or running\n"
                        "• Sat: Strength training\n"
                        "• Sun: Active rest (walk/stretch)\n\n"
                        f"📊 You've done **{sessions} sessions this week**. "
                        + ("Great work! 🔥" if sessions >= 4 else "Try to hit 4–5 sessions! 💪")
                    )
                elif goal == "Gain muscle":
                    plan = (
                        "💪 **Workout Plan for Muscle Gain**\n\n"
                        "• Mon: Chest + Triceps\n"
                        "• Tue: Back + Biceps\n"
                        "• Wed: Rest or light cardio\n"
                        "• Thu: Legs + Glutes\n"
                        "• Fri: Shoulders + Core\n"
                        "• Sat: Full body or weak points\n"
                        "• Sun: Rest\n\n"
                        "🔑 Key: Progressive overload — add weight or reps each week."
                    )
                else:
                    plan = (
                        "🏋️ **General Fitness Plan**\n\n"
                        "• 3× strength training per week\n"
                        "• 2× cardio (30–45 min each)\n"
                        "• 1× active recovery (yoga, stretching)\n"
                        "• 1× full rest day\n\n"
                        "Adjust intensity based on how you feel each week."
                    )
                return plan
            return "🏋️ I recommend 3–5 workouts per week mixing cardio and strength. Set up your profile for a personalised plan!"

        # ── Rest & recovery ───────────────────────────────────────────────────
        if _re.search(r"\b(rest|recovery|rest day|sore|soreness|overtraining)\b", m):
            return (
                "😴 **Rest & Recovery Tips**\n\n"
                "• Muscles grow during rest, not during training\n"
                "• Take 1–2 rest days per week minimum\n"
                "• Sleep 7–9 hours — it's your most powerful recovery tool\n"
                "• Active recovery (walk, yoga, stretching) beats complete inactivity\n"
                "• Soreness = normal. Sharp pain = stop and check\n"
                "• Foam rolling and cold/warm contrast showers help recovery"
            )

        # ── Sleep ────────────────────────────────────────────────────────────
        if _re.search(r"\b(sleep|sleeping|insomnia|tired|fatigue)\b", m):
            return (
                "🌙 **Sleep & Fitness**\n\n"
                "Sleep is the #1 most underrated fitness tool.\n\n"
                "✅ Why it matters:\n"
                "• Growth hormone is released during deep sleep → muscle repair\n"
                "• Poor sleep → higher cortisol → more fat storage\n"
                "• Less sleep → more hunger (ghrelin) → harder to diet\n\n"
                "💡 Tips for better sleep:\n"
                "• Same bedtime every night (even weekends)\n"
                "• No screens 30 min before bed\n"
                "• Keep room cool (18–20°C is optimal)\n"
                "• Avoid caffeine after 2 PM"
            )

        # ── Body fat ─────────────────────────────────────────────────────────
        if _re.search(r"\b(body fat|fat percentage|bf%|bodyfat)\b", m):
            if ctx:
                bf  = ctx["body_fat"]
                gen = ctx["gender"]
                if gen == "Male":
                    cat = "Athletic" if bf < 14 else "Fit" if bf < 18 else "Average" if bf < 25 else "Above average"
                else:
                    cat = "Athletic" if bf < 21 else "Fit" if bf < 25 else "Average" if bf < 32 else "Above average"
                return (
                    f"💪 **Your estimated body fat: {bf}% — {cat}**\n\n"
                    "Body fat ranges (Male / Female):\n"
                    "• Athletic: 6–13% / 14–20%\n"
                    "• Fit: 14–17% / 21–24%\n"
                    "• Average: 18–24% / 25–31%\n"
                    "• Above average: 25%+ / 32%+\n\n"
                    "💡 To lower body fat: combine calorie deficit + strength training."
                )
            return "💪 Body fat % is estimated from your BMI, age, and gender. Set up your profile to see yours!"

        # ── Protein ──────────────────────────────────────────────────────────
        if _re.search(r"\b(how much protein|protein (target|goal|intake|per day))\b", m):
            if ctx:
                return (
                    f"🥩 **Your Protein Target: {ctx['protein_g']} g/day**\n\n"
                    f"That's ~{round(ctx['protein_g'] / ctx['weight'], 1)} g per kg of body weight.\n\n"
                    "🏆 Best protein sources:\n"
                    "• Chicken breast (31g per 100g)\n"
                    "• Tuna canned (26g per 100g)\n"
                    "• Greek yogurt (10g per 100g)\n"
                    "• Eggs (6g per egg)\n"
                    "• Lentils (9g per 100g cooked)\n"
                    "• Cottage cheese (11g per 100g)"
                )
            return "🥩 General rule: 1.2–2.0 g of protein per kg of body weight per day. Higher end for muscle building."

        # ── Supplements ───────────────────────────────────────────────────────
        if _re.search(r"\b(supplement|creatine|protein powder|whey|vitamin|fish oil)\b", m):
            return (
                "💊 **Evidence-Based Supplements**\n\n"
                "The ones actually worth taking:\n\n"
                "🥇 **Creatine monohydrate** — best supplement for strength & muscle. 3–5g/day.\n"
                "🥈 **Whey protein** — convenient way to hit protein targets. Not magic, just protein.\n"
                "🥉 **Vitamin D** — most people are deficient (especially indoors). 1000–2000 IU/day.\n"
                "🐟 **Omega-3 fish oil** — reduces inflammation, supports heart health.\n"
                "☕ **Caffeine** — proven to boost performance. Use before workouts.\n\n"
                "⚠️ Everything else (fat burners, pre-workouts, BCAAs) has weak evidence."
            )

        # ── Motivation ────────────────────────────────────────────────────────
        if _re.search(r"\b(motivat|unmotivated|lazy|can't (get|stay) motivated|no motivation|give up|quit)\b", m):
            return (
                "🔥 **Feeling Unmotivated? That's Normal.**\n\n"
                "Motivation comes and goes — discipline is what gets results.\n\n"
                "💡 Practical tips:\n"
                "• Make it stupidly easy — commit to just 10 minutes\n"
                "• Track your streak — you won't want to break it\n"
                "• Find a workout you actually enjoy\n"
                "• Set a small, achievable goal this week\n"
                "• Remember your *why* — why did you start?\n\n"
                "🏆 \"The hardest part is showing up. Once you're there, you'll do it.\""
            )

        # ── Progress / stats ──────────────────────────────────────────────────
        if _re.search(r"\b(my progress|how am i doing|my stats|my data|summary)\b", m):
            if ctx:
                sessions = ctx["sessions_7d"]
                intake   = ctx["intake_today"]
                target   = ctx["cal_target"]
                net      = ctx["net_today"]
                on_track = "✅ On track!" if abs(net) < 200 else ("⚠️ Over target" if net > 0 else "⚠️ Under target")
                return (
                    f"📊 **Your Quick Summary**\n\n"
                    f"⚖️ Weight: **{ctx['weight']} kg** | BMI: **{ctx['bmi']} ({ctx['bmi_cat']})**\n"
                    f"🎯 Goal: **{ctx['goal']}**\n"
                    f"🏋️ Workouts this week: **{sessions}**\n"
                    f"🍽️ Today's intake: **{intake} kcal** / target {target} kcal — {on_track}\n"
                    f"🔥 TDEE: **{ctx['tdee']} kcal/day**\n"
                    f"💪 Body fat: **{ctx['body_fat']}%**\n"
                    f"💧 Water target: **{ctx['water']} L/day**"
                )
            return "📊 Set up your profile to see your personalised stats!"

        # ── What can you do / help ────────────────────────────────────────────
        if _re.search(r"\b(what can you (do|help)|help me|how (do you|can you)|commands)\b", m):
            return (
                "💬 **I can help you with:**\n\n"
                "• 📊 **BMI** — type 'what is my bmi'\n"
                "• 🍽️ **Calories** — type 'how many calories should i eat'\n"
                "• 🥗 **Macros** — type 'what are my macros'\n"
                "• 🔥 **TDEE** — type 'what is my tdee'\n"
                "• 💧 **Water** — type 'how much water should i drink'\n"
                "• 🏋️ **Workouts** — type 'suggest a workout'\n"
                "• 😴 **Recovery** — type 'rest day tips'\n"
                "• 💊 **Supplements** — type 'what supplements should i take'\n"
                "• 🥩 **Protein** — type 'how much protein do i need'\n"
                "• 📊 **Progress** — type 'show my progress'\n"
                "• 🔥 **Goals** — type 'help me with my goal'"
            )

        # ── Default fallback ──────────────────────────────────────────────────
        return (
            "🤔 I'm not sure I understood that. Here are some things you can ask me:\n\n"
            "• 'What is my BMI?'\n"
            "• 'How many calories should I eat?'\n"
            "• 'What are my macros?'\n"
            "• 'Suggest a workout for me'\n"
            "• 'How much protein do I need?'\n"
            "• 'Show my progress'\n"
            "• 'What supplements should I take?'\n\n"
            "Type **'help'** to see everything I can do!"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CHAT UI
    # ══════════════════════════════════════════════════════════════════════════

    # ── Initialise chat history ───────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        name = profile["name"] if profile else "there"
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "text": (
                    f"👋 Hey {name}! I'm your AI Fitness Coach.\n\n"
                    "Ask me anything about your workouts, nutrition, goals, or health metrics.\n\n"
                    "Type **'help'** to see what I can do!"
                ),
            }
        ]

    # ── Quick-reply suggestion buttons ───────────────────────────────────────
    st.markdown("**⚡ Quick Questions:**")
    qr_cols = st.columns(4)
    quick_replies = [
        "What is my BMI?",
        "How many calories should I eat?",
        "Suggest a workout for me",
        "Show my progress",
        "How much protein do I need?",
        "What supplements should I take?",
        "How much water should I drink?",
        "Rest day tips",
    ]
    for i, qr in enumerate(quick_replies):
        with qr_cols[i % 4]:
            if st.button(qr, key=f"qr_{i}", use_container_width=True):
                st.session_state["chat_history"].append({"role": "user", "text": qr})
                reply = bot_response(qr, ctx)
                st.session_state["chat_history"].append({"role": "assistant", "text": reply})
                st.rerun()

    st.divider()

    # ── Chat history display ──────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                # User bubble — right aligned
                st.markdown(
                    "<div style='display:flex; justify-content:flex-end; margin:8px 0;'>"
                    "<div style='background:linear-gradient(135deg,#6366F1,#8B5CF6);"
                    "color:white; padding:12px 16px; border-radius:18px 18px 4px 18px;"
                    "max-width:70%; font-size:0.9rem; line-height:1.5;'>"
                    + msg["text"] +
                    "</div></div>",
                    unsafe_allow_html=True
                )
            else:
                # Bot bubble — left aligned
                text_html = msg["text"].replace("\n", "<br>")
                # Bold **text**
                text_html = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text_html)
                # Bullet points
                text_html = text_html.replace("• ", "&#8226; ")
                st.markdown(
                    "<div style='display:flex; justify-content:flex-start; margin:8px 0;'>"
                    "<div style='background:linear-gradient(135deg,#1E293B,#0F172A);"
                    "border:1px solid #334155; color:#E2E8F0; padding:12px 16px;"
                    "border-radius:18px 18px 18px 4px; max-width:75%;"
                    "font-size:0.9rem; line-height:1.6;'>"
                    "<span style='color:#6366F1; font-size:1.1rem; margin-right:6px;'>🤖</span>"
                    + text_html +
                    "</div></div>",
                    unsafe_allow_html=True
                )

    st.divider()

    # ── Input box ─────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_input = st.text_input(
                "Message",
                placeholder="Ask me anything about fitness, nutrition, or your health…",
                label_visibility="collapsed",
            )
        with btn_col:
            send = st.form_submit_button("Send 💬", use_container_width=True)

    if send and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "text": user_input.strip()})
        reply = bot_response(user_input.strip(), ctx)
        st.session_state["chat_history"].append({"role": "assistant", "text": reply})
        st.rerun()

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()