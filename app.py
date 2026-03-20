import streamlit as st
import numpy as np
import joblib
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Castor PDI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a140a;
    color: #dceadc;
}
.stApp { background-color: #0a140a; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0.8rem 2rem 0.5rem !important;
    max-width: 1200px !important;
}

/* ── Style Streamlit's column wrappers as panels ── */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    background: #0d1a0d;
    border: 1px solid #1a2e1a;
    border-radius: 14px;
    padding: 1rem 1.2rem 1.2rem !important;
}

/* ── Hero ── */
.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0 0.85rem;
    border-bottom: 1px solid #1a2e1a;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 900;
    color: #93cc8d;
    letter-spacing: -0.2px;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.7rem;
    color: #4a7a4a;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.hero-badge {
    background: #0d1f0d;
    border: 1px solid #254a25;
    border-radius: 8px;
    padding: 0.35rem 0.85rem;
    font-size: 0.68rem;
    color: #5aaf5a;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
    white-space: nowrap;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3d6a3d;
    border-left: 3px solid #2e5a2e;
    padding-left: 0.5rem;
    margin: 0.75rem 0 0.6rem;
    line-height: 1.2;
}
.sec-label:first-child { margin-top: 0; }

/* ── Inputs ── */
div[data-testid="stNumberInput"] { margin-bottom: 0 !important; }
div[data-testid="stNumberInput"] input {
    background: #101e10 !important;
    border: 1px solid #1e3320 !important;
    color: #cde8cd !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
    height: 36px !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #4a9f4a !important;
    box-shadow: 0 0 0 2px rgba(74,159,74,0.14) !important;
}
label {
    color: #6a9a6a !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
}
div[data-testid="stNumberInput"] button {
    background: #141f14 !important;
    border-color: #1e3320 !important;
    color: #4a9f4a !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2a5e2a 0%, #47961a 100%) !important;
    color: #f0fff0 !important;
    border: none !important;
    border-radius: 9px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    margin-top: 0.8rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #347a34 0%, #58b020 100%) !important;
    box-shadow: 0 4px 16px rgba(71,150,26,0.38) !important;
    transform: translateY(-1px) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 12px;
    padding: 1rem 1.2rem 0.9rem;
    text-align: center;
    border: 1px solid;
    margin-bottom: 0.75rem;
}
.result-subtitle {
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.2rem;
}
.result-pdi {
    font-family: 'Playfair Display', serif;
    font-size: 3.4rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.result-risk { font-size: 0.95rem; font-weight: 600; letter-spacing: 0.04em; }

.risk-low      { background:#091d0f; border-color:#1a5228; }
.risk-low      .result-subtitle { color:#3db860; }
.risk-low      .result-pdi      { color:#5dd870; }
.risk-low      .result-risk     { color:#4abf60; }

.risk-moderate { background:#151900; border-color:#424f00; }
.risk-moderate .result-subtitle { color:#a0b000; }
.risk-moderate .result-pdi      { color:#ccd800; }
.risk-moderate .result-risk     { color:#b0c400; }

.risk-high     { background:#1a0e00; border-color:#6a3a00; }
.risk-high     .result-subtitle { color:#d08000; }
.risk-high     .result-pdi      { color:#f5a010; }
.risk-high     .result-risk     { color:#e09010; }

.risk-severe   { background:#1a0000; border-color:#6a1515; }
.risk-severe   .result-subtitle { color:#c03030; }
.risk-severe   .result-pdi      { color:#f05050; }
.risk-severe   .result-risk     { color:#d84040; }

/* ── Risk bar ── */
.rbar-wrap {
    position: relative;
    height: 10px;
    border-radius: 6px;
    overflow: visible;
    margin: 0.3rem 0 0.25rem;
    background: linear-gradient(to right, #1e5228 0%, #424f00 25%, #6a3a00 50%, #6a1515 75%, #8a1a1a 100%);
}
.rbar-pointer {
    position: absolute;
    top: 50%;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 2px solid #0a140a;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 6px rgba(0,0,0,0.6);
}
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: #3a5a3a;
    margin-top: 0.1rem;
    margin-bottom: 0.65rem;
}

/* ── Advisory card ── */
.adv-card {
    background: #0a160a;
    border: 1px solid #1a2e1a;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
}
.adv-header {
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3d6a3d;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.adv-row {
    display: flex;
    gap: 0.5rem;
    align-items: flex-start;
    margin-bottom: 0.38rem;
}
.adv-icon { font-size: 0.9rem; flex-shrink: 0; margin-top: 0.05rem; }
.adv-text { font-size: 0.8rem; color: #8aaa8a; line-height: 1.45; }
.adv-text b { color: #b8d8b8; font-weight: 500; }

/* ── Derived chips ── */
.derived-row { display: flex; gap: 0.5rem; margin-top: 0.4rem; }
.dchip {
    flex: 1;
    background: #0a140a;
    border: 1px solid #1a2e1a;
    border-radius: 8px;
    padding: 0.4rem 0.5rem;
    text-align: center;
}
.dchip .dv {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: #6abf6a;
}
.dchip .dl {
    font-size: 0.59rem;
    color: #3d6a3d;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.05rem;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 1rem;
    text-align: center;
    gap: 0.5rem;
}
.empty-icon { font-size: 2.5rem; opacity: 0.35; }
.empty-text { font-size: 0.82rem; color: #365a36; max-width: 220px; line-height: 1.55; }

/* ── Footer ── */
.footer-bar {
    border-top: 1px solid #162416;
    padding: 0.75rem 0 0.5rem;
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
}
.footer-label {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2e4e2e;
    font-weight: 600;
    white-space: nowrap;
}
.footer-credits { display: flex; gap: 1.8rem; flex-wrap: wrap; }
.fc-person { display: flex; flex-direction: column; }
.fc-name { font-size: 0.78rem; font-weight: 600; color: #5aaa5a; }
.fc-role  { font-size: 0.67rem; color: #3a5a3a; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "castor_elastic_net_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file `castor_elastic_net_model.pkl` not found. "
                 "Place it in the same folder as app.py.")
        st.stop()
    return joblib.load(model_path)

model = load_model()


# ─── Risk data ───────────────────────────────────────────────────────────────
RISK_DATA = {
    "Low": {
        "class": "low", "icon": "🟢", "bar_color": "#5dd870",
        "action": "Routine monitoring",
        "action_icon": "🔍",
        "fungicide": "No fungicide application required",
        "fungicide_icon": "✅",
        "detail": "Crop conditions are currently favourable. Continue weekly observations and standard agronomic practices.",
    },
    "Moderate": {
        "class": "moderate", "icon": "🟡", "bar_color": "#ccd800",
        "action": "Field scouting; apply spray if symptoms appear",
        "action_icon": "👁️",
        "fungicide": "Apply Propiconazole @ 1.0–1.5 ml/L only if symptoms appear",
        "fungicide_icon": "⚗️",
        "detail": "Elevated disease risk. Inspect plants for early lesions every 3–4 days and monitor humidity trends closely.",
    },
    "High": {
        "class": "high", "icon": "🟠", "bar_color": "#f5a010",
        "action": "Preventive fungicide application",
        "action_icon": "⚠️",
        "fungicide": "Apply Propiconazole @ 1.5–2.0 ml/L",
        "fungicide_icon": "🧪",
        "detail": "Significant disease pressure detected. Begin protective spray schedule immediately; reduce canopy humidity where possible.",
    },
    "Severe": {
        "class": "severe", "icon": "🔴", "bar_color": "#f05050",
        "action": "Immediate intervention and strict monitoring",
        "action_icon": "🚨",
        "fungicide": "Apply Propiconazole @ 2.0–2.5 ml/L",
        "fungicide_icon": "🧪",
        "detail": "Epidemic conditions detected. Apply curative doses urgently, consult your agronomist, and consider emergency harvest of mature portions.",
    },
}

def classify_risk(pdi: float) -> str:
    if pdi < 25:   return "Low"
    elif pdi < 50: return "Moderate"
    elif pdi < 75: return "High"
    else:          return "Severe"


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-title">🌿 Castor PDI Predictor</div>
        <div class="hero-sub">Plant Disease Index · Elastic Net Regression Model</div>
    </div>
    <div class="hero-badge">MIT Manipal × UAS Bangalore</div>
</div>
""", unsafe_allow_html=True)


# ─── Two columns ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="medium")


# ════════════════════════════
# LEFT — Inputs
# ════════════════════════════
with left_col:

    st.markdown('<p class="sec-label">🌡 Temperature</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        temp_max = st.number_input("Max Temp (°C)", min_value=0.0, max_value=60.0,
                                   value=35.0, step=0.1, format="%.1f")
    with c2:
        temp_min = st.number_input("Min Temp (°C)", min_value=0.0, max_value=60.0,
                                   value=22.0, step=0.1, format="%.1f")

    st.markdown('<p class="sec-label">💧 Relative Humidity</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        rh_max = st.number_input("Max RH (%)", min_value=0.0, max_value=100.0,
                                  value=85.0, step=0.1, format="%.1f")
    with c4:
        rh_min = st.number_input("Min RH (%)", min_value=0.0, max_value=100.0,
                                  value=45.0, step=0.1, format="%.1f")

    st.markdown('<p class="sec-label">🌧 Atmospheric & Rainfall</p>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0,
                                     value=8.0, step=0.1, format="%.1f")
    with c6:
        sunshine = st.number_input("Sunshine (hrs/day)", min_value=0.0, max_value=24.0,
                                   value=7.5, step=0.1, format="%.1f")

    c7, c8 = st.columns(2)
    with c7:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0,
                                   value=50.0, step=0.1, format="%.1f")
    with c8:
        rainy_days = st.number_input("Rainy Days", min_value=0, max_value=31,
                                     value=5, step=1)

    predict_clicked = st.button("🌾  Predict Disease Index")


# ════════════════════════════
# RIGHT — Results
# ════════════════════════════
with right_col:

    st.markdown('<p class="sec-label">📊 Prediction Output</p>', unsafe_allow_html=True)

    if predict_clicked:
        temp_diff    = temp_max - temp_min
        rh_diff      = rh_max - rh_min
        rain_per_day = rainfall / max(rainy_days, 1)

        features = np.array([[
            temp_max, temp_min,
            rh_max,   rh_min,
            wind_speed, sunshine,
            rainfall,  rainy_days,
            temp_diff, rh_diff, rain_per_day
        ]])

        pdi  = float(model.predict(features)[0])
        pdi  = max(0.0, min(pdi, 100.0))
        risk = classify_risk(pdi)
        rd   = RISK_DATA[risk]

        # Result card
        st.markdown(f"""
        <div class="result-card risk-{rd['class']}">
            <div class="result-subtitle">Plant Disease Index</div>
            <div class="result-pdi">{pdi:.1f}</div>
            <div class="result-risk">{rd['icon']} {risk} Risk</div>
        </div>
        """, unsafe_allow_html=True)

        # Risk bar
        st.markdown(f"""
        <div class="rbar-wrap">
            <div class="rbar-pointer" style="left:{pdi}%;background:{rd['bar_color']};"></div>
        </div>
        <div class="bar-labels">
            <span>0 · Low</span>
            <span>25 · Moderate</span>
            <span>50 · High</span>
            <span>75 · Severe</span>
            <span>100</span>
        </div>
        """, unsafe_allow_html=True)

        # Advisory
        st.markdown(f"""
        <div class="adv-card">
            <div class="adv-header">📋 Advisory &amp; Fungicide Recommendation</div>
            <div class="adv-row">
                <span class="adv-icon">{rd['action_icon']}</span>
                <div class="adv-text"><b>Advisory Action:</b> {rd['action']}</div>
            </div>
            <div class="adv-row">
                <span class="adv-icon">{rd['fungicide_icon']}</span>
                <div class="adv-text"><b>Fungicide:</b> {rd['fungicide']}</div>
            </div>
            <div class="adv-row">
                <span class="adv-icon">📝</span>
                <div class="adv-text">{rd['detail']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Derived feature chips
        st.markdown('<p class="sec-label">⚙️ Derived Features Used</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="derived-row">
            <div class="dchip">
                <div class="dv">{temp_diff:.1f}°C</div>
                <div class="dl">Temp Diff</div>
            </div>
            <div class="dchip">
                <div class="dv">{rh_diff:.1f}%</div>
                <div class="dl">RH Diff</div>
            </div>
            <div class="dchip">
                <div class="dv">{rain_per_day:.1f}mm</div>
                <div class="dl">Rain / Day</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🌱</div>
            <div class="empty-text">
                Enter the weather parameters on the left and click
                <b>Predict Disease Index</b> to see the results and advisory.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
    <div class="footer-label">Developed by</div>
    <div class="footer-credits">
        <div class="fc-person">
            <span class="fc-name">Anurag Dhole</span>
            <span class="fc-role">Researcher, MIT — Manipal</span>
        </div>
        <div class="fc-person">
            <span class="fc-name">Dr. Jadesha G</span>
            <span class="fc-role">Asst. Professor, GKVK — UAS, Bangalore</span>
        </div>
        <div class="fc-person">
            <span class="fc-name">Dr. Deepak D.</span>
            <span class="fc-role">Professor, MIT — Manipal</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
