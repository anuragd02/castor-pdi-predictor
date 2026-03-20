import streamlit as st
import numpy as np
import joblib
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Castor BGM Forecasting & Advisory System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #ffffff;
    color: #1a2e1a;
}
.stApp { background-color: #ffffff; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0.8rem 2rem 0.5rem !important;
    max-width: 1200px !important;
}

/* ── Column panels ── */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    background: #f5faf5;
    border: 2px solid #c0d8c0;
    border-radius: 16px;
    padding: 1.1rem 1.3rem 1.3rem !important;
}

/* ── Hero ── */
.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.4rem 0 0.9rem;
    border-bottom: 2px solid #b0cfb0;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 900;
    color: #1a4f1a;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.7rem;
    color: #3a7a3a;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    font-weight: 700;
}
.hero-badge {
    background: #e8f5e8;
    border: 2px solid #6abf6a;
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-size: 0.68rem;
    color: #1e5c1e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 800;
    white-space: nowrap;
}

/* ── Section labels ── */
.sec-label {
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1e5c1e;
    border-left: 4px solid #2e8a2e;
    padding-left: 0.55rem;
    margin: 0.85rem 0 0.65rem;
    line-height: 1.2;
}

/* ════════════════════════════════════════
   NUMBER INPUT — clean single border fix
   ════════════════════════════════════════ */

/* Remove Streamlit's default wrapper border/shadow entirely */
div[data-testid="stNumberInput"] > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* The actual text input box */
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    border: 2px solid #a8cfa8 !important;
    border-right: none !important;
    border-radius: 10px 0 0 10px !important;
    color: #1a3a1a !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    height: 42px !important;
    padding: 0 0.75rem !important;
    outline: none !important;
    box-shadow: none !important;
    transition: border-color 0.18s ease !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2e8a2e !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Step buttons container — share the border */
div[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
div[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] {
    background: #e8f5e8 !important;
    border: 2px solid #a8cfa8 !important;
    border-left: none !important;
    color: #1e6e1e !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    height: 42px !important;
    width: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
}
div[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"] {
    border-right: 1px solid #c0d8c0 !important;
}
div[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] {
    border-radius: 0 10px 10px 0 !important;
}
div[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"]:hover,
div[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"]:hover {
    background: #d4edda !important;
}

/* Labels */
label, div[data-testid="stNumberInput"] label {
    color: #1e5c1e !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    margin-bottom: 4px !important;
}

div[data-testid="stNumberInput"] { margin-bottom: 0 !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a5c1a 0%, #2e961e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1em !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    margin-top: 0.9rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #246e24 0%, #3ab828 100%) !important;
    box-shadow: 0 5px 18px rgba(46,150,30,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 14px;
    padding: 1.1rem 1.2rem 1rem;
    text-align: center;
    border: 2px solid;
    margin-bottom: 0.8rem;
}
.result-subtitle {
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.result-pdi {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.result-risk { font-size: 1rem; font-weight: 800; letter-spacing: 0.04em; }

/* Low */
.risk-low { background:#edfbf0; border-color:#28a745; }
.risk-low .result-subtitle { color:#1a6b2a; }
.risk-low .result-pdi      { color:#1a8530; }
.risk-low .result-risk     { color:#1a6b2a; }

/* Moderate */
.risk-moderate { background:#fffde8; border-color:#c8a800; }
.risk-moderate .result-subtitle { color:#7a6500; }
.risk-moderate .result-pdi      { color:#9a7f00; }
.risk-moderate .result-risk     { color:#7a6500; }

/* High */
.risk-high { background:#fff4e5; border-color:#e07800; }
.risk-high .result-subtitle { color:#8a4a00; }
.risk-high .result-pdi      { color:#b05e00; }
.risk-high .result-risk     { color:#8a4a00; }

/* Very High */
.risk-veryhigh { background:#fff0f0; border-color:#cc2222; }
.risk-veryhigh .result-subtitle { color:#8a0000; }
.risk-veryhigh .result-pdi      { color:#bb1111; }
.risk-veryhigh .result-risk     { color:#8a0000; }

/* ── Risk bar ── */
.rbar-wrap {
    position: relative;
    height: 12px;
    border-radius: 8px;
    overflow: visible;
    margin: 0.35rem 0 0.3rem;
    background: linear-gradient(to right,
        #28a745 0%,   #28a745 20%,
        #c8a800 20%,  #c8a800 40%,
        #e07800 40%,  #e07800 70%,
        #cc2222 70%,  #cc2222 100%);
}
.rbar-pointer {
    position: absolute;
    top: 50%;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    border: 3px solid #ffffff;
    box-shadow: 0 0 0 1.5px #888, 0 2px 6px rgba(0,0,0,0.25);
    transform: translate(-50%, -50%);
}
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: #4a6a4a;
    font-weight: 700;
    margin-top: 0.15rem;
    margin-bottom: 0.7rem;
}

/* ── Advisory card ── */
.adv-card {
    background: #ffffff;
    border: 2px solid #b0d0b0;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.65rem;
}
.adv-header {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1e5c1e;
    font-weight: 800;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #d0e8d0;
}
.adv-row {
    display: flex;
    gap: 0.55rem;
    align-items: flex-start;
    margin-bottom: 0.42rem;
}
.adv-icon { font-size: 0.95rem; flex-shrink: 0; margin-top: 0.05rem; }
.adv-text { font-size: 0.82rem; color: #2a3a2a; line-height: 1.5; font-weight: 400; }
.adv-text b { color: #1a3a1a; font-weight: 800; }

/* ── Derived chips ── */
.derived-row { display: flex; gap: 0.55rem; margin-top: 0.4rem; }
.dchip {
    flex: 1;
    background: #ffffff;
    border: 2px solid #b0d0b0;
    border-radius: 10px;
    padding: 0.45rem 0.5rem;
    text-align: center;
}
.dchip .dv {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 900;
    color: #1e5c1e;
}
.dchip .dl {
    font-size: 0.59rem;
    color: #3a6a3a;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.05rem;
    font-weight: 700;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3.5rem 1rem;
    text-align: center;
    gap: 0.6rem;
}
.empty-icon { font-size: 2.8rem; opacity: 0.4; }
.empty-text { font-size: 0.84rem; color: #5a8a5a; max-width: 220px; line-height: 1.6; font-weight: 600; }

/* ── Footer ── */
.footer-bar {
    border-top: 2px solid #c0d8c0;
    padding: 0.8rem 0 0.5rem;
    margin-top: 0.8rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
}
.footer-label {
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7a4a;
    font-weight: 800;
    white-space: nowrap;
}
.footer-credits { display: flex; gap: 1.8rem; flex-wrap: wrap; }
.fc-person { display: flex; flex-direction: column; }
.fc-name { font-size: 0.8rem; font-weight: 800; color: #1a5c1a; }
.fc-role  { font-size: 0.67rem; color: #4a7a4a; font-style: italic; font-weight: 600; }
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


# ─── Risk data (0–20 / 20–40 / 40–70 / 70–100) ───────────────────────────────
RISK_DATA = {
    "Low": {
        "class": "low",
        "icon": "🟢",
        "bar_color": "#28a745",
        "range": "0 – 20",
        "action": "Routine monitoring; no fungicide application required",
        "action_icon": "🔍",
        "fungicide": "Not required",
        "fungicide_icon": "✅",
        "detail": "Crop conditions are currently favourable. Continue weekly field observations and maintain standard agronomic practices.",
        "detail_icon": "📝",
        "reason": "Low rainfall (<60 mm) and low humidity (<70%), unfavourable for disease development.",
        "reason_icon": "🌤️",
    },
    "Moderate": {
        "class": "moderate",
        "icon": "🟡",
        "bar_color": "#c8a800",
        "range": "20 – 40",
        "action": "Field scouting; apply fungicide if symptoms appear",
        "action_icon": "👁️",
        "fungicide": "Apply Propiconazole @ 1.0–1.5 ml/L only if symptoms appear",
        "fungicide_icon": "⚗️",
        "detail": "Elevated disease risk. Inspect plants carefully for early lesions or discolouration every 3–4 days and monitor humidity trends.",
        "detail_icon": "📝",
        "reason": "Moderate rainfall (60–100 mm) and humidity (70–75%), favourable for initial disease development.",
        "reason_icon": "🌦️",
    },
    "High": {
        "class": "high",
        "icon": "🟠",
        "bar_color": "#e07800",
        "range": "40 – 70",
        "action": "Preventive fungicide application recommended",
        "action_icon": "⚠️",
        "fungicide": "Apply Propiconazole @ 1.5–2.0 ml/L",
        "fungicide_icon": "🧪",
        "detail": "Significant disease pressure detected. Begin protective spray schedule immediately and reduce canopy humidity where possible.",
        "detail_icon": "📝",
        "reason": "High rainfall (100–150 mm) and humidity (75–80%), favourable for disease development.",
        "reason_icon": "🌧️",
    },
    "Very High": {
        "class": "veryhigh",
        "icon": "🔴",
        "bar_color": "#cc2222",
        "range": "70 – 100",
        "action": "Immediate fungicide application and strict monitoring required",
        "action_icon": "🚨",
        "fungicide": "Apply Propiconazole @ 2.0–2.5 ml/L (critical stage)",
        "fungicide_icon": "🧪",
        "detail": "Epidemic conditions detected. Apply curative doses urgently, consult your agronomist, and consider emergency harvest of mature portions to limit yield loss.",
        "detail_icon": "📝",
        "reason": "Very high rainfall (≥150 mm) and high humidity (≥80%), highly conducive for rapid disease spread.",
        "reason_icon": "⛈️",
    },
}

def classify_risk(pdi: float) -> str:
    if pdi < 20:   return "Low"
    elif pdi < 40: return "Moderate"
    elif pdi < 70: return "High"
    else:          return "Very High"


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div>
        <div class="hero-title">🌿 Castor BGM Forecasting and Advisory System</div>
        <div class="hero-sub">Botrytis Gray Mould · Ensemble Regression Model</div>
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
        # Derived features — mirrors training pipeline exactly
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

        # ── Result card ──
        st.markdown(f"""
        <div class="result-card risk-{rd['class']}">
            <div class="result-subtitle">BGM Disease Severity Index</div>
            <div class="result-pdi">{pdi:.1f}</div>
            <div class="result-risk">{rd['icon']} {risk} Risk &nbsp;·&nbsp; {rd['range']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Risk bar ──
        st.markdown(f"""
        <div class="rbar-wrap">
            <div class="rbar-pointer" style="left:{pdi}%;background:{rd['bar_color']};"></div>
        </div>
        <div class="bar-labels">
            <span>0</span>
            <span>20 · Low</span>
            <span>40 · Moderate</span>
            <span>70 · High</span>
            <span>100 · Very High</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Advisory card — 4 points ──
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
                <span class="adv-icon">{rd['detail_icon']}</span>
                <div class="adv-text"><b>Details:</b> {rd['detail']}</div>
            </div>
            <div class="adv-row">
                <span class="adv-icon">{rd['reason_icon']}</span>
                <div class="adv-text"><b>Reason:</b> {rd['reason']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Derived feature chips ──
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
                <b>Predict Disease Index</b> to see the BGM severity and advisory.
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
