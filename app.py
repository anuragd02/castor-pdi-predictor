import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Castor BGM Forecasting & Advisory System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f0f7f0;
    color: #1a2e1a;
}
.stApp { background-color: #f0f7f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1rem 2rem 0.5rem !important;
    max-width: 1200px !important;
}

/* ── Column panels ── */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    background: #ffffff;
    border-radius: 18px;
    padding: 1.4rem 1.5rem 1.5rem !important;
    box-shadow: 0 2px 12px rgba(30,90,30,0.10), 0 0 0 1.5px #c8e0c8;
}

/* ══════════════════════════════════════
   NUMBER INPUT — only recolour, never
   touch layout / sizing / overflow
   ══════════════════════════════════════ */

/* Label */
div[data-testid="stNumberInput"] label {
    color: #1e5c1e !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
}

/* Input text box */
div[data-testid="stNumberInput"] input {
    background-color: #f6fbf6 !important;
    color: #1a3a1a !important;
    font-weight: 700 !important;
    border-color: #90c890 !important;
    box-shadow: none !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2e8a2e !important;
    box-shadow: 0 0 0 2px rgba(46,138,46,0.15) !important;
}

/* Step buttons */
div[data-testid="stNumberInput"] button {
    color: #1e6e1e !important;
    background-color: #e8f5e8 !important;
    border-color: #90c890 !important;
}
div[data-testid="stNumberInput"] button:hover {
    background-color: #d0ebd0 !important;
    border-color: #2e8a2e !important;
}

div[data-testid="stNumberInput"] { margin-bottom: 2px !important; }

/* ── Hero ── */
.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(30,90,30,0.10), 0 0 0 1.5px #c8e0c8;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 900;
    color: #1a4f1a;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.68rem;
    color: #3a7a3a;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    font-weight: 700;
}
.hero-badge {
    background: #eaf4ea;
    border: 1.5px solid #6abf6a;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-size: 0.68rem;
    color: #1e5c1e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 800;
    white-space: nowrap;
}

/* ── Section labels ── */
.sec-label {
    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1e5c1e;
    border-left: 4px solid #3a9a3a;
    padding: 0.25rem 0.55rem;
    margin: 0.9rem 0 0.6rem;
    line-height: 1.2;
    background: #f2faf2;
    border-radius: 0 6px 6px 0;
    display: block;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a5c1a 0%, #2e961e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.1em !important;
    padding: 0.7rem 1rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: 0 3px 12px rgba(30,150,30,0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #246e24 0%, #3ab828 100%) !important;
    box-shadow: 0 5px 18px rgba(46,150,30,0.40) !important;
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
    font-size: 0.62rem;
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

.risk-low      { background:#edfbf0; border-color:#28a745; }
.risk-low      .result-subtitle { color:#1a6b2a; }
.risk-low      .result-pdi      { color:#1a8530; }
.risk-low      .result-risk     { color:#1a6b2a; }

.risk-moderate { background:#fffde8; border-color:#c8a800; }
.risk-moderate .result-subtitle { color:#7a6500; }
.risk-moderate .result-pdi      { color:#9a7f00; }
.risk-moderate .result-risk     { color:#7a6500; }

.risk-high     { background:#fff4e5; border-color:#e07800; }
.risk-high     .result-subtitle { color:#8a4a00; }
.risk-high     .result-pdi      { color:#b05e00; }
.risk-high     .result-risk     { color:#8a4a00; }

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
    margin: 0.4rem 0 0.3rem;
    background: linear-gradient(to right,
        #28a745 0%,  #28a745 20%,
        #c8a800 20%, #c8a800 40%,
        #e07800 40%, #e07800 70%,
        #cc2222 70%, #cc2222 100%);
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
}
.rbar-pointer {
    position: absolute;
    top: 50%;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 3px solid #ffffff;
    box-shadow: 0 0 0 1.5px #666, 0 2px 6px rgba(0,0,0,0.25);
    transform: translate(-50%, -50%);
}
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: #4a6a4a;
    font-weight: 700;
    margin-top: 0.2rem;
    margin-bottom: 0.75rem;
}

/* ── Advisory card ── */
.adv-card {
    background: #f8fcf8;
    border: 1.5px solid #b8d8b8;
    border-radius: 12px;
    padding: 0.9rem 1.05rem;
    margin-bottom: 0.65rem;
}
.adv-header {
    font-size: 0.64rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1e5c1e;
    font-weight: 800;
    margin-bottom: 0.65rem;
    padding-bottom: 0.45rem;
    border-bottom: 1.5px solid #cce8cc;
}
.adv-row {
    display: flex;
    gap: 0.55rem;
    align-items: flex-start;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px dashed #ddeedd;
}
.adv-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.adv-icon { font-size: 0.95rem; flex-shrink: 0; margin-top: 0.08rem; }
.adv-text { font-size: 0.82rem; color: #2a3a2a; line-height: 1.5; }
.adv-text b { color: #1a4a1a; font-weight: 800; }

/* ── Derived chips ── */
.derived-row { display: flex; gap: 0.6rem; margin-top: 0.5rem; }
.dchip {
    flex: 1;
    background: #f2faf2;
    border: 1.5px solid #b8d8b8;
    border-radius: 10px;
    padding: 0.5rem 0.5rem 0.45rem;
    text-align: center;
}
.dchip .dv {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 900;
    color: #1e5c1e;
}
.dchip .dl {
    font-size: 0.58rem;
    color: #3a6a3a;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.06rem;
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
.empty-icon { font-size: 3rem; opacity: 0.35; }
.empty-text { font-size: 0.84rem; color: #5a8a5a; max-width: 220px; line-height: 1.65; font-weight: 600; }

/* ── Footer ── */
.footer-bar {
    background: #ffffff;
    border-radius: 14px;
    padding: 0.85rem 1.5rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
    box-shadow: 0 2px 12px rgba(30,90,30,0.08), 0 0 0 1.5px #c8e0c8;
}
.footer-label {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a7a4a;
    font-weight: 800;
    white-space: nowrap;
}
.footer-credits { display: flex; gap: 2rem; flex-wrap: wrap; }
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


# ─── Risk data ───────────────────────────────────────────────────────────────
RISK_DATA = {
    "Low": {
        "class": "low", "icon": "🟢", "bar_color": "#28a745", "range": "0 – 20",
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
        "class": "moderate", "icon": "🟡", "bar_color": "#c8a800", "range": "20 – 40",
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
        "class": "high", "icon": "🟠", "bar_color": "#e07800", "range": "40 – 70",
        "action": "Preventive fungicide application recommended",
        "action_icon": "⚠️",
        "fungicide": "Apply Propiconazole @ 1.5–2.0 ml/L",
        "fungicide_icon": "🧪",
        "detail": "Significant disease pressure detected. Begin protective spray schedule immediately.",
        "detail_icon": "📝",
        "reason": "High rainfall (100–150 mm) and humidity (75–80%), favourable for disease development.",
        "reason_icon": "🌧️",
    },
    "Very High": {
        "class": "veryhigh", "icon": "🔴", "bar_color": "#cc2222", "range": "70 – 100",
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
</div>
""", unsafe_allow_html=True)


# ─── Columns ─────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="medium")

# ══════════════════════════════
# LEFT — Inputs
# ══════════════════════════════
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


# ══════════════════════════════
# RIGHT — Results
# ══════════════════════════════
with right_col:
    st.markdown('<p class="sec-label">📊 Prediction Output</p>', unsafe_allow_html=True)

    if predict_clicked:
        temp_diff    = temp_max - temp_min
        rh_diff      = rh_max - rh_min
        rain_per_day = rainfall / max(rainy_days, 1)

        features = np.array([[
            temp_max, temp_min, rh_max, rh_min,
            wind_speed, sunshine, rainfall, rainy_days,
            temp_diff, rh_diff, rain_per_day
        ]])

        pdi  = float(model.predict(features)[0])
        pdi  = max(0.0, min(pdi, 100.0))
        risk = classify_risk(pdi)
        rd   = RISK_DATA[risk]

        st.markdown(f"""
        <div class="result-card risk-{rd['class']}">
            <div class="result-subtitle">BGM Disease Severity Index</div>
            <div class="result-pdi">{pdi:.1f}</div>
            <div class="result-risk">{rd['icon']} {risk} Risk &nbsp;·&nbsp; {rd['range']}</div>
        </div>
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
                Enter weather parameters on the left and click
                <b>Predict Disease Index</b> to see BGM severity and advisory.
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
