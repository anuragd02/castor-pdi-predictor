import streamlit as st
import numpy as np
import joblib
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Castor PDI Predictor",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1a0f;
    color: #e8f0e8;
}

.stApp { background-color: #0f1a0f; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #2a3d2a;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: #a8d5a2;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7a9e7a;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section headers ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5c8c5c;
    border-left: 3px solid #4a7a4a;
    padding-left: 0.6rem;
    margin: 1.8rem 0 1rem;
}

/* ── Input boxes ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSlider"] {
    accent-color: #6abf6a;
}
div[data-testid="stNumberInput"] input {
    background: #192b19 !important;
    border: 1px solid #2e4a2e !important;
    color: #d4e8d4 !important;
    border-radius: 8px !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #6abf6a !important;
    box-shadow: 0 0 0 2px rgba(106,191,106,0.18) !important;
}

/* ── Labels ── */
label, .stNumberInput label {
    color: #9dbf9d !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3d7a3d, #5aa55a) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    margin-top: 1.2rem !important;
    transition: all 0.25s ease !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #4a8f4a, #6abf6a) !important;
    box-shadow: 0 4px 20px rgba(90,165,90,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Result card ── */
.result-card {
    margin: 2rem 0 1rem;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    border: 1px solid;
}
.result-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    font-weight: 500;
}
.result-pdi {
    font-family: 'Playfair Display', serif;
    font-size: 4.5rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-label {
    font-size: 1.15rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* Low */
.risk-low    { background: #0d2b1a; border-color: #2a6b3a; }
.risk-low    .result-title  { color: #5bb870; }
.risk-low    .result-pdi    { color: #6dd67d; }
.risk-low    .result-label  { color: #5bb870; }

/* Moderate */
.risk-moderate { background: #1e2a00; border-color: #5a6e00; }
.risk-moderate .result-title  { color: #c8d400; }
.risk-moderate .result-pdi    { color: #dde800; }
.risk-moderate .result-label  { color: #c8d400; }

/* High */
.risk-high   { background: #2b1500; border-color: #8a4a00; }
.risk-high   .result-title  { color: #f5a623; }
.risk-high   .result-pdi    { color: #ffb833; }
.risk-high   .result-label  { color: #f5a623; }

/* Severe */
.risk-severe { background: #2b0000; border-color: #8a1a1a; }
.risk-severe .result-title  { color: #e05555; }
.risk-severe .result-pdi    { color: #ff6b6b; }
.risk-severe .result-label  { color: #e05555; }

/* ── Risk meter bar ── */
.risk-bar-wrap {
    background: #1c2e1c;
    border-radius: 8px;
    height: 10px;
    width: 100%;
    margin: 1rem 0 0.4rem;
    overflow: hidden;
}
.risk-bar-fill {
    height: 10px;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* ── Interpretation box ── */
.interp-box {
    background: #141f14;
    border: 1px solid #2a3d2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #9dbf9d;
    line-height: 1.6;
}
.interp-box strong { color: #c5e0c5; }

/* ── Divider ── */
hr { border-color: #2a3d2a !important; margin: 1.5rem 0 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #3d5a3d;
    padding: 2rem 0 1rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "castor_elastic_net_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file `castor_elastic_net_model.pkl` not found. "
                 "Please place it in the same folder as app.py.")
        st.stop()
    return joblib.load(model_path)

model = load_model()


# ─── Helpers ────────────────────────────────────────────────────────────────
def classify_risk(pdi: float):
    if pdi < 25:
        return "Low", "low", "🟢", "#6dd67d"
    elif pdi < 50:
        return "Moderate", "moderate", "🟡", "#dde800"
    elif pdi < 75:
        return "High", "high", "🟠", "#ffb833"
    else:
        return "Severe", "severe", "🔴", "#ff6b6b"

def get_advice(risk: str) -> str:
    advice = {
        "Low": (
            "<strong>Crop looks healthy.</strong> Continue routine monitoring. "
            "No immediate disease management intervention is required under the current weather conditions."
        ),
        "Moderate": (
            "<strong>Early warning zone.</strong> Scout fields for initial disease symptoms. "
            "Consider prophylactic spray of recommended fungicide if humid conditions persist beyond 3–5 days."
        ),
        "High": (
            "<strong>Elevated disease pressure.</strong> Apply recommended fungicide/bactericide immediately. "
            "Increase field scouting frequency to every 3–4 days and reduce canopy humidity where possible."
        ),
        "Severe": (
            "<strong>Epidemic conditions detected.</strong> Urgent intervention required. "
            "Apply curative doses of registered pesticides, notify your agronomist, and consider emergency harvesting "
            "of mature portions to limit yield loss."
        ),
    }
    return advice[risk]


# ─── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🌿 Castor PDI Predictor</div>
    <div class="hero-sub">Disease Index Forecasting · Elastic Net Model</div>
</div>
""", unsafe_allow_html=True)


# ─── Input Form ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Temperature Parameters</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    temp_max = st.number_input("Max Temperature (°C)", min_value=0.0, max_value=60.0, value=35.0, step=0.1, format="%.1f")
with col2:
    temp_min = st.number_input("Min Temperature (°C)", min_value=0.0, max_value=60.0, value=22.0, step=0.1, format="%.1f")

st.markdown('<div class="section-label">Humidity Parameters</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    rh_max = st.number_input("Max Relative Humidity (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1, format="%.1f")
with col4:
    rh_min = st.number_input("Min Relative Humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.1, format="%.1f")

st.markdown('<div class="section-label">Atmospheric & Rainfall Parameters</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=8.0, step=0.1, format="%.1f")
with col6:
    sunshine = st.number_input("Sunshine Hours (hrs/day)", min_value=0.0, max_value=24.0, value=7.5, step=0.1, format="%.1f")

col7, col8 = st.columns(2)
with col7:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=50.0, step=0.1, format="%.1f")
with col8:
    rainy_days = st.number_input("Rainy Days", min_value=0, max_value=31, value=5, step=1)


# ─── Predict ────────────────────────────────────────────────────────────────
predict_clicked = st.button("🌾 Predict Disease Index")

if predict_clicked:
    # Derived features (mirrors the training code)
    temp_diff     = temp_max - temp_min
    rh_diff       = rh_max   - rh_min
    rain_per_day  = rainfall / max(rainy_days, 1)

    features = np.array([[
        temp_max, temp_min,
        rh_max,   rh_min,
        wind_speed, sunshine,
        rainfall,  rainy_days,
        temp_diff, rh_diff, rain_per_day
    ]])

    pdi = float(model.predict(features)[0])
    pdi = max(0.0, min(pdi, 100.0))   # clamp to [0, 100]

    risk_label, risk_class, risk_icon, bar_color = classify_risk(pdi)
    bar_pct = min(pdi, 100)

    # Result card
    st.markdown(f"""
    <div class="result-card risk-{risk_class}">
        <div class="result-title">Predicted Plant Disease Index</div>
        <div class="result-pdi">{pdi:.1f}</div>
        <div class="result-label">{risk_icon} {risk_label} Risk</div>
    </div>
    """, unsafe_allow_html=True)

    # Risk meter
    st.markdown(f"""
    <div class="risk-bar-wrap">
        <div class="risk-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#5c8c5c;margin-bottom:0.8rem;">
        <span>0 — Low</span><span>25 — Moderate</span><span>50 — High</span><span>75 — Severe</span>
    </div>
    """, unsafe_allow_html=True)

    # Advisory
    st.markdown(f"""
    <div class="interp-box">
        🌱 <strong>Advisory:</strong><br>{get_advice(risk_label)}
    </div>
    """, unsafe_allow_html=True)

    # Derived values expander
    with st.expander("🔍 View derived feature values used for prediction"):
        st.markdown(f"""
        | Feature | Value |
        |---|---|
        | Temp Difference (Max − Min) | **{temp_diff:.2f} °C** |
        | RH Difference (Max − Min) | **{rh_diff:.2f} %** |
        | Rain per Rainy Day | **{rain_per_day:.2f} mm/day** |
        """)

st.markdown('<div class="footer">Castor Disease Risk Tool · Elastic Net Regression · For research use only</div>',
            unsafe_allow_html=True)
