"""
app.py — Strategic Customer Churn Dashboard (Streamlit)
Kompatibel dengan artefak dari UAS_Churn_Prediction_v2.ipynb
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategic Churn Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Kustom (Tema Gelap Eksekutif) ─────────────────────────────────────────
st.markdown("""
<style>
    /* Global Background & Fonts */
    .stApp {
        background-color: #0E1117;
        color: #F8FAFC;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px; 
        border-radius: 12px; 
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #334155;
    }
    .metric-label { font-size: 14px; opacity: 0.7; margin-bottom: 4px; font-weight: 600; color: #CBD5E1; }
    .metric-value { font-size: 32px; font-weight: 800; color: #38BDF8; }
    
    /* Specific Churn Cards */
    .churn-high { 
        background: linear-gradient(135deg, #450A0A 0%, #2A0404 100%); 
        border-color: #7F1D1D; 
    }
    .churn-high .metric-value { color: #F87171; }
    
    .churn-low  { 
        background: linear-gradient(135deg, #064E3B 0%, #022C22 100%); 
        border-color: #065F46; 
    }
    .churn-low .metric-value { color: #34D399; }
    
    /* Headers & Typography */
    .section-header {
        font-size: 18px; 
        font-weight: 600; 
        color: #E2E8F0;
        border-left: 4px solid #38BDF8; 
        padding-left: 12px; 
        margin: 24px 0 16px;
        background-color: #1E293B;
        padding-top: 8px;
        padding-bottom: 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Artefak ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_artifacts(path: str = "churn_model.joblib"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

artifacts = load_artifacts()

# ── Konstanta Referensi ───────────────────────────────────────────────────────
REF_DATE = pd.Timestamp("2025-01-01")


# ── Feature Engineering ───────────────────────────────────────────────────────
def process_input_data(raw_data, raw_columns):
    df = pd.DataFrame([raw_data])
    
    # Konversi tipe data tanggal
    signup_ts = pd.to_datetime(df["signup_date"])
    purchase_ts = pd.to_datetime(df["last_purchase_date"])
    
    # Feature engineering turunan
    df["days_since_signup"] = (REF_DATE - signup_ts).dt.days
    df["days_since_purchase"] = (REF_DATE - purchase_ts).dt.days
    df["signup_month"] = signup_ts.dt.month
    df["tenure_days"] = (purchase_ts - signup_ts).dt.days
    
    # Cek kupon
    df["has_coupon"] = df["coupon_code"].apply(lambda x: 0 if pd.isna(x) or x == "" else 1)
    
    # Pastikan semua kolom yang dibutuhkan model tersedia
    if raw_columns is not None:
        for col in raw_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Saring hanya kolom yang ada di raw_columns
        available_cols = [c for c in raw_columns if c in df.columns]
        return df[available_cols]
    
    return df


# ── Full Prediction Pipeline ──────────────────────────────────────────────────
def run_prediction(input_df, artifacts):
    model        = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    iqr_bounds   = artifacts.get("iqr_bounds", {})
    top_features = artifacts.get("top_features", [])
    scenario     = artifacts.get("scenario", "Full")

    df = input_df.copy()
    
    # 1. Outlier clipping (batas IQR)
    for col, (lo, hi) in iqr_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # 2. Impute + OHE + scale
    X_arr      = preprocessor.transform(df)
    feat_names = preprocessor.get_feature_names_out()
    X_enc      = pd.DataFrame(X_arr, columns=feat_names)

    # 3. Seleksi top features
    if scenario == "Tuning" and len(top_features) > 0:
        available = [f for f in top_features if f in X_enc.columns]
        X_enc = X_enc[available]

    # 4. Prediksi
    pred  = model.predict(X_enc)[0]
    proba = (model.predict_proba(X_enc)[0][1]
             if hasattr(model, "predict_proba") else float(pred))
    return int(pred), float(proba)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔮 Churn Predictor")
    st.caption("Strategic Customer Analysis Dashboard")
    st.divider()
    if artifacts:
        st.markdown("### 📦 Info Model")
        n_feat = (str(len(artifacts.get("top_features", [])))
                  if artifacts.get("scenario") == "Tuning" else "Semua")
        st.success(
            f"**Model:** {artifacts.get('model_name', '-')}\n\n"
            f"**Skenario:** {artifacts.get('scenario', '-')}\n\n"
            f"**Fitur Prediktor:** {n_feat}"
        )
    else:
        st.warning("⚠️ `churn_model.joblib` tidak ditemukan.")
    st.divider()
    st.caption("© 2026 · Business & Tech Analytics")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Predictive Customer Churn")
st.markdown("Input data interaksi dan finansial klien untuk mengevaluasi probabilitas churn dan menentukan langkah eskalasi sales selanjutnya.")
if artifacts is None:
    st.error("❌ File model `churn_model.joblib` belum ada di direktori yang sama.")
    st.stop()

raw_columns = artifacts.get("raw_columns", [])

# ── Form Berdasarkan Kolom CSV Asli (Vertikal / No Tabs) ──────────────────────
with st.form("form_prediksi"):
    
    # --- SECTION 1: Demografi & Wilayah ---
    st.markdown('<p class="section-header">🧑‍💼 Identitas & Wilayah</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    gender  = c1.selectbox("Gender", ["Male", "Female", "Other"])
    age     = c2.number_input("Age", min_value=10, max_value=100, value=30)
    country = c3.text_input("Country", value="UK")
    city    = c4.text_input("City", value="London")

    # --- SECTION 2: Informasi Langganan ---
    st.markdown('<p class="section-header">📅 Informasi Langganan</p>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    signup_date        = c5.date_input("Signup Date", value=date(2023, 1, 15))
    last_purchase_date = c6.date_input("Last Purchase Date", value=date(2024, 6, 1))
    subscription_type  = c7.selectbox("Subscription Type", ["Monthly", "Annual", "Basic"])
    is_premium_user    = c8.selectbox("Is Premium User?", [1, 0], format_func=lambda x: "Yes (1)" if x==1 else "No (0)")

    # --- SECTION 3: Sumber Akuisisi & Perangkat ---
    st.markdown('<p class="section-header">📱 Sumber Akuisisi & Perangkat</p>', unsafe_allow_html=True)
    c9, c10, c11 = st.columns([2, 2, 1])
    acquisition_channel = c9.selectbox("Acquisition Channel", ["Organic", "Email", "Facebook Ads", "Google Ads", "Referral"])
    device_type         = c10.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])

    # --- SECTION 4: Metrik Interaksi Web/Aplikasi ---
    st.markdown('<p class="section-header">🌐 Metrik Interaksi Web/Aplikasi</p>', unsafe_allow_html=True)
    c12, c13, c14 = st.columns(3)
    total_visits      = c12.number_input("Total Visits", min_value=0, value=10)
    avg_session_time  = c13.number_input("Avg Session Time (mins)", min_value=0.0, value=5.5, format="%.2f")
    pages_per_session = c14.number_input("Pages per Session", min_value=0.0, value=3.2, format="%.2f")

    # --- SECTION 5: Metrik Email & Kepuasan ---
    st.markdown('<p class="section-header">⭐ Metrik Email & Kepuasan</p>', unsafe_allow_html=True)
    c15, c16, c17, c18 = st.columns(4)
    email_open_rate    = c15.number_input("Email Open Rate", min_value=0.0, max_value=1.0, value=0.45, format="%.2f")
    email_click_rate   = c16.number_input("Email Click Rate", min_value=0.0, max_value=1.0, value=0.15, format="%.2f")
    satisfaction_score = c17.number_input("Satisfaction Score", min_value=0, max_value=5, value=4)
    nps_score          = c18.number_input("NPS Score", min_value=0, max_value=10, value=8)

    # --- SECTION 6: Finansial & Transaksi ---
    st.markdown('<p class="section-header">💳 Finansial & Transaksi</p>', unsafe_allow_html=True)
    c19, c20, c21, c22 = st.columns(4)
    total_spent                = c19.number_input("Total Spent ($)", min_value=0.0, value=1250.50, format="%.2f")
    avg_order_value            = c20.number_input("Avg Order Value ($)", min_value=0.0, value=85.20, format="%.2f")
    lifetime_value             = c21.number_input("Lifetime Value ($)", min_value=0.0, value=1500.00, format="%.2f")
    last_3_month_purchase_freq = c22.number_input("Last 3 Month Freq", min_value=0, value=2)

    # --- SECTION 7: Layanan & Promo ---
    st.markdown('<p class="section-header">🛍️ Layanan & Promo</p>', unsafe_allow_html=True)
    c23, c24, c25 = st.columns(3)
    payment_method      = c23.selectbox("Payment Method", ["Card", "PayPal", "UPI", "Bank Transfer"])
    discount_used       = c24.selectbox("Discount Used?", [1, 0], format_func=lambda x: "Yes (1)" if x==1 else "No (0)")
    coupon_code         = c25.text_input("Coupon Code", placeholder="Kosongkan jika tidak ada (opsional)")

    c26, c27, c28, c29 = st.columns(4)
    support_tickets          = c26.number_input("Support Tickets", min_value=0, value=1)
    refund_requested         = c27.selectbox("Refund Requested?", [1, 0], format_func=lambda x: "Yes (1)" if x==1 else "No (0)")
    delivery_delay_days      = c28.number_input("Delivery Delay Days", min_value=0, value=0)
    marketing_spend_per_user = c29.number_input("Marketing Spend ($)", min_value=0.0, value=25.50, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ Eksekusi Prediksi Churn", type="primary", use_container_width=True)


# ── Eksekusi Prediksi ─────────────────────────────────────────────────────────
if submitted:
    if last_purchase_date < signup_date:
        st.error("❌ Last Purchase Date tidak boleh mendahului Signup Date.")
        st.stop()

    # Data dictionary sesuai kolom CSV
    raw_data_input = {
        "gender": gender,
        "age": age,
        "country": country,
        "city": city,
        "signup_date": signup_date,
        "last_purchase_date": last_purchase_date,
        "acquisition_channel": acquisition_channel,
        "device_type": device_type,
        "subscription_type": subscription_type,
        "is_premium_user": is_premium_user,
        "total_visits": total_visits,
        "avg_session_time": avg_session_time,
        "pages_per_session": pages_per_session,
        "email_open_rate": email_open_rate,
        "email_click_rate": email_click_rate,
        "total_spent": total_spent,
        "avg_order_value": avg_order_value,
        "discount_used": discount_used,
        "coupon_code": coupon_code,
        "support_tickets": support_tickets,
        "refund_requested": refund_requested,
        "delivery_delay_days": delivery_delay_days,
        "payment_method": payment_method,
        "satisfaction_score": satisfaction_score,
        "nps_score": nps_score,
        "marketing_spend_per_user": marketing_spend_per_user,
        "lifetime_value": lifetime_value,
        "last_3_month_purchase_freq": last_3_month_purchase_freq
    }

    with st.spinner("Mengalkulasi proyeksi probabilitas..."):
        input_df = process_input_data(raw_data_input, raw_columns)
        pred, proba_churn = run_prediction(input_df, artifacts)

    # ── Tampilan Hasil ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Hasil Prediksi Analisis")

    is_churn   = pred == 1
    card_class = "churn-high" if is_churn else "churn-low"
    verdict    = "🚨 RISIKO CHURN" if is_churn else "✅ AMAN (RETENSI)"
    risk_label = ("🔴 Kritis" if proba_churn > 0.70 else
                  "🟡 Waspada" if proba_churn > 0.40 else "🟢 Terkendali")

    r1, r2, r3 = st.columns(3)
    r1.markdown(
        f'<div class="metric-card {card_class}">'
        f'<div class="metric-label">Proyeksi Pelanggan</div>'
        f'<div class="metric-value">{verdict}</div></div>',
        unsafe_allow_html=True)
    r2.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Probabilitas Churn</div>'
        f'<div class="metric-value">{proba_churn:.1%}</div></div>',
        unsafe_allow_html=True)
    r3.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Status Risiko</div>'
        f'<div class="metric-value">{risk_label}</div></div>',
        unsafe_allow_html=True)

    # Visualisasi Bar / Gauge
    st.markdown("**Pemetaan Skala Risiko**")
    fig, ax = plt.subplots(figsize=(8, 1.2))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.barh([""], [1], color="#1E293B", height=0.4, edgecolor="#334155")
    bar_color = ("#EF4444" if proba_churn > 0.70 else
                 "#F59E0B" if proba_churn > 0.40 else "#10B981")
    ax.barh([""], [proba_churn], color=bar_color, height=0.4)
    ax.set_xlim(0, 1)
    
    ax.axvline(0.4, color="#FBBF24", linewidth=1.5, linestyle=":", alpha=0.7)
    ax.axvline(0.7, color="#F87171", linewidth=1.5, linestyle=":", alpha=0.7)
    
    ax.text(min(proba_churn + 0.02, 0.92), 0,
            f"{proba_churn:.1%}", va="center", fontweight="bold", fontsize=11, color="#F8FAFC")
    
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Expanders untuk Debugging
    with st.expander("📋 Log Data Input (Setelah Feature Engineering)"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
