"""
app.py — Customer Churn Prediction (Streamlit)
Kompatibel dengan artefak dari UAS_Churn_Prediction_v2.ipynb

Pipeline prediksi:
  1. Input form → feature engineering (deterministik, per-baris)
  2. Outlier clipping dengan iqr_bounds dari data latih
  3. preprocessor.transform() (ColumnTransformer: impute + OHE + scale)
  4. Jika skenario Tuning → pilih top_features
  5. model.predict() / model.predict_proba()

Jalankan lokal:
  streamlit run app.py
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
    page_title="Customer Churn Predictor",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Kustom (Tema Terang & Matcha Green) ───────────────────────────────────
st.markdown("""
<style>
    /* Latar belakang utama aplikasi */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Kartu Metrik */
    .metric-card {
        background: #FFFFFF;
        padding: 24px; 
        border-radius: 16px; 
        color: #1E293B; 
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label { font-size: 14px; font-weight: 500; color: #64748B; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 32px; font-weight: 800; color: #0F172A; }
    
    /* Warna Status Churn */
    .churn-high { background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%); border-color: #FCA5A5; }
    .churn-high .metric-value { color: #DC2626; }
    
    .churn-low  { background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border-color: #86EFAC; }
    .churn-low .metric-value { color: #16A34A; }
    
    /* Header Bagian */
    .section-header {
        font-size: 18px; 
        font-weight: 700; 
        color: #334155;
        border-bottom: 2px solid #BBF7D0; 
        padding-bottom: 8px; 
        margin: 24px 0 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Tombol Submit Kustom (Nuansa Matcha) */
    div.stButton > button:first-child {
        background-color: #8BA888;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #718C6E;
        box-shadow: 0 4px 12px rgba(139, 168, 136, 0.3);
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

# ── Konstanta referensi ───────────────────────────────────────────────────────
REF_DATE = pd.Timestamp("2025-01-01")


# ── Feature Engineering ───────────────────────────────────────────────────────
def feature_engineering(signup_date_val, last_purchase_date_val,
                        has_coupon_val, extra_num, nominal, binary,
                        raw_columns):
    signup_ts   = pd.Timestamp(signup_date_val)
    purchase_ts = pd.Timestamp(last_purchase_date_val)
    row = {
        "days_since_signup"  : (REF_DATE - signup_ts).days,
        "days_since_purchase": (REF_DATE - purchase_ts).days,
        "signup_month"       : int(signup_ts.month),
        "tenure_days"        : (purchase_ts - signup_ts).days,
        "has_coupon"         : int(has_coupon_val),
    }
    row.update(nominal)
    row.update(binary)
    row.update(extra_num)
    df = pd.DataFrame([row])
    for col in raw_columns:
        if col not in df.columns:
            df[col] = 0
    return df[raw_columns]


# ── Full Prediction Pipeline ──────────────────────────────────────────────────
def run_prediction(input_df, artifacts):
    model        = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    iqr_bounds   = artifacts["iqr_bounds"]
    top_features = artifacts["top_features"]
    scenario     = artifacts["scenario"]

    df = input_df.copy()
    # 1. Outlier clipping
    for col, (lo, hi) in iqr_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # 2. Preprocessor
    X_arr      = preprocessor.transform(df)
    feat_names = preprocessor.get_feature_names_out()
    X_enc      = pd.DataFrame(X_arr, columns=feat_names)

    # 3. Top features
    if scenario == "Tuning":
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
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=60)
    st.title("🔮 Churn Predictor")
    st.caption("UAS Bengkel Koding — Data Science")
    st.divider()
    if artifacts:
        st.markdown("### 📦 Info Model")
        n_feat = (str(len(artifacts.get("top_features", [])))
                  if artifacts.get("scenario") == "Tuning" else "Semua")
        st.info(
            f"**Model :** {artifacts.get('model_name', '-')}\n\n"
            f"**Skenario :** {artifacts.get('scenario', '-')}\n\n"
            f"**Fitur dipakai :** {n_feat}"
        )
        st.markdown("### 🔄 Alur Prediksi")
        st.markdown(
            "1. Input form\n"
            "2. Feature engineering\n"
            "3. Outlier clipping (IQR)\n"
            "4. Impute + OHE + Scale\n"
            "5. *(Tuning)* Pilih top features\n"
            "6. Model → Prediksi"
        )
    else:
        st.warning("⚠️ `churn_model.joblib` tidak ditemukan.")
    st.divider()
    st.caption("© 2025 · Customer Churn Prediction")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌱 Customer Churn Prediction")
st.markdown(
    "Masukkan data pelanggan di bawah ini, lalu klik **Prediksi Churn** "
    "untuk mengetahui persentase risiko pelanggan meninggalkan layanan."
)

if artifacts is None:
    st.error(
        "❌ `churn_model.joblib` tidak ditemukan. "
        "Jalankan notebook terlebih dahulu untuk menghasilkan artefak model."
    )
    st.stop()

raw_columns  = artifacts.get("raw_columns", [])
numeric_cont = artifacts.get("numeric_cont", [])

KNOWN_NOMINAL = {"gender", "country", "city", "acquisition_channel",
                 "device_type", "subscription_type", "payment_method"}
KNOWN_BINARY  = {"is_premium_user", "discount_used", "refund_requested"}
ENGINEERED    = {"days_since_signup", "days_since_purchase",
                 "signup_month", "tenure_days", "has_coupon"}

other_num = [c for c in raw_columns
             if c not in KNOWN_NOMINAL
             and c not in KNOWN_BINARY
             and c not in ENGINEERED]

LABEL_MAP = {
    "age": "Usia (tahun)",
    "purchase_amount": "Nilai Pembelian",
    "num_purchases": "Jumlah Pembelian",
    "support_calls": "Support Calls",
    "num_support_calls": "Support Calls",
    "satisfaction_score": "Skor Kepuasan (1-10)",
    "browsing_time": "Waktu Browsing (mnt)",
    "items_in_cart": "Item dalam Keranjang",
    "total_spend": "Total Pengeluaran",
    "monthly_spend": "Pengeluaran Bulanan",
    "annual_income": "Pendapatan Tahunan",
    "points_balance": "Saldo Poin",
    "avg_order_value": "Rata-rata Nilai Order",
    "login_frequency": "Frekuensi Login",
}
DEFAULT_MAP = {
    "age": 30.0, "purchase_amount": 500_000.0, "num_purchases": 5.0,
    "support_calls": 1.0, "num_support_calls": 1.0, "satisfaction_score": 7.0,
    "browsing_time": 30.0, "items_in_cart": 2.0, "total_spend": 2_000_000.0,
    "monthly_spend": 300_000.0, "annual_income": 60_000_000.0,
    "points_balance": 100.0, "avg_order_value": 250_000.0, "login_frequency": 10.0,
}


# ── Form Input Terstruktur ────────────────────────────────────────────────────
with st.form("form_prediksi"):
    
    st.markdown('<div class="section-header">👤 Profil & Demografi</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    gender  = c1.selectbox("Gender",  ["Male", "Female", "Other"])
    country = c2.text_input("Country", value="Indonesia")
    city    = c3.text_input("City",    value="Jakarta")

    st.markdown('<div class="section-header">📅 Aktivitas & Informasi Akun</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    signup_date = c4.date_input(
        "Tanggal Signup",
        value=date(2022, 1, 15),
        min_value=date(2015, 1, 1), max_value=date(2025, 1, 1))
    last_purchase_date = c5.date_input(
        "Tanggal Pembelian Terakhir",
        value=date(2024, 6, 1),
        min_value=date(2015, 1, 1), max_value=date(2025, 1, 1))
    has_coupon_raw = c6.radio("Pernah Pakai Kupon?", ["Ya", "Tidak"], horizontal=True)

    st.markdown('<div class="section-header">⚙️ Preferensi Layanan & Channel</div>', unsafe_allow_html=True)
    c7, c8, c9, c10 = st.columns(4)
    subscription_type   = c7.selectbox("Tipe Langganan", ["Basic", "Standard", "Premium"])
    payment_method      = c8.selectbox("Metode Pembayaran", ["Credit Card", "Bank Transfer", "E-Wallet", "Cash"])
    acquisition_channel = c9.selectbox("Saluran Akuisisi", ["Organic", "Referral", "Paid Ads", "Social Media", "Email"])
    device_type         = c10.selectbox("Tipe Perangkat", ["Mobile", "Desktop", "Tablet"])

    st.markdown('<div class="section-header">📊 Metrik Tambahan & Status</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 3])

    with left:
        st.markdown("**Status Khusus**")
        st.write("") # Spacing
        is_premium_user  = st.toggle("Premium User", value=False)
        discount_used    = st.toggle("Pernah Pakai Diskon", value=False)
        refund_requested = st.toggle("Pernah Request Refund", value=False)

    with right:
        num_inputs = {}
        if other_num:
            chunks = [other_num[i:i+3] for i in range(0, len(other_num), 3)]
            for chunk in chunks:
                row_cols = st.columns(len(chunk))
                for idx, col_name in enumerate(chunk):
                    label   = LABEL_MAP.get(col_name, col_name.replace("_", " ").title())
                    default = DEFAULT_MAP.get(col_name, 0.0)
                    num_inputs[col_name] = row_cols[idx].number_input(
                        label, value=float(default), key=f"num_{col_name}")
        else:
            st.info("Tidak ada kolom numerik tambahan.")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "✨ Analisis Risiko Churn", type="primary", use_container_width=True)


# ── Proses Prediksi & Hasil ───────────────────────────────────────────────────
if submitted:
    if last_purchase_date < signup_date:
        st.error("❌ Tanggal pembelian terakhir tidak boleh lebih awal dari signup.")
        st.stop()

    nominal = {
        "gender": gender, "country": country, "city": city,
        "acquisition_channel": acquisition_channel,
        "device_type": device_type,
        "subscription_type": subscription_type,
        "payment_method": payment_method,
    }
    binary = {
        "is_premium_user" : int(is_premium_user),
        "discount_used"   : int(discount_used),
        "refund_requested": int(refund_requested),
    }

    with st.spinner("Memproses prediksi AI…"):
        input_df = feature_engineering(
            signup_date_val        = signup_date,
            last_purchase_date_val = last_purchase_date,
            has_coupon_val         = (has_coupon_raw == "Ya"),
            extra_num              = num_inputs,
            nominal                = nominal,
            binary                 = binary,
            raw_columns            = raw_columns,
        )
        pred, proba_churn = run_prediction(input_df, artifacts)

    # ── Tampilan Hasil ────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">📈 Hasil Analisis AI</div>', unsafe_allow_html=True)

    is_churn   = pred == 1
    card_class = "churn-high" if is_churn else "churn-low"
    verdict    = "BERISIKO CHURN" if is_churn else "TIDAK CHURN"
    risk_label = ("🔴 Tinggi"  if proba_churn > 0.70 else
                  "🟡 Sedang"  if proba_churn > 0.40 else "🟢 Rendah")

    r1, r2, r3 = st.columns(3)
    r1.markdown(
        f'<div class="metric-card {card_class}">'
        f'<div class="metric-label">Prediksi Sistem</div>'
        f'<div class="metric-value">{verdict}</div></div>',
        unsafe_allow_html=True)
    r2.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Probabilitas Churn</div>'
        f'<div class="metric-value">{proba_churn:.1%}</div></div>',
        unsafe_allow_html=True)
    r3.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Tingkat Risiko</div>'
        f'<div class="metric-value">{risk_label}</div></div>',
        unsafe_allow_html=True)

    # Visualisasi Gauge bar (Tema Terang)
    st.markdown("**Visualisasi Risiko Churn**")
    fig, ax = plt.subplots(figsize=(8, 1.4), facecolor="#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    
    ax.barh([""], [1], color="#E2E8F0", height=0.5, edgecolor="none")
    bar_color = ("#EF4444" if proba_churn > 0.70 else
                 "#F59E0B" if proba_churn > 0.40 else "#8BA888")
    ax.barh([""], [proba_churn], color=bar_color, height=0.5, edgecolor="none")
    
    ax.set_xlim(0, 1)
    ax.axvline(0.4, color="#F59E0B", linewidth=2, linestyle=":", alpha=0.8)
    ax.axvline(0.7, color="#EF4444", linewidth=2, linestyle=":", alpha=0.8)
    
    ax.text(min(proba_churn + 0.02, 0.92), 0,
            f"{proba_churn:.1%}", va="center", fontweight="bold", fontsize=11, color="#334155")
    ax.set_title(
        f"Distribusi Probabilitas — {artifacts.get('model_name')}", color="#475569", fontsize=10, pad=10)
    
    patches = [
        mpatches.Patch(color="#8BA888", label="Rendah (< 40%)"),
        mpatches.Patch(color="#F59E0B", label="Sedang (40–70%)"),
        mpatches.Patch(color="#EF4444", label="Tinggi (> 70%)"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.9, facecolor="white", edgecolor="#E2E8F0")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", left=False)
    ax.tick_params(axis="x", colors="#64748B")
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Rekomendasi tindakan
    st.markdown("**💡 Rekomendasi Tindakan Strategis**")
    if proba_churn > 0.70:
        st.error(
            "🚨 **Risiko Tinggi** — Pelanggan ini sangat berisiko churn.\n\n"
            "- Hubungi pelanggan secara personal (email/telepon).\n"
            "- Tawarkan diskon eksklusif atau perpanjangan layanan gratis.\n"
            "- Tindak-lanjuti keluhan / request refund segera."
        )
    elif proba_churn > 0.40:
        st.warning(
            "⚠️ **Risiko Sedang** — Pantau perilaku pelanggan secara berkala.\n\n"
            "- Kirimkan kupon atau penawaran loyalitas.\n"
            "- Dorong upgrade ke paket Premium.\n"
            "- Evaluasi kepuasan melalui survei singkat."
        )
    else:
        st.success(
            "✅ **Risiko Rendah** — Pelanggan cenderung bertahan.\n\n"
            "- Pertahankan pengalaman positif yang sudah ada.\n"
            "- Dorong referral program untuk akuisisi pelanggan baru."
        )

    # Detail expanders
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 Detail Feature Engineering"):
        signup_ts   = pd.Timestamp(signup_date)
        purchase_ts = pd.Timestamp(last_purchase_date)
        st.dataframe(pd.DataFrame({
            "Fitur"      : ["days_since_signup", "days_since_purchase",
                            "signup_month", "tenure_days", "has_coupon"],
            "Penjelasan" : ["Hari sejak signup (ref 2025-01-01)",
                            "Hari sejak pembelian terakhir (ref 2025-01-01)",
                            "Bulan signup",
                            "Durasi signup → pembelian terakhir (hari)",
                            "Pernah pakai kupon (1=Ya, 0=Tidak)"],
            "Nilai"      : [(REF_DATE - signup_ts).days,
                            (REF_DATE - purchase_ts).days,
                            int(signup_ts.month),
                            (purchase_ts - signup_ts).days,
                            int(has_coupon_raw == "Ya")],
        }), use_container_width=True, hide_index=True)

    if artifacts.get("scenario") == "Tuning":
        with st.expander(f"⭐ Top {len(artifacts['top_features'])} Features yang Digunakan"):
            st.dataframe(pd.DataFrame({
                "#"          : range(1, len(artifacts["top_features"]) + 1),
                "Nama Fitur" : artifacts["top_features"],
            }), use_container_width=True, hide_index=True)

    with st.expander("📋 Data Input Mentah (Setelah Pemrosesan)"):
        st.dataframe(input_df.T.rename(columns={0: "Nilai"}), use_container_width=True)
