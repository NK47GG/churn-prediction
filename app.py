"""
app.py — Customer Churn Prediction (Streamlit)

Kompatibel dengan artefak dari notebook churn prediction.
Versi ini dibuat lebih aman:
- path model pakai relative path yang stabil
- load artefak dibungkus try/except
- kalau joblib gagal karena ModuleNotFoundError, error ditampilkan jelas
- UI tetap rapi dan tidak crash total

Jalankan lokal:
    streamlit run app.py
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"

# ── CSS Kustom ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-label { font-size: 13px; opacity: .85; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; }
    .churn-high { background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); }
    .churn-low  { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .section-header {
        font-size: 15px;
        font-weight: 600;
        color: #4a4a6a;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 16px 0 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Load Artefak ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_artifacts(path: str):
    p = Path(path)

    if not p.exists():
        return None, f"File model tidak ditemukan: {p.name}"

    try:
        artifacts = joblib.load(p)
        return artifacts, None
    except ModuleNotFoundError as e:
        msg = (
            "Gagal load `churn_model.joblib` karena ada module/class custom yang tidak tersedia "
            "di environment ini.\n\n"
            f"Detail: {e}\n\n"
            "Solusi biasanya:\n"
            "1) pindahkan class/function custom ke file `.py` terpisah, "
            "2) import file itu sebelum `joblib.load()`, "
            "3) simpan ulang artefak model."
        )
        return None, msg
    except Exception as e:
        msg = (
            "Gagal load `churn_model.joblib`.\n\n"
            f"Detail error: {e}\n\n"
            f"Traceback singkat:\n{traceback.format_exc()}"
        )
        return None, msg


artifacts, load_error = load_artifacts(str(MODEL_PATH))


# ── Konstanta referensi ───────────────────────────────────────────────────────
REF_DATE = pd.Timestamp("2025-01-01")


# ── Feature Engineering ───────────────────────────────────────────────────────
def feature_engineering(
    signup_date_val,
    last_purchase_date_val,
    has_coupon_val,
    extra_num,
    nominal,
    binary,
    raw_columns,
):
    signup_ts = pd.Timestamp(signup_date_val)
    purchase_ts = pd.Timestamp(last_purchase_date_val)

    row = {
        "days_since_signup": (REF_DATE - signup_ts).days,
        "days_since_purchase": (REF_DATE - purchase_ts).days,
        "signup_month": int(signup_ts.month),
        "tenure_days": (purchase_ts - signup_ts).days,
        "has_coupon": int(has_coupon_val),
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
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    iqr_bounds = artifacts["iqr_bounds"]
    top_features = artifacts.get("top_features", [])
    scenario = artifacts.get("scenario", "Unknown")

    df = input_df.copy()

    # 1. Outlier clipping
    for col, bounds in iqr_bounds.items():
        if col in df.columns:
            lo, hi = bounds
            df[col] = df[col].clip(lo, hi)

    # 2. Transform
    X_arr = preprocessor.transform(df)

    try:
        feat_names = preprocessor.get_feature_names_out()
        X_enc = pd.DataFrame(X_arr, columns=feat_names)
    except Exception:
        X_enc = pd.DataFrame(X_arr)

    # 3. Seleksi top features kalau skenario tuning
    if scenario == "Tuning" and top_features:
        available = [f for f in top_features if f in X_enc.columns]
        if available:
            X_enc = X_enc[available]

    # 4. Prediksi
    pred = model.predict(X_enc)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_enc)[0][1]
    else:
        proba = float(pred)

    return int(pred), float(proba)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔮 Churn Predictor")
    st.caption("UAS Bengkel Koding — Data Science")
    st.divider()

    if artifacts:
        st.markdown("### 📦 Info Model")
        n_feat = (
            str(len(artifacts.get("top_features", [])))
            if artifacts.get("scenario") == "Tuning"
            else "Semua"
        )
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
        st.warning("⚠️ Artefak model belum berhasil dimuat.")

    st.divider()
    st.caption("© 2025 · Customer Churn Prediction")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
st.title("Customer Churn Prediction")
st.markdown(
    "Masukkan data pelanggan di bawah ini, lalu klik **Prediksi** "
    "untuk mengetahui apakah pelanggan berisiko *churn*."
)

if load_error:
    st.error("Model gagal dimuat.")
    st.code(load_error)
    st.stop()

if artifacts is None:
    st.error(
        "❌ `churn_model.joblib` tidak ditemukan atau tidak bisa dimuat. "
        "Pastikan file model ada di folder yang sama dengan `app.py`."
    )
    st.stop()


raw_columns = artifacts.get("raw_columns", [])
numeric_cont = artifacts.get("numeric_cont", [])

KNOWN_NOMINAL = {
    "gender",
    "country",
    "city",
    "acquisition_channel",
    "device_type",
    "subscription_type",
    "payment_method",
}
KNOWN_BINARY = {"is_premium_user", "discount_used", "refund_requested"}
ENGINEERED = {
    "days_since_signup",
    "days_since_purchase",
    "signup_month",
    "tenure_days",
    "has_coupon",
}

other_num = [
    c for c in raw_columns
    if c not in KNOWN_NOMINAL and c not in KNOWN_BINARY and c not in ENGINEERED
]

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
    "age": 30.0,
    "purchase_amount": 500_000.0,
    "num_purchases": 5.0,
    "support_calls": 1.0,
    "num_support_calls": 1.0,
    "satisfaction_score": 7.0,
    "browsing_time": 30.0,
    "items_in_cart": 2.0,
    "total_spend": 2_000_000.0,
    "monthly_spend": 300_000.0,
    "annual_income": 60_000_000.0,
    "points_balance": 100.0,
    "avg_order_value": 250_000.0,
    "login_frequency": 10.0,
}


# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("form_prediksi"):
    st.markdown('<p class="section-header">👤 Demografi</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female", "Other"])
    country = c2.text_input("Country", value="Indonesia")
    city = c3.text_input("City", value="Jakarta")

    st.markdown('<p class="section-header">📅 Informasi Akun</p>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    signup_date = c4.date_input(
        "Tanggal Signup",
        value=date(2022, 1, 15),
        min_value=date(2015, 1, 1),
        max_value=date(2025, 1, 1),
    )
    last_purchase_date = c5.date_input(
        "Tanggal Pembelian Terakhir",
        value=date(2024, 6, 1),
        min_value=date(2015, 1, 1),
        max_value=date(2025, 1, 1),
    )
    has_coupon_raw = c6.radio("Pernah Pakai Kupon?", ["Ya", "Tidak"], horizontal=True)

    st.markdown('<p class="section-header">⚙️ Layanan & Channel</p>', unsafe_allow_html=True)
    c7, c8, c9, c10 = st.columns(4)
    subscription_type = c7.selectbox("Tipe Langganan", ["Basic", "Standard", "Premium"])
    payment_method = c8.selectbox("Metode Pembayaran", ["Credit Card", "Bank Transfer", "E-Wallet", "Cash"])
    acquisition_channel = c9.selectbox("Saluran Akuisisi", ["Organic", "Referral", "Paid Ads", "Social Media", "Email"])
    device_type = c10.selectbox("Tipe Perangkat", ["Mobile", "Desktop", "Tablet"])

    st.markdown('<p class="section-header">📊 Status & Data Numerik</p>', unsafe_allow_html=True)
    left, right = st.columns([1, 2])

    with left:
        st.markdown("**Status Akun**")
        is_premium_user = st.checkbox("Premium User")
        discount_used = st.checkbox("Pernah Pakai Diskon")
        refund_requested = st.checkbox("Pernah Request Refund")

    with right:
        num_inputs = {}
        if other_num:
            chunks = [other_num[i:i + 3] for i in range(0, len(other_num), 3)]
            for chunk in chunks:
                row_cols = st.columns(len(chunk))
                for idx, col_name in enumerate(chunk):
                    label = LABEL_MAP.get(col_name, col_name.replace("_", " ").title())
                    default = DEFAULT_MAP.get(col_name, 0.0)
                    num_inputs[col_name] = row_cols[idx].number_input(
                        label,
                        value=float(default),
                        key=f"num_{col_name}",
                    )
        else:
            st.info("Tidak ada kolom numerik tambahan yang terdeteksi dari artefak.")

    submitted = st.form_submit_button(
        "🔮 Prediksi Churn",
        type="primary",
        use_container_width=True,
    )


# ── Prediksi ──────────────────────────────────────────────────────────────────
if submitted:
    if last_purchase_date < signup_date:
        st.error("❌ Tanggal pembelian terakhir tidak boleh lebih awal dari signup.")
        st.stop()

    nominal = {
        "gender": gender,
        "country": country,
        "city": city,
        "acquisition_channel": acquisition_channel,
        "device_type": device_type,
        "subscription_type": subscription_type,
        "payment_method": payment_method,
    }

    binary = {
        "is_premium_user": int(is_premium_user),
        "discount_used": int(discount_used),
        "refund_requested": int(refund_requested),
    }

    with st.spinner("Memproses prediksi…"):
        input_df = feature_engineering(
            signup_date_val=signup_date,
            last_purchase_date_val=last_purchase_date,
            has_coupon_val=(has_coupon_raw == "Ya"),
            extra_num=num_inputs,
            nominal=nominal,
            binary=binary,
            raw_columns=raw_columns,
        )
        pred, proba_churn = run_prediction(input_df, artifacts)

    # ── Tampilan Hasil ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Hasil Prediksi")

    is_churn = pred == 1
    card_class = "churn-high" if is_churn else "churn-low"
    verdict = "⚠️ CHURN" if is_churn else "✅ TIDAK CHURN"
    risk_label = (
        "🔴 Tinggi" if proba_churn > 0.70 else
        "🟡 Sedang" if proba_churn > 0.40 else
        "🟢 Rendah"
    )

    r1, r2, r3 = st.columns(3)
    r1.markdown(
        f'<div class="metric-card {card_class}">'
        f'<div class="metric-label">Prediksi</div>'
        f'<div class="metric-value">{verdict}</div></div>',
        unsafe_allow_html=True,
    )
    r2.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Probabilitas Churn</div>'
        f'<div class="metric-value">{proba_churn:.1%}</div></div>',
        unsafe_allow_html=True,
    )
    r3.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">Tingkat Risiko</div>'
        f'<div class="metric-value">{risk_label}</div></div>',
        unsafe_allow_html=True,
    )

    # Gauge bar
    st.markdown("**Visualisasi Risiko Churn**")
    fig, ax = plt.subplots(figsize=(8, 1.4))
    ax.barh([""], [1], color="#e9ecef", height=0.5)

    bar_color = (
        "#f5576c" if proba_churn > 0.70 else
        "#ffc107" if proba_churn > 0.40 else
        "#43e97b"
    )
    ax.barh([""], [proba_churn], color=bar_color, height=0.5)
    ax.set_xlim(0, 1)
    ax.axvline(0.4, color="#ffc107", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(0.7, color="#f5576c", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(
        min(proba_churn + 0.02, 0.92),
        0,
        f"{proba_churn:.1%}",
        va="center",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_title(
        f"Churn Probability — {artifacts.get('model_name', '-')}"
        f" ({artifacts.get('scenario', '-')})"
    )

    patches = [
        mpatches.Patch(color="#43e97b", label="Rendah (< 40%)"),
        mpatches.Patch(color="#ffc107", label="Sedang (40–70%)"),
        mpatches.Patch(color="#f5576c", label="Tinggi (> 70%)"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", left=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Rekomendasi tindakan
    st.markdown("**💡 Rekomendasi Tindakan**")
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
    with st.expander("🔍 Detail Feature Engineering"):
        signup_ts = pd.Timestamp(signup_date)
        purchase_ts = pd.Timestamp(last_purchase_date)

        st.dataframe(
            pd.DataFrame(
                {
                    "Fitur": [
                        "days_since_signup",
                        "days_since_purchase",
                        "signup_month",
                        "tenure_days",
                        "has_coupon",
                    ],
                    "Penjelasan": [
                        "Hari sejak signup (ref 2025-01-01)",
                        "Hari sejak pembelian terakhir (ref 2025-01-01)",
                        "Bulan signup",
                        "Durasi signup → pembelian terakhir (hari)",
                        "Pernah pakai kupon (1=Ya, 0=Tidak)",
                    ],
                    "Nilai": [
                        (REF_DATE - signup_ts).days,
                        (REF_DATE - purchase_ts).days,
                        int(signup_ts.month),
                        (purchase_ts - signup_ts).days,
                        int(has_coupon_raw == "Ya"),
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if artifacts.get("scenario") == "Tuning" and artifacts.get("top_features"):
        with st.expander(f"⭐ Top {len(artifacts['top_features'])} Features"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "#": range(1, len(artifacts["top_features"]) + 1),
                        "Nama Fitur": artifacts["top_features"],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("📋 Raw Input (setelah feature engineering)"):
        st.dataframe(input_df.T.rename(columns={0: "Nilai"}), use_container_width=True)
