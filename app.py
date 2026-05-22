import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon=":bar_chart:", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load("churn_model.joblib")

artifacts      = load_artifacts()
model          = artifacts["model"]
scaler         = artifacts["scaler"]
label_encoders = artifacts["label_encoders"]
scale_cols     = artifacts["scale_cols"]
top_features   = artifacts["top_features"]
model_name     = artifacts.get("model_name", "Model")
scenario       = artifacts.get("scenario", "-")

st.title("Customer Churn Prediction Dashboard")
st.markdown(f"**Model:** {model_name} | **Skenario:** {scenario}")
st.markdown("---")

st.sidebar.header("Input Data Pelanggan")

age               = st.sidebar.slider("Usia", 18, 80, 35)
total_visits      = st.sidebar.slider("Total Kunjungan", 0, 100, 20)
avg_session_time  = st.sidebar.slider("Rata-rata Waktu Sesi (menit)", 0.0, 60.0, 15.0)
pages_per_session = st.sidebar.slider("Halaman per Sesi", 0.0, 20.0, 5.0)
email_open_rate   = st.sidebar.slider("Email Open Rate", 0.0, 1.0, 0.4)
email_click_rate  = st.sidebar.slider("Email Click Rate", 0.0, 1.0, 0.2)
total_spent       = st.sidebar.number_input("Total Pengeluaran", 0.0, 10000.0, 500.0)
avg_order_value   = st.sidebar.number_input("Rata-rata Nilai Order", 0.0, 1000.0, 80.0)
support_tickets   = st.sidebar.slider("Tiket Support", 0, 20, 2)
satisfaction_score= st.sidebar.slider("Skor Kepuasan (1-5)", 1.0, 5.0, 3.5)
nps_score         = st.sidebar.slider("NPS Score", -100, 100, 30)
lifetime_value    = st.sidebar.number_input("Lifetime Value", 0.0, 10000.0, 1500.0)
last_3_month_purchase_freq = st.sidebar.slider("Frekuensi Pembelian 3 Bulan", 0, 30, 5)
marketing_spend_per_user   = st.sidebar.number_input("Marketing Spend per User", 0.0, 200.0, 25.0)
delivery_delay_days        = st.sidebar.slider("Keterlambatan Pengiriman (hari)", 0, 30, 2)

gender            = st.sidebar.selectbox("Gender", ["Male", "Female"])
country           = st.sidebar.selectbox("Negara", ["India", "Germany", "USA", "UK", "Australia"])
city              = st.sidebar.selectbox("Kota", ["Berlin","Mumbai","London","New York","Sydney","Paris","Tokyo"])
acquisition_channel = st.sidebar.selectbox("Acquisition Channel", ["Email","Organic","Ads","Referral","Social"])
device_type       = st.sidebar.selectbox("Device Type", ["Desktop","Mobile","Tablet"])
subscription_type = st.sidebar.selectbox("Subscription Type", ["Monthly","Annual"])
payment_method    = st.sidebar.selectbox("Payment Method", ["Credit Card","UPI","BKash","PayPal","Debit Card"])

is_premium_user   = int(st.sidebar.checkbox("Premium User?", False))
discount_used     = int(st.sidebar.checkbox("Gunakan Diskon?", False))
refund_requested  = int(st.sidebar.checkbox("Pernah Refund?", False))
has_coupon        = int(st.sidebar.checkbox("Punya Kupon?", False))

days_since_signup   = st.sidebar.slider("Lama Bergabung (hari)", 30, 2000, 500)
days_since_purchase = st.sidebar.slider("Hari Sejak Pembelian Terakhir", 1, 365, 30)
signup_month        = st.sidebar.slider("Bulan Daftar", 1, 12, 6)
tenure_days         = max(0, days_since_signup - days_since_purchase)

input_data = {
    "age": age, "total_visits": total_visits, "avg_session_time": avg_session_time,
    "pages_per_session": pages_per_session, "email_open_rate": email_open_rate,
    "email_click_rate": email_click_rate, "total_spent": total_spent,
    "avg_order_value": avg_order_value, "support_tickets": support_tickets,
    "satisfaction_score": satisfaction_score, "nps_score": nps_score,
    "lifetime_value": lifetime_value, "last_3_month_purchase_freq": last_3_month_purchase_freq,
    "marketing_spend_per_user": marketing_spend_per_user, "delivery_delay_days": delivery_delay_days,
    "is_premium_user": is_premium_user, "discount_used": discount_used,
    "refund_requested": refund_requested, "has_coupon": has_coupon,
    "gender": gender, "country": country, "city": city,
    "acquisition_channel": acquisition_channel, "device_type": device_type,
    "subscription_type": subscription_type, "payment_method": payment_method,
    "days_since_signup": days_since_signup, "days_since_purchase": days_since_purchase,
    "signup_month": signup_month, "tenure_days": tenure_days,
}
input_df = pd.DataFrame([input_data])

cat_cols = ["gender","country","city","acquisition_channel",
            "device_type","subscription_type","payment_method"]
for col in cat_cols:
    if col in label_encoders:
        le = label_encoders[col]
        val = input_df[col].values[0]
        try:
            input_df[col] = le.transform([val])[0]
        except ValueError:
            input_df[col] = 0

sc_present = [c for c in scale_cols if c in input_df.columns]
input_df[sc_present] = scaler.transform(input_df[sc_present])
input_final = input_df[[f for f in top_features if f in input_df.columns]]

pred  = model.predict(input_final)[0]
proba = model.predict_proba(input_final)[0] if hasattr(model, "predict_proba") else None

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Data Input Pelanggan (fitur terpilih)")
    show_cols = [f for f in top_features if f in input_df.columns]
    st.dataframe(input_df[show_cols].T.rename(columns={0: "Nilai"}), height=400)
with col2:
    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.error("PELANGGAN BERPOTENSI CHURN!")
        st.write("Pelanggan ini diprediksi akan **berhenti** menggunakan layanan.")
    else:
        st.success("PELANGGAN DIPREDIKSI TIDAK CHURN")
        st.write("Pelanggan ini diprediksi akan **tetap** menggunakan layanan.")
    if proba is not None:
        st.markdown("---")
        st.metric("Prob. Tidak Churn", f"{proba[0]*100:.1f}%")
        st.metric("Prob. Churn",       f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]), text="Risiko Churn")

st.markdown("---")
st.markdown("**UAS Bengkel Koding Data Science** - Universitas Dian Nuswantoro")
