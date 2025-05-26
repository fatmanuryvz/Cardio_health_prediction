import streamlit as st
import pandas as pd
import sys
import os

# src klasörünü yola ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from fatmanurprojects.DataScience import DataScience

# Başlık
st.title("🩺 Kalp Hastalığı Tahmin Uygulaması")

# Form
with st.form("prediction_form"):
    age = st.slider("Yaş", 30, 90, 50)
    height = st.number_input("Boy (cm)", 140, 210, 170)
    weight = st.number_input("Kilo (kg)", 40, 150, 70)
    ap_hi = st.number_input("Büyük Tansiyon (ap_hi)", 90, 200, 120)
    ap_lo = st.number_input("Küçük Tansiyon (ap_lo)", 60, 130, 80)

    cholesterol_val = st.number_input("Kolesterol (mg/dL)", 100, 400, 220)
    gluc_val = st.number_input("Glikoz (mg/dL)", 70, 300, 100)

    smoke = st.selectbox("Sigara içiyor musun?", ["Hayır", "Evet"])
    alco = st.selectbox("Alkol kullanıyor musun?", ["Hayır", "Evet"])
    active = st.selectbox("Fiziksel olarak aktif misin?", ["Hayır", "Evet"])

    submitted = st.form_submit_button("🧠 Tahmin Et")

def classify_cholesterol(val):
    return 1 if val < 200 else 2 if val <= 239 else 3

def classify_glucose(val):
    return 1 if val < 100 else 2 if val <= 125 else 3

if submitted:
    df = pd.DataFrame([{
        "age_years": age,
        "bmi": weight / ((height / 100) ** 2),
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": classify_cholesterol(cholesterol_val),
        "gluc": classify_glucose(gluc_val),
        "smoke": 1 if smoke == "Evet" else 0,
        "alco": 1 if alco == "Evet" else 0,
        "active": 1 if active == "Evet" else 0
    }])

    model = DataScience.load_model()
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.error("❌ Kalp hastalığı riski VAR!")
    else:
        st.success("✅ Kalp hastalığı riski YOK.")
