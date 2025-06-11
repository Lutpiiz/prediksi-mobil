import streamlit as st
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load("model_mobil.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("Prediksi Harga Mobil Bekas Toyota")

# Ambil opsi dari encoder
model_options = encoders['model'].classes_
transmission_options = encoders['transmission'].classes_
fuel_options = encoders['fuelType'].classes_

with st.form("form_mobil"):
    model_input = st.selectbox("Model Mobil", model_options)
    year = st.number_input("Tahun", min_value=1990, max_value=2025, step=1)
    transmission_input = st.selectbox("Transmisi", transmission_options)
    mileage = st.number_input("Jarak Tempuh (dalam km)", min_value=0)
    fuel_input = st.selectbox("Tipe Bahan Bakar", fuel_options)
    tax = st.number_input("Pajak (£)", min_value=0)
    mpg = st.number_input("Konsumsi BBM (mpg)", min_value=0.0)
    engine_size = st.number_input("Ukuran Mesin (L)", min_value=0.0)
    
    submit = st.form_submit_button("Prediksi")

if submit:
    # Encode data
    model_encoded = encoders['model'].transform([model_input])[0]
    transmission_encoded = encoders['transmission'].transform([transmission_input])[0]
    fuel_encoded = encoders['fuelType'].transform([fuel_input])[0]

    # Gabung semua fitur
    features = np.array([[model_encoded, year, transmission_encoded, mileage, fuel_encoded, tax, mpg, engine_size]])
    harga = model.predict(features)[0]

    st.success(f"Perkiraan Harga Mobil Bekas: £{harga:,.2f}")
