import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("model_mobil_knn.pkl")
scaler = joblib.load("scaler_knn.pkl")

st.title("🚗 Prediksi Harga Mobil Bekas Toyota")

# Mapping hasil encoding dari data asli
model_options = {
    'Auris': 0,
    'Avensis': 1,
    'Aygo': 2,
    'C-HR': 3,
    'Camry': 4,
    'Corolla': 5,
    'GT86': 6,
    'Hilux': 7,
    'IQ': 8,
    'Land Cruiser': 9,
    'Prius': 10,
    'Proace': 11,
    'RAV4': 12,
    'SUPRA': 13,
    'Urban Cruiser': 14,
    'Verso': 15,
    'Yaris': 16
}

transmission_options = {
    'Automatic': 0,
    'Manual': 1,
    'Other': 2,
    'Semi-Auto': 3
}

fuel_options = {
    'Diesel': 0,
    'Electric': 1,
    'Hybrid': 2,
    'Other': 3,
    'Petrol': 4
}

# Form input
with st.form("form_mobil"):
    model_input = st.selectbox("Model Mobil", list(model_options.keys()))
    year = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, step=1)
    transmission_input = st.selectbox("Transmisi", list(transmission_options.keys()))
    mileage = st.number_input("Jarak Tempuh (Mileage dalam mil)", min_value=0)
    fuel_input = st.selectbox("Tipe Bahan Bakar", list(fuel_options.keys()))
    tax = st.number_input("Pajak (£)", min_value=0)
    mpg = st.number_input("MPG (Miles per Gallon)", min_value=0.0)
    engine_size = st.number_input("Ukuran Mesin (Liter)", min_value=0.0)
    
    submit = st.form_submit_button("🔍 Prediksi Harga")

# Prediksi
if submit:
    # Ubah ke angka
    model_val = model_options[model_input]
    transmission_val = transmission_options[transmission_input]
    fuel_val = fuel_options[fuel_input]

    # Gabungkan input jadi array
    features = np.array([[model_val, year, transmission_val, mileage, fuel_val, tax, mpg, engine_size]])

    # Normalisasi input
    features_scaled = scaler.transform(features)

    # Prediksi harga
    harga_prediksi = model.predict(features_scaled)[0]

    st.success(f"💷 Prediksi harga mobil bekas: **£{harga_prediksi:,.2f}**")
