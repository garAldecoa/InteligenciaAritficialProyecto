import streamlit as st
import pandas as pd
import joblib

# Cargar modelo
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="Predicción de Precio de Casas", layout="centered")

st.title("🏠 Predicción de Precio de Casas")
st.write("Ingresa las características de la casa y la IA estimará su precio.")

# Entradas del usuario
longitude = st.number_input("Longitud", value=-122.23, format="%.2f")
latitude = st.number_input("Latitud", value=37.88, format="%.2f")
housing_median_age = st.number_input("Edad media de las casas", min_value=1.0, value=20.0)
total_rooms = st.number_input("Total de cuartos", min_value=1.0, value=2000.0)
total_bedrooms = st.number_input("Total de habitaciones", min_value=1.0, value=400.0)
population = st.number_input("Población", min_value=1.0, value=1000.0)
households = st.number_input("Hogares", min_value=1.0, value=300.0)
median_income = st.number_input("Ingreso medio", min_value=0.1, value=5.0)
ocean_proximity = st.selectbox(
    "Proximidad al océano",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

if st.button("Predecir precio"):
    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Precio estimado de la casa: ${prediction:,.2f}")