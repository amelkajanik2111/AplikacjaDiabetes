import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfiguracja strony
st.set_page_config(page_title="Predykcja Ryzyka Cukrzycy", layout="centered")

@st.cache_resource
def load_assets():
    # Ładowanie modelu, skalera i listy cech
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except Exception as e:
    st.error("Błąd ładowania plików modelu. Upewnij się, że .pkl są w tym samym folderze.")
    st.stop()

st.title("🩺 Asystent Diagnostyki Cukrzycy")
st.write("Wprowadź dane pacjenta, aby ocenić ryzyko wystąpienia cukrzycy.")

# Tworzenie formularza z polami na podstawie Twojego datasetu
with st.form("diabetes_form"):
    st.subheader("Dane zdrowotne i styl życia")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        age = st.slider("Wiek (grupa 1-13)", 1, 13, 5)
        high_bp = st.selectbox("Wysokie ciśnienie?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        high_chol = st.selectbox("Wysoki cholesterol?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        
    with col2:
        gen_hlth = st.slider("Ogólne zdrowie (1-5)", 1, 5, 3)
        phys_act = st.selectbox("Aktywność fizyczna?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        smoker = st.selectbox("Palacz?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        sex = st.selectbox("Płeć", [0, 1], format_func=lambda x: "Mężczyzna" if x==1 else "Kobieta")

    # Dodaj resztę brakujących cech z domyślnymi wartościami (lub stwórz dla nich pola)
    # W Twoim projekcie jest 21 cech wejściowych
    submit = st.form_submit_button("Analizuj ryzyko")

if submit:
    # Przygotowanie danych do predykcji
    # Uwaga: Musisz przekazać WSZYSTKIE cechy w kolejności z features_list.pkl
    input_data = pd.DataFrame([[high_bp, high_chol, 1, bmi, smoker, 0, 0, phys_act, 1, 1, 0, 1, 0, gen_hlth, 0, 0, 0, sex, age, 4, 5]], 
                              columns=features)
    
    # Skalowanie i predykcja
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ Wysokie ryzyko cukrzycy! (Prawdopodobieństwo: {probability:.2%})")
    else:
        st.success(f"✅ Niskie ryzyko cukrzycy. (Prawdopodobieństwo: {probability:.2%})")
