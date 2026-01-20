import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Konfiguracja strony
st.set_page_config(page_title="Diagnostyka Cukrzycy AI", layout="centered")

# Ładowanie plików
@st.cache_resource
def load_assets():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    return model, scaler, features

model, scaler, features_names = load_assets()

st.title("🩺 Inteligentny Asystent Ryzyka Cukrzycy")
st.write("Wprowadź dane pacjenta, aby otrzymać predykcję opartą na modelu XGBoost.")

# Formularz z podziałem na kolumny
col1, col2 = st.columns(2)

with col1:
    high_bp = st.selectbox("Wysokie ciśnienie krwi?", ["Nie", "Tak"])
    high_chol = st.selectbox("Wysoki cholesterol?", ["Nie", "Tak"])
    bmi = st.number_input("BMI (wskaźnik masy ciała)", min_value=10.0, max_value=100.0, value=25.0)
    age = st.slider("Wiek (kategoria 1-13)", 1, 13, 8)

with col2:
    gen_hlth = st.slider("Ogólny stan zdrowia (1-świetny, 5-zły)", 1, 5, 3)
    phys_hlth = st.number_input("Dni złego stanu fizycznego (ostatnie 30 dni)", 0, 30, 0)
    ment_hlth = st.number_input("Dni złego stanu psychicznego (ostatnie 30 dni)", 0, 30, 0)
    income = st.slider("Poziom dochodów (skala 1-8)", 1, 8, 5)

# Przygotowanie danych do predykcji
input_dict = {name: 0 for name in features_names} # Reset wszystkich cech
# Mapowanie wartości z formularza (uproszczone dla przykładu)
input_dict['HighBP'] = 1 if high_bp == "Tak" else 0
input_dict['HighChol'] = 1 if high_chol == "Tak" else 0
input_dict['BMI'] = bmi
input_dict['Age'] = age
input_dict['GenHlth'] = gen_hlth
input_dict['PhysHlth'] = phys_hlth
input_dict['MentHlth'] = ment_hlth
input_dict['Income'] = income

if st.button("Analizuj Ryzyko"):
    df_input = pd.DataFrame([input_dict])
    
    # Skalowanie cech ciągłych (tych samych co w Twoim projekcie)
    cont_feats = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Income']
    df_input[cont_feats] = scaler.transform(df_input[cont_feats])
    
    # Predykcja
    prob = model.predict_proba(df_input)[0][1]
    prediction = model.predict(df_input)[0]
    
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ WYSOKIE RYZYKO: Prawdopodobieństwo wynosi {prob:.2%}")
        st.write("Model sugeruje konsultację lekarską. Pamiętaj, że czułość modelu wynosi ok. 79%.")
    else:
        st.success(f"✅ NISKIE RYZYKO: Prawdopodobieństwo wynosi {prob:.2%}")