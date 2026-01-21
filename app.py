import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Konfiguracja strony
st.set_page_config(page_title="Diagnostyka Cukrzycy AI", layout="centered", page_icon="🩺")

# Funkcja ładowania modeli i list (z cache, aby nie wczytywać ich przy każdym kliknięciu)
@st.cache_resource
def load_assets():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features_names = joblib.load('features_list.pkl')
    return model, scaler, features_names

# Spróbuj załadować pliki
try:
    model, scaler, features_names = load_assets()
except Exception as e:
    st.error(f"Błąd ładowania plików modelu: {e}")
    st.info("Upewnij się, że pliki .pkl znajdują się w tym samym folderze na GitHubie co app.py.")
    st.stop()

st.title("🩺 Inteligentny Asystent Ryzyka Cukrzycy")
st.write("Aplikacja analizuje czynniki ryzyka na podstawie modelu XGBoost wytrenowanego na 250 000 rekordach.")

st.markdown("---")

# Formularz użytkownika
st.subheader("Wprowadź dane pacjenta")
col1, col2 = st.columns(2)

with col1:
    high_bp = st.selectbox("Wysokie ciśnienie krwi?", ["Nie", "Tak"])
    high_chol = st.selectbox("Wysoki cholesterol?", ["Nie", "Tak"])
    bmi = st.number_input("BMI (wskaźnik masy ciała)", min_value=10.0, max_value=80.0, value=25.0)
    age = st.slider("Wiek (1=18-24, ..., 13=80+)", 1, 13, 8)
    heart_disease = st.selectbox("Choroba wieńcowa/Zawał?", ["Nie", "Tak"])

with col2:
    gen_hlth = st.slider("Ogólny stan zdrowia (1-świetny, 5-zły)", 1, 5, 3)
    phys_hlth = st.number_input("Dni złego stanu fizycznego (ostatni miesiąc)", 0, 30, 0)
    ment_hlth = st.number_input("Dni złego stanu psychicznego (ostatni miesiąc)", 0, 30, 0)
    income = st.slider("Poziom dochodów (skala 1-8)", 1, 8, 5)
    phys_activity = st.selectbox("Aktywność fizyczna w ost. 30 dniach?", ["Tak", "Nie"])

# Sekcja obliczeń
if st.button("Analizuj Ryzyko", use_container_width=True):
    # 1. Tworzymy bazowy DataFrame z zerami dla WSZYSTKICH 21 cech
    df_input = pd.DataFrame(0.0, index=[0], columns=features_names)
    
    # 2. Wypełniamy tylko te kolumny, które mamy w formularzu
    # Upewnij się, że nazwy w nawiasach ['...'] są identyczne jak w Twoim pliku CSV!
    df_input['HighBP'] = 1.0 if high_bp == "Tak" else 0.0
    df_input['HighChol'] = 1.0 if high_chol == "Tak" else 0.0
    df_input['BMI'] = float(bmi)
    df_input['Age'] = float(age)
    df_input['GenHlth'] = float(gen_hlth)
    df_input['PhysHlth'] = float(phys_hlth)
    df_input['MentHlth'] = float(ment_hlth)
    df_input['Income'] = float(income)
    
    # Dodajmy te, które wymienił błąd, aby były zainicjalizowane:
    if 'HeartDiseaseorAttack' in features_names:
        df_input['HeartDiseaseorAttack'] = 1.0 if heart_disease == "Tak" else 0.0
    if 'PhysActivity' in features_names:
        df_input['PhysActivity'] = 1.0 if phys_activity == "Tak" else 0.0

    try:
        # KLUCZOWE: Skaler w Twoim projekcie był trenowany na nazwach cech.
        # Musimy upewnić się, że kolejność kolumn w df_input jest IDENTYCZNA jak w features_names.
        df_input = df_input[features_names]
        
        # 3. Skalowanie
        # Transformacja zwraca tablicę numpy, więc musimy ją zamienić z powrotem na DataFrame z nazwami
        scaled_data = scaler.transform(df_input)
        df_final = pd.DataFrame(scaled_data, columns=features_names)
        
        # 4. Predykcja
        prob = model.predict_proba(df_final)[0][1]
        prediction = model.predict(df_final)[0]
        
        # 5. Wyświetlanie wyników
        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **WYSOKIE RYZYKO CUKRZYCY**")
            st.metric("Prawdopodobieństwo", f"{prob:.2%}")
        else:
            st.success(f"✅ **NISKIE RYZYKO CUKRZYCY**")
            st.metric("Prawdopodobieństwo", f"{prob:.2%}")
            
        st.info("Wynik na podstawie modelu XGBoost (Recall: 79%).")

    except Exception as e:
        st.error(f"Błąd dopasowania danych: {e}")
        st.write("Wymagane kolumny przez model:", features_names)
