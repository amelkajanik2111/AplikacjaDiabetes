import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURACJA STRONY ---
st.set_page_config(
    page_title="Predykcja Ryzyka Cukrzycy",
    page_icon="🩺",
    layout="wide"
)

# --- ŁADOWANIE MODELU I SKALERA ---
@st.cache_resource
def load_assets():
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features_list.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_assets()
except Exception as e:
    st.error(f"Błąd ładowania plików: {e}")
    st.stop()

# --- INTERFEJS UŻYTKOWNIKA ---
st.title("🩺 System Przewidywania Ryzyka Cukrzycy")
st.markdown("""
Aplikacja analizuje czynniki zdrowotne na podstawie danych z badania BRFSS i ocenia prawdopodobieństwo wystąpienia cukrzycy.
""")

st.divider()

# Tworzymy formularz, aby uniknąć przeładowania strony przy każdej zmianie pola
with st.form("diabetes_form"):
    st.subheader("Wprowadź dane pacjenta")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bmi = st.number_input("BMI (Wskaźnik masy ciała)", min_value=10.0, max_value=60.0, value=25.0)
        age = st.slider("Wiek (Grupa 1-13)", 1, 13, 5, help="1=18-24, 13=80+")
        sex = st.selectbox("Płeć", options=[0, 1], format_func=lambda x: "Mężczyzna" if x==1 else "Kobieta")
        gen_hlth = st.slider("Ogólny stan zdrowia", 1, 5, 3, help="1=Doskonały, 5=Bardzo słaby")
        phys_hlth = st.slider("Dni słabego zdrowia fizycznego (0-30)", 0, 30, 0)
        ment_hlth = st.slider("Dni słabego zdrowia psychicznego (0-30)", 0, 30, 0)
        diff_walk = st.selectbox("Problemy z chodzeniem / wchodzeniem po schodach?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")

    with col2:
        high_bp = st.selectbox("Wysokie ciśnienie krwi?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        high_chol = st.selectbox("Wysoki cholesterol?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        chol_check = st.selectbox("Badanie cholesterolu w ciągu ostatnich 5 lat?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        heart_disease = st.selectbox("Choroba wieńcowa lub zawał?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        stroke = st.selectbox("Czy kiedykolwiek wystąpił udar?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        any_healthcare = st.selectbox("Posiada ubezpieczenie zdrowotne?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        no_doc_cost = st.selectbox("Brak wizyt u lekarza z powodu kosztów?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")

    with col3:
        phys_activity = st.selectbox("Aktywność fizyczna (ostatnie 30 dni)?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        smoker = st.selectbox("Wypalono co najmniej 100 papierosów w życiu?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        fruits = st.selectbox("Spożycie owoców przynajmniej raz dziennie?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        veggies = st.selectbox("Spożycie warzyw przynajmniej raz dziennie?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        hvy_alcohol = st.selectbox("Nadużywanie alkoholu (mężczyźni >14/tydz, kobiety >7/tydz)?", [0, 1], format_func=lambda x: "Tak" if x==1 else "Nie")
        education = st.slider("Poziom edukacji (1-6)", 1, 6, 4)
        income = st.slider("Poziom dochodów (1-8)", 1, 8, 5)

    submit = st.form_submit_button("ANALIZUJ RYZYKO")

# --- PROCES PREDYKCJI ---
if submit:
    # 1. Tworzymy słownik ze wszystkimi 21 cechami - NAZWY MUSZĄ BYĆ IDENTYCZNE JAK W COLABIE
    input_dict = {
        'HighBP': float(high_bp),
        'HighChol': float(high_chol),
        'CholCheck': float(chol_check),
        'BMI': float(bmi),
        'Smoker': float(smoker),
        'Stroke': float(stroke),
        'HeartDiseaseorAttack': float(heart_disease),
        'PhysActivity': float(phys_activity),
        'Fruits': float(fruits),
        'Veggies': float(veggies),
        'HvyAlcoholConsump': float(hvy_alcohol),
        'AnyHealthcare': float(any_healthcare),
        'NoDocbcCost': float(no_doc_cost),
        'GenHlth': float(gen_hlth),
        'MentHlth': float(ment_hlth),
        'PhysHlth': float(phys_hlth),
        'DiffWalk': float(diff_walk),
        'Sex': float(sex),
        'Age': float(age),
        'Education': float(education),
        'Income': float(income)
    }

    # 2. Konwersja na DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # 3. KLUCZOWY MOMENT: Dopasowanie kolejności kolumn do tej z treningu
    input_df = input_df[features]

    # 4. Skalowanie i predykcja
    try:
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]

        # --- WYŚWIETLENIE WYNIKU ---
        st.subheader("Wynik analizy:")
        
        if prediction == 1:
            st.error(f"### Wysokie ryzyko cukrzycy (Prawdopodobieństwo: {probability:.2%})")
            st.warning("Zalecana konsultacja lekarska i wykonanie badań kontrolnych.")
        else:
            st.success(f"### Niskie ryzyko cukrzycy (Prawdopodobieństwo: {probability:.2%})")
            st.info("Pamiętaj o profilaktyce i zdrowym stylu życia.")

        # Wykres prawdopodobieństwa
        st.progress(probability)
        
    except Exception as e:
        st.error(f"Wystąpił błąd podczas predykcji: {e}")

st.divider()
st.caption("Aplikacja stworzona na podstawie projektu w Google Colab. Model: XGBoost.")
