# app_diabete_portfolio_elegant.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# ===============================
# 🌿 CONFIGURATION GÉNÉRALE
# ===============================
st.set_page_config(
    page_title="🩺 Prédiction du Diabète",
    page_icon="💫",
    layout="wide"
)

# 🌸 Thème personnalisé via CSS
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
    color: #222;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-weight: 600;
    color: #2b2b2b;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
    padding: 1.5rem 1rem;
    border-radius: 10px;
}
.block-container {
    padding-top: 2rem;
}
div.stButton>button {
    background: linear-gradient(90deg, #84fab0 0%, #8fd3f4 100%);
    color: #fff;
    border-radius: 10px;
    height: 3rem;
    font-weight: 600;
    border: none;
}
div.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #8fd3f4 0%, #84fab0 100%);
}
hr {
    border: 0;
    border-top: 1px solid #e5e5e5;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🩺 TITRE ET INTRO
# ===============================
st.title("💫 Application de Prédiction du Diabète")
st.markdown("""
Bienvenue dans une expérience interactive de **Machine Learning appliquée à la santé**.  
Cette application estime le risque de **diabète** à partir de données médicales réelles  
en utilisant un modèle de **forêt aléatoire**.
""")
st.markdown("---")

# ===============================
# 📂 CHARGEMENT DU DATASET
# ===============================
possible_paths = [
    "data/diabetes.csv",
    "diabetes.csv",
    "./diabetes.csv",
    "/app/data/diabetes.csv",
]

csv_path = next((p for p in possible_paths if os.path.exists(p)), None)

if not csv_path:
    st.error("⚠️ Le fichier `diabetes.csv` est introuvable.")
    st.stop()

df = pd.read_csv(csv_path, sep=";") if ";" in open(csv_path).readline() else pd.read_csv(csv_path)
st.success(f"✅ Données chargées depuis : `{csv_path}`")

# Nettoyage automatique des colonnes
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_scaled, y)

# ===============================
# 🎛️ SIDEBAR — PARAMÈTRES UTILISATEUR
# ===============================
st.sidebar.header("🧬 Paramètres du patient")

pregnancies = st.sidebar.number_input("Grossesses", 0, 20, 2)
glucose = st.sidebar.slider("Glucose (mg/dL)", 50, 200, 100)
blood_pressure = st.sidebar.slider("Pression artérielle (mm Hg)", 40, 120, 70)
skin_thickness = st.sidebar.slider("Épaisseur du pli cutané (mm)", 10, 99, 20)
insulin = st.sidebar.slider("Insuline (mu U/ml)", 0, 900, 80)
bmi = st.sidebar.slider("BMI (Indice de Masse Corporelle)", 15.0, 67.0, 30.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Âge (années)", 18, 100, 35)

user_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

st.markdown("### 📋 Données saisies")
st.dataframe(user_data, use_container_width=True)

# ===============================
# 🔮 PRÉDICTION
# ===============================
user_scaled = scaler.transform(user_data)

if st.button("🔮 Lancer la prédiction"):
    pred = model.predict(user_scaled)[0]
    proba = model.predict_proba(user_scaled)[0][1] * 100

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ Risque **élevé** de diabète — probabilité : **{proba:.1f}%**")
    else:
        st.success(f"✅ Risque **faible** de diabète — probabilité : **{proba:.1f}%**")

    # ===============================
    # 📊 IMPORTANCE DES VARIABLES
    # ===============================
    st.markdown("### 📊 Importance des variables")

    importances = pd.DataFrame({
        "Variable": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Importance", y="Variable", data=importances, palette="viridis")
    plt.title("Importance des variables – Random Forest")
    st.pyplot(fig)

# ===============================
# 📈 VISUALISATION DU DATASET
# ===============================
with st.expander("📈 Explorer le dataset"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution du Glucose**")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Glucose"], bins=20, kde=True, color="#5cb0ff", ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("**Distribution du BMI**")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["BMI"], bins=20, kde=True, color="#90dca6", ax=ax2)
        st.pyplot(fig2)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("✨ Application développée avec Streamlit, Pandas et Scikit-learn – Design inspiré d’EchoWrite 🌸")
