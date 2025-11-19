import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# ------------------------------
# Load and prepare data
# ------------------------------
@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel("proyectom.xlsx")
    
    # Target variables
    df["HighPerformance"] = (df["Calificaciones pasadas"] >= 9.2).astype(int)
    
    # Feature engineering - variables normalizadas
    df["eficiencia_estudio_pasado"] = df["Calificaciones pasadas"] / (df["Horas estudio pasadas "] + 1)
    df["intensidad_estudio_actual"] = df["Horas de estudio actuales "] / (df["Materias nuevas"] + 1)
    df["cambio_horas"] = df["Horas de estudio actuales "] - df["Horas estudio pasadas "]
    df["ratio_materias"] = df["Materias nuevas"] / (df["Materias pasadas "] + 1)
    df["tendencia_academica"] = df["Calificaciones pasadas"] * (df["Horas de estudio actuales "] / (df["Horas estudio pasadas "] + 1))
    
    return df

df = load_and_prepare_data()

# Features
feature_cols = [
    "Materias pasadas ",
    "Materias nuevas",
    "Horas de estudio actuales ",
    "Horas estudio pasadas ",
    "Calificaciones pasadas",
    "eficiencia_estudio_pasado",
    "intensidad_estudio_actual",
    "cambio_horas",
    "ratio_materias",
    "tendencia_academica"
]

X = df[feature_cols]

# --------------------------------------------------------
# MODELO 1: REGRESIÓN (CALIFICACIÓN EXACTa) → NO SE TOCA
# --------------------------------------------------------
Y_grade = df["Calificaciones pasadas"]
scaler_reg = StandardScaler()
X_scaled_reg = scaler_reg.fit_transform(X)
model_regression = RandomForestRegressor(
    n_estimators=150, random_state=42, max_depth=6, min_samples_leaf=2
)
model_regression.fit(X_scaled_reg, Y_grade)

# --------------------------------------------------------
# MODELO 2: REGRESIÓN LOGÍSTICA REAL para ≥ 9.2
# --------------------------------------------------------
Y_class = df["HighPerformance"]
scaler_class = StandardScaler()
X_scaled_class = scaler_class.fit_transform(X)

model_classification = LogisticRegression(
    C=1.0,
    max_iter=500,
    solver="lbfgs"
)
model_classification.fit(X_scaled_class, Y_class)

# ------------------------------
# UI
# ------------------------------
st.title("🎓 Predictor de Calificaciones")
st.markdown("*Predice tu calificación esperada y probabilidad de alto rendimiento*")

st.markdown("---")
st.subheader("👤 Información Personal")

col_info1, col_info2 = st.columns(2)

with col_info1:
    gender = st.selectbox("Género", ["Masculino", "Femenino", "Otro"], key="gender")

with col_info2:
    semester = st.selectbox("Semestre actual", list(range(1, 10)), format_func=lambda x: f"{x}° semestre", key="semester")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📚 Semestre Anterior")
    courses_past = st.number_input("Materias cursadas", min_value=1, max_value=15, value=7, key="cp")
    hours_past = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5, key="hp")
    grade_past = st.number_input("Calificación final", min_value=6.0, max_value=10.0, value=9.0, step=0.1, key="gp")

with col2:
    st.subheader("📖 Semestre Actual")
    courses_now = st.number_input("Materias cursando", min_value=1, max_value=15, value=8, key="cn")
    hours_now = st.number_input("Horas de estudio semanales", min_value=1, max_value=30, value=5, key="hn")

# ------------------------------
# Cálculo de features derivadas
# ------------------------------
eficiencia = grade_past / (hours_past + 1)
intensidad = hours_now / (courses_now + 1)
cambio_h = hours_now - hours_past
ratio_mat = courses_now / (courses_past + 1)
tendencia = grade_past * (hours_now / (hours_past + 1))

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔮 Predecir Rendimiento", type="primary"):
    new_data = pd.DataFrame({
        "Materias pasadas ": [courses_past],
        "Materias nuevas": [courses_now],
        "Horas de estudio actuales ": [hours_now],
        "Horas estudio pasadas ": [hours_past],
        "Calificaciones pasadas": [grade_past],
        "eficiencia_estudio_pasado": [eficiencia],
        "intensidad_estudio_actual": [intensidad],
        "cambio_horas": [cambio_h],
        "ratio_materias": [ratio_mat],
        "tendencia_academica": [tendencia]
    })
    
    # --- Predicción de REGRESIÓN ---
    new_data_scaled_reg = scaler_reg.transform(new_data)
    predicted_grade = model_regression.predict(new_data_scaled_reg)[0]
    
    # --- Predicción LOGÍSTICA ---
    new_data_scaled_class = scaler_class.transform(new_data)
    prediction_class = model_classification.predict(new_data_scaled_class)[0]
    probability = model_classification.predict_proba(new_data_scaled_class)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Resultados de la Predicción")
    
    st.info("📌 **Cómo funciona:**\n- 🔴 Regresión: calificación exacta\n- 🟢 Regresión Logística: probabilidad de ≥9.2")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 Calificación Esperada")
        grade_color = "🟢" if predicted_grade >= 9.2 else "🟡" if predicted_grade >= 8.5 else "🔴"
        st.markdown(f"# {grade_color} {predicted_grade:.2f}")
        change = predicted_grade - grade_past
        st.metric("Cambio vs semestre anterior", f"{change:+.2f}", delta=f"{change:+.2f}")

    with col_b:
        st.markdown("### 📈 Alto Rendimiento (≥9.2)")
        prob_color = "🟢" if probability >= 0.7 else "🟡" if probability >= 0.4 else "🔴"
        st.markdown(f"# {prob_color} {probability*100:.1f}%")
        result_text = "✅ SÍ" if prediction_class == 1 else "⚠️ NO"
        st.metric("Predicción", result_text)

    # ------------------------------
    # IMPORTANCIA DE VARIABLES (Coeficientes Logísticos)
    # ------------------------------
    st.markdown("---")
    st.subheader("📈 ¿Qué Afecta Más a tu Alto Rendimiento? (Regresión Logística)")

    feature_names_readable = {
        "Materias pasadas ": "Materias semestre anterior",
        "Materias nuevas": "Materias actuales",
        "Horas de estudio actuales ": "Horas estudio actuales",
        "Horas estudio pasadas ": "Horas estudio anteriores",
        "Calificaciones pasadas": "Calificación pasada",
        "eficiencia_estudio_pasado": "Eficiencia",
        "intensidad_estudio_actual": "Intensidad",
        "cambio_horas": "Cambio en horas",
        "ratio_materias": "Ratio de materias",
        "tendencia_academica": "Tendencia académica"
    }

    coefs = model_classification.coef_[0]

    importance_df = pd.DataFrame({
        "Factor": [feature_names_readable[c] for c in feature_cols],
        "Coeficiente": coefs
    }).sort_values("Coeficiente", ascending=False)

    fig3 = go.Figure(go.Bar(
        x=importance_df["Coeficiente"],
        y=importance_df["Factor"],
        orientation="h"
    ))
    fig3.update_layout(
        title="Importancia (Coeficientes de Regresión Logística)",
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)

# Estadísticas del dataset
with st.expander("📊 Ver estadísticas del dataset"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estudiantes analizados", len(df))
    with col2:
        st.metric("Calificación promedio", f"{df['Calificaciones pasadas'].mean():.2f}")
    with col3:
        st.metric("Alto rendimiento", f"{(Y_class.sum()/len(Y_class)*100):.1f}%")
    with col4:
        st.metric("Horas promedio", f"{df['Horas de estudio actuales '].mean():.1f}")

