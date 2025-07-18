import streamlit as st
import pandas as pd
import io
import plotly.express as px
from utils.model_utils import load_model_components, preprocess_input, predict_churn
from utils.translations import translate_dataframe
from config.config import apply_custom_css
from utils.bulk_analysis_utils import (
    get_example_dataframe, validate_uploaded_dataframe,
    add_predictions_and_risk_levels, classify_risk_level
)
apply_custom_css()

st.set_page_config(
    page_title="Análisis Masivo | Churn Dashboard", layout="wide")

model, categorical_columns, feature_names, ohe = load_model_components()

st.title("Análisis masivo de clientes")

example_df = get_example_dataframe()
csv_data = example_df.to_csv(index=False)
st.download_button(
    label="Descargar CSV de ejemplo",
    data=csv_data,
    file_name="ejemplo_churn.csv",
    mime="text/csv"
)

st.markdown("Sube un archivo CSV con clientes para predecir churn en lote.")
uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Archivo cargado: {uploaded_file.name}")
        st.write("Primeras filas del dataset:")
        st.dataframe(translate_dataframe(df_raw.head()),
                     use_container_width=True)

        required_cols = example_df.columns.tolist()
        missing_cols = validate_uploaded_dataframe(df_raw, required_cols)
        if missing_cols:
            st.error(f"Faltan columnas: {missing_cols}")
            st.stop()

        df_processed = preprocess_input(
            df_raw, categorical_columns, feature_names, ohe)
        preds, probs = predict_churn(model, df_processed)

        df_results = add_predictions_and_risk_levels(
            df_raw, preds, probs, classify_risk_level)

        st.subheader("Resumen del análisis")
        total = len(df_results)
        altos = (df_results["Nivel_Riesgo"] == "ALTO").sum()
        medios = (df_results["Nivel_Riesgo"] == "MEDIO").sum()
        bajos = (df_results["Nivel_Riesgo"] == "BAJO").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Clientes en lote", total)
        col2.metric("Riesgo ALTO", altos, delta=f"{(altos/total)*100:.1f}%")
        col3.metric("Riesgo MEDIO", medios, delta=f"{(medios/total)*100:.1f}%")

        st.subheader("Distribución por nivel de riesgo")
        fig = px.histogram(
            df_results,
            x="Nivel_Riesgo",
            color="Nivel_Riesgo",
            category_orders={"Nivel_Riesgo": ["BAJO", "MEDIO", "ALTO"]},
            title="Distribución de clientes por nivel de riesgo"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top 10 clientes ordenados por riesgo")
        top_clients = df_results.sort_values(
            by="Probabilidad_Churn", ascending=False).head(10)
        st.dataframe(
            translate_dataframe(
                top_clients[["Probabilidad_Churn", "Nivel_Riesgo"] + required_cols]),
            use_container_width=True
        )

        st.subheader("Descargar resultados")
        buffer = io.BytesIO()
        df_results.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Descargar CSV con resultados",
            data=buffer,
            file_name="predicciones_churn.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
