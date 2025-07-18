import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from config.config import apply_custom_css, show_header
from utils.model_utils import load_model_components, get_feature_importance
from utils.translations import translate_dataframe, translate_features
from utils.insights_utils import (
    load_fallback_dataset, get_analysis_dataframe,
    map_columns, summarize_feature_importance,
    plot_feature_importance, plot_variable_distribution,
    plot_segment_comparison
)

st.set_page_config(page_title="Insights | Churn Dashboard", layout="wide")
apply_custom_css()

st.title("Insights y explicabilidad del modelo")
st.markdown(
    """
    Comprende **qué variables influyen más en la probabilidad de churn** y explora cómo se distribuyen
    los clientes en segmentos clave.
    """
)
st.markdown("---")

# Cargar datos y preparar DataFrames
df = get_analysis_dataframe()
df_display = translate_dataframe(df)

col_map, col_map_reverse = map_columns(df, df_display)

with st.expander("Dataset utilizado en los gráficos de esta vista"):
    st.dataframe(df_display.head(), use_container_width=True)

# Cargar modelo e importancia de características
model, categorical_columns, feature_names, ohe = load_model_components()
feat_imp_df = get_feature_importance(model, feature_names)

if feat_imp_df.empty:
    st.error("No se pudo calcular la importancia de características.")
    st.stop()

feat_imp_df["Feature"] = translate_features(feat_imp_df["Feature"])

# Sección 1: Importancia características
st.subheader("Top características más influyentes en el churn")
top_n = st.slider("Número de variables a mostrar",
                  min_value=5, max_value=25, value=15)
top_features = feat_imp_df.sort_values(
    "Abs_Coefficient", ascending=False).head(top_n).reset_index(drop=True)

plot_feature_importance(top_features)

n_up, n_down = summarize_feature_importance(top_features)
st.info(f"{n_up} variables aumentan la probabilidad de churn y {n_down} la reducen.")
st.markdown("---")

# Sección 2: Exploración interactiva
st.subheader("Exploración interactiva por variable")
excluded_cols = {"Probabilidad_Churn", "Nivel_Riesgo"}
eda_cols = [c for c in df_display.columns if c not in excluded_cols]

default_idx = eda_cols.index(
    "Tipo de Contrato") if "Tipo de Contrato" in eda_cols else 0
selected_var = st.selectbox(
    "Selecciona una variable:", eda_cols, index=default_idx)
selected_var_original = col_map_reverse.get(selected_var, selected_var)

plot_variable_distribution(df, df_display, selected_var, selected_var_original)
st.markdown("---")

# Sección 3: Segmentos clave
st.subheader("Segmentos clave de negocio")
segment_cols = [c for c in ["Tipo de Contrato", "Método de Pago",
                            "Servicio de Internet"] if c in df_display.columns]

if segment_cols:
    seg_col = st.selectbox("Segmento a comparar:", segment_cols)
    plot_segment_comparison(df_display, seg_col)
else:
    st.warning("No se encontraron columnas de segmento clave.")

st.markdown("---")
