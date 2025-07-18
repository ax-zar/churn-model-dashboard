import streamlit as st
import pandas as pd
from utils.dashboard_utils import get_dashboard_metrics, get_risk_factors
from utils.plot_utils import (
    plot_model_performance,
    plot_churn_distribution,
    plot_risk_factors,
    plot_churn_reduction
)
from config.config import apply_custom_css, show_header

# Configuración inicial
st.set_page_config(page_title="Home | Churn Dashboard", layout="wide")
apply_custom_css()

# UI principal
st.title("Visión general")
st.markdown(
    "### Bienvenido al dashboard de predicción de churn para clientes de telecomunicaciones")
st.write("Este dashboard presenta insights clave del análisis, rendimiento del modelo y el impacto estimado en el negocio.")
st.markdown("---")

# Obtener métricas y graficar rendimiento del modelo
metrics = get_dashboard_metrics()
st.subheader("Métricas del modelo")
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="F1-Score", value=f"{metrics['f1_score']:.3f}", delta="Óptimo",
            delta_color="normal", help="Balance entre precisión y recall")
col2.metric(label="ROC AUC", value=f"{metrics['roc_auc']:.3f}", delta="Excelente",
            delta_color="normal", help="Capacidad de discriminación del modelo")
col3.metric(label="Precisión", value=f"{metrics['precision']:.1%}",
            help="Proporción de predicciones positivas correctas")
col4.metric(label="Recall",
            value=f"{metrics['recall']:.1%}", help="Porcentaje de churn reales detectados")

fig_perf = plot_model_performance(metrics)
st.plotly_chart(fig_perf, use_container_width=True)

st.markdown("---")

# Distribución de churn
st.subheader("Distribución de churn en el dataset")
churn_data = pd.DataFrame({'Churn': ['No', 'Sí'], 'Count': [5174, 1869]})

col1, col2 = st.columns(2)
with col1:
    fig_churn = plot_churn_distribution(churn_data)
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    total_clients = churn_data['Count'].sum()
    active_clients = churn_data.loc[churn_data['Churn']
                                    == 'No', 'Count'].values[0]
    churn_clients = churn_data.loc[churn_data['Churn']
                                   == 'Sí', 'Count'].values[0]
    st.metric("Total de clientes", f"{total_clients:,}")
    st.metric("Clientes activos",
              f"{active_clients:,} ({active_clients/total_clients:.1%})")
    st.metric("Clientes churn",
              f"{churn_clients:,} ({churn_clients/total_clients:.1%})")

st.markdown("---")

# Resumen Ejecutivo
st.subheader("Resumen ejecutivo")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Hallazgos clave**
    - Contratos **mensuales** presentan mayor churn.
    - Clientes con **cargos altos** son más propensos a abandonar.
    - **Servicios múltiples** reducen el riesgo de abandono.
    - Clientes nuevos (0-6 meses) son los más vulnerables.
    """)

with col2:
    risk_factors = get_risk_factors()
    fig_risk = plot_risk_factors(risk_factors)
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# Impacto en negocio
st.subheader("Impacto estimado en negocio")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Ahorro potencial", value="$1.2M / año",
              help="Retención proactiva de clientes")
with col2:
    st.metric(label="ROI estimado", value="400%",
              help="Retorno de inversión por cada $1 invertido")
with col3:
    st.metric(label="Clientes salvados", value="~460 / mes",
              help="Precisión del 62% en detección")

months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']
without_model = [180, 185, 190, 195, 200, 205]
with_model = [180, 160, 145, 130, 115, 100]
fig_line = plot_churn_reduction(months, without_model, with_model)
st.plotly_chart(fig_line, use_container_width=True)
