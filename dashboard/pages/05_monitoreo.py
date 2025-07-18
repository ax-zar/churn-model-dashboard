import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

from config.config import apply_custom_css, show_header
from utils.monitoring_utils import (
    simulate_metrics_data, calculate_drift,
    plot_metrics_evolution
)

# Configuración de página
st.set_page_config(page_title="Monitoreo | Churn Dashboard", layout="wide")
apply_custom_css()

st.title("Monitoreo de métricas del modelo en el tiempo")
st.warning(
    "**Nota:** Los datos mostrados son *simulados* con fines demostrativos."
)
st.markdown("---")

# Sidebar: configuración
with st.sidebar:
    st.header("Configuración de monitoreo")
    window_days = st.slider("Ventana reciente (días)", 3, 30, 7, step=1)
    st.markdown("**Umbrales mínimos aceptables**")
    thr_acc = st.number_input("Accuracy mín.", 0.0,
                              1.0, 0.75, 0.01, format="%.2f")
    thr_auc = st.number_input("ROC AUC mín.", 0.0, 1.0,
                              0.78, 0.01, format="%.2f")
    thr_f1 = st.number_input("F1 Score mín.", 0.0, 1.0,
                             0.62, 0.01, format="%.2f")

# Datos simulados
df_metrics = simulate_metrics_data()

# Datos recientes según ventana
df_recent = df_metrics.tail(window_days)

# Mostrar KPIs
st.subheader("Métricas recientes")
last = df_metrics.iloc[-1]

if len(df_metrics) > window_days:
    prev_mean = df_metrics.iloc[:-window_days].mean()
else:
    prev_mean = df_metrics.mean()

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{last['Accuracy']:.3f}",
          delta=f"{(last['Accuracy']-prev_mean['Accuracy']):+.3f}")
c2.metric("ROC AUC", f"{last['ROC AUC']:.3f}",
          delta=f"{(last['ROC AUC']-prev_mean['ROC AUC']):+.3f}")
c3.metric("F1 Score", f"{last['F1 Score']:.3f}",
          delta=f"{(last['F1 Score']-prev_mean['F1 Score']):+.3f}")

# Gráfico histórico
st.subheader("Evolución histórica de métricas")
plot_metrics_evolution(df_metrics, df_recent, thr_acc, thr_auc, thr_f1)

# Detección de drift
st.subheader("Detección de drift")
drift_acc, drift_auc, drift_f1 = calculate_drift(
    df_recent, thr_acc, thr_auc, thr_f1)

if drift_acc or drift_auc or drift_f1:
    st.error("Se detecta posible degradación del modelo en la ventana reciente.")
    st.write(
        f"Accuracy reciente: {df_recent['Accuracy'].mean():.3f} {'❌' if drift_acc else '✅'}")
    st.write(
        f"ROC AUC reciente: {df_recent['ROC AUC'].mean():.3f} {'❌' if drift_auc else '✅'}")
    st.write(
        f"F1 Score reciente: {df_recent['F1 Score'].mean():.3f} {'❌' if drift_f1 else '✅'}")
else:
    st.success("No se detecta drift significativo en las métricas recientes.")

# Explicación drift
with st.expander("¿Qué es drift y cómo monitorearlo?"):
    st.markdown(
        """
        **Drift de datos/modelo** ocurre cuando el comportamiento de los datos o el desempeño del modelo cambia con el tiempo.  
        Esto puede provocar predicciones menos confiables si no se detecta y corrige.

        **Buenas prácticas:**
        - Monitorear métricas clave (AUC, F1, Recall) por período.
        - Comparar la distribución de variables vs datos de entrenamiento.
        - Establecer umbrales y alertas automáticas.
        - Reentrenar cuando el desempeño caiga por debajo del umbral.
        """
    )
