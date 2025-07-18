import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import streamlit as st


def simulate_metrics_data():
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=30)

    accuracy = np.clip(
        0.78 + np.cumsum(np.random.normal(0, 0.002, len(dates))), 0.70, 0.88)
    auc = np.clip(
        0.82 + np.cumsum(np.random.normal(0, 0.002, len(dates))), 0.74, 0.92)
    f1 = np.clip(
        0.66 + np.cumsum(np.random.normal(0, 0.003, len(dates))), 0.58, 0.80)

    df_metrics = pd.DataFrame({
        "Fecha": dates,
        "Accuracy": accuracy,
        "ROC AUC": auc,
        "F1 Score": f1,
    }).set_index("Fecha")
    return df_metrics


def plot_metrics_evolution(df_metrics, df_recent, thr_acc, thr_auc, thr_f1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_metrics.index,
                  y=df_metrics["Accuracy"], mode="lines+markers", name="Accuracy"))
    fig.add_trace(go.Scatter(x=df_metrics.index,
                  y=df_metrics["ROC AUC"], mode="lines+markers", name="ROC AUC"))
    fig.add_trace(go.Scatter(x=df_metrics.index,
                  y=df_metrics["F1 Score"], mode="lines+markers", name="F1 Score"))

    # Líneas umbral
    fig.add_hline(y=thr_acc, line_dash="dot",
                  line_color="red", annotation_text="Acc mín.")
    fig.add_hline(y=thr_auc, line_dash="dot",
                  line_color="orange", annotation_text="AUC mín.")
    fig.add_hline(y=thr_f1, line_dash="dot",
                  line_color="purple", annotation_text="F1 mín.")

    # Sombrear ventana reciente
    fig.add_vrect(
        x0=df_recent.index.min(), x1=df_recent.index.max(),
        fillcolor="LightSalmon", opacity=0.15, line_width=0,
        annotation_text="Ventana reciente", annotation_position="top left"
    )

    fig.update_layout(
        height=450,
        yaxis=dict(range=[0.5, 1.0]),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        title="Métricas del modelo por fecha"
    )
    st.plotly_chart(fig, use_container_width=True)


def calculate_drift(df_recent, thr_acc, thr_auc, thr_f1):
    mean_acc = df_recent["Accuracy"].mean()
    mean_auc = df_recent["ROC AUC"].mean()
    mean_f1 = df_recent["F1 Score"].mean()

    drift_acc = mean_acc < thr_acc
    drift_auc = mean_auc < thr_auc
    drift_f1 = mean_f1 < thr_f1
    return drift_acc, drift_auc, drift_f1
