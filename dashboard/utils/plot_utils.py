import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_model_performance(metrics):
    metrics_data = {
        'F1-Score': metrics['f1_score'],
        'ROC AUC': metrics['roc_auc'],
        'Precisión': metrics['precision'],
        'Recall': metrics['recall'],
        'Especificidad': metrics.get('specificity', 0)
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig = go.Figure(data=[go.Bar(
        x=list(metrics_data.keys()),
        y=list(metrics_data.values()),
        marker_color=colors,
        text=[f"{v:.1%}" for v in metrics_data.values()],
        textposition='auto',
        hovertemplate='%{x}: %{y:.2%}<extra></extra>'
    )])

    fig.update_layout(
        title="Rendimiento del modelo",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=400,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig


def plot_churn_distribution(churn_data: pd.DataFrame):
    fig_pie = px.pie(
        churn_data,
        names='Churn',
        values='Count',
        color='Churn',
        color_discrete_map={'No': '#2ca02c', 'Sí': '#d62728'},
        title='Distribución de clientes'
    )
    fig_pie.update_traces(textinfo='percent+label', hole=0.4)
    fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    return fig_pie


def plot_risk_factors(risk_factors: dict):
    fig = go.Figure(go.Bar(
        x=list(risk_factors.values()),
        y=list(risk_factors.keys()),
        orientation='h',
        marker_color='#ff7f0e',
        text=[f"{v:.0%}" for v in risk_factors.values()],
        textposition='auto',
        hovertemplate='%{y}: %{x:.0%}<extra></extra>'
    ))
    fig.update_layout(
        title="Principales factores de riesgo",
        height=350,
        margin=dict(t=50, b=40, l=80, r=40)
    )
    return fig


def plot_churn_reduction(months, without_model, with_model):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=without_model,
        name='Sin modelo',
        mode='lines+markers',
        line=dict(color='#d62728')
    ))
    fig.add_trace(go.Scatter(
        x=months,
        y=with_model,
        name='Con modelo',
        mode='lines+markers',
        line=dict(color='#2ca02c')
    ))
    fig.update_layout(
        title="Reducción proyectada de churn",
        yaxis=dict(title="Clientes en riesgo"),
        height=400,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig
