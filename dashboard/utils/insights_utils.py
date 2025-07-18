import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import streamlit as st


def load_fallback_dataset() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / "data" / "processed" / "clean_telco.csv",
        base_dir / "data" / "raw" / "telco_churn.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue

    # Hardcoded fallback data
    data = [
        ["Female", 0, "Yes", "No", 5, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes",
         "Yes", "Month-to-month", "Yes", "Electronic check", 89.65, 400.5, "0-6", 3, "No"],
        ["Male", 1, "No", "Yes", 40, "Yes", "Yes", "Fiber optic", "Yes", "No", "Yes", "No", "Yes",
         "No", "Two year", "No", "Bank transfer (automatic)", 104.8, 4200.3, "24-48", 4, "No"],
        ["Female", 0, "Yes", "No", 12, "Yes", "No", "DSL", "Yes", "Yes", "No", "No", "No",
         "No", "One year", "Yes", "Credit card (automatic)", 59.95, 750.2, "6-12", 2, "No"],
        ["Male", 0, "No", "No", 2, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "No",
         "No", "Month-to-month", "Yes", "Mailed check", 75.3, 150.6, "0-6", 1, "Yes"],
        ["Female", 0, "No", "No", 60, "Yes", "Yes", "Fiber optic", "Yes", "Yes", "Yes", "Yes", "Yes",
         "Yes", "Two year", "No", "Bank transfer (automatic)", 109.85, 6700.8, "48-72", 6, "No"],
    ]
    cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "tenure_group", "MultipleServices", "Churn"
    ]
    return pd.DataFrame(data, columns=cols)


def get_analysis_dataframe() -> pd.DataFrame:
    if "batch_results" in st.session_state:
        df_ = st.session_state["batch_results"].copy()
    else:
        df_ = load_fallback_dataset()

    if "Churn" not in df_.columns:
        if "Prediccion" in df_.columns:
            df_["Churn"] = np.where(df_["Prediccion"] == 1, "Yes", "No")
        else:
            df_["Churn"] = "No"
    return df_


def map_columns(df_original: pd.DataFrame, df_translated: pd.DataFrame):
    col_map = {k: v for k, v in zip(
        df_original.columns, df_translated.columns)}
    col_map_reverse = {v: k for k, v in col_map.items()}
    return col_map, col_map_reverse


def summarize_feature_importance(df_feat_imp: pd.DataFrame):
    n_up = (df_feat_imp["Coefficient"] > 0).sum()
    n_down = (df_feat_imp["Coefficient"] < 0).sum()
    return n_up, n_down


def plot_feature_importance(df_feat_imp: pd.DataFrame):
    fig = px.bar(
        df_feat_imp,
        x="Abs_Coefficient",
        y="Feature",
        orientation="h",
        title="Importancia de las características",
        labels={"Abs_Coefficient": "Magnitud del impacto",
                "Feature": "Variable"},
        color="Abs_Coefficient",
        color_continuous_scale="viridis"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)


def plot_variable_distribution(df_original: pd.DataFrame, df_translated: pd.DataFrame, selected_var: str, selected_var_original: str):
    is_numeric = pd.api.types.is_numeric_dtype(
        df_original[selected_var_original])

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"**Distribución de {selected_var}**")
        if is_numeric:
            fig_all = px.histogram(df_translated, x=selected_var, nbins=30)
        else:
            count_series = df_translated[selected_var].value_counts(
            ).reset_index()
            count_series.columns = [selected_var, "Cantidad"]
            fig_all = px.bar(count_series, x=selected_var, y="Cantidad")
        st.plotly_chart(fig_all, use_container_width=True)

    with col_right:
        st.markdown(f"**{selected_var} por Churn**")
        churn_col = "Churn"
        if selected_var != churn_col:
            if is_numeric:
                fig_seg = px.box(df_translated, x=churn_col,
                                 y=selected_var, color=churn_col)
            else:
                seg_df = (
                    df_translated.groupby([selected_var, churn_col])
                    .size()
                    .reset_index(name="Cantidad")
                )
                seg_df["Pct"] = seg_df.groupby([selected_var])[
                    "Cantidad"].transform(lambda g: g / g.sum() * 100)
                fig_seg = px.bar(
                    seg_df,
                    x=selected_var,
                    y="Pct",
                    color=churn_col,
                    barmode="group",
                    labels={"Pct": "Porcentaje dentro de categoría"},
                )
            st.plotly_chart(fig_seg, use_container_width=True)


def plot_segment_comparison(df_translated: pd.DataFrame, seg_col: str):
    seg_rate = (
        df_translated.groupby(seg_col)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="Tasa Churn")
    )
    fig_seg_rate = px.bar(seg_rate, x=seg_col,
                          y="Tasa Churn", text="Tasa Churn")
    fig_seg_rate.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig_seg_rate, use_container_width=True)
