import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Configurar rutas relativas
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_model_components():
    """
    Carga todos los componentes del modelo entrenado, incluyendo el encoder OneHotEncoder.
    """
    try:
        # Cargar modelo
        model = joblib.load(MODELS_DIR / "churn_model.pkl")

        # Cargar columnas categóricas
        categorical_columns = joblib.load(
            MODELS_DIR / "categorical_columns.pkl")

        # Cargar nombres de features
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

        # Cargar encoder OneHotEncoder
        ohe = joblib.load(MODELS_DIR / "ohe_encoder.pkl")

        return model, categorical_columns, feature_names, ohe

    except FileNotFoundError as e:
        st.error(f"No se encontraron los archivos del modelo: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.stop()


def preprocess_input(data, categorical_columns, feature_names, ohe):
    """
    Preprocesa los datos de entrada usando el encoder OneHotEncoder cargado.

    Args:
        data: DataFrame con los datos de entrada
        categorical_columns: Lista de columnas categóricas
        feature_names: Lista de nombres de features esperados
        ohe: Encoder OneHotEncoder cargado

    Returns:
        DataFrame procesado listo para predicción
    """
    processed_data = data.copy()

    # Separar categóricas y numéricas
    data_cat = processed_data[categorical_columns]
    data_num = processed_data.drop(columns=categorical_columns)

    # Transformar categóricas con el encoder
    cat_encoded = ohe.transform(data_cat)
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(
        categorical_columns), index=processed_data.index)

    # Unir variables numéricas y categóricas codificadas
    processed_data = pd.concat([data_num, cat_encoded_df], axis=1)

    # Añadir columnas que falten y poner en orden correcto
    for col in feature_names:
        if col not in processed_data.columns:
            processed_data[col] = 0

    processed_data = processed_data[feature_names]

    return processed_data


def predict_churn(model, data):
    """
    Hace predicción de churn para los datos dados.

    Args:
        model: Modelo entrenado
        data: DataFrame procesado

    Returns:
        tuple: (predicciones, probabilidades)
    """
    try:
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)
        return predictions, probabilities
    except Exception as e:
        st.error(f"Error al hacer predicción: {e}")
        return None, None


def get_feature_importance(model, feature_names):
    """
    Obtiene la importancia de las características del modelo.

    Args:
        model: Modelo entrenado (Logistic Regression)
        feature_names: Lista de nombres de features

    Returns:
        DataFrame con importancia de features
    """
    try:
        coefficients = model.coef_[0]

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients),
            'Impact': ['Aumenta Churn' if coef > 0 else 'Reduce Churn' for coef in coefficients]
        }).sort_values('Abs_Coefficient', ascending=False)

        return importance_df
    except Exception as e:
        st.error(f"Error al obtener importancia de features: {e}")
        return pd.DataFrame()


def interpret_prediction(probability, customer_data=None):
    """
    Interpreta la predicción y proporciona contexto.

    Args:
        probability: Probabilidad de churn (0-1)
        customer_data: Datos del cliente (opcional)

    Returns:
        dict con interpretación
    """
    churn_prob = probability[1] if len(probability) > 1 else probability

    if churn_prob < 0.3:
        risk_level = "BAJO"
        color = "green"
        recommendation = "Cliente estable. Continuar con servicio regular."
    elif churn_prob < 0.6:
        risk_level = "MEDIO"
        color = "orange"
        recommendation = "Cliente en riesgo moderado. Considerar ofertas de retención."
    else:
        risk_level = "ALTO"
        color = "red"
        recommendation = "Cliente en alto riesgo. Acción inmediata requerida."

    return {
        'probability': churn_prob,
        'risk_level': risk_level,
        'color': color,
        'recommendation': recommendation,
        'confidence': f"{churn_prob:.1%}"
    }


def get_model_metrics():
    """
    Retorna las métricas del modelo para mostrar en el dashboard.
    """
    return {
        'f1_score': 0.620,
        'roc_auc': 0.840,
        'precision': 0.620,
        'recall': 0.620,
        'specificity': 0.863,
        'accuracy': 0.799
    }


def validate_input_data(data):
    """
    Valida que los datos de entrada sean correctos.

    Args:
        data: DataFrame con datos de entrada

    Returns:
        bool: True si los datos son válidos
    """
    required_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'tenure_group', 'MultipleServices'
    ]

    missing_columns = [
        col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Faltan las siguientes columnas: {missing_columns}")
        return False

    return True
