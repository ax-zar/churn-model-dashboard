# Configuración del dashboard
import streamlit as st

# Configuración de la página
PAGE_CONFIG = {
    "page_title": "Churn Predictor - Telco",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Configuración del tema
THEME_CONFIG = {
    "primary_color": "#1f77b4",
    "background_color": "#ffffff",
    "secondary_background_color": "#f0f2f6",
    "text_color": "#262730"
}

# Opciones para formularios
FORM_OPTIONS = {
    'gender': ['Male', 'Female'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['No', 'Yes', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['No', 'Yes', 'No internet service'],
    'OnlineBackup': ['No', 'Yes', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport': ['No', 'Yes', 'No internet service'],
    'StreamingTV': ['No', 'Yes', 'No internet service'],
    'StreamingMovies': ['No', 'Yes', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    'tenure_group': ['0-6', '6-12', '12-24', '24-48', '48-72']
}

# Rangos para variables numéricas
NUMERIC_RANGES = {
    'tenure': {'min': 0, 'max': 72, 'default': 12},
    'MonthlyCharges': {'min': 18.0, 'max': 120.0, 'default': 65.0},
    'TotalCharges': {'min': 18.0, 'max': 8500.0, 'default': 1500.0},
    'MultipleServices': {'min': 0, 'max': 10, 'default': 3}
}

# Textos para el dashboard
TEXTS = {
    'app_title': '📊 Predictor de Churn - Telco Customer',
    'app_description': 'Sistema de predicción de abandono de clientes basado en Machine Learning',
    'sidebar_title': '🎯 Navegación',
    'model_info': {
        'name': 'Logistic Regression',
        'description': 'Modelo optimizado para balance entre precisión y recall'
    }
}

# Configuración de métricas
METRICS_CONFIG = {
    'f1_threshold': 0.6,
    'roc_threshold': 0.8,
    'precision_threshold': 0.6
}


def apply_custom_css():
    """
    Aplica CSS personalizado para mejorar la apariencia del dashboard.
    """
    st.markdown("""
    <style>
    /* Estilo para métricas */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    /* Estilo para alertas */
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    
    /* Estilo para sidebar */
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Estilo para botones */
    .stButton > button {
        width: 100%;
        border-radius: 0.375rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Ocultar elementos de Streamlit */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Estilo para títulos */
    h1 {
        color: #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h2 {
        color: #495057;
        margin-top: 2rem;
    }
    
    /* Estilo para el header */
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #17a2b8 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


def show_header():
    """
    Muestra el header principal del dashboard con enlaces clave.
    """
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #1f77b4 0%, #17a2b8 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;">
        <h1 style="margin: 0; font-size: 3rem;">Churn Prediction Dashboard</h1>
        <p style="margin: 0; font-size: 2rem; opacity: 0.95;">
            Predicción y análisis del abandono de clientes en telecomunicaciones
        </p>
        <div style="margin-top: 0.8rem;">
            <a href="https://www.kaggle.com/blastchar/telco-customer-churn" target="_blank" style="color: white; margin: 0 10px; text-decoration: underline;">Dataset</a> |
            <a href="https://github.com/ax-zar/churn-model-dashboard" target="_blank" style="color: white; margin: 0 10px; text-decoration: underline;">GitHub</a> |
            <a href="https://www.linkedin.com/in/axzar/" target="_blank" style="color: white; margin: 0 10px; text-decoration: underline;">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
