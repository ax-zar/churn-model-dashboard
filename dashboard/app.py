import streamlit as st
from config.config import apply_custom_css, show_header

# Configuración inicial
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
apply_custom_css()
show_header()

st.markdown("""
Este proyecto presenta una solución **end-to-end** para abordar la problemática de **churn (abandono de clientes) en telecomunicaciones**, 
integrando análisis exploratorio, modelado predictivo y visualización interactiva en un dashboard profesional.

### Contexto del problema
El abandono de clientes impacta directamente en los ingresos recurrentes de las empresas. Detectar patrones de comportamiento 
y anticipar el churn permite diseñar estrategias de retención más efectivas y reducir costos de adquisición.

---

### Objetivos del proyecto
- Predecir la probabilidad de churn para clientes individuales y en lotes masivos.
- Clasificar clientes en **niveles de riesgo (BAJO, MEDIO, ALTO)** para priorizar acciones.
- Proporcionar **interpretabilidad** mediante análisis de características clave.
- Monitorear el desempeño del modelo a lo largo del tiempo para detectar **drift** y degradación.

---

### Flujo de trabajo implementado
1. **Análisis exploratorio de datos (EDA)**  
   - Limpieza y preprocesamiento de datos del dataset **Telco Customer Churn**.
   - Visualización de distribuciones y correlaciones relevantes.
   - Creación de variables derivadas (tenure groups, servicios múltiples).

2. **Entrenamiento y selección del modelo**  
   - Comparación de modelos (Regresión Logística, Random Forest, XGBoost).
   - Validación cruzada y optimización de hiperparámetros.
   - Selección del modelo final **Logistic Regression** por su balance entre interpretabilidad y performance  
     *(F1: 0.63, ROC AUC: 0.84)*.

3. **Construcción del dashboard**  
   - **Predicción individual**: Formulario dinámico para análisis puntual.  
   - **Análisis masivo**: Carga de CSV para evaluar grandes lotes de clientes.  
   - **Insights**: Importancia de características y análisis interactivo por variables.  
   - **Monitoreo**: Simulación de métricas históricas y detección simple de drift.  

---

### Recursos
- **Dataset**: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Repositorio GitHub**: [Proyecto Churn Dashboard](https://github.com/ax-zar/churn-model-dashboard)
- **Perfil LinkedIn**: [Axel Zaragoza](https://www.linkedin.com/in/axzar/)

---
Navega en el **menú lateral** para explorar las funcionalidades del dashboard.
""")
