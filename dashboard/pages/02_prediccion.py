import streamlit as st
import pandas as pd
from utils.model_utils import (
    load_model_components, preprocess_input, predict_churn, interpret_prediction, validate_input_data
)
from config.config import FORM_OPTIONS, NUMERIC_RANGES, apply_custom_css, show_header
from utils.translations import translate_dataframe, translate_options, reverse_translate
from utils.prediction_utils import build_input_dataframe

st.set_page_config(page_title="Predicción | Churn Dashboard", layout="wide")
apply_custom_css()

model, categorical_columns, feature_names, ohe = load_model_components()

st.title("Predicción individual de churn")
st.markdown(
    "Ingrese los datos del cliente para obtener la probabilidad de churn y una recomendación.")
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("Datos del cliente")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox(
            "Género", translate_options(FORM_OPTIONS['gender']))
        senior = st.selectbox("¿Es adulto mayor?", translate_options(
            FORM_OPTIONS['SeniorCitizen']))
        partner = st.selectbox(
            "¿Tiene pareja?", translate_options(FORM_OPTIONS['Partner']))
        dependents = st.selectbox(
            "¿Tiene dependientes?", translate_options(FORM_OPTIONS['Dependents']))

    with col2:
        tenure = st.number_input(
            "Meses de permanencia",
            min_value=NUMERIC_RANGES['tenure']['min'],
            max_value=NUMERIC_RANGES['tenure']['max'],
            value=NUMERIC_RANGES['tenure']['default'],
            step=1
        )
        phone_service = st.selectbox(
            "Servicio telefónico", translate_options(FORM_OPTIONS['PhoneService']))
        multiple_lines = st.selectbox(
            "Líneas múltiples", translate_options(FORM_OPTIONS['MultipleLines']))
        internet_service = st.selectbox(
            "Servicio de Internet", translate_options(FORM_OPTIONS['InternetService']))

    with col3:
        online_security = st.selectbox(
            "Seguridad en línea", translate_options(FORM_OPTIONS['OnlineSecurity']))
        online_backup = st.selectbox(
            "Respaldo en línea", translate_options(FORM_OPTIONS['OnlineBackup']))
        device_protection = st.selectbox(
            "Protección de dispositivos", translate_options(FORM_OPTIONS['DeviceProtection']))
        tech_support = st.selectbox(
            "Soporte técnico", translate_options(FORM_OPTIONS['TechSupport']))

    col4, col5 = st.columns(2)
    with col4:
        streaming_tv = st.selectbox(
            "Streaming TV", translate_options(FORM_OPTIONS['StreamingTV']))
        streaming_movies = st.selectbox(
            "Streaming Películas", translate_options(FORM_OPTIONS['StreamingMovies']))
        contract = st.selectbox(
            "Tipo de contrato", translate_options(FORM_OPTIONS['Contract']))
    with col5:
        paperless_billing = st.selectbox(
            "Facturación electrónica", translate_options(FORM_OPTIONS['PaperlessBilling']))
        payment_method = st.selectbox(
            "Método de pago", translate_options(FORM_OPTIONS['PaymentMethod']))
        monthly_charges = st.number_input(
            "Cargos mensuales",
            min_value=NUMERIC_RANGES['MonthlyCharges']['min'],
            max_value=NUMERIC_RANGES['MonthlyCharges']['max'],
            value=NUMERIC_RANGES['MonthlyCharges']['default'],
            step=1.0,
            format="%.2f"
        )
        total_charges = st.number_input(
            "Cargos totales",
            min_value=NUMERIC_RANGES['TotalCharges']['min'],
            max_value=NUMERIC_RANGES['TotalCharges']['max'],
            value=NUMERIC_RANGES['TotalCharges']['default'],
            step=1.0,
            format="%.2f"
        )

    tenure_group = st.selectbox(
        "Grupo de permanencia", translate_options(FORM_OPTIONS['tenure_group']))
    multiple_services = st.number_input(
        "Cantidad de servicios",
        min_value=NUMERIC_RANGES['MultipleServices']['min'],
        max_value=NUMERIC_RANGES['MultipleServices']['max'],
        value=NUMERIC_RANGES['MultipleServices']['default'],
        step=1
    )

    submitted = st.form_submit_button("Predecir churn")

if submitted:
    input_data = build_input_dataframe({
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group,
        'MultipleServices': multiple_services
    }, reverse_translate)

    if validate_input_data(input_data):
        processed_data = preprocess_input(
            input_data, categorical_columns, feature_names, ohe)
        predictions, probabilities = predict_churn(model, processed_data)

        if predictions is not None:
            prob = probabilities[0]
            interpretation = interpret_prediction(prob)

            st.markdown("---")
            st.subheader("Resultado de la predicción")

            st.markdown(
                f"### Nivel de riesgo: **{interpretation['risk_level']}**")
            st.progress(int(interpretation['probability'] * 100))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilidad de churn",
                          interpretation['confidence'])
            with col2:
                advice = {
                    "ALTO": "**Consejo:** Contactar al cliente y ofrecer beneficios exclusivos.",
                    "MEDIO": "**Consejo:** Mantener comunicación activa y monitorear satisfacción.",
                    "BAJO": "**Consejo:** Continuar con el servicio habitual."
                }
                st.markdown(advice.get(interpretation['risk_level'], ""))

            alert_func = {
                "ALTO": st.error,
                "MEDIO": st.warning,
                "BAJO": st.success
            }
            alert_func.get(interpretation['risk_level'], st.info)(
                {
                    "ALTO": "Cliente en alto riesgo de abandono. Acción inmediata requerida.",
                    "MEDIO": "Cliente con riesgo moderado. Considerar ofertas de retención.",
                    "BAJO": "Cliente estable. No se requieren acciones inmediatas."
                }[interpretation['risk_level']]
            )

            st.write("### Datos del cliente analizado:")
            st.dataframe(translate_dataframe(input_data),
                         use_container_width=True)
