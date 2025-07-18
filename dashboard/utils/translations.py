# utils/translations.py

# Traducción de nombres de columnas
translation_dict = {
    "gender": "Género",
    "SeniorCitizen": "Ciudadano Senior",
    "Partner": "Tiene Pareja",
    "Dependents": "Dependientes",
    "tenure": "Meses Permanencia",
    "PhoneService": "Servicio Telefónico",
    "MultipleLines": "Líneas Múltiples",
    "InternetService": "Servicio de Internet",
    "OnlineSecurity": "Seguridad en Línea",
    "OnlineBackup": "Respaldo en Línea",
    "DeviceProtection": "Protección de Dispositivos",
    "TechSupport": "Soporte Técnico",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Películas",
    "Contract": "Tipo de Contrato",
    "PaperlessBilling": "Facturación Electrónica",
    "PaymentMethod": "Método de Pago",
    "MonthlyCharges": "Cargos Mensuales",
    "TotalCharges": "Cargos Totales",
    "tenure_group": "Grupo de Permanencia",
    "MultipleServices": "Cantidad de Servicios",
    "Churn": "Churn"
}

# Traducción de valores (categorías de select)
value_translation = {
    "Female": "Femenino", "Male": "Masculino",
    "No": "No", "Yes": "Sí",
    "No internet service": "Sin servicio",
    "Fiber optic": "Fibra óptica",
    "DSL": "DSL",
    "Month-to-month": "Mensual",
    "One year": "Un año",
    "Two year": "Dos años",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque físico",
    "Bank transfer (automatic)": "Transferencia bancaria (automática)",
    "Credit card (automatic)": "Tarjeta de crédito (automática)"
}

# Creamos reverse dict para mapear ES → EN
reverse_value_translation = {v: k for k, v in value_translation.items()}


# ------------------ FUNCIONES ------------------

def translate_dataframe(df):
    """
    Devuelve una copia traducida de las columnas para mostrar en UI.
    """
    df_copy = df.copy()
    df_copy.rename(columns=translation_dict, inplace=True)
    return df_copy


def translate_features(features):
    """
    Traduce una lista/serie de nombres de features.
    """
    return [translation_dict.get(f, f) for f in features]


def translate_options(options):
    """
    Traduce lista de opciones (valores de un selectbox).
    """
    return [value_translation.get(opt, opt) for opt in options]


def reverse_translate(value):
    """
    Traduce un valor seleccionado (ES → EN) para enviar al modelo.
    """
    return reverse_value_translation.get(value, value)
