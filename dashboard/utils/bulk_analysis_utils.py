import pandas as pd


import pandas as pd


def get_example_dataframe() -> pd.DataFrame:
    data = [
        ["Female", 0, "Yes", "No", 5, "Yes", "No", "Fiber optic", "No", "No", "No", "No",
         "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 89.65, 400.5, "0-6", 3],
        ["Male", 1, "No", "Yes", 40, "Yes", "Yes", "Fiber optic", "Yes", "No", "Yes", "No",
         "Yes", "No", "Two year", "No", "Bank transfer (automatic)", 104.8, 4200.3, "24-48", 4],
        ["Female", 0, "Yes", "Yes", 60, "No", "No phone service", "DSL", "No", "No", "No", "No",
         "No", "No", "One year", "Yes", "Mailed check", 40.5, 2400.0, "48-72", 1],
        ["Male", 1, "No", "No", 3, "Yes", "No", "No", "No internet service", "No internet service",
         "No internet service", "No internet service", "No internet service", "No internet service",
         "Month-to-month", "Yes", "Electronic check", 20.0, 60.0, "0-6", 0],
        ["Female", 0, "Yes", "Yes", 48, "Yes", "Yes", "Fiber optic", "Yes", "Yes", "Yes", "Yes",
         "Yes", "Yes", "Two year", "No", "Credit card (automatic)", 120.5, 5800.7, "24-48", 5],
        ["Male", 0, "No", "No", 10, "Yes", "No", "DSL", "No", "Yes", "No", "No",
         "No", "No", "Month-to-month", "Yes", "Electronic check", 60.0, 600.0, "6-12", 2],
        ["Female", 1, "Yes", "Yes", 65, "Yes", "Yes", "Fiber optic", "Yes", "Yes", "No", "Yes",
         "Yes", "No", "One year", "No", "Bank transfer (automatic)", 110.0, 7100.3, "48-72", 3],
        ["Male", 0, "No", "No", 30, "Yes", "Yes", "DSL", "No", "Yes", "Yes", "No",
         "Yes", "No", "One year", "Yes", "Mailed check", 85.0, 2550.0, "24-48", 4],
        ["Female", 0, "No", "No", 15, "Yes", "No", "DSL", "No", "No", "No", "No",
         "No", "No", "Month-to-month", "Yes", "Electronic check", 45.0, 675.0, "12-24", 1],
        ["Male", 1, "Yes", "No", 55, "Yes", "Yes", "Fiber optic", "Yes", "Yes", "Yes", "Yes",
         "Yes", "Yes", "Two year", "No", "Credit card (automatic)", 115.0, 6325.0, "48-72", 6],
        ["Female", 0, "Yes", "No", 22, "No", "No phone service", "No", "No internet service", "No internet service",
         "No internet service", "No internet service", "No internet service", "No internet service",
         "Month-to-month", "Yes", "Electronic check", 25.0, 550.0, "12-24", 1],
        ["Male", 0, "No", "No", 8, "Yes", "No", "Fiber optic", "No", "No", "No", "No",
         "No", "No", "Month-to-month", "Yes", "Mailed check", 65.0, 520.0, "6-12", 1],
        ["Female", 1, "No", "No", 38, "Yes", "Yes", "DSL", "Yes", "Yes", "No", "No",
         "No", "No", "One year", "No", "Bank transfer (automatic)", 70.0, 2660.0, "24-48", 3],
        ["Male", 0, "Yes", "Yes", 50, "Yes", "Yes", "Fiber optic", "Yes", "No", "Yes", "Yes",
         "Yes", "Yes", "Two year", "No", "Credit card (automatic)", 110.5, 5525.0, "48-72", 5],
        ["Female", 0, "No", "No", 4, "No", "No phone service", "No", "No internet service", "No internet service",
         "No internet service", "No internet service", "No internet service", "No internet service",
         "Month-to-month", "Yes", "Electronic check", 18.0, 72.0, "0-6", 0],
        ["Male", 1, "Yes", "No", 58, "Yes", "Yes", "Fiber optic", "Yes", "Yes", "Yes", "Yes",
         "Yes", "Yes", "Two year", "No", "Bank transfer (automatic)", 120.0, 6960.0, "48-72", 6],
        ["Female", 0, "No", "No", 12, "Yes", "No", "DSL", "No", "No", "No", "No",
         "No", "No", "Month-to-month", "Yes", "Mailed check", 55.0, 660.0, "6-12", 1],
        ["Male", 0, "No", "No", 20, "Yes", "Yes", "Fiber optic", "No", "No", "No", "No",
         "No", "No", "One year", "No", "Electronic check", 75.0, 1500.0, "12-24", 2],
        ["Female", 1, "Yes", "Yes", 45, "Yes", "Yes", "Fiber optic", "Yes", "Yes", "Yes", "Yes",
         "Yes", "No", "Two year", "No", "Credit card (automatic)", 112.0, 5040.0, "24-48", 5],
        ["Male", 0, "No", "No", 28, "No", "No phone service", "DSL", "No", "No", "No", "No",
         "No", "No", "One year", "Yes", "Mailed check", 40.0, 1120.0, "24-48", 1]
    ]

    columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
        "TotalCharges", "tenure_group", "MultipleServices"
    ]

    return pd.DataFrame(data, columns=columns)


def validate_uploaded_dataframe(df: pd.DataFrame, required_cols: list) -> list:
    return [col for col in required_cols if col not in df.columns]


def classify_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "BAJO"
    elif prob < 0.6:
        return "MEDIO"
    else:
        return "ALTO"


def add_predictions_and_risk_levels(df_raw: pd.DataFrame, preds: list, probs: list, classify_func) -> pd.DataFrame:
    prob_churn = [p[1] for p in probs]
    df_results = df_raw.copy()
    df_results["Probabilidad_Churn"] = prob_churn
    df_results["Prediccion"] = preds
    df_results["Nivel_Riesgo"] = df_results["Probabilidad_Churn"].apply(
        classify_func)
    return df_results
