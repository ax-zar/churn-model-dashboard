def get_dashboard_metrics():
    return {
        'f1_score': 0.627,
        'roc_auc': 0.847,
        'precision': 0.644,
        'recall': 0.495,
        'specificity': 0.75
    }


def get_risk_factors():
    return {
        'Contrato mensual': 0.85,
        'Pago electrónico': 0.72,
        'Sin servicios extra': 0.68,
        'Cliente nuevo': 0.65,
        'Cargos altos': 0.58
    }
