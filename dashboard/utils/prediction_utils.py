import pandas as pd


def build_input_dataframe(inputs: dict, reverse_translate_func) -> pd.DataFrame:
    data = {}
    for key, val in inputs.items():
        if isinstance(val, str):
            data[key] = reverse_translate_func(val)
        else:
            data[key] = val
    return pd.DataFrame([data])
