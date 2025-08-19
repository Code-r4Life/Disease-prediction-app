import joblib
import pandas as pd

def load_heart_model(model_path):
    with open(model_path, 'rb') as f:
        bundle = joblib.load(f)
    return bundle

def preprocess_heart_input(form_data, expected_columns):
    input_dict = {}
    for col in expected_columns:
        val = form_data.get(col)
        input_dict[col] = float(val)
    return input_dict

def predict_heart(bundle, input_dict):
    input_df = pd.DataFrame([input_dict])
    model = bundle['model']
    scaler = bundle['scaler']
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    return "Heart Disease" if int(pred) == 1 else "Normal"
