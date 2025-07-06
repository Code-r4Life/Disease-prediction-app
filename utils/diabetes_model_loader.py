import joblib
import pandas as pd

def load_diabetes_model(model_path):
    model = joblib.load(model_path)
    return model

def preprocess_diabetes_input(form_data, expected_columns):
    input_dict = {}
    for col in expected_columns:
        val = form_data.get(col)

        # Convert to float only if it's a known numeric field
        if col in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_', 'hypertension', 'heart_disease']:
            try:
                input_dict[col] = float(val)
            except ValueError:
                input_dict[col] = 0.0  # or handle as needed
        else:
            input_dict[col] = val  # keep as string (e.g., 'male', 'never')
    return input_dict


def predict_diabetes(model, input_dict):
    input_df=pd.DataFrame([input_dict])
    input_df[model['numeric_cols']]=model['imputer'].transform(input_df[model['numeric_cols']])
    input_df[model['numeric_cols']]=model['scaler'].transform(input_df[model['numeric_cols']])
    input_df[model['encoded_cols']]=model['encoder'].transform(input_df[model['categorical_cols']])
    x_input=input_df[model['numeric_cols']+model['encoded_cols']]
    pred=model['model'].predict(x_input)[0]
    return "Diabetes" if int(pred) == 1 else "Normal"
