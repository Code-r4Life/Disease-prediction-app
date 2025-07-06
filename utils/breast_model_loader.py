import pickle
import numpy as np

def load_breast_model(model_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def preprocess_breast_input(form_data):
    input_values = [float(value) for value in form_data]
    input_array = np.array(input_values).reshape(1, -1)
    return input_array

def predict_breast_cancer(model, scaler, input_array):
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]
    return 'Malignant' if prediction == 1 else 'Benign'
