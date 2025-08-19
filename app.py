from flask import Flask, render_template, request, jsonify
from utils.tumor_model_loader import load_model_tf, predict_brain_tumor
from utils.breast_model_loader import load_breast_model, preprocess_breast_input, predict_breast_cancer
from utils.pneumonia_model_loader import load_model2_tf, predict_pneumonia
from utils.diabetes_model_loader import load_diabetes_model, preprocess_diabetes_input, predict_diabetes
from utils.heart_model_loader import load_heart_model, preprocess_heart_input, predict_heart
from dotenv import load_dotenv
from flask_cors import CORS
import os
import time
import google.generativeai as genai

# === Setup Flask ===
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# === Load .env ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === Load ML Models ===
brain_model = load_model_tf("models/BrainTumor10epochs_categorical.keras")
breast_model, breast_scaler = load_breast_model("models/BreastCancer_model.pkl", "models/BreastCancer_scaler.pkl")
pneumonia_model = load_model2_tf("models/pneumonia10epochs_categorical.keras")
diabetes_model = load_diabetes_model("models/diabetes_model.joblib")
heart_model = load_heart_model("models/stacking_model.pkl")

static_path = 'static/uploads/Brain Tumor'
static_path2 = 'static/uploads/Pneumonia'

# === Gemini Configuration ===
MODEL_NAME = "gemini-1.5-flash"
SESSION_TIMEOUT = 10 * 60  # 10 minutes
generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
system_instruction = """
🧠 Your name is Sam.

🧠 Act as a smart and reliable virtual medical assistant for a disease prediction website.

🎯 Your role is to:
- Understand user queries about medical symptoms.
- Predict possible common illnesses (like cold, flu, pneumonia, TB, etc.) based on described symptoms.
- Explain medical terms (e.g., "What is tuberculosis?") in clear, simple language.
- Respond with empathy and clarity, as a friendly and professional health advisor — not as a doctor.
- Never give final diagnoses. Always suggest that the user consults a real doctor for confirmation.

🚫 Don't:
- Provide prescriptions or medications.
- Act as a licensed healthcare provider.

✅ Do:
- Keep answers simple, educational, and friendly.
- Be concise but supportive.
- Mention "I'm just an AI assistant" if users ask for critical advice.

🗣️ Tone:
Friendly, calm, helpful, like a knowledgeable medical mentor.
"""

# === Start Chat Session ===
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=system_instruction,
)
chat_session = model.start_chat(history=[])
last_active = time.time()
history = []

# === Routes ===
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        input_array = preprocess_breast_input(list(request.form.values()))
        result = predict_breast_cancer(breast_model, breast_scaler, input_array)
        return render_template('breast_cancer.html', prediction=result)
    return render_template('breast_cancer.html')

@app.route('/brain_tumor', methods=['GET', 'POST'])
def brain_tumor():
    prediction = None
    image_url = None
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join(static_path, image.filename)
        image.save(image_path)
        image_url = '/static/uploads/Brain Tumor/' + image.filename
        prediction = predict_brain_tumor(brain_model, image_path)
    return render_template('brain_tumor.html', image_url=image_url, prediction=prediction)

@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    prediction = None
    image_url = None
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join(static_path2, image.filename)
        image.save(image_path)
        image_url = '/static/uploads/Pneumonia/' + image.filename
        prediction = predict_pneumonia(pneumonia_model, image_path)
    return render_template('pneumonia.html', image_url=image_url, prediction=prediction)

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        expected_columns = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_"]
        form_data = request.form
        input_dict = preprocess_diabetes_input(form_data, expected_columns)
        result = predict_diabetes(diabetes_model, input_dict)
        return render_template('diabetes.html', prediction=result)
    return render_template('diabetes.html')

@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        expected_columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        form_data = request.form
        input_dict = preprocess_heart_input(form_data, expected_columns)
        result = predict_heart(heart_model, input_dict)
        return render_template('heart_disease.html', prediction=result)
    return render_template('heart_disease.html')

@app.route("/predict", methods=["POST"])
def predict():
    global last_active, chat_session, history

    user_input = request.get_json().get("message", "").strip()
    if not user_input:
        return jsonify({"answer": "Please type something."})

    current_time = time.time()
    if current_time - last_active > SESSION_TIMEOUT:
        chat_session = model.start_chat(history=[])
        history.clear()

    last_active = current_time

    try:
        response = chat_session.send_message(user_input)
        model_reply = response.text

        history.append({"role": "user", "parts": [user_input]})
        history.append({"role": "model", "parts": [model_reply]})

        return jsonify({"answer": model_reply})
    except Exception as e:
        if "429" in str(e):
            return jsonify({"answer": "You've hit the usage limit. Try again later."})
        return jsonify({"answer": f"Error: {str(e)}"})

# === Run Server ===
if __name__ == '__main__':
    app.run(debug=True)