# 🧠 Disease Prediction Web Application

### 👨‍💻 Developed by: **Shinjan Saha**  
> A machine learning–powered Flask web app that predicts diseases based on user health inputs using clean UI and multiple disease models.

---

## 🚀 Overview

This web application allows users to predict the likelihood of **Breast Cancer**, **Brain Tumor**, **Pneumonia**, and **Diabetes** by submitting relevant medical parameters. The predictions are made using pre-trained ML models that are integrated into a fast and user-friendly interface built with Flask, HTML/CSS, and joblib.

---

## ⚙️ Key Features

- 🎯 **Disease Models**: Supports predictions for:
  - Breast Cancer (tabular data)
  - Brain Tumor (image-based using CNN)
  - Pneumonia (image-based using CNN)
  - Diabetes (tabular data)
  
- 💻 **User-Friendly UI**:  
  - Sliders, dropdowns, and radio buttons for smooth input  
  - Dynamic result rendering and validations  

- 📦 **Reusable Utility Code**:  
  - Modular `utils/` Python files for model loading and preprocessing  
  - Clean separation of logic and templates  

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask  
- **Frontend**: HTML5, CSS3, Vanilla JavaScript 
- **ML Models**: Scikit-learn, TensorFlow, Keras  
- **Environment**: Python 3.10+, joblib, pandas, numpy

---

## 🌐 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/disease-prediction-app.git
cd disease-prediction-app

# (Optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
The app will be available at http://localhost:5000/
```

---

## 📬 Interested in a Similar Project?

I build smart, ML-integrated applications and responsive web platforms. Let’s build something powerful together!

- 📧 shinjansaha00@gmail.com  
- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/shinjan-saha-1bb744319/)  

