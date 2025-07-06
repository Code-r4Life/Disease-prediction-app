from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

def load_model_tf(path):
    return load_model(path)

def preprocess_brain_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = image.reshape(1, 64, 64, 3)
    return image

def predict_brain_tumor(model, image):
    processed_image = preprocess_brain_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    class_names = ['No Tumor', 'Tumor']
    return class_names[class_index]