import tensorflow as tf
import numpy as np
from PIL import Image

# put the model file here  
model_path = "Model/skin_disease_classifier.keras"
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ["Acne", "Actinic_Keratosis", "Benign_tumors", "Eczema", "Lupus", "Monkeypox", "Normal", "Psoriasis", "Sarampion", "SkinCancer", "Tinea", "Varicela", "Vitiligo"]

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224)) 
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def predict(image_path):
    """Predict disease from an image"""
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class
