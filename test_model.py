import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

def test():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'model_paprika_final.h5')
    image_path = os.path.join(os.path.dirname(BASE_DIR), 'cnn_model', 'data', 'Pepper_Dataset', 'Healthy', '00100ffa-095e-4881-aebf-61fe5af7226e___JR_HL 7886.JPG')
    
    print(f"Model path: {model_path}")
    print(f"Image path: {image_path}")
    
    if not os.path.exists(model_path):
        print("Model file not found!")
        return
        
    if not os.path.exists(image_path):
        print("Sample image file not found!")
        return

    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully.")
    
    print("Loading and preprocessing image...")
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print("Running prediction...")
    predictions = model.predict(img_array)
    print(f"Raw predictions: {predictions}")
    
    class_names = ['Bacterial_spot', 'Cercospora Leaf Spot', 'Healthy']
    pred_probs = predictions[0]
    pred_idx = np.argmax(pred_probs)
    
    print(f"Predicted class index: {pred_idx}")
    print(f"Predicted class: {class_names[pred_idx]}")
    print(f"Confidence score: {pred_probs[pred_idx]:.4f}")

if __name__ == '__main__':
    test()
