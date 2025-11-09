import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'serangga_cnn_model_v1.h5'
IMAGE_SIZE = (150, 150)

CLASS_NAMES = ['aphids', 'armyworm', 'bettle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']
NUM_CLASSES = len(CLASS_NAMES)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model Deep Learning berhasil dimuat")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None
    
def preprocesses_image(img):
    """Fungsi untuk preprocessing gambar sebelum diumpankan ke model"""
    img = img.resize(IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    """Melayani file index.html dari folder templates"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk menerima upload gambar dan melakukan prediksi"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error' : 'No selected file'}), 400
    
    if file:
        try:
            img = Image.open(file.stream)
            
            processed_img = preprocesses_image(img)
            predictions = model.predict(processed_img)
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[0][predicted_index])
            
            response = {
                'status': 'success',
                'prediction': predicted_class,
                'confidence' : f"{confidence*100:.2f}"
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': f'Predicted failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)