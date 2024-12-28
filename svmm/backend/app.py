from flask import Flask, request, jsonify,send_from_directory
import numpy as np
import cv2
import pickle
from flask_cors import CORS
from skimage.feature import hog
import os
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})
base_dir = os.path.dirname(os.path.abspath(__file__))  # This will give the absolute path to the backend folder
model_path = os.path.join(base_dir, 'data1.pickle')

# Load the pre-trained model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        @app.route('/')
        def serve_index():
         return send_from_directory(app.static_folder, 'index.html')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

categories = ['Cat', 'Dog']

# Function to extract HOG features
def extract_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Unable to decode image. Ensure the uploaded file is valid.")

        resized_image = cv2.resize(image, (100, 100))
        features = extract_features(resized_image).reshape(1, -1)

        prediction = model.predict(features)
        return jsonify({'prediction': categories[prediction[0]]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
