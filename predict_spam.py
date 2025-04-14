from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS  # Import for CORS support

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the tokenizer
tokenizer_path = 'tokenizer.pickle'
try:
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully.")
except FileNotFoundError:
    print(f"Tokenizer file not found at: {tokenizer_path}")
    tokenizer = None

# Load the trained model
model_path = 'spam_detection_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data['message']

        # Check if model and tokenizer are loaded
        if model is None or tokenizer is None:
            return jsonify({"error": "Model or tokenizer not loaded"}), 500

        sequences = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(sequences, maxlen=50, padding='post')
        prediction = model.predict(padded)
        
        # Get the prediction score
        score = float(prediction[0][0])
        label = "Spam" if score > 0.5 else "Not Spam"
        
        # Return the prediction with confidence score
        return jsonify({
            "prediction": label,
            "confidence": score if label == "Spam" else 1 - score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)