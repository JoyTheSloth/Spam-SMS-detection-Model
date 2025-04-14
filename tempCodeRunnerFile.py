from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load the pre-trained model
model = load_model('spam_detection_model.h5')

# Load the tokenizer used during training
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(["Placeholder text"])  # Replace this with your tokenizer data (you can re-fit or load it as necessary)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from the request
    message = data.get('message')  # The message to classify

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Preprocess the message
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=50)

    # Make prediction
    prediction = model.predict(padded_sequence)
    label = 'Spam' if prediction > 0.5 else 'Not Spam'

    return jsonify({"message": message, "prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
