import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Load the dataset from CSV file
data_path = "C:/Users/funnn/Desktop/AI SEM 6 project/spam-detector/SpamTrain.csv"
df = pd.read_csv(data_path, encoding='latin1')

# Ensure that your dataset contains the columns 'v2' for messages and 'v1' for labels
print(df.head())  # Inspect the first few rows of your data

# 2. Preprocess the data
# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['v2'])  # Use 'v2' for messages

X = tokenizer.texts_to_sequences(df['v2'])  # Convert text to sequences
X = pad_sequences(X, padding='post', maxlen=50)  # Pad sequences to ensure uniform input length

y = np.array(df['v1'])  # Labels are in 'v1' (1 = Spam, 0 = Not Spam)

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define the neural network model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=50),  # Embedding layer for word vectors
    LSTM(64, return_sequences=True),  # LSTM layer for learning sequence patterns
    GRU(32),  # GRU layer for sequence modeling
    Dropout(0.5),  # Dropout for regularization
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification (Spam/Not Spam)
])

# 5. Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

# 7. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 8. Save the model
model.save('spam_detection_model.h5')

# 9. Use the model for prediction
def predict_spam(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=50)
    predictions = model.predict(padded_sequences)
    labels = ['Spam' if pred > 0.5 else 'Not Spam' for pred in predictions]
    return labels

# Example usage of the prediction function
texts = ["Congratulations, you've won a free iPhone!", "Can we reschedule our meeting?"]
predictions = predict_spam(texts)
print(predictions)
