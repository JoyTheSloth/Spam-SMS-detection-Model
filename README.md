## 🧠 Spam Detector - AI-Powered Message Classification

Detect spam messages with a simple and elegant web interface powered by machine learning.

### 🚀 Features
- 🔍 Real-time spam detection using a trained LSTM + GRU neural network
- 🧠 Built with TensorFlow and Keras
- 🌐 Interactive frontend with Tailwind CSS
- 🗂 Flask API for backend prediction
- 🔁 Live result display with modern UI

---

### 📁 Project Structure

```
/spam-detector
├── app.py                   # (Optional) Model training script
├── predict_spam.py         # Flask API for serving predictions
├── tokenizer.pickle        # Saved tokenizer used during training
├── spam_detection_model.h5 # Trained spam detection model
├── SpamTrain.csv           # Training data
├── SpamTest.csv            # Test data
├── templates/
│   ├── index.html          # Frontend input page
│   └── result.html         # Result display page
├── static/                 # Optional styles/scripts
```

---

### ⚙️ How to Run

1. Install dependencies
   ```bash
   pip install flask tensorflow flask-cors
   ```

2. Start the Flask server
   ```bash
   python predict_spam.py
   ```

3. Launch the frontend
   ```bash
   python -m http.server 8000
   ```

4. Visit in browser
   ```
   http://localhost:8000/index.html
   ```

---

### 💡 How It Works

- User types a message into the frontend
- The input is sent to the Flask API via a POST request
- The server tokenizes and classifies the message as "Spam" or "Not Spam"
- The result is shown on a separate result page

---

### 🛠 Tech Stack

- Frontend: HTML, Tailwind CSS, JavaScript
- Backend: Python, Flask, TensorFlow
- Model: LSTM + GRU hybrid for spam classification
- Deployment Ready: Can be served locally or hosted via Flask + frontend hosting (Netlify, Vercel, etc.)

---
