# 🔢 Handwritten Digit Recognition – CNN Based Web Application

A Flask-based web application that recognizes handwritten digits (**0–9**) drawn by the user in a browser using a Convolutional Neural Network (CNN) model trained on the MNIST dataset.

---

## 🚀 Features

- **Handwritten Digit Prediction**: Recognizes digits drawn by the user
- **Deep Learning Model**: CNN model trained on the MNIST dataset
- **Interactive Canvas Interface**: Users draw digits directly in the browser
- **Real-time Predictions**: Instant digit recognition through the web interface
- **Probability Distribution**: Displays prediction confidence and probability for all digits

---

## 📋 Requirements

- Python 3.7 or higher
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- HTML5 / CSS3 / JavaScript

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd "Handwritten Digit Recognition"
```

### 2. Install dependencies

Option A: Using pip
```bash
pip install -r requirements.txt
```
Option B: Using conda
```bash
conda install --file requirements.txt
```

### 3. Ensure model files are present
Make sure you have these files in the project root:

digit_model.h5 – Trained CNN model
digit_model_fixed.h5 – Improved trained model

Note: If these files are missing, train the model first using the Digit_Recognition_Training.ipynb notebook.

---

🎯 Usage

Running the Application
Start the Flask server
```bash
python app.py
```
Open your browser and navigate to:
```bash
http://localhost:5000
```
Draw a handwritten digit on the canvas.

Click “Predict Digit” to get the prediction.

The system will display:

Predicted digit
Confidence score
Probability distribution for digits (0–9)

---

🔌 API Endpoint

Predict Digit
Endpoint: POST /predict
Request Body
```bash
{
    "image": "base64_encoded_canvas_image"
}
```

---

Response
```bash
{
    "prediction": 7,
    "confidence": 0.985,
    "success": true
}
```
Health Check: GET /health

---


📁 Project Structure
```
Handwritten Digit Recognition/
│
├── app.py                              # Flask backend
├── requirements.txt                    # Python dependencies
├── digit_model.h5                      # Trained CNN model
├── digit_model_fixed.h5                # Updated trained model
├── templates/
│   └── index.html                      # Full Frontend UI with canvas
├── Dataset/
│   └── MNIST_Dataset                   # Handwritten digits dataset
├── Digit_Recognition_Training.ipynb    # Model training notebook
└── README.md                           # Documentation
```
---

🤖 Model Information

Algorithm: Convolutional Neural Network (CNN)
Dataset: MNIST
Images: 28×28 grayscale handwritten digits

Classes:

- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9

Model Architecture

Conv2D (32 filters, 3×3, ReLU)
MaxPooling2D (2×2)
Flatten Layer
Dense Layer (128 neurons, ReLU)
Output Layer (10 neurons, Softmax)

Optimizer

Adam

Loss Function

Categorical Crossentropy

---

🧪 Testing

To test the application:

Run the server:
```bash
python app.py
```

Open:
```bash
http://localhost:5000
```
Draw a digit on the canvas.

Click Predict Digit to see the prediction.

---

🐛 Troubleshooting

Model not loading
Ensure the model file exists in the root directory
Check TensorFlow and Python versions

Port already in use
```bash
app.run(port=5001)
```
Dependency issues
```bash
python -m pip install --upgrade pip
pip install flask tensorflow numpy pillow
```
📊 Model Insights

The model predicts digits based on handwritten patterns learned from the MNIST dataset.

It analyzes:

Pixel intensity patterns
Stroke shapes
Digit structure
Spatial relationships in the image

---

📝 Notes

The system expects digits drawn clearly in the canvas
Images are resized to 28×28 grayscale before prediction
Blurry or unclear drawings may reduce accuracy
Designed for educational and demonstration purposes

---

📄 License

This project is for educational purposes only.

---

👨‍💻 Author

Daniyal Muneer
Data Scientist | Machine Learning & AI Applications

---

🙏 Acknowledgments

- MNIST dataset for handwritten digit images
- TensorFlow / Keras for deep learning framework
- Flask for backend web development
- HTML5 Canvas API for interactive drawing interface

---
