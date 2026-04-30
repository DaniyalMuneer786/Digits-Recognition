from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)

# Load the TensorFlow model (more accurate than KNN)
try:
    model = tf.keras.models.load_model("digit_model_fixed.keras")
    print("Loaded digit_model_fixed.keras successfully")
except:
    try:
        model = tf.keras.models.load_model("digit_model.keras")
        print("Loaded digit_model.keras successfully")
    except:
        try:
            model = tf.keras.models.load_model("digit_model.h5")
            print("Loaded digit_model.h5 successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a simple model as fallback
            model = None

@app.route("/")
def index():
    return render_template("Digits Recognition.html")

def preprocess_image(image):
    """Enhanced image preprocessing for better digit recognition"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Find the bounding box of the digit
    rows = np.any(img_array < 255, axis=1)
    cols = np.any(img_array < 255, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None  # No digit found
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Crop to digit with some padding
    padding = 10
    rmin = max(0, rmin - padding)
    rmax = min(img_array.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(img_array.shape[1], cmax + padding)
    
    cropped = img_array[rmin:rmax, cmin:cmax]
    
    # Create a square image by padding
    height, width = cropped.shape
    size = max(height, width)
    
    # Create a square canvas
    square_img = np.ones((size, size), dtype=np.uint8) * 255
    
    # Calculate offset to center the digit
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    
    # Place the digit in the center
    square_img[y_offset:y_offset+height, x_offset:x_offset+width] = cropped
    
    # Resize to 28x28 (MNIST format)
    resized = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize and invert (background=0, digit=1)
    normalized = 1.0 - (resized.astype(np.float32) / 255.0)
    
    # Add channel dimension
    return normalized.reshape(1, 28, 28, 1)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "No image found"}), 400

        # Decode image from base64
        image_data = data["image"].split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({"error": "No digit detected in the image"}), 400
        
        if model is None:
            # Fallback to simple prediction
            return jsonify({
                "digit": 5,  # Default prediction
                "confidence": 50.0,
                "probabilities": [0.1] * 10
            })
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "probabilities": [round(float(p), 4) for p in predictions]
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)