#!/usr/bin/env python3
"""
Test script to verify model loading and basic functionality
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")
    
    model_files = [
        "digit_model_fixed.keras",
        "digit_model.keras", 
        "digit_model.h5"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                print(f"Attempting to load {model_file}...")
                model = tf.keras.models.load_model(model_file)
                print(f"✅ Successfully loaded {model_file}")
                
                # Test with a dummy input
                dummy_input = np.random.random((1, 28, 28, 1))
                prediction = model.predict(dummy_input, verbose=0)
                print(f"✅ Model prediction shape: {prediction.shape}")
                print(f"✅ Predicted digit: {np.argmax(prediction[0])}")
                return model
                
            except Exception as e:
                print(f"❌ Failed to load {model_file}: {e}")
    
    print("❌ No models could be loaded")
    return None

def test_preprocessing():
    """Test the image preprocessing function"""
    print("\nTesting image preprocessing...")
    
    # Create a dummy image (white background with some black pixels)
    dummy_img = np.ones((200, 200), dtype=np.uint8) * 255
    # Add some "digit" pixels
    dummy_img[50:150, 50:150] = 0
    
    # Convert to PIL Image
    pil_img = Image.fromarray(dummy_img)
    
    # Test preprocessing
    try:
        from app import preprocess_image
        processed = preprocess_image(pil_img)
        
        if processed is not None:
            print(f"✅ Preprocessing successful")
            print(f"✅ Output shape: {processed.shape}")
            print(f"✅ Value range: {processed.min():.3f} to {processed.max():.3f}")
        else:
            print("❌ Preprocessing returned None")
            
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")

if __name__ == "__main__":
    print("=== Digit Recognition Model Test ===\n")
    
    # Test model loading
    model = test_model_loading()
    
    # Test preprocessing
    test_preprocessing()
    
    print("\n=== Test Complete ===") 