import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
MODEL_PATH = "model.h5"  # Update this to your saved model's path
model = load_model(MODEL_PATH)

# Constants
IMAGE_SIZE = 224
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Define a function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an uploaded image for model prediction.

    Args:
        image (numpy.ndarray): The uploaded image in BGR format.
        target_size (tuple): The target size for the image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, target_size)  # Resize to target size
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Streamlit app
st.title("Diabetic Retinopathy Detection")
st.markdown("Upload a retinal image to detect the level of diabetic retinopathy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    # Predict
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])  # Get class with the highest probability
    confidence_scores = {CLASSES[i]: f"{prediction[0][i]:.2f}" for i in range(len(CLASSES))}

    # Display results
    st.write("### Prediction Results")
    st.write(f"**Predicted Class:** {CLASSES[predicted_class]}")
    st.write("**Confidence Scores:**")
    st.json(confidence_scores)

# Add footer
st.markdown("---")
st.markdown("**Developed with ❤️ by Your Name**")
