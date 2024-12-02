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
st.title("🩺 Diabetic Retinopathy Detection")
st.markdown("""
Welcome to the **Diabetic Retinopathy Detection App**.  
Upload a retinal image to detect the level of diabetic retinopathy using a pre-trained deep learning model.
""")

# Sidebar information
st.sidebar.header("About Diabetic Retinopathy")
st.sidebar.write("""
Diabetic retinopathy is a complication of diabetes that affects the eyes. It occurs when high blood sugar levels damage the blood vessels in the retina. The detection categories include:
- **No DR**: No diabetic retinopathy detected.
- **Mild**: Early signs of retinopathy, such as small retinal changes.
- **Moderate**: Increased retinal damage.
- **Severe**: Severe retinal damage that could lead to vision loss.
- **Proliferative DR**: Advanced stage with new blood vessel growth.

Early detection can help prevent vision loss. This app provides AI-based predictions to assist in diagnosis.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing the image...")

    # Preprocess the image
    processed_image = preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    # Predict
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])  # Get class with the highest probability

    # Display results
    st.markdown("## 🏆 Prediction Results")
    st.write(f"### **Predicted Category:** {CLASSES[predicted_class]}")

    # Add detailed description
    if predicted_class == 0:
        st.success("Great news! The model detected **No Diabetic Retinopathy** in the uploaded image.")
    elif predicted_class == 1:
        st.warning("**Mild Diabetic Retinopathy** detected. Consider scheduling an eye check-up for further evaluation.")
    elif predicted_class == 2:
        st.warning("**Moderate Diabetic Retinopathy** detected. It is recommended to consult an ophthalmologist.")
    elif predicted_class == 3:
        st.error("**Severe Diabetic Retinopathy** detected. Immediate medical attention is advised.")
    elif predicted_class == 4:
        st.error("**Proliferative Diabetic Retinopathy** detected. This is a serious condition requiring urgent treatment.")

# Add footer
st.markdown("---")
st.markdown("_Note: This app is for educational purposes only and should not replace professional medical advice._")
