import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

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
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetic Retinopathy Detection")
st.markdown("""
Welcome to the **Diabetic Retinopathy Detection App**!  
Upload a **retinal image** to detect the level of diabetic retinopathy using an advanced AI model.
""")

# Sidebar information
st.sidebar.header("About Diabetic Retinopathy")
st.sidebar.write("""
Diabetic retinopathy is a complication of diabetes that affects the retina due to high blood sugar levels. The detection categories include:
- **No DR**: No diabetic retinopathy detected.
- **Mild**: Early retinal changes.
- **Moderate**: Visible retinal damage.
- **Severe**: Severe retinal damage, possible vision loss.
- **Proliferative DR**: Advanced stage with new blood vessel growth.

**Note**: Early detection is key to preventing vision loss.
""")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/7/73/Diabetic_retinopathy%2C_fundus_photograph.jpg", caption="Example of Retinal Image", use_column_width=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload a retinal image (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Validate image (check if it looks like a retina image)
    if image.shape[0] < 300 or image.shape[1] < 300:  # Basic dimension check
        st.error("⚠️ Please upload a valid retinal image.")
    else:
        col1, col2 = st.columns([2, 1])

        # Display uploaded image
        with col1:
            st.image(image, caption="📷 Uploaded Retinal Image", use_column_width=True)
            st.write("🔍 Analyzing the image...")

        # Preprocess the image
        processed_image = preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))

        # Predict
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])  # Get class with the highest probability

        # Display results
        with col2:
            st.markdown("## 🏆 Prediction Results")
            st.write(f"### **Predicted Category:** {CLASSES[predicted_class]}")

            # Add detailed description
            if predicted_class == 0:
                st.success("Great news! The model detected **No Diabetic Retinopathy**.")
            elif predicted_class == 1:
                st.warning("**Mild Diabetic Retinopathy** detected. Consider an eye check-up.")
            elif predicted_class == 2:
                st.warning("**Moderate Diabetic Retinopathy** detected. Consult an ophthalmologist.")
            elif predicted_class == 3:
                st.error("**Severe Diabetic Retinopathy** detected. Immediate medical attention is advised.")
            elif predicted_class == 4:
                st.error("**Proliferative Diabetic Retinopathy** detected. Urgent treatment required.")

  
        # Additional Retina Image Segmentation/Visualization (if available)
        st.markdown("---")
        st.markdown("### 🧪 Future Work")
        st.write("Retina segmentation, blood vessel mapping, and lesion detection can provide more detailed diagnostic insights.")

# Add footer
st.markdown("---")
st.markdown("_This app is for educational purposes only. For medical advice, please consult a professional._")
