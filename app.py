import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt

# Load the pre-trained model
MODEL_PATH = "model.h5"  # Update this to your saved model's path
model = load_model(MODEL_PATH)

# Constants
IMAGE_SIZE = 224
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Define a function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, target_size)  # Resize to target size
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Function to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
dr_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_pQxTwc.json")
success_animation = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_j1adxtyb.json")
warning_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_km2eyzfg.json")
error_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_cjyektyf.json")

# Streamlit app
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="🩺",
    layout="wide"
)

# Header
st.title("🩺 Diabetic Retinopathy Detection")
st_lottie(dr_animation, height=200, key="dr_anim")

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

# File uploader
uploaded_file = st.file_uploader("📂 Upload a retinal image (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image.shape[0] < 300 or image.shape[1] < 300:  # Basic dimension check
        st.error("⚠️ Please upload a valid retinal image.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image, caption="📷 Uploaded Retinal Image", use_column_width=True)
            st.write("🔍 Analyzing the image...")

        # Preprocess the image
        processed_image = preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))

        # Predict
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction[0])

        # Display results
        with col2:
            st.markdown("## 🏆 Prediction Results")
            st.write(f"### **Predicted Category:** {CLASSES[predicted_class]}")

            # Add detailed description with animations
            if predicted_class == 0:
                st.success("Great news! The model detected **No Diabetic Retinopathy**.")
                st_lottie(success_animation, height=150, key="success")
            elif predicted_class == 1:
                st.warning("**Mild Diabetic Retinopathy** detected. Consider an eye check-up.")
                st_lottie(warning_animation, height=150, key="mild_warning")
            elif predicted_class == 2:
                st.warning("**Moderate Diabetic Retinopathy** detected. Consult an ophthalmologist.")
                st_lottie(warning_animation, height=150, key="moderate_warning")
            elif predicted_class == 3:
                st.error("**Severe Diabetic Retinopathy** detected. Immediate medical attention is advised.")
                st_lottie(error_animation, height=150, key="severe_error")
            elif predicted_class == 4:
                st.error("**Proliferative Diabetic Retinopathy** detected. Urgent treatment required.")
                st_lottie(error_animation, height=150, key="proliferative_error")

        st.markdown("---")
        st.markdown("### 🧪 Future Work")
        st.write("Retina segmentation, blood vessel mapping, and lesion detection can provide more detailed diagnostic insights.")

# Footer
st.markdown("---")
st.markdown("_This app is for educational purposes only. For medical advice, please consult a professional._")
