import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests
from PIL import Image

# Title and description
st.title("Microaneurysm Detection")
st.write("Upload an image to detect the severity of diabetic retinopathy.")

# Load the model from GitHub
@st.cache_resource
def load_model_from_github():
    url = "https://github.com/your-username/your-repo-name/raw/main/model.h5"  # Replace with your actual GitHub file URL
    response = requests.get(url, stream=True)
    with open("model.h5", "wb") as f:
        f.write(response.content)
    model = load_model("model.h5")
    return model

model = load_model_from_github()

# Define severity labels (customize as per your model)
severity_labels = ["Normal", "Mild", "Moderate", "Severe", "Proliferated"]

# Image upload
uploaded_file = st.file_uploader("Upload an image (JPG/PNG format)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Resize as per your model input size
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    st.write("Predicting...")
    predictions = model.predict(img_array)
    severity_index = np.argmax(predictions, axis=1)[0]
    severity = severity_labels[severity_index]

    # Display result
    st.write(f"Prediction: **{severity}**")
    st.bar_chart(predictions[0])  # Display prediction probabilities as a bar chart
