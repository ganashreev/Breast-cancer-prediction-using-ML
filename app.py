import streamlit as st
from PIL import Image
import numpy as np
import time

# Import backend functions
from model import load_trained_model, preprocess_image, predict_idc_cancer, get_project_accuracy

# Page settings
st.set_page_config(page_title="Breast Cancer AI", layout="wide")

st.title("🧠 Breast Cancer Detection using VGG16")

# Load model
model = load_trained_model()
accuracy = get_project_accuracy()

# Sidebar
st.sidebar.title("📊 Project Info")
st.sidebar.write(f"Model: VGG16")
st.sidebar.write(f"Accuracy: {accuracy}%")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            time.sleep(1)

            processed = preprocess_image(image)
            prediction, confidence = predict_idc_cancer(model, processed)

        # Result
        if "Malignant" in prediction:
            st.error(f"⚠️ {prediction}")
        else:
            st.success(f"✅ {prediction}")

        st.write(f"Confidence: {confidence:.2f}%")