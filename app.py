import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF warnings

@st.cache_resource
def load_my_model():
    return load_model("skin_cancer_cnn.h5")

model = load_my_model()

st.title("Skin Cancer Detection")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("Predicting..."):
            prediction = model.predict(img_array)
        label = "Cancer" if prediction[0][0] > 0.5 else "No Cancer"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        st.success(f"Prediction: {label} ({confidence:.2f})")
