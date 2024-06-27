import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('crop_health_model.h5')

# Define class names
class_names = ['Healthy', 'Diseased']  # Update this list based on your actual class names

st.title("Crop Health Prediction")

st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 2em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Upload an Image of a Crop Leaf</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]

    st.write(f"Health Status of the crop: {predicted_class}")
