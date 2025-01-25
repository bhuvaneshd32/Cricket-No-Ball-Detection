import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model_path = r'final_cnn.keras'
loaded_model = tf.keras.models.load_model(model_path)

def predict_no_ball(image):
    image = image.resize((64, 64))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    result = loaded_model.predict(image)
    return 'No Ball' if result[0][0] >= 0.5 else 'Legal Ball'

st.title("No Ball Detection System")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_no_ball(image)
    st.write(f"Prediction: {prediction}")
