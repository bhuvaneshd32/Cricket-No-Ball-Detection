import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image

model_path = 'C:/Users/91822/Desktop/college/capstone/final_model1.keras'
loaded_model = tf.keras.models.load_model(model_path)

def predict_no_ball(image):
    image = image.resize((64, 64))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    result = loaded_model.predict(image)
    return 'No Ball' if result[0][0] == 1 else 'Legal Ball'

st.title("No Ball Detection System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi","mov"])

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.video(video_path)

    if st.button("Take Screenshot"):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            st.image(image, caption='Screenshot for Prediction', use_column_width=True)
            prediction = predict_no_ball(image)
            st.write(f"Prediction: {prediction}")
        cap.release()
