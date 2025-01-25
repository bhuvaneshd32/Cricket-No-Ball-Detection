import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model_path = r'final_cnn.keras'
loaded_model = tf.keras.models.load_model(model_path)

# Prediction function
def predict_no_ball(image):
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    result = loaded_model.predict(image)
    return 'No Ball' if result[0][0] >= 0.5 else 'Legal Ball'

# Directory containing test images
image_dir = r'Final-test'

# Get list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]

# Streamlit app configuration
st.set_page_config(page_title="No Ball Detection System", layout="centered", initial_sidebar_state="expanded")
st.title("🏏 No Ball Detection System")
st.write("Upload an image or select one from the test set to detect if it's a **No Ball** or a **Legal Ball**!")

# Tabs for image selection
tab1, tab2 = st.tabs(["📂 Select from Test Set", "📤 Upload Your Own Image"])

# Variables for image and prediction
selected_image = None
prediction = None

# Tab 1: Select image from test set
with tab1:
    st.subheader("Select an Image from the Test Set")
    selected_image_name = st.selectbox("Select an image for prediction", image_files)
    if selected_image_name:
        image_path = os.path.join(image_dir, selected_image_name)
        selected_image = Image.open(image_path)
        st.image(selected_image, caption=f"Selected Image: {selected_image_name}", use_column_width=True)

# Tab 2: Upload your own image
with tab2:
    st.subheader("Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpeg", "jpg", "png"])
    if uploaded_file:
        selected_image = Image.open(uploaded_file)
        st.image(selected_image, caption="Uploaded Image", use_column_width=True)

# Test and Predict button
if selected_image:
    st.markdown("### 🧪 Prediction Results")
    test_button = st.button("Test the Image")
    if test_button:
        with st.spinner("Analyzing the image..."):
            prediction = predict_no_ball(selected_image)

        # Display the prediction in a clean and professional way
        st.write("---")  # Separator line
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #f4f4f4; border-radius: 10px;">
            <h2 style="color: {'green' if prediction == 'Legal Ball' else 'red'};">Prediction: {prediction}</h2>
        </div>
        """, unsafe_allow_html=True)

# Add some styling and credits in the sidebar
st.sidebar.markdown(
    """
    ## About
    This system uses a trained Convolutional Neural Network (CNN) to detect **No Balls** in cricket images.
    - Upload your own image or select one from the test set.

    """
)
