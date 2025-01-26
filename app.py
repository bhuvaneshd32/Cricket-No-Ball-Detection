import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from PIL import Image

model_path = r'CNN Model\final_model.keras'
loaded_model = load_model(model_path)

def predict_no_ball(image_pil):
    from tensorflow.keras.preprocessing import image
    from tensorflow import convert_to_tensor, expand_dims

    image_pil = image_pil.resize((64, 64))
    image_array = image.img_to_array(image_pil)
    image_tensor = convert_to_tensor(image_array)
    image_tensor = image_tensor / 255.0

    image_tensor = expand_dims(image_tensor, axis=0)

    result = loaded_model.predict(image_tensor)

    return 'No Ball' if result[0][0] >= 0.5 else 'Legal Ball'

st.title("No Ball Detection System")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

# Upload video
if uploaded_video is not None:
    video = uploaded_video.read()
    temp_video = r"temp\temp_video.mp4"
    temp_frame = r"temp\temp_frame.jpg"

    import os
    os.makedirs("temp", exist_ok=True)

    with open(temp_video, "wb") as f:
        f.write(video)
    
    # Open video with OpenCV
    cap = cv2.VideoCapture(temp_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # Control playback using Streamlit slider
    time = st.slider("Pause at time (s)", 0.0, duration, 0.0)
    frame_number = int(time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame at the selected time
    success, frame = cap.read()
    if success:
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption=f"Frame at {time:.2f} seconds", use_column_width=True)
    
    if st.button("Submit Frame Time"):
        st.session_state["paused_time"] = time
        st.write(f"Recorded frame time: {time:.2f} seconds")

        def get_frame_at_time(video_path, t):
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Set the video position in milliseconds
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            
            # Read the frame at the specified time
            success, frame = cap.read()
            cap.release()
            
            if success:
                # Convert frame to RGB if necessary
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return None

        frame = get_frame_at_time(temp_video, time)
        if frame is not None:
            st.image(frame, caption='Uploaded Image', use_column_width=True)
            cv2.imwrite(temp_frame, frame)

            image = Image.open(temp_frame)

            prediction = predict_no_ball(image)
            st.write(f"Prediction: {prediction}")   
    
    cap.release()

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_no_ball(image)
    st.write(f"Prediction: {prediction}")
