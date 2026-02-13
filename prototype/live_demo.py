import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os

# MUJ PBL Branding
st.set_page_config(page_title="Gestellence Live", layout="centered")
st.title("Gestellence – AI Sign Language Translator")
st.write("Academic Project 2026 - Dept. of CSE, MUJ")

MODEL_PATH = "hand_landmarker.task"

# Cache the model so it doesn't reload every time you take a photo
@st.cache_resource
def load_landmarker():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7
    )
    return mp_vision.HandLandmarker.create_from_options(options)

landmarker = load_landmarker()

def detect_gesture(landmarks):
    # Your exact gesture logic from localhost
    thumb_tip, index_tip = landmarks[4], landmarks[8]
    middle_tip, ring_tip = landmarks[12], landmarks[16]
    pinky_tip, index_base = landmarks[20], landmarks[6]
    middle_base, ring_base, pinky_base = landmarks[10], landmarks[14], landmarks[18]

    if (index_tip.y < index_base.y and middle_tip.y < middle_base.y and
        ring_tip.y < ring_base.y and pinky_tip.y < pinky_base.y):
        return "STOP"
    if (index_tip.y > index_base.y and middle_tip.y > middle_base.y and
        ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y):
        return "YES"
    # ... add your NO, CALL ME, and OK logic here ...
    return "UNKNOWN"

# The Browser Camera Input
img_file = st.camera_input("Show a gesture to the camera")

if img_file:
    # Process the image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            gesture = detect_gesture(hand_lms)
            st.metric(label="Detected Gesture", value=gesture)
            # Visual feedback in the app
            if gesture == "STOP":
                st.warning("Action: STOP")
            elif gesture == "YES":
                st.success("Action: YES")
    else:
        st.info("No hand detected. Please hold your hand up clearly.")