import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import av
import threading

# ── Download model once ───────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading hand landmark model (~25 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

model_path = download_model()

# ── Manual drawing ────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand(frame, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 150, 255), 1)

# ── Gesture detection ─────────────────────────────────────────────────────────
def detect_gesture(landmarks):
    thumb_tip  = landmarks[4]
    index_tip  = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip   = landmarks[16]
    pinky_tip  = landmarks[20]
    index_base  = landmarks[6]
    middle_base = landmarks[10]
    ring_base   = landmarks[14]
    pinky_base  = landmarks[18]

    if (index_tip.y < index_base.y and middle_tip.y < middle_base.y and
        ring_tip.y < ring_base.y and pinky_tip.y < pinky_base.y):
        return "STOP"
    if (index_tip.y > index_base.y and middle_tip.y > middle_base.y and
        ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y):
        return "YES"
    if (index_tip.y < index_base.y and middle_tip.y < middle_base.y and
        ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y):
        return "NO"
    if (thumb_tip.y < index_base.y and pinky_tip.y < pinky_base.y and
        index_tip.y > index_base.y and middle_tip.y > middle_base.y and
        ring_tip.y > ring_base.y):
        return "CALL ME"
    dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    if (dist < 0.05 and middle_tip.y < middle_base.y and
        ring_tip.y < ring_base.y and pinky_tip.y < pinky_base.y):
        return "OK"
    return "UNKNOWN"

# ── Video Processor ───────────────────────────────────────────────────────────
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_gesture = "No Hand"
        self._latest_landmarks = []
        self._timestamp_ms = 0

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            result_callback=self._callback,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def _callback(self, result, output_image, timestamp_ms):
        with self._lock:
            self._latest_landmarks = result.hand_landmarks if result.hand_landmarks else []
            if self._latest_landmarks:
                self._latest_gesture = detect_gesture(self._latest_landmarks[0])
            else:
                self._latest_gesture = "No Hand"

    @property
    def gesture(self):
        with self._lock:
            return self._latest_gesture

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._landmarker.detect_async(mp_image, self._timestamp_ms)
        self._timestamp_ms += 33

        with self._lock:
            landmarks_copy = list(self._latest_landmarks)
            gesture = self._latest_gesture

        for hand_landmarks in landmarks_copy:
            draw_hand(img, hand_landmarks, h, w)

        cv2.putText(img, gesture, (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    (0, 255, 0), 3, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gestellence", page_icon="🤙", layout="centered")
st.title("🤙 Gestellence – Live Gesture Translation")
st.caption("Academic Project · MUJ 2026")

st.info(
    "**Supported gestures:** ✋ STOP | ✊ YES | ✌️ NO | 🤙 CALL ME | 👌 OK",
    icon="ℹ️"
)

ctx = webrtc_streamer(
    key="gestellence-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    st.markdown(f"### Detected: `{ctx.video_processor.gesture}`")