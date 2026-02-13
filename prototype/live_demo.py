import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import base64
import cv2
import numpy as np
import tempfile

# ── Download model ────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (~25 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

download_model()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gestellence", page_icon="🤙", layout="centered")
st.title("🤙 Gestellence – Live Gesture Translation")
st.caption("Academic Project · MUJ 2026")

st.info(
    "**Supported gestures:** ✋ STOP | ✊ YES | ✌️ NO | 🤙 CALL ME | 👌 OK",
    icon="ℹ️"
)

# ── Inject browser-side MediaPipe (runs 100% in browser, zero WebRTC needed) ──
components.html("""
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      background: #0e1117;
      font-family: 'Segoe UI', sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    #container {
      position: relative;
      width: 640px;
      max-width: 100vw;
    }
    video, canvas {
      width: 100%;
      border-radius: 12px;
    }
    canvas {
      position: absolute;
      top: 0; left: 0;
    }
    #gesture-box {
      margin-top: 12px;
      padding: 14px 28px;
      background: #1e2130;
      border-radius: 10px;
      border: 1px solid #2e3250;
      text-align: center;
      width: 100%;
      box-sizing: border-box;
    }
    #gesture-label {
      font-size: 2.2rem;
      font-weight: 700;
      color: #00e676;
      letter-spacing: 2px;
    }
    #status {
      font-size: 0.8rem;
      color: #888;
      margin-top: 4px;
    }
    #start-btn {
      margin-top: 14px;
      padding: 12px 36px;
      background: #ff4b4b;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    #start-btn:hover { background: #e03c3c; }
  </style>
</head>
<body>

<div id="container">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>
</div>

<div id="gesture-box">
  <div id="gesture-label">No Hand</div>
  <div id="status">Click START to begin</div>
</div>

<button id="start-btn" onclick="startCamera()">▶ START CAMERA</button>

<script type="module">
import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video   = document.getElementById("video");
const canvas  = document.getElementById("canvas");
const ctx     = canvas.getContext("2d");
const label   = document.getElementById("gesture-label");
const status  = document.getElementById("status");

let handLandmarker = null;
let running = false;
let lastTimestamp = 0;

// Gesture colours
const GESTURE_COLORS = {
  "STOP":    "#ff4b4b",
  "YES":     "#00e676",
  "NO":      "#ff9800",
  "CALL ME": "#29b6f6",
  "OK":      "#ce93d8",
  "UNKNOWN": "#ffffff",
  "No Hand": "#555555",
};

// Hand connections
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

function detectGesture(lm) {
  const thumbTip  = lm[4],  indexTip  = lm[8];
  const middleTip = lm[12], ringTip   = lm[16], pinkyTip  = lm[20];
  const indexBase = lm[6],  middleBase= lm[10];
  const ringBase  = lm[14], pinkyBase = lm[18];

  const up   = (tip, base) => tip.y < base.y;
  const down = (tip, base) => tip.y > base.y;

  if (up(indexTip,indexBase) && up(middleTip,middleBase) &&
      up(ringTip,ringBase)   && up(pinkyTip,pinkyBase))   return "STOP";

  if (down(indexTip,indexBase) && down(middleTip,middleBase) &&
      down(ringTip,ringBase)   && down(pinkyTip,pinkyBase))  return "YES";

  if (up(indexTip,indexBase)   && up(middleTip,middleBase) &&
      down(ringTip,ringBase)   && down(pinkyTip,pinkyBase)) return "NO";

  if (thumbTip.y < indexBase.y && up(pinkyTip,pinkyBase) &&
      down(indexTip,indexBase) && down(middleTip,middleBase) &&
      down(ringTip,ringBase))                               return "CALL ME";

  const dist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
  if (dist < 0.05 && up(middleTip,middleBase) &&
      up(ringTip,ringBase) && up(pinkyTip,pinkyBase))       return "OK";

  return "UNKNOWN";
}

function drawHand(landmarks, W, H) {
  ctx.strokeStyle = "rgba(0,200,200,0.8)";
  ctx.lineWidth = 2;
  for (const [a, b] of CONNECTIONS) {
    const p1 = landmarks[a], p2 = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(p1.x * W, p1.y * H);
    ctx.lineTo(p2.x * W, p2.y * H);
    ctx.stroke();
  }
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * W, lm.y * H, 5, 0, 2 * Math.PI);
    ctx.fillStyle = "white";
    ctx.fill();
    ctx.strokeStyle = "#0096ff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

async function initModel() {
  status.textContent = "Loading MediaPipe model...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.7,
    minHandPresenceConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });
  status.textContent = "Model ready ✓";
}

function predict() {
  if (!running) return;

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  const W = canvas.width, H = canvas.height;

  // Mirror
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -W, 0, W, H);
  ctx.restore();

  const now = performance.now();
  if (now === lastTimestamp) { requestAnimationFrame(predict); return; }
  lastTimestamp = now;

  const results = handLandmarker.detectForVideo(video, now);

  if (results.landmarks && results.landmarks.length > 0) {
    let gesture = "UNKNOWN";
    for (const lm of results.landmarks) {
      // Mirror x coords for drawing
      const mirrored = lm.map(p => ({...p, x: 1 - p.x}));
      drawHand(mirrored, W, H);
      gesture = detectGesture(lm);
    }
    label.textContent = gesture;
    label.style.color = GESTURE_COLORS[gesture] || "#fff";
    status.textContent = `${results.landmarks.length} hand(s) detected`;
  } else {
    label.textContent = "No Hand";
    label.style.color = GESTURE_COLORS["No Hand"];
    status.textContent = "Show your hand to the camera";
  }

  requestAnimationFrame(predict);
}

window.startCamera = async function() {
  const btn = document.getElementById("start-btn");
  if (running) {
    running = false;
    btn.textContent = "▶ START CAMERA";
    status.textContent = "Stopped.";
    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    return;
  }

  if (!handLandmarker) await initModel();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    running = true;
    btn.textContent = "⏹ STOP";
    status.textContent = "Camera running...";
    requestAnimationFrame(predict);
  } catch (err) {
    status.textContent = "Camera error: " + err.message;
  }
};
</script>
</body>
</html>
""", height=620, scrolling=True)

st.markdown("---")
st.markdown(
    "Built with [MediaPipe](https://mediapipe.dev) · "
    "[Streamlit](https://streamlit.io) · MUJ 2026"
)