import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request, os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model (~25 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def draw_hand(frame, landmarks, h, w):
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 200), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, (0, 150, 255), 1)


# ── Gesture logic ─────────────────────────────────────────────────────────────
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

    if (index_tip.y < index_base.y and
        middle_tip.y < middle_base.y and
        ring_tip.y < ring_base.y and
        pinky_tip.y < pinky_base.y):
        return "STOP"

    if (index_tip.y > index_base.y and
        middle_tip.y > middle_base.y and
        ring_tip.y > ring_base.y and
        pinky_tip.y > pinky_base.y):
        return "YES"

    if (index_tip.y < index_base.y and
        middle_tip.y < middle_base.y and
        ring_tip.y > ring_base.y and
        pinky_tip.y > pinky_base.y):
        return "NO"

    if (thumb_tip.y < index_base.y and
        pinky_tip.y < pinky_base.y and
        index_tip.y > index_base.y and
        middle_tip.y > middle_base.y and
        ring_tip.y > ring_base.y):
        return "CALL ME"

    dist = ((thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    if (dist < 0.05 and
        middle_tip.y < middle_base.y and
        ring_tip.y < ring_base.y and
        pinky_tip.y < pinky_base.y):
        return "OK"

    return "UNKNOWN"


latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=result_callback,
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
timestamp_ms = 0
gesture_text = "No Hand"

print("Camera running — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Empty frame, skipping.")
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker.detect_async(mp_image, timestamp_ms)
    timestamp_ms += 33

    if latest_result and latest_result.hand_landmarks:
        gesture_text = "UNKNOWN"
        for hand_landmarks in latest_result.hand_landmarks:
            draw_hand(frame, hand_landmarks, h, w)
            gesture_text = detect_gesture(hand_landmarks)
    else:
        gesture_text = "No Hand"

    cv2.putText(frame, gesture_text,
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.8,
                (0, 255, 0),
                3)

    cv2.imshow("Gestellence Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()