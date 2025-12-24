# fatigue_dashboard3.py
# optional.py
#this code conisits of head tilt alert
"""
Streamlit dashboard for internship project:
Detecting Fatigue from Facial Landmarks — multi-page app.

Pages (sidebar - now reduced):
 - Ethical & Privacy Considerations
 - Real-time Webcam Detection (Continuous)
 - Driver Drowsy Alert Mechanism

Uses MediaPipe FaceMesh for landmark extraction and supports image upload + browser webcam (st.camera_input).
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import io
import multiprocessing
from scipy.spatial import distance as dist
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import base64
from datetime import datetime 

import threading

# --- Robust audio playback ---
try:
    from playsound import playsound
except Exception:
    playsound = None

try:
    import pygame
except Exception:
    pygame = None


def safe_play_audio(path):
    """Play alarm sound using playsound or pygame fallback."""
    if not os.path.exists(path):
        print(f"⚠️ Audio file missing: {path}")
        return

    def _threaded_play():
        try:
            if playsound:
                playsound(path, block=True)
            elif pygame:
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
            else:
                print("❌ No supported audio library found (install playsound or pygame).")
        except Exception as e:
            print(f"Audio playback failed: {e}")

    threading.Thread(target=_threaded_play, daemon=True).start()


# Optional imports for CNN
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional audio
try:
    import simpleaudio as sa
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

st.set_page_config(page_title="Detecting Fatigue Dashboard", layout="wide")

# -----------------------------
# Utility: MediaPipe FaceMesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

@st.cache_resource
def get_face_mesh(max_num_faces=1):
    return mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=max_num_faces)

def get_landmarks_from_image(img_bgr):
    """Return list of landmarks (as mp Landmarks) or None"""
    face_mesh = get_face_mesh()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks  # iterable of landmarks

# -----------------------------
# Feature calculators
# -----------------------------
LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
OUTER_MOUTH_IDX = list(range(61,68)) + list(range(291,298))
INNER_MOUTH_IDX = list(range(78,89)) + list(range(308,319))
POSE_IDX = [1, 199, 33, 263, 61, 291]

def eye_aspect_ratio_landmarks(landmarks, idxs, img_w, img_h):
    pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in idxs])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3]) + 1e-8
    return float((A + B) / (2.0 * C))

def mouth_aspect_ratio_landmarks(landmarks, outer_idx, inner_idx, img_w, img_h):
    pts_o = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in outer_idx])
    pts_i = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in inner_idx])
    # robust vertical measure
    try:
        vertical = (np.linalg.norm(pts_i[13] - pts_i[19]) +
                    np.linalg.norm(pts_i[14] - pts_i[18]) +
                    np.linalg.norm(pts_i[15] - pts_i[17])) / 3.0
    except Exception:
        vertical = np.linalg.norm(pts_o[2 % len(pts_o)] - pts_o[6 % len(pts_o)])
    width = np.linalg.norm(pts_o[0] - pts_o[6]) + 1e-8
    return float(vertical / width)

def estimate_head_pose(landmarks, img_shape):
    h, w = img_shape[:2]
    try:
        pts2d = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in POSE_IDX], dtype=np.float64)
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        focal_length = w
        center = (w/2, h/2)
        cam = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0,0,1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1))
        ok, rvec, tvec = cv2.solvePnP(model_points, pts2d, cam, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.degrees(np.arctan2(rmat[2,1], rmat[2,2]))
            y = np.degrees(np.arctan2(-rmat[2,0], sy))
            z = np.degrees(np.arctan2(rmat[1,0], rmat[0,0]))
        else:
            x = np.degrees(np.arctan2(-rmat[1,2], rmat[1,1]))
            y = np.degrees(np.arctan2(-rmat[2,0], sy))
            z = 0
        return float(x), float(y), float(z)
    except Exception:
        return None, None, None

# -----------------------------
# Model loading helpers
# -----------------------------

# --- Define Model Paths (Using the user's absolute paths for the .pkl files) ---
RF_MODEL_PATH = "models/rf_pipeline.pkl"
CNN_MODEL_PATH = "models/cnn_model.h5"
# Note: Keeping the user's explicit absolute paths for these models as they seem required.
EYE_BLINK_MODEL_PATH = "models/eye_blink_model.pkl"
YAWN_MODEL_PATH = "models/yawn_model.pkl"
# --- End Define Model Paths ---


@st.cache_resource
def load_rf_model(path=RF_MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load RF model ({path}): {e}")
    else:
        st.warning(f"⚠️ RF model not found at path: {path}")
    return None

@st.cache_resource
def load_cnn_model(path=CNN_MODEL_PATH):
    if TF_AVAILABLE and os.path.exists(path):
        try:
            return load_model(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load CNN model ({path}): {e}")
    elif TF_AVAILABLE:
        st.warning(f"⚠️ CNN model not found at path: {path}")
    return None

# --- Custom Model Loaders using Absolute Paths ---
@st.cache_resource
def load_eye_blink_model(path=EYE_BLINK_MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load Eye Blink model ({path}): {e}")
    else:
        st.warning(f"⚠️ Eye blink model not found at path: {path}")
    return None

@st.cache_resource
def load_yawn_model(path=YAWN_MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load Yawn model ({path}): {e}")
    else:
        st.warning(f"⚠️ Yawn model not found at path: {path}")
    return None
# -----------------------------
# End Model loading helpers
# -----------------------------

# -----------------------------
# Helpers: display images / plots
# -----------------------------
def image_to_bytes(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr)
    return buf.tobytes()

def plot_landmarks_scatter(landmarks, img_w, img_h):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    plt.figure(figsize=(4,5))
    plt.scatter(xs, [-y for y in ys], s=8)
    for i,(x,y) in enumerate(zip(xs,ys)):
        plt.text(x, -y, str(i), fontsize=6)
    plt.title("Facial Landmarks (indexed)")
    plt.axis('off'); st.pyplot(plt.gcf()); plt.clf()

# -----------------------------
# Page logic
# -----------------------------
st.sidebar.title("Navigation")
# Reduced pages list
pages = [
    "Real-time Webcam Detection (Continuous)", # Page 14 (Continuous Webcam)
    "Driver Drowsy Alert Mechanism"        # Page 15
]
choice = st.sidebar.radio("Go to", pages)

# common models
rf_model = load_rf_model()
cnn_model = load_cnn_model()
eye_blink_model = load_eye_blink_model()
yawn_model = load_yawn_model()



# ------------------- PAGE 14: Real-time Webcam Detection (Continuous) -------------------
if choice == "Real-time Webcam Detection (Continuous)":
    st.title("📹 Real-time Webcam Detection (Continuous)")
    st.markdown(
        "This demo runs a live webcam feed, processing each frame to detect fatigue using EAR (eye), MAR (mouth), and head tilt analysis."
    )
    st.info("Ensure eye_blink_model.pkl, yawn_model.pkl, and alarm.wav are correctly located in your working directory.")

    import threading, os, time, cv2, numpy as np
    from playsound import playsound

    # --- Alarm setup ---
    ALARM_ACTIVE_FLAG = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")
    EVENTS_FILE = os.path.join(BASE_DIR, "drowsy_events.csv")

    def play_alarm_sound(path):
        """Play alarm sound safely in a separate thread."""
        global ALARM_ACTIVE_FLAG
        if ALARM_ACTIVE_FLAG:
            return
        ALARM_ACTIVE_FLAG = True
        try:
            safe_path = os.path.normpath(path)
            playsound(safe_path)
        except Exception as e:
            print("Audio error:", e)
        ALARM_ACTIVE_FLAG = False

    def play_alarm_non_blocking(path):
        """Play alarm only if not already active."""
        global ALARM_ACTIVE_FLAG
        if not ALARM_ACTIVE_FLAG:
            threading.Thread(target=play_alarm_sound, args=(path,), daemon=True).start()

    # --- Event logging for dashboard ---
    def log_event(event_type, extra=None):
        import pandas as pd
        now = pd.Timestamp.now().isoformat()
        entry = {"timestamp": now, "event": event_type}
        if extra:
            entry.update(extra)
        df = pd.DataFrame([entry])
        file_exists = os.path.exists(EVENTS_FILE)
        df.to_csv(EVENTS_FILE, mode='a', header=not file_exists, index=False)

    # user behaviour trends dashboard
    import pandas as pd
    if os.path.exists(EVENTS_FILE):
        st.subheader("User Behaviour Trends (Past Days/Months)")
        df_evt = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"], on_bad_lines="skip")
        df_evt["day"] = pd.to_datetime(df_evt["timestamp"]).dt.date
        daily = df_evt.groupby(["day", "event"]).size().unstack(fill_value=0)
        st.bar_chart(daily)

    # --- Streamlit control buttons ---
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Webcam", key="start_rt_webcam"):
            st.session_state.run_webcam = True
    with col2:
        if st.button("Stop Webcam", key="stop_rt_webcam"):
            st.session_state.run_webcam = False

    # --- Webcam processing loop ---
    idx = 0
    for index in range(5):  # Check indices 0 through 4
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            idx = index
            cap.release()
            break
    else:
        print(f"No camera at index {idx}")

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            st.error("❌ Could not open webcam.")
            st.session_state.run_webcam = False
        else:
            frame_placeholder = st.empty()
            FRAME_RATE = 30

            consecutive_eye_closed = 0
            consecutive_yawn = 0
            consecutive_tilt = 0  # NEW: head-tilt counter

            try:
                while st.session_state.run_webcam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠ Frame capture failed.")
                        break

                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    color_default = (255, 255, 255)
                    fatigue_pct_eye = 0
                    fatigue_pct_yawn = 0
                    fatigue_pct_tilt = 0      # NEW: head-tilt fatigue %
                    status_text_eye = ""
                    status_text_yawn = ""
                    status_text_tilt = ""      # NEW: head-tilt status text

                    lms_all = get_landmarks_from_image(frame)
                    if lms_all:
                        lm = lms_all[0].landmark
                        ear = (
                            eye_aspect_ratio_landmarks(lm, LEFT_EYE_IDX, w, h)
                            + eye_aspect_ratio_landmarks(lm, RIGHT_EYE_IDX, w, h)
                        ) / 2.0
                        mar = mouth_aspect_ratio_landmarks(
                            lm, OUTER_MOUTH_IDX, INNER_MOUTH_IDX, w, h
                        )
                        pitch, yaw, roll = estimate_head_pose(lm, frame.shape)

                        # --- Predictions ---
                        eye_pred = int(
                            eye_blink_model.predict(np.array([ear]).reshape(1, -1))[0]
                        )
                        mar_pred = int(
                            yawn_model.predict(np.array([mar]).reshape(1, -1))[0]
                        )
                        ear_label = "OPEN" if eye_pred == 1 else "CLOSED"
                        yawn_label = "YAWN" if mar_pred == 1 else "NO YAWN"

                        # --- Log events for dashboard ---
                        log_event("EYE_" + ear_label)

                        # --- Head tilt detection (down or strong sideways) ---
                        head_tilt = (
                            pitch is not None
                            and (pitch > 20 or abs(yaw) > 20 or abs(roll) > 20)
                        )
                        eye_closed = ear_label == "CLOSED"

                        # Eye closed counter
                        if eye_closed:
                            consecutive_eye_closed += 1
                        else:
                            consecutive_eye_closed = 0

                        # Yawn counter (model + MAR threshold)
                        yawn_detected = (mar_pred == 1 and mar > 0.35)
                        if yawn_detected:
                            consecutive_yawn += 1
                            if consecutive_yawn / FRAME_RATE >= 3:
                                log_event("YAWN_ALARM")
                            else:
                                log_event("YAWN_WARNING")
                        else:
                            consecutive_yawn = 0

                        # Head tilt counter
                        if head_tilt:
                            consecutive_tilt += 1
                        else:
                            consecutive_tilt = 0

                        eye_closed_sec = consecutive_eye_closed / FRAME_RATE
                        yawn_sec = consecutive_yawn / FRAME_RATE
                        tilt_sec = consecutive_tilt / FRAME_RATE  # NEW timer

                        # --- Eye-blink fatigue logic ---
                        if eye_closed_sec >= 3:
                            fatigue_pct_eye = 100
                            status_text_eye = "🚨 FATIGUE DETECTED – TAKE A BREAK!"
                            color_status_eye = (0, 0, 255)
                            play_alarm_non_blocking(ALARM_PATH)
                        elif 0 < eye_closed_sec < 3:
                            fatigue_pct_eye = 50
                            status_text_eye = "STATUS: FATIGUE WARNING!!"
                            color_status_eye = (0, 165, 255)
                        else:
                            fatigue_pct_eye = 0
                            status_text_eye = "STATUS: NORMAL (ATTENTIVE)"
                            color_status_eye = (0, 255, 0)

                        # --- Yawn logic ---
                        if yawn_detected and yawn_sec >= 3:
                            fatigue_pct_yawn = 100
                            status_text_yawn = "⚠ YAWN ALERT — TAKE A BREAK!"
                            color_status_yawn = (0, 0, 255)
                            play_alarm_non_blocking(ALARM_PATH)
                        elif yawn_detected and 0 < yawn_sec < 3:
                            fatigue_pct_yawn = 50
                            status_text_yawn = "YAWN DETECTED! WARNING"
                            color_status_yawn = (0, 165, 255)
                        else:
                            fatigue_pct_yawn = 0
                            status_text_yawn = "NO YAWN"
                            color_status_yawn = (0, 255, 0)

                        # --- Head tilt fatigue logic (NEW) ---
                        if tilt_sec >= 3:
                            fatigue_pct_tilt = 100
                            status_text_tilt = "⚠ HEAD TILT ALERT — KEEP YOUR HEAD UP!"
                            color_status_tilt = (0, 0, 255)
                            play_alarm_non_blocking(ALARM_PATH)
                            log_event("TILT_ALARM")
                        elif 0 < tilt_sec < 3:
                            fatigue_pct_tilt = 50
                            status_text_tilt = "HEAD TILT WARNING"
                            color_status_tilt = (0, 165, 255)
                            log_event("TILT_WARNING")
                        else:
                            fatigue_pct_tilt = 0
                            status_text_tilt = "NO TILT"
                            color_status_tilt = (0, 255, 0)

                        # --- Timers text ---
                        cv2.putText(
                            frame,
                            f"Eye closed timer: {eye_closed_sec:.1f}s",
                            (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Yawn timer: {yawn_sec:.1f}s",
                            (10, 230),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Tilt timer: {tilt_sec:.1f}s",  # NEW timer display
                            (10, 260),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )

                        # --- Fatigue overlays ---
                        cv2.putText(
                            frame,
                            f"STATUS EYEBLINK: {ear_label}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"EAR: {ear:.3f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"MAR: {mar:.3f} ({yawn_label})",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color_default,
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Fatigue (Eye): {fatigue_pct_eye}%",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255) if fatigue_pct_eye >= 60 else (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Fatigue (Yawn): {fatigue_pct_yawn}%",
                            (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255) if fatigue_pct_yawn >= 60 else (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"Fatigue (Tilt): {fatigue_pct_tilt}%",  # NEW %
                            (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255) if fatigue_pct_tilt >= 60 else (0, 255, 0),
                            2,
                        )

                        # --- Status overlays ---
                        cv2.putText(
                            frame,
                            status_text_eye,
                            (10, 300),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            color_status_eye,
                            2,
                        )
                        cv2.putText(
                            frame,
                            status_text_yawn,
                            (10, 330),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            color_status_yawn,
                            2,
                        )
                        cv2.putText(
                            frame,
                            status_text_tilt,  # NEW tilt status line
                            (10, 360),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            color_status_tilt,
                            2,
                        )

                        # --- Red overlay when any fatigue is 100% ---
                        if (
                            fatigue_pct_eye == 100
                            or fatigue_pct_yawn == 100
                            or fatigue_pct_tilt == 100
                        ):
                            overlay = frame.copy()
                            cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 255), -1)
                            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                            cv2.putText(
                                frame,
                                "🚨 FATIGUE ALERT 🚨",
                                (w // 2 - 200, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (255, 255, 255),
                                3,
                            )

                    # --- Stream frame to Streamlit ---
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        frame_rgb, channels="RGB", use_container_width=True
                    )
                    time.sleep(0.03)

            finally:
                cap.release()
                st.session_state.run_webcam = False
                st.info("✅ Webcam stopped.")

        # quick model sanity check (optional)
        model = joblib.load(
            r"C:\Users\User\Downloads\fatigue_detection\fatigue\fatigueapi\models\eye_blink_model.pkl"
        )
        test_ears = np.array([[0.15], [0.25], [0.35]])
        preds = model.predict(test_ears)
        for ear, pred in zip(test_ears.flatten(), preds):
            print(f"EAR={ear:.2f} -> Model Prediction={pred}")

    # import pandas as pd
    # if os.path.exists(EVENTS_FILE):
    #     st.subheader("User Behaviour Trends (Past Days/Months)")
    #     df_evt = pd.read_csv(EVENTS_FILE, parse_dates=["timestamp"], on_bad_lines="skip")
    #     df_evt["day"] = pd.to_datetime(df_evt["timestamp"]).dt.date
    #     daily = df_evt.groupby(["day", "event"]).size().unstack(fill_value=0)
    #     st.bar_chart(daily)


# ------------------- PAGE 15: Driver Drowsy Alert Mechanism -------------------
# elif choice == "Driver Drowsy Alert Mechanism":
#     import streamlit as st
#     import cv2, joblib, numpy as np, mediapipe as mp, os, time, threading
#     from playsound import playsound


#     st.title("🚨 Driver Drowsy Alert Mechanism")
#     st.markdown("""
#     *EAR & MAR based fatigue detection*  
#     - Real-time EAR / MAR metrics  
#     - Head-pose readiness  
#     - 3s sustained eye-closure / yawn alarm trigger  
#     - Continuous visual warnings  
#     """)


#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     MODELS_DIR = os.path.join(BASE_DIR, "models")
#     ALARM_FILE = os.path.join(BASE_DIR, "alarm.wav")


#     EYE_BLINK_MODEL_PATH = os.path.join(MODELS_DIR, "eye_blink_model.pkl")
#     YAWN_MODEL_PATH = os.path.join(MODELS_DIR, "yawn_model.pkl")
#     ALARM_SOUND_PATH = ALARM_FILE


#     for name, path in {
#         "Eye Blink Model": EYE_BLINK_MODEL_PATH,
#         "Yawn Model": YAWN_MODEL_PATH,
#         "Alarm Sound": ALARM_SOUND_PATH,
#     }.items():
#         if not os.path.exists(path):
#             st.warning(f"⚠ {name} not found: {path}")
#         else:
#             st.success(f"✅ {name} loaded: {os.path.basename(path)}")


#     # --- Detection constants ---
#     ALARM_COOLDOWN = 5.0
#     EYE_OPEN_RECOVERY_FRAMES = 5
#     CRITICAL_EYE_CLOSE_FRAMES = 6
#     DANGER_DELAY = 3.0  # seconds of continuous closure before alarm
#     YAWN_DELAY = 3.0    # seconds of continuous yawn before alarm
#     HEAD_TILT_THRESHOLD = 15  # degrees tilt threshold for alert
#     HEAD_TILT_DELAY = 3.0     # seconds of sustained head tilt before alarm


#     DANGER_ACTIVE = False
#     ALARM_ACTIVE_FLAG = False


#     # --- Helper functions ---
#     def play_sound_thread(path):
#         global ALARM_ACTIVE_FLAG
#         ALARM_ACTIVE_FLAG = True
#         try:
#             playsound(path)
#         except Exception as e:
#             print("Audio error:", e)
#         ALARM_ACTIVE_FLAG = False


#     def play_alarm_non_blocking(path, last_time, cooldown):
#         global ALARM_ACTIVE_FLAG
#         t = time.time()
#         if (t - last_time) > cooldown and not ALARM_ACTIVE_FLAG:
#             if os.path.exists(path):
#                 threading.Thread(target=play_sound_thread, args=(path,), daemon=True).start()
#                 return t
#         return last_time


#     st.info("Click below to start detection.")


#     start_btn = st.button("▶ Start Drowsy Detection")
#     stop_btn = st.button("⏹ Stop Detection")


#     if "running" not in st.session_state:
#         st.session_state.running = False


#     if start_btn:
#         st.session_state.running = True
#         st.success("✅ Detection started.")
#     if stop_btn:
#         st.session_state.running = False
#         st.warning("🛑 Detection stopped.")


#     FRAME_WINDOW = st.image([])


#     if st.session_state.running:
#         eye_model = joblib.load(EYE_BLINK_MODEL_PATH)
#         yawn_model = joblib.load(YAWN_MODEL_PATH)
#         mp_face = mp.solutions.face_mesh
#         cap = cv2.VideoCapture(0)


#         last_audio_alert_time = time.time() - ALARM_COOLDOWN
#         consecutive_closed, open_recovery = 0, 0
#         danger_start_time, yawn_start_time = None, None  # Timers for eyes and yawn
#         head_tilt_start_time = None  # Timer for head tilt


#         # Landmark indices for head pose tilt approx
#         LEFT_EYE_IDX = 33
#         RIGHT_EYE_IDX = 263


#         with mp_face.FaceMesh(
#             refine_landmarks=True, max_num_faces=1,
#             min_detection_confidence=0.6, min_tracking_confidence=0.6
#         ) as face_mesh:
#             while st.session_state.running:
#                 ret, frame = cap.read()
#                 if not ret:
#                     st.error("❌ Camera not detected.")
#                     break


#                 frame = cv2.flip(frame, 1)
#                 h, w = frame.shape[:2]
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 results = face_mesh.process(rgb)

#                 # default timer values each loop
#                 elapsed_danger = 0
#                 elapsed_yawn = 0
#                 elapsed_tilt = 0

#                 if results.multi_face_landmarks:
#                     lm = results.multi_face_landmarks[0].landmark

#                     def ratio(idxs):
#                         pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idxs])
#                         A = np.linalg.norm(pts[1]-pts[5])
#                         B = np.linalg.norm(pts[2]-pts[4])
#                         C = np.linalg.norm(pts[0]-pts[3]) + 1e-8
#                         return (A+B)/(2*C)

#                     LEFT = [362,385,387,263,373,380]
#                     RIGHT = [33,160,158,133,153,144]
#                     MOUTH = [78,308,13,14,87,317]
#                     ear = np.mean([ratio(LEFT), ratio(RIGHT)])
#                     mar = ratio(MOUTH)

#                     # --- Yawn and Eye Prediction ---
#                     eye_state = int(eye_model.predict([[ear]])[0])
#                     yawn_state = int(yawn_model.predict([[mar]])[0])

#                     # --- Stability control for yawn ---
#                     YAWN_THRESHOLD = 0.58
#                     yawn_raw = (mar > YAWN_THRESHOLD)

#                     # --- Eye blink handling ---
#                     if eye_state == 0:
#                         consecutive_closed += 1
#                         open_recovery = 0
#                     else:
#                         consecutive_closed = 0
#                         open_recovery += 1

#                     is_danger = (consecutive_closed >= CRITICAL_EYE_CLOSE_FRAMES)

#                     now = time.time()

#                     # --------- EYE BLINK STATUS + TIMER ----------
#                     eye_blink_detected = (eye_state == 0)  # 0 = closed
#                     if eye_blink_detected:
#                         if danger_start_time is None:
#                             danger_start_time = now
#                         elapsed_danger = now - danger_start_time
#                     else:
#                         danger_start_time = None
#                         elapsed_danger = 0

#                     # --------- YAWN STATUS + TIMER ----------
#                     yawn_detected = (yawn_state == 1 and yawn_raw)
#                     if yawn_detected:
#                         if yawn_start_time is None:
#                             yawn_start_time = now
#                         elapsed_yawn = now - yawn_start_time
#                     else:
#                         yawn_start_time = None
#                         elapsed_yawn = 0

#                     # --- Head tilt detection (unchanged) ---
#                     left_eye = np.array([lm[LEFT_EYE_IDX].x * w, lm[LEFT_EYE_IDX].y * h])
#                     right_eye = np.array([lm[RIGHT_EYE_IDX].x * w, lm[RIGHT_EYE_IDX].y * h])
#                     eye_vector = right_eye - left_eye
#                     angle_rad = np.arctan2(eye_vector[1], eye_vector[0])
#                     angle_deg = np.degrees(angle_rad)

#                     head_tilted = abs(angle_deg) > HEAD_TILT_THRESHOLD

#                     if head_tilted:
#                         if head_tilt_start_time is None:
#                             head_tilt_start_time = now
#                         elapsed_tilt = now - head_tilt_start_time
#                     else:
#                         head_tilt_start_time = None
#                         elapsed_tilt = 0

#                     # --- Display status and alerts ---
#                     if is_danger:
#                         status_text = "STATUS: DANGER: IMMEDIATE ALERT"
#                         status_color = (0, 0, 255)
#                         cv2.putText(frame, "WAKE UP! PULL OVER!", (80, int(h / 2)),
#                                     cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 4)
#                     elif consecutive_closed > 0 and consecutive_closed < CRITICAL_EYE_CLOSE_FRAMES:
#                         status_text = "STATUS: STRAINING (LOW BLINKS)"
#                         status_color = (0, 165, 255)
#                     else:
#                         status_text = "STATUS: NORMAL (ATTENTIVE)"
#                         status_color = (0, 255, 0)

#                     cv2.putText(frame, status_text, (10, 60),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
#                     cv2.putText(frame, f"EAR: {ear:.2f}", (10, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#                     cv2.putText(frame, f"MAR: {mar:.2f}", (200, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#                     # --- EYE BLINK YES/NO + TIMER overlay ---
#                     cv2.putText(
#                         frame,
#                         f"EYE BLINK: {'YES' if eye_blink_detected else 'NO'}",
#                         (10, 135),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8,
#                         (0, 0, 255) if eye_blink_detected else (0, 255, 0),
#                         2,
#                     )
#                     if elapsed_danger > 0:
#                         cv2.putText(
#                             frame,
#                             f"EYE TIMER: {elapsed_danger:.1f}s",
#                             (10, 165),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.8,
#                             (0, 0, 255) if elapsed_danger >= DANGER_DELAY else (0, 255, 255),
#                             2,
#                         )

#                     # --- YAWN YES/NO + TIMER overlay ---
#                     cv2.putText(
#                         frame,
#                         f"YAWN: {'YES' if yawn_detected else 'NO'}",
#                         (10, 195),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.8,
#                         (0, 0, 255) if yawn_detected else (0, 255, 0),
#                         2,
#                     )
#                     if elapsed_yawn > 0:
#                         cv2.putText(
#                             frame,
#                             f"YAWN TIMER: {elapsed_yawn:.1f}s",
#                             (10, 225),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.8,
#                             (0, 0, 255) if elapsed_yawn >= YAWN_DELAY else (0, 255, 255),
#                             2,
#                         )

#                     # Head Tilt Display (unchanged)
#                     if head_tilted:
#                         cv2.putText(frame, f"HEAD TILT: YES ({angle_deg:.1f} deg)", (10, 255),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#                     else:
#                         cv2.putText(frame, "HEAD TILT: NO", (10, 255),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#                     # --- Alarm triggering for sustained conditions ---
#                     if (elapsed_danger >= DANGER_DELAY) or (elapsed_yawn >= YAWN_DELAY) or (elapsed_tilt >= HEAD_TILT_DELAY):
#                         last_audio_alert_time = play_alarm_non_blocking(
#                             ALARM_SOUND_PATH, last_audio_alert_time, ALARM_COOLDOWN
#                         )
#                         # Optional status message when timers exceed 3s
#                         if elapsed_danger >= DANGER_DELAY:
#                             cv2.putText(frame, "⚠ POSSIBLE EYE BLINK FATIGUE", (50, int(h * 0.82)),
#                                         cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
#                         if elapsed_yawn >= YAWN_DELAY:
#                             cv2.putText(frame, "⚠ POSSIBLE YAWN FATIGUE", (50, int(h * 0.87)),
#                                         cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
#                     elif eye_blink_detected or yawn_detected or head_tilted:
#                         cv2.putText(frame, "⚠ WARNING: POSSIBLE FATIGUE", (50, int(h * 0.9)),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

#                     if elapsed_tilt >= HEAD_TILT_DELAY:
#                         cv2.putText(frame, "⚠ ALERT: HEAD TILT DETECTED!", (50, int(h * 0.85)),
#                                     cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

#                 FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


#             cap.release()
#             st.success("🟢 Detection stopped successfully.")

# ------------------- PAGE 15: Driver Drowsy Alert Mechanism -------------------
elif choice == "Driver Drowsy Alert Mechanism":
    import streamlit as st
    import cv2, joblib, numpy as np, mediapipe as mp, os, time, threading
    from playsound import playsound


    st.title("🚨 Driver Drowsy Alert Mechanism")
    st.markdown("""
    *EAR & MAR based fatigue detection*  
    - Real-time EAR / MAR metrics  
    - Head-pose readiness  
    - 3s sustained eye-closure / yawn alarm trigger  
    - Continuous visual warnings  
    """)


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    ALARM_FILE = os.path.join(BASE_DIR, "alarm.wav")


    EYE_BLINK_MODEL_PATH = os.path.join(MODELS_DIR, "eye_blink_model.pkl")
    YAWN_MODEL_PATH = os.path.join(MODELS_DIR, "yawn_model.pkl")
    ALARM_SOUND_PATH = ALARM_FILE


    for name, path in {
        "Eye Blink Model": EYE_BLINK_MODEL_PATH,
        "Yawn Model": YAWN_MODEL_PATH,
        "Alarm Sound": ALARM_SOUND_PATH,
    }.items():
        if not os.path.exists(path):
            st.warning(f"⚠ {name} not found: {path}")
        else:
            st.success(f"✅ {name} loaded: {os.path.basename(path)}")


    # --- Detection constants ---
    ALARM_COOLDOWN = 5.0
    EYE_OPEN_RECOVERY_FRAMES = 5
    CRITICAL_EYE_CLOSE_FRAMES = 6
    DANGER_DELAY = 3.0  # seconds of continuous closure before alarm
    YAWN_DELAY = 3.0    # seconds of continuous yawn before alarm
    HEAD_TILT_THRESHOLD = 15  # degrees tilt threshold for alert
    HEAD_TILT_DELAY = 3.0     # seconds of sustained head tilt before alarm


    DANGER_ACTIVE = False
    ALARM_ACTIVE_FLAG = False


    # --- Helper functions ---
    def play_sound_thread(path):
        global ALARM_ACTIVE_FLAG
        ALARM_ACTIVE_FLAG = True
        try:
            playsound(path)
        except Exception as e:
            print("Audio error:", e)
        ALARM_ACTIVE_FLAG = False


    def play_alarm_non_blocking(path, last_time, cooldown):
        global ALARM_ACTIVE_FLAG
        t = time.time()
        if (t - last_time) > cooldown and not ALARM_ACTIVE_FLAG:
            if os.path.exists(path):
                threading.Thread(target=play_sound_thread, args=(path,), daemon=True).start()
                return t
        return last_time


    st.info("Click below to start detection.")


    start_btn = st.button("▶ Start Drowsy Detection")
    stop_btn = st.button("⏹ Stop Detection")


    if "running" not in st.session_state:
        st.session_state.running = False


    if start_btn:
        st.session_state.running = True
        st.success("✅ Detection started.")
    if stop_btn:
        st.session_state.running = False
        st.warning("🛑 Detection stopped.")


    FRAME_WINDOW = st.image([])


    if st.session_state.running:
        eye_model = joblib.load(EYE_BLINK_MODEL_PATH)
        yawn_model = joblib.load(YAWN_MODEL_PATH)
        mp_face = mp.solutions.face_mesh
        cap = cv2.VideoCapture(0)


        last_audio_alert_time = time.time() - ALARM_COOLDOWN
        consecutive_closed, open_recovery = 0, 0
        danger_start_time, yawn_start_time = None, None  # Timers for eyes and yawn
        head_tilt_start_time = None  # Timer for head tilt


        # Landmark indices for head pose tilt approx
        LEFT_EYE_IDX = 33
        RIGHT_EYE_IDX = 263


        with mp_face.FaceMesh(
            refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        ) as face_mesh:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Camera not detected.")
                    break


                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)


                # defaults
                yawn_detected = False
                eye_blink_detected = False
                elapsed_danger = 0
                elapsed_yawn = 0
                elapsed_tilt = 0


                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark


                    def ratio(idxs):
                        pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idxs])
                        A = np.linalg.norm(pts[1]-pts[5])
                        B = np.linalg.norm(pts[2]-pts[4])
                        C = np.linalg.norm(pts[0]-pts[3]) + 1e-8
                        return (A+B)/(2*C)


                    LEFT = [362,385,387,263,373,380]
                    RIGHT = [33,160,158,133,153,144]
                    MOUTH = [78,308,13,14,87,317]
                    ear = np.mean([ratio(LEFT), ratio(RIGHT)])
                    mar = ratio(MOUTH)


                    # --- Yawn and Eye Prediction ---
                    eye_state = int(eye_model.predict([[ear]])[0])
                    yawn_state = int(yawn_model.predict([[mar]])[0])


                    # --- Stability control for yawn ---
                    YAWN_THRESHOLD = 0.58
                    yawn_detected_raw = (mar > YAWN_THRESHOLD)


                    # --- Eye blink handling (for status + timer) ---
                    # eye_state == 0 -> closed by your model
                    eye_blink_detected = (eye_state == 0)
                    if eye_blink_detected:
                        consecutive_closed += 1
                        open_recovery = 0
                    else:
                        consecutive_closed = 0
                        open_recovery += 1


                    is_danger = (consecutive_closed >= CRITICAL_EYE_CLOSE_FRAMES)


                    # --- Drowsy / Danger timers (eye + yawn) ---
                    now = time.time()

                    # Eye timer
                    if eye_blink_detected:
                        if danger_start_time is None:
                            danger_start_time = now
                        elapsed_danger = now - danger_start_time
                    else:
                        danger_start_time = None
                        elapsed_danger = 0

                    # Yawn timer
                    yawn_detected = (yawn_state == 1 and yawn_detected_raw)
                    if yawn_detected:
                        if yawn_start_time is None:
                            yawn_start_time = now
                        elapsed_yawn = now - yawn_start_time
                    else:
                        yawn_start_time = None
                        elapsed_yawn = 0


                    # --- Head tilt detection ---
                    left_eye = np.array([lm[LEFT_EYE_IDX].x * w, lm[LEFT_EYE_IDX].y * h])
                    right_eye = np.array([lm[RIGHT_EYE_IDX].x * w, lm[RIGHT_EYE_IDX].y * h])
                    eye_vector = right_eye - left_eye
                    angle_rad = np.arctan2(eye_vector[1], eye_vector[0])
                    angle_deg = np.degrees(angle_rad)

                    head_tilted = abs(angle_deg) > HEAD_TILT_THRESHOLD

                    if head_tilted:
                        if head_tilt_start_time is None:
                            head_tilt_start_time = now
                        elapsed_tilt = now - head_tilt_start_time
                    else:
                        head_tilt_start_time = None
                        elapsed_tilt = 0


                    # --- Display status (overall) ---
                    if is_danger:
                        status_text = "STATUS: DANGER: IMMEDIATE ALERT"
                        status_color = (0, 0, 255)
                        cv2.putText(frame, "WAKE UP! PULL OVER!", (80, int(h / 2)),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 4)
                    elif consecutive_closed > 0 and consecutive_closed < CRITICAL_EYE_CLOSE_FRAMES:
                        status_text = "STATUS: STRAINING (LOW BLINKS)"
                        status_color = (0, 165, 255)
                    else:
                        status_text = "STATUS: NORMAL (ATTENTIVE)"
                        status_color = (0, 255, 0)


                    cv2.putText(frame, status_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (200, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


                    # --- EYE BLINK status + timer overlay ---
                    cv2.putText(
                        frame,
                        f"EYE BLINK: {'YES' if eye_blink_detected else 'NO'}",
                        (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255) if eye_blink_detected else (0, 255, 0),
                        2,
                    )
                    if elapsed_danger > 0:
                        cv2.putText(
                            frame,
                            f"EYE TIMER: {elapsed_danger:.1f}s",
                            (10, 165),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255) if elapsed_danger >= DANGER_DELAY else (0, 255, 255),
                            2,
                        )


                    # --- YAWN status + timer overlay ---
                    cv2.putText(
                        frame,
                        f"YAWN: {'YES' if yawn_detected else 'NO'}",
                        (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255) if yawn_detected else (0, 255, 0),
                        2,
                    )
                    if elapsed_yawn > 0:
                        cv2.putText(
                            frame,
                            f"YAWN TIMER: {elapsed_yawn:.1f}s",
                            (10, 225),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255) if elapsed_yawn >= YAWN_DELAY else (0, 255, 255),
                            2,
                        )


                    # --- Head Tilt Display + timer ---
                    if head_tilted:
                        cv2.putText(frame, f"HEAD TILT: YES ({angle_deg:.1f} deg)", (10, 255),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "HEAD TILT: NO", (10, 255),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if elapsed_tilt > 0:
                        cv2.putText(
                            frame,
                            f"TILT TIMER: {elapsed_tilt:.1f}s",
                            (10, 285),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255) if elapsed_tilt >= HEAD_TILT_DELAY else (0, 255, 255),
                            2,
                        )


                    # --- Alarm triggering for sustained conditions ---
                    # Eye blink sustained > 3s  -> Possible eye blink fatigue
                    # Yawn sustained > 3s      -> Possible yawn
                    # Head tilt sustained > 3s -> Head tilt alert
                    if (
                        (eye_blink_detected and elapsed_danger >= DANGER_DELAY)
                        or (yawn_detected and elapsed_yawn >= YAWN_DELAY)
                        or (head_tilted and elapsed_tilt >= HEAD_TILT_DELAY)
                    ):
                        last_audio_alert_time = play_alarm_non_blocking(
                            ALARM_SOUND_PATH, last_audio_alert_time, ALARM_COOLDOWN
                        )
                        # specific status messages
                        if eye_blink_detected and elapsed_danger >= DANGER_DELAY:
                            cv2.putText(frame, "⚠ POSSIBLE EYE BLINK FATIGUE", (50, int(h * 0.82)),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
                        if yawn_detected and elapsed_yawn >= YAWN_DELAY:
                            cv2.putText(frame, "⚠ POSSIBLE YAWN", (50, int(h * 0.88)),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
                        if head_tilted and elapsed_tilt >= HEAD_TILT_DELAY:
                            cv2.putText(frame, "⚠ ALERT: HEAD TILT DETECTED!", (50, int(h * 0.94)),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
                    elif eye_blink_detected or yawn_detected or head_tilted:
                        cv2.putText(frame, "⚠ WARNING: POSSIBLE FATIGUE", (50, int(h * 0.9)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)


                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


            cap.release()
            st.success("🟢 Detection stopped successfully.")

# ------------------- END OF FILE -------------------
