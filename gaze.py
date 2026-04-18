"""
gaze.py — Webcam gaze estimation with MediaPipe Tasks API + OpenCV fallback.

Strategy:
  1. Try MediaPipe FaceLandmarker (iris-level accuracy)
  2. If iris landmarks unavailable, fall back to face bounding-box center
  3. If MediaPipe fails entirely, fall back to OpenCV Haar eye detection
"""

import cv2
import numpy as np
import threading
import time
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model ──────────────────────────────────────────────────────────
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[OcuMind] Downloading FaceLandmarker model (~12 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[OcuMind] Model downloaded.")

# ── Landmark indices ───────────────────────────────────────────────
LEFT_IRIS            = [468, 469, 470, 471, 472]
RIGHT_IRIS           = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS     = [33, 133]
RIGHT_EYE_CORNERS    = [362, 263]
LEFT_EYE_TB          = [159, 145]
RIGHT_EYE_TB         = [386, 374]
# Face bounding landmarks (fallback when iris not available)
FACE_OVAL            = [10, 338, 297, 332, 284, 251, 389, 356,
                        454, 323, 361, 288, 397, 365, 379, 378]


class GazeTracker:

    def __init__(self, camera_index: int = 0):
        self.camera_index  = camera_index
        self.cap           = None
        self.running       = False
        self._thread       = None
        self._lock         = threading.Lock()
        self._landmarker   = None
        self._mp_ok        = False      # did MediaPipe init succeed?
        self._haar_face    = None
        self._haar_eye     = None
        self._frame_count  = 0
        self._detect_count = 0

        self._gaze = {
            "x": 0.5, "y": 0.5,
            "raw_x": 0.5, "raw_y": 0.5,
            "detected": False,
            "timestamp": 0.0,
            "method": "none",
        }
        self._smooth_buf = []
        self._smooth_n   = 4

    # ── Public API ─────────────────────────────────────────────────

    def start(self):
        self._init_mediapipe()
        self._init_haar()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[OcuMind] GazeTracker started (MediaPipe: {self._mp_ok})")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def get_gaze(self) -> dict:
        with self._lock:
            return dict(self._gaze)

    # ── Init ────────────────────────────────────────────────────────

    def _init_mediapipe(self):
        try:
            _ensure_model()
            base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                result_callback=self._on_result,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
            self._mp_ok = True
        except Exception as e:
            print(f"[OcuMind] MediaPipe init failed: {e}")
            self._mp_ok = False

    def _init_haar(self):
        """Load OpenCV Haar cascades as fallback."""
        try:
            cv2_data = cv2.data.haarcascades
            self._haar_face = cv2.CascadeClassifier(cv2_data + "haarcascade_frontalface_default.xml")
            self._haar_eye  = cv2.CascadeClassifier(cv2_data + "haarcascade_eye.xml")
            print("[OcuMind] Haar cascades loaded (fallback ready)")
        except Exception as e:
            print(f"[OcuMind] Haar load failed: {e}")

    # ── Capture loop ────────────────────────────────────────────────

    def _capture_loop(self):
        last_log = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            self._frame_count += 1

            # Log detection rate every 5 s
            now = time.time()
            if now - last_log > 5:
                rate = self._detect_count / max(1, self._frame_count) * 100
                print(f"[OcuMind] Detection rate: {rate:.0f}%  "
                      f"({self._detect_count}/{self._frame_count} frames)")
                self._frame_count  = 0
                self._detect_count = 0
                last_log = now

            if self._mp_ok:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts     = int(time.time() * 1000)   # real wall-clock ms
                try:
                    self._landmarker.detect_async(mp_img, ts)
                except Exception as e:
                    # Timestamp must be strictly increasing; skip if duplicate
                    pass
            else:
                # Pure OpenCV fallback
                gaze = self._haar_estimate(frame)
                if gaze:
                    gaze["detected"]  = True
                    gaze["timestamp"] = time.time()
                    gaze["method"]    = "haar"
                    self._detect_count += 1
                    with self._lock:
                        self._gaze = gaze
                else:
                    with self._lock:
                        self._gaze["detected"] = False

            time.sleep(1 / 30)

    # ── MediaPipe callback ──────────────────────────────────────────

    def _on_result(self, result, output_image, timestamp_ms):
        try:
            if not result.face_landmarks:
                with self._lock:
                    self._gaze["detected"] = False
                return

            lm = result.face_landmarks[0]
            n  = len(lm)

            if n >= 478:
                # Full iris landmarks available
                gaze = self._iris_gaze(lm)
                gaze["method"] = "iris"
            elif n >= 468:
                # Face mesh without iris — use eye corner midpoints
                gaze = self._eye_corner_gaze(lm)
                gaze["method"] = "eye_corners"
            else:
                # Minimal landmarks — use face centroid
                gaze = self._face_centroid_gaze(lm)
                gaze["method"] = "centroid"

            gaze["detected"]  = True
            gaze["timestamp"] = time.time()
            self._detect_count += 1

            with self._lock:
                self._gaze = gaze

        except Exception as e:
            print(f"[OcuMind] Callback error: {e}")
            with self._lock:
                self._gaze["detected"] = False

    # ── Gaze estimation methods ─────────────────────────────────────

    def _iris_gaze(self, lm) -> dict:
        def center(idxs):
            return (np.mean([lm[i].x for i in idxs]),
                    np.mean([lm[i].y for i in idxs]))

        li_x, li_y = center(LEFT_IRIS)
        ri_x, ri_y = center(RIGHT_IRIS)

        # Horizontal: iris within eye
        l_h = (li_x - lm[LEFT_EYE_CORNERS[0]].x) / (
            abs(lm[LEFT_EYE_CORNERS[1]].x - lm[LEFT_EYE_CORNERS[0]].x) + 1e-9)
        r_h = (ri_x - lm[RIGHT_EYE_CORNERS[0]].x) / (
            abs(lm[RIGHT_EYE_CORNERS[1]].x - lm[RIGHT_EYE_CORNERS[0]].x) + 1e-9)
        avg_h = (l_h + r_h) / 2.0

        # Vertical: iris within eyelid span
        l_v = (li_y - lm[LEFT_EYE_TB[0]].y) / (
            abs(lm[LEFT_EYE_TB[1]].y - lm[LEFT_EYE_TB[0]].y) + 1e-9)
        r_v = (ri_y - lm[RIGHT_EYE_TB[0]].y) / (
            abs(lm[RIGHT_EYE_TB[1]].y - lm[RIGHT_EYE_TB[0]].y) + 1e-9)
        avg_v = (l_v + r_v) / 2.0

        return self._smoothed(avg_h, avg_v,
                              (li_x + ri_x) / 2, (li_y + ri_y) / 2)

    def _eye_corner_gaze(self, lm) -> dict:
        """Use midpoint between eye corners as proxy for gaze."""
        lx = (lm[LEFT_EYE_CORNERS[0]].x + lm[LEFT_EYE_CORNERS[1]].x) / 2
        ly = (lm[LEFT_EYE_CORNERS[0]].y + lm[LEFT_EYE_CORNERS[1]].y) / 2
        rx = (lm[RIGHT_EYE_CORNERS[0]].x + lm[RIGHT_EYE_CORNERS[1]].x) / 2
        ry = (lm[RIGHT_EYE_CORNERS[0]].y + lm[RIGHT_EYE_CORNERS[1]].y) / 2
        mx, my = (lx + rx) / 2, (ly + ry) / 2
        return self._smoothed(mx, my, mx, my)

    def _face_centroid_gaze(self, lm) -> dict:
        """Use face centroid as very rough gaze proxy."""
        xs = [lm[i].x for i in range(min(len(lm), 20))]
        ys = [lm[i].y for i in range(min(len(lm), 20))]
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        return self._smoothed(mx, my, mx, my)

    def _haar_estimate(self, frame) -> dict | None:
        """OpenCV Haar fallback — returns None if no face found."""
        if self._haar_face is None:
            return None
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w  = frame.shape[:2]
        faces = self._haar_face.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if not len(faces):
            return None
        fx, fy, fw, fh = faces[0]
        # Detect eyes within face ROI
        roi   = gray[fy:fy + fh, fx:fx + fw]
        eyes  = self._haar_eye.detectMultiScale(roi, 1.05, 3)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            ex = np.mean([e[0] + e[2] / 2 for e in eyes]) + fx
            ey = np.mean([e[1] + e[3] / 2 for e in eyes]) + fy
        else:
            ex = fx + fw / 2
            ey = fy + fh * 0.35       # approximate eye region
        nx, ny = float(ex / w), float(ey / h)
        return self._smoothed(nx, ny, nx, ny)

    def _smoothed(self, x, y, raw_x, raw_y) -> dict:
        self._smooth_buf.append((x, y))
        if len(self._smooth_buf) > self._smooth_n:
            self._smooth_buf.pop(0)
        sx = float(np.mean([p[0] for p in self._smooth_buf]))
        sy = float(np.mean([p[1] for p in self._smooth_buf]))
        return {
            "x":     float(np.clip(sx, 0.0, 1.0)),
            "y":     float(np.clip(sy, 0.0, 1.0)),
            "raw_x": float(raw_x),
            "raw_y": float(raw_y),
        }