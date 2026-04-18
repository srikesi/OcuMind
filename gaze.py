"""
gaze.py — Webcam gaze estimation.

Uses RAW iris position in camera-image space (not iris-within-eye ratio).
Raw image coords give full X and Y movement. Calibration maps the
raw range → screen space.
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

# Iris landmark indices (require refine_landmarks / iris-capable model)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Fallback: eye-corner midpoints when iris not in model
LEFT_EYE_CORNERS  = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]


class GazeTracker:

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap          = None
        self.running      = False
        self._thread      = None
        self._lock        = threading.Lock()
        self._landmarker  = None
        self._mp_ok       = False
        self._haar_face   = None
        self._haar_eye    = None
        self._last_ts_ms  = 0          # enforce strictly increasing timestamps

        self._gaze = {
            "x": 0.5, "y": 0.5,
            "raw_x": 0.5, "raw_y": 0.5,
            "detected": False,
            "timestamp": 0.0,
            "method": "none",
        }
        self._smooth_buf = []
        self._smooth_n   = 4

    # ── Public ──────────────────────────────────────────────────────

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
        print(f"[OcuMind] GazeTracker started  mediapipe={self._mp_ok}")

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
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
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
        try:
            d = cv2.data.haarcascades
            self._haar_face = cv2.CascadeClassifier(d + "haarcascade_frontalface_default.xml")
            self._haar_eye  = cv2.CascadeClassifier(d + "haarcascade_eye.xml")
        except Exception:
            pass

    # ── Capture loop ────────────────────────────────────────────────

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)   # mirror so left/right feel natural

            if self._mp_ok:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Strictly increasing timestamp required by LIVE_STREAM
                ts = int(time.time() * 1000)
                if ts <= self._last_ts_ms:
                    ts = self._last_ts_ms + 1
                self._last_ts_ms = ts

                try:
                    self._landmarker.detect_async(mp_img, ts)
                except Exception:
                    pass
            else:
                gaze = self._haar_estimate(frame)
                if gaze:
                    gaze.update({"detected": True, "timestamp": time.time(), "method": "haar"})
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
                gaze = self._raw_iris_gaze(lm)
                gaze["method"] = "iris"
            elif n >= 468:
                gaze = self._eye_corner_gaze(lm)
                gaze["method"] = "eye_corners"
            else:
                gaze = self._face_centroid(lm)
                gaze["method"] = "centroid"

            gaze.update({"detected": True, "timestamp": time.time()})
            with self._lock:
                self._gaze = gaze

        except Exception as e:
            print(f"[OcuMind] callback error: {e}")

    # ── Estimation methods ──────────────────────────────────────────

    def _raw_iris_gaze(self, lm) -> dict:
        """
        Use raw iris center position in image space (0-1 per axis).

        WHY raw coords instead of iris-within-eye ratio:
        - Iris-within-eye vertical range is ~10 px → noisy, eyelid-dependent
        - Raw image Y moves ~100 px between looking up vs down → reliable
        - Calibration maps the raw range to screen space
        """
        def center(idxs):
            return (float(np.mean([lm[i].x for i in idxs])),
                    float(np.mean([lm[i].y for i in idxs])))

        li_x, li_y = center(LEFT_IRIS)
        ri_x, ri_y = center(RIGHT_IRIS)

        raw_x = (li_x + ri_x) / 2.0
        raw_y = (li_y + ri_y) / 2.0

        return self._smoothed(raw_x, raw_y)

    def _eye_corner_gaze(self, lm) -> dict:
        """Midpoint of eye corners — fallback when iris landmarks absent."""
        lx = (lm[LEFT_EYE_CORNERS[0]].x  + lm[LEFT_EYE_CORNERS[1]].x)  / 2
        ly = (lm[LEFT_EYE_CORNERS[0]].y  + lm[LEFT_EYE_CORNERS[1]].y)  / 2
        rx = (lm[RIGHT_EYE_CORNERS[0]].x + lm[RIGHT_EYE_CORNERS[1]].x) / 2
        ry = (lm[RIGHT_EYE_CORNERS[0]].y + lm[RIGHT_EYE_CORNERS[1]].y) / 2
        return self._smoothed((lx + rx) / 2, (ly + ry) / 2)

    def _face_centroid(self, lm) -> dict:
        xs = [lm[i].x for i in range(min(len(lm), 50))]
        ys = [lm[i].y for i in range(min(len(lm), 50))]
        return self._smoothed(float(np.mean(xs)), float(np.mean(ys)))

    def _haar_estimate(self, frame) -> dict | None:
        if self._haar_face is None:
            return None
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w  = frame.shape[:2]
        faces = self._haar_face.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if not len(faces):
            return None
        fx, fy, fw, fh = faces[0]
        roi  = gray[fy:fy + fh, fx:fx + fw]
        eyes = self._haar_eye.detectMultiScale(roi, 1.05, 3)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            ex   = np.mean([e[0] + e[2] / 2 for e in eyes]) + fx
            ey   = np.mean([e[1] + e[3] / 2 for e in eyes]) + fy
        else:
            ex = fx + fw / 2
            ey = fy + fh * 0.35
        return self._smoothed(float(ex / w), float(ey / h))

    def _smoothed(self, x: float, y: float) -> dict:
        self._smooth_buf.append((x, y))
        if len(self._smooth_buf) > self._smooth_n:
            self._smooth_buf.pop(0)
        sx = float(np.clip(np.mean([p[0] for p in self._smooth_buf]), 0, 1))
        sy = float(np.clip(np.mean([p[1] for p in self._smooth_buf]), 0, 1))
        return {"x": sx, "y": sy, "raw_x": x, "raw_y": y}