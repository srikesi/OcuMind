"""
gaze.py — Webcam-based iris gaze estimation using MediaPipe 0.10.x Tasks API.

mediapipe 0.10+ removed mp.solutions. We now use the Tasks API with
FaceLandmarker in LIVE_STREAM mode.

On first run, the model file (~12 MB) is downloaded automatically.
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


# ── Landmark indices (same as classic face mesh) ───────────────────
LEFT_IRIS   = [468, 469, 470, 471, 472]
RIGHT_IRIS  = [473, 474, 475, 476, 477]

LEFT_EYE_LEFT_CORNER   = 33
LEFT_EYE_RIGHT_CORNER  = 133
RIGHT_EYE_LEFT_CORNER  = 362
RIGHT_EYE_RIGHT_CORNER = 263

LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOTTOM = 374


class GazeTracker:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap     = None
        self.running = False
        self._thread = None
        self._lock   = threading.Lock()
        self._landmarker = None

        self._gaze = {
            "x": 0.5, "y": 0.5,
            "raw_x": 0.5, "raw_y": 0.5,
            "detected": False,
            "timestamp": 0.0,
        }
        self._smooth_buf = []
        self._smooth_n   = 3

    def start(self):
        _ensure_model()
        self._build_landmarker()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def get_gaze(self) -> dict:
        with self._lock:
            return dict(self._gaze)

    def _build_landmarker(self):
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    def _capture_loop(self):
        ts = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts    += 33
            self._landmarker.detect_async(mp_img, ts)

    def _on_result(self, result, output_image, timestamp_ms):
        if not result.face_landmarks:
            with self._lock:
                self._gaze["detected"] = False
            return
        lm   = result.face_landmarks[0]
        gaze = self._estimate_gaze(lm)
        gaze["detected"]  = True
        gaze["timestamp"] = time.time()
        with self._lock:
            self._gaze = gaze

    def _estimate_gaze(self, lm) -> dict:
        def iris_center(indices):
            return (float(np.mean([lm[i].x for i in indices])),
                    float(np.mean([lm[i].y for i in indices])))

        li_x, li_y = iris_center(LEFT_IRIS)
        ri_x, ri_y = iris_center(RIGHT_IRIS)

        l_h = (li_x - lm[LEFT_EYE_LEFT_CORNER].x) / (abs(lm[LEFT_EYE_RIGHT_CORNER].x - lm[LEFT_EYE_LEFT_CORNER].x) + 1e-9)
        r_h = (ri_x - lm[RIGHT_EYE_LEFT_CORNER].x) / (abs(lm[RIGHT_EYE_RIGHT_CORNER].x - lm[RIGHT_EYE_LEFT_CORNER].x) + 1e-9)
        avg_h = (l_h + r_h) / 2.0

        l_v = (li_y - lm[LEFT_EYE_TOP].y) / (abs(lm[LEFT_EYE_BOTTOM].y - lm[LEFT_EYE_TOP].y) + 1e-9)
        r_v = (ri_y - lm[RIGHT_EYE_TOP].y) / (abs(lm[RIGHT_EYE_BOTTOM].y - lm[RIGHT_EYE_TOP].y) + 1e-9)
        avg_v = (l_v + r_v) / 2.0

        self._smooth_buf.append((avg_h, avg_v))
        if len(self._smooth_buf) > self._smooth_n:
            self._smooth_buf.pop(0)
        avg_h = float(np.mean([p[0] for p in self._smooth_buf]))
        avg_v = float(np.mean([p[1] for p in self._smooth_buf]))

        return {
            "x":     float(np.clip(avg_h, 0.0, 1.0)),
            "y":     float(np.clip(avg_v, 0.0, 1.0)),
            "raw_x": float((li_x + ri_x) / 2.0),
            "raw_y": float((li_y + ri_y) / 2.0),
        }