"""
gaze.py — Webcam-based iris gaze estimation using MediaPipe FaceMesh.

Outputs normalized gaze coordinates (x, y) in [0, 1] range,
where (0,0) is top-left and (1,1) is bottom-right of the visual field.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time


# MediaPipe FaceMesh landmark indices (refine_landmarks=True required)
LEFT_IRIS   = [468, 469, 470, 471, 472]   # left iris ring + center
RIGHT_IRIS  = [473, 474, 475, 476, 477]   # right iris ring + center

# Eye corner landmarks (used for eye-relative normalization)
LEFT_EYE_LEFT_CORNER  = 33
LEFT_EYE_RIGHT_CORNER = 133
RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER = 263

# Eyelid top/bottom landmarks (vertical gaze)
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOTTOM = 374


class GazeTracker:
    """
    Runs MediaPipe FaceMesh in a background thread.
    Call start() to begin capture, get_gaze() for the latest estimate.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,   # REQUIRED for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = None
        self.running = False
        self._thread = None
        self._lock = threading.Lock()

        # Latest gaze reading
        self._gaze = {
            "x": 0.5, "y": 0.5,
            "raw_x": 0.5, "raw_y": 0.5,
            "detected": False,
            "timestamp": 0.0,
        }

        # Smoothing buffer
        self._smooth_buf = []
        self._smooth_n = 3          # frames to average

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
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
        """Thread-safe snapshot of latest gaze data."""
        with self._lock:
            return dict(self._gaze)

    def is_camera_available(self) -> bool:
        cap = cv2.VideoCapture(self.camera_index)
        ok = cap.isOpened()
        cap.release()
        return ok

    # ------------------------------------------------------------------ #
    #  Background capture loop                                            #
    # ------------------------------------------------------------------ #

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)          # mirror so left/right are natural
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                gaze = self._estimate_gaze(lm)
                gaze["detected"]  = True
                gaze["timestamp"] = time.time()
            else:
                gaze = {"x": 0.5, "y": 0.5,
                        "raw_x": 0.5, "raw_y": 0.5,
                        "detected": False,
                        "timestamp": time.time()}

            with self._lock:
                self._gaze = gaze

    # ------------------------------------------------------------------ #
    #  Gaze estimation from landmarks                                     #
    # ------------------------------------------------------------------ #

    def _estimate_gaze(self, lm) -> dict:
        """
        Compute normalized gaze (0–1 per axis) from iris + eye corner landmarks.

        Strategy:
          - Compute iris center as the mean of the 5 iris landmarks per eye.
          - Compute the iris position *within* the eye bounding box (horizontal
            & vertical) to get an eye-relative ratio.
          - Average left and right eye estimates.
          - The eye-relative ratio already codes gaze direction in a way that
            is largely invariant to small head translations.
        """

        def iris_center(indices):
            xs = [lm[i].x for i in indices]
            ys = [lm[i].y for i in indices]
            return np.mean(xs), np.mean(ys)

        li_x, li_y = iris_center(LEFT_IRIS)
        ri_x, ri_y = iris_center(RIGHT_IRIS)

        # ---- Horizontal ratio (iris within eye width) ----
        l_left  = lm[LEFT_EYE_LEFT_CORNER].x
        l_right = lm[LEFT_EYE_RIGHT_CORNER].x
        r_left  = lm[RIGHT_EYE_LEFT_CORNER].x
        r_right = lm[RIGHT_EYE_RIGHT_CORNER].x

        l_eye_w = abs(l_right - l_left) + 1e-9
        r_eye_w = abs(r_right - r_left) + 1e-9

        l_h_ratio = (li_x - l_left) / l_eye_w
        r_h_ratio = (ri_x - r_left) / r_eye_w
        avg_h = (l_h_ratio + r_h_ratio) / 2.0

        # ---- Vertical ratio (iris within eye height) ----
        l_top    = lm[LEFT_EYE_TOP].y
        l_bottom = lm[LEFT_EYE_BOTTOM].y
        r_top    = lm[RIGHT_EYE_TOP].y
        r_bottom = lm[RIGHT_EYE_BOTTOM].y

        l_eye_h = abs(l_bottom - l_top) + 1e-9
        r_eye_h = abs(r_bottom - r_top) + 1e-9

        l_v_ratio = (li_y - l_top) / l_eye_h
        r_v_ratio = (ri_y - r_top) / r_eye_h
        avg_v = (l_v_ratio + r_v_ratio) / 2.0

        # Smooth over recent frames
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
