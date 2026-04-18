"""
gaze.py — Webcam gaze estimation.

Design:
  - Use raw iris centre position in image space as the gaze signal.
    It's the most stable/low-noise signal MediaPipe gives us.
  - Apply one-euro-style smoothing (low-pass when still, responsive
    when moving) to kill jitter without adding sluggishness.
  - Track head position separately so the frontend can warn the user
    when they drift far from their calibration pose.
  - Calibration (polynomial fit) maps raw iris position → screen.
    It inherently encodes "user's head in this pose, iris here = this
    screen point". So calibration is mandatory for accuracy, and the
    user should try to keep their head roughly still.

Fallbacks:
  - If iris landmarks not available: eye-corner midpoint
  - If MediaPipe fails entirely: OpenCV Haar cascade
"""

import cv2
import numpy as np
import threading
import time
import os
import urllib.request
from typing import Optional

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


# Iris landmarks (present when the iris-capable face_landmarker.task is loaded)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye corners — used for fallback gaze and head reference
LEFT_EYE_CORNERS  = [33, 133]     # outer, inner
RIGHT_EYE_CORNERS = [362, 263]    # inner, outer

# Top/bottom eyelid landmarks — used for vertical gaze ratio
LEFT_EYE_TB  = [159, 145]         # top lid, bottom lid
RIGHT_EYE_TB = [386, 374]         # top lid, bottom lid

# Under-eye landmarks (top of cheek) — don't move with blinks/expressions.
# Used as a more stable vertical reference than eyelids.
LEFT_UNDEREYE  = 230              # below left eye
RIGHT_UNDEREYE = 450              # below right eye
# Brow landmarks — upper stable reference
LEFT_BROW      = 52
RIGHT_BROW     = 282

# Nose tip — stable landmark used as head-position reference
NOSE_TIP = 1


# ─────────────────────────────────────────────────────────────────────
#  One-Euro filter — adaptive low-pass.
#  Smooths hard during fixation, stays responsive during saccades.
#  Reference: https://gery.casiez.net/1euro/
# ─────────────────────────────────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, freq: float = 30.0,
                 mincutoff: float = 1.0, beta: float = 0.007,
                 dcutoff: float = 1.0):
        self.freq      = freq
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self._x_prev   = None
        self._dx_prev  = 0.0
        self._t_prev   = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te  = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        dt = max(t - self._t_prev, 1e-6)
        self.freq = 1.0 / dt

        dx     = (x - self._x_prev) * self.freq
        a_d    = self._alpha(self.dcutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a      = self._alpha(cutoff, self.freq)
        x_hat  = a * x + (1 - a) * self._x_prev

        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        self._t_prev  = t
        return x_hat

    def reset(self):
        self._x_prev  = None
        self._dx_prev = 0.0
        self._t_prev  = None


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
        self._last_ts_ms  = 0

        # One-Euro filters. Y gets heavier smoothing (lower mincutoff)
        # because vertical gaze signal is inherently noisier — the eye
        # is physically ~3× narrower vertically than horizontally, so
        # the same landmark jitter is 3× worse in vertical.
        self._filter_x = OneEuroFilter(mincutoff=1.0, beta=0.04)
        self._filter_y = OneEuroFilter(mincutoff=0.6, beta=0.03)

        self._gaze = {
            "x": 0.5, "y": 0.5,
            "raw_x": 0.5, "raw_y": 0.5,
            "detected": False,
            "timestamp": 0.0,
            "method": "none",
            "head_x": 0.5, "head_y": 0.5,
        }

    # ── Public API ──────────────────────────────────────────────────

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

    def reset_filter(self):
        """Clear smoothing state — e.g. between calibration points."""
        self._filter_x.reset()
        self._filter_y.reset()

    # ── Init ────────────────────────────────────────────────────────

    def _init_mediapipe(self):
        try:
            _ensure_model()
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,   # enables head pose
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

            frame = cv2.flip(frame, 1)

            if self._mp_ok:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
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

            # 3D head pose matrix (4x4) if MediaPipe provided it
            tmat = None
            if (result.facial_transformation_matrixes
                    and len(result.facial_transformation_matrixes) > 0):
                tmat = np.array(result.facial_transformation_matrixes[0])

            if n >= 478:
                gaze = self._iris_gaze(lm, tmat)
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

    def _iris_gaze(self, lm, tmat=None) -> dict:
        """
        Hybrid gaze estimation:
          1. Iris-in-eye ratio (head-invariant by construction)
          2. Subtract residual head yaw/pitch from 3D transform matrix
          3. One-Euro filter the result

        The 3D matrix from MediaPipe gives us actual head rotation
        (not just translation). Subtracting a small fraction of the
        yaw/pitch angles cancels the residual head-motion leak that
        ratio-only methods can't fully eliminate.
        """
        def mean_xy(idxs):
            return (float(np.mean([lm[i].x for i in idxs])),
                    float(np.mean([lm[i].y for i in idxs])))

        li_x, li_y = mean_xy(LEFT_IRIS)
        ri_x, ri_y = mean_xy(RIGHT_IRIS)

        # Eye socket corners
        l_outer, l_inner = lm[LEFT_EYE_CORNERS[0]],  lm[LEFT_EYE_CORNERS[1]]
        r_inner, r_outer = lm[RIGHT_EYE_CORNERS[0]], lm[RIGHT_EYE_CORNERS[1]]

        # Iris-in-eye HORIZONTAL ratio (this already works well) — use eye corners
        l_w = abs(l_inner.x - l_outer.x) + 1e-6
        r_w = abs(r_outer.x - r_inner.x) + 1e-6
        l_rx = (li_x - min(l_outer.x, l_inner.x)) / l_w
        r_rx = (ri_x - min(r_inner.x, r_outer.x)) / r_w

        # Iris-in-eye VERTICAL ratio — use BROW to UNDER-EYE instead of
        # eyelids. Eyelids move with blinks/expressions (noisy).
        # Brow and cheek landmarks are stable to expression changes.
        l_brow   = lm[LEFT_BROW]
        r_brow   = lm[RIGHT_BROW]
        l_under  = lm[LEFT_UNDEREYE]
        r_under  = lm[RIGHT_UNDEREYE]

        l_h = abs(l_under.y - l_brow.y) + 1e-6
        r_h = abs(r_under.y - r_brow.y) + 1e-6
        l_ry = (li_y - l_brow.y) / l_h
        r_ry = (ri_y - r_brow.y) / r_h

        ratio_x = (l_rx + r_rx) / 2.0
        ratio_y = (l_ry + r_ry) / 2.0

        # Rescale. Horizontal ratio typically ~0.3–0.7. Vertical ratio
        # with brow-to-cheek reference typically ~0.4–0.7 (iris lives
        # in the upper half of brow-to-cheek span).
        raw_x = (ratio_x - 0.3) / 0.4
        raw_y = (ratio_y - 0.4) / 0.3

        # ── Head pose subtraction using 3D transformation matrix ────
        # Extract yaw and pitch from the rotation part of the 4x4 matrix.
        # Standard rotation matrix → euler angle extraction.
        head_yaw   = 0.0
        head_pitch = 0.0
        if tmat is not None and tmat.shape == (4, 4):
            R = tmat[:3, :3]
            # pitch = rotation around X axis, yaw = rotation around Y axis
            # Using the standard XYZ euler convention
            head_pitch = float(np.arctan2(-R[1, 2], R[2, 2]))  # radians
            head_yaw   = float(np.arctan2(-R[0, 2], np.sqrt(R[1, 2]**2 + R[2, 2]**2)))

        # Subtract head rotation from raw gaze. Pitch affects vertical
        # gaze more strongly than yaw affects horizontal (perspective
        # foreshortening when head tilts up/down), so pitch gets higher gain.
        HEAD_YAW_GAIN   = 1.2
        HEAD_PITCH_GAIN = 2.0
        raw_x += head_yaw   * HEAD_YAW_GAIN
        raw_y += head_pitch * HEAD_PITCH_GAIN

        # Clip the pre-filter value to a sensible range
        raw_x = float(np.clip(raw_x, -0.3, 1.3))
        raw_y = float(np.clip(raw_y, -0.3, 1.3))

        # One-Euro smoothing
        t = time.time()
        sx = float(np.clip(self._filter_x(raw_x, t), 0.0, 1.0))
        sy = float(np.clip(self._filter_y(raw_y, t), 0.0, 1.0))

        head_x = float(lm[NOSE_TIP].x)
        head_y = float(lm[NOSE_TIP].y)

        return {
            "x": sx, "y": sy,
            "raw_x": raw_x, "raw_y": raw_y,
            "head_x": head_x, "head_y": head_y,
            "head_yaw":   head_yaw,
            "head_pitch": head_pitch,
        }

    def _eye_corner_gaze(self, lm) -> dict:
        lx = (lm[LEFT_EYE_CORNERS[0]].x  + lm[LEFT_EYE_CORNERS[1]].x)  / 2
        ly = (lm[LEFT_EYE_CORNERS[0]].y  + lm[LEFT_EYE_CORNERS[1]].y)  / 2
        rx = (lm[RIGHT_EYE_CORNERS[0]].x + lm[RIGHT_EYE_CORNERS[1]].x) / 2
        ry = (lm[RIGHT_EYE_CORNERS[0]].y + lm[RIGHT_EYE_CORNERS[1]].y) / 2
        raw_x = (lx + rx) / 2
        raw_y = (ly + ry) / 2
        t = time.time()
        sx = float(np.clip(self._filter_x(raw_x, t), 0.0, 1.0))
        sy = float(np.clip(self._filter_y(raw_y, t), 0.0, 1.0))
        return {"x": sx, "y": sy, "raw_x": raw_x, "raw_y": raw_y,
                "head_x": raw_x, "head_y": raw_y}

    def _face_centroid(self, lm) -> dict:
        xs = [lm[i].x for i in range(min(len(lm), 50))]
        ys = [lm[i].y for i in range(min(len(lm), 50))]
        raw_x = float(np.mean(xs))
        raw_y = float(np.mean(ys))
        t = time.time()
        sx = float(np.clip(self._filter_x(raw_x, t), 0.0, 1.0))
        sy = float(np.clip(self._filter_y(raw_y, t), 0.0, 1.0))
        return {"x": sx, "y": sy, "raw_x": raw_x, "raw_y": raw_y,
                "head_x": raw_x, "head_y": raw_y}

    def _haar_estimate(self, frame) -> Optional[dict]:
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
        raw_x = float(ex / w)
        raw_y = float(ey / h)
        t = time.time()
        sx = float(np.clip(self._filter_x(raw_x, t), 0.0, 1.0))
        sy = float(np.clip(self._filter_y(raw_y, t), 0.0, 1.0))
        return {"x": sx, "y": sy, "raw_x": raw_x, "raw_y": raw_y,
                "head_x": raw_x, "head_y": raw_y}