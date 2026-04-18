"""
session.py — Session recording and export.
"""

import json
import csv
import os
import time
from typing import List, Optional


SESSION_DIR = os.path.join(os.path.dirname(__file__), "data", "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)


class SessionManager:

    def __init__(self):
        self.session_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.mode: str = "circular"
        self.frames: List[dict] = []
        self.metrics_history: List[dict] = []
        self.calibration_data: Optional[dict] = None

    # ------------------------------------------------------------------ #

    def start(self, mode: str = "circular"):
        self.session_id   = f"session_{int(time.time())}"
        self.start_time   = time.time()
        self.mode         = mode
        self.frames       = []
        self.metrics_history = []

    def record_frame(self, gaze_x: float, gaze_y: float,
                     target_x: float, target_y: float,
                     metrics: dict):
        self.frames.append({
            "t":        round(time.time() - self.start_time, 3),
            "gaze_x":   round(gaze_x,   4),
            "gaze_y":   round(gaze_y,   4),
            "target_x": round(target_x, 4),
            "target_y": round(target_y, 4),
        })
        self.metrics_history.append(metrics)

    def finish(self) -> dict:
        """Finalise session and return summary dict."""
        if not self.metrics_history:
            return {}
        last  = self.metrics_history[-1]
        final = {
            "session_id":  self.session_id,
            "mode":        self.mode,
            "duration_s":  round(time.time() - self.start_time, 1),
            "n_frames":    len(self.frames),
            "final_score": last.get("score", 0),
            "accuracy":    last.get("accuracy", 0),
            "smoothness":  last.get("smoothness", 0),
            "latency_ms":  last.get("latency_ms", 0),
            "stability":   last.get("stability", 0),
        }
        self._save(final)
        return final

    # ------------------------------------------------------------------ #

    def export_json(self) -> str:
        path = os.path.join(SESSION_DIR, f"{self.session_id}.json")
        payload = {
            "meta":    {"session_id": self.session_id, "mode": self.mode,
                        "start_time": self.start_time},
            "calibration": self.calibration_data,
            "frames":  self.frames,
            "metrics": self.metrics_history,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def export_csv(self) -> str:
        path = os.path.join(SESSION_DIR, f"{self.session_id}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t", "gaze_x", "gaze_y",
                                                    "target_x", "target_y"])
            writer.writeheader()
            writer.writerows(self.frames)
        return path

    def list_sessions(self) -> List[dict]:
        sessions = []
        for fname in sorted(os.listdir(SESSION_DIR)):
            if fname.endswith("_summary.json"):
                fpath = os.path.join(SESSION_DIR, fname)
                with open(fpath) as f:
                    sessions.append(json.load(f))
        return sessions[-20:]  # last 20

    # ------------------------------------------------------------------ #

    def _save(self, summary: dict):
        path = os.path.join(SESSION_DIR, f"{self.session_id}_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
