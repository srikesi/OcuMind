"""
app.py — OcuMind Flask + SocketIO server.

Run with:  python app.py
Then open  http://localhost:5050
"""

import json
import time
import threading

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit

from gaze        import GazeTracker
from calibration import CalibrationManager
from metrics     import MetricsEngine
from session     import SessionManager
from insights    import InsightEngine


# ------------------------------------------------------------------ #
#  App initialisation                                                  #
# ------------------------------------------------------------------ #

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

tracker  = GazeTracker()
calib    = CalibrationManager()
engine   = MetricsEngine()
session  = SessionManager()
insights = InsightEngine()

# Global state
_state = {
    "streaming":  False,
    "in_session": False,
    "session_mode": "circular",
    "calibrated": False,
    "calib_collecting": False,
    "calib_point": None,
    "calib_buf":  [],
    "calib_buf_target": 50,
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "calibrated": _state["calibrated"],
        "in_session": _state["in_session"],
        "streaming":  _state["streaming"],
        "camera_ok":  tracker.cap is not None and tracker.cap.isOpened()
                      if tracker.cap else False,
    })

@app.route("/api/sessions")
def list_sessions():
    return jsonify(session.list_sessions())

@app.route("/api/export/<session_id>/<fmt>")
def export_session(session_id, fmt):
    if fmt == "json":
        path = session.export_json()
    else:
        path = session.export_csv()
    return send_file(path, as_attachment=True)

@socketio.on("connect")
def on_connect():
    if not _state["streaming"]:
        _start_gaze_stream()

@socketio.on("disconnect")
def on_disconnect():
    pass

@socketio.on("start_camera")
def on_start_camera():
    if not (tracker.cap and tracker.cap.isOpened()):
        try:
            tracker.start()
            emit("camera_ready", {"ok": True})
        except Exception as e:
            emit("camera_ready", {"ok": False, "error": str(e)})
    else:
        emit("camera_ready", {"ok": True})

    if not _state["streaming"]:
        _start_gaze_stream()

@socketio.on("stop_camera")
def on_stop_camera():
    tracker.stop()
    _state["streaming"] = False

# ---- Calibration ---- #

@socketio.on("calib_start_point")
def on_calib_start_point(data):
    _state["calib_collecting"] = True
    _state["calib_point"]      = (data["x"], data["y"])
    _state["calib_buf"]        = []

@socketio.on("calib_commit_point")
def on_calib_commit_point():
    import numpy as np
    buf = _state["calib_buf"]
    if buf and len(buf) >= 5:
        arr = np.array(buf)
        med = np.median(arr, axis=0)
        dists = np.linalg.norm(arr - med, axis=1)
        mad = np.median(dists) + 1e-6
        keep = arr[dists < 2.5 * mad]
        if len(keep) < 3:
            keep = arr
        rx, ry = float(np.mean(keep[:, 0])), float(np.mean(keep[:, 1]))
        sx, sy = _state["calib_point"]
        calib.add_sample(sx, sy, rx, ry)
    _state["calib_collecting"] = False
    _state["calib_buf"]        = []
    emit("calib_point_done", {"n": calib.sample_count()})

@socketio.on("calib_finish")
def on_calib_finish():
    ok = calib.fit()
    _state["calibrated"] = ok
    emit("calib_result", {"ok": ok, "n_samples": calib.sample_count()})

@socketio.on("calib_reset")
def on_calib_reset():
    calib.reset()
    _state["calibrated"] = False
    emit("calib_reset_done", {})

# ---- Training session ---- #

@socketio.on("session_start")
def on_session_start(data):
    mode = data.get("mode", "circular")
    _state["in_session"]    = True
    _state["session_mode"]  = mode
    engine.reset()
    session.start(mode)
    emit("session_started", {"mode": mode})

@socketio.on("session_frame")
def on_session_frame(data):
    if not _state["in_session"]:
        return

    gaze = tracker.get_gaze()
    if not gaze["detected"]:
        return

    if _state["calibrated"]:
        cx, cy = calib.map(gaze["x"], gaze["y"])
    else:
        cx, cy = gaze["x"], gaze["y"]

    tx = data.get("target_x", 0.5)
    ty = data.get("target_y", 0.5)
    moving = data.get("moving", True)

    engine.update(cx, cy, tx, ty, target_moving=moving)
    snap = engine.snapshot()
    session.record_frame(cx, cy, tx, ty, snap)

    emit("metrics_update", {
        "gaze_x":  round(cx, 4),
        "gaze_y":  round(cy, 4),
        **snap,
    })

@socketio.on("session_stop")
def on_session_stop():
    _state["in_session"] = False
    summary = session.finish()
    
    # Run comprehensive holistic clinical analysis
    clinical_data = insights.analyze_session(session.frames, session.mode)
    
    if "error" not in clinical_data:
        summary["clinical_report"] = clinical_data["report"]
        # REMOVED: summary["clinical_flags"] = clinical_data["flags"]
        
    emit("session_summary", summary)

def _gaze_broadcast():
    _state["streaming"] = True
    while _state["streaming"]:
        gaze = tracker.get_gaze()

        if _state["calib_collecting"] and gaze["detected"]:
            buf = _state["calib_buf"]
            if len(buf) < _state["calib_buf_target"]:
                buf.append((gaze["x"], gaze["y"]))
            else:
                socketio.emit("calib_auto_commit", {"ready": True})

        if gaze["detected"]:
            if _state["calibrated"]:
                cx, cy = calib.map(gaze["x"], gaze["y"])
            else:
                cx, cy = gaze["x"], gaze["y"]
            socketio.emit("gaze_raw", {
                "x": round(cx, 4),
                "y": round(cy, 4),
                "detected": True,
            })
        else:
            socketio.emit("gaze_raw", {"detected": False})

        time.sleep(1 / 30)

def _start_gaze_stream():
    if not (tracker.cap and tracker.cap.isOpened()):
        tracker.start()
    socketio.start_background_task(_gaze_broadcast)

if __name__ == "__main__":
    print("=" * 55)
    print("  OcuMind — Eye Tracking Rehabilitation App")
    print("  http://localhost:5050")
    print("=" * 55)
    tracker.start()
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)