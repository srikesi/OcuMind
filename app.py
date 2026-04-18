"""
app.py — OcuMind Flask + SocketIO server.

Run with:  python app.py
Then open  http://localhost:5050
"""

import eventlet
eventlet.monkey_patch()          # must be FIRST

import json
import time
import threading

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit

from gaze        import GazeTracker
from calibration import CalibrationManager
from metrics     import MetricsEngine
from session     import SessionManager


# ------------------------------------------------------------------ #
#  App initialisation                                                  #
# ------------------------------------------------------------------ #

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

tracker  = GazeTracker()
calib    = CalibrationManager()
engine   = MetricsEngine()
session  = SessionManager()

# Global state
_state = {
    "streaming":  False,   # gaze broadcast loop running
    "in_session": False,   # training session active
    "session_mode": "circular",
    "calibrated": False,
    "calib_collecting": False,   # currently collecting a calib point
    "calib_point": None,         # (screen_x, screen_y) of current point
    "calib_buf":  [],            # raw gaze samples for current point
    "calib_buf_target": 30,      # samples to collect per point
}

# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  SocketIO events                                                     #
# ------------------------------------------------------------------ #

@socketio.on("connect")
def on_connect():
    print("[SocketIO] Client connected")
    if not _state["streaming"]:
        _start_gaze_stream()


@socketio.on("disconnect")
def on_disconnect():
    print("[SocketIO] Client disconnected")


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
    """
    Frontend says 'user is now looking at screen position (sx, sy)'.
    We collect ~30 gaze frames then average them.
    """
    _state["calib_collecting"] = True
    _state["calib_point"]      = (data["x"], data["y"])
    _state["calib_buf"]        = []


@socketio.on("calib_commit_point")
def on_calib_commit_point():
    """Average collected samples and add to CalibrationManager."""
    buf = _state["calib_buf"]
    if buf:
        rx = sum(p[0] for p in buf) / len(buf)
        ry = sum(p[1] for p in buf) / len(buf)
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
    """
    Frontend sends current target position each animation frame.
    We fuse with latest gaze, compute metrics, and emit back.
    """
    if not _state["in_session"]:
        return

    gaze = tracker.get_gaze()
    if not gaze["detected"]:
        return

    # Apply calibration mapping
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
    emit("session_summary", summary)


# ------------------------------------------------------------------ #
#  Background gaze broadcast loop                                      #
# ------------------------------------------------------------------ #

def _gaze_broadcast():
    """
    Runs in a background greenlet.
    Broadcasts raw gaze ~30 times/sec so the frontend can render
    the gaze dot even outside a training session (calibration, idle).
    """
    _state["streaming"] = True
    while _state["streaming"]:
        gaze = tracker.get_gaze()

        # Feed calibration buffer if active
        if _state["calib_collecting"] and gaze["detected"]:
            buf = _state["calib_buf"]
            if len(buf) < _state["calib_buf_target"]:
                buf.append((gaze["x"], gaze["y"]))
            else:
                # Auto-commit when buffer full
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

        eventlet.sleep(1 / 30)   # ~30 Hz


def _start_gaze_stream():
    if not (tracker.cap and tracker.cap.isOpened()):
        tracker.start()
    socketio.start_background_task(_gaze_broadcast)


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 55)
    print("  OcuMind — Eye Tracking Rehabilitation App")
    print("  http://localhost:5050")
    print("=" * 55)
    tracker.start()
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)
