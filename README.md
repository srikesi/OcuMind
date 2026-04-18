# 👁 OcuMind — Eye Tracking Rehabilitation App

Real-time webcam-based eye tracking for visuomotor training, smooth pursuit analysis, and rehabilitation-style visual control exercises.

---

## 🖥 Requirements
- **macOS** (tested on MacBook M1/M2/Intel)
- Python 3.10–3.12
- Webcam (built-in or external)
- Chrome or Firefox (Safari may have WebSocket issues)

---

## ⚡️ Quick Start

### 1. Clone / download the project
```bash
cd ~/Desktop
# If you downloaded a zip, just unzip it. Otherwise:
git clone <repo-url> ocumind
cd ocumind
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First install takes ~2–3 minutes (MediaPipe is large).

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5050
```

> **Allow camera access** when the browser asks — this is required.

---

## 🎯 How to Use

### Calibration (recommended)
1. Click **Calibrate** on the welcome screen
2. A dot will appear at 9 positions across the screen
3. Look at each dot and hold still while the countdown runs
4. After all 9 points, calibration fits automatically
5. Session starts automatically

### Skip calibration
Click **Skip (demo)** to jump straight to training with raw uncalibrated gaze.
Accuracy metrics will be lower but the exercise still runs.

### Exercises
| Mode | Description |
|------|------------|
| 🔵 Circular | Smooth pursuit in a circle — good for Parkinson's / MS |
| ➡ Linear sweep | Horizontal saccade training |
| ∞ Figure-8 | Binocular coordination training |
| 🎲 Random | Fixation and attention shifting |

### Metrics
| Metric | Meaning |
|--------|---------|
| 🎯 Accuracy | How close gaze is to target (higher = better) |
| 🌀 Smoothness | How jitter-free eye movement is (higher = better) |
| 📉 Stability | Fixation steadiness during stationary phases |
| ⏱ Latency | Estimated delay between target and gaze (ms) |
| Score | Weighted composite 0–100 |

### Export
After each session, export raw frame data as **JSON** or **CSV** for further analysis.

---

## 📁 Project Structure

```
ocumind/
├── app.py              # Flask + SocketIO server (entry point)
├── gaze.py             # MediaPipe iris tracking
├── calibration.py      # Polynomial gaze-to-screen mapping
├── metrics.py          # Accuracy / smoothness / latency / stability
├── session.py          # Session recording and export
├── requirements.txt
├── templates/
│   └── index.html      # Full single-page frontend
├── static/
│   ├── css/style.css
│   └── js/app.js       # Stimulus engine, gaze rendering, charts
└── data/
    └── sessions/       # Saved session JSON/CSV files
```

---

## 🔧 Troubleshooting

**Camera not detected**
```bash
# Test that Python can open the camera
python3 -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

**mediapipe install fails on Apple Silicon**
```bash
pip install mediapipe --no-cache-dir
```

**Port 5050 already in use**
Edit the last line of `app.py` and change `port=5050` to another port.

**Gaze drifts / poor accuracy**
- Re-run calibration
- Ensure good lighting on your face
- Keep your head relatively still during calibration
- Sit ~50–70 cm from the screen

---

## ⚠️ Safety Notice

OcuMind is **not a medical device** and does not diagnose or treat any neurological condition. It is intended for research, rehabilitation support exploration, and cognitive/visual training experiments only.

---

## 🧠 Technical Notes

### Gaze Estimation
MediaPipe FaceMesh with `refine_landmarks=True` provides 478 facial landmarks including 10 iris landmarks (5 per eye). The iris center is computed relative to eye corner landmarks to produce a gaze direction estimate that is partially invariant to small head movements.

### Calibration
A degree-2 polynomial regression maps eye-relative iris positions to screen coordinates, fitted via least-squares on 9 fixation samples.

### Metrics
- **Accuracy**: rolling mean Euclidean distance (gaze ↔ target), normalised to 0–100%
- **Smoothness**: 1 − normalised velocity standard deviation over a 30-frame window
- **Latency**: cross-correlation lag between target and gaze x-trajectories
- **Stability**: mean gaze error during stationary target phases
- **Score**: 0.5 × accuracy + 0.3 × smoothness + 0.2 × stability
