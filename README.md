# 🧠 Ocular Loop

Ocular Loop is a real-time eye-tracking rehabilitation and cognitive training application that uses webcam-based gaze estimation to guide users through smooth pursuit and fixation exercises.

The system displays moving visual targets (e.g., a drifting dot) while tracking the user’s eye movements to measure:
- accuracy  
- smoothness  
- latency  
- tracking stability  

Designed as a lightweight, browser-based or desktop-compatible tool, Ocular Loop supports research and rehabilitation contexts involving ocular motor control and visuomotor coordination.

---

## 💡 Use Cases

Ocular Loop is particularly relevant for individuals experiencing impairments in eye movement control, including:

- Parkinsonian disorders (impaired smooth pursuit, delayed saccades)  
- Post-stroke recovery (ocular motor and attentional deficits)  
- Multiple Sclerosis (visual fatigue, impaired coordination)  
- Traumatic Brain Injury (TBI) affecting gaze stability  
- General neurocognitive fatigue or attention disorders  

> ⚠️ Note: This application is **not a diagnostic tool**. It is intended for training and monitoring visual-motor function.

---

## ✨ Features

- 🎯 Smooth pursuit and fixation exercises  
- ⚡ Real-time gaze tracking via webcam  
- 📊 Performance metrics (accuracy, latency, stability)  
- 📈 Multi-session progress tracking (planned/extendable)  
- 🧠 Cognitive + visual-motor training  

---

## 👥 Team

**Ishika Aggarwal · Marie Cho · Sri Kesiraju**

---

## 🚀 Getting Started

Follow these steps to run Ocular Loop locally.

---

### 1. Prerequisites

Ensure you have Python3 installed

```bash
python3 --version
```

python3 -m venv venv

source venv/bin/activate 

pip install flask==3.0.3 flask-socketio==5.3.6 opencv-python==4.10.0.84 \
  mediapipe==0.10.33 numpy python-engineio==4.9.1 \
  python-socketio==5.11.3 simple-websocket

python app.py

http://localhost:5050
