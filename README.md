Ishika Aggarwal, Marie Cho, Sri Kesiraju

# 1. Make sure you have Python 3.10–3.13 installed (NOT 3.14 — too new)
python3 --version

# 2. Download and unzip the project, then navigate into it
cd ~/Downloads/ocumind

# 3. Create a virtual environment
python3 -m venv venv

# 4. Activate it
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows (use this instead)

# 5. Install dependencies
pip install flask==3.0.3 flask-socketio==5.3.6 opencv-python==4.10.0.84 \
  mediapipe==0.10.33 numpy python-engineio==4.9.1 \
  python-socketio==5.11.3 simple-websocket

# 6. Run the app
python app.py

# 7. Open in Chrome (not Safari)
#    http://localhost:5050