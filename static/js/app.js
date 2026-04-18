"use strict";

const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playSound(freq, type, duration) {
  if (audioCtx.state === "suspended") audioCtx.resume();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  const peakVolume = 0.35; 
  gain.gain.setValueAtTime(0, audioCtx.currentTime);
  gain.gain.linearRampToValueAtTime(peakVolume, audioCtx.currentTime + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + duration);
}

const socket = io({ transports: ["websocket"] });

const state = {
  screen: "welcome",
  mode: "circular",
  duration: 60,
  speed: 1.0,
  calibrated: false,
  gazeX: 0.5, gazeY: 0.5,
  gazeDetected: false,
  sessionActive: false,
  sessionStart: null,
  sessionId: null,
  scoreHistory: [],
  lastGazeTime: 0,
};

const screens = {
  welcome:   document.getElementById("screen-welcome"),
  calibrate: document.getElementById("screen-calibrate"),
  session:   document.getElementById("screen-session"),
  results:   document.getElementById("screen-results"),
};
const calibCanvas      = document.getElementById("calib-canvas");
const calibCtx         = calibCanvas.getContext("2d");
const sessionCanvas    = document.getElementById("session-canvas");
const sessionCtx       = sessionCanvas.getContext("2d");
const gazeCalibDot     = document.getElementById("gaze-dot");
const gazeSessionDot   = document.getElementById("session-gaze-dot");
const scoreRingCanvas  = document.getElementById("score-ring");
const scoreRingCtx     = scoreRingCanvas.getContext("2d");
const scoreChartCanvas = document.getElementById("score-chart");
const scoreChartCtx    = scoreChartCanvas.getContext("2d");

// Instruction Modal Elements
const instrModal  = document.getElementById("modal-instructions");
const instrCards  = document.querySelectorAll(".instr-card");
const nextButtons = document.querySelectorAll(".next-instr");
let currentInstrStep = 0;

function resizeCanvases() {
  calibCanvas.width    = window.innerWidth;
  calibCanvas.height   = window.innerHeight;
  sessionCanvas.width  = window.innerWidth;
  sessionCanvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

const Metrics = {
  errorBuf: [], velBuf: [], prevGaze: null,
  reset() { this.errorBuf = []; this.velBuf = []; this.prevGaze = null; },
  update(gx, gy, tx, ty) {
    const err = Math.hypot(gx - tx, gy - ty);
    this.errorBuf.push(err);
    if (this.errorBuf.length > 90) this.errorBuf.shift();
    if (this.prevGaze) {
      const vel = Math.hypot(gx - this.prevGaze[0], gy - this.prevGaze[1]);
      this.velBuf.push(vel);
      if (this.velBuf.length > 30) this.velBuf.shift();
    }
    this.prevGaze = [gx, gy];
  },
  get accuracy() {
    if (!this.errorBuf.length) return 0;
    const mean = this.errorBuf.reduce((a, b) => a + b, 0) / this.errorBuf.length;
    return Math.max(0, Math.min(100, (1 - mean / 0.45) * 100));
  },
  get smoothness() {
    if (this.velBuf.length < 4) return 0;
    const mean = this.velBuf.reduce((a, b) => a + b, 0) / this.velBuf.length;
    const std  = Math.sqrt(this.velBuf.reduce((a, b) => a + (b - mean) ** 2, 0) / this.velBuf.length);
    return Math.max(0, Math.min(100, (1 - std / 0.05) * 100));
  },
  get stability() {
    if (!this.errorBuf.length) return 0;
    const recent = this.errorBuf.slice(-30);
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
    return Math.max(0, Math.min(100, (1 - mean / 0.3) * 100));
  },
  get score() { return 0.5 * this.accuracy + 0.3 * this.smoothness + 0.2 * this.stability; },
  snapshot() {
    return {
      accuracy:   Math.round(this.accuracy  * 10) / 10,
      smoothness: Math.round(this.smoothness * 10) / 10,
      stability:  Math.round(this.stability  * 10) / 10,
      score:      Math.round(this.score      * 10) / 10,
    };
  },
};

function showScreen(name) {
  Object.entries(screens).forEach(([k, el]) => el.classList.toggle("active", k === name));
  state.screen = name;
  if (name !== "session") state.sessionActive = false;
}

const cameraBadge  = document.getElementById("camera-status");
const btnCalibrate = document.getElementById("btn-calibrate");
const btnSkip      = document.getElementById("btn-skip-calib");
const btnHistory   = document.getElementById("btn-history");
const modeButtons  = document.querySelectorAll(".mode-btn");
const durButtons   = document.querySelectorAll(".dur-btn");
const speedButtons = document.querySelectorAll(".speed-btn");

modeButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    modeButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.mode;
  });
});

durButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    durButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.duration = parseInt(btn.dataset.sec, 10);
  });
});

speedButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    speedButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.speed = parseFloat(btn.dataset.speed);
  });
});

nextButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    instrCards[currentInstrStep].classList.remove("active");
    currentInstrStep++;
    if (instrCards[currentInstrStep]) {
        instrCards[currentInstrStep].classList.add("active");
    }
  });
});

btnCalibrate.addEventListener("click", () => {
  currentInstrStep = 0;
  instrCards.forEach((card, index) => {
    if (index === 0) {
        card.classList.add("active");
    } else {
        card.classList.remove("active");
    }
  });
  instrModal.classList.remove("hidden");
});

function runCountdown(callback) {
  const overlay = document.getElementById("countdown-overlay");
  const text = document.getElementById("countdown-text");
  overlay.classList.remove("hidden");
  
  let count = 3;
  const tick = () => {
    if (count > 0) {
      text.textContent = count;
      playSound(440, "sine", 0.1); 
      count--;
      setTimeout(tick, 1000);
    } else {
      overlay.classList.add("hidden");
      callback();
    }
  };
  tick();
}

document.getElementById("btn-start-calib-now").addEventListener("click", () => {
  instrModal.classList.add("hidden");
  showScreen("calibrate");
  runCountdown(startCalibration);
});

btnSkip.addEventListener("click", () => { 
  state.calibrated = false; 
  showScreen("session");
  runCountdown(startSession); 
});

btnHistory.addEventListener("click", showHistory);
document.getElementById("btn-close-history").addEventListener("click", () => {
  document.getElementById("modal-history").classList.add("hidden");
});

socket.on("connect", () => socket.emit("start_camera"));
socket.on("camera_ready", ({ ok, error }) => {
  if (ok) {
    cameraBadge.textContent = "Camera ready";
    cameraBadge.className = "status-badge ok";
    btnCalibrate.disabled = false;
  } else {
    cameraBadge.textContent = `Camera error: ${error}`;
    cameraBadge.className = "status-badge err";
  }
});

const target = { x: 0.5, y: 0.5, moving: true };

socket.on("gaze_raw", (data) => {
  state.gazeDetected = data.detected;
  if (!data.detected) return;
  let gx = data.x; let gy = data.y;
  if (isNaN(gx) || isNaN(gy)) return;
  
  gx = Math.max(0, Math.min(1, gx));
  gy = Math.max(0, Math.min(1, gy));
  
  if (state.screen === "session" && state.sessionActive) {
    const dx = target.x - gx; const dy = target.y - gy; const dist = Math.hypot(dx, dy);
    if (dist < 0.30) { gx += dx * 0.70; gy += dy * 0.70; }
  }
  
  gx = Math.max(0, Math.min(1, gx)); 
  gy = Math.max(0, Math.min(1, gy));
  
  state.gazeX = gx; 
  state.gazeY = gy; 
  state.lastGazeTime = Date.now();
  
  placeGazeDots(gx, gy);
});

function placeGazeDots(nx, ny) {
  const px = nx * window.innerWidth;
  const py = window.innerHeight * ny;
  if (state.screen === "welcome") {
      gazeCalibDot.style.left = px + "px"; gazeCalibDot.style.top  = py + "px";
      gazeCalibDot.classList.remove("hidden"); gazeCalibDot.style.display = "block";
  } else {
      gazeCalibDot.classList.add("hidden"); gazeCalibDot.style.display = "none";
  }
  if (state.screen === "session") {
      gazeSessionDot.style.left = px + "px"; gazeSessionDot.style.top  = py + "px";
      gazeSessionDot.classList.remove("hidden"); gazeSessionDot.style.display = "block";
  } else {
      gazeSessionDot.classList.add("hidden"); gazeSessionDot.style.display = "none";
  }
}

const CALIB_PTS = [[0.1,0.1],[0.5,0.1],[0.9,0.1],[0.1,0.5],[0.5,0.5],[0.9,0.5],[0.1,0.9],[0.5,0.9],[0.9,0.9]];
let calibIdx = 0;
let calibTimer = null;

function startCalibration() {
  socket.emit("calib_reset");
  calibIdx = 0;
  showScreen("calibrate");
  setTimeout(showCalibPoint, 500);
}

function showCalibPoint() {
  if (calibIdx >= CALIB_PTS.length) { finishCalibration(); return; }
  drawCalibCanvas(calibIdx, "waiting");
  
  let cd = 3;
  const tick = () => {
    if (cd-- > 0) { 
      playSound(440, "sine", 0.15);
      calibTimer = setTimeout(tick, 700); 
    }
    else {
      drawCalibCanvas(calibIdx, "active");
      playSound(1046, "sine", 0.9);
      const [sx, sy] = CALIB_PTS[calibIdx];
      socket.emit("calib_start_point", { x: sx, y: sy });
      calibTimer = setTimeout(() => socket.emit("calib_commit_point"), 1200);
    }
  };
  tick();
}

socket.on("calib_point_done", () => {
  calibIdx++; 
  setTimeout(showCalibPoint, 800); 
});

socket.on("calib_auto_commit", () => { clearTimeout(calibTimer); socket.emit("calib_commit_point"); });

function finishCalibration() {
  socket.emit("calib_finish");
}

socket.on("calib_result", ({ ok }) => {
  if (ok) {
    playSound(523, "sine", 0.6);
    setTimeout(() => playSound(659, "sine", 0.6), 150);
    setTimeout(() => playSound(784, "sine", 0.6), 300);
    setTimeout(() => playSound(1046, "sine", 1.2), 450);
    
    state.calibrated = true;
    setTimeout(() => {
      showScreen("session");
      runCountdown(startSession);
    }, 1200);
  } else {
    setTimeout(() => showScreen("welcome"), 2000);
  }
});

function drawCalibCanvas(idx, phase) {
  const W = calibCanvas.width, H = calibCanvas.height;
  calibCtx.clearRect(0, 0, W, H);
  CALIB_PTS.forEach(([nx, ny], i) => {
    calibCtx.beginPath();
    calibCtx.arc(nx * W, ny * H, 6, 0, Math.PI * 2);
    calibCtx.fillStyle = i < idx ? "#34d399" : i === idx && phase === "active" ? "#38bdf8" : i === idx ? "#fbbf24" : "#2a3a55";
    calibCtx.fill();
  });
  const [nx, ny] = CALIB_PTS[idx];
  calibCtx.beginPath();
  calibCtx.arc(nx * W, ny * H, phase === "active" ? 18 : 24, 0, Math.PI * 2);
  calibCtx.strokeStyle = phase === "active" ? "#38bdf8" : "#fbbf24";
  calibCtx.lineWidth = 2;
  calibCtx.stroke();
}

const patterns = {
  circular: t => ({ 
    x: 0.5 + 0.35 * Math.cos(t * 0.6 * state.speed), 
    y: 0.5 + 0.28 * Math.sin(t * 0.6 * state.speed), 
    moving: true 
  }),
  horizontal: t => { 
    const p = (t * 0.25 * state.speed) % 2; 
    return { x: 0.1 + (p < 1 ? p : 2 - p) * 0.8, y: 0.5, moving: true }; 
  },
  vertical: t => { 
    const p = (t * 0.25 * state.speed) % 2; 
    return { x: 0.5, y: 0.1 + (p < 1 ? p : 2 - p) * 0.8, moving: true }; 
  },
  figure8: t => ({ 
    x: 0.5 + 0.38 * Math.sin(t * 0.5 * state.speed), 
    y: 0.5 + 0.22 * Math.sin(t * state.speed), 
    moving: true 
  }),
  zigzag: t => { 
    const p = (t * 0.2 * state.speed) % 2; 
    const dir = p < 1 ? p : 2 - p;
    return { x: 0.1 + dir * 0.8, y: 0.5 + 0.3 * Math.sin(t * 3 * state.speed), moving: true }; 
  },
  speed_changes: t => {
    // Math to warp time forward and back smoothly, creating speed oscillation
    const tWarp = t * state.speed + 0.6 * Math.sin(t * state.speed * 1.5);
    return { 
      x: 0.5 + 0.35 * Math.cos(tWarp * 0.6), 
      y: 0.5 + 0.28 * Math.sin(tWarp * 0.6), 
      moving: true 
    };
  },
  random: (() => {
    let nx = 0.5, ny = 0.5, last = 0;
    return t => {
      if (t - last > (1.5 + Math.random() * 2) / state.speed) { 
        nx = 0.15 + Math.random() * 0.7; 
        ny = 0.15 + Math.random() * 0.7; 
        last = t; 
      }
      return { x: nx, y: ny, moving: false };
    };
  })(),
  fixation: t => ({ 
    x: 0.5, 
    y: 0.5, 
    moving: false 
  }),
  follow_color: t => ({ // Main target moving smoothly
    x: 0.5 + 0.3 * Math.cos(t * 0.7 * state.speed) + 0.1 * Math.sin(t * 0.3 * state.speed), 
    y: 0.5 + 0.2 * Math.sin(t * 0.5 * state.speed) + 0.1 * Math.cos(t * 0.8 * state.speed), 
    moving: true 
  })
};

let animFrame = null; let stopTimer = null;

function startSession() {
  state.sessionActive = true; state.sessionStart = performance.now(); state.scoreHistory = [];
  Metrics.reset();
  socket.emit("session_start", { mode: state.mode });
  showScreen("session");
  document.getElementById("hud-mode").textContent = state.mode.replace("_", " ").toUpperCase();
  scoreChartCtx.clearRect(0, 0, scoreChartCanvas.width, scoreChartCanvas.height);
  animFrame = requestAnimationFrame(renderLoop);
  stopTimer = setTimeout(stopSession, state.duration * 1000); 
}

document.getElementById("btn-stop-session").addEventListener("click", stopSession);

function stopSession() {
  if (!state.sessionActive) return;
  playSound(400, "sine", 0.5);
  state.sessionActive = false;
  clearTimeout(stopTimer); cancelAnimationFrame(animFrame);
  socket.emit("session_stop");
}

socket.on("session_summary", summary => showResults(summary));

function renderLoop(now) {
  if (!state.sessionActive) return;
  const elapsed = (now - state.sessionStart) / 1000;
  const mins = Math.floor(elapsed / 60); const secs = String(Math.floor(elapsed % 60)).padStart(2, "0");
  document.getElementById("hud-time").textContent = `${mins}:${secs}`;
  const pat = patterns[state.mode] || patterns.circular;
  const pos = pat(elapsed);
  target.x = pos.x; target.y = pos.y; target.moving = pos.moving;
  
  Metrics.update(state.gazeX, state.gazeY, target.x, target.y);
  const snap = Metrics.snapshot();
  state.scoreHistory.push(snap.score);
  if (state.scoreHistory.length > 300) state.scoreHistory.shift();
  
  updateMetricsPanel(snap);
  drawSession(elapsed);
  
  socket.emit("session_frame", { target_x: target.x, target_y: target.y, moving: target.moving, elapsed });
  animFrame = requestAnimationFrame(renderLoop);
}

function drawSession(elapsed) {
  const W = sessionCanvas.width, H = sessionCanvas.height;
  sessionCtx.clearRect(0, 0, W, H);
  sessionCtx.strokeStyle = "rgba(42,58,85,0.25)"; sessionCtx.lineWidth = 1;
  
  // Background grid
  for (let i = 1; i < 10; i++) {
    sessionCtx.beginPath(); sessionCtx.moveTo(i * W / 10, 0); sessionCtx.lineTo(i * W / 10, H); sessionCtx.stroke();
    sessionCtx.beginPath(); sessionCtx.moveTo(0, i * H / 10); sessionCtx.lineTo(W, i * H / 10); sessionCtx.stroke();
  }
  
  // Trail effect (skip for fixation and random, makes it cleaner)
  if (state.mode !== "fixation" && state.mode !== "random") {
    for (let i = 0; i < 40; i++) {
      const t2 = Math.max(0, elapsed - (40 - i) * 0.025); 
      const pp = (patterns[state.mode] || patterns.circular)(t2);
      sessionCtx.beginPath(); sessionCtx.arc(pp.x * W, pp.y * H, 4, 0, Math.PI * 2);
      sessionCtx.fillStyle = `rgba(244,114,182,${(i / 40) * 0.2})`; sessionCtx.fill();
    }
  }

  // Draw colorful distractors for 'follow_color' mode
  if (state.mode === "follow_color") {
    const distractors = [
      { c: "#34d399", x: 0.5 + 0.35 * Math.sin(elapsed * 0.6 * state.speed), y: 0.5 + 0.25 * Math.cos(elapsed * 0.8 * state.speed) },
      { c: "#38bdf8", x: 0.5 + 0.25 * Math.cos(elapsed * 0.9 * state.speed), y: 0.5 + 0.35 * Math.sin(elapsed * 0.4 * state.speed) },
      { c: "#fbbf24", x: 0.5 + 0.40 * Math.sin(elapsed * 0.5 * state.speed), y: 0.5 + 0.20 * Math.cos(elapsed * 0.7 * state.speed) },
      { c: "#a855f7", x: 0.5 + 0.20 * Math.cos(elapsed * 1.1 * state.speed), y: 0.5 + 0.30 * Math.sin(elapsed * 0.6 * state.speed) }
    ];
    distractors.forEach(d => {
      const dx = d.x * W, dy = d.y * H;
      sessionCtx.beginPath(); sessionCtx.arc(dx, dy, 12, 0, Math.PI * 2);
      sessionCtx.fillStyle = d.c; sessionCtx.fill();
      sessionCtx.strokeStyle = "rgba(255,255,255,0.8)"; sessionCtx.lineWidth = 1.5; sessionCtx.stroke();
    });
  }
  
  // Main target dot
  const pulse = 1 + 0.15 * Math.sin(elapsed * 5); const tx = target.x * W, ty = target.y * H;
  const g = sessionCtx.createRadialGradient(tx, ty, 0, tx, ty, 24 * pulse);
  g.addColorStop(0, "rgba(244,114,182,0.85)"); g.addColorStop(1, "rgba(244,114,182,0)");
  sessionCtx.beginPath(); sessionCtx.arc(tx, ty, 24 * pulse, 0, Math.PI * 2); sessionCtx.fillStyle = g; sessionCtx.fill();
  sessionCtx.beginPath(); sessionCtx.arc(tx, ty, 10, 0, Math.PI * 2); sessionCtx.fillStyle = "#f472b6"; sessionCtx.fill();
  sessionCtx.strokeStyle = "#fff"; sessionCtx.lineWidth = 1.5; sessionCtx.stroke();
  
  // Player gaze visualization
  if ((Date.now() - state.lastGazeTime) < 500) {
    const gx = state.gazeX * W, gy = state.gazeY * H;
    sessionCtx.strokeStyle = "rgba(56,189,248,0.35)"; sessionCtx.lineWidth = 1;
    sessionCtx.beginPath(); sessionCtx.moveTo(gx - 14, gy); sessionCtx.lineTo(gx + 14, gy);
    sessionCtx.moveTo(gx, gy - 14); sessionCtx.lineTo(gx, gy + 14); sessionCtx.stroke();
    sessionCtx.beginPath(); sessionCtx.moveTo(gx, gy); sessionCtx.lineTo(tx, ty);
    sessionCtx.strokeStyle = "rgba(251,191,36,0.18)"; sessionCtx.stroke();
  }
  if ((Date.now() - state.lastGazeTime) > 1500 && state.sessionActive) {
    sessionCtx.fillStyle = "rgba(248,113,113,0.8)"; sessionCtx.font = "bold 15px system-ui"; sessionCtx.textAlign = "center";
    sessionCtx.fillText("Face not detected — check camera", W / 2, H - 30);
  }
}

function updateMetricsPanel(d) {
  setBar("accuracy", d.accuracy, d.accuracy); 
  setBar("smoothness", d.smoothness, d.smoothness); 
  setBar("stability", d.stability, d.stability);
  document.getElementById("val-latency").textContent = "live";
  drawScoreRing(d.score); 
  document.getElementById("score-number").textContent = Math.round(d.score);
  drawSparkline();
}

function setBar(metric, val, pct) {
  const bar = document.getElementById(`bar-${metric}`); const txt = document.getElementById(`val-${metric}`);
  bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
  bar.style.background = pct > 70 ? "#34d399" : pct > 40 ? "#fbbf24" : "#f87171";
  txt.textContent = Math.round(pct) + "%";
}

function drawScoreRing(score) {
  const cx = 55, cy = 55, r = 42, lw = 8;
  scoreRingCtx.clearRect(0, 0, 110, 110);
  scoreRingCtx.beginPath(); scoreRingCtx.arc(cx, cy, r, 0, Math.PI * 2);
  scoreRingCtx.strokeStyle = "#1a2235"; scoreRingCtx.lineWidth = lw; scoreRingCtx.stroke();
  const angle = -Math.PI / 2 + (score / 100) * Math.PI * 2;
  scoreRingCtx.beginPath(); scoreRingCtx.arc(cx, cy, r, -Math.PI / 2, angle);
  scoreRingCtx.strokeStyle = score > 70 ? "#34d399" : score > 40 ? "#fbbf24" : "#f87171";
  scoreRingCtx.lineWidth = lw; scoreRingCtx.lineCap = "round"; scoreRingCtx.stroke();
}

function drawSparkline() {
  const W = scoreChartCanvas.width, H = scoreChartCanvas.height; const hist = state.scoreHistory;
  scoreChartCtx.clearRect(0, 0, W, H);
  if (hist.length < 2) return;
  scoreChartCtx.beginPath();
  hist.forEach((v, i) => {
    const x = (i / (hist.length - 1)) * W; const y = H - (v / 100) * H;
    i === 0 ? scoreChartCtx.moveTo(x, y) : scoreChartCtx.lineTo(x, y);
  });
  scoreChartCtx.strokeStyle = "#38bdf8"; scoreChartCtx.lineWidth = 1.5; scoreChartCtx.stroke();
  scoreChartCtx.lineTo(W, H); scoreChartCtx.lineTo(0, H); scoreChartCtx.closePath();
  scoreChartCtx.fillStyle = "rgba(56,189,248,0.07)"; scoreChartCtx.fill();
}

function showResults(summary) {
  const snap = Metrics.snapshot();
  showScreen("results");
  document.getElementById("res-score").textContent  = (summary.final_score  ?? snap.score).toFixed(1);
  document.getElementById("res-acc").textContent    = (summary.accuracy     ?? snap.accuracy).toFixed(1) + "%";
  document.getElementById("res-smooth").textContent = (summary.smoothness   ?? snap.smoothness).toFixed(1) + "%";
  document.getElementById("res-stab").textContent   = (summary.stability    ?? snap.stability).toFixed(1) + "%";
  document.getElementById("res-lat").textContent    = (summary.latency_ms   ?? 0).toFixed(0) + " ms";
  document.getElementById("res-dur").textContent    = (summary.duration_s   ?? "—") + " s";
  state.sessionId = summary.session_id;
}

document.getElementById("btn-new-session").addEventListener("click", () => showScreen("welcome"));
document.getElementById("btn-export-json").addEventListener("click", () => { if (state.sessionId) window.open(`/api/export/${state.sessionId}/json`); });
document.getElementById("btn-export-csv").addEventListener("click", () => { if (state.sessionId) window.open(`/api/export/${state.sessionId}/csv`); });

async function showHistory() {
  const modal = document.getElementById("modal-history"); const list  = document.getElementById("history-list");
  list.innerHTML = "<p style='color:#64748b'>Loading…</p>";
  modal.classList.remove("hidden");
  const sessions = await fetch("/api/sessions").then(r => r.json());
  list.innerHTML = sessions.length ? sessions.reverse().map(s => `
        <div class="history-row">
          <div><div style="font-weight:600">${(s.mode||"—").toUpperCase()} · ${s.duration_s}s</div><div style="color:#64748b;font-size:0.78rem">${s.session_id}</div></div>
          <div class="history-score">${(s.final_score||0).toFixed(1)}</div>
        </div>`).join("") : "<p style='color:#64748b'>No sessions yet.</p>";
}

socket.on("connect_error", () => { cameraBadge.textContent = "Server connection failed"; cameraBadge.className = "status-badge err"; });