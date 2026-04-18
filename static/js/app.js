"use strict";

// ── SocketIO ────────────────────────────────────────────────────
const socket = io({ transports: ["websocket"] });

// ── State ───────────────────────────────────────────────────────
const state = {
  screen: "welcome",
  mode: "circular",
  calibrated: false,
  gazeX: 0.5, gazeY: 0.5,
  gazeDetected: false,
  sessionActive: false,
  sessionStart: null,
  sessionId: null,
  scoreHistory: [],
  lastGazeTime: 0,
};

// ── DOM ─────────────────────────────────────────────────────────
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

function resizeCanvases() {
  calibCanvas.width    = window.innerWidth;
  calibCanvas.height   = window.innerHeight;
  sessionCanvas.width  = window.innerWidth;
  sessionCanvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

// ════════════════════════════════════════════════════════════════
//  CLIENT-SIDE METRICS ENGINE
//  Computes everything from gaze + target positions in JS —
//  no server round-trip needed, always live.
// ════════════════════════════════════════════════════════════════
const Metrics = {
  errorBuf:  [],   // rolling accuracy errors
  velBuf:    [],   // rolling gaze velocities
  prevGaze:  null, // [x, y]

  reset() {
    this.errorBuf = [];
    this.velBuf   = [];
    this.prevGaze = null;
  },

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
    // short-window accuracy (last 30 frames) as fixation proxy
    if (!this.errorBuf.length) return 0;
    const recent = this.errorBuf.slice(-30);
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
    return Math.max(0, Math.min(100, (1 - mean / 0.3) * 100));
  },

  get score() {
    return 0.5 * this.accuracy + 0.3 * this.smoothness + 0.2 * this.stability;
  },

  snapshot() {
    return {
      accuracy:   Math.round(this.accuracy  * 10) / 10,
      smoothness: Math.round(this.smoothness * 10) / 10,
      stability:  Math.round(this.stability  * 10) / 10,
      score:      Math.round(this.score      * 10) / 10,
    };
  },
};

// ════════════════════════════════════════════════════════════════
//  SCREEN NAV
// ════════════════════════════════════════════════════════════════
function showScreen(name) {
  Object.entries(screens).forEach(([k, el]) => el.classList.toggle("active", k === name));
  state.screen = name;
}

// ════════════════════════════════════════════════════════════════
//  WELCOME
// ════════════════════════════════════════════════════════════════
const cameraBadge  = document.getElementById("camera-status");
const btnCalibrate = document.getElementById("btn-calibrate");
const btnSkip      = document.getElementById("btn-skip-calib");
const btnHistory   = document.getElementById("btn-history");
const modeButtons  = document.querySelectorAll(".mode-btn");

modeButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    modeButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.mode;
  });
});

btnCalibrate.addEventListener("click", startCalibration);
btnSkip.addEventListener("click", () => { state.calibrated = false; startSession(); });
btnHistory.addEventListener("click", showHistory);
document.getElementById("btn-close-history").addEventListener("click", () => {
  document.getElementById("modal-history").classList.add("hidden");
});

socket.on("connect", () => socket.emit("start_camera"));

socket.on("camera_ready", ({ ok, error }) => {
  if (ok) {
    cameraBadge.textContent = "✅ Camera ready";
    cameraBadge.className   = "status-badge ok";
    btnCalibrate.disabled   = false;
  } else {
    cameraBadge.textContent = `❌ Camera error: ${error}`;
    cameraBadge.className   = "status-badge err";
  }
});

// ════════════════════════════════════════════════════════════════
//  GAZE DOT — raw broadcast from server
// ════════════════════════════════════════════════════════════════
socket.on("gaze_raw", (data) => {
  state.gazeDetected = data.detected;
  if (!data.detected) return;
  state.gazeX = data.x;
  state.gazeY = data.y;
  state.lastGazeTime = Date.now();
  placeGazeDots(data.x, data.y);
});

function placeGazeDots(nx, ny) {
  const px = nx * window.innerWidth;
  const py = ny * window.innerHeight;
  [gazeCalibDot, gazeSessionDot].forEach(d => {
    d.style.left = px + "px";
    d.style.top  = py + "px";
    d.classList.remove("hidden");
  });
}

// ════════════════════════════════════════════════════════════════
//  CALIBRATION
// ════════════════════════════════════════════════════════════════
const CALIB_PTS = [
  [0.05,0.05],[0.5,0.05],[0.95,0.05],
  [0.05,0.5], [0.5,0.5], [0.95,0.5],
  [0.05,0.95],[0.5,0.95],[0.95,0.95],
];
let calibIdx = 0;
let calibTimer = null;

function startCalibration() {
  socket.emit("calib_reset");
  calibIdx = 0;
  showScreen("calibrate");
  gazeCalibDot.classList.remove("hidden");
  setTimeout(showCalibPoint, 500);
}

function showCalibPoint() {
  if (calibIdx >= CALIB_PTS.length) { finishCalibration(); return; }
  document.getElementById("calib-progress").textContent = `Point ${calibIdx + 1} / ${CALIB_PTS.length}`;
  drawCalibCanvas(calibIdx, "waiting");
  let cd = 3;
  const tick = () => {
    document.getElementById("calib-instruction").textContent = `Look at the dot — ${cd}`;
    if (cd-- > 0) { calibTimer = setTimeout(tick, 700); }
    else {
      drawCalibCanvas(calibIdx, "active");
      document.getElementById("calib-instruction").textContent = "Hold still…";
      const [sx, sy] = CALIB_PTS[calibIdx];
      socket.emit("calib_start_point", { x: sx, y: sy });
      calibTimer = setTimeout(() => socket.emit("calib_commit_point"), 1800);
    }
  };
  tick();
}

socket.on("calib_point_done", () => { calibIdx++; setTimeout(showCalibPoint, 350); });
socket.on("calib_auto_commit", () => { clearTimeout(calibTimer); socket.emit("calib_commit_point"); });

function finishCalibration() {
  document.getElementById("calib-instruction").textContent = "Calibrating…";
  socket.emit("calib_finish");
}
socket.on("calib_result", ({ ok }) => {
  if (ok) {
    state.calibrated = true;
    document.getElementById("calib-instruction").textContent = "✅ Done!";
    setTimeout(startSession, 800);
  } else {
    document.getElementById("calib-instruction").textContent = "⚠ Failed — try again.";
    setTimeout(() => showScreen("welcome"), 2000);
  }
});

function drawCalibCanvas(idx, phase) {
  const W = calibCanvas.width, H = calibCanvas.height;
  calibCtx.clearRect(0, 0, W, H);
  CALIB_PTS.forEach(([nx, ny], i) => {
    calibCtx.beginPath();
    calibCtx.arc(nx * W, ny * H, 6, 0, Math.PI * 2);
    calibCtx.fillStyle = i < idx ? "#34d399" :
                         i === idx && phase === "active" ? "#38bdf8" :
                         i === idx ? "#fbbf24" : "#2a3a55";
    calibCtx.fill();
  });
  const [nx, ny] = CALIB_PTS[idx];
  calibCtx.beginPath();
  calibCtx.arc(nx * W, ny * H, phase === "active" ? 18 : 24, 0, Math.PI * 2);
  calibCtx.strokeStyle = phase === "active" ? "#38bdf8" : "#fbbf24";
  calibCtx.lineWidth = 2;
  calibCtx.stroke();
}

// ════════════════════════════════════════════════════════════════
//  STIMULUS PATTERNS
// ════════════════════════════════════════════════════════════════
const patterns = {
  circular: t => ({
    x: 0.5 + 0.35 * Math.cos(t * 0.6),
    y: 0.5 + 0.28 * Math.sin(t * 0.6),
    moving: true,
  }),
  linear: t => {
    const p = (t * 0.25) % 2;
    return { x: 0.1 + (p < 1 ? p : 2 - p) * 0.8, y: 0.5, moving: true };
  },
  figure8: t => ({
    x: 0.5 + 0.38 * Math.sin(t * 0.5),
    y: 0.5 + 0.22 * Math.sin(t),
    moving: true,
  }),
  random: (() => {
    let nx = 0.5, ny = 0.5, last = 0;
    return t => {
      if (t - last > 1.5 + Math.random() * 2) {
        nx = 0.15 + Math.random() * 0.7;
        ny = 0.15 + Math.random() * 0.7;
        last = t;
      }
      return { x: nx, y: ny, moving: false };
    };
  })(),
};

// ════════════════════════════════════════════════════════════════
//  SESSION
// ════════════════════════════════════════════════════════════════
const SESSION_SECS = 60;
let animFrame   = null;
let stopTimer   = null;
const target    = { x: 0.5, y: 0.5, moving: true };

function startSession() {
  state.sessionActive = true;
  state.sessionStart  = performance.now();
  state.scoreHistory  = [];
  Metrics.reset();
  socket.emit("session_start", { mode: state.mode });
  showScreen("session");
  document.getElementById("hud-mode").textContent = state.mode.toUpperCase();
  scoreChartCtx.clearRect(0, 0, scoreChartCanvas.width, scoreChartCanvas.height);
  animFrame = requestAnimationFrame(renderLoop);
  stopTimer = setTimeout(stopSession, SESSION_SECS * 1000);
}

document.getElementById("btn-stop-session").addEventListener("click", stopSession);

function stopSession() {
  if (!state.sessionActive) return;
  state.sessionActive = false;
  clearTimeout(stopTimer);
  cancelAnimationFrame(animFrame);
  socket.emit("session_stop");
}

socket.on("session_summary", summary => showResults(summary));

// ── Main render loop ────────────────────────────────────────────
function renderLoop(now) {
  if (!state.sessionActive) return;

  const elapsed = (now - state.sessionStart) / 1000;
  const mins = Math.floor(elapsed / 60);
  const secs = String(Math.floor(elapsed % 60)).padStart(2, "0");
  document.getElementById("hud-time").textContent = `${mins}:${secs}`;

  const pat = patterns[state.mode] || patterns.circular;
  const pos = pat(elapsed);
  target.x = pos.x; target.y = pos.y; target.moving = pos.moving;

  // ── Client-side metrics — always update every frame ─────────
  Metrics.update(state.gazeX, state.gazeY, target.x, target.y);
  const snap = Metrics.snapshot();
  state.scoreHistory.push(snap.score);
  if (state.scoreHistory.length > 300) state.scoreHistory.shift();
  updateMetricsPanel(snap);

  drawSession(elapsed);

  // Send to server for recording (fire-and-forget)
  socket.emit("session_frame", {
    target_x: target.x, target_y: target.y,
    moving: target.moving, elapsed,
  });

  animFrame = requestAnimationFrame(renderLoop);
}

// ── Canvas drawing ───────────────────────────────────────────────
function drawSession(elapsed) {
  const W = sessionCanvas.width, H = sessionCanvas.height;
  sessionCtx.clearRect(0, 0, W, H);

  // Grid
  sessionCtx.strokeStyle = "rgba(42,58,85,0.25)";
  sessionCtx.lineWidth = 1;
  for (let i = 1; i < 10; i++) {
    sessionCtx.beginPath();
    sessionCtx.moveTo(i * W / 10, 0); sessionCtx.lineTo(i * W / 10, H); sessionCtx.stroke();
    sessionCtx.beginPath();
    sessionCtx.moveTo(0, i * H / 10); sessionCtx.lineTo(W, i * H / 10); sessionCtx.stroke();
  }

  // Target trail
  for (let i = 0; i < 40; i++) {
    const t2 = elapsed - (40 - i) * 0.025;
    const pp = (patterns[state.mode] || patterns.circular)(t2);
    sessionCtx.beginPath();
    sessionCtx.arc(pp.x * W, pp.y * H, 4, 0, Math.PI * 2);
    sessionCtx.fillStyle = `rgba(244,114,182,${(i / 40) * 0.2})`;
    sessionCtx.fill();
  }

  // Target dot (pulsing pink)
  const pulse = 1 + 0.15 * Math.sin(elapsed * 5);
  const tx = target.x * W, ty = target.y * H;
  const g = sessionCtx.createRadialGradient(tx, ty, 0, tx, ty, 24 * pulse);
  g.addColorStop(0, "rgba(244,114,182,0.85)");
  g.addColorStop(1, "rgba(244,114,182,0)");
  sessionCtx.beginPath(); sessionCtx.arc(tx, ty, 24 * pulse, 0, Math.PI * 2);
  sessionCtx.fillStyle = g; sessionCtx.fill();
  sessionCtx.beginPath(); sessionCtx.arc(tx, ty, 10, 0, Math.PI * 2);
  sessionCtx.fillStyle = "#f472b6"; sessionCtx.fill();
  sessionCtx.strokeStyle = "#fff"; sessionCtx.lineWidth = 1.5; sessionCtx.stroke();

  // Gaze crosshair + error line
  if ((Date.now() - state.lastGazeTime) < 500) {
    const gx = state.gazeX * W, gy = state.gazeY * H;
    sessionCtx.strokeStyle = "rgba(56,189,248,0.35)";
    sessionCtx.lineWidth = 1;
    sessionCtx.beginPath();
    sessionCtx.moveTo(gx - 14, gy); sessionCtx.lineTo(gx + 14, gy);
    sessionCtx.moveTo(gx, gy - 14); sessionCtx.lineTo(gx, gy + 14);
    sessionCtx.stroke();
    sessionCtx.beginPath(); sessionCtx.moveTo(gx, gy); sessionCtx.lineTo(tx, ty);
    sessionCtx.strokeStyle = "rgba(251,191,36,0.18)"; sessionCtx.stroke();
  }

  // "No gaze" warning
  if ((Date.now() - state.lastGazeTime) > 1500 && state.sessionActive) {
    sessionCtx.fillStyle = "rgba(248,113,113,0.8)";
    sessionCtx.font = "bold 15px system-ui";
    sessionCtx.textAlign = "center";
    sessionCtx.fillText("⚠ Face not detected — check camera", W / 2, H - 30);
  }
}

// ── Metrics UI ──────────────────────────────────────────────────
function updateMetricsPanel(d) {
  setBar("accuracy",  d.accuracy,  d.accuracy);
  setBar("smoothness", d.smoothness, d.smoothness);
  setBar("stability", d.stability, d.stability);
  document.getElementById("val-latency").textContent = "live";
  drawScoreRing(d.score);
  document.getElementById("score-number").textContent = Math.round(d.score);
  drawSparkline();
}

function setBar(metric, val, pct) {
  const bar = document.getElementById(`bar-${metric}`);
  const txt = document.getElementById(`val-${metric}`);
  bar.style.width      = Math.max(0, Math.min(100, pct)) + "%";
  bar.style.background = pct > 70 ? "#34d399" : pct > 40 ? "#fbbf24" : "#f87171";
  txt.textContent      = Math.round(pct) + "%";
}

function drawScoreRing(score) {
  const cx = 55, cy = 55, r = 42, lw = 8;
  scoreRingCtx.clearRect(0, 0, 110, 110);
  scoreRingCtx.beginPath();
  scoreRingCtx.arc(cx, cy, r, 0, Math.PI * 2);
  scoreRingCtx.strokeStyle = "#1a2235"; scoreRingCtx.lineWidth = lw; scoreRingCtx.stroke();
  const angle = -Math.PI / 2 + (score / 100) * Math.PI * 2;
  scoreRingCtx.beginPath();
  scoreRingCtx.arc(cx, cy, r, -Math.PI / 2, angle);
  scoreRingCtx.strokeStyle = score > 70 ? "#34d399" : score > 40 ? "#fbbf24" : "#f87171";
  scoreRingCtx.lineWidth = lw; scoreRingCtx.lineCap = "round"; scoreRingCtx.stroke();
}

function drawSparkline() {
  const W = scoreChartCanvas.width, H = scoreChartCanvas.height;
  const hist = state.scoreHistory;
  scoreChartCtx.clearRect(0, 0, W, H);
  if (hist.length < 2) return;
  scoreChartCtx.beginPath();
  hist.forEach((v, i) => {
    const x = (i / (hist.length - 1)) * W;
    const y = H - (v / 100) * H;
    i === 0 ? scoreChartCtx.moveTo(x, y) : scoreChartCtx.lineTo(x, y);
  });
  scoreChartCtx.strokeStyle = "#38bdf8"; scoreChartCtx.lineWidth = 1.5; scoreChartCtx.stroke();
  scoreChartCtx.lineTo(W, H); scoreChartCtx.lineTo(0, H); scoreChartCtx.closePath();
  scoreChartCtx.fillStyle = "rgba(56,189,248,0.07)"; scoreChartCtx.fill();
}

// ════════════════════════════════════════════════════════════════
//  RESULTS
// ════════════════════════════════════════════════════════════════
function showResults(summary) {
  // Use client-side metrics as ground truth if server returns blanks
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
document.getElementById("btn-export-json").addEventListener("click", () => {
  if (state.sessionId) window.open(`/api/export/${state.sessionId}/json`);
});
document.getElementById("btn-export-csv").addEventListener("click", () => {
  if (state.sessionId) window.open(`/api/export/${state.sessionId}/csv`);
});

// ════════════════════════════════════════════════════════════════
//  HISTORY
// ════════════════════════════════════════════════════════════════
async function showHistory() {
  const modal = document.getElementById("modal-history");
  const list  = document.getElementById("history-list");
  list.innerHTML = "<p style='color:#64748b'>Loading…</p>";
  modal.classList.remove("hidden");
  const sessions = await fetch("/api/sessions").then(r => r.json());
  list.innerHTML = sessions.length
    ? sessions.reverse().map(s => `
        <div class="history-row">
          <div>
            <div style="font-weight:600">${(s.mode||"—").toUpperCase()} · ${s.duration_s}s</div>
            <div style="color:#64748b;font-size:0.78rem">${s.session_id}</div>
          </div>
          <div class="history-score">${(s.final_score||0).toFixed(1)}</div>
        </div>`).join("")
    : "<p style='color:#64748b'>No sessions yet.</p>";
}

socket.on("connect_error", () => {
  cameraBadge.textContent = "❌ Server connection failed";
  cameraBadge.className   = "status-badge err";
});