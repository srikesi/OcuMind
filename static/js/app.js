/**
 * app.js — OcuMind frontend
 *
 * Manages:
 *  - SocketIO connection to Flask backend
 *  - Screen transitions (welcome → calibration → session → results)
 *  - 9-point calibration UI
 *  - Stimulus animation engine (circular, linear, random, figure8)
 *  - Live gaze dot rendering
 *  - Real-time metrics panel + score ring + sparkline chart
 */

"use strict";

// ── SocketIO connection ─────────────────────────────────────────
const socket = io({ transports: ["websocket"] });

// ── State ───────────────────────────────────────────────────────
const state = {
  screen:       "welcome",
  mode:         "circular",
  calibrated:   false,
  gazeX: 0.5,   gazeY: 0.5,
  gazeDetected: false,
  sessionActive: false,
  sessionStart:  null,
  sessionId:     null,
  scoreHistory:  [],
  lastMetrics:   {},
};

// ── DOM references ───────────────────────────────────────────────
const screens = {
  welcome:   document.getElementById("screen-welcome"),
  calibrate: document.getElementById("screen-calibrate"),
  session:   document.getElementById("screen-session"),
  results:   document.getElementById("screen-results"),
};
const calibCanvas   = document.getElementById("calib-canvas");
const calibCtx      = calibCanvas.getContext("2d");
const sessionCanvas = document.getElementById("session-canvas");
const sessionCtx    = sessionCanvas.getContext("2d");
const gazeCalibDot  = document.getElementById("gaze-dot");
const gazeSessionDot= document.getElementById("session-gaze-dot");
const scoreRingCanvas = document.getElementById("score-ring");
const scoreRingCtx    = scoreRingCanvas.getContext("2d");
const scoreChartCanvas = document.getElementById("score-chart");
const scoreChartCtx    = scoreChartCanvas.getContext("2d");

// ── Resize canvases to window ────────────────────────────────────
function resizeCanvases() {
  calibCanvas.width  = window.innerWidth;
  calibCanvas.height = window.innerHeight;
  sessionCanvas.width  = window.innerWidth;
  sessionCanvas.height = window.innerHeight;
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

// ════════════════════════════════════════════════════════════════
//  SCREEN NAVIGATION
// ════════════════════════════════════════════════════════════════
function showScreen(name) {
  Object.entries(screens).forEach(([k, el]) => {
    el.classList.toggle("active", k === name);
  });
  state.screen = name;
}

// ════════════════════════════════════════════════════════════════
//  WELCOME SCREEN
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

btnCalibrate.addEventListener("click", () => startCalibration());
btnSkip.addEventListener("click", () => {
  state.calibrated = false;
  startSession();
});
btnHistory.addEventListener("click", () => showHistory());
document.getElementById("btn-close-history").addEventListener("click", () => {
  document.getElementById("modal-history").classList.add("hidden");
});

// Camera startup feedback
socket.on("connect", () => {
  socket.emit("start_camera");
});
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
//  GAZE DOT (raw broadcast — shown everywhere)
// ════════════════════════════════════════════════════════════════
socket.on("gaze_raw", (data) => {
  state.gazeDetected = data.detected;
  if (!data.detected) return;
  state.gazeX = data.x;
  state.gazeY = data.y;
  updateGazeDots(data.x, data.y);
});

function updateGazeDots(nx, ny) {
  const px = nx * window.innerWidth;
  const py = ny * window.innerHeight;
  const dots = [gazeCalibDot, gazeSessionDot];
  dots.forEach(d => {
    d.style.left = px + "px";
    d.style.top  = py + "px";
    d.classList.remove("hidden");
  });
}

// ════════════════════════════════════════════════════════════════
//  CALIBRATION
// ════════════════════════════════════════════════════════════════
const CALIB_POINTS_9 = [
  [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
  [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
  [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
];

let calibState = {
  pointIndex:   0,
  phase:        "idle",      // idle | countdown | collecting | done
  countdownVal: 3,
  countdownTimer: null,
  collectTimer:   null,
};

function startCalibration() {
  socket.emit("calib_reset");
  calibState.pointIndex = 0;
  calibState.phase      = "idle";
  showScreen("calibrate");
  gazeCalibDot.classList.remove("hidden");
  setTimeout(() => showNextCalibPoint(), 500);
}

function showNextCalibPoint() {
  const idx = calibState.pointIndex;
  if (idx >= CALIB_POINTS_9.length) {
    finishCalibration();
    return;
  }
  document.getElementById("calib-progress").textContent =
    `Point ${idx + 1} / ${CALIB_POINTS_9.length}`;
  calibState.phase = "countdown";
  calibState.countdownVal = 3;
  drawCalibPoint(idx, "waiting");
  runCountdown(idx);
}

function runCountdown(idx) {
  document.getElementById("calib-instruction").textContent =
    `Look at the dot — ${calibState.countdownVal}`;
  calibState.countdownVal--;
  if (calibState.countdownVal >= 0) {
    calibState.countdownTimer = setTimeout(() => runCountdown(idx), 700);
  } else {
    // Countdown done → collect
    calibState.phase = "collecting";
    drawCalibPoint(idx, "active");
    document.getElementById("calib-instruction").textContent =
      "Hold still…";

    const [sx, sy] = CALIB_POINTS_9[idx];
    socket.emit("calib_start_point", { x: sx, y: sy });

    // Commit after 1.2 s of collection
    calibState.collectTimer = setTimeout(() => {
      socket.emit("calib_commit_point");
    }, 1200);
  }
}

socket.on("calib_point_done", () => {
  calibState.pointIndex++;
  calibState.phase = "idle";
  setTimeout(() => showNextCalibPoint(), 400);
});

// Auto-commit if backend buffer fills first
socket.on("calib_auto_commit", () => {
  clearTimeout(calibState.collectTimer);
  socket.emit("calib_commit_point");
});

function finishCalibration() {
  document.getElementById("calib-instruction").textContent = "Calibrating…";
  socket.emit("calib_finish");
}

socket.on("calib_result", ({ ok, n_samples }) => {
  if (ok) {
    state.calibrated = true;
    document.getElementById("calib-instruction").textContent =
      "✅ Calibration complete!";
    setTimeout(() => startSession(), 900);
  } else {
    document.getElementById("calib-instruction").textContent =
      `⚠️ Calibration failed (${n_samples} samples). Try again.`;
    setTimeout(() => showScreen("welcome"), 2000);
  }
});

// Draw calibration target dot on canvas
function drawCalibPoint(idx, phase) {
  const W = calibCanvas.width, H = calibCanvas.height;
  calibCtx.clearRect(0, 0, W, H);

  // Draw all dots dimly
  CALIB_POINTS_9.forEach(([nx, ny], i) => {
    const x = nx * W, y = ny * H;
    calibCtx.beginPath();
    calibCtx.arc(x, y, 6, 0, Math.PI * 2);
    calibCtx.fillStyle = i < idx ? "#34d399" :
                         i === idx && phase === "active" ? "#38bdf8" :
                         i === idx ? "#fbbf24" : "#2a3a55";
    calibCtx.fill();
  });

  // Big animated ring on active point
  const [nx, ny] = CALIB_POINTS_9[idx];
  const x = nx * W, y = ny * H;
  calibCtx.beginPath();
  calibCtx.arc(x, y, phase === "active" ? 18 : 24, 0, Math.PI * 2);
  calibCtx.strokeStyle = phase === "active" ? "#38bdf8" : "#fbbf24";
  calibCtx.lineWidth = 2;
  calibCtx.stroke();
}

// ════════════════════════════════════════════════════════════════
//  SESSION / STIMULUS ENGINE
// ════════════════════════════════════════════════════════════════
let animFrame    = null;
let sessionTimer = null;
const SESSION_DURATION_S = 60;   // default 60 s

// Stimulus state
const target = { x: 0.5, y: 0.5, moving: true };

// Pattern generators — return {x, y} in [0,1] given elapsed seconds
const patterns = {
  circular: (t) => ({
    x: 0.5 + 0.35 * Math.cos(t * 0.6),
    y: 0.5 + 0.30 * Math.sin(t * 0.6),
    moving: true,
  }),
  linear: (t) => {
    const p = (t * 0.25) % 2;   // 0→1→0 sawtooth
    const x = p < 1 ? p : 2 - p;
    return { x: 0.1 + x * 0.8, y: 0.5, moving: true };
  },
  figure8: (t) => ({
    x: 0.5 + 0.38 * Math.sin(t * 0.5),
    y: 0.5 + 0.22 * Math.sin(t * 1.0),
    moving: true,
  }),
  random: (() => {
    let nextX = 0.5, nextY = 0.5, lastSwitch = 0;
    return (t) => {
      if (t - lastSwitch > 1.5 + Math.random() * 2) {
        nextX = 0.15 + Math.random() * 0.7;
        nextY = 0.15 + Math.random() * 0.7;
        lastSwitch = t;
      }
      return { x: nextX, y: nextY, moving: false };
    };
  })(),
};

function startSession() {
  state.sessionActive = true;
  state.sessionStart  = performance.now();
  state.scoreHistory  = [];

  socket.emit("session_start", { mode: state.mode });
  showScreen("session");
  document.getElementById("hud-mode").textContent = state.mode.toUpperCase();

  // Reset sparkline background
  scoreChartCtx.clearRect(0, 0, scoreChartCanvas.width, scoreChartCanvas.height);

  animFrame = requestAnimationFrame(renderSession);

  sessionTimer = setTimeout(() => stopSession(), SESSION_DURATION_S * 1000);
}

document.getElementById("btn-stop-session").addEventListener("click", stopSession);

function stopSession() {
  state.sessionActive = false;
  clearTimeout(sessionTimer);
  cancelAnimationFrame(animFrame);
  socket.emit("session_stop");
}

socket.on("session_summary", (summary) => {
  showResults(summary);
});

// ─── Main animation loop ────────────────────────────────────────
function renderSession(now) {
  if (!state.sessionActive) return;

  const elapsed = (now - state.sessionStart) / 1000;

  // Update timer display
  const mins = Math.floor(elapsed / 60);
  const secs = Math.floor(elapsed % 60).toString().padStart(2, "0");
  document.getElementById("hud-time").textContent = `${mins}:${secs}`;

  // Compute target position from selected pattern
  const pattern = patterns[state.mode] || patterns.circular;
  const pos = pattern(elapsed);
  target.x = pos.x;
  target.y = pos.y;
  target.moving = pos.moving;

  drawSessionFrame(elapsed);

  // Send target to backend for metrics computation
  socket.emit("session_frame", {
    target_x: target.x,
    target_y: target.y,
    moving:   target.moving,
    elapsed:  elapsed,
  });

  animFrame = requestAnimationFrame(renderSession);
}

// ─── Canvas drawing ─────────────────────────────────────────────
function drawSessionFrame(elapsed) {
  const W = sessionCanvas.width, H = sessionCanvas.height;
  sessionCtx.clearRect(0, 0, W, H);

  // Subtle grid
  sessionCtx.strokeStyle = "rgba(42,58,85,0.3)";
  sessionCtx.lineWidth   = 1;
  for (let i = 1; i < 10; i++) {
    sessionCtx.beginPath();
    sessionCtx.moveTo(i * W / 10, 0);
    sessionCtx.lineTo(i * W / 10, H);
    sessionCtx.stroke();
    sessionCtx.beginPath();
    sessionCtx.moveTo(0, i * H / 10);
    sessionCtx.lineTo(W, i * H / 10);
    sessionCtx.stroke();
  }

  // Draw ghost trail of target path
  const TRAIL = 40;
  for (let i = 0; i < TRAIL; i++) {
    const t2 = elapsed - (TRAIL - i) * 0.025;
    const pp = (patterns[state.mode] || patterns.circular)(t2);
    const alpha = i / TRAIL * 0.25;
    sessionCtx.beginPath();
    sessionCtx.arc(pp.x * W, pp.y * H, 5, 0, Math.PI * 2);
    sessionCtx.fillStyle = `rgba(244,114,182,${alpha})`;
    sessionCtx.fill();
  }

  // Target dot (pink, pulsing)
  const pulse = 1 + 0.15 * Math.sin(elapsed * 5);
  const tx = target.x * W, ty = target.y * H;

  const grad = sessionCtx.createRadialGradient(tx, ty, 0, tx, ty, 22 * pulse);
  grad.addColorStop(0, "rgba(244,114,182,0.9)");
  grad.addColorStop(1, "rgba(244,114,182,0)");
  sessionCtx.beginPath();
  sessionCtx.arc(tx, ty, 22 * pulse, 0, Math.PI * 2);
  sessionCtx.fillStyle = grad;
  sessionCtx.fill();

  sessionCtx.beginPath();
  sessionCtx.arc(tx, ty, 10, 0, Math.PI * 2);
  sessionCtx.fillStyle = "#f472b6";
  sessionCtx.fill();
  sessionCtx.strokeStyle = "#fff";
  sessionCtx.lineWidth = 1.5;
  sessionCtx.stroke();

  // Gaze crosshair (if detected)
  if (state.gazeDetected) {
    const gx = state.gazeX * W, gy = state.gazeY * H;
    sessionCtx.strokeStyle = "rgba(56,189,248,0.4)";
    sessionCtx.lineWidth = 1;
    sessionCtx.beginPath();
    sessionCtx.moveTo(gx - 14, gy); sessionCtx.lineTo(gx + 14, gy);
    sessionCtx.moveTo(gx, gy - 14); sessionCtx.lineTo(gx, gy + 14);
    sessionCtx.stroke();

    // Error line gaze→target
    sessionCtx.beginPath();
    sessionCtx.moveTo(gx, gy);
    sessionCtx.lineTo(tx, ty);
    sessionCtx.strokeStyle = "rgba(251,191,36,0.2)";
    sessionCtx.lineWidth = 1;
    sessionCtx.stroke();
  }
}

// ─── Metrics update from backend ────────────────────────────────
socket.on("metrics_update", (data) => {
  state.lastMetrics = data;
  state.gazeX = data.gaze_x;
  state.gazeY = data.gaze_y;
  updateGazeDots(data.gaze_x, data.gaze_y);
  updateMetricsPanel(data);
});

function updateMetricsPanel(d) {
  // Bars
  setBar("accuracy",   d.accuracy,   d.accuracy);
  setBar("smoothness", d.smoothness, d.smoothness);
  setBar("stability",  d.stability,  d.stability);

  // Latency text
  document.getElementById("val-latency").textContent = d.latency_ms.toFixed(0) + " ms";

  // Score ring
  drawScoreRing(d.score);
  document.getElementById("score-number").textContent = Math.round(d.score);

  // Sparkline
  state.scoreHistory.push(d.score);
  if (state.scoreHistory.length > 200) state.scoreHistory.shift();
  drawSparkline();
}

function setBar(metric, val, pct) {
  const bar = document.getElementById(`bar-${metric}`);
  const txt = document.getElementById(`val-${metric}`);
  const w   = Math.max(0, Math.min(100, pct));
  bar.style.width = w + "%";
  bar.style.background = w > 70 ? "#34d399" : w > 40 ? "#fbbf24" : "#f87171";
  txt.textContent = Math.round(pct) + "%";
}

// Score ring (arc)
function drawScoreRing(score) {
  const W = 110, H = 110, cx = 55, cy = 55, r = 42, lw = 8;
  scoreRingCtx.clearRect(0, 0, W, H);

  // Background track
  scoreRingCtx.beginPath();
  scoreRingCtx.arc(cx, cy, r, 0, Math.PI * 2);
  scoreRingCtx.strokeStyle = "#1a2235";
  scoreRingCtx.lineWidth   = lw;
  scoreRingCtx.stroke();

  // Filled arc
  const angle = -Math.PI / 2 + (score / 100) * Math.PI * 2;
  const color = score > 70 ? "#34d399" : score > 40 ? "#fbbf24" : "#f87171";
  scoreRingCtx.beginPath();
  scoreRingCtx.arc(cx, cy, r, -Math.PI / 2, angle);
  scoreRingCtx.strokeStyle = color;
  scoreRingCtx.lineWidth   = lw;
  scoreRingCtx.lineCap     = "round";
  scoreRingCtx.stroke();
}

// Sparkline chart
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
  scoreChartCtx.strokeStyle = "#38bdf8";
  scoreChartCtx.lineWidth   = 1.5;
  scoreChartCtx.stroke();

  // Fill
  scoreChartCtx.lineTo(W, H); scoreChartCtx.lineTo(0, H);
  scoreChartCtx.closePath();
  scoreChartCtx.fillStyle = "rgba(56,189,248,0.08)";
  scoreChartCtx.fill();
}

// ════════════════════════════════════════════════════════════════
//  RESULTS
// ════════════════════════════════════════════════════════════════
function showResults(summary) {
  showScreen("results");
  document.getElementById("res-score").textContent  = summary.final_score?.toFixed(1) ?? "—";
  document.getElementById("res-acc").textContent    = (summary.accuracy ?? "—") + "%";
  document.getElementById("res-smooth").textContent = (summary.smoothness ?? "—") + "%";
  document.getElementById("res-stab").textContent   = (summary.stability ?? "—") + "%";
  document.getElementById("res-lat").textContent    = (summary.latency_ms ?? "—") + " ms";
  document.getElementById("res-dur").textContent    = (summary.duration_s ?? "—") + " s";
  state.sessionId = summary.session_id;
}

document.getElementById("btn-new-session").addEventListener("click", () => {
  showScreen("welcome");
});
document.getElementById("btn-export-json").addEventListener("click", () => {
  if (state.sessionId)
    window.open(`/api/export/${state.sessionId}/json`);
});
document.getElementById("btn-export-csv").addEventListener("click", () => {
  if (state.sessionId)
    window.open(`/api/export/${state.sessionId}/csv`);
});

// ════════════════════════════════════════════════════════════════
//  HISTORY
// ════════════════════════════════════════════════════════════════
async function showHistory() {
  const modal = document.getElementById("modal-history");
  const list  = document.getElementById("history-list");
  list.innerHTML = "<p style='color:#64748b'>Loading…</p>";
  modal.classList.remove("hidden");

  const resp = await fetch("/api/sessions");
  const sessions = await resp.json();

  if (!sessions.length) {
    list.innerHTML = "<p style='color:#64748b'>No sessions yet.</p>";
    return;
  }
  list.innerHTML = sessions.reverse().map(s => `
    <div class="history-row">
      <div>
        <div style="font-weight:600">${s.mode?.toUpperCase()} · ${s.duration_s}s</div>
        <div style="color:#64748b;font-size:0.78rem">${s.session_id}</div>
      </div>
      <div class="history-score">${s.final_score?.toFixed(1)}</div>
    </div>
  `).join("");
}

// ════════════════════════════════════════════════════════════════
//  MISC
// ════════════════════════════════════════════════════════════════
socket.on("disconnect", () => {
  console.warn("[OcuMind] Socket disconnected");
});
socket.on("connect_error", (err) => {
  cameraBadge.textContent = "❌ Server connection failed";
  cameraBadge.className   = "status-badge err";
});
