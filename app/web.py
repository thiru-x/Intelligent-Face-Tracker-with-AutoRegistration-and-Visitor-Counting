"""Localhost-only browser dashboard for face visitor monitoring."""

import ipaddress
import json
import logging
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2

from app import database
from app.event_logger import configure_logging, flush_audit_outbox
from app.pipeline import VideoPipeline, load_config


LOGGER = logging.getLogger("katomaran.web")
MAX_REQUEST_BYTES = 8192

PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#101816">
  <meta name="description" content="Local-first face visitor monitoring dashboard">
  <title>Framewise — Local Visitor Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      font: 15px/1.5 Inter, "Segoe UI", system-ui, sans-serif;
      background: #101816;
      color: #edf4f0;
      --muted: #93a69c;
      --line: #293831;
      --panel: #17211c;
      --panel-raised: #1c2922;
      --mint: #a4f4c5;
      --mint-ink: #13241a;
      --amber: #f4ca7c;
      --red: #ff9e91;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(ellipse at 88% 0%, #22382b 0, transparent 36rem),
        #101816;
      min-height: 100vh;
    }
    button, input { font: inherit; }
    button:focus-visible, input:focus-visible, a:focus-visible {
      outline: 2px solid var(--mint);
      outline-offset: 3px;
    }
    .shell { width: min(1320px, 100% - 48px); margin: 0 auto; }
    .topbar {
      min-height: 76px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #26352d;
    }
    .brand { display: flex; gap: 11px; align-items: center; color: inherit;
      text-decoration: none; font-size: 1.02rem; font-weight: 750; }
    .brand-mark {
      width: 37px; height: 37px; border-radius: 12px; display: grid;
      place-items: center; color: var(--mint); background: #20382a;
    }
    .brand-copy { display: grid; line-height: 1.15; }
    .brand-copy small { color: var(--muted); font-size: .59rem;
      letter-spacing: .16em; margin-top: 5px; }
    .topbar-right { display: flex; gap: 10px; align-items: center; }
    .local-tag, .status {
      display: inline-flex; align-items: center; gap: 8px;
      border: 1px solid var(--line); border-radius: 999px;
      padding: 7px 12px; font-size: .75rem; font-weight: 650;
    }
    .local-tag { color: #b9e8c9; background: #192820; }
    .local-dot, .state-dot {
      width: 7px; height: 7px; border-radius: 50%; background: #829188;
    }
    .local-dot { background: var(--mint); }
    .status { color: #bdc9c1; text-transform: capitalize; }
    .status[data-state="running"] { color: var(--mint); border-color: #42694e; }
    .status[data-state="running"] .state-dot {
      background: var(--mint); box-shadow: 0 0 12px #a4f4c5a0;
      animation: pulse 1.7s ease-in-out infinite;
    }
    .status[data-state="error"] { color: var(--red); border-color: #69413b; }
    .status[data-state="error"] .state-dot { background: var(--red); }
    .status[data-state="reconnecting"],
    .status[data-state="starting"] { color: var(--amber); }
    .status[data-state="reconnecting"] .state-dot,
    .status[data-state="starting"] .state-dot { background: var(--amber); }
    main { padding: 38px 0 28px; }
    .intro {
      display: flex; align-items: flex-end; justify-content: space-between;
      gap: 24px; margin-bottom: 25px;
    }
    .eyebrow {
      margin: 0 0 9px; color: var(--mint); font-size: .72rem;
      font-weight: 750; letter-spacing: .17em; text-transform: uppercase;
    }
    h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.15rem);
      line-height: 1.05; letter-spacing: -.045em; font-weight: 680; }
    h1 em { color: var(--mint); font-style: normal; }
    .intro-copy { max-width: 620px; color: #a5b5ac; margin: 12px 0 0; }
    .model-tags { display: flex; gap: 7px; flex-wrap: wrap;
      justify-content: flex-end; padding-bottom: 3px; }
    .model-tags span {
      color: #afc0b5; border: 1px solid var(--line); border-radius: 7px;
      padding: 6px 9px; font-size: .67rem; letter-spacing: .055em;
    }
    .metrics {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 13px; margin-bottom: 17px;
    }
    .metric {
      position: relative; overflow: hidden; min-height: 116px;
      border: 1px solid var(--line); border-radius: 14px;
      background: linear-gradient(145deg, #1a261f, #15201a);
      padding: 17px 19px;
    }
    .metric::after {
      content: ""; position: absolute; width: 90px; height: 90px;
      border-radius: 50%; right: -36px; bottom: -58px;
      background: #a4f4c512;
    }
    .metric-label { display: flex; justify-content: space-between; gap: 8px;
      color: #a5b5ab; font-size: .78rem; }
    .metric-label span:last-child { color: #73887a; font-size: .65rem; }
    .metric strong { display: block; margin-top: 8px; font-size: 1.9rem;
      line-height: 1.05; letter-spacing: -.045em; font-variant-numeric: tabular-nums; }
    .metric strong small { color: var(--muted); font-size: .8rem;
      font-weight: 550; letter-spacing: 0; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.8fr)
      minmax(300px, .82fr); gap: 16px; align-items: stretch; }
    .panel {
      border: 1px solid var(--line); border-radius: 15px;
      background: linear-gradient(150deg, #1a251f, #17211c);
    }
    .feed-panel { padding: 17px; min-width: 0; }
    .panel-head { display: flex; align-items: center; justify-content: space-between;
      gap: 16px; margin-bottom: 14px; }
    .panel-title { margin: 0; font-size: .96rem; font-weight: 690; }
    .panel-subtitle { color: var(--muted); font-size: .72rem; margin-top: 3px; }
    .feed-badge { color: #a9b7ae; font-size: .7rem; }
    .feed-stage {
      min-height: 350px; position: relative; display: grid; place-items: center;
      overflow: hidden; border-radius: 11px;
      border: 1px solid #2b3c32;
      background:
        linear-gradient(#16221d9c, #16221d9c),
        linear-gradient(90deg, #25352c 1px, transparent 1px),
        linear-gradient(#25352c 1px, transparent 1px),
        #101914;
      background-size: auto, 36px 36px, 36px 36px, auto;
    }
    .feed-stage::before {
      content: ""; position: absolute; inset: 22px;
      border: 1px solid #a4f4c51a; border-radius: 5px; pointer-events: none;
    }
    .feed-stage img { width: 100%; height: 100%; max-height: 57vh;
      object-fit: contain; position: absolute; inset: 0; }
    .feed-live {
      position: absolute; top: 13px; left: 13px; z-index: 1;
      display: flex; align-items: center; gap: 7px; border-radius: 6px;
      background: #111914db; color: var(--mint); padding: 6px 9px;
      font-size: .65rem; font-weight: 750; letter-spacing: .08em;
    }
    .feed-live[hidden] { display: none; }
    .feed-live span { width: 6px; height: 6px; border-radius: 50%;
      background: var(--mint); }
    .placeholder {
      z-index: 1; display: grid; justify-items: center; gap: 10px;
      max-width: 310px; padding: 24px; color: #a6b6ac; text-align: center;
    }
    .placeholder[hidden] { display: none; }
    .placeholder-icon {
      width: 48px; height: 48px; display: grid; place-items: center;
      border: 1px solid #34483b; border-radius: 15px; color: var(--mint);
      background: #1c2b22;
    }
    .placeholder strong { color: #dce9e0; font-size: .9rem; }
    .placeholder span { font-size: .75rem; }
    .feed-stats { display: flex; flex-wrap: wrap; gap: 8px 21px;
      padding: 12px 2px 0; color: var(--muted); font-size: .72rem; }
    .feed-stats strong { color: #dce9e0; font-weight: 650;
      font-variant-numeric: tabular-nums; }
    .source-panel { padding: 18px; }
    .source-panel h2 { margin: 0; font-size: 1rem; }
    .source-panel > p { margin: 6px 0 17px; color: var(--muted); font-size: .75rem; }
    .tabs {
      display: grid; grid-template-columns: 1fr 1fr; gap: 4px;
      padding: 4px; border-radius: 9px; background: #111a15;
    }
    .tab {
      border: 0; border-radius: 7px; color: #9caea3; background: transparent;
      padding: 9px 7px; font-size: .73rem; font-weight: 650; cursor: pointer;
    }
    .tab[aria-selected="true"] { color: #e8f4ec; background: #293a2f; }
    .source-form[hidden] { display: none; }
    label { display: block; margin: 17px 0 7px; color: #d5e0d9;
      font-size: .75rem; font-weight: 600; }
    input {
      width: 100%; min-height: 43px; border: 1px solid #34463b;
      border-radius: 8px; background: #111a15; color: #edf4f0;
      padding: 10px 12px; font-size: .78rem;
    }
    input::placeholder { color: #697d70; }
    .field-help { display: block; color: #8fa397; margin-top: 7px;
      font-size: .67rem; line-height: 1.55; }
    .actions { display: grid; grid-template-columns: 1fr auto; gap: 9px;
      margin-top: 18px; }
    .button {
      min-height: 43px; border: 1px solid transparent; border-radius: 8px;
      padding: 9px 12px; cursor: pointer; font-size: .77rem;
      font-weight: 720; transition: background .18s, transform .18s;
    }
    .button:hover:not(:disabled) { transform: translateY(-1px); }
    .button:disabled { opacity: .45; cursor: not-allowed; }
    .button-primary { background: var(--mint); color: var(--mint-ink); }
    .button-primary:hover:not(:disabled) { background: #b9f9d1; }
    .button-stop { color: #ffc0b7; border-color: #68433d;
      background: #30211f; }
    .button-stop:hover:not(:disabled) { background: #402925; }
    .message { min-height: 36px; margin: 13px 0 0; color: var(--amber);
      font-size: .73rem; }
    .privacy-note {
      display: flex; align-items: flex-start; gap: 9px; margin-top: 13px;
      border-top: 1px solid var(--line); padding-top: 13px;
      color: #93a69a; font-size: .68rem;
    }
    .privacy-note svg { flex: 0 0 15px; color: var(--mint); }
    .events-panel { margin-top: 16px; padding: 18px; }
    .event-count {
      border: 1px solid var(--line); border-radius: 999px;
      color: #afc0b5; padding: 4px 9px; font-size: .67rem;
    }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; min-width: 620px; border-collapse: collapse;
      font-size: .74rem; }
    th, td { text-align: left; padding: 11px 9px;
      border-bottom: 1px solid #293831; vertical-align: middle; }
    th { color: #91a397; font-size: .65rem; font-weight: 650;
      letter-spacing: .07em; text-transform: uppercase; }
    td { color: #d9e4dc; }
    tbody tr:last-child td { border-bottom: 0; }
    .empty-cell { color: var(--muted); text-align: center; padding: 24px; }
    .event-type { display: inline-block; border-radius: 999px;
      padding: 3px 8px; font-size: .62rem; font-weight: 750; }
    .event-entry { color: var(--mint); background: #203a2b; }
    .event-exit { color: #ffc2b8; background: #3b2927; }
    .footer-note { display: flex; justify-content: space-between; gap: 16px;
      padding: 18px 2px 25px; color: #829489; font-size: .68rem; }
    @keyframes pulse { 50% { opacity: .48; } }
    @media (max-width: 900px) {
      .intro { align-items: flex-start; flex-direction: column; }
      .model-tags { justify-content: flex-start; }
      .workspace { grid-template-columns: minmax(0, 1fr) minmax(260px, .82fr); }
    }
    @media (max-width: 700px) {
      .shell { width: min(100% - 28px, 600px); }
      .topbar { min-height: 66px; }
      .brand-copy small { display: none; }
      .local-tag { font-size: 0; padding: 10px; }
      .local-tag .local-dot { width: 8px; height: 8px; }
      main { padding-top: 28px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; }
      .metric { min-height: 103px; padding: 14px; }
      .metric strong { font-size: 1.65rem; }
      .workspace { grid-template-columns: 1fr; }
      .feed-stage { min-height: 240px; }
      .events-panel { padding: 14px; }
      .footer-note { flex-direction: column; gap: 5px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: .01ms !important;
        transition-duration: .01ms !important; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="Framewise dashboard home">
        <span class="brand-mark" aria-hidden="true">
          <svg width="21" height="21" viewBox="0 0 24 24" fill="none">
            <path d="M4 8V5h3m10 0h3v3m0 8v3h-3M7 19H4v-3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
            <circle cx="12" cy="10" r="2.5" stroke="currentColor" stroke-width="1.6"/>
            <path d="M7.5 17c.6-2 2.1-3 4.5-3s3.9 1 4.5 3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
          </svg>
        </span>
        <span class="brand-copy">Framewise<small>VISITOR INTELLIGENCE</small></span>
      </a>
      <div class="topbar-right">
        <span class="local-tag"><span class="local-dot"></span>LOCAL ONLY</span>
        <span id="state" class="status" data-state="idle">
          <span class="state-dot"></span><span id="stateText">Idle</span>
        </span>
      </div>
    </header>
    <main>
      <section class="intro" aria-labelledby="page-title">
        <div>
          <p class="eyebrow">Live monitoring / local-first</p>
          <h1 id="page-title">Know who arrives.<br><em>Count every visit.</em></h1>
          <p class="intro-copy">Face detection, persistent identity matching and timestamped visitor events — processed on this machine.</p>
        </div>
        <div class="model-tags" aria-label="Processing components">
          <span>YOLO FACE DETECTION</span><span>INSIGHTFACE EMBEDDINGS</span><span>SQLITE EVENTS</span>
        </div>
      </section>
      <section class="metrics" aria-label="Visitor overview">
        <article class="metric">
          <div class="metric-label"><span>Unique identities</span><span>ALL TIME</span></div>
          <strong id="unique">0</strong>
        </article>
        <article class="metric">
          <div class="metric-label"><span>Total visits</span><span>ENTRY EVENTS</span></div>
          <strong id="visits">0</strong>
        </article>
        <article class="metric">
          <div class="metric-label"><span>Present now</span><span>THIS SOURCE</span></div>
          <strong id="present">0</strong>
        </article>
        <article class="metric">
          <div class="metric-label"><span>Processing rate</span><span>PROCESSED FPS</span></div>
          <strong><span id="fps">0.0</span><small> fps</small></strong>
        </article>
      </section>
      <section class="workspace" aria-label="Live feed and source controls">
        <article class="panel feed-panel">
          <div class="panel-head">
            <div><h2 class="panel-title">Processed camera feed</h2>
              <div class="panel-subtitle">Live annotated frames from the active source</div></div>
            <span id="feedLabel" class="feed-badge">Waiting for source</span>
          </div>
          <div class="feed-stage">
            <span id="feedLive" class="feed-live" hidden><span></span>LIVE FEED</span>
            <img id="stream" alt="Live processed camera or video feed" hidden>
            <div id="placeholder" class="placeholder">
              <span class="placeholder-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                  <rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" stroke-width="1.6"/>
                  <path d="m10 9 5 3-5 3V9Z" fill="currentColor"/>
                </svg>
              </span>
              <strong>Ready when you are</strong>
              <span>Choose a video, RTSP stream or local camera to start processing.</span>
            </div>
          </div>
          <div class="feed-stats">
            <span>Frames read <strong id="frames">0</strong></span>
            <span>Source FPS <strong id="sourceFps">—</strong></span>
            <span>Faces in frame <strong id="faces">0</strong></span>
          </div>
        </article>
        <aside class="panel source-panel">
          <h2>Video source</h2>
          <p>Choose how to connect the footage for this session.</p>
          <div class="tabs" role="tablist" aria-label="Source type">
            <button id="videoTab" class="tab" type="button" role="tab"
              aria-selected="true" aria-controls="videoForm">Video / RTSP</button>
            <button id="cameraTab" class="tab" type="button" role="tab"
              aria-selected="false" aria-controls="cameraForm">Webcam</button>
          </div>
          <div id="videoForm" class="source-form" role="tabpanel" aria-labelledby="videoTab">
            <label for="source">Video file path or RTSP URL</label>
            <input id="source" type="text" autocomplete="off"
              placeholder="C:\Videos\visitors.mp4 or rtsp://camera/stream">
            <small class="field-help">Leave blank to use the configured default source. RTSP credentials are used for this session and are not saved by the dashboard.</small>
          </div>
          <div id="cameraForm" class="source-form" role="tabpanel"
            aria-labelledby="cameraTab" hidden>
            <label for="camera">Camera device index</label>
            <input id="camera" type="number" min="0" step="1" value="0">
            <small class="field-help">Use 0 for the default webcam, or try another index if multiple cameras are connected.</small>
          </div>
          <div class="actions">
            <button id="start" class="button button-primary" type="button">Start processing</button>
            <button id="stop" class="button button-stop" type="button" disabled>Stop</button>
          </div>
          <p id="message" class="message" role="status" aria-live="polite"></p>
          <div class="privacy-note">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M12 3 5 6v5c0 4.5 2.8 8.2 7 10 4.2-1.8 7-5.5 7-10V6l-7-3Z" stroke="currentColor" stroke-width="1.7"/>
              <path d="m9 12 2 2 4-4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Local dashboard bound to 127.0.0.1. Stopping a source records an exit for active visits.</span>
          </div>
        </aside>
      </section>
      <section class="panel events-panel" aria-labelledby="events-title">
        <div class="panel-head">
          <div><h2 id="events-title" class="panel-title">Recent entry &amp; exit events</h2>
            <div class="panel-subtitle">Timestamped activity persisted in the local database</div></div>
          <span id="eventCount" class="event-count">0 events</span>
        </div>
        <div class="table-wrap">
          <table><thead><tr><th>Time (UTC)</th><th>Event</th><th>Face ID</th>
            <th>Frame</th><th>Media time</th></tr></thead>
            <tbody id="events"><tr><td colspan="5" class="empty-cell">No events recorded yet</td></tr></tbody>
          </table>
        </div>
      </section>
      <footer class="footer-note">
        <span>Detection and identity matching are separate stages. A tracker ID is temporary; a face ID persists in SQLite.</span>
        <span>Local-first · SQLite · Structured audit trail</span>
      </footer>
    </main>
  </div>
<script>
const $ = (id) => document.getElementById(id);
const message = (text) => { $("message").textContent = text || ""; };
async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Request failed");
  return body;
}
function renderEvents(events) {
  const body = $("events");
  body.replaceChildren();
  $("eventCount").textContent = `${events.length} ${events.length === 1 ? "event" : "events"}`;
  if (!events.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 5;
    cell.className = "empty-cell";
    cell.textContent = "No events recorded yet";
    return;
  }
  for (const event of events) {
    const row = body.insertRow();
    row.insertCell().textContent = event.timestamp;
    const typeCell = row.insertCell();
    const badge = document.createElement("span");
    badge.className = `event-type event-${event.event_type}`;
    badge.textContent = event.event_type.toUpperCase();
    typeCell.appendChild(badge);
    row.insertCell().textContent = event.face_id;
    row.insertCell().textContent = event.frame_number ?? "—";
    row.insertCell().textContent = event.source_timestamp == null
      ? "—" : `${Number(event.source_timestamp).toFixed(2)}s`;
  }
}
async function refresh() {
  try {
    const data = await request("/api/status");
    $("state").dataset.state = data.state;
    $("stateText").textContent = data.state;
    $("feedLabel").textContent = data.state === "running"
      ? "Source connected" : data.state === "reconnecting"
        ? "Reconnecting…" : data.state === "error"
          ? "Source error" : "Waiting for source";
    $("faces").textContent = data.faces_detected ?? 0;
    $("present").textContent = data.present ?? 0;
    $("unique").textContent = data.unique_identities ?? data.database_unique_identities;
    $("visits").textContent = data.visits ?? data.database_visits;
    $("fps").textContent = Number(data.processing_fps || 0).toFixed(1);
    $("frames").textContent = data.frames_read || 0;
    $("sourceFps").textContent = Number(data.source_fps || 0) > 0
      ? Number(data.source_fps).toFixed(1) : "—";
    $("start").disabled = ["starting", "running", "reconnecting", "stopping"].includes(data.state);
    $("stop").disabled = !["starting", "running", "reconnecting"].includes(data.state);
    $("stream").hidden = !data.has_frame;
    $("placeholder").hidden = data.has_frame;
    $("feedLive").hidden = !data.has_frame;
    if (data.has_frame && !$("stream").src) $("stream").src = "/video.mjpg";
    if (data.error) message(data.error);
    renderEvents(data.recent_events || []);
  } catch (error) {
    message(error.message);
  }
}
async function start(payload) {
  message("");
  try {
    await request("/api/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    $("stream").src = `/video.mjpg?t=${Date.now()}`;
    message("Connecting to the selected source…");
    await refresh();
  } catch (error) {
    message(error.message);
  }
}
function selectSource(sourceType) {
  const useCamera = sourceType === "camera";
  $("videoTab").setAttribute("aria-selected", String(!useCamera));
  $("cameraTab").setAttribute("aria-selected", String(useCamera));
  $("videoForm").hidden = useCamera;
  $("cameraForm").hidden = !useCamera;
}
$("videoTab").addEventListener("click", () => selectSource("video"));
$("cameraTab").addEventListener("click", () => selectSource("camera"));
$("start").addEventListener("click", () => {
  if (!$("cameraForm").hidden) {
    const camera = Number($("camera").value);
    if (!Number.isInteger(camera) || camera < 0) {
      message("Enter a non-negative camera index.");
      return;
    }
    start({camera});
    return;
  }
  const source = $("source").value.trim();
  start(source ? {source} : {});
});
$("source").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !$("start").disabled) $("start").click();
});
$("camera").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !$("start").disabled) $("start").click();
});
$("stop").addEventListener("click", async () => {
  try {
    await request("/api/stop", {method: "POST"});
    message("Stop requested. Active visits will be closed.");
  } catch (error) { message(error.message); }
});
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""


class DashboardController:
    """Own one background video-processing worker and latest JPEG frame."""

    def __init__(self):
        config = load_config()
        configure_logging(
            config.get("log_directory", "logs"),
            config.get("face_image_quality", 95),
        )
        database.configure_database_path(
            config.get("database_path", "data/visitors.db")
        )
        database.initialize_database()
        flush_audit_outbox()
        self.lock = threading.RLock()
        self.status = {
            "state": "idle",
            "frames_read": 0,
            "frames_processed": 0,
            "faces_detected": 0,
            "present": 0,
            "unique_this_run": 0,
            "unique_identities": database.get_unique_visitor_count(),
            "visits": database.get_visit_count(),
            "source_fps": 0.0,
            "processing_fps": 0.0,
            "error": None,
        }
        self.latest_frame = None
        self.frame_version = 0
        self.frame_condition = threading.Condition(self.lock)
        self.stop_event = None
        self.worker = None

    def snapshot(self):
        with self.lock:
            result = dict(self.status)
            result["has_frame"] = self.latest_frame is not None
        result["database_unique_identities"] = (
            database.get_unique_visitor_count()
        )
        result["database_visits"] = database.get_visit_count()
        result["recent_events"] = database.get_recent_events(20)
        return result

    def start(self, source=None, camera=None):
        if camera is not None:
            if isinstance(camera, bool) or not isinstance(camera, int) or camera < 0:
                raise ValueError("camera must be a non-negative integer")
            video_source = camera
        elif source is None:
            video_source = None
        elif isinstance(source, str) and source.strip():
            video_source = source.strip()
        else:
            raise ValueError("source must be a non-empty string")

        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise RuntimeError("Video processing is already active")

            self.stop_event = threading.Event()
            stop_event = self.stop_event
            self.latest_frame = None
            self.status = {
                **self.status,
                "state": "starting",
                "error": None,
                "frames_read": 0,
                "frames_processed": 0,
                "faces_detected": 0,
                "present": 0,
                "unique_this_run": 0,
                "unique_identities": database.get_unique_visitor_count(),
                "visits": database.get_visit_count(),
                "processing_fps": 0.0,
            }
            self.worker = threading.Thread(
                target=self._run,
                args=(video_source, stop_event),
                name="visitor-video-pipeline",
                daemon=True,
            )
            self.worker.start()

    def _run(self, source, stop_event):
        try:
            pipeline = VideoPipeline()
            with self.lock:
                if self.status["state"] == "starting":
                    self.status["state"] = "running"
            pipeline.process_video(
                video_source=source,
                display=False,
                frame_callback=self._publish_frame,
                status_callback=self._publish_status,
                stop_event=stop_event,
            )
        except Exception as exc:
            LOGGER.error(
                "Browser video processing failed (%s)",
                type(exc).__name__,
            )
            with self.lock:
                self.status["state"] = "error"
                if VideoPipeline._is_network_source(source):
                    self.status["error"] = (
                        "Network camera stream failed. Check the RTSP URL, "
                        "credentials, network, camera availability, and "
                        "configured reconnect limits."
                    )
                else:
                    self.status["error"] = str(exc)
        finally:
            with self.lock:
                if self.status["state"] in {"running", "stopping", "starting"}:
                    self.status["state"] = "stopped"
                self.status["unique_identities"] = (
                    database.get_unique_visitor_count()
                )
                self.status["visits"] = database.get_visit_count()
                self.frame_condition.notify_all()

    def _publish_frame(self, frame):
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if not success:
            raise RuntimeError("Could not encode a video frame for the browser")
        with self.frame_condition:
            self.latest_frame = encoded.tobytes()
            self.frame_version += 1
            self.frame_condition.notify_all()

    def _publish_status(self, status):
        with self.lock:
            if self.status["state"] not in {"stopping", "error"}:
                self.status.update(status)

    def stop(self):
        with self.lock:
            if self.worker is None or not self.worker.is_alive():
                return False
            if self.stop_event is not None:
                self.stop_event.set()
            self.status["state"] = "stopping"
            self.frame_condition.notify_all()
            return True

    def next_frame(self, previous_version, timeout=1.0):
        with self.frame_condition:
            self.frame_condition.wait_for(
                lambda: (
                    self.frame_version > previous_version
                    or self.status["state"] in {"stopped", "error", "idle"}
                ),
                timeout=timeout,
            )
            if self.frame_version <= previous_version:
                return None, previous_version
            return self.latest_frame, self.frame_version

    def shutdown(self):
        self.stop()
        worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=10)


def _is_loopback_host(host):
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_server(host="127.0.0.1", port=8000, controller=None):
    if not _is_loopback_host(host):
        raise ValueError("The dashboard can only bind to a localhost address")
    if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")

    controller = controller or DashboardController()

    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "LocalVisitorDashboard/1.0"

        def _allow_local_host(self):
            host_header = self.headers.get("Host", "")
            hostname = urlsplit("//" + host_header).hostname
            if hostname and _is_loopback_host(hostname):
                return True
            self._send_json({"error": "Only localhost requests are allowed"}, 403)
            return False

        def log_message(self, format_string, *args):
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

        def _send_json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length") from exc
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body must be between 1 and 8192 bytes")
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("Request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object")
            return payload

        def do_GET(self):
            if not self._allow_local_host():
                return
            path = urlsplit(self.path).path
            if path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/html; charset=utf-8"
                )
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/status":
                self._send_json(controller.snapshot())
                return

            if path == "/video.mjpg":
                self._stream_video()
                return

            self._send_json({"error": "Not found"}, status=404)

        def _stream_video(self):
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            version = 0
            try:
                while True:
                    frame, version = controller.next_frame(version, timeout=1)
                    if frame is None:
                        if controller.snapshot()["state"] in {
                            "stopped", "error", "idle"
                        }:
                            break
                        continue
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: "
                        + str(len(frame)).encode("ascii")
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            if not self._allow_local_host():
                return
            path = urlsplit(self.path).path
            if path == "/api/start":
                try:
                    payload = self._read_json()
                    unknown = set(payload) - {"source", "camera"}
                    if unknown:
                        raise ValueError("Only source or camera is allowed")
                    if "source" in payload and "camera" in payload:
                        raise ValueError("Choose either a source or camera")
                    if "source" in payload and (
                        not isinstance(payload["source"], str)
                        or not payload["source"].strip()
                    ):
                        raise ValueError("source must be a non-empty string")
                    if "camera" in payload and (
                        isinstance(payload["camera"], bool)
                        or not isinstance(payload["camera"], int)
                        or payload["camera"] < 0
                    ):
                        raise ValueError(
                            "camera must be a non-negative integer"
                        )
                    controller.start(
                        source=payload.get("source"),
                        camera=payload.get("camera"),
                    )
                except (ValueError, RuntimeError) as exc:
                    self._send_json({"error": str(exc)}, status=400)
                    return
                self._send_json({"state": "starting"}, status=202)
                return

            if path == "/api/stop":
                started = controller.stop()
                self._send_json({
                    "state": "stopping" if started else controller.snapshot()["state"]
                })
                return

            self._send_json({"error": "Not found"}, status=404)

    class LocalThreadingHTTPServer(ThreadingHTTPServer):
        address_family = (
            socket.AF_INET6 if ":" in host else socket.AF_INET
        )

    server = LocalThreadingHTTPServer((host, port), RequestHandler)
    server.daemon_threads = True
    server.controller = controller
    return server


def run_server(host="127.0.0.1", port=8000, open_browser=True):
    server = create_server(host, port)
    address, actual_port = server.server_address[:2]
    address_part = f"[{address}]" if ":" in address else address
    url = f"http://{address_part}:{actual_port}/"
    print(f"Local visitor dashboard: {url}")
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Dashboard shutdown requested")
    finally:
        server.controller.shutdown()
        server.server_close()
    return url


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()
