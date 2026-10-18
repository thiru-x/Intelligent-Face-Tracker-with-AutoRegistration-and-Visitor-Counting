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

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Visitor Monitor</title>
  <style>
    :root { color-scheme: dark; font: 16px/1.5 system-ui, sans-serif;
      background: #10151b; color: #e7edf4; }
    body { margin: 0 auto; max-width: 1180px; padding: 28px 20px; }
    h1 { margin: 0; font-size: 1.7rem; }
    .subtle, small { color: #9ba9b8; }
    header { display: flex; justify-content: space-between; gap: 20px;
      align-items: center; margin-bottom: 22px; }
    .status { border: 1px solid #3c4b5a; border-radius: 20px;
      padding: 5px 12px; text-transform: capitalize; }
    .layout { display: grid; grid-template-columns: minmax(0, 2fr)
      minmax(260px, 1fr); gap: 18px; }
    section { background: #19212a; border: 1px solid #293542;
      border-radius: 12px; padding: 18px; }
    .video { min-height: 310px; display: grid; place-items: center;
      background: #080b0f; border-radius: 8px; overflow: hidden; }
    .video img { width: 100%; max-height: 65vh; object-fit: contain; }
    .metrics { display: grid; grid-template-columns: repeat(2, 1fr);
      gap: 10px; margin-top: 16px; }
    .metric { background: #111820; border-radius: 8px; padding: 12px; }
    .metric strong { display: block; font-size: 1.55rem; }
    label { display: block; font-size: .9rem; margin: 14px 0 5px; }
    input { box-sizing: border-box; width: 100%; padding: 10px;
      border: 1px solid #465564; border-radius: 6px;
      background: #0f151c; color: inherit; }
    button { border: 0; border-radius: 6px; padding: 10px 15px;
      margin: 12px 8px 0 0; cursor: pointer; color: #08121b;
      background: #6fd3b0; font-weight: 700; }
    button.stop { background: #ff987f; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    .message { min-height: 24px; color: #ffd38b; margin-top: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: .88rem; }
    th, td { padding: 9px 5px; border-bottom: 1px solid #303b47;
      text-align: left; vertical-align: top; }
    th { color: #aebdcb; }
    .events { margin-top: 18px; overflow-x: auto; }
    @media (max-width: 760px) {
      .layout { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
      .video { min-height: 200px; }
    }
  </style>
</head>
<body>
  <header>
    <div><h1>Local Visitor Monitor</h1>
      <div class="subtle">Video and visitor data stay on this machine.</div>
    </div>
    <span id="state" class="status">idle</span>
  </header>
  <main class="layout">
    <section>
      <div class="video">
        <img id="stream" alt="Live processed camera/video feed" hidden>
        <span id="placeholder" class="subtle">Start a video source to view output</span>
      </div>
      <div class="metrics">
        <div class="metric"><small>Faces detected</small><strong id="faces">0</strong></div>
        <div class="metric"><small>Currently present</small><strong id="present">0</strong></div>
        <div class="metric"><small>Unique identities</small><strong id="unique">0</strong></div>
        <div class="metric"><small>Total visits</small><strong id="visits">0</strong></div>
      </div>
      <p class="subtle">Processed FPS: <strong id="fps">0.0</strong>
        &nbsp; | &nbsp; Frames read: <strong id="frames">0</strong></p>
    </section>
    <section>
      <h2>Video source</h2>
      <label for="source">Video path or RTSP stream URL</label>
      <input id="source" placeholder="rtsp://camera-address:554/stream1">
      <small>RTSP credentials are used locally and are not saved in the dashboard.</small>
      <button id="start">Start processing</button>
      <hr>
      <label for="camera">Local camera index</label>
      <input id="camera" type="number" min="0" step="1" value="0">
      <button id="cameraStart">Start camera</button>
      <button id="stop" class="stop" disabled>Stop</button>
      <div id="message" class="message" role="status"></div>
      <p class="subtle">The server binds to 127.0.0.1 only. Press Stop to close
        an active visit and record its exit.</p>
    </section>
  </main>
  <section class="events">
    <h2>Recent entry/exit events</h2>
    <table><thead><tr><th>Time (UTC)</th><th>Type</th><th>Face ID</th>
      <th>Frame</th><th>Media time</th></tr></thead>
      <tbody id="events"><tr><td colspan="5" class="subtle">No events yet</td></tr></tbody>
    </table>
  </section>
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
  if (!events.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 5;
    cell.className = "subtle";
    cell.textContent = "No events yet";
    return;
  }
  for (const event of events) {
    const row = body.insertRow();
    for (const value of [
      event.timestamp,
      event.event_type.toUpperCase(),
      event.face_id,
      event.frame_number ?? "-",
      event.source_timestamp == null ? "-" : `${event.source_timestamp.toFixed(2)}s`
    ]) row.insertCell().textContent = value;
  }
}
async function refresh() {
  try {
    const data = await request("/api/status");
    $("state").textContent = data.state;
    $("faces").textContent = data.faces_detected ?? 0;
    $("present").textContent = data.present ?? 0;
    $("unique").textContent = data.unique_identities ?? data.database_unique_identities;
    $("visits").textContent = data.visits ?? data.database_visits;
    $("fps").textContent = Number(data.processing_fps || 0).toFixed(1);
    $("frames").textContent = data.frames_read || 0;
    $("start").disabled = ["starting", "running", "reconnecting", "stopping"].includes(data.state);
    $("cameraStart").disabled = $("start").disabled;
    $("stop").disabled = !["starting", "running", "reconnecting"].includes(data.state);
    $("stream").hidden = !data.has_frame;
    $("placeholder").hidden = data.has_frame;
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
    message("Processing started.");
    await refresh();
  } catch (error) {
    message(error.message);
  }
}
$("start").addEventListener("click", () => {
  const source = $("source").value.trim();
  start(source ? {source} : {});
});
$("cameraStart").addEventListener("click", () => {
  const camera = Number($("camera").value);
  if (!Number.isInteger(camera) || camera < 0) {
    message("Enter a non-negative camera index.");
    return;
  }
  start({camera});
});
$("stop").addEventListener("click", async () => {
  try {
    await request("/api/stop", {method: "POST"});
    message("Stop requested.");
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
