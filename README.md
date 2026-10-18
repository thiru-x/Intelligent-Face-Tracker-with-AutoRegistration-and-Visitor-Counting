# Local Face Visitor Counter

This project processes a local video file or camera stream, detects faces with
the configured face-trained YOLO model, computes embeddings with InsightFace,
and matches returning visitors against embeddings stored in SQLite.

YOLO detects **faces**, not identities. A persistent `face_id` is assigned by
embedding comparison; tracker IDs are temporary and are never used as person
identity. InsightFace supplies face embeddings; its boxes are paired
one-to-one with YOLO boxes by IoU overlap. A YOLO box without a sufficiently
overlapping embedding is not registered as a person.

## Windows setup

Use Python 3.11 and create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository expects `models/yolov8n-face.pt` to be present. Model files
must be obtained from a trusted source and used in accordance with their
licenses. InsightFace may download its configured recognition model the
first time it starts; that is a model download, not an upload of camera frames.
The browser dashboard binds to loopback only and does not transmit face data
to a remote service.

Run the configured video source:

```powershell
python -m app.main
```

To view the output and counters in your browser, run:

```powershell
python -m app.main --web
```

The dashboard opens at `http://127.0.0.1:8000/`. It supports starting the
configured video source, entering a local video path/RTSP URL, or selecting
a camera index; live processed frames, counts, and recent entry/exit records
are shown in the page. Use **Stop** to close the active visit cleanly. Change
the port if needed with `python -m app.main --web --port 8080`. The server
rejects non-loopback bind addresses; do not proxy or expose this biometric
dashboard to a network.

Use a camera or a different local video:

```powershell
python -m app.main --camera 0
python -m app.main --source "C:\Videos\visitors.mp4"
python -m app.main --source "rtsp://192.168.1.50:554/stream1"
```

Press `q` in the desktop display or `Ctrl+C` in the terminal to stop. Camera
and stream processing can run without a desktop display using `--no-display`.
RTSP is only used when explicitly selected.

### Connecting an RTSP camera

In the browser dashboard, enter the camera's RTSP URL in **Video path or RTSP
stream URL** and start processing. A common URL shape is
`rtsp://CAMERA_IP:554/PATH`; the exact path and credentials depend on the
camera/NVR vendor. For cameras requiring authentication, use the RTSP
username/password provided by the camera administrator. The URL is held in
memory for the active run and is not written to logs or the database. Prefer
the dashboard over a command that contains a password, because shell commands
can be retained in terminal history.

The default RTSP transport is TCP, which is generally more reliable across
lossy networks and firewalls. Set `rtsp_transport` to `udp` in `config.json`
only when UDP is required by the camera/network. Open and read timeouts default
to 10 seconds. On a stream interruption, the app attempts five reconnects,
waiting one second between attempts; adjust `network_open_timeout_seconds`,
`network_read_timeout_seconds`, `network_reconnect_attempts`, and
`network_reconnect_delay_seconds` for the camera/network. Set
`use_rtsp` to `true` and `rtsp_url` in `config.json` to use that source when
starting the configured source; otherwise enter the URL in the dashboard or
pass it with `--source`.

The camera must be reachable from this machine and provide a stream supported
by the installed OpenCV FFmpeg backend. If it does not connect, verify the
camera's RTSP setting, exact stream path, credentials, firewall/VLAN access,
and that another application is not exclusively holding the camera. A failed
live stream is retried within the configured limit; the browser dashboard
shows a reconnecting state and reports an error if retries are exhausted.

## What the counts mean

- **Unique identities**: persistent face records in `faces`; a returning
  matched identity does not increment this.
- **Unique this run**: distinct `face_id` values seen in this invocation.
- **Visits**: entry events. A known person returning after an exit keeps their
  `face_id` and creates another visit.
- **Present**: distinct identities observed in the current processed frame.
- **Processed FPS**: processed frames divided by total wall-clock run time.
- **Inference FPS**: processed frames divided by combined YOLO and embedding
  inference time; it excludes capture, database, and display time.
- **Capture/read FPS**: frames read divided by time spent in OpenCV frame reads;
  for a video file this is decode throughput, not the video's encoded FPS.
- **Source-reported FPS**: FPS reported by the source/backend. It may be missing
  or inaccurate for live streams.

The SQLite database is initialized and safely migrated at startup. New face
records and entry/exit events are written during processing to the configured
database (default `data/visitors.db`, WAL mode). Timestamps in
`events.timestamp` are UTC;
`source_timestamp` is elapsed media time for video files and is `NULL` for a
live camera/stream. `frame_number` is the 1-based source frame number. When a
source ends or the user quits, active visits receive exit events.

The configured `log_directory` is the root for `events.log`, the rotating
`application.log`, and dated `entries/` and `exits/` face-image directories.
`events.log` contains structured JSON-lines audit records for detection,
embedding generation, tracking, recognition decisions, registration, and
entry/exit. Audit records are committed to the SQLite `audit_log` outbox before
they are appended and flushed to disk. If the log file is temporarily
unavailable, the committed rows remain pending and are retried at the next
startup; each record has an `audit_id` so a replay after a crash between file
append and database acknowledgement can be deduplicated. Face crops are saved
before their visit event is inserted; if the event insert fails, the unreferenced
crop is removed. Image-write failure prevents an entry/exit event from being
recorded without its required crop.

Check current persisted counts in a separate PowerShell terminal with:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/visitors.db'); print('Unique identities:', c.execute('select count(*) from faces').fetchone()[0]); print('Visits:', c.execute('select count(*) from events where event_type=?', ('entry',)).fetchone()[0]); print('Events:', c.execute('select count(*) from events').fetchone()[0]); c.close()"
```

Event rows are written as they happen, so querying the database while the
process is running shows new registrations and visit events. Use a SQLite
viewer that supports WAL mode or query through SQLite; do not delete the
database or its `-wal`/`-shm` files while the app is running.

## Configuration

Edit `config.json`:

| Setting | Meaning |
| --- | --- |
| `video_source` | Default file path or source string |
| `use_rtsp`, `rtsp_url` | Optional configured RTSP source |
| `yolo_face_model` | Face-trained YOLO weights path |
| `face_detection_confidence` | YOLO face-box confidence filter |
| `detection_image_size` | YOLO inference image size (defaults to 960) |
| `min_face_size` | Minimum YOLO box width and height in pixels |
| `model_name` | InsightFace embedding model name |
| `recognition_threshold` | Minimum cosine similarity to reuse a `face_id` |
| `embedding_box_iou_threshold` | Minimum overlap to pair a YOLO box with its embedding |
| `detection_skip_frames` | Process every Nth source frame |
| `tracker_max_lost_frames` | Tracker retention in processed frames |
| `camera_exit_timeout_seconds` | Absence time before an exit is recorded |
| `rtsp_transport` | FFmpeg RTSP transport: `tcp` (default) or `udp` |
| `network_open_timeout_seconds` | Timeout for opening a network stream |
| `network_read_timeout_seconds` | Timeout for a network frame read |
| `network_reconnect_attempts` | Bounded retries after network stream failure |
| `network_reconnect_delay_seconds` | Delay between reconnect attempts |
| `database_path`, `log_directory` | Project-relative persistence locations |
| `device` | `CPU` or `GPU`; GPU mode requires compatible runtime providers |

Starting thresholds are not universal guarantees. YOLO confidence filters
detection boxes; the recognition threshold is a separate cosine similarity
decision, where larger values are closer matches. Validate both on consented
samples from the target camera, lighting, distance, and pose. Review false
accepts as well as false rejects before using the system for consequential
decisions. Frame skipping and inference resolution trade speed for the chance
of missing small or fast-moving faces. Measure FPS on the deployment hardware.

## Data and privacy

Face embeddings and saved face crops are sensitive biometric data. Obtain
appropriate consent, restrict local file/database access, define a retention
period, and remove records through an approved retention process. This project
does not automatically delete historical data. Face IDs are pseudonymous and
are not proof of a person's legal identity. Detection and recognition can fail
because of occlusion, pose, lighting, camera quality, threshold choice, or
similar-looking faces; do not use this prototype as the sole basis for
high-impact decisions.

This project is part of a hackathon run by https://katomaran.com