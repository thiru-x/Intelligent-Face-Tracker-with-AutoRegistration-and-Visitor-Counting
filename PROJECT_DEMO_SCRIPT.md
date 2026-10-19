# Loom Walkthrough: Project Demo Script

**Suggested length:** 7–9 minutes  
**Audience:** hackathon reviewers and technical interviewers  
**Recording setup:** record the browser and VS Code/terminal, close personal tabs,
hide RTSP credentials and real visitor data, and use a consented sample video.

The bracketed notes are on-screen actions, not spoken narration. Adapt the
wording to your own voice. Do not claim a live RTSP camera was tested unless you
have actually connected and demonstrated one.

## Before recording

1. Confirm the checked-out project and its model weights are available locally.
   The model weights, local database, logs, and sample videos are intentionally
   excluded from Git.
2. Start the app with `python -m app.main --web`; show the local URL
   `http://127.0.0.1:8000/` in the browser.
3. Use a short, consented sample video with more than one arrival if possible.
   If the sample does not contain a clear departure and return, demonstrate the
   persisted metrics and explain the return-visit test rather than implying it
   happened in that recording.
4. Keep the RTSP source field empty unless you are deliberately showing a
   working camera. Never reveal a real camera password or URL containing
   credentials.
5. Keep a terminal ready for the test command and a local SQLite viewer or
   read-only query if you want to show persisted events.

## 0:00–0:35 — Intro

**[Show the dashboard, then the project title.]**

> Hello, I’m [name]. This is my submission for the Katomaran Intelligent Face
> Tracker with Auto-Registration and Visitor Counting hackathon. The project
> takes a local video, webcam, or RTSP stream; detects faces; matches returning
> visitors using face embeddings; and records timestamped entry and exit
> events. The processing, database, images, logs, and dashboard run locally on
> this machine.

## 0:35–1:15 — Problem statement and requirements

**[Briefly show the PDF’s objective and core modules, then return to the app.]**

> The problem statement asks for more than a face detector. It requires
> detection, persistent recognition, tracking, automatic registration, unique
> visitor counting, timestamped entry and exit images, database records, a
> mandatory audit log, configurable frame skipping, and a clear local
> architecture. Live RTSP is also an interview scenario. The browser interface
> is optional in the statement, but I added it so the stream, counts, and
> events are visible during a demonstration.

## 1:15–2:05 — Project structure

**[Show the repository tree in VS Code. Expand `app/` and `tests/`.]**

> The application is intentionally a single-process Python project. In
> `app/`, `detector.py` wraps YOLO face detection; `recognizer.py` provides
> InsightFace embeddings and matching; `tracker.py` handles short-lived
> OpenCV track IDs; and `visitor_counter.py` owns persistent face identities
> and visit transitions. `pipeline.py` connects source capture, inference,
> association, tracking, counting, and rendered frames.
>
> `database.py` initializes and migrates SQLite, stores face embeddings and
> visit events, and provides the durable audit outbox. `event_logger.py`
> writes face crops and structured logs. `web.py` serves the localhost API,
> live MJPEG feed, and dashboard. `main.py` is the command-line entry point.
> Configuration is in `config.json`; `tests/` covers the database, pipeline,
> tracker, visitor transitions, audit resilience, and web endpoints.

## 2:05–3:10 — Architecture and identity matching

**[Open `docs/architecture.svg` in the browser or show it in the README.]**

> The architecture flows from a video file, camera, or RTSP stream into
> OpenCV capture. The pipeline attaches frame numbers and source timestamps
> where available, and applies the configured frame-skip setting before the
> more expensive processing stages.
>
> YOLO detects face boxes; it does not identify people. InsightFace generates
> face embeddings. In this implementation InsightFace performs its own face
> detection, so its boxes are associated one-to-one with YOLO boxes using an
> IoU threshold. This avoids pretending that a detection-only YOLO model
> provides recognition or that a tracker ID is a person ID.
>
> The tracker IDs are temporary and help associate observations across
> nearby frames. Persistent identity comes from comparing compatible
> embeddings with cosine similarity. If the best match reaches the configured
> threshold, the existing `face_id` is reused; otherwise the system registers
> a new face. A returning person therefore keeps their persistent face ID even
> if the process or tracker is restarted.

## 3:10–4:15 — Browser demonstration

**[Return to the dashboard. Select Video / RTSP, enter a local sample path or
leave it empty for the configured default, then start processing.]**

> This is the local browser dashboard. I can choose a video path or RTSP URL,
> or switch to the webcam tab and choose a camera index. RTSP credentials are
> used for the active run and are not written to the database or audit log.
> The server binds to loopback rather than exposing this biometric interface
> to the network.
>
> As frames are processed, the dashboard shows the annotated feed, total
> identities, visits, currently present visitors, processed FPS, source FPS,
> and recent timestamped events. The values update from the processing worker
> and the SQLite database; they are not placeholders.

## 4:15–5:05 — Counts and visit semantics

**[Point to each metric and then to the recent event table.]**

> These counters have different meanings. Unique identities are persistent
> face records in SQLite. Visits count entry events. Present is the number of
> distinct recognized identities currently seen in the processed frame. A
> repeated observation of the same active visitor does not create another
> entry. When an identity is absent longer than the configured exit timeout,
> an exit is logged. If that face is recognized later, the same face ID is
> reused and a new visit is counted.
>
> The event table shows UTC event time and, when available, the source frame
> and media timestamp. For a live camera, source media time is not available,
> so the media timestamp is intentionally blank.

## 5:05–6:05 — Configuration and performance

**[Show `config.json`, highlighting these values.]**

> The main tuning settings are visible in `config.json`. The current
> `detection_skip_frames` value is 8, so the pipeline processes one frame out
> of each eight-frame interval. Face detection confidence is 0.2,
> `recognition_threshold` is 0.45, and the minimum IoU for pairing YOLO
> detections with InsightFace detections is 0.3. These are starting values,
> not universal accuracy guarantees.
>
> A higher cosine similarity means a closer embedding match. Raising the
> recognition threshold can reduce false matches but may increase missed
> matches; lowering it can do the reverse. These values need calibration on
> consented footage representative of the deployment camera, distance,
> lighting, and pose.
>
> FPS is also split by meaning. Processed FPS measures processed frames over
> the complete wall-clock run. Inference FPS isolates YOLO plus embedding
> inference, while capture/read FPS measures frame-read throughput. The
> dashboard shows processed and source-reported FPS. Increasing frame skip
> reduces inference work but can miss small or fast-moving faces.

## 6:05–7:05 — Persistence and audit resilience

**[Show the event table, then `logs/` and a read-only database view if
available. Do not show real biometric records.]**

> SQLite stores face IDs, binary embeddings with model metadata, and
> entry/exit events. Each event includes a UTC timestamp and a saved face-crop
> path. Crops are organized in dated entry and exit folders. `events.log` is
> a structured JSON-lines audit trail for detection, embeddings, tracking,
> recognition, registration, and visit events. `application.log` contains
> rotating operational diagnostics.
>
> To make audit delivery resilient, an event is first committed to SQLite's
> audit outbox. It is then appended and flushed to the audit file, and only
> then acknowledged in the database. A pending record can be retried after an
> interruption without duplicating an audit line. If saving the required
> face crop fails, the visit event is not recorded; if its database insert
> fails after the crop is written, the unreferenced crop is removed when
> possible.

## 7:05–7:50 — Validation and resource expectations

**[Show the test output and, if available, a real run's printed FPS summary.]**

> The automated tests use synthetic frames and simulated failures to check
> identity reuse, return visits, exits, SQLite persistence, audit replay,
> image-write failure, database failure, and localhost API behavior. They do
> not certify face-recognition accuracy or substitute for a physical RTSP
> camera test.
>
> The expensive work is YOLO inference and InsightFace embedding extraction.
> On CPU, performance depends heavily on processor, input resolution,
> frame-skip, and the selected inference providers. A compatible GPU runtime
> can accelerate inference, but GPU mode requires a working provider and its
> memory use depends on the model and runtime. I do not claim fixed CPU/GPU
> utilization numbers without measuring the target hardware. The application
> reports inference timing and FPS so this can be measured on the deployment
> machine.

## 7:50–8:20 — Privacy, limitations, and outro

**[Return to the dashboard or show the architecture diagram.]**

> Face embeddings and face crops are sensitive biometric data. This project
> is local-first, but local storage still needs appropriate consent, access
> controls, and a retention policy. The system can make false matches or miss
> faces because of camera quality, lighting, pose, occlusion, threshold
> selection, or similar-looking faces, so it should not be treated as proof
> of legal identity or as the sole basis for a high-impact decision.
>
> That completes the walkthrough of the architecture, code structure,
> live dashboard, visitor counting, and resilient event logging. Thank you
> for reviewing my Katomaran hackathon submission.

## Recording checklist

- [ ] The audio is clear and the browser text is readable at recording size.
- [ ] The demo uses consented footage and reveals no real biometric data.
- [ ] The RTSP URL contains no visible password or private camera address.
- [ ] The shown counts and events were generated by the displayed run.
- [ ] The video distinguishes automated test coverage from hardware testing.
- [ ] Add the final Loom URL to the README before submission.
