# Phased Build Prompts: Local Face Visitor System

Use these prompts with a coding agent one phase at a time. The repository is
already a partial implementation; audit and extend it rather than replacing it.

## Project context

- This project lives in `app/` and currently includes a face-trained YOLO
  detector, an InsightFace-based embedding recognizer, a lightweight tracker,
  a visitor counter, SQLite persistence, and tests.
- YOLO face weights detect face bounding boxes. They do not identify a person.
  Persistent identity matching requires face embeddings from a recognition
  model and a similarity threshold. Keep detection, tracking, and identity
  recognition as separate, clearly named responsibilities.
- Relevant files include `app/detector.py`, `app/recognizer.py`,
  `app/tracker.py`, `app/visitor_counter.py`, `app/database.py`,
  `app/event_logger.py`, and `app/pipeline.py`. `app/main.py` and
  `README.md` may need implementation or updates.
- Preserve a simple local-first design: one Python process, OpenCV, the
  existing model dependencies, and SQLite. Do not add a distributed service,
  vector database, or heavyweight web stack without a demonstrated need.
- Work incrementally, keep existing user data, and never commit biometric
  images, embeddings, database records, credentials, or model weights.

## Prompt 0 - Use throughout the project

> Act as a senior Python, computer-vision, and SQLite engineer. Build a
> maintainable local visitor-monitoring application in this existing
> repository. First inspect the files and tests relevant to the requested
> phase. Do not rewrite working components or make unrelated changes.
>
> YOLO face weights are for face detection only. Use a compatible face
> recognition/embedding model to create normalized embeddings and compare
> them with stored embeddings using cosine similarity. Never treat a tracker
> ID as a persistent person ID. An uncertain or low-quality match must remain
> unknown; do not silently attach it to the nearest person.
>
> Keep these metrics distinct:
> - **Unique identities**: number of persistent `face_id` records in SQLite.
> - **Visits**: number of entry events; a recognized person returning after an
>   exit starts a new visit.
> - **Currently present**: distinct identities active in the current camera
>   session.
> - **Frames per second**: report capture/read FPS, processed FPS, and
>   inference/processing FPS separately; do not label skipped-frame throughput
>   as source-video FPS.
>
> Persist event wall-clock timestamps in UTC. For video files, also preserve
> the source media timestamp and frame number; for live cameras use a
> monotonic clock for elapsed-time decisions. Do not mix these clock domains.
> Initialize/migrate SQLite safely when the application starts and persist
> each event during execution. Use parameterized SQL, transactions, foreign
> keys, WAL where appropriate, and explicit errors.
>
> Keep the application local by default (bind web endpoints to `127.0.0.1`).
> Treat face images and embeddings as sensitive biometric data: document
> consent, local storage, access, and retention considerations; do not send
> them to external services. Do not claim production readiness without
> measurable validation.
>
> For this phase, implement only the requested scope. Add or update focused
> tests, run the relevant existing tests, and report changed files, commands
> and results, limitations, and any decisions that require calibration. Never
> report a test as passing unless it was actually run.

## Phase 1 - Baseline audit and acceptance criteria

> Apply Prompt 0. Audit the existing detector, embedding recognizer, tracker,
> visitor counter, database, pipeline, configuration, and tests. Trace one
> frame from source read through detection, identity matching, persistence,
> entry/exit handling, and displayed counters. Identify actual bugs and
> mismatched responsibilities, especially whether YOLO detections are truly
> used downstream or merely counted while another detector supplies boxes.
>
> Do not make broad code changes in this phase. Return a concise current-state
> summary, a prioritized implementation sequence, and testable acceptance
> criteria for: repeated observations, a returning known person, a new person,
> unknown/low-quality faces, one exit after the timeout, persistent database
> updates, video timestamps, and localhost startup. Call out any potentially
> destructive handling of existing `data/` or `logs/` and preserve them.

## Phase 2 - Configuration, startup, and reliable database lifecycle

> Apply Prompt 0 and the agreed Phase 1 acceptance criteria. Make configuration
> validation consistent and central enough to avoid different modules
> silently interpreting the same setting differently. Ensure database and
> log paths resolve predictably from the project root and can be overridden
> without changing source code.
>
> Ensure startup creates the SQLite parent directory, initializes the schema,
> enables foreign keys, and applies explicit, versioned, non-destructive
> migrations before video processing begins. Keep the schema simple. Preserve
> existing visitor records. Add the minimum event metadata needed to store
> UTC event time, source media time/frame number where available, and visit
> identity. Make visitor and event counts explicit queries with documented
> semantics. Do not store an embedding as SQL text; keep validated float32
> binary data with its dimension and recognition-model/version metadata.
>
> Add focused database tests for fresh initialization, repeated initialization,
> migration of a representative existing database, valid embedding round-trip,
> invalid embedding rejection, foreign keys, UTC event records, and persistent
> count changes. Do not delete or rewrite the real `data/visitors.db`.

## Phase 3 - Face detection, embeddings, and persistent identity matching

> Apply Prompt 0. Establish one coherent inference path. Use the configured
> YOLO face model as the source of face boxes. For each valid box, use the
> recognition model's documented face alignment/embedding API to produce an
> embedding for that same face; do not independently detect faces a second
> time and associate results by guesswork. If the chosen embedding library
> cannot safely embed a supplied box, explain the limitation and implement a
> reliable, tested alignment path or propose a decision before changing the
> architecture.
>
> Validate and normalize embeddings consistently at creation and load time.
> Persist a stable `face_id` for a new identity and reuse it on future matches.
> Compare only compatible embedding dimensions and model versions. Make the
> cosine-similarity threshold configurable, define clearly that a higher
> score is a closer match, and treat a below-threshold score as unknown.
> Do not use uncalibrated thresholds as a claim of identity accuracy.
>
> Add tests for normalized and malformed embeddings, identical and dissimilar
> vectors, shape/model mismatches, just-below/just-above threshold behavior,
> and a known person being assigned the same database `face_id` after the
> recognizer/process is recreated. Add an offline calibration utility or
> documented procedure using consented validation samples; do not tune a
> threshold to the bundled demo video alone.

## Phase 4 - Tracking, visit counting, and timestamped entry/exit events

> Apply Prompt 0. Integrate detection, embeddings, and the short-lived tracker
> so track IDs reduce repeated recognition work but never become persistent
> person IDs. Resolve tracks to identity only through validated embeddings
> and the persistent matcher. Guard against track-ID reuse, crossings, missed
> frames, duplicate observations, and two people being assigned to one
> identity in the same frame.
>
> Define a visit as beginning when an identity becomes present and ending
> after the configured absence timeout. Log exactly one entry per visit and
> at most one exit for that visit. If that known person returns later, retain
> their `face_id`, create another entry event, and increment the visit count,
> not the unique-identity count. Do not create a new identity merely because
> the tracker was reset. Keep the unique-person, total-visit, current-presence,
> and current-run unique counts separate in APIs and display.
>
> Use monotonic elapsed time for live-camera absence timeouts and media
> timestamps/frame numbers for video-file events. Store UTC wall-clock event
> timestamps separately. Avoid a fake enormous timestamp for end-of-stream
> cleanup; provide an explicit end-of-source close operation. Make writes
> happen during processing, not only after the process exits.
>
> Add deterministic tests with an injectable clock/database for: repeated
> frames, a known return after exit, a new identity, one exit after timeout,
> no duplicate exit, identity across tracker reset, and correct UTC/source
> timestamp fields.

## Phase 5 - Local run command and usable operator interface

> Apply Prompt 0. Provide a straightforward Windows-friendly setup and run
> path using the project's existing dependency manifest and a single documented
> command. Implement `app/main.py` as the stable entry point. Support the
> configured local video source and camera index; make RTSP opt-in and never
> print credentials. Validate source/model/config failures with actionable
> errors and release OpenCV/model resources on normal exit, errors, and `q`.
>
> Show face boxes, stable `face_id` values, similarity/unknown status where
> useful, current presence, unique identities, visits, and processing FPS
> without conflating these counters. Keep headless operation available for
> servers/tests. If adding a web dashboard or API, keep it optional and
> localhost-only by default, use existing project patterns, and do not expose
> camera/biometric data to the network without explicit authentication and
> user approval.
>
> Update `README.md` with environment setup, model-file requirements and
> licensing caveat, configuration reference, startup/shutdown instructions,
> camera permissions, database/log locations, and troubleshooting. Do not
> download or replace model weights automatically without explicit consent.

## Phase 6 - FPS, quality tuning, and operational safeguards

> Apply Prompt 0. Measure the actual pipeline before optimizing. Use
> `time.perf_counter()` to separately report capture/read, YOLO inference,
> embedding inference, persistence, display, and end-to-end processing time.
> Report rolling processing FPS and source FPS independently. Include device,
> input resolution, inference image size, processed-frame interval, and
> warm-up behavior in benchmark output.
>
> Make frame skipping configurable and explain its trade-off: it can improve
> throughput but may miss small/fast faces and affects tracker continuity.
> Express tracking expiry in elapsed time or convert seconds to frames using
> the effective processed rate; do not assume skipped frames are processed.
> Keep confidence, minimum face size, and identity similarity threshold as
> separate settings with separate meanings. Add safe validation and sensible
> documented starting values, not promises of universally correct values.
> Measure CPU and optional GPU behavior without silently falling back when a
> requested device is unavailable.
>
> Add a repeatable benchmark command and a troubleshooting/tuning guide. Test
> that metrics are finite and FPS denominators are correct; do not set
> pass/fail FPS targets without stating the machine, video, resolution, and
> hardware used. Review disk growth and provide a documented retention
> approach for biometric snapshots without automatically deleting data.

## Phase 7 - End-to-end verification and delivery

> Apply Prompt 0. Run the focused tests, then the full available test suite.
> Add an end-to-end test using a short synthetic or consented fixture that
> proves a person is registered once, repeated detections retain the same
> `face_id`, each completed visit creates one entry and one exit, return visits
> increment visits without inflating unique identities, and SQLite updates are
> visible while the application is running.
>
> Verify the documented localhost command from a clean environment as far as
> available in this workspace. Check startup with a missing video/model,
> headless operation, graceful shutdown, persistence after restart, and that
> generated data/log files are not accidentally included in source control.
> Update documentation to distinguish demonstrated behavior from known
> limitations. Finish with a concise report of tests run, measured performance
> conditions, limitations, and how to run the app.

## Tuning reference for the operator

- **YOLO confidence** filters candidate face detections; lower values may find
  more difficult faces but increase false detections. It is not an identity
  match threshold.
- **Minimum face size** affects whether small/distant faces are processed.
  Use the actual source resolution and camera placement when tuning it.
- **Embedding cosine similarity** compares identity features. Higher means
  more similar for normalized embeddings, but the correct acceptance
  threshold depends on the model, camera, lighting, pose, and validation data.
  Check false accepts as well as false rejects before deployment.
- **Tracker IoU/lost duration** controls short-term association, not identity.
  Tune lost duration in seconds for the effective processing rate.
- **Frame skip / inference image size / device** affect latency, throughput,
  and detection quality. Measure end-to-end FPS on the target machine.
- **SQLite** should be initialized at startup and written during processing.
  Unique identities come from the identity table; visit totals come from entry
  events (or a visit table if one is added with an explicit migration).
- Face IDs are pseudonymous identifiers, not proof of a person's legal
  identity. Obtain appropriate consent and define access and retention before
  collecting real biometric data.