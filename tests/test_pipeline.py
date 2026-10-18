import json
import threading

import numpy as np

import app.pipeline as pipeline_module
from app import database
from app import event_logger
from app.pipeline import VideoPipeline


def test_yolo_boxes_pair_to_matching_embeddings_one_to_one():
    yolo_faces = [
        {"bbox": (0, 0, 100, 100), "confidence": 0.9},
        {"bbox": (100, 0, 200, 100), "confidence": 0.8},
    ]
    features = [
        {"bbox": (5, 5, 95, 95), "embedding": [1.0, 0.0]},
        {"bbox": (105, 5, 195, 95), "embedding": [0.0, 1.0]},
    ]

    result = VideoPipeline._match_yolo_faces_to_embeddings(
        yolo_faces,
        features,
        minimum_iou=0.3,
    )

    assert result == {0: 0, 1: 1}


def test_yolo_box_without_overlapping_embedding_is_unmatched():
    yolo_faces = [{"bbox": (0, 0, 50, 50), "confidence": 0.9}]
    features = [{"bbox": (80, 80, 120, 120), "embedding": [1.0, 0.0]}]

    result = VideoPipeline._match_yolo_faces_to_embeddings(
        yolo_faces,
        features,
        minimum_iou=0.3,
    )

    assert result == {}


def test_rtsp_sources_use_ffmpeg_and_bounded_open_read_timeouts(monkeypatch):
    pipeline = VideoPipeline.__new__(VideoPipeline)
    pipeline.network_open_timeout_ms = 7000
    pipeline.network_read_timeout_ms = 9000
    calls = []

    class FakeCapture:
        pass

    def fake_video_capture(*args):
        calls.append(args)
        return FakeCapture()

    monkeypatch.setattr(
        pipeline_module.cv2,
        "VideoCapture",
        fake_video_capture,
    )

    assert pipeline._is_live_source("rtsps://camera/stream")
    assert pipeline._is_network_source("rtsp://camera/stream")
    pipeline._open_capture("rtsp://camera/stream")

    assert calls == [(
        "rtsp://camera/stream",
        pipeline_module.cv2.CAP_FFMPEG,
        [
            pipeline_module.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            7000,
            pipeline_module.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            9000,
        ],
    )]


def test_network_capture_retries_and_reports_reconnecting(monkeypatch):
    pipeline = VideoPipeline.__new__(VideoPipeline)
    pipeline.network_reconnect_attempts = 2
    pipeline.network_reconnect_delay_seconds = 0.001
    captures = [False, True]

    class FakeCapture:
        def __init__(self, opened):
            self.opened = opened
            self.released = False

        def isOpened(self):
            return self.opened

        def release(self):
            self.released = True

    monkeypatch.setattr(
        pipeline,
        "_open_capture",
        lambda source: FakeCapture(captures.pop(0)),
    )
    statuses = []
    capture = pipeline._reopen_network_capture(
        "rtsp://camera/stream",
        stop_event=None,
        status_callback=statuses.append,
        reason="Camera unavailable",
    )

    assert capture.isOpened()
    assert len(statuses) == 2
    assert statuses[-1]["state"] == "reconnecting"
    assert "(2/2)" in statuses[-1]["error"]


def test_video_pipeline_persists_entry_and_exit_during_processing(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "pipeline.db"
    log_directory = tmp_path / "pipeline-logs"
    event_logger.configure_logging(log_directory, face_image_quality=75)
    monkeypatch.setattr(
        pipeline_module,
        "load_config",
        lambda: {
            "database_path": str(database_path),
            "log_directory": str(log_directory),
            "face_image_quality": 75,
            "video_source": "unused.mp4",
            "detection_skip_frames": 1,
            "tracker_max_lost_frames": 3,
            "camera_exit_timeout_seconds": 0.15,
            "embedding_box_iou_threshold": 0.3,
        },
    )

    class FakeDetector:
        def detect(self, frame):
            if int(frame[0, 0, 0]) < 2:
                return [{"bbox": (10, 10, 50, 50), "confidence": 0.9}]
            return []

    class FakeRecognizer:
        embedding_model = "test-model"

        def extract_faces(self, frame):
            if int(frame[0, 0, 0]) < 2:
                return [{
                    "bbox": (10, 10, 50, 50),
                    "embedding": np.array([1.0, 0.0], dtype=np.float32),
                    "confidence": 0.9,
                }]
            return []

        def match_embedding(self, embedding):
            for face_id, stored in database.get_all_faces():
                if np.allclose(embedding, stored):
                    return face_id, 1.0
            return None, -1.0

    class FakeCapture:
        def __init__(self, source):
            self.frames = [
                np.full((64, 64, 3), value, dtype=np.uint8)
                for value in range(4)
            ]
            self.position = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == pipeline_module.cv2.CAP_PROP_FPS:
                return 10.0
            if property_id == pipeline_module.cv2.CAP_PROP_POS_MSEC:
                return self.position * 100.0
            return 0.0

        def read(self):
            if self.position >= len(self.frames):
                return False, None
            frame = self.frames[self.position]
            self.position += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(pipeline_module, "FaceDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "FaceRecognizer", FakeRecognizer)
    monkeypatch.setattr(
        pipeline_module.cv2, "VideoCapture", FakeCapture
    )

    summary = VideoPipeline().process_video(
        video_source="synthetic.mp4",
        display=False,
    )

    assert summary["frames_read"] == 4
    assert summary["frames_processed"] == 4
    assert summary["unique_visitors"] == 1
    assert summary["registered_identities"] == 1
    assert summary["visits"] == 1
    assert np.isfinite(summary["processing_fps"])
    assert np.isfinite(summary["inference_fps"])

    with database.get_connection() as connection:
        events = connection.execute(
            """
            SELECT event_type, source_timestamp, frame_number
            FROM events ORDER BY event_id
            """
        ).fetchall()

    assert [event[0] for event in events] == ["entry", "exit"]
    assert events[0][1:] == (0.1, 1)
    assert events[1][1:] == (0.4, 4)
    audit_records = [
        json.loads(line)
        for line in (log_directory / "events.log").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    audit_types = {record["event_type"] for record in audit_records}
    for event_type in (
        "FACE_DETECTED",
        "FACE_EMBEDDING_GENERATED",
        "FACE_TRACKED",
        "FACE_RECOGNIZED",
        "FACE_REGISTERED",
        "VISITOR_ENTRY",
        "VISITOR_EXIT",
    ):
        assert event_type in audit_types
    assert sum(
        record["event_type"] == "VISITOR_ENTRY"
        for record in audit_records
    ) == 1
    with database.get_connection() as connection:
        image_paths = connection.execute(
            "SELECT image_path FROM events ORDER BY event_id"
        ).fetchall()
    assert len(image_paths) == 2
    assert all(
        (pipeline_module.PROJECT_ROOT / path[0]).is_file()
        for path in image_paths
    )


def test_rtsp_pipeline_reconnects_and_publishes_state(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "rtsp-pipeline.db"
    log_directory = tmp_path / "rtsp-logs"
    event_logger.configure_logging(log_directory)
    monkeypatch.setattr(
        pipeline_module,
        "load_config",
        lambda: {
            "database_path": str(database_path),
            "log_directory": str(log_directory),
            "detection_skip_frames": 1,
            "tracker_max_lost_frames": 3,
            "camera_exit_timeout_seconds": 3,
            "embedding_box_iou_threshold": 0.3,
            "network_open_timeout_seconds": 2,
            "network_read_timeout_seconds": 2,
            "network_reconnect_attempts": 2,
            "network_reconnect_delay_seconds": 0.01,
            "rtsp_transport": "tcp",
        },
    )

    class FakeDetector:
        def detect(self, frame):
            return []

    class FakeRecognizer:
        embedding_model = "test-model"

        def extract_faces(self, frame):
            return []

    class FakeCapture:
        def __init__(self, reads, opened=True):
            self.reads = reads
            self.position = 0
            self.opened = opened
            self.released = False

        def isOpened(self):
            return self.opened

        def get(self, property_id):
            if property_id == pipeline_module.cv2.CAP_PROP_FPS:
                return 0.0
            return 0.0

        def read(self):
            if self.position >= len(self.reads):
                return False, None
            result = self.reads[self.position]
            self.position += 1
            return result

        def release(self):
            self.released = True

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    captures = [
        FakeCapture([(False, None)]),
        FakeCapture([], opened=False),
        FakeCapture([(True, frame)]),
    ]
    capture_calls = []

    def fake_video_capture(*args):
        capture_calls.append(args)
        return captures.pop(0)

    monkeypatch.setattr(pipeline_module, "FaceDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "FaceRecognizer", FakeRecognizer)
    monkeypatch.setattr(
        pipeline_module.cv2, "VideoCapture", fake_video_capture
    )

    stop_event = threading.Event()
    statuses = []
    pipeline = VideoPipeline()
    pipeline.process_video(
        video_source="rtsp://camera.local/stream1",
        display=False,
        frame_callback=lambda image: stop_event.set(),
        status_callback=statuses.append,
        stop_event=stop_event,
    )

    assert len(capture_calls) == 3
    assert all(call[1] == pipeline_module.cv2.CAP_FFMPEG for call in capture_calls)
    assert any(status["state"] == "reconnecting" for status in statuses)
    assert statuses[-1]["state"] == "stopped"
