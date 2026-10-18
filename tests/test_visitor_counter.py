
import sqlite3

import numpy as np
import pytest

from app import database
from app import event_logger
from app import visitor_counter as visitor_counter_module
from app.visitor_counter import VisitorCounter



class FakeRecognizer:
    def match_embedding(self, embedding):
        for face_id, known_embedding in database.get_all_faces():
            if np.allclose(embedding, known_embedding):
                return face_id, 1.0

        return None, -1.0


@pytest.fixture
def counter(tmp_path, monkeypatch):
    db_path = tmp_path / "visitors.db"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    event_logger.configure_logging(tmp_path / "logs", face_image_quality=75)
    database.initialize_database()

    recognizer = FakeRecognizer()
    visitor_counter = VisitorCounter(
        recognizer=recognizer,
        exit_timeout_seconds=3,
    )

    return visitor_counter, recognizer


def test_new_face_is_registered_and_entry_logged(counter, tmp_path):
    visitor_counter, recognizer = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    result = visitor_counter.process_faces(
        [{"bbox": (10, 10, 50, 50), "embedding": embedding}],
        frame,
        now=100,
    )

    assert len(result) == 1
    assert result[0]["is_new_visitor"] is True
    assert result[0]["face_id"].startswith("FACE_")
    assert database.get_unique_visitor_count() == 1
    assert database.get_event_count() == 1
    with database.get_connection() as connection:
        image_path = connection.execute(
            "SELECT image_path FROM events WHERE event_type = 'entry'"
        ).fetchone()[0]
    assert image_path
    assert (tmp_path / "logs" / "entries").is_dir()
    assert (tmp_path / "logs" / "events.log").is_file()


def test_repeated_observation_does_not_duplicate_entry(counter):
    visitor_counter, _ = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    face = {"bbox": (10, 10, 50, 50), "embedding": embedding}

    visitor_counter.process_faces([face], frame, now=100)
    visitor_counter.process_faces([face], frame, now=101)

    assert database.get_unique_visitor_count() == 1
    assert database.get_event_count() == 1


def test_audit_delivery_failure_does_not_duplicate_entry_on_retry(
    counter,
    monkeypatch,
):
    visitor_counter, _ = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    face = {"bbox": (10, 10, 50, 50), "embedding": embedding}
    flush = event_logger.flush_audit_outbox
    delivery_attempts = 0

    def fail_first_delivery():
        nonlocal delivery_attempts
        delivery_attempts += 1
        if delivery_attempts == 1:
            raise OSError("simulated audit-file outage")
        flush()

    monkeypatch.setattr(
        visitor_counter_module,
        "flush_audit_outbox",
        fail_first_delivery,
    )

    with pytest.raises(OSError, match="simulated audit-file outage"):
        visitor_counter.process_faces([face], frame, now=100)

    assert database.get_event_count() == 1
    assert len(visitor_counter.active_visitors) == 1

    visitor_counter.process_faces([face], frame, now=101)

    assert database.get_event_count() == 1
    assert database.get_visit_count() == 1
    assert database.get_pending_audit_count() == 0


def test_tracked_face_stays_present_during_embedding_miss(counter):
    visitor_counter, _ = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    visitor_counter.process_faces(
        [{
            "bbox": (10, 10, 50, 50),
            "embedding": embedding,
            "track_id": 1,
        }],
        frame,
        now=100,
    )

    result = visitor_counter.process_faces(
        [{
            "bbox": (11, 10, 51, 50),
            "embedding": None,
            "track_id": 1,
        }],
        frame,
        now=101,
    )

    assert len(result) == 1
    assert result[0]["face_id"].startswith("FACE_")
    assert database.get_event_count() == 1
    assert visitor_counter.active_visitors


def test_missing_visitor_generates_one_exit(counter):
    visitor_counter, _ = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    face = {"bbox": (10, 10, 50, 50), "embedding": embedding}

    visitor_counter.process_faces([face], frame, now=100)
    visitor_counter.expire_missing_visitors(now=102)
    assert database.get_event_count() == 1

    visitor_counter.expire_missing_visitors(now=103)
    assert database.get_event_count() == 2

    visitor_counter.expire_missing_visitors(now=104)
    assert database.get_event_count() == 2


def test_returning_person_keeps_face_id_and_increments_visit_count(counter):
    visitor_counter, recognizer = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    face = {
        "bbox": (10, 10, 50, 50),
        "embedding": embedding,
        "track_id": 1,
    }

    first_visit = visitor_counter.process_faces(
        [face],
        frame,
        now=100,
        source_timestamp=1.25,
        frame_number=38,
    )[0]
    visitor_counter.expire_missing_visitors(
        now=103,
        source_timestamp=4.25,
        frame_number=128,
    )

    restarted_counter = VisitorCounter(
        recognizer=recognizer,
        exit_timeout_seconds=3,
    )
    returning_face = {**face, "track_id": 99}
    second_visit = restarted_counter.process_faces(
        [returning_face],
        frame,
        now=104,
        source_timestamp=5.25,
        frame_number=158,
    )[0]

    assert first_visit["face_id"] == second_visit["face_id"]
    assert second_visit["is_new_visitor"] is False
    assert second_visit["is_new_visit"] is True
    assert database.get_unique_visitor_count() == 1
    assert database.get_visit_count() == 2
    assert database.get_event_count() == 3


def test_source_end_closes_each_active_visit_once(counter):
    visitor_counter, _ = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    visitor_counter.process_faces(
        [{"bbox": (10, 10, 50, 50), "embedding": embedding}],
        frame,
        now=100,
        source_timestamp=1.0,
        frame_number=30,
    )

    visitor_counter.close_active_visits(
        source_timestamp=2.0,
        frame_number=60,
    )
    visitor_counter.close_active_visits(
        source_timestamp=2.0,
        frame_number=60,
    )

    assert database.get_event_count() == 2
    assert visitor_counter.active_visitors == {}

    with database.get_connection() as connection:
        exit_source_time, exit_frame = connection.execute(
            """
            SELECT source_timestamp, frame_number
            FROM events WHERE event_type = 'exit'
            """
        ).fetchone()
    assert exit_source_time == 2.0
    assert exit_frame == 60


def test_failed_face_image_write_does_not_create_visit_event(
    counter,
    monkeypatch,
):
    visitor_counter, _ = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(
        event_logger.cv2, "imwrite", lambda *args, **kwargs: False
    )

    with pytest.raises(OSError, match="Failed to save face image"):
        visitor_counter.process_faces(
            [{"bbox": (10, 10, 50, 50), "embedding": embedding}],
            frame,
            now=100,
        )

    assert database.get_unique_visitor_count() == 1
    assert database.get_visit_count() == 0
    assert database.get_event_count() == 0
    assert visitor_counter.active_visitors == {}


def test_database_failure_removes_unreferenced_face_crop(
    counter,
    monkeypatch,
):
    visitor_counter, _ = counter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def fail_database_write(**kwargs):
        raise sqlite3.OperationalError("simulated database interruption")

    monkeypatch.setattr(database, "log_event", fail_database_write)

    with pytest.raises(sqlite3.OperationalError):
        visitor_counter.process_faces(
            [{"bbox": (10, 10, 50, 50), "embedding": embedding}],
            frame,
            now=100,
        )

    assert database.get_event_count() == 0
    assert list((event_logger.LOG_DIR / "entries").rglob("*.jpg")) == []
    assert visitor_counter.active_visitors == {}
