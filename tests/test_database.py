
import sqlite3

import numpy as np
import pytest

import app.database as database
from app import event_logger


@pytest.fixture
def clean_database(tmp_path, monkeypatch):
    test_db = tmp_path / "test_visitors.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)
    database.initialize_database()
    return test_db


def test_face_registration_and_embedding_round_trip(clean_database):
    embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    database.register_face("TEST_FACE_001", embedding)

    faces = database.get_all_faces()
    assert len(faces) == 1
    assert faces[0][0] == "TEST_FACE_001"
    np.testing.assert_allclose(faces[0][1], embedding)


def test_entry_and_exit_events(clean_database):
    database.register_face(
        "TEST_FACE_002",
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )

    database.log_event("TEST_FACE_002", "entry", "entries/test.jpg")
    database.log_event("TEST_FACE_002", "exit", "exits/test.jpg")

    assert database.get_event_count() == 2
    assert database.get_unique_visitor_count() == 1


def test_duplicate_face_id_is_rejected(clean_database):
    embedding = np.array([0.1, 0.2], dtype=np.float32)
    database.register_face("TEST_FACE_003", embedding)

    with pytest.raises(Exception):
        database.register_face("TEST_FACE_003", embedding)


def test_existing_database_is_migrated_without_losing_rows(tmp_path):
    old_db = tmp_path / "legacy.db"
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    with sqlite3.connect(old_db) as connection:
        connection.executescript("""
            CREATE TABLE faces (
                face_id TEXT PRIMARY KEY,
                first_seen TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dimension INTEGER NOT NULL
            );
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT
            );
        """)
        connection.execute(
            "INSERT INTO faces VALUES (?, ?, ?, ?)",
            (
                "LEGACY_FACE",
                "2026-01-01T00:00:00+00:00",
                embedding.tobytes(),
                embedding.size,
            ),
        )
        connection.execute(
            "INSERT INTO events "
            "(face_id, event_type, timestamp) VALUES (?, ?, ?)",
            ("LEGACY_FACE", "entry", "2026-01-01T00:00:00+00:00"),
        )

    database.initialize_database(old_db)
    database.initialize_database(old_db)

    with sqlite3.connect(old_db) as connection:
        face_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(faces)")
        }
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(events)")
        }
        audit_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(audit_log)")
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        face_count = connection.execute(
            "SELECT COUNT(*) FROM faces"
        ).fetchone()[0]
        event_count = connection.execute(
            "SELECT COUNT(*) FROM events"
        ).fetchone()[0]

    assert "embedding_model" in face_columns
    assert {"source_timestamp", "frame_number"} <= event_columns
    assert {"audit_id", "event_type", "delivered_at"} <= audit_columns
    assert version == database.SCHEMA_VERSION
    assert face_count == 1
    assert event_count == 1


def test_event_persists_source_metadata_and_visit_count(clean_database):
    database.register_face(
        "TEST_FACE_004",
        np.array([1.0, 0.0], dtype=np.float32),
        embedding_model="test-model",
    )

    database.log_event(
        "TEST_FACE_004",
        "entry",
        source_timestamp=12.5,
        frame_number=375,
    )

    with database.get_connection() as connection:
        timestamp, media_time, frame_number = connection.execute(
            """
            SELECT timestamp, source_timestamp, frame_number
            FROM events WHERE face_id = ?
            """,
            ("TEST_FACE_004",),
        ).fetchone()

    assert timestamp.endswith("+00:00")
    assert media_time == 12.5
    assert frame_number == 375
    assert database.get_visit_count() == 1
    assert database.get_all_faces_with_metadata()[0][2] == "test-model"


def test_event_rejects_invalid_source_metadata(clean_database):
    database.register_face(
        "TEST_FACE_005",
        np.array([1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="source_timestamp"):
        database.log_event(
            "TEST_FACE_005",
            "entry",
            source_timestamp=float("nan"),
        )
    with pytest.raises(ValueError, match="frame_number"):
        database.log_event(
            "TEST_FACE_005",
            "entry",
            frame_number=1.5,
        )


def test_audit_outbox_retries_file_failure_without_losing_database_event(
    clean_database,
    tmp_path,
    monkeypatch,
):
    log_directory = tmp_path / "audit-logs"
    event_logger.configure_logging(log_directory)
    database.register_face(
        "TEST_FACE_AUDIT",
        np.array([1.0, 0.0], dtype=np.float32),
    )
    event_logger.flush_audit_outbox()

    original_writer = event_logger._write_audit_lines

    def fail_writer(records):
        raise OSError("simulated disk interruption")

    with monkeypatch.context() as patch:
        patch.setattr(event_logger, "_write_audit_lines", fail_writer)
        with pytest.raises(OSError, match="simulated disk interruption"):
            database.log_event("TEST_FACE_AUDIT", "entry")
            event_logger.flush_audit_outbox()

    assert database.get_event_count() == 1
    assert database.get_pending_audit_count() == 1

    monkeypatch.setattr(
        event_logger, "_write_audit_lines", original_writer
    )
    event_logger.flush_audit_outbox()

    assert database.get_pending_audit_count() == 0
    audit_text = (log_directory / "events.log").read_text(encoding="utf-8")
    assert '"event_type":"VISITOR_ENTRY"' in audit_text
    assert '"face_id":"TEST_FACE_AUDIT"' in audit_text


def test_audit_outbox_deduplicates_after_append_before_delivery_marker(
    clean_database,
    tmp_path,
    monkeypatch,
):
    log_directory = tmp_path / "audit-replay"
    event_logger.configure_logging(log_directory)
    database.register_face(
        "TEST_FACE_REPLAY",
        np.array([1.0, 0.0], dtype=np.float32),
    )
    event_logger.flush_audit_outbox()

    original_marker = database.mark_audit_events_delivered

    def fail_delivery_marker(audit_ids):
        raise OSError("simulated interruption after append")

    with monkeypatch.context() as patch:
        patch.setattr(
            database, "mark_audit_events_delivered", fail_delivery_marker
        )
        with pytest.raises(OSError, match="after append"):
            database.log_event("TEST_FACE_REPLAY", "entry")
            event_logger.flush_audit_outbox()

    assert database.get_pending_audit_count() == 1
    monkeypatch.setattr(
        database, "mark_audit_events_delivered", original_marker
    )
    event_logger.flush_audit_outbox()

    lines = (log_directory / "events.log").read_text(
        encoding="utf-8"
    ).splitlines()
    entry_lines = [
        line for line in lines if '"event_type":"VISITOR_ENTRY"' in line
    ]
    assert len(entry_lines) == 1
    assert database.get_pending_audit_count() == 0
