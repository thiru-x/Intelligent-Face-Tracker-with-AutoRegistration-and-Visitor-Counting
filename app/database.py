
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "visitors.db"
SCHEMA_VERSION = 2


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _insert_audit_record(
    connection,
    event_type,
    face_id=None,
    frame_number=None,
    details=None,
    timestamp=None,
):
    """Insert an audit record in the caller's current transaction."""
    payload = details or {}
    if not isinstance(payload, dict):
        raise ValueError("Audit details must be a dictionary")
    connection.execute(
        """
        INSERT INTO audit_log (
            timestamp, event_type, face_id, frame_number, details
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            timestamp or _utc_now(),
            str(event_type),
            face_id,
            frame_number,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        ),
    )


def configure_database_path(path):
    """Set the database path for this process."""
    global DATABASE_PATH
    configured_path = Path(path)
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    DATABASE_PATH = configured_path


def get_connection(database_path=None):
    """Open a database connection and enable foreign keys."""
    path = Path(database_path) if database_path is not None else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(database_path=None):
    """Create or safely migrate the database without removing stored data."""
    with get_connection(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {version} is newer than "
                f"supported version {SCHEMA_VERSION}"
            )

        connection.executescript("""
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                first_seen TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                embedding_model TEXT NOT NULL DEFAULT 'unknown'
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id TEXT NOT NULL,
                event_type TEXT NOT NULL
                    CHECK (event_type IN ('entry', 'exit')),
                timestamp TEXT NOT NULL,
                image_path TEXT,
                source_timestamp REAL,
                frame_number INTEGER,
                FOREIGN KEY (face_id) REFERENCES faces(face_id)
            );

            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                face_id TEXT,
                frame_number INTEGER,
                details TEXT NOT NULL,
                delivered_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_audit_log_pending
            ON audit_log(delivered_at, audit_id);

            CREATE INDEX IF NOT EXISTS idx_events_face_id
            ON events(face_id);

            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events(timestamp);
        """)

        face_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(faces)")
        }
        if "embedding_model" not in face_columns:
            connection.execute(
                "ALTER TABLE faces ADD COLUMN embedding_model TEXT "
                "NOT NULL DEFAULT 'unknown'"
            )

        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(events)")
        }
        if "source_timestamp" not in event_columns:
            connection.execute(
                "ALTER TABLE events ADD COLUMN source_timestamp REAL"
            )
        if "frame_number" not in event_columns:
            connection.execute(
                "ALTER TABLE events ADD COLUMN frame_number INTEGER"
            )

        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            """,
            (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def register_face(
    face_id,
    embedding,
    embedding_model="unknown",
    frame_number=None,
):
    """Register a face and store its numeric embedding."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Embedding must contain finite numeric values")

    timestamp = _utc_now()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO faces (
                face_id, first_seen, embedding, embedding_dimension,
                embedding_model
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                face_id,
                timestamp,
                sqlite3.Binary(vector.tobytes()),
                int(vector.size),
                str(embedding_model),
            ),
        )
        _insert_audit_record(
            connection,
            "FACE_REGISTERED",
            face_id=face_id,
            frame_number=frame_number,
            details={
                "embedding_dimension": int(vector.size),
                "embedding_model": str(embedding_model),
            },
            timestamp=timestamp,
        )

    return face_id


def _read_faces(include_model):
    """Read and validate embeddings from SQLite."""
    columns = "face_id, embedding, embedding_dimension"
    if include_model:
        columns += ", embedding_model"
    with get_connection() as connection:
        rows = connection.execute(
            f"SELECT {columns} FROM faces ORDER BY first_seen"
        ).fetchall()

    faces = []
    for row in rows:
        face_id, blob, dimension = row[:3]
        embedding = np.frombuffer(blob, dtype=np.float32).copy()

        if embedding.size != dimension:
            raise ValueError(
                f"Corrupted embedding dimension for {face_id}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError(f"Corrupted embedding values for {face_id}")

        if include_model:
            faces.append((face_id, embedding, row[3]))
        else:
            faces.append((face_id, embedding))

    return faces


def get_all_faces():
    """Return registered face IDs and embeddings (legacy-compatible shape)."""
    return _read_faces(include_model=False)


def get_all_faces_with_metadata():
    """Return registered face IDs, embeddings, and embedding model names."""
    return _read_faces(include_model=True)


def log_event(
    face_id,
    event_type,
    image_path=None,
    source_timestamp=None,
    frame_number=None,
    preceding_audit_events=None,
):
    """Persist one entry or exit event with UTC and optional media time."""
    if event_type not in {"entry", "exit"}:
        raise ValueError("event_type must be 'entry' or 'exit'")
    if source_timestamp is not None:
        try:
            source_timestamp = float(source_timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "source_timestamp must be finite or None"
            ) from exc
        if not np.isfinite(source_timestamp):
            raise ValueError("source_timestamp must be finite or None")

    if frame_number is not None:
        try:
            parsed_frame_number = int(frame_number)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "frame_number must be a non-negative integer or None"
            ) from exc
        if (
            isinstance(frame_number, bool)
            or parsed_frame_number != frame_number
            or parsed_frame_number < 0
        ):
            raise ValueError(
                "frame_number must be a non-negative integer or None"
            )
        frame_number = parsed_frame_number

    timestamp = _utc_now()

    with get_connection() as connection:
        for audit_event in preceding_audit_events or ():
            _insert_audit_record(
                connection,
                event_type=audit_event["event_type"],
                face_id=audit_event.get("face_id"),
                frame_number=audit_event.get("frame_number"),
                details=audit_event.get("details", {}),
                timestamp=timestamp,
            )

        cursor = connection.execute(
            """
            INSERT INTO events (
                face_id, event_type, timestamp, image_path,
                source_timestamp, frame_number
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                face_id,
                event_type,
                timestamp,
                image_path,
                source_timestamp,
                frame_number,
            ),
        )

        event_id = cursor.lastrowid
        _insert_audit_record(
            connection,
            f"VISITOR_{event_type.upper()}",
            face_id=face_id,
            frame_number=frame_number,
            details={
                "event_id": event_id,
                "image_path": image_path,
                "source_timestamp": source_timestamp,
            },
            timestamp=timestamp,
        )

        return event_id


def record_audit_events(events):
    """Persist a batch of processing audit records in one transaction."""
    if not isinstance(events, (list, tuple)):
        raise TypeError("events must be a list or tuple")
    if not events:
        return 0

    timestamp = _utc_now()
    with get_connection() as connection:
        for event in events:
            if not isinstance(event, dict):
                raise TypeError("Each audit event must be a dictionary")
            frame_number = event.get("frame_number")
            if frame_number is not None:
                try:
                    parsed_frame = int(frame_number)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "frame_number must be a non-negative integer or None"
                    ) from exc
                if (
                    isinstance(frame_number, bool)
                    or parsed_frame != frame_number
                    or parsed_frame < 0
                ):
                    raise ValueError(
                        "frame_number must be a non-negative integer or None"
                    )
                frame_number = parsed_frame
            _insert_audit_record(
                connection,
                event_type=event.get("event_type"),
                face_id=event.get("face_id"),
                frame_number=frame_number,
                details=event.get("details", {}),
                timestamp=event.get("timestamp", timestamp),
            )
    return len(events)


def get_pending_audit_events(limit=500):
    """Read audit records not yet delivered to the append-only audit file."""
    if not 1 <= int(limit) <= 5000:
        raise ValueError("limit must be between 1 and 5000")
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT audit_id, timestamp, event_type, face_id,
                   frame_number, details
            FROM audit_log
            WHERE delivered_at IS NULL
            ORDER BY audit_id
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()


def mark_audit_events_delivered(audit_ids):
    """Mark audit rows delivered after their file write succeeds."""
    if not audit_ids:
        return
    timestamp = _utc_now()
    placeholders = ",".join("?" for _ in audit_ids)
    with get_connection() as connection:
        connection.execute(
            f"""
            UPDATE audit_log SET delivered_at = ?
            WHERE audit_id IN ({placeholders}) AND delivered_at IS NULL
            """,
            [timestamp, *audit_ids],
        )


def get_pending_audit_count():
    with get_connection() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM audit_log WHERE delivered_at IS NULL"
        ).fetchone()[0]


def get_unique_visitor_count():
    """Count unique registered face identities."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM faces"
        ).fetchone()

    return row[0]


def get_event_count():
    """Return the total number of persisted entry/exit events."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM events"
        ).fetchone()

    return row[0]


def get_visit_count():
    """Count recorded visits, defined as persisted entry events."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'entry'"
        ).fetchone()

    return row[0]


def get_recent_events(limit=20):
    """Return recent entry/exit events for local operator display."""
    try:
        limit = int(limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("limit must be between 1 and 100") from exc
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT event_id, face_id, event_type, timestamp,
                   source_timestamp, frame_number
            FROM events
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "event_id": row[0],
            "face_id": row[1],
            "event_type": row[2],
            "timestamp": row[3],
            "source_timestamp": row[4],
            "frame_number": row[5],
        }
        for row in rows
    ]
