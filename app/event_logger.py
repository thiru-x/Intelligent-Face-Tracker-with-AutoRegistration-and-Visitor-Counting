"""Configured application logging and durable structured audit events."""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2

from app import database


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
EVENTS_LOG = LOG_DIR / "events.log"
APPLICATION_LOG = LOG_DIR / "application.log"
_FILE_HANDLER_MARKER = "_katomaran_application_handler"
_AUDIT_LOCK = threading.Lock()
_AUDIT_IDS_ON_DISK = None
_FACE_IMAGE_QUALITY = 95
logger = logging.getLogger("katomaran")


def _resolve_project_path(path):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def configure_logging(log_directory="logs", face_image_quality=95):
    """Configure the application log and biometric evidence directory."""
    global LOG_DIR, EVENTS_LOG, APPLICATION_LOG, _FACE_IMAGE_QUALITY
    global _AUDIT_IDS_ON_DISK

    LOG_DIR = _resolve_project_path(log_directory)
    EVENTS_LOG = LOG_DIR / "events.log"
    APPLICATION_LOG = LOG_DIR / "application.log"
    _AUDIT_IDS_ON_DISK = None
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    quality = int(face_image_quality)
    if not 1 <= quality <= 100:
        raise ValueError("face_image_quality must be between 1 and 100")
    _FACE_IMAGE_QUALITY = quality

    for handler in logger.handlers:
        if (
            getattr(handler, _FILE_HANDLER_MARKER, False)
            and Path(handler.baseFilename).resolve()
            == APPLICATION_LOG.resolve()
        ):
            return EVENTS_LOG

    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, _FILE_HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    file_handler = RotatingFileHandler(
        APPLICATION_LOG,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    setattr(file_handler, _FILE_HANDLER_MARKER, True)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(file_handler)
    return EVENTS_LOG


def _write_audit_lines(records):
    global _AUDIT_IDS_ON_DISK
    if not records:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _AUDIT_LOCK:
        if _AUDIT_IDS_ON_DISK is None:
            known_ids = set()
            if EVENTS_LOG.is_file():
                with EVENTS_LOG.open("r", encoding="utf-8") as audit_file:
                    for line in audit_file:
                        try:
                            existing = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        audit_id = (
                            existing.get("audit_id")
                            if isinstance(existing, dict)
                            else None
                        )
                        if isinstance(audit_id, int):
                            known_ids.add(audit_id)
            _AUDIT_IDS_ON_DISK = known_ids

        new_records = [
            record for record in records
            if record[0] not in _AUDIT_IDS_ON_DISK
        ]
        if not new_records:
            return

        lines = b"".join(
            (
                json.dumps(
                    {
                        "audit_id": record[0],
                        "timestamp": record[1],
                        "event_type": record[2],
                        "face_id": record[3],
                        "frame_number": record[4],
                        "details": json.loads(record[5]),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for record in new_records
        )
        descriptor = os.open(
            EVENTS_LOG,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            remaining = memoryview(lines)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("Could not append audit records")
                remaining = remaining[written:]
            os.fsync(descriptor)
            _AUDIT_IDS_ON_DISK.update(
                record[0] for record in new_records
            )
        finally:
            os.close(descriptor)


def flush_audit_outbox():
    """Deliver committed audit rows; undelivered rows remain retryable."""
    delivered = 0
    while True:
        records = database.get_pending_audit_events()
        if not records:
            return delivered

        _write_audit_lines(records)
        database.mark_audit_events_delivered(
            [record[0] for record in records]
        )
        delivered += len(records)


def write_audit_events(events):
    """Persist an audit batch before writing it to the append-only file."""
    inserted = database.record_audit_events(events)
    if inserted:
        flush_audit_outbox()
    return inserted


def log_event(event_type, face_id=None, details=None, frame_number=None):
    """Write one structured audit record through the durable outbox."""
    return write_audit_events([{
        "event_type": event_type,
        "face_id": face_id,
        "frame_number": frame_number,
        "details": details or {},
    }])


def save_face_image(face_image, event_type, face_id):
    """Save an event crop under the configured dated image directory."""
    if face_image is None or face_image.size == 0:
        raise ValueError(f"Cannot save an empty face image for {face_id}")

    if event_type not in {"entries", "exits"}:
        raise ValueError("event_type must be 'entries' or 'exits'")

    now = datetime.now(timezone.utc)
    date_folder = now.strftime("%Y-%m-%d")
    output_dir = LOG_DIR / event_type / date_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now.strftime("%H%M%S_%f")
    image_path = output_dir / (
        f"{face_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
    )
    success = cv2.imwrite(
        str(image_path),
        face_image,
        [cv2.IMWRITE_JPEG_QUALITY, _FACE_IMAGE_QUALITY],
    )
    if not success:
        raise OSError(f"Failed to save face image: {image_path}")

    resolved_path = image_path.resolve()
    try:
        relative_path = resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        relative_path = str(resolved_path)
    logger.info(
        "FACE_IMAGE_SAVED | face_id=%s | type=%s | path=%s",
        face_id,
        event_type,
        relative_path,
    )
    return relative_path


def remove_face_image(image_path):
    """Remove an image whose corresponding database event was not committed."""
    if image_path is None:
        return
    candidate = _resolve_project_path(image_path).resolve()
    try:
        candidate.relative_to(LOG_DIR.resolve())
    except ValueError as exc:
        raise ValueError("Image cleanup path is outside the log directory") from exc
    candidate.unlink(missing_ok=True)


def log_system_event(event_type, face_id=None, details=None):
    """Write a non-biometric operational message to application.log."""
    logger.info(
        "%s | face_id=%s | details=%s",
        event_type,
        face_id or "-",
        json.dumps(details or {}, sort_keys=True),
    )
