
import json
import logging
import time
import uuid
from pathlib import Path

import numpy as np

from app import database
from app.event_logger import flush_audit_outbox
from app.event_logger import save_face_image
from app.event_logger import remove_face_image
from app.event_logger import write_audit_events


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("katomaran.visitors")


def load_config():
    """Load application configuration."""
    config_path = PROJECT_ROOT / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")

    return config


class VisitorCounter:
    """Register face identities and record entry/exit events."""

    def __init__(self, recognizer, exit_timeout_seconds=None):
        config = load_config()
        self.recognizer = recognizer

        self.exit_timeout_seconds = float(
            exit_timeout_seconds
            if exit_timeout_seconds is not None
            else config.get("camera_exit_timeout_seconds", 3)
        )

        if (
            not np.isfinite(self.exit_timeout_seconds)
            or self.exit_timeout_seconds <= 0
        ):
            raise ValueError(
                "exit_timeout_seconds must be a finite positive number"
            )

        # Persistent face ID -> last observation time.
        self.active_visitors = {}

        # Persistent face ID -> most recent valid face crop.
        self.last_face_images = {}

        # Tracker ID -> persistent face ID.
        self.track_to_face_id = {}

        # Tracker ID -> last observation time.
        self.track_last_seen = {}

    @staticmethod
    def _valid_embedding(embedding):
        """Check that an embedding is finite, non-empty and non-zero."""
        if embedding is None:
            return False

        try:
            vector = np.array(
                embedding, dtype=np.float32, copy=True
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return False

        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return False

        norm = float(np.linalg.norm(vector))
        return np.isfinite(norm) and norm > 1e-12

    @staticmethod
    def _crop_face(frame, bbox):
        """Safely crop a face from a BGR frame."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or bbox is None
        ):
            return None

        try:
            coordinates = tuple(int(value) for value in bbox)
        except (TypeError, ValueError, OverflowError):
            return None

        if len(coordinates) != 4:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = coordinates

        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size else None

    def _identify_or_register(self, embedding, frame_number=None):
        """Match an embedding or register a new persistent identity."""
        if not self._valid_embedding(embedding):
            raise ValueError("Cannot register an invalid face embedding")

        vector = np.asarray(
            embedding, dtype=np.float32
        ).reshape(-1)
        vector /= np.linalg.norm(vector)

        face_id, similarity = self.recognizer.match_embedding(vector)

        if face_id is not None:
            return face_id, float(similarity), False

        new_face_id = f"FACE_{uuid.uuid4().hex[:12].upper()}"

        database.register_face(
            new_face_id,
            vector,
            embedding_model=getattr(
                self.recognizer, "embedding_model", "unknown"
            ),
            frame_number=frame_number,
        )

        return new_face_id, float(similarity), True

    def _record_event(
        self,
        face_id,
        event_type,
        face_image,
        source_timestamp=None,
        frame_number=None,
        preceding_audit_events=None,
    ):
        """Save event evidence and persist the event."""
        if event_type not in {"entry", "exit"}:
            raise ValueError("event_type must be 'entry' or 'exit'")

        category = "entries" if event_type == "entry" else "exits"

        image_path = save_face_image(
            face_image,
            category,
            face_id,
        )
        try:
            database.log_event(
                face_id=face_id,
                event_type=event_type,
                image_path=image_path,
                source_timestamp=source_timestamp,
                frame_number=frame_number,
                preceding_audit_events=preceding_audit_events,
            )
        except Exception:
            try:
                remove_face_image(image_path)
            except OSError:
                LOGGER.exception(
                    "Could not remove orphaned event image for %s",
                    face_id,
                )
            raise

    def process_faces(
        self,
        recognized_faces,
        frame,
        now=None,
        source_timestamp=None,
        frame_number=None,
        audit_events=None,
    ):
        """
        Process face observations.

        Each face dictionary should contain:
            bbox: (x1, y1, x2, y2)
            embedding: face embedding vector
            track_id: optional tracker ID

        `now` is injectable for deterministic tests. Use one consistent
        time base throughout a processing session.
        """
        current_time = time.monotonic() if now is None else float(now)

        if not np.isfinite(current_time):
            raise ValueError("now must be a finite timestamp")

        self.last_source_timestamp = source_timestamp
        self.last_frame_number = frame_number
        processed = []
        observed_face_ids = set()
        observed_track_ids = set()
        write_audit_events(audit_events or ())
        pending_audit_events = []

        for face in recognized_faces:
            if not isinstance(face, dict):
                continue

            embedding = face.get("embedding")
            bbox = face.get("bbox")
            track_id = face.get("track_id")

            if bbox is None:
                continue

            # If a tracker ID has been absent beyond the timeout, do not
            # blindly reuse its cached identity when it reappears.
            if track_id is not None and track_id in self.track_to_face_id:
                last_seen = self.track_last_seen.get(track_id)

                if (
                    last_seen is not None
                    and current_time - last_seen >= self.exit_timeout_seconds
                ):
                    self.track_to_face_id.pop(track_id, None)
                    self.track_last_seen.pop(track_id, None)

            valid_embedding = self._valid_embedding(embedding)
            has_tracked_identity = (
                track_id is not None
                and track_id in self.track_to_face_id
            )
            if not valid_embedding and not has_tracked_identity:
                write_audit_events([{
                    "event_type": "FACE_RECOGNITION_SKIPPED",
                    "frame_number": frame_number,
                    "details": {
                        "track_id": track_id,
                        "reason": "embedding unavailable",
                    },
                }])
                continue

            recognition_audit = None
            if track_id is not None and track_id in self.track_to_face_id:
                face_id = self.track_to_face_id[track_id]
                similarity = None
                is_new = False
                recognition_method = "active_track"
            else:
                face_id, similarity, is_new = (
                    self._identify_or_register(
                        embedding,
                        frame_number=frame_number,
                    )
                )
                recognition_method = (
                    "new_identity" if is_new else "embedding_match"
                )
                recognition_audit = {
                    "event_type": "FACE_RECOGNIZED",
                    "face_id": face_id,
                    "frame_number": frame_number,
                    "details": {
                        "method": recognition_method,
                        "similarity": similarity,
                        "is_new_identity": is_new,
                    },
                }

                if track_id is not None:
                    self.track_to_face_id[track_id] = face_id

            if face_id in observed_face_ids:
                if recognition_audit is not None:
                    pending_audit_events.append(recognition_audit)
                pending_audit_events.append({
                    "event_type": "DUPLICATE_IDENTITY_SUPPRESSED",
                    "face_id": face_id,
                    "frame_number": frame_number,
                    "details": {"track_id": track_id},
                })
                LOGGER.warning(
                    "Ignoring duplicate face identity %s in one frame",
                    face_id,
                )
                continue

            face_image = self._crop_face(frame, bbox)

            # Log an entry only when this identity becomes active.
            is_new_visit = face_id not in self.active_visitors
            if is_new_visit:
                preceding_audit_events = list(pending_audit_events)
                pending_audit_events.clear()
                if recognition_audit is not None:
                    preceding_audit_events.append(recognition_audit)
                self._record_event(
                    face_id,
                    "entry",
                    face_image,
                    source_timestamp=source_timestamp,
                    frame_number=frame_number,
                    preceding_audit_events=preceding_audit_events,
                )
            elif recognition_audit is not None:
                pending_audit_events.append(recognition_audit)

            self.active_visitors[face_id] = current_time
            observed_face_ids.add(face_id)

            if track_id is not None:
                self.track_last_seen[track_id] = current_time
                observed_track_ids.add(track_id)

            if face_image is not None:
                self.last_face_images[face_id] = face_image

            processed.append({
                **face,
                "face_id": face_id,
                "similarity": similarity,
                "is_new_visitor": is_new,
                "is_new_visit": is_new_visit,
            })

        self.expire_missing_visitors(
            now=current_time,
            observed_face_ids=observed_face_ids,
            observed_track_ids=observed_track_ids,
            source_timestamp=source_timestamp,
            frame_number=frame_number,
            flush_outbox=False,
        )

        if pending_audit_events:
            write_audit_events(pending_audit_events)
        else:
            flush_audit_outbox()
        return processed

    def expire_missing_visitors(
        self,
        now=None,
        observed_face_ids=None,
        observed_track_ids=None,
        source_timestamp=None,
        frame_number=None,
        flush_outbox=True,
    ):
        """Record exits for identities absent longer than the timeout."""
        current_time = time.monotonic() if now is None else float(now)

        if not np.isfinite(current_time):
            raise ValueError("now must be a finite timestamp")

        if source_timestamp is None:
            source_timestamp = getattr(self, "last_source_timestamp", None)
        if frame_number is None:
            frame_number = getattr(self, "last_frame_number", None)

        observed_faces = set(observed_face_ids or ())
        observed_tracks = set(observed_track_ids or ())

        expired_ids = [
            face_id
            for face_id, last_seen in self.active_visitors.items()
            if face_id not in observed_faces
            and current_time - last_seen >= self.exit_timeout_seconds
        ]

        for face_id in expired_ids:
            face_image = self.last_face_images.get(face_id)
            self._record_event(
                face_id,
                "exit",
                face_image,
                source_timestamp=source_timestamp,
                frame_number=frame_number,
            )

            self.active_visitors.pop(face_id, None)
            self.last_face_images.pop(face_id, None)

            # Remove tracker mappings associated with the expired visit.
            stale_ids = [
                track_id
                for track_id, mapped_face_id
                in self.track_to_face_id.items()
                if mapped_face_id == face_id
            ]

            for track_id in stale_ids:
                self.track_to_face_id.pop(track_id, None)
                self.track_last_seen.pop(track_id, None)

        # Clear tracks that have independently gone stale.
        stale_tracks = [
            track_id
            for track_id, last_seen in self.track_last_seen.items()
            if track_id not in observed_tracks
            and current_time - last_seen >= self.exit_timeout_seconds
        ]

        for track_id in stale_tracks:
            self.track_last_seen.pop(track_id, None)
            self.track_to_face_id.pop(track_id, None)

        if flush_outbox:
            flush_audit_outbox()

        return expired_ids

    def close_active_visits(self, source_timestamp=None, frame_number=None):
        """Record exits for all active visits when a source ends."""
        if source_timestamp is None:
            source_timestamp = getattr(self, "last_source_timestamp", None)
        if frame_number is None:
            frame_number = getattr(self, "last_frame_number", None)

        closed_face_ids = list(self.active_visitors)
        for face_id in closed_face_ids:
            self._record_event(
                face_id,
                "exit",
                self.last_face_images.get(face_id),
                source_timestamp=source_timestamp,
                frame_number=frame_number,
            )

        self.active_visitors.clear()
        self.last_face_images.clear()
        self.track_to_face_id.clear()
        self.track_last_seen.clear()
        flush_audit_outbox()
        return closed_face_ids

    @staticmethod
    def unique_visitor_count():
        """Return the total number of registered identities in SQLite."""
        return database.get_unique_visitor_count()

    @staticmethod
    def visit_count():
        """Return persisted visit count (number of entry events)."""
        return database.get_visit_count()
