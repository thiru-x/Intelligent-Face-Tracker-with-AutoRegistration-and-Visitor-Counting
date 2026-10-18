import json
import logging
import math
import os
import time
from pathlib import Path

import cv2

from app.detector import FaceDetector
from app.recognizer import FaceRecognizer
from app.tracker import FaceTracker
from app.visitor_counter import VisitorCounter
from app import database
from app.event_logger import (
    configure_logging,
    flush_audit_outbox,
    write_audit_events,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("katomaran.pipeline")


def _positive_config_float(config, key, default):
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a positive finite number")
    return value


def load_config():
    config_path = PROJECT_ROOT / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    return config


def calculate_iou(box_a, box_b):
    """Calculate intersection over union for two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


class VideoPipeline:
    def __init__(self):
        self.config = load_config()
        configure_logging(
            self.config.get("log_directory", "logs"),
            self.config.get("face_image_quality", 95),
        )
        self.network_open_timeout_ms = int(
            max(
                1,
                _positive_config_float(
                    self.config, "network_open_timeout_seconds", 10
                ) * 1000,
            )
        )
        self.network_read_timeout_ms = int(
            max(
                1,
                _positive_config_float(
                    self.config, "network_read_timeout_seconds", 10
                ) * 1000,
            )
        )
        try:
            self.network_reconnect_attempts = int(
                self.config.get("network_reconnect_attempts", 5)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "network_reconnect_attempts must be a non-negative integer"
            ) from exc
        if self.network_reconnect_attempts < 0:
            raise ValueError(
                "network_reconnect_attempts must be a non-negative integer"
            )
        self.network_reconnect_delay_seconds = _positive_config_float(
            self.config, "network_reconnect_delay_seconds", 1
        )
        self.rtsp_transport = str(
            self.config.get("rtsp_transport", "tcp")
        ).lower()
        if self.rtsp_transport not in {"tcp", "udp"}:
            raise ValueError("rtsp_transport must be 'tcp' or 'udp'")
        # FFmpeg reads this process-level option when opening an RTSP stream.
        # Respect an explicit deployment override.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{self.rtsp_transport}",
        )

        database.configure_database_path(
            self.config.get("database_path", "data/visitors.db")
        )
        database.initialize_database()
        flush_audit_outbox()

        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()

        self.tracker = FaceTracker(
            max_lost_frames=max(
                1,
                int(self.config.get("tracker_max_lost_frames", 30)),
            )
        )

        self.visitors = VisitorCounter(
            self.recognizer,
            exit_timeout_seconds=float(
                self.config.get("camera_exit_timeout_seconds", 3)
            ),
        )

        self.frame_skip = max(
            1,
            int(self.config.get("detection_skip_frames", 1)),
        )
        self.embedding_box_iou_threshold = float(
            self.config.get("embedding_box_iou_threshold", 0.3)
        )
        if not 0.0 <= self.embedding_box_iou_threshold <= 1.0:
            raise ValueError(
                "embedding_box_iou_threshold must be between 0 and 1"
            )

        # IDs observed during this process_video() session only.
        # This is intentionally separate from the full SQLite identity count.
        self.session_visitor_ids = set()
        self.registered_identity_count = database.get_unique_visitor_count()
        self.persisted_visit_count = database.get_visit_count()

    @staticmethod
    def _match_tracks_to_faces(tracks, faces):
        """Assign each detected face to at most one track using IoU."""
        candidates = []

        for track_index, track in enumerate(tracks):
            for face_index, face in enumerate(faces):
                overlap = calculate_iou(track["bbox"], face["bbox"])
                if overlap > 0:
                    candidates.append((overlap, track_index, face_index))

        candidates.sort(reverse=True)

        assignments = {}
        used_tracks = set()
        used_faces = set()

        for _, track_index, face_index in candidates:
            if track_index in used_tracks or face_index in used_faces:
                continue

            assignments[track_index] = face_index
            used_tracks.add(track_index)
            used_faces.add(face_index)

        return assignments

    @staticmethod
    def _match_yolo_faces_to_embeddings(
        yolo_faces,
        face_features,
        minimum_iou,
    ):
        """Pair YOLO boxes with one-to-one InsightFace embeddings."""
        candidates = []
        for yolo_index, yolo_face in enumerate(yolo_faces):
            for feature_index, feature in enumerate(face_features):
                overlap = calculate_iou(
                    yolo_face["bbox"], feature["bbox"]
                )
                if overlap >= minimum_iou:
                    candidates.append((overlap, yolo_index, feature_index))

        candidates.sort(reverse=True)
        yolo_to_feature = {}
        used_features = set()
        for _, yolo_index, feature_index in candidates:
            if (
                yolo_index in yolo_to_feature
                or feature_index in used_features
            ):
                continue
            yolo_to_feature[yolo_index] = feature_index
            used_features.add(feature_index)

        return yolo_to_feature

    @staticmethod
    def _is_live_source(video_source):
        if isinstance(video_source, int):
            return True

        if isinstance(video_source, str):
            return video_source.lower().startswith(
                ("rtsp://", "rtsps://", "http://", "https://")
            )

        return False

    @staticmethod
    def _is_network_source(video_source):
        return (
            isinstance(video_source, str)
            and video_source.lower().startswith(
                ("rtsp://", "rtsps://", "http://", "https://")
            )
        )

    def _open_capture(self, video_source):
        """Open a video source with bounded network open/read timeouts."""
        if not self._is_network_source(video_source):
            return cv2.VideoCapture(video_source)

        timeout_parameters = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.network_open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.network_read_timeout_ms,
        ]
        return cv2.VideoCapture(
            video_source,
            cv2.CAP_FFMPEG,
            timeout_parameters,
        )

    def _reopen_network_capture(
        self,
        video_source,
        stop_event,
        status_callback,
        reason,
    ):
        """Try bounded network reconnections and report progress."""
        for attempt in range(1, self.network_reconnect_attempts + 1):
            if stop_event is not None and stop_event.is_set():
                return None

            if status_callback is not None:
                status_callback({
                    "state": "reconnecting",
                    "error": (
                        f"{reason}; reconnecting "
                        f"({attempt}/{self.network_reconnect_attempts})"
                    ),
                })

            if stop_event is not None:
                if stop_event.wait(self.network_reconnect_delay_seconds):
                    return None
            else:
                time.sleep(self.network_reconnect_delay_seconds)

            candidate = self._open_capture(video_source)
            if candidate.isOpened():
                return candidate
            candidate.release()

        return None

    @staticmethod
    def _video_timestamp(capture, frame_number, fps, previous_timestamp):
        """
        Return a monotonically increasing elapsed-video timestamp.

        CAP_PROP_POS_MSEC is preferred for files. Frame position is the
        fallback if the backend reports an invalid or repeated timestamp.
        """
        milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = milliseconds / 1000.0

        if not math.isfinite(timestamp) or timestamp < 0:
            timestamp = (frame_number - 1) / fps

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            timestamp = (frame_number - 1) / fps

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            timestamp = previous_timestamp + (1.0 / fps)

        return timestamp

    def process_video(
        self,
        video_source=None,
        display=True,
        frame_callback=None,
        status_callback=None,
        stop_event=None,
    ):
        # Reset the per-run visitor set whenever a new source is processed.
        self.session_visitor_ids.clear()
        self.registered_identity_count = database.get_unique_visitor_count()
        self.persisted_visit_count = database.get_visit_count()
        self.tracker = FaceTracker(
            max_lost_frames=max(
                1,
                int(self.config.get("tracker_max_lost_frames", 30)),
            )
        )

        if video_source is None:
            if self.config.get("use_rtsp", False):
                video_source = self.config.get("rtsp_url", "")
                if not video_source:
                    raise ValueError(
                        "use_rtsp is true, but rtsp_url is empty"
                    )
            else:
                video_source = self.config.get(
                    "video_source",
                    "sample_video.mp4",
                )

        if isinstance(video_source, str) and "://" not in video_source:
            candidate = Path(video_source)
            if not candidate.is_absolute():
                video_source = str(PROJECT_ROOT / candidate)

        live_source = self._is_live_source(video_source)
        network_source = self._is_network_source(video_source)
        capture = self._open_capture(video_source)

        if not capture.isOpened():
            capture.release()
            capture = None
            if not network_source:
                raise RuntimeError(
                    "Could not open video source. Check its path or camera "
                    "permissions."
                )
            capture = self._reopen_network_capture(
                video_source,
                stop_event,
                status_callback,
                "Could not open camera stream",
            )
            if capture is None:
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("Video source startup was cancelled")
                raise RuntimeError(
                    "Could not open network camera stream after "
                    "reconnection attempts"
                )

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 0.0
            timestamp_fps = 30.0
        else:
            timestamp_fps = source_fps

        frame_number = 0
        frames_processed = 0
        yolo_detection_total = 0
        insightface_detection_total = 0
        unmatched_yolo_total = 0
        capture_read_seconds = 0.0
        yolo_inference_seconds = 0.0
        embedding_inference_seconds = 0.0
        visitor_processing_seconds = 0.0
        display_seconds = 0.0
        previous_video_timestamp = None
        last_source_timestamp = None
        reconnected = False
        run_started = time.perf_counter()

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                if capture is None:
                    raise RuntimeError(
                        "Network camera stream could not be reopened"
                    )

                read_started = time.perf_counter()
                success, frame = capture.read()
                capture_read_seconds += time.perf_counter() - read_started
                if not success:
                    if stop_event is not None and stop_event.is_set():
                        break
                    if not network_source:
                        break
                    capture.release()
                    capture = None
                    capture = self._reopen_network_capture(
                        video_source,
                        stop_event,
                        status_callback,
                        "Camera stream interrupted",
                    )

                    if capture is None:
                        if stop_event is not None and stop_event.is_set():
                            break
                        raise RuntimeError(
                            "Network camera stream disconnected and "
                            "reconnection attempts were exhausted"
                        )
                    reconnected = True
                    continue

                if reconnected:
                    # Close visits that timed out while the live source was
                    # disconnected; a returning observation starts a visit.
                    self.visitors.expire_missing_visitors(
                        now=time.monotonic(),
                        source_timestamp=None,
                        frame_number=frame_number,
                    )
                    reconnected = False

                frame_number += 1

                if live_source:
                    timestamp = time.monotonic()
                else:
                    timestamp = self._video_timestamp(
                        capture,
                        frame_number,
                        timestamp_fps,
                        previous_video_timestamp,
                    )

                previous_video_timestamp = timestamp
                last_source_timestamp = timestamp

                # Skip expensive processing on frames not selected by config.
                if (frame_number - 1) % self.frame_skip != 0:
                    continue

                frames_processed += 1

                yolo_started = time.perf_counter()
                yolo_faces = self.detector.detect(frame)
                yolo_inference_seconds += time.perf_counter() - yolo_started
                yolo_detection_total += len(yolo_faces)

                embedding_started = time.perf_counter()
                face_features = self.recognizer.extract_faces(frame)
                embedding_inference_seconds += (
                    time.perf_counter() - embedding_started
                )
                insightface_detection_total += len(face_features)

                frame_audit_events = [
                    {
                        "event_type": "FACE_DETECTED",
                        "frame_number": frame_number,
                        "details": {
                            "bbox": list(face["bbox"]),
                            "confidence": float(face["confidence"]),
                        },
                    }
                    for face in yolo_faces
                ]
                frame_audit_events.extend(
                    {
                        "event_type": "FACE_EMBEDDING_GENERATED",
                        "frame_number": frame_number,
                        "details": {
                            "bbox": list(face["bbox"]),
                            "confidence": float(face["confidence"]),
                            "embedding_dimension": int(
                                len(face["embedding"])
                            ),
                            "embedding_model": getattr(
                                self.recognizer,
                                "embedding_model",
                                "unknown",
                            ),
                        },
                    }
                    for face in face_features
                )

                # Match embeddings to YOLO boxes one-to-one. The embedding
                # model has its own face detector; IoU prevents assigning its
                # embedding to an unrelated YOLO detection.
                yolo_to_feature = self._match_yolo_faces_to_embeddings(
                    yolo_faces,
                    face_features,
                    self.embedding_box_iou_threshold,
                )

                unmatched_yolo_total += (
                    len(yolo_faces) - len(yolo_to_feature)
                )

                tracks = self.tracker.update(yolo_faces)
                track_assignments = self._match_tracks_to_faces(
                    tracks,
                    yolo_faces,
                )

                recognized_faces = []

                for track_index, yolo_index in track_assignments.items():
                    feature_index = yolo_to_feature.get(yolo_index)
                    track = tracks[track_index]
                    frame_audit_events.append({
                        "event_type": "FACE_TRACKED",
                        "frame_number": frame_number,
                        "details": {
                            "track_id": track["track_id"],
                            "bbox": list(yolo_faces[yolo_index]["bbox"]),
                            "has_embedding": feature_index is not None,
                        },
                    })
                    embedding = (
                        face_features[feature_index]["embedding"]
                        if feature_index is not None
                        else None
                    )

                    recognized_faces.append(
                        {
                            "bbox": tuple(
                                int(value)
                                for value in yolo_faces[yolo_index]["bbox"]
                            ),
                            "embedding": embedding,
                            "track_id": track["track_id"],
                            "confidence": yolo_faces[yolo_index][
                                "confidence"
                            ],
                        }
                    )

                visitor_started = time.perf_counter()
                visitors = self.visitors.process_faces(
                    recognized_faces,
                    frame,
                    now=timestamp,
                    source_timestamp=None if live_source else timestamp,
                    frame_number=frame_number,
                    audit_events=frame_audit_events,
                )
                visitor_processing_seconds += (
                    time.perf_counter() - visitor_started
                )

                # Count each identity at most once in this processing session.
                self.session_visitor_ids.update(
                    visitor["face_id"] for visitor in visitors
                )
                self.registered_identity_count += sum(
                    1 for visitor in visitors if visitor["is_new_visitor"]
                )
                self.persisted_visit_count += sum(
                    1 for visitor in visitors if visitor["is_new_visit"]
                )

                if display or frame_callback is not None:
                    display_started = time.perf_counter()
                    visitors_by_track = {
                        visitor.get("track_id"): visitor
                        for visitor in visitors
                    }
                    for track_index, yolo_index in track_assignments.items():
                        track = tracks[track_index]
                        visitor = visitors_by_track.get(track["track_id"])
                        x1, y1, x2, y2 = yolo_faces[yolo_index]["bbox"]
                        label = (
                            visitor["face_id"]
                            if visitor is not None
                            else "UNKNOWN"
                        )
                        color = (0, 200, 0) if visitor is not None else (
                            0, 165, 255
                        )
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                    visible_count = len(
                        {visitor["face_id"] for visitor in visitors}
                    )

                    elapsed = max(
                        time.perf_counter() - run_started, 1e-9
                    )
                    processing_fps = frames_processed / elapsed
                    status = (
                        f"Faces: {len(yolo_faces)} "
                        f"| Known present: {visible_count} "
                        f"| Unique all-time: {self.registered_identity_count} "
                        f"| Unique this run: {len(self.session_visitor_ids)} "
                        f"| Visits: {self.persisted_visit_count} "
                        f"| Process FPS: {processing_fps:.1f}"
                    )

                    cv2.putText(
                        frame,
                        status,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                    if display:
                        cv2.imshow("Katomaran Face Tracker", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    if frame_callback is not None:
                        frame_callback(frame)

                    if status_callback is not None:
                        status_callback({
                            "state": "running",
                            "error": None,
                            "frames_read": frame_number,
                            "frames_processed": frames_processed,
                            "faces_detected": len(yolo_faces),
                            "present": visible_count,
                            "unique_this_run": len(
                                self.session_visitor_ids
                            ),
                            "unique_identities": (
                                self.registered_identity_count
                            ),
                            "visits": self.persisted_visit_count,
                            "source_fps": source_fps,
                            "processing_fps": frames_processed / max(
                                time.perf_counter() - run_started, 1e-9
                            ),
                        })

                    display_seconds += time.perf_counter() - display_started

        finally:
            try:
                self.visitors.close_active_visits(
                    source_timestamp=(
                        None if live_source else last_source_timestamp
                    ),
                    frame_number=frame_number,
                )
            finally:
                if capture is not None:
                    capture.release()
                if display:
                    cv2.destroyAllWindows()

        elapsed_seconds = max(time.perf_counter() - run_started, 1e-9)
        registered_identities = self.visitors.unique_visitor_count()

        summary = {
            "frames_read": frame_number,
            "frames_processed": frames_processed,
            "yolo_detections": yolo_detection_total,
            "insightface_detections": insightface_detection_total,
            "unmatched_yolo_detections": unmatched_yolo_total,
            # Distinct face IDs observed during this video-processing session.
            "unique_visitors": len(self.session_visitor_ids),
            # All identities accumulated in the persistent SQLite database.
            "registered_identities": registered_identities,
            "visits": self.visitors.visit_count(),
            "source_video_fps": source_fps,
            "capture_read_fps": (
                frame_number / capture_read_seconds
                if capture_read_seconds > 0
                else 0.0
            ),
            "processing_fps": frames_processed / elapsed_seconds,
            "inference_fps": (
                frames_processed
                / (yolo_inference_seconds + embedding_inference_seconds)
                if yolo_inference_seconds + embedding_inference_seconds > 0
                else 0.0
            ),
            "capture_read_seconds": capture_read_seconds,
            "yolo_inference_seconds": yolo_inference_seconds,
            "embedding_inference_seconds": embedding_inference_seconds,
            "visitor_processing_seconds": visitor_processing_seconds,
            "display_seconds": display_seconds,
            "elapsed_seconds": elapsed_seconds,
        }

        if status_callback is not None:
            status_callback({
                "state": "stopped",
                "frames_read": frame_number,
                "frames_processed": frames_processed,
                "faces_detected": 0,
                "present": 0,
                "unique_this_run": len(self.session_visitor_ids),
                "unique_identities": registered_identities,
                "visits": summary["visits"],
                "source_fps": source_fps,
                "processing_fps": summary["processing_fps"],
            })

        print(f"Total video frames read: {summary['frames_read']}")
        print(f"Frames processed: {summary['frames_processed']}")
        print(f"YOLO detections: {summary['yolo_detections']}")
        print(f"InsightFace detections: {summary['insightface_detections']}")
        print(f"YOLO detections without embeddings: {summary['unmatched_yolo_detections']}")
        print(f"Unique visitors this run: {summary['unique_visitors']}")
        print(f"Total registered identities in database: {summary['registered_identities']}")
        print(f"Total visits: {summary['visits']}")
        print(f"Source-reported FPS: {summary['source_video_fps']:.2f}")
        print(f"Capture/read FPS: {summary['capture_read_fps']:.2f}")
        print(f"Processed FPS: {summary['processing_fps']:.2f}")
        print(f"YOLO+embedding inference FPS: {summary['inference_fps']:.2f}")
        print(
            "Inference seconds: "
            f"YOLO={summary['yolo_inference_seconds']:.3f}, "
            "embedding="
            f"{summary['embedding_inference_seconds']:.3f}"
        )

        return summary


if __name__ == "__main__":
    VideoPipeline().process_video()
