
from dataclasses import dataclass
from typing import Optional

BBox = tuple[int, int, int, int]


def calculate_iou(box_a: BBox, box_b: BBox) -> float:
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


def _center(box: BBox) -> tuple[float, float]:
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _predict_bbox(track: "Track") -> BBox:
    """Predict the next box using a conservative constant-velocity model."""
    x1, y1, x2, y2 = track.bbox
    vx, vy = track.velocity

    # Limit extrapolation so a missed detection cannot send a track
    # arbitrarily far away.
    steps = min(track.lost_frames + 1, 3)
    dx = round(vx * steps)
    dy = round(vy * steps)

    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    confidence: float
    lost_frames: int = 0
    face_id: Optional[str] = None
    velocity: tuple[float, float] = (0.0, 0.0)


class FaceTracker:
    """
    Lightweight face tracker using IoU, motion prediction and one-to-one
    greedy assignment.

    Track IDs identify temporary tracks, not verified human identities.
    """

    def __init__(
        self,
        max_lost_frames: int = 30,
        iou_threshold: float = 0.3,
    ):
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames cannot be negative")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.max_lost_frames = int(max_lost_frames)
        self.iou_threshold = float(iou_threshold)
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1

    @staticmethod
    def _validate_detection(detection: dict) -> Optional[dict]:
        """Validate a detection and return a normalized representation."""
        if not isinstance(detection, dict):
            raise TypeError("Each detection must be a dictionary")

        if "bbox" not in detection:
            raise ValueError("Each detection must contain 'bbox'")

        raw_bbox = detection["bbox"]

        try:
            if len(raw_bbox) != 4:
                raise ValueError(
                    "Each bounding box must have four coordinates"
                )

            bbox = tuple(int(value) for value in raw_bbox)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Bounding box must contain four numeric coordinates"
            ) from exc

        x1, y1, x2, y2 = bbox

        # Ignore degenerate boxes instead of creating invalid tracks.
        if x2 <= x1 or y2 <= y1:
            return None

        try:
            confidence = float(detection.get("confidence", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Detection confidence must be numeric") from exc

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1")

        return {"bbox": bbox, "confidence": confidence}

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Update tracks from current-frame detections.

        Input:
            [{"bbox": (x1, y1, x2, y2), "confidence": 0.95}]

        Output:
            [{"track_id": 1, "bbox": (...), "confidence": 0.95,
              "face_id": None}]

        Only tracks visible in the current frame are returned.
        """
        if detections is None:
            detections = []

        valid_detections = []

        for detection in detections:
            normalized = self._validate_detection(detection)
            if normalized is not None:
                valid_detections.append(normalized)

        # Save predicted positions before increasing the lost-frame count.
        predicted_boxes = {
            track_id: _predict_bbox(track)
            for track_id, track in self.tracks.items()
        }

        for track in self.tracks.values():
            track.lost_frames += 1

        # Candidate associations: predicted-box overlap first.
        candidates = []

        for track_id, track in self.tracks.items():
            predicted_box = predicted_boxes[track_id]

            for detection_index, detection in enumerate(valid_detections):
                detection_box = detection["bbox"]
                overlap = calculate_iou(predicted_box, detection_box)

                if overlap >= self.iou_threshold:
                    candidates.append(
                        (overlap, track_id, detection_index)
                    )

        # Highest-IoU associations are accepted first. Each track and
        # detection can participate in at most one association per frame.
        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            track = self.tracks[track_id]
            detection = valid_detections[detection_index]

            old_center = _center(track.bbox)
            new_center = _center(detection["bbox"])

            measured_velocity = (
                new_center[0] - old_center[0],
                new_center[1] - old_center[1],
            )

            # Smooth motion estimates to reduce sudden jumps.
            previous_vx, previous_vy = track.velocity
            measured_vx, measured_vy = measured_velocity

            track.velocity = (
                0.5 * previous_vx + 0.5 * measured_vx,
                0.5 * previous_vy + 0.5 * measured_vy,
            )

            track.bbox = detection["bbox"]
            track.confidence = detection["confidence"]
            track.lost_frames = 0

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        # Create a new track for each unmatched valid detection.
        for detection_index, detection in enumerate(valid_detections):
            if detection_index in matched_detections:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = Track(
                track_id=track_id,
                bbox=detection["bbox"],
                confidence=detection["confidence"],
            )

            matched_tracks.add(track_id)

        # Expire tracks that exceeded the allowed number of missed frames.
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.lost_frames > self.max_lost_frames
        ]

        for track_id in expired_ids:
            del self.tracks[track_id]
            matched_tracks.discard(track_id)

        # Return only tracks that were matched or created this frame.
        return [
            {
                "track_id": track.track_id,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "face_id": track.face_id,
            }
            for track_id, track in sorted(self.tracks.items())
            if track_id in matched_tracks
        ]
