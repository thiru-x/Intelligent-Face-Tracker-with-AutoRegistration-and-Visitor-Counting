
from app.tracker import FaceTracker, calculate_iou


def test_identical_boxes_have_iou_one():
    box = (10, 10, 50, 50)
    assert calculate_iou(box, box) == 1.0


def test_non_overlapping_boxes_have_iou_zero():
    assert calculate_iou(
        (0, 0, 10, 10),
        (20, 20, 30, 30),
    ) == 0.0


def test_tracker_keeps_id_for_moving_face():
    tracker = FaceTracker(iou_threshold=0.2)

    first = tracker.update([
        {"bbox": (10, 10, 50, 50), "confidence": 0.9}
    ])
    second = tracker.update([
        {"bbox": (12, 10, 52, 50), "confidence": 0.95}
    ])

    assert first[0]["track_id"] == second[0]["track_id"]


def test_tracker_creates_new_id_for_new_face():
    tracker = FaceTracker(iou_threshold=0.3)

    first = tracker.update([
        {"bbox": (0, 0, 40, 40), "confidence": 0.9}
    ])
    second = tracker.update([
        {"bbox": (100, 100, 140, 140), "confidence": 0.9}
    ])

    assert first[0]["track_id"] != second[0]["track_id"]


def test_tracker_handles_empty_frame():
    tracker = FaceTracker()

    tracker.update([
        {"bbox": (0, 0, 40, 40), "confidence": 0.9}
    ])

    assert tracker.update([]) == []
