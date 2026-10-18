
import cv2
from app.detector import FaceDetector
from app.recognizer import FaceRecognizer


def iou(box_a, box_b):
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


detector = FaceDetector()
recognizer = FaceRecognizer()
capture = cv2.VideoCapture("sample_video.mp4")

if not capture.isOpened():
    raise RuntimeError("Cannot open sample_video.mp4")

try:
    for frame_number in range(1, 181):
        success, frame = capture.read()
        if not success:
            break

        if frame_number % 3 != 0:
            continue

        yolo_faces = detector.detect(frame)
        insight_faces = recognizer.extract_faces(frame)

        matches = 0

        for yolo in yolo_faces:
            best_iou = max(
                (
                    iou(yolo["bbox"], face["bbox"])
                    for face in insight_faces
                ),
                default=0.0,
            )

            if best_iou >= 0.3:
                matches += 1

        print(
            f"Frame {frame_number}: "
            f"YOLO={len(yolo_faces)}, "
            f"InsightFace={len(insight_faces)}, "
            f"overlapping={matches}"
        )
finally:
    capture.release()
