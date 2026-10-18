
import cv2

from app.detector import FaceDetector
from app.recognizer import FaceRecognizer


def draw_box(image, bbox, color, label):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


detector = FaceDetector()
recognizer = FaceRecognizer()
capture = cv2.VideoCapture("sample_video.mp4")

if not capture.isOpened():
    raise RuntimeError("Cannot open sample_video.mp4")

try:
    for frame_number in range(1, 118):
        success, frame = capture.read()
        if not success:
            raise RuntimeError("Video ended before frame 117")

    image = frame.copy()
    yolo_faces = detector.detect(frame)
    insight_faces = recognizer.extract_faces(frame)

    for index, face in enumerate(yolo_faces, start=1):
        draw_box(
            image,
            face["bbox"],
            (0, 255, 0),
            f"YOLO {index}: {face['confidence']:.2f}",
        )

    for index, face in enumerate(insight_faces, start=1):
        draw_box(
            image,
            face["bbox"],
            (255, 100, 0),
            f"InsightFace {index}",
        )

    output_path = "debug_matching_overlay.jpg"
    if not cv2.imwrite(output_path, image):
        raise RuntimeError("Could not save overlay image")

    print("YOLO detections:", len(yolo_faces))
    print("InsightFace detections:", len(insight_faces))
    print("Saved:", output_path)

finally:
    capture.release()
