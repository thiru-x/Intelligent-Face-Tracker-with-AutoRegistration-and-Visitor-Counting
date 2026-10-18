
import cv2
from pathlib import Path

from app.detector import FaceDetector
from app.recognizer import FaceRecognizer

video_path = Path("sample_video.mp4")

if not video_path.is_file():
    raise FileNotFoundError(f"Video not found: {video_path.resolve()}")

detector = FaceDetector()
recognizer = FaceRecognizer()

capture = cv2.VideoCapture(str(video_path))

if not capture.isOpened():
    raise RuntimeError(f"Could not open video: {video_path.resolve()}")

frame_number = 0
frames_checked = 0
yolo_total = 0
insightface_total = 0
debug_image_saved = False

try:
    while frames_checked < 60:
        success, frame = capture.read()
        if not success:
            break

        frame_number += 1

        # Sample every third frame to keep this diagnostic manageable.
        if frame_number % 3 != 0:
            continue

        frames_checked += 1

        yolo_faces = detector.detect(frame)
        insight_faces = recognizer.extract_faces(frame)

        yolo_total += len(yolo_faces)
        insightface_total += len(insight_faces)

        print(
            f"Frame {frame_number}: "
            f"YOLO={len(yolo_faces)}, "
            f"InsightFace={len(insight_faces)}"
        )

        # Save one annotated frame for visual inspection.
        if yolo_faces and not debug_image_saved:
            preview = frame.copy()

            for face in yolo_faces:
                x1, y1, x2, y2 = face["bbox"]
                cv2.rectangle(
                    preview, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                cv2.putText(
                    preview,
                    f'{face["confidence"]:.2f}',
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            output_path = Path("debug_detections.jpg")
            if cv2.imwrite(str(output_path), preview):
                print(f"Saved annotated frame: {output_path.resolve()}")
                debug_image_saved = True

finally:
    capture.release()

print("\n--- Diagnostic summary ---")
print("Frames checked:", frames_checked)
print("Total YOLO detections:", yolo_total)
print("Total InsightFace detections:", insightface_total)
print("Database was not modified by this script.")
