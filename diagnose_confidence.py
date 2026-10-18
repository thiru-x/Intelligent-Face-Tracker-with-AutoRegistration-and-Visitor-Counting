import cv2
from app.detector import FaceDetector

VIDEO_PATH = "sample_video.mp4"
FRAME_NUMBERS = [215, 429, 644, 858, 1073, 1287, 1502]

thresholds = [0.50, 0.30, 0.15]

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

try:
    for frame_number in FRAME_NUMBERS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_number}")
            continue

        print(f"\nFrame {frame_number}")

        for threshold in thresholds:
            detector = FaceDetector(confidence=threshold)
            detections = detector.detect(frame)

            print(
                f"  Confidence {threshold:.2f}: "
                f"{len(detections)} detections"
            )

            # Save a preview for each confidence threshold.
            preview = frame.copy()

            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                confidence = detection["confidence"]

                cv2.rectangle(
                    preview,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    preview,
                    f"{confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            output_path = (
                f"diagnosis_frame_{frame_number}_"
                f"conf_{threshold:.2f}.jpg"
            )

            cv2.imwrite(output_path, preview)

finally:
    cap.release()

print("\nDone. Open the generated JPG previews and compare them.")