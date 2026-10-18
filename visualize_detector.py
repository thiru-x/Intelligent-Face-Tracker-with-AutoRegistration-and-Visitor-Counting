
import cv2

from app.detector import FaceDetector

VIDEO_PATH = "sample_video.mp4"
OUTPUT_PATH = "detector_preview.jpg"


def main():
    detector = FaceDetector()
    capture = cv2.VideoCapture(VIDEO_PATH)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    try:
        # Choose a frame where the diagnostic found detections.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 429)
        success, frame = capture.read()

        if not success:
            raise RuntimeError("Could not read frame 429")

        detections = detector.detect(frame)

        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"Face {index}: {confidence:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if not cv2.imwrite(OUTPUT_PATH, frame):
            raise RuntimeError("Could not save preview image")

        print("Frame:", 429)
        print("Detections:", len(detections))
        print("Saved:", OUTPUT_PATH)

    finally:
        capture.release()


if __name__ == "__main__":
    main()
