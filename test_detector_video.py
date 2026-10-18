import cv2

from app.detector import FaceDetector

VIDEO_PATH = "sample_video.mp4"

def main():
    detector = FaceDetector()
    capture = cv2.VideoCapture(VIDEO_PATH)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", frame_count)

    # Sample up to 10 frames spread across the video.
    sample_count = min(10, frame_count)
    if sample_count == 0:
        capture.release()
        raise RuntimeError("The video contains no readable frames")

    positions = sorted(
        set(
            round(i * (frame_count - 1) / max(sample_count - 1, 1))
            for i in range(sample_count)
        )
    )

    total_detections = 0

    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()

            if not ok:
                print(f"Frame {position}: could not read")
                continue

            detections = detector.detect(frame)
            total_detections += len(detections)

            print(
                f"Frame {position}: "
                f"{len(detections)} detections"
            )

    finally:
        capture.release()

    print("Total detections across sampled frames:", total_detections)


if __name__ == "__main__":
    main()
