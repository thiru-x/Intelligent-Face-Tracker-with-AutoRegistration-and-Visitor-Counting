
import cv2
from ultralytics import YOLO
from pathlib import Path

model_path = Path("models/yolov8n-face.pt")
video_path = Path("sample_video.mp4")

model = YOLO(str(model_path))
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

print("Model classes:", model.names)
print("Video resolution:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "x",
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

try:
    for frame_number in [9, 19, 29, 49, 79, 99]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()

        if not ok:
            print(f"Cannot read frame {frame_number}")
            continue

        print(f"\nFrame {frame_number + 1}")

        for confidence in [0.50, 0.20, 0.05]:
            results = model.predict(
                source=frame,
                conf=confidence,
                imgsz=960,
                max_det=300,
                device="cpu",
                verbose=False,
            )

            boxes = results[0].boxes
            count = 0 if boxes is None else len(boxes)

            print(f"  conf={confidence:.2f}: {count} detections")

            preview = frame.copy()

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )
                    cv2.rectangle(
                        preview, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

            if confidence == 0.05:
                cv2.imwrite(
                    f"raw_yolo_frame_{frame_number + 1}.jpg",
                    preview,
                )
finally:
    cap.release()

print("\nDone. Inspect the raw_yolo_frame_*.jpg previews.")
