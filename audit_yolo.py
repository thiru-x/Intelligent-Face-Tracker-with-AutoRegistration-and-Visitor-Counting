
import cv2
from pathlib import Path
from ultralytics import YOLO
from app.detector import FaceDetector

VIDEO = Path("sample_video.mp4").resolve()
MODEL = Path("models/yolov8n-face.pt").resolve()

raw_model = YOLO(str(MODEL))
detector = FaceDetector()
cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO}")

print("Video:", VIDEO)
print("Model classes:", raw_model.names)

try:
    for frame_no in [9, 19, 29, 49, 79, 99]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()

        if not ok:
            print(f"Could not read frame {frame_no + 1}")
            continue
        raw = raw_model.predict(
            source=frame,
            conf=detector.confidence,
            imgsz=960,
            device="cpu",
            verbose=False,
                )[0]
                
        raw_count = 0 if raw.boxes is None else len(raw.boxes)
        if raw.boxes is not None:
            for box in raw.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(
                    f"  Raw box: {x2-x1:.1f}x{y2-y1:.1f} px, "
                    f"confidence={float(box.conf[0]):.3f}"
                )
            print(f"  Wrapper minimum face size: {detector.min_face_size}px")
        wrapped_count = len(detector.detect(frame))

        print(
            f"Frame {frame_no + 1}: "
            f"raw YOLO={raw_count}, "
            f"FaceDetector={wrapped_count}"
        )
finally:
    cap.release()
