
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Load the project configuration."""
    config_path = PROJECT_ROOT / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")

    return config


class FaceDetector:
    """Detect faces using a compatible, face-trained YOLO model."""

    def __init__(
        self,
        model_path=None,
        confidence=None,
        device=None,
    ):
        config = load_config()

        configured_path = (
            model_path or config.get("yolo_face_model")
        )

        if not configured_path:
            raise ValueError(
                "Missing 'yolo_face_model' in config.json"
            )

        self.model_path = Path(configured_path)

        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path

        self.confidence = float(
            confidence
            if confidence is not None
            else config.get("face_detection_confidence", 0.5)
        )

        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(
                "Detection confidence must be greater than 0 "
                "and at most 1"
            )

        self.device = str(
            device or config.get("device", "CPU")
        ).strip().upper()

        if self.device not in {"CPU", "GPU"}:
            raise ValueError(
                "Device must be either 'CPU' or 'GPU'"
            )

        self.image_size = int(config.get("detection_image_size", 960))

        if self.image_size < 32:
            raise ValueError(
                "detection_image_size must be at least 32"
            )

        self.min_face_size = int(config.get("min_face_size", 12))

        if self.min_face_size < 1:
            raise ValueError("min_face_size must be positive")

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO face model not found: {self.model_path}\n"
                "Check config.json and provide compatible "
                "face-trained YOLO weights."
            )

        # Loading can fail if the weights are corrupt or incompatible.
        self.model = YOLO(str(self.model_path))

    def detect(self, frame):
        """
        Detect faces in a BGR OpenCV frame.

        Returns:
            List of dictionaries containing:
            - bbox: (x1, y1, x2, y2)
            - confidence: detection confidence
        """
        if frame is None:
            return []

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")

        if frame.size == 0:
            return []

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame must be a color image with shape (height, width, 3)"
            )

        height, width = frame.shape[:2]

        if height == 0 or width == 0:
            return []

        # Ultralytics uses 'cpu' for CPU inference and '0' for the
        # first CUDA GPU. A GPU request will raise an error if the
        # installed runtime cannot use that GPU.
        inference_device = "0" if self.device == "GPU" else "cpu"

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=inference_device,
            imgsz=self.image_size,
            max_det=300,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            coordinates = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()

            for coords, score in zip(coordinates, scores):
                if not np.all(np.isfinite(coords)):
                    continue

                score = float(score)

                if not np.isfinite(score):
                    continue

                x1, y1, x2, y2 = (
                    int(round(float(value))) for value in coords
                )

                # Clamp coordinates to the image boundaries.
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    continue

                box_width = x2 - x1
                box_height = y2 - y1

                if (
                    box_width < self.min_face_size
                    or box_height < self.min_face_size
                ):
                    continue

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": score,
                })

        return detections
