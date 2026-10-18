
import json
from pathlib import Path

import numpy as np
from insightface.app import FaceAnalysis

from app.database import get_all_faces_with_metadata


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Load application configuration."""
    config_path = PROJECT_ROOT / "config.json"

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


class FaceRecognizer:
    """Extract face embeddings and match registered identities."""

    def __init__(self, model_name=None, threshold=None, device=None):
        config = load_config()

        self.model_name = model_name or config.get(
            "model_name", "buffalo_l"
        )
        self.embedding_model = self.model_name
        self.threshold = float(
            threshold
            if threshold is not None
            else config.get("recognition_threshold", 0.45)
        )
        self.device = str(
            device or config.get("device", "CPU")
        ).upper()
        self.min_face_detection_score = float(
            config.get("min_face_detection_score", 0.35)
        )
        self.min_face_width = int(config.get("min_face_width", 12))
        self.min_face_height = int(config.get("min_face_height", 12))

        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError(
                "Recognition threshold must be between -1.0 and 1.0"
            )

        if not 0.0 <= self.min_face_detection_score <= 1.0:
            raise ValueError(
                "min_face_detection_score must be between 0 and 1"
            )
        if self.min_face_width < 1 or self.min_face_height < 1:
            raise ValueError("Minimum face dimensions must be positive")

        if self.device not in {"CPU", "GPU"}:
            raise ValueError("device must be 'CPU' or 'GPU'")

        available_providers = set(
            __import__("onnxruntime").get_available_providers()
        )

        if self.device == "GPU" and (
            "CUDAExecutionProvider" not in available_providers
        ):
            raise RuntimeError(
                "GPU was requested, but CUDAExecutionProvider is "
                "unavailable. Install compatible ONNX Runtime GPU "
                "dependencies or set device to CPU in config.json."
            )

        providers = (
            ["CPUExecutionProvider"]
            if self.device == "CPU"
            else [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        )

        self.app = FaceAnalysis(
            name=self.model_name,
            providers=providers,
        )

        ctx_id = 0 if self.device == "GPU" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    @staticmethod
    def _valid_embedding(embedding):
        """Convert an embedding into a validated 1-D float32 vector."""
        if embedding is None:
            return None

        try:
            vector = np.asarray(
                embedding, dtype=np.float32
            ).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return None

        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None

        norm = float(np.linalg.norm(vector))

        if not np.isfinite(norm) or norm <= 1e-12:
            return None

        return vector / norm

    @staticmethod
    def cosine_similarity(first, second):
        """Calculate cosine similarity safely; invalid inputs score -1."""
        first_vector = FaceRecognizer._valid_embedding(first)
        second_vector = FaceRecognizer._valid_embedding(second)

        if first_vector is None or second_vector is None:
            return -1.0

        if first_vector.shape != second_vector.shape:
            return -1.0

        first_norm = float(np.linalg.norm(first_vector))
        second_norm = float(np.linalg.norm(second_vector))

        denominator = first_norm * second_norm

        if not np.isfinite(denominator) or denominator <= 1e-12:
            return -1.0

        similarity = float(
            np.dot(first_vector, second_vector) / denominator
        )

        if not np.isfinite(similarity):
            return -1.0

        return float(np.clip(similarity, -1.0, 1.0))

    def extract_faces(self, frame):
        """
        Detect faces and return bounding boxes, embeddings,
        and confidence scores.
        """
        if (
            frame is None
            or not hasattr(frame, "size")
            or frame.size == 0
        ):
            return []

        results = []
        for face in self.app.get(frame):
            embedding = getattr(face, "normed_embedding", None)

            if embedding is None:
                embedding = getattr(face, "embedding", None)

            vector = self._valid_embedding(embedding)

            if vector is None:
                continue

            try:
                bbox_values = np.asarray(
                    face.bbox, dtype=np.float64
                ).reshape(-1)
            except (TypeError, ValueError, OverflowError):
                continue

            if (
                bbox_values.size != 4
                or not np.all(np.isfinite(bbox_values))
            ):
                continue

            x1, y1, x2, y2 = (
                int(value) for value in bbox_values
            )

            if x2 <= x1 or y2 <= y1:
                continue

            try:
                confidence = float(face.det_score)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(confidence):
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            if confidence < self.min_face_detection_score:
                continue

            if (
                box_width < self.min_face_width
                or box_height < self.min_face_height
            ):
                continue

            results.append({
                "bbox": (x1, y1, x2, y2),
                "embedding": vector.copy(),
                "confidence": confidence,
            })

        return results

    def match_embedding(self, embedding):
        """
        Return (face_id, similarity) for the best accepted match.

        Unknown or invalid embeddings return (None, score).
        """
        query = self._valid_embedding(embedding)

        if query is None:
            return None, -1.0

        best_id = None
        best_score = -1.0

        # Read current records so newly registered identities are
        # available during subsequent comparisons.
        for face_id, stored_embedding, stored_model in (
            get_all_faces_with_metadata()
        ):
            if stored_model not in {"unknown", self.embedding_model}:
                continue

            stored = self._valid_embedding(stored_embedding)

            if stored is None or stored.shape != query.shape:
                continue

            score = self.cosine_similarity(query, stored)

            if score > best_score:
                best_id = face_id
                best_score = score

        if best_id is not None and best_score >= self.threshold:
            return best_id, best_score

        return None, best_score

    def recognize_frame(self, frame):
        """Detect faces, extract embeddings, and match identities."""
        recognized = []

        for face in self.extract_faces(frame):
            face_id, score = self.match_embedding(
                face["embedding"]
            )

            recognized.append({
                **face,
                "face_id": face_id,
                "similarity": score,
            })

        return recognized
