"""YOLO adapter kept independent from the API and persistence layers."""

from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path
import time
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger("visionwise.inference")


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]
    position: str


class Detector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model: Any | None = None
        self.error: str | None = None
        self.device = "cpu"

    def load(self) -> None:
        if not self.model_path.is_file():
            self.error = "Model weights were not found at the configured path."
            return
        try:
            from ultralytics import YOLO
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = YOLO(str(self.model_path))
            self.error = None
        except Exception as exc:
            logger.exception("Could not initialize the local YOLO model")
            self.error = "Could not initialize the local YOLO model. Check API dependencies and logs."

    def status(self) -> dict[str, Any]:
        return {
            "available": self.model is not None,
            "name": "YOLO11n",
            "weights": self.model_path.name,
            "class_count": len(self.model.names) if self.model is not None else None,
            "device": self.device,
            "error": self.error,
        }

    def predict(self, image_bytes: bytes, confidence: float) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(self.error or "Model is not loaded")
        from PIL import Image
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("The uploaded frame is not a valid image.") from exc
        started = time.perf_counter()
        results = self.model.predict(source=image, conf=confidence, device=self.device, verbose=False)
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections: list[Detection] = []
        width, height = image.size
        names = self.model.names
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                center_x = (x1 + x2) / 2 / max(width, 1)
                position = "left" if center_x < 0.36 else "right" if center_x > 0.64 else "center"
                class_id = int(box.cls[0].item())
                detections.append(Detection(
                    label=str(names[class_id]), confidence=round(float(box.conf[0].item()), 4),
                    box=[round(x1 / max(width, 1), 4), round(y1 / max(height, 1), 4),
                         round(x2 / max(width, 1), 4), round(y2 / max(height, 1), 4)], position=position,
                ))
        detections.sort(key=lambda item: (item.box[0] + item.box[2]) / 2)
        return {"detections": detections, "inference_ms": round(elapsed_ms, 1), "model": self.status()}
