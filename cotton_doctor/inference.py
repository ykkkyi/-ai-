"""Local image/video/camera inference, sharing the same detector as the web app."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class LocalPredictor:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45,
                 device: str = "cpu"):
        self.model = YOLO(model_path, task="detect")
        self.conf = conf
        self.iou = iou
        self.device = device
        self.lock = threading.Lock()

    def __call__(self, image: np.ndarray) -> tuple[dict, bytes]:
        with self.lock:
            result = self.model.predict(image, conf=self.conf, iou=self.iou,
                                        device=self.device, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.cpu()
            for xyxy, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                index = int(class_id)
                detections.append({
                    "class_id": index, "class_name": result.names[index],
                    "confidence": round(float(confidence), 5),
                    "xyxy": [round(float(value), 2) for value in xyxy],
                })
        ok, buffer = cv2.imencode(".jpg", result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode annotated image")
        return {"detections": detections,
                "inference_ms": round(float(result.speed["inference"]), 3)}, buffer.tobytes()


def predict_source(model: Path, source: str, output: Path, *, device: str = "cpu",
                   conf: float = 0.25, iou: float = 0.45,
                   max_frames: int | None = None) -> dict:
    """Save annotated images or video plus JSONL; release cameras even on interruption."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    predictor = LocalPredictor(str(model), conf=conf, iou=iou, device=device)
    source_path = Path(source)
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if source_path.is_dir() or source_path.suffix.lower() in suffixes:
        images = ([source_path] if source_path.is_file() else
                  sorted(p for p in source_path.iterdir() if p.suffix.lower() in suffixes))
        if not images:
            raise ValueError("No image files found")
        if max_frames:
            images = images[:max_frames]
        with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
            for index, image_path in enumerate(images):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Cannot read {image_path}")
                result, rendered = predictor(image)
                (output / f"{index:06d}-{image_path.stem}.jpg").write_bytes(rendered)
                stream.write(json.dumps({"source": str(image_path), **result}, ensure_ascii=False) + "\n")
        return {"images": len(images), "output": str(output.resolve())}
    capture = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    writer = None
    frames = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video or camera {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
            while max_frames is None or frames < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                result, encoded = predictor(frame)
                rendered = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if writer is None:
                    height, width = rendered.shape[:2]
                    writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise OSError("Cannot create MP4 output with the installed OpenCV codecs")
                writer.write(rendered)
                stream.write(json.dumps({"frame": frames, "source_time_seconds": frames / fps,
                                         **result}, ensure_ascii=False) + "\n")
                frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not frames:
        raise ValueError("Video/camera returned no readable frames")
    return {"frames": frames, "output": str(output.resolve())}
