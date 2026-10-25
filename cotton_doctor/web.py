"""Small local FastAPI interface for image-based detector inference."""

from __future__ import annotations

import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


class LocalPredictor:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.lock = threading.Lock()

    def __call__(self, image: np.ndarray) -> tuple[dict, bytes]:
        with self.lock:
            result = self.model.predict(image, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.cpu()
            for xyxy, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                index = int(class_id)
                detections.append({
                    "class_id": index, "class_name": self.model.names[index],
                    "confidence": round(float(confidence), 5),
                    "xyxy": [round(float(value), 2) for value in xyxy],
                })
        rendered = result.plot()
        ok, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Failed to encode annotated image")
        return {"detections": detections, "inference_ms": round(float(result.speed["inference"]), 3)}, buffer.tobytes()


def create_app(predictor=None, *, max_upload_mb: int = 12, result_cache_size: int = 8) -> FastAPI:
    static_dir = Path(__file__).parent / "static"
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if predictor is None:
            model_path = os.environ.get("COTTON_MODEL", "weights/best.pt")
            if not os.path.isfile(model_path):
                raise RuntimeError(f"Model not found: {model_path}. Set COTTON_MODEL to a trained .pt weight.")
            app.state.predictor = LocalPredictor(
                model_path, float(os.environ.get("COTTON_CONF", "0.25")),
                float(os.environ.get("COTTON_IOU", "0.45")),
            )
        yield

    app = FastAPI(title="棉卫士 Cotton Doctor", version="0.1.0", lifespan=lifespan)
    app.state.predictor = predictor
    app.state.result_cache = OrderedDict()
    app.state.max_upload_bytes = max_upload_mb * 1024 * 1024
    app.state.result_cache_size = result_cache_size
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.predictor is not None}

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)):  # noqa: B008
        if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
            raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, or BMP image.")
        content = await file.read(app.state.max_upload_bytes + 1)
        if len(content) > app.state.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds the upload size limit.")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")
        try:
            response, rendered = app.state.predictor(image)
        except Exception as error:
            raise HTTPException(status_code=500, detail="Detection failed.") from error
        result_id = uuid.uuid4().hex
        app.state.result_cache[result_id] = rendered
        while len(app.state.result_cache) > app.state.result_cache_size:
            app.state.result_cache.popitem(last=False)
        return {"result_id": result_id, **response}

    @app.get("/api/results/{result_id}.jpg")
    def result_image(result_id: str):
        content = app.state.result_cache.get(result_id)
        if content is None:
            raise HTTPException(status_code=404, detail="Result expired or not found.")
        app.state.result_cache.move_to_end(result_id)
        from fastapi.responses import Response

        return Response(content=content, media_type="image/jpeg")

    return app


app = create_app()
