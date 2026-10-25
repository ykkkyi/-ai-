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
from starlette.concurrency import run_in_threadpool

from .inference import LocalPredictor


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
                os.environ.get("COTTON_DEVICE", "cpu"),
            )
        yield

    app = FastAPI(title="棉卫士 Cotton Doctor", version="0.1.0", lifespan=lifespan)
    app.state.predictor = predictor
    app.state.result_cache = OrderedDict()
    app.state.cache_lock = threading.Lock()
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
            response, rendered = await run_in_threadpool(app.state.predictor, image)
        except Exception as error:
            raise HTTPException(status_code=500, detail="Detection failed.") from error
        result_id = uuid.uuid4().hex
        with app.state.cache_lock:
            app.state.result_cache[result_id] = rendered
            while len(app.state.result_cache) > app.state.result_cache_size:
                app.state.result_cache.popitem(last=False)
        return {"result_id": result_id, **response}

    @app.get("/api/results/{result_id}.jpg")
    def result_image(result_id: str):
        with app.state.cache_lock:
            content = app.state.result_cache.get(result_id)
            if content is None:
                raise HTTPException(status_code=404, detail="Result expired or not found.")
            app.state.result_cache.move_to_end(result_id)
        from fastapi.responses import Response

        return Response(content=content, media_type="image/jpeg")

    return app


app = create_app(max_upload_mb=int(os.environ.get("COTTON_MAX_UPLOAD_MB", "12")))
