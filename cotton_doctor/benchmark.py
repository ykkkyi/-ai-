"""Repeatable model-size and end-to-end latency measurements."""

from __future__ import annotations

import hashlib
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def _sync(device: str) -> None:
    if str(device) != "cpu":
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def benchmark(model_path: Path, manifest: Path, *, imgsz: int = 640, device: str = "0",
              warmup: int = 10, limit: int | None = None) -> dict:
    """Measure complete image decode + preprocessing + inference + postprocessing latency."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    if warmup < 1 or (limit is not None and limit < 1):
        raise ValueError("Use at least one warmup and one timed image")
    if model_path.suffix != ".pt":
        raise ValueError("This benchmark currently measures PyTorch .pt models")
    paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No images found in {manifest}")
    model = YOLO(str(model_path))

    def run(path: Path) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {path}")
        model.predict(image, imgsz=imgsz, device=device, batch=1, half=False,
                      conf=0.25, iou=0.45, verbose=False)

    for index in range(warmup):
        run(paths[index % len(paths)])
    _sync(device)
    timings = []
    for path in paths:
        start = time.perf_counter()
        run(path)
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    size_bytes = model_path.stat().st_size
    params = sum(parameter.numel() for parameter in model.model.parameters())
    cpu_name = platform.processor() or platform.machine()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    return {
        "model": str(model_path.resolve()), "manifest": str(manifest.resolve()),
        "images": len(paths), "imgsz": imgsz, "device": device,
        "warmup": warmup, "model_file_mib": round(size_bytes / 2**20, 3),
        "weight_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "parameters": params, "latency_ms_median": round(statistics.median(timings), 3),
        "latency_ms_p95": round(float(np.percentile(timings, 95)), 3),
        "throughput_fps": round(1000 * len(timings) / sum(timings), 3),
        "batch": 1, "precision": "FP32", "confidence": 0.25, "nms_iou": 0.45,
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "platform": platform.platform(),
        "hardware": (cpu_name if str(device) == "cpu"
                     else torch.cuda.get_device_name(int(str(device).split(",")[0]))),
        "timing_boundary": "cv2 decode + Ultralytics preprocess + model + NMS; includes one image call",
    }

