"""Compare raw FP32 detection outputs to catch export/preprocessing mismatches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def verify_export(weights: Path, onnx: Path, manifest: Path, *, imgsz: int = 640,
                  limit: int = 8, atol: float = 1e-3, rtol: float = 1e-3) -> dict:
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    if limit < 1:
        raise ValueError("Compare at least one image")
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    if model_input.type != "tensor(float)":
        raise ValueError("This agreement check currently supports FP32 ONNX exports")
    model = YOLO(str(weights), task="detect").model.cpu().float().eval()
    model.fuse(verbose=False)
    paths = [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()][:limit]
    if not paths:
        raise ValueError("No images in the manifest")
    errors = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {path}")
        image = LetterBox(new_shape=(imgsz, imgsz), auto=False)(image=image)
        tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        with torch.inference_mode():
            reference = model(torch.from_numpy(tensor))
        if isinstance(reference, tuple):
            reference = reference[0]
        reference = reference.numpy()
        exported = session.run(None, {model_input.name: tensor})[0]
        if reference.shape != exported.shape:
            raise AssertionError(f"Exported shape {exported.shape} differs from {reference.shape}")
        if not np.allclose(reference, exported, rtol=rtol, atol=atol):
            raise AssertionError(f"Export disagreement for {path}; max abs error "
                                 f"{np.max(np.abs(reference - exported))}")
        errors.append(float(np.max(np.abs(reference - exported))))
    return {"status": "passed", "images": len(paths), "imgsz": imgsz, "precision": "FP32",
            "atol": atol, "rtol": rtol, "max_absolute_error": max(errors),
            "weight_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            "output": "raw boxes and class scores before NMS", "onnxruntime": ort.__version__}
