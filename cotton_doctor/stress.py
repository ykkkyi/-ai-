"""Build controlled image-stress subsets with unchanged YOLO boxes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def _variant(image: np.ndarray, name: str, index: int) -> np.ndarray:
    if name == "dark":
        return np.clip(image.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if name == "bright":
        return np.clip(image.astype(np.float32) * 1.45, 0, 255).astype(np.uint8)
    if name == "low_contrast":
        return np.clip(image.astype(np.float32) * 0.55 + 70, 0, 255).astype(np.uint8)
    if name == "blur":
        return cv2.GaussianBlur(image, (0, 0), 1.8)
    if name == "scale_050":
        height, width = image.shape[:2]
        small = cv2.resize(image, (max(1, width // 2), max(1, height // 2)),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if name == "occlusion":
        result = image.copy()
        height, width = result.shape[:2]
        rng = np.random.default_rng(1000 + index)
        box_width, box_height = int(width * 0.18), int(height * 0.18)
        x = int(rng.integers(0, max(1, width - box_width)))
        y = int(rng.integers(0, max(1, height - box_height)))
        result[y:y + box_height, x:x + box_width] = 0
        return result
    raise ValueError(f"Unknown stress variant: {name}")


def make_stress_dataset(data_yaml: Path, output: Path, *, split: str = "val",
                        variants: tuple[str, ...] = ("dark", "bright", "low_contrast", "occlusion",
                                                     "blur", "scale_050"),
                        limit: int | None = None) -> dict:
    if split not in {"val", "test"} or not variants:
        raise ValueError("Choose val or test and at least one variant")
    if limit is not None and limit < 1:
        raise ValueError("Image limit must be positive")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    manifest = root / config[split]
    paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError("The source manifest is empty")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory to avoid stale stress images: {output}")
    output.mkdir(parents=True, exist_ok=True)
    configs = {}
    for variant in variants:
        images = output / variant / "images"
        labels = output / variant / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(paths):
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read {source}")
            destination = images / f"{index:06d}.png"
            if not cv2.imwrite(str(destination), _variant(image, variant, index)):
                raise OSError(f"Failed to write {destination}")
            shutil.copy2(source.parent.parent / "labels" / f"{source.stem}.txt",
                         labels / f"{destination.stem}.txt")
        stress_config = {"path": str((output / variant).resolve()),
                         "train": "images", "val": "images", "test": "images",
                         "nc": config["nc"], "names": config["names"]}
        variant_config = output / variant / "data.yaml"
        variant_config.write_text(yaml.safe_dump(stress_config, sort_keys=False, allow_unicode=True),
                                   encoding="utf-8")
        configs[variant] = str(variant_config.resolve())
    (output / "source.json").write_text(
        json.dumps({"source_data": str(data_yaml.resolve()), "source_split": split,
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "images": len(paths), "variants": variants, "seed": 1000,
                    "settings": {"dark_gain": 0.45, "bright_gain": 1.45,
                                 "low_contrast_gain": 0.55, "low_contrast_offset": 70,
                                 "occlusion_rect_width_height_fraction": 0.18,
                                 "blur_sigma": 1.8, "resolution_fraction": 0.5},
                    "interpretation": "Synthetic stress on original full boxes. Occlusion can cover "
                                      "targets; it is not a real field occlusion annotation. "
                                      "scale_050 tests resolution loss, not geometric box-scale change. "
                                      "Measure box scales separately and validate real field conditions."},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return {"path": str(output.resolve()), "images": len(paths), "variants": variants,
            "configs": configs}
