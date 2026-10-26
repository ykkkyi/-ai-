"""Build a deterministic derived training set with target-aware partial occlusions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def _label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    result = []
    for line in _label_path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            values = line.split()
            result.append((int(float(values[0])), *(float(value) for value in values[1:5])))
    return result


def make_occlusion_train(data_yaml: Path, output: Path, *, fraction: float = 0.25,
                         seed: int = 42, rectangle_fraction: float = 0.18) -> dict:
    """Append target-overlapping occluded copies to the original training manifest."""
    if not 0 < fraction <= 1 or not 0 < rectangle_fraction <= 1:
        raise ValueError("fraction and rectangle_fraction must be in (0, 1]")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    source_manifest = root / config["train"]
    source_paths = [Path(line.strip()) for line in source_manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    if not source_paths:
        raise ValueError("The training manifest is empty")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    derived_images = output / "images"
    derived_labels = output / "labels"
    derived_images.mkdir(parents=True, exist_ok=True)
    derived_labels.mkdir(parents=True, exist_ok=True)
    selected = []
    for index, source in enumerate(source_paths):
        if index / len(source_paths) >= fraction:
            break
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {source}")
        boxes = _boxes(source)
        if not boxes:
            continue
        # Largest target gives a stable, useful occlusion when an image has mixed labels.
        _, cx, cy, _, _ = max(boxes, key=lambda box: box[3] * box[4])
        image_height, image_width = image.shape[:2]
        rng = np.random.default_rng(seed + index)
        rect_width = max(1, int(image_width * rectangle_fraction))
        rect_height = max(1, int(image_height * rectangle_fraction))
        center_x = int(cx * image_width + rng.integers(-rect_width // 4, rect_width // 4 + 1))
        center_y = int(cy * image_height + rng.integers(-rect_height // 4, rect_height // 4 + 1))
        x1 = max(0, min(image_width - rect_width, center_x - rect_width // 2))
        y1 = max(0, min(image_height - rect_height, center_y - rect_height // 2))
        # Use the local image median as a neutral textured-field occluder.
        patch_color = np.median(image[max(0, y1 - 8):min(image_height, y1 + rect_height + 8),
                                      max(0, x1 - 8):min(image_width, x1 + rect_width + 8)], axis=(0, 1))
        image[y1:y1 + rect_height, x1:x1 + rect_width] = patch_color.astype(np.uint8)
        destination = derived_images / f"occluded_{index:06d}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Failed to write {destination}")
        shutil.copy2(_label_path(source), derived_labels / f"{destination.stem}.txt")
        selected.append({"source": str(source), "derived": str(destination),
                         "rectangle": [x1, y1, rect_width, rect_height]})

    if not selected:
        raise ValueError("No labeled training images were selected")
    train_paths = source_paths + [Path(row["derived"]) for row in selected]
    train_paths.sort(key=lambda path: hashlib.sha256(f"{seed}:{path}".encode()).hexdigest())
    train_manifest = output / "train.txt"
    train_manifest.write_text("\n".join(str(path) for path in train_paths) + "\n", encoding="utf-8")
    new_config = dict(config)
    new_config["path"] = str(root.resolve())
    new_config["train"] = str(train_manifest.resolve())
    (output / "data.yaml").write_text(yaml.safe_dump(new_config, sort_keys=False, allow_unicode=True),
                                       encoding="utf-8")
    summary = {
        "source_data": str(data_yaml.resolve()), "source_train_manifest": str(source_manifest.resolve()),
        "output": str(output.resolve()), "seed": seed, "fraction": fraction,
        "rectangle_fraction": rectangle_fraction, "source_images": len(source_paths),
        "derived_images": len(selected), "train_images_per_epoch": len(train_paths),
        "validation_manifest": str((root / config["val"]).resolve()),
        "test_manifest": str((root / config["test"]).resolve()),
        "policy": "Original training paths retained; derived copies occlude a rectangle near the largest box. "
                  "Validation and test manifests are unchanged.",
        "examples": selected[:20],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
