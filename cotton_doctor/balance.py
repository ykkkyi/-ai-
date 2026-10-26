"""Create deterministic, non-destructive class-balanced training manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def _label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _classes(image: Path) -> set[int]:
    label = _label_path(image)
    if not label.is_file():
        raise FileNotFoundError(f"Missing label for {image}: {label}")
    classes = set()
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            classes.add(int(float(line.split()[0])))
    return classes


def make_balanced_train(data_yaml: Path, output: Path, *, target_class: int = 4,
                        factor: float = 2.0, seed: int = 42) -> dict:
    """Oversample images containing one class while retaining the original paths and split."""
    if factor < 1.0:
        raise ValueError("factor must be at least 1.0")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    train_manifest = root / config["train"]
    source_paths = [Path(line.strip()) for line in train_manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    if not source_paths:
        raise ValueError("The training manifest is empty")
    target_paths = [path for path in source_paths if target_class in _classes(path)]
    if not target_paths:
        raise ValueError(f"No training image contains class {target_class}")

    # The extra entries are deterministic and do not alter validation/test membership.
    extra_count = round((factor - 1.0) * len(target_paths))
    selected = list(source_paths)
    selected.extend(target_paths[index % len(target_paths)] for index in range(extra_count))
    # Stable permutation avoids putting all repeated images in a single tail of an epoch.
    selected.sort(key=lambda path: hashlib.sha256(f"{seed}:{path}".encode()).hexdigest())

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.txt"
    train_path.write_text("\n".join(str(path) for path in selected) + "\n", encoding="utf-8")
    new_config = dict(config)
    new_config["path"] = str(root.resolve())
    new_config["train"] = str(train_path.resolve())
    (output / "data.yaml").write_text(
        yaml.safe_dump(new_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    summary = {
        "source_data": str(data_yaml.resolve()), "source_train_manifest": str(train_manifest.resolve()),
        "output": str(output.resolve()), "seed": seed, "target_class": target_class,
        "factor": factor, "source_images": len(source_paths), "target_images": len(target_paths),
        "extra_samples": extra_count, "balanced_images_per_epoch": len(selected),
        "validation_manifest": str((root / config["val"]).resolve()),
        "test_manifest": str((root / config["test"]).resolve()),
        "policy": "Original training paths retained; only target-class image entries are repeated. "
                  "Validation and test manifests are unchanged.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
