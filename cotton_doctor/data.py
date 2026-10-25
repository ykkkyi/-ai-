"""Read-only dataset checks and non-destructive exact-duplicate manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}


def load_class_names(dataset: Path) -> list[str]:
    data_file = dataset / "data.yaml"
    config = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError(f"Expected a non-empty class-name list in {data_file}")
    if config.get("nc", len(names)) != len(names):
        raise ValueError("The nc value does not match the number of class names")
    return names


def _label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _read_labels(path: Path, class_count: int) -> tuple[list[tuple[float, ...]], list[str]]:
    boxes: list[tuple[float, ...]] = []
    issues: list[str] = []
    if not path.exists():
        return boxes, ["missing_label"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError("expected five finite values")
            class_float, x, y, width, height = values
            class_id = int(class_float)
            if class_float != class_id or not 0 <= class_id < class_count:
                raise ValueError("invalid class index")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("coordinates out of range")
            if x - width / 2 < -1e-5 or y - height / 2 < -1e-5:
                raise ValueError("box begins outside image")
            if x + width / 2 > 1.00001 or y + height / 2 > 1.00001:
                raise ValueError("box ends outside image")
            boxes.append((float(class_id), x, y, width, height))
        except (ValueError, OverflowError) as error:
            issues.append(f"line {line_number}: {error}")
    return boxes, issues


def audit_dataset(dataset: Path) -> dict[str, Any]:
    """Inspect image readability, YOLO labels, split counts, and byte-identical files."""
    root = dataset.expanduser().resolve()
    names = load_class_names(root)
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    split_stats: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    for split in SPLITS:
        image_root = root / split / "images"
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_root}")
        images = sorted(path for path in image_root.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        class_boxes: Counter[int] = Counter()
        class_images: Counter[int] = Counter()
        dimensions: Counter[str] = Counter()
        empty_images = 0
        known_stems = {image.stem for image in images}
        orphan_labels = sorted(path.name for path in (root / split / "labels").glob("*.txt")
                               if path.stem not in known_stems)

        for image in images:
            record_issues: list[str] = []
            try:
                with Image.open(image) as decoded:
                    decoded.verify()
                    width, height = decoded.size
                dimensions[f"{width}x{height}"] += 1
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                width = height = 0
                record_issues.append(f"image decode: {error}")
            boxes, label_issues = _read_labels(_label_path(image), len(names))
            record_issues.extend(label_issues)
            classes = sorted({int(box[0]) for box in boxes})
            empty_images += not boxes
            for box in boxes:
                class_boxes[int(box[0])] += 1
            class_images.update(classes)
            try:
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
            except OSError as error:
                digest = ""
                record_issues.append(f"image read: {error}")
            record = {
                "split": split,
                "image": str(image.relative_to(root)),
                "sha256": digest,
                "width": width,
                "height": height,
                "boxes": boxes,
                "classes": classes,
                "issues": record_issues,
            }
            records.append(record)
            if digest:
                hashes[digest].append(record)
            if record_issues:
                issues.append({"image": record["image"], "issues": record_issues})
        split_stats[split] = {
            "images": len(images),
            "boxes": sum(class_boxes.values()),
            "empty_images": empty_images,
            "class_box_counts": {str(index): class_boxes[index] for index in range(len(names))},
            "class_image_counts": {str(index): class_images[index] for index in range(len(names))},
            "dimensions": dict(dimensions),
            "orphan_labels": orphan_labels,
        }

    duplicate_groups = []
    for digest, group in hashes.items():
        if len(group) < 2:
            continue
        annotations = {
            tuple(sorted(tuple(round(value, 6) for value in box) for box in record["boxes"]))
            for record in group
        }
        duplicate_groups.append({
            "sha256": digest,
            "cross_split": len({record["split"] for record in group}) > 1,
            "annotation_disagreement": len(annotations) > 1,
            "images": [{"split": record["split"], "image": record["image"]} for record in group],
        })

    return {
        "dataset": str(root),
        "names_as_provided": names,
        "splits": split_stats,
        "total_images": len(records),
        "total_valid_boxes": sum(stats["boxes"] for stats in split_stats.values()),
        "image_issues": issues,
        "exact_duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": sum(group["cross_split"] for group in duplicate_groups),
        "duplicate_annotation_disagreements": sum(
            group["annotation_disagreement"] for group in duplicate_groups
        ),
        "scope": "Image decoding, YOLO label validity, split statistics, and exact SHA-256 duplicates. "
                 "No near-duplicate detection or agricultural diagnosis.",
        "_records": records,
    }


def write_audit(audit: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    records = audit["_records"]
    public_audit = {key: value for key, value in audit.items() if key != "_records"}
    (output / "dataset-audit.json").write_text(
        json.dumps(public_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "image-manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "image", "sha256", "width", "height",
                                                    "box_count", "classes", "issues"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "split": record["split"], "image": record["image"], "sha256": record["sha256"],
                "width": record["width"], "height": record["height"],
                "box_count": len(record["boxes"]), "classes": json.dumps(record["classes"]),
                "issues": json.dumps(record["issues"], ensure_ascii=False),
            })


def write_exact_split(audit: dict[str, Any], output: Path) -> dict[str, Any]:
    """Write image lists, retaining each exact file once and preferring test > val > train."""
    if audit["image_issues"]:
        raise ValueError("Resolve image or label issues before preparing training lists")
    if audit["duplicate_annotation_disagreements"]:
        raise ValueError("Resolve duplicate-label conflicts before preparing training lists")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audit["_records"]:
        groups[record["sha256"]].append(record)

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for digest, group in sorted(groups.items()):
        keep = min(group, key=lambda item: (SPLIT_PRIORITY[item["split"]], item["image"]))
        selected.append(keep)
        excluded.extend({"image": item["image"], "original_split": item["split"],
                         "kept_image": keep["image"], "kept_split": keep["split"],
                         "sha256": digest} for item in group if item is not keep)

    output.mkdir(parents=True, exist_ok=True)
    root = Path(audit["dataset"])
    split_hashes: dict[str, set[str]] = {}
    stats: dict[str, Any] = {}
    class_names = audit["names_as_provided"]
    for split in SPLITS:
        rows = sorted((item for item in selected if item["split"] == split),
                      key=lambda item: item["image"])
        (output / f"{split}.txt").write_text(
            "\n".join(str(root / item["image"]) for item in rows) + "\n", encoding="utf-8"
        )
        box_counts: Counter[int] = Counter(int(box[0]) for item in rows for box in item["boxes"])
        image_counts: Counter[int] = Counter(cls for item in rows for cls in item["classes"])
        stats[split] = {
            "images": len(rows), "boxes": sum(box_counts.values()),
            "empty_images": sum(not item["boxes"] for item in rows),
            "class_box_counts": {str(i): box_counts[i] for i in range(len(class_names))},
            "class_image_counts": {str(i): image_counts[i] for i in range(len(class_names))},
        }
        split_hashes[split] = {item["sha256"] for item in rows}
    if any(split_hashes[a] & split_hashes[b]
           for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]):
        raise AssertionError("Exact duplicate hashes still cross the generated splits")

    report = {
        "policy": "one image per SHA-256; preserve test > valid > train; lexical tie break",
        "source_dataset": audit["dataset"],
        "original_images": len(audit["_records"]),
        "unique_images": len(selected),
        "excluded_copies": len(excluded),
        "cross_split_sha256_intersection": 0,
        "splits": stats,
        "limitation": "Exact-byte duplicate cleanup only. Group near-duplicates, plants, scenes, and capture "
                      "sessions before treating the final test set as field-independent.",
    }
    (output / "split-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "excluded-copies.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "original_split", "kept_image",
                                                    "kept_split", "sha256"])
        writer.writeheader()
        writer.writerows(excluded)
    config = {
        "path": str(output.resolve()), "train": "train.txt", "val": "valid.txt",
        "test": "test.txt", "nc": len(class_names), "names": class_names,
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return report
