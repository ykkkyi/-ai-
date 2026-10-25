"""Conservative visual grouping before a deterministic, non-destructive split.

pHash matches are candidates, not proof that two photographs show the same plant.
We co-locate candidates instead of silently deleting their images or annotations.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .data import SPLIT_PRIORITY, SPLITS, _label_path


def perceptual_hashes(path: Path) -> list[int]:
    """Return 64-bit DCT hashes for all eight rotations/reflections."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    hashes = []
    for mirrored in (small, np.fliplr(small)):
        for rotation in range(4):
            coefficients = cv2.dct(np.ascontiguousarray(np.rot90(mirrored, rotation)))[:8, :8]
            ac = coefficients.ravel()[1:]
            bits = np.r_[False, ac > np.median(ac)]
            hashes.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    return hashes


def visual_components(hashes: np.ndarray, distance: int) -> tuple[list[list[int]], list[tuple]]:
    """Union candidate edges, including transitive matches across rotations/flips."""
    if not 0 <= distance <= 16:
        raise ValueError("pHash distance must be between 0 and 16")
    parents = list(range(len(hashes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # Byte lookup keeps compatibility with NumPy 1.26, which lacks bitwise_count.
    popcounts = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
    edges = []
    for i in range(len(hashes) - 1):
        xor = np.ascontiguousarray(hashes[i + 1:] ^ hashes[i, 0])
        counts = popcounts[xor.view(np.uint8)].reshape(xor.shape + (8,)).sum(axis=-1)
        best = counts.min(axis=1)
        for offset in np.flatnonzero(best <= distance):
            j = i + 1 + int(offset)
            parents[find(j)] = find(i)
            edges.append((i, j, int(best[offset])))
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(hashes)):
        groups[find(i)].append(i)
    return list(groups.values()), edges


def assign_groups(groups: list[list[int]], records: list[dict], class_count: int,
                  seed: int, fractions: tuple[float, float, float]) -> dict[int, str]:
    """Greedily balance image and multilabel class-image counts, never splitting groups."""
    if min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Split fractions must be positive and sum to one")
    if len(groups) < 3:
        raise ValueError("At least three visual groups are required")
    features = np.zeros((len(groups), class_count + 1), dtype=np.float64)
    for i, group in enumerate(groups):
        features[i, 0] = len(group)
        for index in group:
            for cls in records[index]["classes"]:
                features[i, cls + 1] += 1
    targets = np.outer(fractions, features.sum(axis=0))
    allocated = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    tie_breaks = rng.random(len(groups))
    order = sorted(range(len(groups)), key=lambda i: (-len(groups[i]), tie_breaks[i]))
    assignment = {}
    for index in order:
        before = (allocated - targets) ** 2 / (targets + 1)
        after = (allocated + features[index] - targets) ** 2 / (targets + 1)
        costs = (after - before).sum(axis=1)
        split_index = int(np.argmin(costs))
        allocated[split_index] += features[index]
        for member in groups[index]:
            assignment[member] = SPLITS[split_index]
    if set(assignment.values()) != set(SPLITS):
        raise ValueError("Grouping produced an empty split; use more independent source images")
    return assignment


def write_grouped_split(audit: dict[str, Any], output: Path, *, distance: int = 6,
                        seed: int = 42, workers: int = 4,
                        fractions: tuple[float, float, float] = (0.7, 0.2, 0.1)) -> dict:
    """Keep exact files once, co-locate near candidates, write manifests and provenance."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new split version instead of overwriting frozen manifests: {output}")
    if audit["image_issues"] or audit["duplicate_annotation_disagreements"]:
        raise ValueError("Resolve invalid images/labels and exact-duplicate label conflicts first")
    if not 0 <= distance <= 16 or workers < 1:
        raise ValueError("Use pHash distance 0–16 and at least one worker")
    # Stable representative selection independent of filesystem enumeration order.
    exact: dict[str, list[dict]] = defaultdict(list)
    for item in audit["_records"]:
        exact[item["sha256"]].append(item)
    records = sorted((min(items, key=lambda r: (SPLIT_PRIORITY[r["split"]], r["image"]))
                      for items in exact.values()), key=lambda r: r["image"])
    root = Path(audit["dataset"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = np.array(list(pool.map(perceptual_hashes,
                                        (root / r["image"] for r in records))), dtype=np.uint64)
    groups, edges = visual_components(hashes, distance)
    names = audit["names_as_provided"]
    assignments = assign_groups(groups, records, len(names), seed, fractions)
    group_ids = {}
    for group in groups:
        group_id = hashlib.sha256("\n".join(sorted(records[i]["sha256"] for i in group))
                                  .encode()).hexdigest()[:20]
        for i in group:
            group_ids[i] = group_id
    output.mkdir(parents=True, exist_ok=True)
    provenance = []
    for i, record in enumerate(records):
        provenance.append({"split": assignments[i], "source_split": record["split"],
                           "image": record["image"], "sha256": record["sha256"],
                           "label_sha256": hashlib.sha256(
                               _label_path(root / record["image"]).read_bytes()).hexdigest(),
                           "group_id": group_ids[i]})
    manifest_text = json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(manifest_text.encode()).hexdigest()
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)
    stats = {}
    for split in SPLITS:
        selected = [r for i, r in enumerate(records) if assignments[i] == split]
        (output / f"{split}.txt").write_text(
            "\n".join(str(root / r["image"]) for r in selected) + "\n", encoding="utf-8")
        boxes = Counter(int(box[0]) for r in selected for box in r["boxes"])
        images = Counter(cls for r in selected for cls in r["classes"])
        stats[split] = {"images": len(selected), "boxes": sum(boxes.values()),
                        "groups": len({group_ids[i] for i in assignments if assignments[i] == split}),
                        "class_box_counts": {str(i): boxes[i] for i in range(len(names))},
                        "class_image_counts": {str(i): images[i] for i in range(len(names))}}
    report = {"schema_version": 2, "seed": seed, "target_fractions": fractions,
              "names_as_provided": names, "original_images": len(audit["_records"]),
              "unique_sha256_images": len(records), "visual_candidate_groups": len(groups),
              "multi_image_groups": sum(len(g) > 1 for g in groups),
              "largest_group": max(map(len, groups)), "candidate_pairs": len(edges),
              "pairs_crossing_previous_splits": sum(records[i]["split"] != records[j]["split"]
                                                    for i, j, _ in edges),
              "phash_distance": distance, "transforms": "four 90-degree rotations and reflection",
              "split_manifest_sha256": fingerprint, "cross_split_candidate_groups": 0,
              "splits": stats,
              "policy": "One copy per SHA-256; union DCT pHash candidates; assign whole connected "
                        "components to new seeded, class-balanced train/valid/test splits.",
              "limitation": "Perceptual candidates are conservatively co-located, not deleted. "
                            "pHash can miss crops, perspective changes and same-plant photos. "
                            "This is not proof of plant/session independence or field generalization. "
                            "The original split is replaced; restart models from pretrained weights."}
    (output / "split-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    with (output / "near-pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_a", "image_b", "distance", "previous_split_a", "previous_split_b"])
        for i, j, d in edges:
            writer.writerow([records[i]["image"], records[j]["image"], d,
                             records[i]["split"], records[j]["split"]])
    config = {"path": str(output.resolve()), "train": "train.txt", "val": "valid.txt",
              "test": "test.txt", "nc": len(names), "names": names}
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                                      encoding="utf-8")
    return report
