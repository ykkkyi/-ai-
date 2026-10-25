import csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from cotton_doctor.data import audit_dataset
from cotton_doctor.grouping import perceptual_hashes, visual_components, write_grouped_split


def test_visual_graph_keeps_transitive_matches_together():
    hashes = np.array([[0], [1], [3], [255]], dtype=np.uint64)
    groups, _ = visual_components(hashes, distance=1)
    assert sorted(groups) == [[0, 1, 2], [3]]


def test_phash_recognizes_rotation_and_jpeg_reencoding(tmp_path: Path):
    rng = np.random.default_rng(42)
    image = cv2.resize(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8), (128, 128))
    original = tmp_path / "source.png"
    rotated = tmp_path / "rotated.jpg"
    cv2.imwrite(str(original), image)
    cv2.imwrite(str(rotated), cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    hashes = np.array([perceptual_hashes(original), perceptual_hashes(rotated)], dtype=np.uint64)
    groups, _ = visual_components(hashes, distance=6)
    assert groups == [[0, 1]]


def test_grouped_split_is_repeatable_and_never_splits_visual_groups(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 2, "names": ["a", "b"]}))
    rng = np.random.default_rng(17)
    before = {}
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for i in range(12):
            path = root / split / "images" / f"{i}.png"
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            label = root / split / "labels" / f"{i}.txt"
            label.write_text(f"{i % 2} 0.5 0.5 0.5 0.5\n")
            before[path] = path.read_bytes()
            before[label] = label.read_bytes()
    source = root / "train/images/0.png"
    rotated = root / "valid/images/0.png"
    cv2.imwrite(str(rotated), cv2.rotate(cv2.imread(str(source)), cv2.ROTATE_90_CLOCKWISE))
    before[rotated] = rotated.read_bytes()
    audit = audit_dataset(root)
    first = write_grouped_split(audit, tmp_path / "first", workers=1)
    second = write_grouped_split(audit, tmp_path / "second", workers=1)
    assert first["split_manifest_sha256"] == second["split_manifest_sha256"]
    assert first["cross_split_candidate_groups"] == 0
    with (tmp_path / "first/manifest.csv").open() as stream:
        rows = {row["image"]: row for row in csv.DictReader(stream)}
    assert rows["train/images/0.png"]["group_id"] == rows["valid/images/0.png"]["group_id"]
    assert rows["train/images/0.png"]["split"] == rows["valid/images/0.png"]["split"]
    assert all(stats["images"] for stats in first["splits"].values())
    assert all(path.read_bytes() == content for path, content in before.items())
