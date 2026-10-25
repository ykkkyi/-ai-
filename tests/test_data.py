from pathlib import Path

import yaml
from PIL import Image

from cotton_doctor.data import audit_dataset, write_exact_split


def _add_image(root: Path, split: str, name: str, label: str, color: tuple[int, int, int]) -> Path:
    image = root / split / "images" / f"{name}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    labels = root / split / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(image)
    (labels / f"{name}.txt").write_text(label, encoding="utf-8")
    return image


def test_audit_finds_cross_split_exact_duplicates_and_validates_yolo_labels(tmp_path: Path):
    names = ["leaf", "aphid"]
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"nc": 2, "names": names}), encoding="utf-8")
    duplicate = _add_image(tmp_path, "train", "same-train", "0 0.5 0.5 0.5 0.5\n", (1, 2, 3))
    _add_image(tmp_path, "valid", "same-valid", "0 0.5 0.5 0.5 0.5\n", (1, 2, 3))
    _add_image(tmp_path, "test", "distinct", "1 0.5 0.5 0.25 0.25\n", (8, 9, 10))

    report = audit_dataset(tmp_path)

    assert report["total_images"] == 3
    assert report["total_valid_boxes"] == 3
    assert report["cross_split_duplicate_groups"] == 1
    assert report["duplicate_annotation_disagreements"] == 0
    assert duplicate.is_file()


def test_prepare_keeps_test_duplicate_and_does_not_modify_source(tmp_path: Path):
    names = ["leaf", "aphid"]
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"nc": 2, "names": names}), encoding="utf-8")
    images = {
        split: _add_image(tmp_path, split, f"same-{split}", "0 0.5 0.5 0.5 0.5\n", (11, 22, 33))
        for split in ("train", "valid", "test")
    }
    before = {path: path.read_bytes() for path in images.values()}

    report = write_exact_split(audit_dataset(tmp_path), tmp_path / "clean-split")

    assert report["unique_images"] == 1
    assert report["splits"]["test"]["images"] == 1
    assert report["splits"]["train"]["images"] == 0
    assert (tmp_path / "clean-split/test.txt").read_text().strip().endswith(images["test"].name)
    assert all(path.read_bytes() == value for path, value in before.items())
    config = yaml.safe_load((tmp_path / "clean-split/data.yaml").read_text())
    assert config["names"] == names


def test_audit_records_invalid_labels(tmp_path: Path):
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"nc": 1, "names": ["leaf"]}), encoding="utf-8")
    _add_image(tmp_path, "train", "bad", "2 0.5 0.5 0.5 0.5\n", (1, 2, 3))
    _add_image(tmp_path, "valid", "empty", "", (1, 2, 3))
    _add_image(tmp_path, "test", "empty", "", (1, 2, 3))

    report = audit_dataset(tmp_path)

    assert len(report["image_issues"]) == 1
    assert "invalid class index" in report["image_issues"][0]["issues"][0]
