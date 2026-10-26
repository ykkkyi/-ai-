from pathlib import Path

import yaml

from cotton_doctor.balance import make_balanced_train


def _write_sample(root: Path, split: str, name: str, class_id: int) -> Path:
    image = root / split / "images" / f"{name}.jpg"
    label = root / split / "labels" / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    label.write_text(f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return image


def test_balanced_manifest_only_repeats_target_class_and_keeps_eval_manifests(tmp_path: Path):
    root = tmp_path / "source"
    train = [_write_sample(root, "train", "target", 4), _write_sample(root, "train", "other", 0)]
    valid = _write_sample(root, "valid", "valid", 4)
    test = _write_sample(root, "test", "test", 4)
    (root / "train.txt").write_text("\n".join(map(str, train)) + "\n", encoding="utf-8")
    (root / "valid.txt").write_text(f"{valid}\n", encoding="utf-8")
    (root / "test.txt").write_text(f"{test}\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "train.txt", "val": "valid.txt",
                                    "test": "test.txt", "nc": 5, "names": ["a"] * 5}), encoding="utf-8")

    summary = make_balanced_train(data, tmp_path / "balanced", target_class=4, factor=2.0)

    assert summary["source_images"] == 2
    assert summary["target_images"] == 1
    assert summary["extra_samples"] == 1
    assert summary["balanced_images_per_epoch"] == 3
    lines = (tmp_path / "balanced/train.txt").read_text().splitlines()
    assert lines.count(str(train[0])) == 2
    assert lines.count(str(train[1])) == 1
    config = yaml.safe_load((tmp_path / "balanced/data.yaml").read_text())
    assert config["val"] == "valid.txt"
    assert config["test"] == "test.txt"
