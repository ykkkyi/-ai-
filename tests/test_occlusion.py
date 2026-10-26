from pathlib import Path

import cv2
import numpy as np
import yaml

from cotton_doctor.occlusion import make_occlusion_train


def test_occlusion_training_copies_preserve_source_and_eval_manifests(tmp_path: Path):
    root = tmp_path / "source"
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    source = image_dir / "sample.png"
    original = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    cv2.imwrite(str(source), original)
    (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    valid = root / "valid.txt"
    test = root / "test.txt"
    valid.write_text(f"{source}\n", encoding="utf-8")
    test.write_text(f"{source}\n", encoding="utf-8")
    train = root / "train.txt"
    train.write_text(f"{source}\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "train": "train.txt", "val": "valid.txt",
                                    "test": "test.txt", "nc": 1, "names": ["leaf"]}), encoding="utf-8")

    summary = make_occlusion_train(data, tmp_path / "occluded", fraction=1.0, seed=42)

    assert summary["source_images"] == 1
    assert summary["derived_images"] == 1
    assert summary["train_images_per_epoch"] == 2
    assert np.array_equal(cv2.imread(str(source)), original)
    derived = cv2.imread(str(tmp_path / "occluded/images/occluded_000000.png"))
    assert not np.array_equal(derived, original)
    assert (tmp_path / "occluded/labels/occluded_000000.txt").read_text() == (
        label_dir / "sample.txt").read_text()
