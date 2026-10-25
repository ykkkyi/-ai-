from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cotton_doctor.stress import make_stress_dataset


def test_stress_preserves_boxes_and_keeps_originals_intact(tmp_path: Path):
    images = tmp_path / "original" / "images"
    labels = tmp_path / "original" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    source = images / "sample.png"
    cv2.imwrite(str(source), np.full((64, 64, 3), 100, dtype=np.uint8))
    label = labels / "sample.txt"
    label.write_text("0 0.5 0.5 0.6 0.6\n")
    original = source.read_bytes()
    (tmp_path / "valid.txt").write_text(str(source) + "\n")
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "val": "valid.txt", "nc": 1,
                                    "names": ["leaf"]}))
    result = make_stress_dataset(data, tmp_path / "stress", variants=("dark", "occlusion"))
    assert result["images"] == 1
    assert source.read_bytes() == original
    for variant in result["variants"]:
        assert (tmp_path / "stress" / variant / "labels" / "000000.txt").read_bytes() == label.read_bytes()
    dark = cv2.imread(str(tmp_path / "stress/dark/images/000000.png"))
    assert dark.mean() == 45
    with pytest.raises(FileExistsError):
        make_stress_dataset(data, tmp_path / "stress")
