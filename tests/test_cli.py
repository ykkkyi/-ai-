import pytest

from cotton_doctor.cli import _training_hyperparameters, build_parser


def test_training_profiles_keep_baseline_and_change_only_robust_knobs():
    baseline = _training_hyperparameters("baseline")
    robust = _training_hyperparameters("robust")

    assert robust["hsv_s"] > baseline["hsv_s"]
    assert robust["hsv_v"] > baseline["hsv_v"]
    assert {key for key in baseline if baseline[key] != robust[key]} == {"hsv_s", "hsv_v"}


def test_train_parser_exposes_profile():
    args = build_parser().parse_args(["train", "--profile", "robust"])
    assert args.profile == "robust"


def test_unknown_training_profile_is_rejected():
    with pytest.raises(ValueError):
        _training_hyperparameters("unknown")
