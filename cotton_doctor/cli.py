"""Command-line entry points for dataset preparation, training, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from .data import audit_dataset, write_audit, write_exact_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cotton-doctor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Check image files, YOLO labels, and exact duplicates")
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--output", type=Path, default=Path("data/processed/audit"))

    prepare = subparsers.add_parser("prepare", help="Create a non-destructive exact-deduplicated split")
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("--output", type=Path, default=Path("data/splits/exact"))

    grouped = subparsers.add_parser("prepare-grouped", help="Group visual near-duplicates before splitting")
    grouped.add_argument("dataset", type=Path)
    grouped.add_argument("--output", type=Path, default=Path("data/splits/grouped-v2"))
    grouped.add_argument("--distance", type=int, default=6)
    grouped.add_argument("--seed", type=int, default=42)
    grouped.add_argument("--workers", type=int, default=4)

    balanced = subparsers.add_parser("balance-train", help="Create a deterministic class-balanced train manifest")
    balanced.add_argument("--data", type=Path, default=Path("data/splits/grouped-v2/data.yaml"))
    balanced.add_argument("--output", type=Path, default=Path("data/splits/grouped-v2-balanced-fusarium"))
    balanced.add_argument("--target-class", type=int, default=4)
    balanced.add_argument("--factor", type=float, default=2.0)
    balanced.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="Train a YOLO detection model")
    train.add_argument("--data", type=Path, default=Path("configs/data.local.yaml"))
    train.add_argument("--model", default="yolov8n.pt")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--profile", choices=("baseline", "robust"), default="baseline",
                       help="Training augmentation profile; robust targets illumination and occlusion")
    train.add_argument("--name", default="E0_yolov8n_seed42")
    train.add_argument("--project", type=Path, default=Path("runs"))
    train.add_argument("--resume", type=Path, help="Resume this last.pt checkpoint with its saved settings")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a frozen model on val or test")
    evaluate.add_argument("model", type=Path)
    evaluate.add_argument("--data", type=Path, default=Path("configs/data.local.yaml"))
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=1)
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--project", type=Path, default=Path("runs/evaluation"))
    evaluate.add_argument("--name", default="frozen_validation")
    evaluate.add_argument("--allow-test", action="store_true",
                          help="Required to explicitly consume the held-out final test split")

    export = subparsers.add_parser("export", help="Export a trained model to ONNX")
    export.add_argument("model", type=Path)
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--half", action="store_true")
    export.add_argument("--opset", type=int, default=17)

    benchmark = subparsers.add_parser("benchmark", help="Measure end-to-end inference latency and model size")
    benchmark.add_argument("model", type=Path)
    benchmark.add_argument("--manifest", type=Path, default=Path("data/splits/grouped-v2/valid.txt"))
    benchmark.add_argument("--imgsz", type=int, default=640)
    benchmark.add_argument("--device", default="0")
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--limit", type=int, default=100)
    benchmark.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))

    stress = subparsers.add_parser("make-stress", help="Create illumination/occlusion stress subsets")
    stress.add_argument("--data", type=Path, default=Path("data/splits/grouped-v2/data.yaml"))
    stress.add_argument("--output", type=Path, default=Path("data/stress/val"))
    stress.add_argument("--split", choices=("val", "test"), default="val")
    stress.add_argument("--limit", type=int, default=None)

    predict = subparsers.add_parser("predict", help="Detect images, a video, or a USB camera")
    predict.add_argument("model", type=Path)
    predict.add_argument("--source", required=True, help="Image/folder/video path or camera index 0")
    predict.add_argument("--output", type=Path, default=Path("runs/predict/local"))
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.45)
    predict.add_argument("--max-frames", type=int, default=None)

    check = subparsers.add_parser("verify-export", help="Compare FP32 PyTorch/ONNX raw outputs")
    check.add_argument("weights", type=Path)
    check.add_argument("onnx", type=Path)
    check.add_argument("--manifest", type=Path, default=Path("data/splits/grouped-v2/valid.txt"))
    check.add_argument("--imgsz", type=int, default=640)
    check.add_argument("--limit", type=int, default=8)
    check.add_argument("--output", type=Path, default=Path("runs/export-check.json"))

    return parser


def _training_hyperparameters(profile: str) -> dict[str, float | int | bool]:
    """Return the explicit augmentation knobs used by a reproducible experiment."""
    if profile == "baseline":
        return {
            "degrees": 5.0, "translate": 0.1, "scale": 0.3, "fliplr": 0.5, "flipud": 0.0,
            "hsv_h": 0.01, "hsv_s": 0.25, "hsv_v": 0.25, "mosaic": 0.5,
            "close_mosaic": 10, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0,
        }
    if profile == "robust":
        return {
            "degrees": 5.0, "translate": 0.1, "scale": 0.3, "fliplr": 0.5, "flipud": 0.0,
            "hsv_h": 0.01, "hsv_s": 0.45, "hsv_v": 0.50, "mosaic": 0.5,
            "close_mosaic": 10, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.25,
        }
    raise ValueError(f"Unknown training profile: {profile}")


def _train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    if args.resume:
        result = YOLO(str(args.resume)).train(resume=True)
        print(f"Resumed training completed: {result.save_dir}")
        return 0
    model = YOLO(args.model)
    augmentation = _training_hyperparameters(args.profile)
    result = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=args.seed, deterministic=True,
        optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=True, cache=False, **augmentation,
        patience=25, pretrained=True, amp=True, project=str(args.project.resolve()), name=args.name,
        exist_ok=False, save=True, val=True, plots=True,
    )
    print(f"Training completed: {result.save_dir}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.split == "test" and not args.allow_test:
        raise SystemExit("Test data is held out. Pass --allow-test after freezing model and thresholds.")
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(args.project.resolve()), name=args.name,
        exist_ok=False, plots=True, save_json=True,
    )
    summary = {
        "model": str(args.model.resolve()), "data": str(args.data.resolve()), "split": args.split,
        "imgsz": args.imgsz, "batch": args.batch,
        "weight_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": [
            {"class_id": int(class_id), "class_name": metrics.names[int(class_id)],
             "precision": float(metrics.box.p[i]), "recall": float(metrics.box.r[i]),
             "f1": float(metrics.box.f1[i]),
             "ap50": float(metrics.box.ap50[i]), "ap50_95": float(metrics.box.ap[i])}
            for i, class_id in enumerate(metrics.box.ap_class_index)
        ],
        "speed_ms_per_image": metrics.speed,
        "precision_recall_policy": "Ultralytics max mean F1 validation curve point; "
                                   "not fixed-confidence precision or image classification accuracy",
    }
    report_path = args.data.parent / "split-report.json"
    if report_path.is_file():
        summary["data_split"] = json.loads(report_path.read_text(encoding="utf-8"))
    output = Path(metrics.save_dir) / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _export(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    exported = model.export(format="onnx", imgsz=args.imgsz, half=args.half,
                            dynamic=False, simplify=False, opset=args.opset)
    print(f"Exported ONNX model: {exported}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "balance-train":
        from .balance import make_balanced_train

        print(json.dumps(make_balanced_train(args.data, args.output, target_class=args.target_class,
                                              factor=args.factor, seed=args.seed),
                         ensure_ascii=False, indent=2))
        return 0
    if args.command in {"audit", "prepare", "prepare-grouped"}:
        audit = audit_dataset(args.dataset)
        if args.command == "audit":
            write_audit(audit, args.output)
            summary = {key: value for key, value in audit.items() if key not in {"_records", "exact_duplicate_groups"}}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == "prepare":
            write_audit(audit, args.output.parent / "audit")
            report = write_exact_split(audit, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            from .grouping import write_grouped_split

            report = write_grouped_split(audit, args.output, distance=args.distance,
                                         seed=args.seed, workers=args.workers)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        return _train(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "export":
        return _export(args)
    if args.command == "benchmark":
        from .benchmark import benchmark

        result = benchmark(args.model, args.manifest, imgsz=args.imgsz, device=args.device,
                           warmup=args.warmup, limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "make-stress":
        from .stress import make_stress_dataset

        print(json.dumps(make_stress_dataset(args.data, args.output, split=args.split, limit=args.limit),
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "predict":
        from .inference import predict_source

        result = predict_source(args.model, args.source, args.output, device=args.device,
                                 conf=args.conf, iou=args.iou, max_frames=args.max_frames)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify-export":
        from .export_check import verify_export

        result = verify_export(args.weights, args.onnx, args.manifest, imgsz=args.imgsz, limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
