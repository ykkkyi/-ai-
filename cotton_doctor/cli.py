"""Command-line entry points for dataset preparation, training, and evaluation."""

from __future__ import annotations

import argparse
import json
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

    train = subparsers.add_parser("train", help="Train a YOLO detection model")
    train.add_argument("--data", type=Path, default=Path("configs/data.local.yaml"))
    train.add_argument("--model", default="yolov8n.pt")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--name", default="E0_yolov8n_seed42")
    train.add_argument("--project", type=Path, default=Path("runs"))

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

    return parser


def _train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    model = YOLO(args.model)
    result = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=args.seed, deterministic=True,
        optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=True, degrees=5.0, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.25, hsv_v=0.25,
        mosaic=0.5, close_mosaic=10, mixup=0.0, copy_paste=0.0, cache=False,
        patience=25, pretrained=True, amp=True, project=str(args.project.resolve()), name=args.name,
        exist_ok=False, save=True, val=True, plots=True,
    )
    print(f"Training completed: {result.save_dir}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
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
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
    }
    output = args.project.resolve() / args.name / "summary.json"
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
    if args.command in {"audit", "prepare"}:
        audit = audit_dataset(args.dataset)
        if args.command == "audit":
            write_audit(audit, args.output)
            summary = {key: value for key, value in audit.items() if key not in {"_records", "exact_duplicate_groups"}}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            write_audit(audit, args.output.parent / "audit")
            report = write_exact_split(audit, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        return _train(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "export":
        return _export(args)
    raise AssertionError(f"Unhandled command: {args.command}")
