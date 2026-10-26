"""Evaluate the robust augmentation candidate without consuming the held-out test set."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from cotton_doctor.benchmark import benchmark as measure_benchmark

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("runs/E2_robust_yolo11n_seed42/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/splits/grouped-v2/data.yaml"))
    parser.add_argument("--stress-root", type=Path, default=Path("data/stress/suite-val"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--project", type=Path, default=Path("runs/robust-ablation"))
    parser.add_argument("--report", type=Path, default=Path("reports/robust-ablation.json"))
    parser.add_argument("--label", default="robust", help="Short name used in report headings and run names")
    parser.add_argument("--baseline-speed", type=Path,
                        help="Optional JSON list of same-protocol baseline repeats")
    parser.add_argument("--epochs-completed", type=int,
                        help="Actual completed epochs recorded by the training results.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cli(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "cotton_doctor.cli", *args], cwd=ROOT, check=True)


def evaluate(model: Path, data: Path, project: Path, name: str, device: str) -> dict:
    summary = project / name / "summary.json"
    if not summary.is_file():
        run_cli("evaluate", model, "--data", data, "--split", "val", "--batch", "8",
                "--device", device, "--project", project, "--name", name)
    return read_json(summary)


def benchmark(model: Path, manifest: Path, project: Path, name: str, device: str, limit: int,
              repeats: int) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output = project / f"{name}.json"
    if output.is_file():
        return read_json(output)
    rows = [measure_benchmark(model, manifest, device=device, limit=limit) for _ in range(repeats)]
    ordered = sorted(rows, key=lambda row: row["latency_ms_median"])
    result = dict(ordered[len(ordered) // 2])
    result["repeats"] = rows
    result["repeat_policy"] = "Median row after sorting repeated 100-image runs by median latency"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    model = args.model.resolve()
    data = args.data.resolve()
    stress_root = args.stress_root.resolve()
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    val = evaluate(model, data, project / "evaluation", f"{args.label}_val", args.device)
    source = read_json(stress_root / "source.json")
    stress = {}
    for variant in source["variants"]:
        stress[variant] = evaluate(model, stress_root / variant / "data.yaml", project / "evaluation",
                                   f"{args.label}_{variant}", args.device)
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    manifest = Path(config["path"]) / config["val"]
    speed = {
        "gpu": benchmark(model, manifest, project, f"{args.label}_gpu_speed", args.device,
                         args.limit, args.repeats),
        "cpu": benchmark(model, manifest, project, f"{args.label}_cpu_speed", "cpu",
                         args.limit, args.repeats),
    }
    baseline = read_json(ROOT / "reports/baseline-comparison.json")
    base_row = next(row for row in baseline["models"] if row["model"] == "yolo11n")
    speed_reference = "baseline-comparison.json"
    if args.baseline_speed:
        baseline_repeats = read_json(ROOT / args.baseline_speed)
        if isinstance(baseline_repeats, list):
            baseline_speed = sorted(
                baseline_repeats, key=lambda row: row["latency_ms_median"]
            )[len(baseline_repeats) // 2]
        else:
            baseline_speed = baseline_repeats
        base_row = dict(base_row)
        base_row["benchmark_gpu"] = baseline_speed
        speed_reference = str(args.baseline_speed)
    report = {
        "scope": f"YOLO11n {args.label} ablation on grouped-v2; final test remains held out",
        "model": str(model),
        "profile": args.label,
        "training": {"epochs_completed": args.epochs_completed},
        "data": baseline["data"],
        "validation": val,
        "stress_validation": {key: value for key, value in stress.items()},
        "benchmark": speed,
        "baseline_reference": base_row,
        "speed_reference": speed_reference,
        "final_test_evaluated": False,
    }
    report_path = (ROOT / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    base_metrics = base_row["metrics"]
    robust_metrics = val["metrics"]
    base_speed = base_row["benchmark_gpu"]
    robust_speed = speed["gpu"]
    lines = [
        f"# {args.label} 消融结果", "",
        f"该报告只使用分组验证集及其合成压力子集；最终测试集仍然保留。实际完成训练轮数：{args.epochs_completed or '未提供'}。",
        "与基线相比，模型、数据划分、输入尺寸、优化器、seed 和训练预算保持一致；具体训练差异见 JSON。", "",
        f"| 指标 | YOLO11n 基线 | {args.label} | 变化 |", "|---|---:|---:|---:|",
        f"| Precision | {base_metrics['metrics/precision(B)']:.2%} | {robust_metrics['metrics/precision(B)']:.2%} | {robust_metrics['metrics/precision(B)'] - base_metrics['metrics/precision(B)']:+.2%} |",
        f"| Recall | {base_metrics['metrics/recall(B)']:.2%} | {robust_metrics['metrics/recall(B)']:.2%} | {robust_metrics['metrics/recall(B)'] - base_metrics['metrics/recall(B)']:+.2%} |",
        f"| mAP50 | {base_metrics['metrics/mAP50(B)']:.2%} | {robust_metrics['metrics/mAP50(B)']:.2%} | {robust_metrics['metrics/mAP50(B)'] - base_metrics['metrics/mAP50(B)']:+.2%} |",
        f"| mAP50-95 | {base_metrics['metrics/mAP50-95(B)']:.2%} | {robust_metrics['metrics/mAP50-95(B)']:.2%} | {robust_metrics['metrics/mAP50-95(B)'] - base_metrics['metrics/mAP50-95(B)']:+.2%} |",
        f"| GPU median latency | {base_speed['latency_ms_median']:.3f} ms | {robust_speed['latency_ms_median']:.3f} ms | {robust_speed['latency_ms_median'] - base_speed['latency_ms_median']:+.3f} ms |",
        f"| GPU throughput | {base_speed['throughput_fps']:.2f} FPS | {robust_speed['throughput_fps']:.2f} FPS | {robust_speed['throughput_fps'] - base_speed['throughput_fps']:+.2f} FPS |",
        "", "## 压力集 mAP50", "", f"| 条件 | 基线 | {args.label} | 变化 |", "|---|---:|---:|---:|",
    ]
    for variant in source["variants"]:
        base_value = base_row["stress_validation"][variant]["metrics/mAP50(B)"]
        robust_value = stress[variant]["metrics"]["metrics/mAP50(B)"]
        lines.append(f"| {variant} | {base_value:.2%} | {robust_value:.2%} | {robust_value - base_value:+.2%} |")
    lines += ["", "仅当干净验证集和困难条件均支持时，才将该配置作为候选最终模型；压力集结果不能替代真实田间留出集。"]
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "validation": robust_metrics,
                      "final_test_evaluated": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
