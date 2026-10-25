"""Serial baseline experiments. Never evaluates the held-out final test set.

Run from the repository root. --publish-reports explicitly enables publishing just
the two generated aggregate reports, with no dataset, image, or weight artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=Path("data/splits/grouped-v2/data.yaml"))
    result.add_argument("--epochs", type=int, default=120)
    result.add_argument("--device", default="0")
    result.add_argument("--wait-pid", type=int, default=None)
    result.add_argument("--skip-stress", action="store_true")
    result.add_argument("--publish-reports", action="store_true")
    result.add_argument("--plan", action="store_true", help="Print steps without starting jobs")
    return result


def state(stage: str, **extra) -> None:
    target = ROOT / "runs/baseline-suite/state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"stage": stage, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  **extra}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stage, extra, flush=True)


def run(label: str, *args: str) -> None:
    state(label)
    logs = ROOT / "runs/baseline-suite/logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{label}.log").open("a", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "cotton_doctor.cli", *map(str, args)],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def completed(run_dir: Path) -> bool:
    import torch

    checkpoint = run_dir / "weights/last.pt"
    if not checkpoint.exists():
        return False
    # Ultralytics finalizes both normal and early-stopped runs by stripping
    # optimizer state and setting epoch=-1 (ultralytics.utils.torch_utils).
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return saved.get("epoch") == -1 and saved.get("optimizer") is None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def publish_reports(paths: list[Path]) -> None:
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    if staged.strip():
        raise RuntimeError("Other staged changes exist; aggregate reports remain local")
    subprocess.run(["git", "add", "--", *(str(path.relative_to(ROOT)) for path in paths)],
                   cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False,
                             cwd=ROOT).returncode
    if changed == 1:
        subprocess.run(["git", "commit", "-m", "Report grouped cotton detection baselines"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=True)
    elif changed != 0:
        raise RuntimeError("Could not inspect the staged report diff")


def execute(args: argparse.Namespace) -> None:
    if args.wait_pid:
        state("waiting_for_existing_training", pid=args.wait_pid)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(10)
    import yaml

    data = args.data.resolve()
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    split_report = read_json(data.parent / "split-report.json")
    manifest = Path(config["path"]) / config["val"]
    validation = ROOT / "runs/baseline-suite/evaluation"
    report_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rows = []
    for index, model in enumerate(("yolov8n", "yolo11n")):
        name = f"E{index}_grouped_{model}_seed42"
        folder = ROOT / "runs" / name
        if folder.exists() and not completed(folder):
            if (folder / "weights/last.pt").exists():
                run(f"train_{model}", "train", "--resume", folder / "weights/last.pt")
            else:
                raise RuntimeError(f"Incomplete run has no checkpoint: {folder}")
        elif not folder.exists():
            run(f"train_{model}", "train", "--data", data, "--model", f"{model}.pt",
                "--epochs", args.epochs, "--device", args.device, "--name", name)
        if not completed(folder):
            raise RuntimeError(f"Training did not finalize: {folder}")
        weight = folder / "weights/best.pt"
        eval_name = f"{model}_val"
        summary_path = validation / eval_name / "summary.json"
        if not summary_path.exists():
            run(f"validate_{model}", "evaluate", weight, "--data", data, "--split", "val",
                "--batch", "8", "--device", args.device, "--project", validation, "--name", eval_name)
        summary = read_json(summary_path)
        with (folder / "results.csv").open(newline="", encoding="utf-8") as stream:
            actual_epochs = len(list(csv.DictReader(stream)))
        rows.append({"model": model, "run": name, "epochs_completed": actual_epochs,
                     "weight_sha256": summary["weight_sha256"], "metrics": summary["metrics"],
                     "per_class": summary["per_class"], "environment": {
                         key: summary[key] for key in ("python", "torch", "ultralytics")},
                     "stress_validation": {}})
    # No training runs share the GPU with timed inference.
    for row in rows:
        weight = ROOT / "runs" / row["run"] / "weights/best.pt"
        for device in dict.fromkeys((args.device, "cpu")):
            label = "cpu" if device == "cpu" else "gpu"
            output = ROOT / "runs/baseline-suite" / f"{row['model']}-{label}-speed.json"
            run(f"benchmark_{row['model']}_{label}", "benchmark", weight, "--manifest", manifest,
                "--device", device, "--limit", "100", "--output", output)
            speed = read_json(output)
            row[f"benchmark_{label}"] = {k: v for k, v in speed.items() if k not in {"model", "manifest"}}
    if not args.skip_stress:
        stress_root = ROOT / "data/stress/suite-val"
        if not (stress_root / "source.json").exists():
            run("prepare_stress", "make-stress", "--data", data, "--output", stress_root)
        for row in rows:
            for variant in read_json(stress_root / "source.json")["variants"]:
                name = f"{row['model']}_{variant}"
                summary_path = validation / name / "summary.json"
                if not summary_path.exists():
                    run(f"stress_{name}", "evaluate", ROOT / "runs" / row["run"] / "weights/best.pt",
                        "--data", stress_root / variant / "data.yaml", "--split", "val", "--batch", "8",
                        "--device", args.device, "--project", validation, "--name", name)
                row["stress_validation"][variant] = read_json(summary_path)["metrics"]
    chosen = max(rows, key=lambda row: row["metrics"]["metrics/mAP50-95(B)"])
    weights = ROOT / "weights"
    weights.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "runs" / chosen["run"] / "weights/best.pt", weights / "best.pt")
    run("export_chosen_baseline", "export", weights / "best.pt")
    agreement = ROOT / "runs/baseline-suite/export-agreement.json"
    run("verify_chosen_baseline", "verify-export", weights / "best.pt", weights / "best.onnx",
        "--manifest", manifest, "--output", agreement)
    report = {"scope": "Two standard pretrained baselines on grouped-v2; no novel method is evaluated yet",
              "report_code_commit": report_commit, "data": split_report, "models": rows,
              "selected_baseline": chosen["model"], "selection": "validation mAP50-95 only",
              "pr_policy": "Ultralytics validation precision/recall use the max mean F1 curve point; "
                           "deployment speed uses fixed confidence=0.25, NMS IoU=0.45",
              "final_test_evaluated": False, "export_agreement": read_json(agreement)}
    json_path = ROOT / "reports/baseline-comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 分组数据基线结果", "", "本报告由实际训练与验证生成；最终测试集尚未评测。",
             "标准模型对比尚不代表创新方法有效，合成困难测试也不能替代田间实测。", "",
             "| 模型 | 完成轮数 | 验证 mAP50 | 验证 mAP50–95 | 部署端中位延迟 ms | 模型 MiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        speed = row.get("benchmark_gpu", row.get("benchmark_cpu"))
        lines.append(f"| {row['model']} | {row['epochs_completed']} | "
                     f"{row['metrics']['metrics/mAP50(B)']:.2%} | "
                     f"{row['metrics']['metrics/mAP50-95(B)']:.2%} | "
                     f"{speed['latency_ms_median']} | {speed['model_file_mib']} |")
    lines += ["", f"当前按验证 mAP50–95 选择：{chosen['model']}。", "",
              "延迟含图片解码、预处理、模型与 NMS，batch=1；完整环境、逐类指标与合成压力测试见 JSON。",
              "ONNX 数值一致性通过不等于 ONNX 精度或目标边缘设备速度已经验证。"]
    md_path = ROOT / "reports/baseline-comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state("reports_ready", selected=chosen["model"])
    if args.publish_reports:
        publish_reports([json_path, md_path])
        state("reports_published", selected=chosen["model"])


def main() -> None:
    args = parser().parse_args()
    os.chdir(ROOT)
    if args.plan:
        print("Wait if requested → train/resume YOLOv8n → validate → train/resume YOLO11n → validate "
              "→ GPU/CPU benchmark → synthetic validation stress → select by val → export/check "
              "→ aggregate reports" + (" → publish reports" if args.publish_reports else ""))
        return
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    lock = ROOT / "runs/baseline-suite/active.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        execute(args)
    except Exception as error:
        state("failed", error=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
