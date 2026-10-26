# Cotton Doctor

棉卫士是一个面向棉花病虫害巡检的可复现目标检测项目，目标是在普通RGB图片、视频和本地摄像头上检测棉花病害与虫害相关目标，并将模型部署为本地网页服务。

仓库已实现数据审计、视觉近重复分组、训练/验证、困难条件压力集、速度测量、ONNX 一致性检查和本地图片/视频推理。当前正推进 YOLOv8n/YOLO11n 的分组数据基线；鲁棒训练、蒸馏和边缘设备实测按后续独立实验推进。

**状态**

- [x] 数据审计、SHA-256 去重、旋转/翻转 pHash 分组和可复算清单
- [x] 图片上传网页、视频推理、JSON/CSV 导出和相机入口
- [x] 合成困难集、端到端计时和 ONNX 原始输出一致性工具
- [ ] 完成 YOLOv8n/YOLO11n 分组数据基线与比较报告
- [ ] 完成真实田间光照、遮挡和尺度分组评价
- [ ] 完成单因素消融与小模型优化
- [ ] 完成目标边缘设备的 ONNX/TensorRT 实测

本仓库不包含原始棉花图片、标签、下载的数据压缩包、训练权重或推理缓存。公开数据集和预训练权重的许可应分别核对后再用于训练、分发。

## Requirements

- Python 3.11–3.14
- NVIDIA GPU and a matching PyTorch build for CUDA training; CPU can run the web app but trains slowly.
- Raw dataset in YOLO format, outside Git.

For the current RTX 5060 Laptop / CUDA 13.0 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cu130.txt
```

On other systems, install the PyTorch build matching your driver from the official [PyTorch installer](https://pytorch.org/get-started/locally/), then install the project:

```bash
python -m pip install -e '.[dev]'
```

## Dataset

The expected dataset layout is:

```text
dataset/
  data.yaml
  train/images/  train/labels/
  valid/images/  valid/labels/
  test/images/   test/labels/
```

Class IDs are kept in the dataset's original order. The supplied dataset has 8 classes and 6,097 images. A byte-level SHA-256 audit found 300 duplicate groups across the original train/validation/test sets. This project never moves or edits source images. Before the final split is frozen, also review near-duplicates, plants, capture sessions, label semantics, and missed boxes; exact hash deduplication alone does not establish an independent field test set.

Prepare local manifests from your own dataset path:

```bash
cotton-doctor audit /path/to/dataset --output data/processed/audit
cotton-doctor prepare /path/to/dataset --output data/splits/exact
cotton-doctor prepare-grouped /path/to/dataset --output data/splits/grouped-v2
```

The exact-only split is a diagnostic intermediate. Grouped-v2 co-locates pHash candidates, including rotation/reflection matches, then creates new seeded class-balanced splits: **3,801 train / 1,085 validation / 544 test**, covering 2,828 visual candidate groups. It does not delete visually similar images. The [versioned summary](reports/grouped-v2-summary.json) and [relative-path, hash-only manifest](reports/grouped-v2-manifest.csv) are public; source images, annotations and local absolute-path manifests are excluded. Review crops, plant/session grouping and agricultural label semantics before claiming field generalization.

## Train and evaluate

Use the generated manifest as the local config:

```bash
cp data/splits/grouped-v2/data.yaml configs/data.local.yaml
cotton-doctor train --data configs/data.local.yaml --model yolov8n.pt --epochs 120 --imgsz 640 --batch 8 --device 0 --name E0_grouped_yolov8n_seed42
```

The training entry point also has a controlled `robust` augmentation profile for the
illumination/occlusion ablation. It keeps the baseline optimizer, image size, seed and
geometry settings fixed, while increasing HSV brightness/saturation jitter and enabling
limited random erasing:

```bash
cotton-doctor train --data data/splits/grouped-v2/data.yaml --model yolo11n.pt \
  --epochs 120 --imgsz 640 --batch 8 --device 0 --seed 42 \
  --profile robust --name E2_robust_yolo11n_seed42
```

The `robust` run is an ablation candidate, not the selected model until its validation,
stress and deployment measurements are compared with `yolo11n` under the same protocol.

After reviewing validation metrics and errors:

```bash
cotton-doctor evaluate runs/E0_grouped_yolov8n_seed42/weights/best.pt --data configs/data.local.yaml --split val --imgsz 640 --batch 1
```

Only after model, confidence threshold, and deployment configuration have been frozen, evaluate the held-out test set explicitly:

```bash
cotton-doctor evaluate runs/E0_grouped_yolov8n_seed42/weights/best.pt --data configs/data.local.yaml --split test --allow-test
```

Run both baselines, synthetic validation stress checks, CPU/GPU timing and ONNX checks sequentially:

```bash
python scripts/run_baseline_suite.py --plan
python scripts/run_baseline_suite.py
```

Progress is saved in `runs/baseline-suite/state.json`. The suite selects a baseline by validation mAP50–95 and writes `reports/baseline-comparison.{json,md}` when complete; it never evaluates the final test set. Optional `--publish-reports` commits and pushes only these two aggregate reports. If a separate training job is active, use `--wait-pid <training-pid>` to wait for it.

## Local web demo

Copy the chosen model to `weights/best.pt` or set `COTTON_MODEL` to its path:

```bash
mkdir -p weights
cp runs/E0_grouped_yolov8n_seed42/weights/best.pt weights/best.pt
uvicorn cotton_doctor.web:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The service binds to loopback and uses CPU by default; set `COTTON_DEVICE=0` for GPU. It accepts uploads or browser camera captures and returns annotated images and box coordinates, with JSON/CSV/image downloads. Actual image and four-frame video inference passed integration checks. Physical USB camera and target edge hardware still need testing.

```bash
cotton-doctor predict weights/best.pt --source /path/to/cotton.mp4 --output runs/predict/video
cotton-doctor predict weights/best.pt --source 0 --max-frames 300 --output runs/predict/camera
cotton-doctor make-stress --data data/splits/grouped-v2/data.yaml
cotton-doctor benchmark weights/best.pt --manifest data/splits/grouped-v2/valid.txt --device 0
cotton-doctor export weights/best.pt
cotton-doctor verify-export weights/best.pt weights/best.onnx
```

Install ONNX dependencies with `python -m pip install -e '.[export]'`. Synthetic `scale_050` measures resolution loss, not geometric object-size change; occlusion stress retains the original full-object labels. Interpret these separately from real field difficulty groups.

## Evaluation protocol

Report mAP50, mAP50-95, precision, recall, F1, and per-class metrics. Select the operating confidence/NMS thresholds on validation data only. Evaluate real difficult subsets separately from synthetic stress tests. For deployment report device, software versions, input size, batch size, warm-up, median/P95 latency, sustained throughput, peak memory, and whether preprocessing and NMS are included. Model size or FLOPs alone does not establish a speed improvement.

## Data semantics and limitations

The supplied labels mix affected-leaf boxes, insect boxes, and plant-level scenes. In the inspected Aphids examples the box covers a large affected-leaf region, so detections must not be interpreted as aphid counts. The original names `Bacterial Blight` and misspelled `Grey milddew` require agricultural review before Chinese display names are finalized. A box detector cannot infer lesion area, severity percentage, field incidence, pathogen confirmation, or pesticide prescriptions from these labels.

## Development

```bash
pytest
ruff check cotton_doctor tests scripts
```

Chinese starting steps are in [docs/上手指南.md](docs/上手指南.md). See [the complete project and knowledge plan](docs/project-plan.md), [the implementation roadmap](docs/roadmap.md), [the staged learning path](docs/learning-path.md), and [recent paper notes](docs/references.md). [Integration evidence](reports/integration-checks.json) covers software checks on a provisional model, not final accuracy.

The corn-doctor project inspired the training → web → export → camera workflow. Its main pipeline classifies whole images; this repository implements its own eight-class box detector and application. Its scores and weights are not this project's results.

## License and attribution

Project code is released under the MIT License. This does not grant rights to redistribute the supplied dataset, third-party pretrained weights, or derived model weights. Check their own licenses and competition rules. Literature links are cited for research discussion; reported metrics are from their respective papers and are not results on this project's data.
