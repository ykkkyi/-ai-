# Cotton Doctor

棉卫士是一个面向棉花病虫害巡检的可复现目标检测项目，目标是在普通RGB图片、视频和本地摄像头上检测棉花病害与虫害相关目标，并将模型部署为本地网页服务。

当前仓库先完成可复现工程和 YOLOv8n 基线。数据审计已发现原划分之间有完全重复图片，项目在训练前提供非破坏性的去重清单生成。数据清理以后才训练正式基线；光照鲁棒训练、蒸馏和边缘部署均按单独对照实验推进。

**状态**

- [x] 建立数据审计、精确去重和训练/评估命令
- [x] 初始化本地图片检测网页和接口
- [ ] 对精确去重并人工复核后的数据训练 YOLOv8n/YOLO11n 基线
- [ ] 完成光照、遮挡和尺度分组评价
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
```

The prepare command writes a local `data.yaml`, absolute image-list manifests, an audit report, and an excluded-copy list. Generated data and reports are ignored by Git. The split policy keeps one copy per SHA-256 and preserves `test > valid > train`; inspect `data/splits/exact/split-report.json` and `data/processed/audit/image-manifest.csv` before training.

## Train and evaluate

Use the generated manifest as the local config:

```bash
cp data/splits/exact/data.yaml configs/data.local.yaml
cotton-doctor train --data configs/data.local.yaml --model yolov8n.pt --epochs 120 --imgsz 640 --batch 8 --device 0 --name E0_yolov8n_seed42
```

After reviewing validation metrics and errors:

```bash
cotton-doctor evaluate runs/E0_yolov8n_seed42/weights/best.pt --data configs/data.local.yaml --split val --imgsz 640 --batch 1
```

Only after model, confidence threshold, and deployment configuration have been frozen, evaluate the held-out test set explicitly:

```bash
cotton-doctor evaluate runs/E0_yolov8n_seed42/weights/best.pt --data configs/data.local.yaml --split test --allow-test
```

## Local web demo

Copy the chosen model to `weights/best.pt` or set `COTTON_MODEL` to its path:

```bash
mkdir -p weights
cp runs/E0_yolov8n_seed42/weights/best.pt weights/best.pt
uvicorn cotton_doctor.web:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The service binds to loopback by default; no cloud account or WeChat mini program is required. It accepts single-image uploads and returns the annotated image and box coordinates. Video, USB camera, ONNX, and edge-engine paths are planned next.

## Evaluation protocol

Report mAP50, mAP50-95, precision, recall, F1, and per-class metrics. Select the operating confidence/NMS thresholds on validation data only. Evaluate real difficult subsets separately from synthetic stress tests. For deployment report device, software versions, input size, batch size, warm-up, median/P95 latency, sustained throughput, peak memory, and whether preprocessing and NMS are included. Model size or FLOPs alone does not establish a speed improvement.

## Data semantics and limitations

The supplied labels mix affected-leaf boxes, insect boxes, and plant-level scenes. In the inspected Aphids examples the box covers a large affected-leaf region, so detections must not be interpreted as aphid counts. The original names `Bacterial Blight` and misspelled `Grey milddew` require agricultural review before Chinese display names are finalized. A box detector cannot infer lesion area, severity percentage, field incidence, pathogen confirmation, or pesticide prescriptions from these labels.

## Development

```bash
pytest
ruff check cotton_doctor tests
```

Chinese starting steps are in [docs/上手指南.md](docs/上手指南.md). See [the implementation roadmap](docs/roadmap.md), [the staged learning path](docs/learning-path.md), and [recent paper notes](docs/references.md).

## License and attribution

Project code is released under the MIT License. This does not grant rights to redistribute the supplied dataset, third-party pretrained weights, or derived model weights. Check their own licenses and competition rules. Literature links are cited for research discussion; reported metrics are from their respective papers and are not results on this project's data.
