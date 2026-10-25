# Learning path

Learn each topic just before the project needs it. The goal is to be able to run, inspect, and explain each experiment, not to reimplement every standard library.

| Stage | Knowledge | Practical pass condition |
|---|---|---|
| Python and environment | Python functions/data structures/files, virtual environments, Git, WSL paths | Read one image, call a pretrained detector, save the rendered result |
| Data | YOLO normalized boxes, class IDs, train/val/test, leakage, class imbalance, image display | Explain every class's annotation unit and visualize labels; exact hashes do not cross splits |
| Detection training | PyTorch tensors, gradients, optimizer, epoch/batch/LR, CNN/feature maps, YOLO backbone/neck/head | Run a short smoke training, interpret loss/validation curves, reload best weights |
| Evaluation | IoU, TP/FP/FN, precision, recall, F1, AP/mAP, confidence/NMS | Report per-class metrics and select thresholds on validation data only |
| Robustness methods | OpenCV/NumPy augmentation, paired views, feature masks, loss weighting | Show a controlled ablation on real lighting/occlusion subsets |
| Lightweight models | depthwise/pointwise convolution, feature pyramid, teacher/student knowledge distillation | Compare model size and target-device latency, not FLOPs alone |
| Deployment | ONNX, FP16/INT8, preprocessing, letterbox, NMS, latency/P95, memory | Compare exported predictions and run a sustained benchmark on the named device |
| Application | HTTP/JSON, FastAPI, HTML/CSS/JavaScript, OpenCV video/camera | Another person can upload an image and export the annotated result |
| Research communication | controlled comparisons, seeds, error review, citation/metric scope | Every claim points to a saved run, data version, and hardware setup |

Suggested reading order: LCDDN-YOLO for cotton-specific lightweight detection; CM-YOLO for scale errors and the YOLO11n comparison; IRMKD for teacher-student background (it is classification, not detection); RGC-YOLO and YOLO-Citrus for edge deployment practices. Details and caveats are in `references.md`.
