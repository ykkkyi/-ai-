# Implementation roadmap

The project advances one reproducible milestone at a time. No accuracy, latency, or edge-support claim is published until it is backed by a local run and its exact environment is recorded.

## Milestone 1: dataset and baseline

1. Run the read-only image/label audit.
2. Deduplicate byte-identical files with the test-first policy.
3. Review near-duplicate scenes, class semantics, empty-label images, and missed boxes.
4. Freeze a versioned train/validation/test manifest and record its hash.
5. Train YOLOv8n and YOLO11n with the same input size, training budget, and pretrained-data policy.
6. Save per-class AP/recall, PR operating point, confusion analysis, training logs, and the exact software/hardware versions.

## Milestone 2: difficulty analysis and robust training

Label real illumination, occlusion, and scale conditions on a field holdout. First compare ordinary augmentation, agricultural lighting augmentation, lighting-paired consistency, and limited occlusion augmentation one change at a time. Keep generated stress images separate from the real-field results. Measure both missed detections and false positives.

## Milestone 3: lightweight model and distillation

Retain the high-resolution P3 path initially. Test selected P4/P5 depthwise-pointwise replacements only if device measurements support them. Compare no distillation, ordinary teacher-student distillation, and foreground/class-balanced distillation. The deployment graph must exclude the teacher and training-only adapters.

## Milestone 4: export and application

Verify PyTorch/ONNX output agreement, then measure FP16 on the chosen target. Try INT8 only with calibration images from training data and report per-class/stress-set accuracy loss. Add the browser UI, file/video/camera paths, and exportable JSON/CSV results.

## Milestone 5: frozen evaluation and competition package

Freeze weights, confidence/NMS settings, preprocessing, and export format before final test evaluation. Publish exact data provenance, per-class results, real versus synthetic stress results, model size, and measured end-to-end latency. Package the demo instructions, model metadata, technical proposal, slides, and a 3–5 minute demo video.

## Current audit facts

The supplied dataset has 6,097 640×640 images and 12,265 YOLO boxes. There are 300 exact duplicate groups across splits (667 duplicate groups total, including within-split duplicates). Exact deduplication leaves 5,430 files: train 3,669, validation 1,152, test 609. These counts are provisional until near-duplicate and label review is complete. The supplied repository ZIP contained a README only and no reproducible source code or weights.
