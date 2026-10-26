Record each run in `results-template.csv` or a versioned JSON file. Do not report unmeasured values. Keep validation experiments separate from the single frozen final test evaluation. Every result should identify its data manifest, model/weight, code commit, software versions, hardware, input size, confidence/NMS policy, and timing boundaries. The frozen YOLO11n test result is [`final-test-yolo11n.md`](final-test-yolo11n.md), with machine-readable details in [`final-test-yolo11n.json`](final-test-yolo11n.json).

`grouped-v2-summary.json` and `grouped-v2-manifest.csv` document the current grouped split. The CSV includes relative image names, hashes and group IDs, with no source pixels or box coordinates. `integration-checks.json` records software checks with provisional weights and synthetic fixtures; it is not an agricultural accuracy or edge-performance result.

The baseline suite generates `baseline-comparison.json` and `.md` only after successful training, validation, benchmarking and export checks. These reports keep the final test set unopened for model evaluation and do not establish that a proposed novel method works.

`robust-ablation.json` and `.md` contain the first controlled augmentation ablation. The
profile improves recall and low-contrast stress slightly but does not improve overall mAP
or occlusion robustness, so the frozen YOLO11n baseline remains the current candidate.

`balanced-fusarium-ablation.json` and `.md` record the second ablation. Repeating rare-class
training images raises Fusarium Wilt recall while leaving the deployment graph unchanged;
it does not improve synthetic occlusion and must be selected only if the competition score
weights rare-class recall enough to justify the small overall mAP tradeoff.
