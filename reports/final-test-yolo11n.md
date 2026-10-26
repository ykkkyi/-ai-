# YOLO11n 冻结测试集结果

该报告是在模型、输入尺寸、置信度/NMS策略和部署路径冻结后，对 grouped-v2 测试集进行的一次性评估。测试集没有用于模型选择或阈值调参。

## 配置

| 项目 | 值 |
|---|---|
| 模型 | `runs/E1_grouped_yolo11n_seed42/weights/best.pt` |
| 数据清单 | `data/splits/grouped-v2/data.yaml` |
| 测试图片 | 544 |
| 测试框 | 842 |
| 输入尺寸 | 640×640 |
| batch | 8 |
| 权重 SHA-256 | `c228f71c11bebaf35608062c4810a87b85f4ecc160b4e8402f4261facd287eb4` |
| 软件 | Python 3.14.4，PyTorch 2.14.0+cu130，Ultralytics 8.4.189 |

## 总体结果

| 指标 | 测试集结果 |
|---|---:|
| Precision | 79.47% |
| Recall | 64.68% |
| mAP50 | 69.69% |
| mAP50-95 | 39.07% |

## 逐类结果

| 类别 | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| leaf spot | 55.66% | 45.14% | 48.92% | 22.43% |
| Aphids | 85.16% | 86.73% | 84.30% | 43.52% |
| Armyworm | 88.27% | 81.82% | 86.31% | 43.07% |
| Bacterial Blight | 88.36% | 66.59% | 75.88% | 42.11% |
| Fusarium Wilt | 47.15% | 18.25% | 20.44% | 8.19% |
| Grey milddew | 91.02% | 86.96% | 85.31% | 46.69% |
| Healthy | 80.88% | 56.93% | 77.49% | 43.76% |
| Leaf Curl | 99.23% | 75.00% | 78.86% | 62.77% |

## 解读

测试集总体结果高于验证集，但 Fusarium Wilt 和 leaf spot 仍是主要漏检类别。E3 类别平衡消融已证明可以提高 Fusarium Wilt Recall，但会牺牲总体 Precision、mAP50 和速度，因此保留为按比赛评分函数切换的备选模型，不把两套结果混成一个最终成绩。

测试集结果仅代表当前 grouped-v2 视觉分组和数据来源。pHash 分组不能证明植株或采集会话完全独立；真实田间留出集和指定边缘设备仍需单独验收。

评估原始 JSON：`reports/final-test-yolo11n.json`。可复现命令：

```bash
cd /home/ykkkyi/yky/falcons/cotton-doctor
source .venv/bin/activate
python -m cotton_doctor.cli evaluate \
  runs/E1_grouped_yolo11n_seed42/weights/best.pt \
  --data data/splits/grouped-v2/data.yaml --split test --allow-test \
  --imgsz 640 --batch 8 --device 0 \
  --project runs/final-test --name yolo11n_frozen_test
```
