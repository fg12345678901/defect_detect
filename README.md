# 缺陷检测 (defect_detect)

该仓库提供了一套用于表面缺陷检测的训练与推理脚本，支持分类模型和分割模型，并带有一个基于 Flask 的简易 Web 界面。项目中的数据处理脚本可以将常见公开数据集转换成统一的目录结构，便于模型训练和评估。

![1](./md_images/1.png)

## 目录结构

- `data/`            数据集预处理脚本，将原始数据转换为训练/验证集
- `dataset/`         经过处理后的示例数据集，结构如下：
    - `<task>/train/train_images`
    - `<task>/train/train_masks`
    - `<task>/val/val_images`
    - `<task>/val/val_masks`
- `engine/`          训练和推理脚本
- `models/`          分类与分割模型定义
- `losses/`          损失函数实现
- `utils/`           指标计算与可视化工具
- `webapp/`          Web 演示界面
- `logs/`            训练过程生成的模型权重和 TensorBoard 日志

## 依赖安装

项目基于 Python 3.8+。建议使用 `virtualenv` 或 `conda` 创建独立环境，然后安装 PyTorch 及其它依赖：

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

`requirements.txt` 中已列出了 Flask、segmentation_models_pytorch 等常用依赖，如有需要可根据实际环境增删。

## 数据集准备

`data/` 目录下提供了多个脚本示例，用于将原始数据集转换为统一的 one-hot 掩膜格式。处理完成后，数据应放置在 `dataset/<task>/` 下，其结构示例如下：

```
 dataset/phone/
 ├─ train/
 │  ├─ train_images/
 │  └─ train_masks/
 └─ val/
    ├─ val_images/
    └─ val_masks/
```

分类数据集可使用规范化后的整个dataset，也可另行指定。

## 训练

### 分类模型

```bash
python -m engine.train_cls \
       --data_dir dataset/ \
       --epochs 20 \
       --batch 8 \
       --run_name Classify_test
```

### 分割模型

```bash
python -m engine.train_seg \
       --run_name "solar-panel_model" \
       --train_dir "dataset/solar-panel/train/train_images" \
       --train_mask_dir "dataset/solar-panel/train/train_masks" \
       --val_dir "dataset/solar-panel/val/val_images" \
       --val_mask_dir "dataset/solar-panel/val/val_masks" \
       --classes 29 \
       --epochs 150 \
       --crop_size 256 \
       --lr 1e-4 \
       --batch 8
```

训练完成的模型和日志将保存在 `logs/<run_name>/` 目录下。

## 推理与评估

### 分割模型推理

```bash
python -m engine.infer \
       --task phone \
       --model logs/phone_model_final/best_model.pth \
       --classes 4 \
       --images dataset/phone/val/val_images \
       --gt_masks dataset/phone/val/val_masks \
       --pred_vis logs/phone_model_final/predictions
```

### 分类模型推理

```bash
python -m engine.infer_cls \
       --model_path logs/Classify/best_model.pth \
       --test_dir dataset/Classify_test
```

以上命令将在指定目录下保存可视化结果，并在终端打印评估指标。


## 预训练模型下载

分类器以及四个分割模型均已上传至 Google Drive：

<https://drive.google.com/drive/folders/1iDLOm0Uc5CsU0P06J9y7t3dgQyumtTD6?usp=sharing>

包含的模型及其类别数如下：

- `logs/Classify/best_model.pth` – 分类器
- `logs/phone_model_final/best_model.pth` – 4 类分割模型（含背景）
- `logs/magnetic_model/best_model.pth` – 6 类分割模型（含背景）
- `logs/solar-panel_model/best_model.pth` – 29 类分割模型（含背景）
- `logs/steel_model/best_model_v11.pth` – 5 类分割模型（含背景）

## 数据集来源

仓库示例数据整理自以下公开数据集，表中列出了对应的本地目录和类别数：

| 本地目录 | 原始数据集名称 | 类别数（含背景） | 链接 |
|----------|----------------|-----------------|------|
| `phone` | MSD – Mobile‑Screen Defect（油渍、划痕、污渍） | 4 | <https://github.com/jianzhang96/MSD?tab=readme-ov-file> |
| `magnetic` | Magnetic‑Tile Surface Defect (MT‑Defect) | 6 | <https://github.com/MitraDP/Detection-of-Surface-Defects-in-Magnetic-Tile-Images> |
| `steel` | Severstal: Steel Defect Detection | 5 | <https://www.kaggle.com/competitions/severstal-steel-defect-detection> |
| `solar-panel` | Benchmark EL Images (太阳能电池 EL 多缺陷语义分割) | 29 | <https://github.com/TheMakiran/BenchmarkELimages> |

## Web 演示

`webapp/` 目录提供了一个基于 Flask 的简单前端，可对上传的图片进行分类和分割演示：

```bash
cd webapp
python app.py
```

启动后访问 `http://localhost:5000/` 即可使用界面上传图片查看结果。


