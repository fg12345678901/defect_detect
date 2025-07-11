# engine/infer.py (已复现并包含智能填充)
"""
通用推理与评估脚本

使用示例 (phone 任务):
-----------------------------------------------
python -m engine.infer `
       --task magnetic `
       --model "logs/magnetic_model/best_model.pth" `
       --classes 6 `
       --images "dataset/magnetic/val/val_images" `
       --gt_masks "dataset/magnetic/val/val_masks" `
       --pred_vis "logs/magnetic_model/predictions"

python -m engine.infer `
       --task phone `
       --model "logs/phone_model_final/best_model.pth" `
       --classes 4 `
       --images "dataset/phone/val/val_images" `
       --gt_masks "dataset/phone/val/val_masks" `
       --pred_vis "logs/phone_model_final/predictions"

# engine/infer.py (保留原始填充逻辑，并增加JSON颜色地图功能)

通用推理与评估脚本

使用示例 (使用自定义颜色JSON):
-----------------------------------------------
python -m engine.infer `
       --task solar-panel `
       --model "logs/solar-panel_model/best_model.pth" `
       --classes 29 `
       --images "dataset/solar-panel/val/val_images" `
       --gt_masks "dataset/solar-panel/val/val_masks" `
       --pred_vis "logs/solar-panel_model/predictions" `
       --color_map "dataset/solar-panel/color_map.json"
"""
from __future__ import annotations
import argparse, os, time, json  # <--- 新增 import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import albumentations as A
import cv2, numpy as np, torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F

# 导入您的自定义模块
from models.segmenter import get_unet_model
from utils.viz import visualize_prediction_3d
from utils.metrics import DiceMeter, IoUMeter


# 保持原始的预处理流程，不包含填充
def _val_tf():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


@torch.no_grad()
def infer_and_evaluate(
        model: torch.nn.Module,
        imgs: List[Path],
        vis_dir: Path,
        device: torch.device,
        gt_dir: Optional[Path] = None,
        num_classes: int = 4,
        task: str = "steel",
        color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,  # <--- 新增参数
):
    """
    执行推理，只保存可视化结果并打印指标。
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    dice_meter, iou_meter = DiceMeter(num_classes), IoUMeter(num_classes)
    tf = _val_tf()
    times = []

    for p in tqdm(imgs, desc=f"正在对 {task} 任务进行推理"):
        img_rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        tensor = tf(image=img_rgb)["image"].unsqueeze(0).to(device)

        # --- 保留您指定的原始手动填充逻辑 ---
        H_orig, W_orig = tensor.shape[-2:]
        pad_h = (32 - H_orig % 32) % 32
        pad_w = (32 - W_orig % 32) % 32
        tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h))
        # --- 填充逻辑结束 ---

        t0 = time.perf_counter()
        logits, _ = model(tensor_padded)
        logits = logits[..., :H_orig, :W_orig]
        if device.type == 'cuda': torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        pred = logits.argmax(1).squeeze(0).cpu().numpy()
        pred_3d_chw = np.zeros((num_classes, pred.shape[0], pred.shape[1]), np.uint8)
        for c in range(num_classes):
            pred_3d_chw[c] = (pred == c)

        # --- 可视化：将加载的自定义颜色字典传递给可视化函数 ---
        visualize_prediction_3d(img_rgb, pred_3d_chw, p.name, str(vis_dir), task=task, color_map_override=color_map)

        # (指标计算逻辑不变)
        if gt_dir:
            gt_path = gt_dir / f"{p.stem}.npy"
            if gt_path.exists():
                gt_3d_hwc = np.load(gt_path)
                gt_3d_chw = np.transpose(gt_3d_hwc, (2, 0, 1))
                dice_meter.update(pred_3d_chw, gt_3d_chw)
                iou_meter.update(pred_3d_chw, gt_3d_chw)

    # (终端打印结果逻辑不变)
    avg_time = np.mean(times)
    print(f"\n[速度] 平均每张耗时={avg_time * 1000:.1f} ms | FPS={1 / avg_time:.2f}")
    if gt_dir:
        dice_means, overall_dice = dice_meter.get_scores()
        iou_means, overall_iou = iou_meter.get_scores()
        print(f"[指标] Overall Dice: {overall_dice:.4f} | Overall IoU: {overall_iou:.4f}")
        for i, c in enumerate(range(1, num_classes)):
            print(f"      Class {c} -> Dice: {dice_means[i]:.4f}, IoU: {iou_means[i]:.4f}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- 核心参数 ---
    ap.add_argument("--images", required=True, help="待推理的图片目录路径")
    ap.add_argument("--model", required=True, help="模型权重文件 (.pth) 的路径")
    ap.add_argument("--classes", type=int, required=True, help="模型的类别数 (包括背景)")
    ap.add_argument("--task", type=str, default="steel", choices=["steel", "phone", "magnetic", "solar-panel"],
                    help="用于从预设中选择颜色。若提供--color_map，则此项仅作参考。")

    # --- 新增参数：用于加载外部颜色配置文件 ---
    ap.add_argument("--color_map", type=str, default=None,
                    help="自定义颜色映射JSON文件的路径 (例如为 solar-panel 任务)")

    # --- 可选参数 ---
    ap.add_argument("--gt_masks", default=None, help="真值掩膜目录的路径，如果提供则会计算并打印评估指标")
    ap.add_argument("--pred_vis", default="predictions", help="保存可视化结果的目录名称")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 新增逻辑：加载自定义颜色地图 ---
    custom_color_map = None
    if args.color_map:
        if os.path.exists(args.color_map):
            print(f"正在从 {args.color_map} 加载自定义颜色...")
            with open(args.color_map, 'r') as f:
                # JSON加载的键是字符串，需要转回整型的类别ID
                custom_color_map = {int(k): tuple(v) for k, v in json.load(f).items()}
        else:
            print(f"警告：找不到指定的颜色配置文件 {args.color_map}，将使用默认颜色。")

    # --- 加载模型 ---
    print(f"正在加载模型: {args.model}")
    model = get_unet_model("efficientnet-b4", None, 3, args.classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    # --- 准备文件列表 ---
    image_dir = Path(args.images)
    imgs = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}])

    # --- 执行推理和评估 ---
    infer_and_evaluate(
        model=model,
        imgs=imgs,
        vis_dir=Path(args.pred_vis),
        device=device,
        gt_dir=Path(args.gt_masks) if args.gt_masks else None,
        num_classes=args.classes,
        task=args.task,
        color_map=custom_color_map,  # <--- 将加载的颜色字典传递下去
    )
    print(f"\n推理完成！可视化结果已保存到 '{args.pred_vis}' 目录。")


if __name__ == "__main__":
    main()