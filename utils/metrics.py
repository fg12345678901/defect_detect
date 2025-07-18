# utils/metrics.py
"""
Dice / IoU 评估工具
--------------------------------------------
‣  calc_dice, calc_iou      —— 单幅二值掩膜指标
‣  DiceMeter, IoUMeter      —— 多图宏平均 (忽略背景=0)
"""

from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


def calc_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = float((pred & gt).sum())
    union = float(pred.sum() + gt.sum())
    return (2.0 * inter + eps) / (union + eps)


def calc_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = float((pred & gt).sum())
    union = float(((pred | gt) > 0).sum())
    return (inter + eps) / (union + eps)


class _BaseMeter:
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.val_sum = np.zeros(num_classes, dtype=np.float64)
        self.count = np.zeros(num_classes, dtype=np.int32)

    def _update_one(self, pred_3d: np.ndarray, gt_3d: np.ndarray, fn):
        for c in range(1, self.num_classes):          # 跳过背景 0
            score = fn(pred_3d[c] > 0, gt_3d[c] > 0)
            self.val_sum[c] += score
            self.count[c] += 1

    def get_scores(self):
        means = np.divide(
            self.val_sum[1:],  # 背景不返回
            np.maximum(self.count[1:], 1),
            dtype=np.float64,
        )
        overall = means.mean() if means.size else 0.0
        return means.tolist(), float(overall)


class DiceMeter(_BaseMeter):
    def update(self, pred_3d: np.ndarray, gt_3d: np.ndarray):
        super()._update_one(pred_3d, gt_3d, calc_dice)


class IoUMeter(_BaseMeter):
    def update(self, pred_3d: np.ndarray, gt_3d: np.ndarray):
        super()._update_one(pred_3d, gt_3d, calc_iou)


def evaluate_classifier(model, loader, criterion, device, class_names: list):
    """
    Evaluate a classification model and return loss, macro F1,
    and a detailed report dict.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return avg_loss, macro_f1, report_dict
