# data/smart_crop.py
"""
Steel-surface datasets (full-image + multi-crop) and Albumentations pipelines.

‣ BaseDefectDataset        – abstract parent
‣ SteelDefectDataset       – 1 image → 1 sample（用于验证 / 推理）
‣ SteelDefectMultiCropDataset – 1 image → n patches（训练用多裁剪）
‣ SteelValDataset          – alias of SteelDefectDataset (kept for backward-compat)
‣ get_training_augmentation / get_validation_augmentation
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# ----------------------------------------------------------------------
# helpers ───────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
def find_defect_bboxes(mask_1ch: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """Return bounding boxes of all connected defect components.

    Each item: (x, y, w, h, area) – background (0) is ignored.
    """
    mask_bin = (mask_1ch > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )
    bboxes = []
    for label_id in range(1, num_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]
        bboxes.append((x, y, w, h, area))
    return bboxes


def random_crop_any(
    image: np.ndarray, mask: np.ndarray, crop_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Free random crop (used if no defect)."""
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return image, mask
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return (
        image[top : top + crop_size, left : left + crop_size],
        mask[top : top + crop_size, left : left + crop_size],
    )


def random_crop_full_in_diff_pos(
    image: np.ndarray,
    mask: np.ndarray,
    box: Tuple[int, int, int, int, int],
    crop_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop one patch so that the given bounding-box is fully inside."""
    h, w = image.shape[:2]
    x, y, bw, bh, _ = box

    if bw > crop_size or bh > crop_size:  # defect bigger than patch – center align
        cx, cy = x + bw // 2, y + bh // 2
        left = max(0, min(w - crop_size, cx - crop_size // 2))
        top = max(0, min(h - crop_size, cy - crop_size // 2))
        return (
            image[top : top + crop_size, left : left + crop_size],
            mask[top : top + crop_size, left : left + crop_size],
        )

    # feasible ranges (defect entirely inside patch)
    left_min = max(0, x + bw - crop_size)
    left_max = min(x, w - crop_size)
    top_min = max(0, y + bh - crop_size)
    top_max = min(y, h - crop_size)
    if left_min > left_max:
        left_min, left_max = 0, max(0, w - crop_size)
    if top_min > top_max:
        top_min, top_max = 0, max(0, h - crop_size)

    left = random.randint(left_min, left_max) if left_max >= left_min else 0
    top = random.randint(top_min, top_max) if top_max >= top_min else 0
    return (
        image[top : top + crop_size, left : left + crop_size],
        mask[top : top + crop_size, left : left + crop_size],
    )


# ----------------------------------------------------------------------
# datasets ──────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
class BaseDefectDataset(Dataset):
    """Abstract dataset that returns (tensor image, long mask)."""

    def __init__(
        self,
        image_ids: List[str],
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.BasicTransform] = None,
    ):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def _read_image(self, image_id: str) -> np.ndarray:
        return np.array(Image.open(os.path.join(self.images_dir, image_id)).convert("RGB"))

    def _read_mask(self, image_id: str) -> np.ndarray:
        npy = np.load(
            os.path.join(
                self.masks_dir,
                image_id.replace(".jpg", ".npy").replace(".png", ".npy"),
            )
        )
        if npy.ndim == 3 and npy.shape[2] > 1:
            return npy.argmax(axis=2).astype(np.uint8)
        return npy.astype(np.uint8)


class SteelDefectDataset(BaseDefectDataset):
    """Return whole image (for val / inference)."""

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self._read_image(image_id)
        msk = self._read_mask(image_id)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]
        msk = msk.long()
        return img, msk


class SteelDefectMultiCropDataset(BaseDefectDataset):
    """n random defect-aware crops per image (training)."""

    def __init__(
        self,
        image_ids: List[str],
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.BasicTransform] = None,
        crop_size: int = 256,
        n_crops: int = 2,
    ):
        super().__init__(image_ids, images_dir, masks_dir, transform)
        self.crop_size = crop_size
        self.n_crops = n_crops
        print("Pre-loading masks for multi-crop dataset …")
        for _ in tqdm(self.image_ids):
            pass  # placeholder, keep lazy loading

    def __len__(self) -> int:
        return len(self.image_ids) * self.n_crops

    def __getitem__(self, idx):
        real_idx, patch_idx = divmod(idx, self.n_crops)
        image_id = self.image_ids[real_idx]

        img_np = self._read_image(image_id)
        msk_1ch = self._read_mask(image_id)

        # bboxes = find_defect_bboxes(msk_1ch)
        bboxes = find_defect_bboxes(msk_1ch)
        bboxes.sort(key=lambda b: b[-1], reverse=True)

        if not bboxes:
            crop_img, crop_msk = random_crop_any(img_np, msk_1ch, self.crop_size)
        elif len(bboxes) >= self.n_crops:
            crop_img, crop_msk = random_crop_full_in_diff_pos(
                img_np, msk_1ch, bboxes[patch_idx], self.crop_size
            )
        else:  # fewer defects than n_crops → reuse
            crop_img, crop_msk = random_crop_full_in_diff_pos(
                img_np, msk_1ch, bboxes[patch_idx % len(bboxes)], self.crop_size
            )

        if self.transform:
            aug = self.transform(image=crop_img, mask=crop_msk)
            crop_img, crop_msk = aug["image"], aug["mask"]
        crop_msk = crop_msk.long()
        return crop_img, crop_msk


SteelValDataset = SteelDefectDataset  # alias


# ----------------------------------------------------------------------
# albumentations pipelines ──────────────────────────────────────────────
# ----------------------------------------------------------------------
_IMAGENET_STATS = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def get_training_augmentation() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(**_IMAGENET_STATS),
            ToTensorV2(),
        ]
    )



def get_validation_augmentation() -> A.Compose:
    """
    修改后的验证集数据增强。
    1. 使用 PadIfNeeded 智能填充到32的倍数。
    2. 标准化。
    3. 转换为Tensor。
    """
    return A.Compose(
        [
            # 新增：智能填充模块
            A.PadIfNeeded(
                min_height=None,          # 我们不指定固定的最小高度
                min_width=None,           # 也不指定固定的最小宽度
                pad_height_divisor=32,    # 保证高度是32的倍数
                pad_width_divisor=32,     # 保证宽度是32的倍数
                border_mode=cv2.BORDER_CONSTANT, # 填充模式
                value=0,                  # 用黑色(0)填充图片
                mask_value=0              # 用背景(0)填充掩膜
            ),
            # 原有的处理流程
            A.Normalize(**_IMAGENET_STATS),
            ToTensorV2(),
        ]
    )





# import os
# import random
# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from typing import List, Tuple
#
# # --- 1. 配置参数 ---
#
# # 请根据您的目录结构确认这些路径
# VAL_IMAGES_DIR = r'D:\PythonProjects\defect_detect\dataset\phone\val\val_images'
# VAL_MASKS_DIR = r'D:\PythonProjects\defect_detect\dataset\phone\val\val_masks'
#
# # 可视化设置
# CROP_SIZE = 256  # 要可视化的裁剪尺寸，应与训练时一致
# N_IMAGES_TO_SHOW = 5  # 从验证集中随机选择几张图片进行可视化
# N_CROPS_PER_IMAGE = 3  # 为每张图片生成几个不同的裁剪示例
#
#
# # --- 2. 完全复刻您代码中的核心函数 ---
# # (!!! 保证这部分与您提供的代码一字不差 !!!)
#
#
# # --- 辅助函数 ---
# def _read_image(image_path: str) -> np.ndarray:
#     return np.array(Image.open(image_path).convert("RGB"))
#
#
# def _read_mask(mask_path: str) -> np.ndarray:
#     npy = np.load(mask_path)
#     if npy.ndim == 3 and npy.shape[2] > 1:
#         return npy.argmax(axis=2).astype(np.uint8)
#     return npy.astype(np.uint8)
#
#
# #--- 3. 可视化主函数 ---
#
# def visualize_cropping_for_image(image_id: str):
#     """为单个图像创建并显示裁剪可视化。"""
#     image_path = os.path.join(VAL_IMAGES_DIR, image_id)
#     mask_path = os.path.join(VAL_MASKS_DIR, image_id.replace('.jpg', '.npy'))
#
#     if not os.path.exists(image_path) or not os.path.exists(mask_path):
#         print(f"警告: 找不到文件对 {image_id}，跳过。")
#         return
#
#     original_image = _read_image(image_path)
#     mask_1ch = _read_mask(mask_path)
#
#     bboxes = find_defect_bboxes(mask_1ch)
#     bboxes.sort(key=lambda b: b[-1], reverse=True)
#
#     fig, axes = plt.subplots(N_CROPS_PER_IMAGE, 2, figsize=(10, 5 * N_CROPS_PER_IMAGE))
#     fig.suptitle(f'Cropping Visualization for: {image_id}', fontsize=16)
#
#     for i in range(N_CROPS_PER_IMAGE):
#         ax_left, ax_right = axes[i] if N_CROPS_PER_IMAGE > 1 else axes
#
#         img_with_boxes = original_image.copy()
#
#         # --- 核心步骤 1: 完全按照您的逻辑调用函数来获得裁剪结果 ---
#         cropped_image = None
#         target_box = None
#
#         if not bboxes:
#             cropped_image, _ = random_crop_any(original_image, mask_1ch, CROP_SIZE)
#             ax_left.set_title(f'Crop {i + 1}: No Defects (random_crop_any)')
#         else:
#             target_box = bboxes[i % len(bboxes)]
#             cropped_image, _ = random_crop_full_in_diff_pos(original_image, mask_1ch, target_box, CROP_SIZE)
#             ax_left.set_title(f'Crop {i + 1}: Targeting Defect (yellow)')
#
#         # --- 核心步骤 2: 反向查找裁剪位置，而不修改原始逻辑 ---
#         # 使用模板匹配找到 cropped_image 在 original_image 中的位置
#         result = cv2.matchTemplate(original_image, cropped_image, cv2.TM_CCOEFF_NORMED)
#         _, _, _, max_loc = cv2.minMaxLoc(result)
#         crop_left, crop_top = max_loc
#
#         # --- 核心步骤 3: 根据找到的位置进行可视化 ---
#         if target_box:
#             # 绘制所有缺陷框（蓝色）
#             for x, y, w, h, _ in bboxes:
#                 cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             # 高亮目标缺陷框（黄色）
#             x, y, w, h, _ = target_box
#             cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 255, 0), 3)
#
#         # 绘制找到的裁剪区域（红色）
#         cv2.rectangle(img_with_boxes, (crop_left, crop_top), (crop_left + CROP_SIZE, crop_top + CROP_SIZE), (255, 0, 0),
#                       3)
#
#         # 显示图像
#         ax_left.imshow(img_with_boxes)
#         ax_left.set_xticks([])
#         ax_left.set_yticks([])
#         ax_right.imshow(cropped_image)
#         ax_right.set_title(f'Resulting Patch ({CROP_SIZE}x{CROP_SIZE})')
#         ax_right.set_xticks([])
#         ax_right.set_yticks([])
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()
#
#
# # --- 4. 执行可视化 ---
# if __name__ == '__main__':
#     if not os.path.exists(VAL_IMAGES_DIR):
#         print(f"错误：找不到验证集图片目录 '{VAL_IMAGES_DIR}'")
#     else:
#         validation_ids = [f for f in os.listdir(VAL_IMAGES_DIR) if f.lower().endswith('.jpg')]
#         if len(validation_ids) == 0:
#             print("错误：在验证集目录中没有找到.jpg图片。")
#         else:
#             if len(validation_ids) < N_IMAGES_TO_SHOW:
#                 print(f"警告：验证集图片数量 ({len(validation_ids)}) 少于要求显示的数量 ({N_IMAGES_TO_SHOW})。")
#                 sample_ids = validation_ids
#             else:
#                 sample_ids = random.sample(validation_ids, N_IMAGES_TO_SHOW)
#
#             print(f"将为以下 {len(sample_ids)} 张图片进行可视化: {sample_ids}")
#             for image_id in sample_ids:
#                 visualize_cropping_for_image(image_id)
