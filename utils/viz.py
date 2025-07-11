# utils/viz.py
"""
可视化辅助：
‣ visualize_prediction_3d —— 原图叠加彩色缺陷掩膜
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Optional, Dict, Tuple

# 预设的颜色集保持不变
TASK_COLOR_MAPS = {
    "steel": {
        0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0),
    },
    "phone": {
        0: (0, 0, 0), 1: (128, 0, 0), 2: (128, 128, 0), 3: (0, 128, 0),
    },
    "magnetic": {
        0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0), 5: (128, 0, 255),
    }
}


def visualize_prediction_3d(
    image_np: np.ndarray,
    pred_mask_3d: np.ndarray,
    image_name: str,
    save_dir: str,
    task: str = "steel",
    alpha: float = 0.3,
    color_map_override: Optional[Dict[int, Tuple[int, int, int]]] = None, # <--- 新增参数
):
    """
    根据指定的任务（task），使用不同的颜色集进行可视化。
    如果提供了 color_map_override，则优先使用它。
    """
    os.makedirs(save_dir, exist_ok=True)
    h, w, _ = image_np.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # --- 核心修改：智能选择颜色集 ---
    if color_map_override is not None:
        color_map = color_map_override
    else:
        # 如果没有提供覆盖字典，则使用内部预设
        color_map = TASK_COLOR_MAPS.get(task, TASK_COLOR_MAPS["steel"])
    # --------------------------------

    num_classes = pred_mask_3d.shape[0]
    for c in range(1, num_classes):
        color = color_map.get(c, (255, 255, 255))
        overlay[pred_mask_3d[c] > 0] = color

    blended = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(
        os.path.join(save_dir, image_name),
        cv2.cvtColor(blended, cv2.COLOR_RGB2BGR),
    )