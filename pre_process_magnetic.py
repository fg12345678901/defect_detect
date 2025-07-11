#!/usr/bin/env python
# magnetic_prepare.py
# -------------------
# 将 Magnetic-tile-defect-datasets.-master 转成
#  (H, W, 6) one-hot 掩膜 (.npy) +  原图 (.jpg) + 可视化 (.jpg)
# author: <you>
# -------------------------------------------------------------

import os, shutil, random
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------- 1. 路径与参数 ----------
BASE_DIR      = 'dataset/magnetic'   # 输出根目录
RAW_ROOT_DIR  = os.path.join(
    'dataset',
    'magnetic_tmp',                  # 你解压出来的临时目录
    'Magnetic-tile-defect-datasets.-master'
)

# 需要处理的子目录及其在 one-hot 掩膜中的通道编号
# 通道 0 = 背景
DEFECT_DIRS = {
    'MT_Blowhole/Imgs': 1,
    'MT_Break/Imgs'   : 2,
    'MT_Crack/Imgs'   : 3,
    'MT_Fray/Imgs'    : 4,
    'MT_Uneven/Imgs'  : 5,
    # 'MT_Free'  : 6,   # 如果想把“无缺陷”算一类就打开
}

VAL_SIZE      = 0.2     # 验证集占比
RANDOM_STATE  = 42

COLOR_MAP = {           # 用于可视化上色
    1: (255,   0,   0), # Blowhole  → 红
    2: (  0, 255,   0), # Break     → 绿
    3: (  0,   0, 255), # Crack     → 蓝
    4: (255, 255,   0), # Fray      → 黄
    5: (128,   0, 255), # Uneven    → 紫
}

# ---------- 2. 收集样本 ----------
print('Scanning dataset ...')
samples = []   # [{'img':..., 'mask':..., 'cls':class_id}, ...]

for folder, cls_id in DEFECT_DIRS.items():
    folder_dir = os.path.join(RAW_ROOT_DIR, folder)
    if not os.path.isdir(folder_dir):
        print(f'  [WARN] dir not found: {folder_dir}')
        continue

    for fname in os.listdir(folder_dir):
        if not fname.lower().endswith('.jpg'):
            continue
        img_path  = os.path.join(folder_dir, fname)
        base      = os.path.splitext(fname)[0]
        mask_path = os.path.join(folder_dir, base + '.png')   # 与图片同名 .png

        if not os.path.isfile(mask_path):          # 无缺陷或 MT_Free
            mask_path = None

        samples.append({'img': img_path,
                        'mask': mask_path,
                        'cls' : cls_id})

print(f'Collected {len(samples)} images.')

# ---------- 3. 训练 / 验证划分 ----------
train_set, val_set = train_test_split(
    samples, test_size=VAL_SIZE,
    random_state=RANDOM_STATE, shuffle=True
)
splits = {'train': train_set, 'val': val_set}
print(f'Train: {len(train_set)},  Val: {len(val_set)}')

# ---------- 4. 核心处理 ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process(split_name, split_samples):
    img_out   = os.path.join(BASE_DIR, split_name, f'{split_name}_images')
    mask_out  = os.path.join(BASE_DIR, split_name, f'{split_name}_masks')
    vis_out   = os.path.join(BASE_DIR, split_name, f'{split_name}_masks_visualization')
    for d in (img_out, mask_out, vis_out):
        ensure_dir(d)

    print(f'\nProcessing {split_name} set ...')
    for item in tqdm(split_samples):
        img_src = item['img']
        mask_src= item['mask']          # 可能为 None
        cls_id  = item['cls']
        fname   = os.path.basename(img_src)
        base, _ = os.path.splitext(fname)

        # ---------- 读取原图 ----------
        img = Image.open(img_src).convert('RGB')
        img_np = np.asarray(img)
        H, W = img_np.shape[:2]

        # ---------- 构造 one-hot 掩膜 ----------
        num_classes = max(DEFECT_DIRS.values()) + 1  # 背景+缺陷
        mask_np = np.zeros((H, W, num_classes), dtype=np.uint8)

        if mask_src and os.path.isfile(mask_src):
            mask_bin = np.array(Image.open(mask_src).convert('L'))  # 0/255
            mask_bin = (mask_bin > 0).astype(np.uint8)
            mask_np[:, :, cls_id] = mask_bin

        # 背景通道
        defect_any = np.any(mask_np[:, :, 1:], axis=2)
        mask_np[:, :, 0] = (~defect_any).astype(np.uint8)

        # ---------- 保存 ----------
        shutil.copy(img_src, os.path.join(img_out, fname))
        np.save(os.path.join(mask_out, base + '.npy'), mask_np)

        # ---------- 可视化 ----------
        color_mask = np.zeros_like(img_np)
        for c in range(1, num_classes):
            color = COLOR_MAP.get(c, (255,255,255))
            color_mask[mask_np[:,:,c]==1] = color
        overlay = cv2.addWeighted(img_np, 0.7, color_mask, 0.3, 0)
        cv2.imwrite(os.path.join(vis_out, fname),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# -------- 5. 执行 --------
for split_name, split_samples in splits.items():
    process(split_name, split_samples)

print('\nAll done! Dataset prepared at:', BASE_DIR)
