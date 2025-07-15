#!/usr/bin/env python
# solar_panel_prepare.py
# ----------------------
# 将 BenchmarkELimages 之 dataset_20221008 → 统一 one-hot 掩膜格式
# author: <you>

import os, shutil, csv, random
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------- 1. 路径 & 参数 ----------
RAW_ROOT_DIR = 'dataset/solar-panel_tmp/BenchmarkELimages-main/dataset_20221008'
OUT_DIR      = 'dataset/solar-panel'

VAL_SIZE     = 0.2
RANDOM_STATE = 42

CSV_PATH = os.path.join(RAW_ROOT_DIR, 'ListOfClassesAndColorCodes_20221008.csv')

# ---------- 2. 解析 CSV → 映射 ----------
value2label = {}   # 像素值(Gray 或 Label) → class_id
color_map   = {}   # class_id → (R,G,B)

with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = int(row['Label'])
        gray  = int(row['Gray'])
        r, g, b = int(row['Red']), int(row['Green']), int(row['Blue'])
        value2label[label] = label
        value2label[gray]  = label
        color_map[label] = (r, g, b)

NUM_CLASSES = max(value2label.values()) + 1  # 29

# ---------- 3. 汇总全部样本 ----------
def collect(split):
    img_dir  = os.path.join(RAW_ROOT_DIR, f'el_images_{split}')
    mask_dir = os.path.join(RAW_ROOT_DIR, f'el_masks_{split}')
    pairs = []
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path  = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname.rsplit('.', 1)[0] + '.png')
        if not os.path.isfile(mask_path):
            print(f'[WARN] mask missing: {mask_path}')
            continue
        pairs.append({'img': img_path, 'mask': mask_path})
    return pairs

samples = collect('train') + collect('val') + collect('test')
print(f'Collected {len(samples)} total images.')

# ---------- 4. 重新划分 train / val ----------
train_set, val_set = train_test_split(
    samples, test_size=VAL_SIZE,
    random_state=RANDOM_STATE, shuffle=True
)
print(f'Train: {len(train_set)},  Val: {len(val_set)}')
splits = {'train': train_set, 'val': val_set}

# ---------- 5. 工具函数 ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def process(split_name, split_samples):
    img_out  = os.path.join(OUT_DIR, split_name, f'{split_name}_images')
    msk_out  = os.path.join(OUT_DIR, split_name, f'{split_name}_masks')
    vis_out  = os.path.join(OUT_DIR, split_name, f'{split_name}_masks_visualization')
    for d in (img_out, msk_out, vis_out): ensure_dir(d)

    print(f'\nProcessing {split_name} set …')
    for item in tqdm(split_samples):
        img_np = np.asarray(Image.open(item["img"]).convert('RGB'))
        mask_np_raw = np.asarray(Image.open(item["mask"]).convert('L'))

        H, W = mask_np_raw.shape
        mask_onehot = np.zeros((H, W, NUM_CLASSES), dtype=np.uint8)

        # 转 one-hot
        for v in np.unique(mask_np_raw):
            cls = value2label.get(int(v), None)
            if cls is None:      # 未知像素值
                continue
            mask_onehot[:, :, cls] = (mask_np_raw == v).astype(np.uint8)

        # 背景保证完整
        defect_any = np.any(mask_onehot[:, :, 1:], axis=2)
        mask_onehot[:, :, 0] = (~defect_any).astype(np.uint8)

        base = os.path.splitext(os.path.basename(item["img"]))[0]
        # 1. 原图
        shutil.copy(item["img"], os.path.join(img_out, base + '.png'))
        # 2. 掩膜
        np.save(os.path.join(msk_out,  base + '.npy'), mask_onehot)
        # 3. 可视化
        color_mask = np.zeros_like(img_np)
        for cls in range(1, NUM_CLASSES):
            color = color_map.get(cls, (255, 255, 255))
            color_mask[mask_onehot[:, :, cls] == 1] = color
        overlay = cv2.addWeighted(img_np, 0.7, color_mask, 0.3, 0)
        cv2.imwrite(os.path.join(vis_out, base + '.png'),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# # ---------- 6. 清空旧目录并执行 ----------
# for sub in ('train', 'val'):
#     dir_to_rm = os.path.join(OUT_DIR, sub)
#     if os.path.exists(dir_to_rm): shutil.rmtree(dir_to_rm)

for name, data in splits.items():
    process(name, data)

print('\nDone! Converted dataset saved to:', OUT_DIR)
