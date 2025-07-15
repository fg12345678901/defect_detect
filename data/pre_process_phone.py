import os
import shutil
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

# --- 1. 配置参数 ---

# 基础路径设置
BASE_DIR = 'dataset/phone'
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw')
OUTPUT_DIR = BASE_DIR  # 输出到 dataset/phone/ 下

# 缺陷类别定义 (这决定了处理的顺序和通道的顺序)
# 第0通道是背景, 第1是oil, 第2是scratch, 第3是stain
DEFECT_TYPES = ['oil', 'scratch', 'stain']

# !! 关键 !!: 根据颜色分析报告更新此字典
COLOR_TO_CLASS_ID = {
    (128, 0, 0): 1,   # 暗红色 -> oil (类别1)
    (128, 128, 0): 2, # 暗黄色 -> scratch (类别2)
    (0, 128, 0): 3,   # 暗绿色 -> stain (类别3)
}

# 数据集划分比例
VAL_SIZE = 0.2
RANDOM_STATE = 42  # 为了每次划分结果都一样，方便复现

# --- 2. 收集所有文件路径 ---

print("开始收集所有原始图片和真值文件...")
all_files = []
raw_image_dir = RAW_DATA_DIR
ground_truth_dir = os.path.join(RAW_DATA_DIR, 'ground_truth')

for defect_idx, defect_name in enumerate(DEFECT_TYPES):
    defect_class_id = defect_idx + 1
    image_folder = os.path.join(raw_image_dir, defect_name)

    if not os.path.isdir(image_folder):
        print(f"警告: 找不到目录 {image_folder}")
        continue

    for image_filename in os.listdir(image_folder):
        if image_filename.lower().endswith('.jpg'):
            # 构建路径
            raw_image_path = os.path.join(image_folder, image_filename)
            gt_filename = image_filename.replace('.jpg', '.png')
            ground_truth_path = os.path.join(ground_truth_dir, gt_filename)

            # 确认真值文件存在
            if os.path.exists(ground_truth_path):
                all_files.append({
                    'raw_path': raw_image_path,
                    'gt_path': ground_truth_path,
                    'filename': os.path.splitext(image_filename)[0]  # e.g., "Oil_0001"
                })
            else:
                print(f"警告: 找不到对应的真值文件 {ground_truth_path}")

print(f"成功收集到 {len(all_files)} 个文件对。")

# --- 3. 划分训练集和验证集 ---

print(f"按 {1 - VAL_SIZE:.0%}:{VAL_SIZE:.0%} 的比例划分数据集...")
train_files, val_files = train_test_split(all_files, test_size=VAL_SIZE, random_state=RANDOM_STATE)
print(f"训练集数量: {len(train_files)}")
print(f"验证集数量: {len(val_files)}")


# --- 4. 定义核心处理函数 ---

def process_and_save_data(file_list, dataset_type):
    """
    处理文件列表并保存到指定的数据集目录 (train 或 val)

    参数:
        file_list (list): 包含文件路径字典的列表
        dataset_type (str): 'train' 或 'val'
    """
    # 创建输出目录
    image_out_dir = os.path.join(OUTPUT_DIR, dataset_type, f'{dataset_type}_images')
    mask_out_dir = os.path.join(OUTPUT_DIR, dataset_type, f'{dataset_type}_masks')
    vis_out_dir = os.path.join(OUTPUT_DIR, dataset_type, f'{dataset_type}_masks_visualization')

    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(vis_out_dir, exist_ok=True)

    print(f"\n开始处理 {dataset_type} 数据集...")
    for item in tqdm(file_list, desc=f"处理 {dataset_type} 集"):
        try:
            # 读取原始图和真值图
            raw_image = Image.open(item['raw_path']).convert("RGB")
            gt_image = Image.open(item['gt_path']).convert("RGB")

            raw_image_np = np.array(raw_image)
            gt_image_np = np.array(gt_image)

            H, W, _ = raw_image_np.shape

            # --- 生成 H*W*4 的 NumPy 掩膜 ---
            # 通道: 0=背景, 1=oil, 2=scratch, 3=stain
            num_classes = len(DEFECT_TYPES) + 1
            mask_np = np.zeros((H, W, num_classes), dtype=np.uint8)

            for color, class_id in COLOR_TO_CLASS_ID.items():
                # 找到当前颜色在真值图中的所有像素
                # np.all会检查最后一个维度（颜色通道）是否完全匹配
                class_mask = np.all(gt_image_np == color, axis=-1)
                mask_np[:, :, class_id] += class_mask.astype(np.uint8) # 使用 += 防止万一有重叠

            # 设置背景通道: 如果所有缺陷类别都为0, 则背景为1
            defects_mask_combined = np.any(mask_np[:, :, 1:] > 0, axis=2)
            mask_np[:, :, 0] = np.where(defects_mask_combined, 0, 1)

            # --- 保存处理结果 ---
            filename = item['filename']

            # 1. 复制原始图片
            shutil.copy(item['raw_path'], os.path.join(image_out_dir, f"{filename}.jpg"))

            # 2. 保存 NumPy 掩膜数组
            np.save(os.path.join(mask_out_dir, f"{filename}.npy"), mask_np)

            # 3. 生成并保存可视化图片
            # gt_image_np 本身就是彩色的掩膜，可以直接使用
            overlay = cv2.addWeighted(raw_image_np, 0.7, gt_image_np, 0.3, 0)
            # OpenCV 使用 BGR 格式保存，需要转换
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(vis_out_dir, f"{filename}.jpg"), overlay_bgr)

        except Exception as e:
            print(f"处理文件 {item['filename']} 时出错: {e}")


# --- 5. 执行处理流程 ---

# 在开始前，确保旧数据被清除 (或者手动删除)
print("正在清除旧的 train/val 目录 (如果存在)...")
if os.path.exists(os.path.join(BASE_DIR, 'train')):
    shutil.rmtree(os.path.join(BASE_DIR, 'train'))
if os.path.exists(os.path.join(BASE_DIR, 'val')):
    shutil.rmtree(os.path.join(BASE_DIR, 'val'))
print("旧目录已清除。")


process_and_save_data(train_files, 'train')
process_and_save_data(val_files, 'val')

print("\n所有处理完成！")
print("数据已成功保存到以下目录结构:")
print(f"- {os.path.join(OUTPUT_DIR, 'train')}")
print(f"  - train_images/")
print(f"  - train_masks/")
print(f"  - train_masks_visualization/")
print(f"- {os.path.join(OUTPUT_DIR, 'val')}")
print(f"  - val_images/")
print(f"  - val_masks/")
print(f"  - val_masks_visualization/")


# import numpy as np
# import os
#
# # --- 请在这里修改为您要查看的 .npy 文件的实际路径 ---
# # 这是根据我们之前脚本生成的示例路径
# file_path = 'dataset/steel/train/train_masks/0a1cade03.npy'
# # ---------------------------------------------------------
#
# # 设置打印选项，防止因数组过大而省略内容（可选项）
# # threshold=np.inf 表示不省略任何元素
# # np.set_printoptions(threshold=np.inf)
#
# # 检查文件是否存在
# if not os.path.exists(file_path):
#     print(f"错误：找不到文件 '{file_path}'")
#     print("请确认文件路径是否正确，以及之前的处理脚本是否已成功运行。")
# else:
#     try:
#         # 使用 np.load() 加载 .npy 文件
#         data = np.load(file_path)
#
#         # 打印相关信息
#         print(f"成功加载文件: '{file_path}'")
#         print("-" * 30)
#
#         # 1. 打印数组的形状 (非常重要)
#         print(f"数组的形状 (Shape): {data.shape}")
#
#         # 2. 打印数组的数据类型
#         print(f"数组的数据类型 (Data Type): {data.dtype}")
#
#         print("-" * 30)
#
#         # 3. 打印数组的完整内容
#         print("数组内容:")
#         print(data)
#
#     except Exception as e:
#         print(f"加载或打印文件时发生错误: {e}")