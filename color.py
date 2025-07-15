# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import collections
#
# # --- 配置 ---
#
# # 指向您存放原始 .png 格式真值图的目录
# # 这是我们第一次处理数据时用的原始输入目录
# GROUND_TRUTH_DIR = 'dataset/phone/raw/ground_truth'
#
# # 您的缺陷类别和文件名前缀的对应关系
# # 注意: 'Scr' 对应 'scratch', 'Sta' 对应 'stain'
# DEFECT_TYPES_MAP = {
#     'Oil': 'oil',
#     'Scr': 'scratch',
#     'Sta': 'stain'
# }
#
# # 我们要忽略的背景色，通常是黑色
# BG_COLOR = (0, 0, 0)
#
#
# # --- 主逻辑 ---
#
# def analyze_mask_colors():
#     """
#     遍历所有真值掩膜图片，找出每种缺陷使用的精确RGB颜色。
#     """
#     if not os.path.isdir(GROUND_TRUTH_DIR):
#         print(f"错误：找不到目录 '{GROUND_TRUTH_DIR}'")
#         print("请确认路径是否正确，以及该目录下是否存放了原始的.png掩膜文件。")
#         return
#
#     # 使用集合来自动去重
#     colors_per_class = collections.defaultdict(set)
#     all_mask_files = [f for f in os.listdir(GROUND_TRUTH_DIR) if f.lower().endswith('.png')]
#
#     if not all_mask_files:
#         print(f"错误：在目录 '{GROUND_TRUTH_DIR}' 中没有找到任何.png文件。")
#         return
#
#     print(f"开始分析目录 '{GROUND_TRUTH_DIR}' 中的 {len(all_mask_files)} 个掩膜文件...")
#
#     for filename in tqdm(all_mask_files, desc="正在分析颜色"):
#         try:
#             img_path = os.path.join(GROUND_TRUTH_DIR, filename)
#             img = Image.open(img_path).convert("RGB")
#             img_np = np.array(img)
#
#             # 将 (H, W, 3) 的图像数组重塑为 (N, 3) 的像素列表
#             # 然后寻找唯一的行，即唯一的颜色
#             unique_colors = np.unique(img_np.reshape(-1, 3), axis=0)
#
#             # 从文件名中识别类别
#             file_class_key = filename.split('_')[0]
#             defect_type = DEFECT_TYPES_MAP.get(file_class_key)
#
#             if not defect_type:
#                 continue  # 如果文件名不符合预期，则跳过
#
#             for color in unique_colors:
#                 color_tuple = tuple(color)
#                 # 如果不是背景色，就记录下来
#                 if color_tuple != BG_COLOR:
#                     colors_per_class[defect_type].add(color_tuple)
#
#         except Exception as e:
#             print(f"\n处理文件 {filename} 时出错: {e}")
#
#     # --- 报告结果 ---
#     print("\n\n✅ --- 颜色分析报告 --- ✅")
#     print("在所有真值掩膜图片中找到了以下非背景颜色：\n")
#
#     has_found_any_color = False
#     for defect_type, color_set in colors_per_class.items():
#         if color_set:
#             has_found_any_color = True
#             print(f"🎨 缺陷类别 '{defect_type}':")
#             if len(color_set) > 1:
#                 print(f"  >>> 警告: 为此类缺陷找到了多种颜色，请确认! {color_set}")
#             else:
#                 print(f"  └── 精确颜色 (R, G, B): {list(color_set)[0]}")
#         else:
#             print(f"⚪️ 缺陷类别 '{defect_type}': 未找到任何非背景色!")
#
#     if not has_found_any_color:
#         print("\n‼️ 严重警告：未在任何掩膜中找到任何非背景(非黑)的颜色。")
#         print("这说明您的.png掩膜文件可能全是黑色的，请打开几张确认一下。")
#         return
#
#     print("\n\n📋 --- 建议的颜色映射字典 --- 📋")
#     print("请将以下字典完整复制到您【生成.npy的数据处理脚本】中，")
#     print("替换掉旧的 `COLOR_TO_CLASS_ID` 字典。\n")
#
#     # 按照 oil, scratch, stain 的顺序生成字典
#     defect_order = ['oil', 'scratch', 'stain']
#
#     print("COLOR_TO_CLASS_ID = {")
#     for i, defect_type in enumerate(defect_order):
#         class_id = i + 1
#         color_set = colors_per_class.get(defect_type)
#
#         if color_set and len(color_set) == 1:
#             color_str = str(list(color_set)[0])
#             print(f"    {color_str}: {class_id},   # {defect_type}")
#         else:
#             # 如果没找到颜色或找到多种颜色，留空让用户手动填写
#             print(f"    # (请为 '{defect_type}' 手动填入正确颜色): {class_id},")
#     print("}")
#
#
# if __name__ == '__main__':
#     analyze_mask_colors()

'''颜色映射'''
# import os
# import csv
# import json
#
# # --- 1. 配置路径 (与您的 solar_panel_prepare.py 保持一致) ---
#
# # 指向包含原始CSV文件的目录
# RAW_DATA_DIR = 'dataset/solar-panel_tmp/BenchmarkELimages-main/dataset_20221008'
# CSV_PATH = os.path.join(RAW_DATA_DIR, 'ListOfClassesAndColorCodes_20221008.csv')
#
# # 定义最终的输出目录，我们将把 color_map.json 保存到这里
# # 这与您处理后的数据集目录一致，方便管理
# OUTPUT_DIR = 'dataset/solar-panel'
#
#
# def create_color_map_json():
#     """
#     读取 solar-panel 任务的颜色配置文件(CSV)，并生成一个 color_map.json 文件。
#     """
#     print(f"正在读取颜色配置文件: {CSV_PATH}")
#
#     # 检查CSV文件是否存在
#     if not os.path.exists(CSV_PATH):
#         print(f"错误：找不到CSV文件 '{CSV_PATH}'")
#         print("请确认您的原始数据集路径是否正确。")
#         return
#
#     # 初始化颜色字典，并手动添加所有任务通用的背景类
#     # 背景(class 0) 通常是黑色的
#     color_map = {0: (0, 0, 0)}
#
#     # --- 2. 解析CSV文件 (此逻辑与您的脚本完全相同) ---
#     try:
#         with open(CSV_PATH, mode='r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 label = int(row['Label'])
#                 r = int(row['Red'])
#                 g = int(row['Green'])
#                 b = int(row['Blue'])
#                 color_map[label] = (r, g, b)
#
#         # -1 是因为我们手动加了背景类
#         print(f"成功解析了 {len(color_map) - 1} 个缺陷类别的颜色。")
#
#     except Exception as e:
#         print(f"读取或解析CSV时出错: {e}")
#         return
#
#     # --- 3. 保存为 JSON 文件 ---
#     # 确保输出目录存在
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     save_path = os.path.join(OUTPUT_DIR, 'color_map.json')
#
#     try:
#         # 将字典写入JSON文件
#         with open(save_path, 'w', encoding='utf-8') as f:
#             # 使用 indent=4 参数使JSON文件格式更美观，易于阅读
#             # sort_keys=True 保证每次生成的json文件键的顺序都一样
#             json.dump(color_map, f, indent=4, sort_keys=True)
#
#         print(f"\n✅ 成功！颜色映射文件已保存到: {save_path}")
#
#     except Exception as e:
#         print(f"\n❌ 保存JSON文件时出错: {e}")
#
#
# if __name__ == "__main__":
#     create_color_map_json()


import os
import shutil
import random
from pathlib import Path

# --- 配置区 ---
# 源数据集的根目录
SOURCE_ROOT = Path("dataset")
# 新测试集的目标目录
DEST_DIR = SOURCE_ROOT / "Classify_test"
# 你要处理的所有类别名称
CATEGORIES = ["steel", "phone", "magnetic", "solar-panel"]
# 每个类别要抽取的图片数量
NUM_IMAGES_PER_CLASS = 50


def create_test_dataset():
    """
    从每个类别的验证集中随机抽取图片，复制并重命名，
    以创建一个新的、统一的测试集。
    """
    print(f"开始创建测试集...")
    print(f"目标文件夹: {DEST_DIR}")

    # 1. 创建目标文件夹，如果不存在的话
    DEST_DIR.mkdir(exist_ok=True)

    total_copied_count = 0

    # 2. 遍历每个类别
    for category in CATEGORIES:
        print(f"\n处理类别: '{category}'")

        # 构建源验证集图片的路径
        source_val_dir = SOURCE_ROOT / category / "val" / "val_images"

        # 检查源路径是否存在
        if not source_val_dir.is_dir():
            print(f"  [警告] 源文件夹未找到, 跳过: {source_val_dir}")
            continue

        # 3. 获取源文件夹下所有图片文件 (.jpg 和 .png)
        try:
            image_files = [f for f in source_val_dir.iterdir() if f.suffix.lower() in ['.jpg', '.png']]
        except FileNotFoundError:
            print(f"  [警告] 无法访问文件夹, 跳过: {source_val_dir}")
            continue

        if not image_files:
            print(f"  [警告] 在 {source_val_dir} 中未找到任何图片, 跳过。")
            continue

        # 4. 随机抽取指定数量的图片
        if len(image_files) < NUM_IMAGES_PER_CLASS:
            print(f"  [警告] 图片数量 ({len(image_files)}) 少于期望的 {NUM_IMAGES_PER_CLASS} 张, 将复制所有图片。")
            selected_files = image_files
        else:
            selected_files = random.sample(image_files, NUM_IMAGES_PER_CLASS)

        # 5. 复制并重命名文件
        copied_count_per_class = 0
        for i, src_path in enumerate(selected_files, start=1):
            # 获取文件后缀 (例如: .jpg)
            extension = src_path.suffix
            # 创建新文件名 (例如: steel_0001.jpg)
            new_filename = f"{category}_{i:04d}{extension}"
            dest_path = DEST_DIR / new_filename

            # 执行复制操作 (shutil.copy2 会同时复制元数据)
            shutil.copy2(src_path, dest_path)
            copied_count_per_class += 1

        print(f"  成功复制 {copied_count_per_class} 张图片到测试集。")
        total_copied_count += copied_count_per_class

    print("\n" + "=" * 40)
    print("✅ 脚本执行完毕!")
    print(f"总共复制了 {total_copied_count} 张图片。")
    print(f"新的测试集已在以下路径准备就绪: {DEST_DIR}")
    print("=" * 40)


if __name__ == "__main__":
    create_test_dataset()