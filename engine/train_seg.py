# # engine/train_seg.py
# from __future__ import annotations
# import argparse, os, numpy as np
# import torch, torch.optim as optim
# from pathlib import Path
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from tqdm import tqdm
#
# # ⚠ 绝对包路径
# from data.steel import (
#     SteelDefectMultiCropDataset,
#     SteelValDataset,
#     get_training_augmentation,
#     get_validation_augmentation,
# )
# from losses.combined import CombinedLossWithContrast
# from models.segmenter import get_unet_model
#
#
# # ----------------------------------------------------------------------
# # sampler util ──────────────────────────────────────────────────────────
# # ----------------------------------------------------------------------
# def make_weights_balanced(image_ids, masks_dir, num_classes: int = 5):
#     """为每张整图算一个逆频权重，背景 = 0"""
#     class_count = [0] * num_classes
#     img_classes = []
#     for img in tqdm(image_ids, desc="scan masks"):
#         mask = np.load(
#             os.path.join(masks_dir, img.replace(".jpg", ".npy").replace(".png", ".npy"))
#         ).argmax(2)
#         uniq = np.unique(mask)
#         img_classes.append(uniq)
#         for c in uniq:
#             class_count[c] += 1
#
#     weights = [max(1.0 / class_count[c] for c in uniq) for uniq in img_classes]
#     return torch.DoubleTensor(weights)
#
#
# # ----------------------------------------------------------------------
# # train / val loop ──────────────────────────────────────────────────────
# # ----------------------------------------------------------------------
# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     losses = 0.0
#     for imgs, msks in tqdm(loader, leave=False):
#         imgs, msks = imgs.to(device), msks.to(device)
#         optimizer.zero_grad()
#         loss = criterion(model(imgs), msks)
#         loss.backward()
#         optimizer.step()
#         losses += loss.item()
#     return losses / len(loader)
#
#
# @torch.no_grad()
# def evaluate(model, loader, criterion, device):
#     model.eval()
#     losses = 0.0
#     for imgs, msks in tqdm(loader, leave=False):
#         imgs, msks = imgs.to(device), msks.to(device)
#         losses += criterion(model(imgs), msks).item()
#     return losses / len(loader)
#
#
# # ----------------------------------------------------------------------
# # main ──────────────────────────────────────────────────────────────────
# # ----------------------------------------------------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_dir", required=True)
#     ap.add_argument("--train_mask_dir", required=True)
#     ap.add_argument("--val_dir", required=True)
#     ap.add_argument("--val_mask_dir", required=True)
#     ap.add_argument("--epochs", type=int, default=200)
#     ap.add_argument("--batch", type=int, default=32)
#     ap.add_argument("--out", type=str, default="best_model.pth")
#     args = ap.parse_args()
#
#     # --- 新增代码：确保输出目录存在 ---
#     # 使用 pathlib 获取输出文件的父目录
#     output_dir = Path(args.out).parent
#     # 创建该目录，parents=True会一并创建所有上级目录，exist_ok=True表示如果目录已存在则不报错
#     output_dir.mkdir(parents=True, exist_ok=True)
#     # --------------------------------
#
#     train_imgs = [f for f in os.listdir(args.train_dir) if f.lower().endswith((".jpg", ".png"))]
#     val_imgs   = [f for f in os.listdir(args.val_dir)   if f.lower().endswith((".jpg", ".png"))]
#
#     N_CROPS = 3
#     train_ds = SteelDefectMultiCropDataset(
#         image_ids=train_imgs,
#         images_dir=args.train_dir,
#         masks_dir=args.train_mask_dir,
#         transform=get_training_augmentation(),
#         crop_size=256,
#         n_crops=N_CROPS,
#     )
#     val_ds = SteelValDataset(
#         image_ids=val_imgs,
#         images_dir=args.val_dir,
#         masks_dir=args.val_mask_dir,
#         transform=get_validation_augmentation(),
#     )
#
#     # ---------- 过采样逻辑已被注释掉 ----------
#     # w_img = make_weights_balanced(train_imgs, args.train_mask_dir)
#     # w_full = torch.repeat_interleave(w_img, repeats=N_CROPS)
#     # sampler = WeightedRandomSampler(w_full, num_samples=len(train_ds), replacement=True)
#
#     # 移除 sampler，并加入 shuffle=True 来随机化训练数据
#     train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
#                               num_workers=4, pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=args.batch // 4, shuffle=False,
#                               num_workers=4, pin_memory=True)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = get_unet_model("efficientnet-b4", "imagenet", 3, 4).to(device)
#
#     criterion = CombinedLossWithContrast()
#     optimiser = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
#     sched = optim.lr_scheduler.ReduceLROnPlateau(optimiser, "min", factor=0.5, patience=10)
#
#     best = 1e9
#     for ep in range(1, args.epochs + 1):
#         tr = train_one_epoch(model, train_loader, criterion, optimiser, device)
#         vl = evaluate(model, val_loader, criterion, device)
#         sched.step(vl)
#         print(f"[{ep:03d}/{args.epochs}] train={tr:.4f}  val={vl:.4f}")
#
#         if vl < best:
#             best = vl
#             torch.save(model.state_dict(), args.out)
#             print(f"  ↳ save best → {args.out}")
#
#     print("done.")
#
#
# if __name__ == "__main__":
#     main()


# engine/train_seg.py
"""
通用分割模型训练器 (支持 TensorBoard, 过采样, 续训, 可变类别数等功能)

使用示例
-----
# 训练手机模型 (4个类别)
python -m engine.train_seg `
       --run_name "phone_model_final" `
       --train_dir "dataset/phone/train/train_images" `
       --train_mask_dir "dataset/phone/train/train_masks" `
       --val_dir "dataset/phone/val/val_images" `
       --val_mask_dir "dataset/phone/val/val_masks" `
       --classes 4 `
       --epochs 200 `
       --batch 8 `
       --n_crops 3 `
       --lr 1e-4

# 训练钢铁模型 (5个类别，并启用过采样)
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
"""
from __future__ import annotations
import argparse, os, numpy as np
from datetime import datetime

import torch, torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ⚠ 绝对包路径
from data.steel import (
    SteelDefectMultiCropDataset,
    SteelValDataset,
    get_training_augmentation,
    get_validation_augmentation,
)
from losses.combined import CombinedLossWithContrast
from models.segmenter import get_unet_model
from utils.metrics import DiceMeter, IoUMeter  # <--- 从您提供的文件导入


# ----------------------------------------------------------------------
# sampler util (可选) ──────────────────────────────────────────────────
# ----------------------------------------------------------------------
def make_weights_balanced(image_ids, masks_dir, num_classes: int):
    """为每张整图算一个逆频权重，以实现过采样"""
    print("正在为过采样计算类别权重...")
    class_count = [0] * num_classes
    for img in tqdm(image_ids, desc="扫描掩膜"):
        mask = np.load(os.path.join(masks_dir, img.replace('.jpg', '.npy')))
        if mask.ndim == 3:
            mask = mask.argmax(axis=2)
        uniq = np.unique(mask)
        for c in uniq:
            if c < num_classes:
                class_count[c] += 1
    class_count = [max(c, 1) for c in class_count]  # 避免除以0

    weights = []
    for img in image_ids:
        mask = np.load(os.path.join(masks_dir, img.replace('.jpg', '.npy')))
        if mask.ndim == 3:
            mask = mask.argmax(axis=2)
        uniq = np.unique(mask)
        weights.append(max(1.0 / class_count[c] for c in uniq if c < num_classes))

    return torch.DoubleTensor(weights)


# ----------------------------------------------------------------------
# train / val loop ──────────────────────────────────────────────────────
# ----------------------------------------------------------------------
def run_one_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """运行一个 epoch (训练或验证)，并返回包含总损失和核心指标的字典。"""
    model.train(is_train)

    num_classes = model.unet.segmentation_head[0].out_channels
    meter_dice = DiceMeter(num_classes=num_classes)
    meter_iou = IoUMeter(num_classes=num_classes)
    total_loss_sum = 0.0

    progress_bar = tqdm(loader, desc="训练" if is_train else "验证", leave=False)
    for imgs, msks in progress_bar:
        imgs, msks = imgs.to(device), msks.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(imgs)
            total_loss = criterion(outputs, msks)

            if is_train:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        # --- 计算指标 ---
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        pred_mask_2d = logits.argmax(dim=1)
        pred_mask_chw = F.one_hot(pred_mask_2d, num_classes=num_classes).permute(0, 3, 1, 2).cpu().numpy()
        gt_mask_chw = F.one_hot(msks.long(), num_classes=num_classes).permute(0, 3, 1, 2).cpu().numpy()

        for i in range(pred_mask_chw.shape[0]):
            meter_dice.update(pred_mask_chw[i], gt_mask_chw[i])
            meter_iou.update(pred_mask_chw[i], gt_mask_chw[i])

        # --- 只累加总损失 ---
        total_loss_sum += total_loss.item()

    # --- 整合最终结果 ---
    final_metrics = {}
    final_metrics['total_loss'] = total_loss_sum / len(loader)
    _, final_metrics['dice_score'] = meter_dice.get_scores()
    _, final_metrics['iou_score'] = meter_iou.get_scores()

    return final_metrics


# ----------------------------------------------------------------------
# main ──────────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--train_mask_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--val_mask_dir", required=True)
    ap.add_argument("--run_name", type=str, default=None, help="为本次运行命名。如果留空，将自动生成带时间戳的名称。")
    ap.add_argument("--classes", type=int, default=4, help="模型输出类别数 (缺陷数+1个背景)")
    ap.add_argument("--n_crops", type=int, default=3, help="训练时每张图的裁剪数")
    ap.add_argument("--oversample", action='store_true', help="启用基于类别频率的过采样")
    ap.add_argument("--resume", type=str, default=None, help="预训练模型 .pth 文件的路径")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5, help="学习率")
    ap.add_argument("--crop_size", type=int, default=256, help="裁剪大小")
    args = ap.parse_args()

    if args.run_name is None:
        args.run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    log_dir = Path("logs") / args.run_name
    checkpoint_path = log_dir / "best_model.pth"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"所有日志和模型将保存在: {log_dir}")

    train_imgs = [f for f in os.listdir(args.train_dir) if f.lower().endswith((".jpg", ".png"))]
    val_imgs = [f for f in os.listdir(args.val_dir) if f.lower().endswith((".jpg", ".png"))]

    train_ds = SteelDefectMultiCropDataset(
        image_ids=train_imgs, images_dir=args.train_dir, masks_dir=args.train_mask_dir,
        transform=get_training_augmentation(), crop_size=args.crop_size, n_crops=args.n_crops,
    )
    val_ds = SteelValDataset(
        image_ids=val_imgs, images_dir=args.val_dir, masks_dir=args.val_mask_dir,
        transform=get_validation_augmentation(),
    )

    sampler = None
    if args.oversample:
        weights = make_weights_balanced(train_imgs, args.train_mask_dir, num_classes=args.classes)
        w_full = torch.repeat_interleave(weights, repeats=args.n_crops)
        sampler = WeightedRandomSampler(w_full, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              shuffle=(sampler is None), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}, 类别数: {args.classes}")
    model = get_unet_model("efficientnet-b4", "imagenet", 3, classes=args.classes).to(device)

    if args.resume:
        if os.path.exists(args.resume):
            print(f"正在从 {args.resume} 加载预训练权重...")
            model.load_state_dict(torch.load(args.resume, map_location=device))
        else:
            print(f"警告：找不到指定的权重文件 {args.resume}，将从头开始训练。")

    criterion = CombinedLossWithContrast()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10, verbose=True)

    best_metric = -1.0

    for ep in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, None, device, is_train=False)

        scheduler.step(val_metrics['total_loss'])

        print(f"[{ep:03d}/{args.epochs}] "
              f"Train -> Loss: {train_metrics['total_loss']:.4f}, Dice: {train_metrics['dice_score']:.4f}, IoU: {train_metrics['iou_score']:.4f} | "
              f"Val -> Loss: {val_metrics['total_loss']:.4f}, Dice: {val_metrics['dice_score']:.4f}, IoU: {val_metrics['iou_score']:.4f}")

        # --- TensorBoard 日志记录 (已精简) ---
        writer.add_scalar('Hyperparameters/Learning_Rate', optimizer.param_groups[0]['lr'], ep)
        writer.add_scalars('Loss/Total', {'train': train_metrics['total_loss'], 'val': val_metrics['total_loss']}, ep)
        writer.add_scalars('Metrics/Dice_Score',
                           {'train': train_metrics['dice_score'], 'val': val_metrics['dice_score']}, ep)
        writer.add_scalars('Metrics/IoU_Score', {'train': train_metrics['iou_score'], 'val': val_metrics['iou_score']},
                           ep)

        # --- 保存最佳模型 ---
        current_metric = val_metrics['dice_score']
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↳ Val Dice improved to {best_metric:.4f}. Saving best model to {checkpoint_path}")

    writer.close()
    print("训练完成。")


if __name__ == "__main__":
    main()