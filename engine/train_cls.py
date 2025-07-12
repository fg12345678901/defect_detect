# engine/train_cls.py
"""
Image classification trainer for multi-category classification.

Usage Example:
python -m engine.train_cls \
    --data_dir dataset/ \
    --epochs 20 \
    --batch 8 \
    --run_name Classify_test
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- Local Module Imports ---
from models.classifier import create_classifier
from losses.cls_loss import get_cls_loss
from utils.metrics import evaluate_classifier


# ----------------------------------------------------------------------
# Custom Dataset Class (Remains here for data handling)
# ----------------------------------------------------------------------
class CustomImageDataset(Dataset):
    """
    Custom dataset for a structure like:
    root/class_x/train/train_images/xxx.png
    root/class_x/val/val_images/xxy.png
    """

    def __init__(self, root_dir: str, split: str, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            image_dir = self.root_dir / class_name / self.split / f"{self.split}_images"
            if not image_dir.is_dir():
                print(f"Warning: Directory not found, skipping: {image_dir}")
                continue
            for img_path in image_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ----------------------------------------------------------------------
# Data Loading (Remains here for data handling)
# ----------------------------------------------------------------------
def build_loaders(root_dir: str, batch: int, img_size: int = 380):
    """Create dataloaders using our CustomImageDataset."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    t_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = CustomImageDataset(root_dir, split='train', transform=t_train)
    val_ds = CustomImageDataset(root_dir, split='val', transform=t_val)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("One of the datasets is empty. Check your --data_dir path and folder structure.")
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=os.cpu_count() // 2,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=os.cpu_count() // 2, pin_memory=True)
    return train_loader, val_loader


# ----------------------------------------------------------------------
# Main Training Logic
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root directory of the dataset (e.g., 'dataset/')")
    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    ap.add_argument("--batch", type=int, default=8, help="Batch size (reduce if you run out of VRAM)")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--run_name", type=str, default=None, help="Name for the training run (for logging)")
    args = ap.parse_args()

    # --- Setup ---
    if args.run_name is None:
        args.run_name = Path(args.data_dir).name + "_efficientnet_b4"
    log_dir = Path("logs") / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    checkpoint_path = log_dir / "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data, Model, Loss, Optimizer ---
    train_loader, val_loader = build_loaders(args.data_dir, args.batch)
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    print(f"Training with {len(train_loader.dataset)} images, validating with {len(val_loader.dataset)} images.")

    model = create_classifier(num_classes).to(device)
    criterion = get_cls_loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep:02d}/{args.epochs} [Training]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        train_loss = running_loss / len(train_loader)

        # --- Validation & Logging ---
        val_loss, val_f1, report_dict = evaluate_classifier(model, val_loader, criterion, device, class_names)

        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalar("loss/val", val_loss, ep)
        writer.add_scalar("F1_Score/macro", val_f1, ep)

        for class_name, metrics in report_dict.items():
            if class_name in class_names:
                writer.add_scalar(f"Recall/{class_name}", metrics['recall'], ep)
                writer.add_scalar(f"Precision/{class_name}", metrics['precision'], ep)
                writer.add_scalar(f"F1_Score/{class_name}", metrics['f1-score'], ep)

        print(f"\n[{ep:03d}/{args.epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_F1={val_f1:.4f}")
        print("  " + "-" * 50)
        for class_name in class_names:
            metrics = report_dict[class_name]
            print(f"  - {class_name:<15} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f}")
        print("  " + "-" * 50)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↳ Val F1 improved to {best_f1:.4f}. Saving model to {checkpoint_path}")

    writer.close()
    print("\nTraining finished.")


if __name__ == "__main__":
    main()
