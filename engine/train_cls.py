"""engine/train_cls.py
Image classification trainer for defect type classification.

Usage Example:
python -m engine.train_cls \
    --train_dir dataset/steel/train/train_images \
    --val_dir dataset/steel/val/val_images \
    --epochs 20 \
    --batch 32

The directory passed to --train_dir and --val_dir should contain
subdirectories named after each category (e.g. steel, phone, magnetic,
solar-panel). Images can have arbitrary sizes; they will be resized to a
fixed size during training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score


# ----------------------------------------------------------------------
# util functions -------------------------------------------------------
# ----------------------------------------------------------------------
def build_loaders(train_dir: str, val_dir: str, batch: int, img_size: int = 224):
    """Create dataloaders using torchvision ImageFolder."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    t_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=t_train)
    val_ds = datasets.ImageFolder(val_dir, transform=t_val)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)


def evaluate(model, loader, criterion, device):
    """Evaluate model and return average loss and macro F1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), f1


# ----------------------------------------------------------------------
# main -----------------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--run_name", type=str, default=None)
    args = ap.parse_args()

    if args.run_name is None:
        args.run_name = Path(args.train_dir).stem + "_cls"

    log_dir = Path("logs") / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    checkpoint_path = log_dir / "best_model.pth"

    train_loader, val_loader, num_classes = build_loaders(
        args.train_dir, args.val_dir, args.batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        # ----- training -----
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # ----- validation -----
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalar("loss/val", val_loss, ep)
        writer.add_scalar("F1/val", val_f1, ep)
        print(f"[{ep:03d}/{args.epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ↳ Val F1 improved to {best_f1:.4f}. Saving model to {checkpoint_path}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
