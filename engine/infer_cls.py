# engine/infer_cls.py
"""
Inference script for the classification model.

This script can operate in two modes:
1.  Batch Evaluation: Evaluates a whole directory of test images.
    (Labels are derived from filenames).
2.  Single Image Prediction: Predicts the class for a single image file.

Usage Example (Batch Evaluation):
python -m engine.infer_cls \
    --model_path logs/Classify_test/best_model.pth \
    --test_dir dataset/Classify_test/

Usage Example (Single Image Prediction):
python -m engine.infer_cls `
    --model_path logs/Classify/best_model.pth `
    --image_path dataset/magnetic/val/val_images/exp3_num_322643.jpg `
    --data_dir dataset/
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# --- Local Module Imports ---
from models.classifier import create_classifier
from losses.cls_loss import get_cls_loss
from utils.metrics import evaluate_classifier


# ----------------------------------------------------------------------
# Custom Dataset for Inference (for batch evaluation)
# ----------------------------------------------------------------------
class InferenceDataset(Dataset):
    """
    Custom dataset for a flat test directory where the class name
    is the prefix of the filename, e.g., 'steel_0001.jpg'.
    """

    def __init__(self, test_dir: str, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.image_paths = [f for f in self.test_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in directory: {test_dir}")
        class_names = sorted(list(set([p.name.split('_')[0] for p in self.image_paths])))
        self.classes = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for path in self.image_paths:
            class_name = path.name.split('_')[0]
            label_idx = self.class_to_idx[class_name]
            self.samples.append((path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ----------------------------------------------------------------------
# Core Logic Functions
# ----------------------------------------------------------------------
def evaluate_directory(args, device):
    """Handles evaluating a whole directory of test images."""
    print(f"--- Starting Batch Evaluation on Directory: {args.test_dir} ---")
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        test_dataset = InferenceDataset(args.test_dir, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    except FileNotFoundError as e:
        print(e)
        return

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes in test set: {class_names}")

    model = create_classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    criterion = get_cls_loss()
    test_loss, test_f1, report_dict = evaluate_classifier(model, test_loader, criterion, device, class_names)

    print("\n" + "=" * 50)
    print("          INFERENCE RESULTS          ")
    print("=" * 50)
    print(f"\nOverall Macro F1-Score: {test_f1:.4f}\n")

    header = f"{'CLASS':<15} | {'RECALL (ACCURACY)':<20} | {'PRECISION':<15} | {'F1-SCORE':<15}"
    print(header)
    print("-" * len(header))

    for class_name in class_names:
        metrics = report_dict[class_name]
        recall, precision, f1 = metrics['recall'], metrics['precision'], metrics['f1-score']
        print(f"{class_name:<15} | {recall:<20.4f} | {precision:<15.4f} | {f1:<15.4f}")


# **↓↓↓ 修改点 1：在函数定义中加入 args 参数 ↓↓↓**
def classify_single_image(args, device):
    """Handles classifying a single image file."""
    print(f"--- Predicting Single Image: {args.image_path} ---")
    if not args.data_dir:
        print("Error: --data_dir is required for single image prediction to map class indices to names.")
        return

    # Define a set of directories to ignore
    IGNORE_DIRS = {'Classify_test'}
    # Filter out the ignored directories when finding classes
    class_names = sorted([
        d.name for d in Path(args.data_dir).iterdir()
        if d.is_dir() and d.name not in IGNORE_DIRS
    ])
    num_classes = len(class_names)
    if num_classes == 0:
        print(f"Error: No valid class subdirectories found in {args.data_dir}")
        return
    print(f"Found {num_classes} potential classes: {class_names}")

    # Load Model
    model = create_classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Prepare Image
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.open(args.image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image_path}")
        return

    # Add batch dimension and send to device
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        prediction_idx = outputs.argmax(dim=1).item()

    predicted_class = class_names[prediction_idx]
    confidence = probabilities[prediction_idx].item()

    print("\n" + "=" * 30)
    print("     PREDICTION RESULT     ")
    print("=" * 30)
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Confidence:      {confidence:.2%}")
    print("=" * 30)


def main():
    ap = argparse.ArgumentParser(description="Inference script for classification model.")
    ap.add_argument("--model_path", required=True, type=str, help="Path to the trained model (.pth file).")
    ap.add_argument("--test_dir", type=str, default=None,
                    help="Path to the test dataset directory for batch evaluation.")
    ap.add_argument("--image_path", type=str, default=None, help="Path to a single image for classification.")
    ap.add_argument("--data_dir", type=str, default="dataset/",
                    help="Path to original data source to infer class names (for single image mode).")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for batch evaluation.")
    ap.add_argument("--img_size", type=int, default=380, help="Image size the model expects.")
    args = ap.parse_args()

    # --- Mode Selection ---
    if not args.image_path and not args.test_dir:
        print(
            "Error: You must provide either --test_dir for batch evaluation or --image_path for single image prediction.")
        return

    if args.image_path and args.test_dir:
        print("Error: Please provide either --test_dir or --image_path, not both.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test_dir:
        evaluate_directory(args, device)
    elif args.image_path:
        # **↓↓↓ 修改点 2：在函数调用时传入 args 参数 ↓↓↓**
        classify_single_image(args, device)

    print("\nInference finished.")


if __name__ == "__main__":
    main()
