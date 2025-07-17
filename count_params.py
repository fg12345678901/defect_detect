import argparse
import torch
from typing import Any, Dict


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch state dict from a .pth file."""
    obj: Any = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # If the values are tensors, we assume this is already a state dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        # Common wrappers
        for key in ["state_dict", "model", "model_state_dict", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    raise ValueError(f"Unable to find state dict in file: {path}")


def count_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    """Return total number of parameters in the state dict."""
    return int(sum(v.numel() for v in state_dict.values()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Count parameters in a .pth model file")
    ap.add_argument("model_path", help="Path to the model weights (.pth)")
    args = ap.parse_args()

    state_dict = load_state_dict(args.model_path)
    total = count_parameters(state_dict)
    size_mb = total * 4 / 1e6  # assuming float32 weights
    print(f"Total parameters: {total:,}")
    print(f"Approx. storage size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
