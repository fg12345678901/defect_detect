# losses/combined.py
"""
Composite loss = CE + Dice + λ·Contrast (pixel-level).

Dice excludes background channel (id 0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import PixelWiseSupervisedContrastiveLoss


class _DiceLossNoBg(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits : (N,C,H,W) – raw, NOT softmaxed
        targets: (N,H,W)   – long
        """
        probs = F.softmax(logits, dim=1)
        targets_1hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        probs_fg, targets_fg = probs[:, 1:], targets_1hot[:, 1:]

        inter = (probs_fg * targets_fg).sum(dim=(0, 2, 3))
        denom = probs_fg.sum(dim=(0, 2, 3)) + targets_fg.sum(dim=(0, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class CombinedLossWithContrast(nn.Module):
    def __init__(
        self,
        contrastive_weight: float = 0.2,
        ignore_bg_in_contrast: bool = False,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = _DiceLossNoBg()
        self.contrast = PixelWiseSupervisedContrastiveLoss(
            temperature=0.07,
            ignore_background=ignore_bg_in_contrast,
            max_samples=15_000,
        )
        self.weight = contrastive_weight

    def forward(self, inputs, targets):
        """inputs = (logits, emb) tuple produced by model."""
        logits, emb = inputs
        return (
            self.ce(logits, targets)
            + self.dice(logits, targets)
            + self.weight * self.contrast(emb, targets)
        )
