# losses/contrastive.py
"""
Pixel-wise supervised contrastive loss (ignore background option).

Ported from old train2.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelWiseSupervisedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        ignore_background: bool = False,
        max_samples: int | None = 15_000,
    ):
        super().__init__()
        self.temperature = temperature
        self.ignore_background = ignore_background
        self.max_samples = max_samples

    def forward(self, emb: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        emb:     (N, C, H, W) – L2-normalised inside.
        targets: (N, H, W)   – long mask.
        """
        n, c, h, w = emb.shape
        feats = F.normalize(emb.permute(0, 2, 3, 1).reshape(-1, c), dim=1)
        labels = targets.view(-1)

        if self.ignore_background:
            valid_mask = labels != 0
            feats, labels = feats[valid_mask], labels[valid_mask]

        total = feats.shape[0]
        if self.max_samples and total > self.max_samples:
            idx = torch.randperm(total, device=feats.device)[: self.max_samples]
            feats, labels, total = feats[idx], labels[idx], self.max_samples

        if total < 2:  # nothing to contrast
            return torch.tensor(0.0, device=feats.device, requires_grad=True)

        sim = torch.matmul(feats, feats.t()) / self.temperature  # (total,total)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = label_eq.float()
        pos_mask.fill_diagonal_(0)

        exp_sim = torch.exp(sim)
        sum_pos = (exp_sim * pos_mask).sum(dim=1)
        sum_all = exp_sim.sum(dim=1) - torch.exp(sim.diagonal())

        valid = sum_pos > 0
        loss = -torch.log(sum_pos[valid] / sum_all[valid]).mean()
        return loss
