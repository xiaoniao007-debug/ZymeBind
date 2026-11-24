"""监督微调：多模态融合 + 多标签 EC 分类头（方案 3.5 / 5.1 阶段二）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """多标签 Focal Loss（与 DeepECtransformer 一致），缓解 EC 长尾。"""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor | None = None,
                pos_weight: torch.Tensor | None = None) -> torch.Tensor:
        """weights: 逐标签权重（B×C），用于多功能酶的注释粒度降权。
        pos_weight: 逐类别正例权重（C,），缓解 6,110 类极端长尾下"全预测 0"退化解。
        """
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none",
                                                pos_weight=pos_weight)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        if weights is not None:
            loss = loss * weights
        return loss.mean()


class ModalityFusion(nn.Module):
    """把各输入模态向量投影为 token 后做自注意力融合，[CLS] 作为融合表示。

    投影层按 in_dims 的键动态构建：当前缓存训练传入 4 个模态
    （sequence / text / threedi / domain）；研究计划 3.1 的五模态属端到端设计。
    """

    def __init__(self, in_dims: dict[str, int], out_dim: int = 512,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.ModuleDict({k: nn.Linear(d, out_dim) for k, d in in_dims.items()})
        self.cls = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(out_dim, n_heads, out_dim * 4, dropout,
                                       batch_first=True, norm_first=True),
            num_layers=2,
        )
        self.out_dim = out_dim

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = [self.proj[k](v).unsqueeze(1) for k, v in feats.items()]
        tokens = torch.cat([self.cls.expand(len(tokens[0]), -1, -1)] + tokens, dim=1)
        return self.encoder(tokens)[:, 0]


class ECClassifier(nn.Module):
    def __init__(self, in_dims: dict[str, int], n_classes: int, fusion_dim: int = 512,
                 n_heads: int = 8, dropout: float = 0.1, hidden: int = 1024):
        super().__init__()
        self.fusion = ModalityFusion(in_dims, fusion_dim, n_heads, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.fusion(feats))
