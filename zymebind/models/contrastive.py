"""ZymeBind 统一多模态对比学习模块（方案 3.3）。

以序列模态为锚点，对 (结构, 文本, 结构域, EC 本体) 四个模态做 InfoNCE 对齐。
借鉴 FunBind 的 MoE 投影 + diversity loss：投影由 1~N 个专家组成，gating 网络
按样本动态加权，diversity loss 惩罚专家输出两两余弦相似度以防专家塌缩。
n_experts=1（默认）时退化为原 ProjectionHead，state_dict 键逐字不变，
既有 _hierprior / _mlphead / _longbudget / _epochs60 checkpoint 可直接加载。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    @property
    def first_linear(self) -> nn.Linear:
        """供阶段二把对比投影知识迁移到融合层；property 不注册重复模块。"""
        return self.net[0]

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MoEProjectionHead(nn.Module):
    """FunBind 式专家混合投影：MLP + BN + GELU + Dropout 专家 + 样本级 gating。"""

    def __init__(self, in_dim: int, out_dim: int, n_experts: int = 3,
                 hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or out_dim * 2
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, out_dim))
            for _ in range(n_experts)])
        self.gate = nn.Sequential(nn.Linear(in_dim, n_experts), nn.Softmax(dim=-1))

    @property
    def first_linear(self) -> nn.Linear:
        return self.experts[0][0]

    def forward(self, x, return_experts: bool = False):
        outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, D)
        w = self.gate(x).unsqueeze(-1)                           # (B, E, 1)
        out = (outs * w).sum(1)
        if return_experts:
            return F.normalize(out, dim=-1), F.normalize(outs, dim=-1)
        return F.normalize(out, dim=-1)

    @staticmethod
    def diversity_loss(expert_out: torch.Tensor) -> torch.Tensor:
        """惩罚同模态专家输出的非对角余弦相似度，防止所有专家学成同一个函数。"""
        sim = expert_out @ expert_out.transpose(1, 2)            # (B, E, E)
        e = sim.size(1)
        eye = torch.eye(e, device=sim.device, dtype=torch.bool).unsqueeze(0)
        off = sim.masked_fill(eye, 0.0)
        return (off ** 2).sum() / (sim.size(0) * e * (e - 1))


class ZymeBindContrastive(nn.Module):
    """五模态编码 + 投影头 + InfoNCE（序列为锚）。"""

    def __init__(self, encoders: dict[str, nn.Module], in_dims: dict[str, int],
                 proj_dim: int = 256, temperature: float = 0.2,
                 n_experts: int = 1, diversity_weight: float = 0.0):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        if n_experts <= 1:
            self.heads = nn.ModuleDict({k: ProjectionHead(d, proj_dim)
                                        for k, d in in_dims.items()})
        else:
            self.heads = nn.ModuleDict({k: MoEProjectionHead(d, proj_dim, n_experts)
                                        for k, d in in_dims.items()})
        self.log_tau = nn.Parameter(torch.tensor(temperature).log())
        self.diversity_weight = diversity_weight
        self._expert_outputs: dict[str, torch.Tensor] = {}

    @staticmethod
    def info_nce(anchor: torch.Tensor, other: torch.Tensor, tau: float) -> torch.Tensor:
        logits = anchor @ other.t() / tau
        labels = torch.arange(anchor.size(0), device=anchor.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """batch 需含各模态已编码的输入；返回 {modality: projected embedding}。"""
        embs, experts = {}, {}
        for name, enc in self.encoders.items():
            if name not in batch:
                continue
            x = enc(**batch[name])
            head = self.heads[name]
            if isinstance(head, MoEProjectionHead):
                embs[name], experts[name] = head(x, return_experts=True)
            else:
                embs[name] = head(x)
        self._expert_outputs = experts
        return embs

    def loss(self, embs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        tau = self.log_tau.exp().clamp(0.05, 1.0)
        anchor = embs["sequence"]
        total, logs = 0.0, {}
        for name in ("structure", "text", "domain", "ec_ontology"):
            if name not in embs:
                continue
            l = self.info_nce(anchor, embs[name], tau)
            logs[f"nce_{name}"] = float(l.detach())
            total = total + l
        if self.diversity_weight > 0 and self._expert_outputs:
            div = torch.stack([MoEProjectionHead.diversity_loss(v)
                               for v in self._expert_outputs.values()]).mean()
            total = total + self.diversity_weight * div
            logs["diversity"] = float(div.detach())
        return total, logs
