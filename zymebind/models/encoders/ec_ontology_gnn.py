"""EC 本体模态编码器：EC 层级 DAG 上的图嵌入（方案 3.2）。

节点特征不用随机 nn.Embedding：随机初始化经 2 层 GCN 平滑后会让所有 EC 嵌入
塌缩（实测随机两类余弦约 0.43），训练未见 EC 又被 InfoNCE 当负例系统性推远，
Mode 1 零样本检索退化为随机。本版以 EC 四级前缀的确定性层级先验作基底，
叠加可学习残差，并用输入残差连接阻止 GCN 抹平个体区分度：
共享前缀的节点天然相近，训练未见 EC 因此可继承已见兄弟类的语义。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None


def level_prior(ec2id: dict[str, int], dim: int) -> torch.Tensor:
    """按 EC 四级前缀构造确定性层级特征（固定 seed，可复现）。

    例：`1.1.1.1` 与 `1.1.1.2` 共享前三级子向量，仅第四级不同，
    故层级相近又彼此可区分；浅层前缀节点取实际存在的级数。
    """
    g = torch.Generator().manual_seed(42)
    tables = [torch.randn(1024, dim, generator=g) for _ in range(4)]
    out = torch.zeros(len(ec2id), dim)
    for ec, j in ec2id.items():
        parts = ec.split(".")
        for lvl in range(min(len(parts), 4)):
            key = ".".join(parts[:lvl + 1])
            h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 1024
            out[j] += tables[lvl][h] / (lvl + 1)
    return F.normalize(out, dim=-1)


class ECOntologyEncoder(nn.Module):
    def __init__(self, vocab_path: str, dim: int = 256, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        vocab = json.loads(Path(vocab_path).read_text())
        self.ec2id: dict[str, int] = vocab["ec2id"]
        self.n_nodes = len(self.ec2id)
        self.register_buffer("prior", level_prior(self.ec2id, dim))
        self.delta = nn.Parameter(torch.zeros(self.n_nodes, dim))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if GCNConv is None:
                raise ImportError("需要 torch-geometric")
            self.convs.append(GCNConv(dim, dim))
            self.norms.append(nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = dim

    @staticmethod
    def load_edge_index(edges_tsv: str, ec2id: dict[str, int]) -> torch.Tensor:
        src, dst = [], []
        with open(edges_tsv) as fh:
            next(fh)
            for line in fh:
                p, c = line.split()
                if p in ec2id and c in ec2id:
                    # 双向：子类 -> 父类 与 父类 -> 子类 都建边（DAG 双向消息传递）
                    src += [ec2id[p], ec2id[c]]
                    dst += [ec2id[c], ec2id[p]]
        return torch.tensor([src, dst], dtype=torch.long)

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x0 = F.normalize(self.prior + self.delta, dim=-1)
        x = x0
        for conv, norm in zip(self.convs, self.norms):
            neigh = self.dropout(norm(torch.relu(conv(x, edge_index))))
            x = F.normalize(x + neigh, dim=-1)
        return x
