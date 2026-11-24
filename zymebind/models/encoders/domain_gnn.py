"""结构域模态编码器：InterPro/Pfam 注释 -> 域节点 GCN 聚合（方案 3.2）。

每个蛋白表示为其所含域节点的子图；用共享的域嵌入 + 2 层 GCN 聚合邻居域信息，
再对蛋白所含域做注意力池化得到蛋白级域嵌入。
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
except ImportError:  # 允许无 PyG 时退化
    GCNConv = None


class DomainGNNEncoder(nn.Module):
    def __init__(self, n_domains: int, embedding_dim: int = 256,
                 hidden: int = 256, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_domains, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim = embedding_dim
        for _ in range(layers):
            if GCNConv is None:
                raise ImportError("需要 torch-geometric")
            self.convs.append(GCNConv(dim, hidden))
            self.norms.append(nn.LayerNorm(hidden))
            dim = hidden
        self.att = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, domain_ids, edge_index, batch):
        """domain_ids: [N] 全局域节点 id（子图）; edge_index: [2,E]; batch: [N] 所属蛋白索引。"""
        x = self.embedding(domain_ids)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(torch.relu(conv(x, edge_index)))
            x = self.dropout(x)
        # 注意力池化
        w = torch.softmax(self.att(x).squeeze(-1), dim=0)
        n_prot = int(batch.max()) + 1
        pooled = torch.zeros(n_prot, x.size(-1), device=x.device)
        pooled.index_add_(0, batch, x * w.unsqueeze(-1))
        return pooled
