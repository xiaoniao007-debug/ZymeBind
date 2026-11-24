"""L3 稀疏自编码器：在 ESM-2 残基嵌入上发现单语义特征（方案 6.3 L3，借鉴 InterPLM）。

架构: 1280-d -> k-d (k>>1280, 如 16384) 加 L1 稀疏惩罚 -> 重建 1280-d。
输出特征激活矩阵，供 sae_interpretation.py 做 LLM 自动描述与 HDBSCAN 聚类。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int = 1280, expansion: int = 12, l1_coef: float = 1e-3):
        super().__init__()
        d_hidden = d_in * expansion
        self.encoder = nn.Linear(d_in, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.l1_coef = l1_coef
        self.d_hidden = d_hidden
        nn.init.kaiming_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

    def forward(self, x: torch.Tensor):
        pre = self.encoder(x)
        acts = torch.relu(pre)
        recon = self.decoder(acts)
        return recon, acts

    def loss(self, x: torch.Tensor):
        recon, acts = self(x)
        mse = (recon - x).pow(2).mean()
        l1 = acts.abs().mean()
        return mse + self.l1_coef * l1, {"mse": float(mse), "l1": float(l1),
                                        "l0": float((acts > 0).float().sum(-1).mean())}


def train_sae(embeddings: torch.Tensor, out_path: str, d_in: int = 1280,
              expansion: int = 12, epochs: int = 30, batch_size: int = 4096,
              lr: float = 1e-4, device: str = "cuda"):
    """在池化后的残基嵌入矩阵 [N, 1280] 上训练 SAE。"""
    model = SparseAutoencoder(d_in, expansion).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(embeddings)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = embeddings[idx].to(device)
            loss, logs = model.loss(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"[sae] epoch {ep+1}/{epochs} loss={tot/(n/batch_size):.5f} {logs}")
    torch.save({"state_dict": model.state_dict(), "d_in": d_in,
                "expansion": expansion}, out_path)
    return model
