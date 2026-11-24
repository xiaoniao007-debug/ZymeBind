#!/usr/bin/env python
"""P1 #5b：L3 稀疏自编码器（Top-K SAE，对标 Nature Methods 2025 InterPLM）单语义生物学特征发现。

在锁定单模型 m2_dualplm_large 的 768 维 fused 表征上（纯 seq_struct 无元数据视图，
确保学到的单语义特征源自序列+结构而非文本捷径）训练 4096 维 Top-K (k=32) SAE，
并在 common_clean/val 上定量验证单个稀疏神经元与以下真实生化概念的对齐度（F1 / AUROC）：
  1. EC3 催化机制大类（val 中样本 >= 15 的 EC3 类）
  2. 辅因子 / 金属离子（ATP, NAD/NADP, FAD, Mg2+, Zn2+, Fe-S, PLP, Heme, SAM, CoA）
  3. 高频 InterPro 催化结构域（val 中样本 >= 25 的前 50 个 InterPro）

输出：results/val_interpretability_l3_sae_monosemantic.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import EmbeddingCache  # noqa: E402
from data.dataset import load_split  # noqa: E402
from evaluation.eval_comprehensive_val import build_inputs  # noqa: E402
from models.zymebind_v2 import load_v2_checkpoint  # noqa: E402
from training_zymebind import TIERS  # noqa: E402
from training_zymebind_v4_dualplm import ProstT5AACache  # noqa: E402

COFACTOR_PATTERNS = {
    "ATP": re.compile(r"\bATP\b", re.I),
    "NAD_NADP": re.compile(r"\bNADP?\b|\bNADH\b|\bNADPH\b", re.I),
    "FAD_FMN": re.compile(r"\bFAD\b|\bFMN\b|\bflavin\b", re.I),
    "Mg2+": re.compile(r"Mg\(2\+\)|magnesium", re.I),
    "Zn2+": re.compile(r"Zn\(2\+\)|zinc", re.I),
    "Fe_S": re.compile(r"\[4Fe-4S\]|\[2Fe-2S\]|iron-sulfur", re.I),
    "PLP": re.compile(r"pyridoxal\s+5'-phosphate|\bPLP\b", re.I),
    "Heme": re.compile(r"\bheme\b", re.I),
    "SAM": re.compile(r"S-adenosyl-L-methionine|\bAdoMet\b", re.I),
    "CoA": re.compile(r"\bCoA\b|coenzyme A", re.I),
}


class TopKSAE(nn.Module):
    """Top-K 稀疏自编码器（Gao et al. 2024 / InterPLM 2025 风格）。"""

    def __init__(self, d_in: int = 768, d_latent: int = 4096, k: int = 32):
        super().__init__()
        self.d_in = d_in
        self.d_latent = d_latent
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, d_latent, bias=True)
        self.dec = nn.Linear(d_latent, d_in, bias=False)
        nn.init.kaiming_uniform_(self.enc.weight, a=0.0)
        with torch.no_grad():
            self.dec.weight.copy_(F.normalize(self.enc.weight.t(), dim=0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu(self.enc(x - self.b_pre))
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.dec(z) + self.b_pre
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.dec.weight.data = F.normalize(self.dec.weight.data, dim=0)


@torch.no_grad()
def extract_fused_seq_struct(stage2, rows, caches, dom_cache, device: str, bs: int = 512) -> np.ndarray:
    seq, tri, txt, dom = build_inputs(rows, "p1m2_benchclusterfree", True, caches, dom_cache)
    z_txt = np.zeros_like(txt)
    z_dom = np.zeros_like(dom)
    outs = []
    for i in range(0, len(rows), bs):
        feats = {
            "sequence": torch.from_numpy(seq[i:i + bs]).float().to(device),
            "structure": torch.from_numpy(tri[i:i + bs]).float().to(device),
            "text": torch.from_numpy(z_txt[i:i + bs]).float().to(device),
            "domain": torch.from_numpy(z_dom[i:i + bs]).float().to(device),
        }
        out = stage2.forward_detailed(feats)
        outs.append(out["fused"].float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def best_monosemantic_match(Z: np.ndarray, y: np.ndarray) -> dict:
    """对二值概念标签 y (N,)，找到相关性/F1 最高的单个 SAE 稀疏神经元并算精确 F1 & AUROC。"""
    y_f = y.astype(np.float32)
    pos = float(y_f.sum())
    if pos < 10 or pos > len(y) - 10:
        return {}
    # 向量化计算每个神经元与 y 的二值激活 F1（激活 > 0 即触发）
    act = (Z > 0).astype(np.float32)  # (N, D)
    tp = (act.T @ y_f)                # (D,)
    pred_pos = act.sum(axis=0)        # (D,)
    f1_all = (2.0 * tp) / np.maximum(pred_pos + pos, 1e-9)

    # 在 top-10 候选神经元中用连续激活值算精确 AUROC 与最优阈值 F1
    top_idx = np.argsort(-f1_all)[:10]
    best = {"f1": -1.0}
    for j in top_idx:
        scores = Z[:, j]
        if scores.max() <= 0:
            continue
        auroc = float(roc_auc_score(y, scores))
        # 5 个分位阈值选最佳 F1
        f1_j = float(f1_all[j])
        for thr in np.percentile(scores[scores > 0], [10, 30, 50, 70, 85]):
            p_b = (scores >= thr).astype(np.float32)
            t_p = float((p_b * y_f).sum())
            f1_t = (2.0 * t_p) / max(float(p_b.sum()) + pos, 1e-9)
            if f1_t > f1_j:
                f1_j = f1_t
        if f1_j > best["f1"]:
            best = {
                "feature_id": int(j),
                "f1": round(f1_j, 4),
                "auroc": round(auroc, 4),
                "n_pos": int(pos),
                "n_active": int(pred_pos[j]),
            }
    return best


def main():
    device = "cuda:0"
    torch.manual_seed(42)
    np.random.seed(42)

    train_rows = load_split(ROOT / TIERS["p1m2_benchclusterfree"][0] / "train.tsv")
    val_rows = load_split(ROOT / "data/processed/common_clean/val.tsv")
    # 随机抽取 60,000 条 M2 train 蛋白训练 SAE（覆盖全部主要酶家族且 2 分钟内收敛）
    rng = np.random.default_rng(42)
    tr_sub = [train_rows[i] for i in rng.choice(len(train_rows), size=min(60000, len(train_rows)), replace=False)]

    caches = (EmbeddingCache("sequence"), EmbeddingCache("text"),
              EmbeddingCache("threedi_observed_afdbv6"), ProstT5AACache())
    dom_cache = {}
    _, _, stage2, _, _ = load_v2_checkpoint(ROOT / "checkpoints/zymebind_v4large_dualplm_p1m2_benchclusterfree.pt", device)

    print("[L3 SAE] Extracting 768d fused (seq_struct, metadata-free) representations...", flush=True)
    X_tr = extract_fused_seq_struct(stage2, tr_sub, caches, dom_cache, device)
    X_va = extract_fused_seq_struct(stage2, val_rows, caches, dom_cache, device)
    del stage2
    torch.cuda.empty_cache()

    sae = TopKSAE(d_in=X_tr.shape[1], d_latent=4096, k=32).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr)), batch_size=512, shuffle=True)

    t0 = time.time()
    for ep in range(1, 21):
        sae.train()
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            xh, _ = sae(xb)
            loss = F.mse_loss(xh, xb)
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            tot += float(loss.item()) * len(xb)
        if ep in (1, 5, 10, 15, 20):
            print(f"  [SAE ep {ep:02d}/20] mse={tot/len(X_tr):.5f} ({time.time()-t0:.0f}s)", flush=True)

    # 在 common_clean/val 上评估重建 R²、稀疏度与单语义生物学概念对齐
    sae.eval()
    with torch.no_grad():
        Xv_t = torch.from_numpy(X_va).to(device)
        Xh_t, Zv_t = sae(Xv_t)
        mse_va = float(F.mse_loss(Xh_t, Xv_t).item())
        var_va = float(Xv_t.var(dim=0).mean().item())
        r2_va = 1.0 - mse_va / max(var_va, 1e-9)
        Z_va = Zv_t.cpu().numpy()
        dead_frac = float(((Z_va > 0).sum(axis=0) == 0).mean())
        l0_mean = float((Z_va > 0).sum(axis=1).mean())
    print(f"[L3 SAE val] R2={r2_va:.4f} MSE={mse_va:.5f} L0={l0_mean:.1f} dead_features={dead_frac*100:.1f}%", flush=True)

    # 1. EC3 催化机制对齐（n >= 15）
    ec3_cnt = Counter(".".join(e.split(".")[:3]) for r in val_rows for e in r["ec_numbers"] if e.count(".") == 3)
    ec3_targets = [c for c, n in ec3_cnt.most_common() if n >= 15]
    ec3_matches = {}
    for c in ec3_targets:
        y = np.array([any(e.startswith(c + ".") for e in r["ec_numbers"]) for r in val_rows], dtype=np.int32)
        m = best_monosemantic_match(Z_va, y)
        if m:
            ec3_matches[c] = m

    # 2. 辅因子 / 金属离子对齐
    cof_matches = {}
    for cname, pat in COFACTOR_PATTERNS.items():
        y = np.array([
            bool(pat.search((r.get("cofactor") or "") + " " + (r.get("catalytic_activity") or "")))
            for r in val_rows
        ], dtype=np.int32)
        m = best_monosemantic_match(Z_va, y)
        if m:
            cof_matches[cname] = m

    # 3. 高频 InterPro 催化结构域对齐（前 50 个，n >= 25）
    ipr_cnt = Counter(it for r in val_rows for it in (r.get("domains") or {}).get("InterPro", []))
    ipr_targets = [d for d, n in ipr_cnt.most_common(50) if n >= 25]
    ipr_matches = {}
    for d in ipr_targets:
        y = np.array([d in (r.get("domains") or {}).get("InterPro", []) for r in val_rows], dtype=np.int32)
        m = best_monosemantic_match(Z_va, y)
        if m:
            ipr_matches[d] = m

    def agg_matches(d: dict) -> dict:
        if not d:
            return {"n_concepts": 0}
        f1s = [v["f1"] for v in d.values()]
        aucs = [v["auroc"] for v in d.values()]
        top = sorted(d.items(), key=lambda kv: kv[1]["f1"], reverse=True)[:10]
        return {
            "n_concepts": len(d),
            "mean_f1": round(float(np.mean(f1s)), 4),
            "mean_auroc": round(float(np.mean(aucs)), 4),
            "frac_f1_ge_0.7": round(float(np.mean([x >= 0.7 for x in f1s])), 4),
            "frac_auroc_ge_0.9": round(float(np.mean([x >= 0.9 for x in aucs])), 4),
            "top10": [{"concept": k, **v} for k, v in top],
        }

    summary = {
        "eval_set": "data/processed/common_clean/val.tsv",
        "model": "m2_dualplm_large (seq_struct 768d fused, ZERO text/domain input)",
        "sae_cfg": {"d_in": 768, "d_latent": 4096, "top_k": 32, "n_train": len(X_tr)},
        "reconstruction": {"r2": round(r2_va, 4), "mse": round(mse_va, 5), "l0_mean": round(l0_mean, 1), "dead_frac": round(dead_frac, 4)},
        "ec3_mechanisms": agg_matches(ec3_matches),
        "cofactors": agg_matches(cof_matches),
        "interpro_domains": agg_matches(ipr_matches),
    }
    print("  [EC3 mechanisms]", {k: v for k, v in summary["ec3_mechanisms"].items() if k != "top10"})
    print("  [Cofactors]     ", {k: v for k, v in summary["cofactors"].items() if k != "top10"})
    print("  [InterPro]      ", {k: v for k, v in summary["interpro_domains"].items() if k != "top10"})

    out_p = ROOT / "results/val_interpretability_l3_sae_monosemantic.json"
    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    torch.save(sae.state_dict(), ROOT / "checkpoints/sae_l3_m2_dualplm_large.pt")
    print(f"[DONE] saved {out_p}", flush=True)


if __name__ == "__main__":
    main()
