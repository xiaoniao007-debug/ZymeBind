#!/usr/bin/env python
"""L2 跨模态精确 Shapley 贡献分解（2^4=16 子集全枚举，无采样误差）。

在 p1m1_benchclusterfree/test 上对比 ZymeBind v1 与 ZymeBind v2：
  1. 全局 4 模态（sequence / structure / text / domain）精确 Shapley 贡献与占比；
  2. 按 EC1 七大酶类分层（EC 1–7）；
  3. 按训练集 EC4 频次分层（常见 >20 / 中等 6-20 / 罕见 1-5）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import CachedFeatureDataset, EmbeddingCache, cached_collate  # noqa: E402
from data.dataset import build_domain_vocab, load_split  # noqa: E402
from evaluation.zeroshot_eval import build_mode2, load_checkpoint  # noqa: E402
from models.zymebind_v2 import MODALITIES, load_v2_checkpoint  # noqa: E402
from training_cached import load_cfg  # noqa: E402
from training_zymebind import TIERS, ordered_feats  # noqa: E402

EC1_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


def exact_shapley_4mod(v_subset: np.ndarray) -> np.ndarray:
    """给定 16 个子集的标量值 v_subset (N, 16)，返回 4 个模态的精确 Shapley 值 (N, 4)。

    mask bit i 对应 MODALITIES[i] (0=seq, 1=struct, 2=text, 3=domain)。
    """
    n_mod = 4
    N = v_subset.shape[0]
    phi = np.zeros((N, n_mod), dtype=np.float64)
    fact = [math.factorial(k) for k in range(n_mod + 1)]
    denom = float(fact[n_mod])

    for i in range(n_mod):
        bit = 1 << i
        for mask in range(1 << n_mod):
            if mask & bit:
                continue
            s = bin(mask).count("1")
            w = (fact[s] * fact[n_mod - s - 1]) / denom
            phi[:, i] += w * (v_subset[:, mask | bit] - v_subset[:, mask])
    return phi


@torch.no_grad()
def compute_subset_values(model, loader: DataLoader, ec2id: dict[str, int],
                          device: str) -> tuple[np.ndarray, list[dict]]:
    """对全部 16 个模态子集计算每个测试样本在其真实 EC4 上的平均预测概率。"""
    model.eval()
    v_all = []
    meta_all = []

    for batch in loader:
        feats_full = ordered_feats(batch, device)
        B = next(iter(feats_full.values())).size(0)

        # 筛选至少有 1 个词表内完整 EC4 标签的样本
        target_ids = []
        valid_idx = []
        for i, ecs in enumerate(batch["ec_numbers"]):
            hits = [ec2id[e] for e in ecs if e in ec2id and e.count(".") == 3]
            if hits:
                valid_idx.append(i)
                target_ids.append(hits)
                meta_all.append({
                    "accession": batch["accession"][i],
                    "ecs": [e for e in ecs if e in ec2id and e.count(".") == 3],
                })
        if not valid_idx:
            continue

        v_batch = np.zeros((len(valid_idx), 16), dtype=np.float32)
        for mask in range(16):
            sub_feats = {}
            for m_i, k in enumerate(MODALITIES):
                if (mask >> m_i) & 1:
                    sub_feats[k] = feats_full[k]
                else:
                    sub_feats[k] = torch.zeros_like(feats_full[k])
            logits, _ = model(sub_feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            for r_i, orig_i in enumerate(valid_idx):
                v_batch[r_i, mask] = float(np.mean([probs[orig_i, c] for c in target_ids[r_i]]))
        v_all.append(v_batch)

    return np.concatenate(v_all, axis=0), meta_all


def summarize_shapley(phi: np.ndarray, v_subset: np.ndarray) -> dict:
    mean_phi = phi.mean(axis=0)
    abs_phi = np.abs(phi).mean(axis=0)
    pos_sum = max(float(np.maximum(mean_phi, 0.0).sum()), 1e-9)
    abs_sum = max(float(abs_phi.sum()), 1e-9)
    return {
        "n": int(phi.shape[0]),
        "mean_prob_empty": round(float(v_subset[:, 0].mean()), 4),
        "mean_prob_full": round(float(v_subset[:, 15].mean()), 4),
        "mean_prob_seq_only": round(float(v_subset[:, 1].mean()), 4),
        "mean_prob_seq_struct": round(float(v_subset[:, 3].mean()), 4),
        "shapley_mean": {k: round(float(mean_phi[i]), 4) for i, k in enumerate(MODALITIES)},
        "shapley_share_pct": {k: round(float(max(mean_phi[i], 0.0) / pos_sum * 100.0), 1) for i, k in enumerate(MODALITIES)},
        "shapley_abs_share_pct": {k: round(float(abs_phi[i] / abs_sum * 100.0), 1) for i, k in enumerate(MODALITIES)},
    }


def analyze_model(name: str, model, loader, ec2id, train_ec_counts: Counter, device: str) -> dict:
    v_sub, meta = compute_subset_values(model, loader, ec2id, device)
    phi = exact_shapley_4mod(v_sub)

    out = {
        "model": name,
        "overall": summarize_shapley(phi, v_sub),
        "by_ec1": {},
        "by_train_freq": {},
    }

    # 按 EC1 分层
    ec1_idx = defaultdict(list)
    freq_idx = defaultdict(list)
    for i, m in enumerate(meta):
        ec1 = m["ecs"][0].split(".")[0]
        ec1_idx[ec1].append(i)
        max_cnt = max(train_ec_counts.get(e, 0) for e in m["ecs"])
        if max_cnt == 0:
            bucket = "unseen_0"
        elif max_cnt <= 5:
            bucket = "rare_1_5"
        elif max_cnt <= 20:
            bucket = "medium_6_20"
        else:
            bucket = "common_gt20"
        freq_idx[bucket].append(i)

    for ec1 in sorted(ec1_idx.keys()):
        idx = ec1_idx[ec1]
        out["by_ec1"][f"EC{ec1}_{EC1_NAMES.get(ec1, '')}"] = summarize_shapley(phi[idx], v_sub[idx])
    for b in ("common_gt20", "medium_6_20", "rare_1_5", "unseen_0"):
        if b in freq_idx:
            idx = freq_idx[b]
            out["by_train_freq"][b] = summarize_shapley(phi[idx], v_sub[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="p1m1_benchclusterfree")
    ap.add_argument("--v1-ckpt", default="checkpoints/zymebind_p1m1_benchclusterfree_hierprior.pt")
    ap.add_argument("--v2-ckpt", default="checkpoints/zymebind_v2_p1m1_benchclusterfree.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/v2_interpretability_shapley_l2_p1m1_test.json")
    args = ap.parse_args()

    cfg = load_cfg("configs/pretrain.yaml")
    split_rel, vocab_rel, _ = TIERS[args.tier]
    sdir = ROOT / split_rel
    vpath = ROOT / vocab_rel
    vocab = json.loads(vpath.read_text())
    ec2id = vocab["ec2id"]

    train_rows = load_split(sdir / "train.tsv")
    test_rows = load_split(sdir / "test.tsv")
    domain2id = build_domain_vocab(train_rows)
    train_ec_counts = Counter(e for r in train_rows for e in r["ec_numbers"] if e.count(".") == 3)

    seq_c = EmbeddingCache("sequence")
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")
    te_ds = CachedFeatureDataset(test_rows, ec2id, domain2id, seq_c, txt_c, anc_soft=0.0, threedi_cache=tdi_c)
    te_dl = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=cached_collate)

    # 1. ZymeBind v1
    ckpt_v1, vp, _, edge_idx = load_checkpoint(Path(args.v1_ckpt), args.tier, args.device)
    m_v1 = build_mode2(ckpt_v1, vp, edge_idx, ec2id, args.device, cfg)
    res_v1 = analyze_model("zymebind_v1_hierprior", m_v1, te_dl, ec2id, train_ec_counts, args.device)
    print("\n=== L2 Exact Shapley: ZymeBind v1 (test) ===")
    print(json.dumps(res_v1["overall"], indent=2))

    # 2. ZymeBind v2
    _, _, m_v2, _, _ = load_v2_checkpoint(Path(args.v2_ckpt), args.device)
    res_v2 = analyze_model("zymebind_v2", m_v2, te_dl, ec2id, train_ec_counts, args.device)
    print("\n=== L2 Exact Shapley: ZymeBind v2 (test) ===")
    print(json.dumps(res_v2["overall"], indent=2))
    print("\n=== ZymeBind v2 by Train Freq ===")
    for k, v in res_v2["by_train_freq"].items():
        print(f"  {k:12s} (n={v['n']:4d}): shares={v['shapley_share_pct']} | full={v['mean_prob_full']} ss={v['mean_prob_seq_struct']} seq={v['mean_prob_seq_only']}")

    out = {"tier": args.tier, "split": "test", "v1": res_v1, "v2": res_v2}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[DONE] saved {args.out}")


if __name__ == "__main__":
    main()
