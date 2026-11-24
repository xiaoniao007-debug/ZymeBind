#!/usr/bin/env python
"""L1 残基级催化位点归因（Integrated Gradients 穿透 ESM-2 650M），只用内部 held-out test。

做法：
  - 在线跑 ESM-2 650M，池化方式与离线缓存逐字一致（mean(last_hidden * attention_mask)），
    并对每条蛋白校验在线池化向量与缓存向量的余弦相似度（防止口径漂移）；
  - 池化向量送入 Mode 2 分类头，**纯序列语境**（3Di / text / domain 全零）；
  - 归因目标 = 模型自身预测概率最高的 EC4（不使用真值标签选目标）；
  - Integrated Gradients：在 word-embedding 输出处插值，基线 = 各残基位置替换为 <mask> 的嵌入
    （CLS/EOS 保持不变），步数 --ig-steps；残基分数 = |Σ_d (x-x')·avg_grad|；
  - 真值：UniProt Active site（ACT_SITE）与 ACT_SITE ∪ Binding site；
  - 指标：AUROC、AUPRC（附阳性率）、Top-10 命中率、Top-10 召回、Enrichment@10；
    随机打分基线用 50 次置换经验估计；按预测是否正确分层。

对比模型：ZymeBind v1 (hierprior) 与 ZymeBind v2（同一 p1m1_benchclusterfree 划分）。
不触碰 NEW-392 / Price-149。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import EmbeddingCache  # noqa: E402
from data.dataset import load_split  # noqa: E402
from evaluation.zeroshot_eval import build_mode2, load_checkpoint  # noqa: E402
from models.zymebind_v2 import load_v2_checkpoint  # noqa: E402
from training_cached import load_cfg  # noqa: E402
from training_zymebind import TIERS  # noqa: E402

ESM_NAME = "facebook/esm2_t33_650M_UR50D"


def site_metrics(scores: np.ndarray, pos: list[int]) -> dict:
    L = len(scores)
    y = np.zeros(L, dtype=np.int32)
    y[pos] = 1
    order = np.argsort(-scores)
    top10 = y[order[:10]]
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "prevalence": float(y.mean()),
        "top10_hit": float(top10.sum() > 0),
        "top10_recall": float(top10.sum() / len(pos)),
        "enrich10": float(top10.mean() / y.mean()),
    }


def random_metrics(L: int, pos: list[int], rng: np.random.Generator, n: int = 50) -> dict:
    acc = []
    for _ in range(n):
        acc.append(site_metrics(rng.random(L), pos))
    return {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}


def agg(items: list[dict]) -> dict:
    if not items:
        return {"n": 0}
    keys = items[0].keys()
    out = {"n": len(items)}
    for k in keys:
        out[k] = round(float(np.mean([x[k] for x in items])), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="p1m1_benchclusterfree")
    ap.add_argument("--v1-ckpt", default="checkpoints/zymebind_p1m1_benchclusterfree_hierprior.pt")
    ap.add_argument("--v2-ckpt", default="checkpoints/zymebind_v2_p1m1_benchclusterfree.pt")
    ap.add_argument("--max-proteins", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=800)
    ap.add_argument("--ig-steps", type=int, default=16)
    ap.add_argument("--ig-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/interp_l1_ig_v1_v2_p1m1_test.json")
    args = ap.parse_args()
    dev = args.device
    rng = np.random.default_rng(42)

    split_rel, vocab_rel, _ = TIERS[args.tier]
    ec2id = json.loads((ROOT / vocab_rel).read_text())["ec2id"]
    id2ec = {int(v): k for k, v in ec2id.items()}
    ec4_mask = torch.tensor([id2ec[i].count(".") == 3 for i in range(len(ec2id))], device=dev)

    sites = json.loads((ROOT / "data/processed/active_sites_eval.json").read_text())["annotations"]
    test_rows = load_split(ROOT / split_rel / "test.tsv")
    cands = [r for r in test_rows
             if r["accession"] in sites and sites[r["accession"]]["act_site"]
             and 30 <= len(r["sequence"]) <= args.max_len and r["ec_numbers"]]
    cands.sort(key=lambda r: r["accession"])  # 确定性
    cands = cands[:args.max_proteins]
    print(f"[L1] candidates with ACT_SITE (len<={args.max_len}): {len(cands)}", flush=True)

    cfg = load_cfg("configs/pretrain.yaml")
    ck1, vp1, _, ei1 = load_checkpoint(Path(args.v1_ckpt), args.tier, dev)
    m_v1 = build_mode2(ck1, vp1, ei1, ec2id, dev, cfg).eval()
    _, _, m_v2, _, ck2 = load_v2_checkpoint(Path(args.v2_ckpt), dev)
    m_v2.eval()
    models = {"v1_hierprior": (m_v1, ck1["in_dims"]), "v2": (m_v2, ck2["in_dims"])}

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ESM_NAME, local_files_only=True)
    esm = AutoModel.from_pretrained(ESM_NAME, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    for p in esm.parameters():
        p.requires_grad_(False)
    emb_layer = esm.embeddings.word_embeddings
    seq_cache = EmbeddingCache("sequence")

    def pooled_from(ids, mask, interp):
        handle = emb_layer.register_forward_hook(lambda m, i, o: interp)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = esm(input_ids=ids, attention_mask=mask).last_hidden_state
        finally:
            handle.remove()
        h = h.float()
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    res = {name: {"act_site": [], "act_or_bind": [], "act_site_correct": [], "act_site_wrong": [], "cases": []}
           for name in models}
    rand = {"act_site": [], "act_or_bind": []}
    cos_checks = []
    t0 = time.time()

    for ci, r in enumerate(cands):
        acc, seq = r["accession"], r["sequence"]
        L = len(seq)
        enc = tok([seq], return_tensors="pt").to(dev)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        T = ids.size(1)
        with torch.no_grad():
            x = emb_layer(ids).detach()
            base_ids = ids.clone()
            base_ids[:, 1:T - 1] = tok.mask_token_id
            xb = emb_layer(base_ids).detach()
            pooled_full = pooled_from(ids, mask, x)
        cached = seq_cache.get(acc).to(dev)
        cos_checks.append(float(F.cosine_similarity(pooled_full[0], cached, dim=0).item()))

        act = [p - 1 for p in sites[acc]["act_site"] if 1 <= p <= L]
        both = sorted(set(act + [p - 1 for p in sites[acc]["binding"] if 1 <= p <= L]))
        if not act:
            continue
        rand["act_site"].append(random_metrics(L, act, rng))
        if len(both) < L:
            rand["act_or_bind"].append(random_metrics(L, both, rng))

        for name, (model, in_dims) in models.items():
            zeros = {k: torch.zeros(1, d, device=dev) for k, d in in_dims.items() if k != "sequence"}
            with torch.no_grad():
                feats = {"sequence": pooled_full, **zeros}
                logits, _ = model(feats)
                lg = logits[0].masked_fill(~ec4_mask, float("-inf"))
                tgt = int(lg.argmax().item())
            pred_ec = id2ec[tgt]
            correct = pred_ec in r["ec_numbers"]

            alphas = torch.linspace(1.0 / args.ig_steps, 1.0, args.ig_steps, device=dev)
            grad_sum = torch.zeros_like(x[0])
            for b0 in range(0, args.ig_steps, args.ig_batch):
                al = alphas[b0:b0 + args.ig_batch]
                k = al.numel()
                interp = (xb + al.view(-1, 1, 1) * (x - xb)).detach().requires_grad_(True)
                pooled = pooled_from(ids.repeat(k, 1), mask.repeat(k, 1), interp)
                feats = {"sequence": pooled, **{kk: v.repeat(k, 1) for kk, v in zeros.items()}}
                logits, _ = model(feats)
                logits[:, tgt].sum().backward()
                grad_sum += interp.grad.sum(0).detach()
            ig = ((x[0] - xb[0]) * grad_sum / args.ig_steps).sum(-1)  # (T,)
            score = ig[1:T - 1].abs().float().cpu().numpy()
            if len(score) != L:
                continue
            m_act = site_metrics(score, act)
            res[name]["act_site"].append(m_act)
            (res[name]["act_site_correct"] if correct else res[name]["act_site_wrong"]).append(m_act)
            if len(both) < L:
                res[name]["act_or_bind"].append(site_metrics(score, both))
            res[name]["cases"].append({
                "accession": acc, "length": L, "true_ec": r["ec_numbers"], "pred_ec": pred_ec,
                "correct": correct, "act_site_1based": [p + 1 for p in act],
                "top10_1based": [int(i) + 1 for i in np.argsort(-score)[:10]],
                "auroc": round(m_act["auroc"], 4), "enrich10": round(m_act["enrich10"], 2),
            })
        if (ci + 1) % 25 == 0:
            print(f"  [{ci+1}/{len(cands)}] {time.time()-t0:.0f}s  "
                  f"v1 AUROC={np.mean([m['auroc'] for m in res['v1_hierprior']['act_site']]):.3f} "
                  f"v2 AUROC={np.mean([m['auroc'] for m in res['v2']['act_site']]):.3f} "
                  f"cos(min)={min(cos_checks):.4f}", flush=True)

    out = {
        "tier": args.tier, "split": "test", "context": "seq_only (3Di/text/domain zero)",
        "method": f"Integrated Gradients through ESM-2 650M word embeddings, {args.ig_steps} steps, <mask> baseline",
        "target": "model's own top-1 predicted EC4 (no label peeking)",
        "n_candidates": len(cands),
        "cache_consistency_cosine": {"min": round(min(cos_checks), 5), "mean": round(float(np.mean(cos_checks)), 5)},
        "random_baseline": {k: agg(v) for k, v in rand.items()},
        "models": {},
    }
    for name, d in res.items():
        d["cases"].sort(key=lambda c: (c["correct"], c["auroc"]), reverse=True)
        out["models"][name] = {
            "act_site": agg(d["act_site"]),
            "act_or_bind": agg(d["act_or_bind"]),
            "act_site_pred_correct": agg(d["act_site_correct"]),
            "act_site_pred_wrong": agg(d["act_site_wrong"]),
            "top_cases": d["cases"][:12],
        }
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("cache_consistency_cosine", "random_baseline")}, indent=2))
    for name, d in out["models"].items():
        print(name, json.dumps({k: d[k] for k in ("act_site", "act_or_bind", "act_site_pred_correct", "act_site_pred_wrong")}, indent=1))
    print(f"[DONE] {args.out}")


if __name__ == "__main__":
    main()
