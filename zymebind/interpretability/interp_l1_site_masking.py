#!/usr/bin/env python
"""L1 因果检验：遮蔽真实催化位点 vs 遮蔽等量随机残基，对目标 EC4 预测概率的影响（内部 test）。

- 模型：ZymeBind v2（p1m1_benchclusterfree），纯序列语境（3Di/text/domain 全零）；
- 目标：模型自身 top-1 预测 EC4（不偷看真值）；只统计预测正确的蛋白为主分析，全部蛋白为副分析；
- 遮蔽：把位点残基 token 换成 <mask>，重新跑 ESM-2 + 同口径均值池化；
- 对照：从非位点残基中均匀抽取等量位置，20 次取均值；
- 统计：Δ = p0 − p_masked；配对 Wilcoxon（act vs 随机均值）、act 下降大于随机的蛋白比例、
  以及每条蛋白的经验置换 p 值中位数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import load_split  # noqa: E402
from models.zymebind_v2 import load_v2_checkpoint  # noqa: E402
from training_zymebind import TIERS  # noqa: E402

ESM_NAME = "facebook/esm2_t33_650M_UR50D"
N_RAND = 20


def main():
    dev = "cuda:0"
    tier = "p1m1_benchclusterfree"
    rng = np.random.default_rng(42)
    ec2id = json.loads((ROOT / TIERS[tier][1]).read_text())["ec2id"]
    id2ec = {int(v): k for k, v in ec2id.items()}
    ec4_mask = torch.tensor([id2ec[i].count(".") == 3 for i in range(len(ec2id))], device=dev)
    sites = json.loads((ROOT / "data/processed/active_sites_eval.json").read_text())["annotations"]
    rows = load_split(ROOT / TIERS[tier][0] / "test.tsv")
    cands = sorted([r for r in rows if r["accession"] in sites and sites[r["accession"]]["act_site"]
                    and 30 <= len(r["sequence"]) <= 800 and r["ec_numbers"]], key=lambda r: r["accession"])[:400]

    _, _, model, _, ck = load_v2_checkpoint(ROOT / "checkpoints/zymebind_v2_p1m1_benchclusterfree.pt", dev)
    model.eval()
    in_dims = ck["in_dims"]
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ESM_NAME, local_files_only=True)
    esm = AutoModel.from_pretrained(ESM_NAME, torch_dtype=torch.bfloat16, local_files_only=True).to(dev).eval()

    @torch.no_grad()
    def probs_for(ids_batch: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(ids_batch)
        out = []
        for i in range(0, ids_batch.size(0), 16):
            b = ids_batch[i:i + 16]
            m = mask[i:i + 16]
            h = esm(input_ids=b, attention_mask=m).last_hidden_state.float()
            pooled = h.mean(1)  # 无 padding，等价于 mean(h * attention_mask)
            feats = {"sequence": pooled, **{k: torch.zeros(len(b), d, device=dev) for k, d in in_dims.items() if k != "sequence"}}
            logits, _ = model(feats)
            out.append(torch.sigmoid(logits))
        return torch.cat(out)

    recs = {"act_site": [], "act_or_bind": []}
    for ci, r in enumerate(cands):
        seq = r["sequence"]
        L = len(seq)
        ids = tok([seq], return_tensors="pt")["input_ids"].to(dev)  # (1, L+2)
        p_base = probs_for(ids)[0]
        tgt = int(p_base.masked_fill(~ec4_mask, -1).argmax().item())
        correct = id2ec[tgt] in r["ec_numbers"]
        act = sorted({p - 1 for p in sites[r["accession"]]["act_site"] if 1 <= p <= L})
        both = sorted(set(act) | {p - 1 for p in sites[r["accession"]]["binding"] if 1 <= p <= L})
        for key, pos in (("act_site", act), ("act_or_bind", both)):
            k = len(pos)
            if k == 0 or k > L // 3:
                continue
            others = np.setdiff1d(np.arange(L), pos)
            variants = [pos] + [rng.choice(others, size=k, replace=False).tolist() for _ in range(N_RAND)]
            batch = ids.repeat(len(variants), 1)
            for vi, pp in enumerate(variants):
                batch[vi, torch.tensor(pp, device=dev) + 1] = tok.mask_token_id
            p = probs_for(batch)[:, tgt].cpu().numpy()
            p0 = float(p_base[tgt].item())
            d_site = p0 - p[0]
            d_rand = p0 - p[1:]
            recs[key].append({
                "accession": r["accession"], "correct": bool(correct), "n_sites": k, "p0": p0,
                "drop_site": float(d_site), "drop_rand_mean": float(d_rand.mean()),
                "perm_p": float((np.sum(d_rand >= d_site) + 1) / (N_RAND + 1)),
            })
        if (ci + 1) % 50 == 0:
            a = recs["act_site"]
            print(f"  [{ci+1}/{len(cands)}] act: site_drop={np.mean([x['drop_site'] for x in a]):.4f} "
                  f"rand_drop={np.mean([x['drop_rand_mean'] for x in a]):.4f}", flush=True)

    def summarize(items):
        if len(items) < 5:
            return {"n": len(items)}
        ds = np.array([x["drop_site"] for x in items])
        dr = np.array([x["drop_rand_mean"] for x in items])
        stat = wilcoxon(ds, dr, alternative="greater")
        return {
            "n": len(items),
            "mean_p0": round(float(np.mean([x["p0"] for x in items])), 4),
            "mean_drop_site": round(float(ds.mean()), 4),
            "mean_drop_random": round(float(dr.mean()), 4),
            "median_drop_site": round(float(np.median(ds)), 4),
            "median_drop_random": round(float(np.median(dr)), 4),
            "frac_site_gt_random": round(float((ds > dr).mean()), 4),
            "median_perm_p": round(float(np.median([x["perm_p"] for x in items])), 4),
            "frac_perm_p_le_0.05": round(float(np.mean([x["perm_p"] <= 0.05 for x in items])), 4),
            "wilcoxon_greater_p": float(stat.pvalue),
        }

    out = {"model": "zymebind_v2_p1m1_benchclusterfree", "split": "test", "context": "seq_only",
           "n_random_controls": N_RAND, "results": {}}
    for key, items in recs.items():
        out["results"][key] = {
            "pred_correct": summarize([x for x in items if x["correct"]]),
            "all": summarize(items),
        }
    p = ROOT / "results/interp_l1_site_masking_v2_p1m1_test.json"
    out["per_protein"] = recs
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["results"], indent=2))
    print(f"[DONE] {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
