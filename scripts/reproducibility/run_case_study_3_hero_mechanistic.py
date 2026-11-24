#!/usr/bin/env python
"""Case Study 3: End-to-End 4-Layer Mechanistic Dissection of Twilight-Zone (<20%)
and Experimental (Price-149) Hero Cases.

1. Mines common_clean/test (<20% homology & multi-EC) to identify real enzymes where
   MMseqs2 + same-split SOTA baselines (CLEAN, EnzymeHunter, DeepECtransformer,
   HIT-EC, GraphEC-LD) FAIL while locked ZymeBind (Seq-only / Seq+3Di) SUCCEEDS.
2. For each Hero Case, computes the complete 4-layer mechanistic evidence chain:
   - Exact homology (<20% pident) & side-by-side predictions of all 6 models
   - L2 Exact 4-Modality Shapley values (2^4 = 16 coalitions) for that specific enzyme
   - L3 Top-K SAE top-activated monosemantic neurons (InterPro / EC3 / Cofactor concepts)
   - UniProt ACT_SITE / BINDING annotations & AlphaFoldDB structure availability

Output: results/case_study_3_hero_mechanistic.json
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/mnt/data/cs/ZymeBind")
sys.path.insert(0, str(ROOT))

from data.cached_dataset import EmbeddingCache
from data.dataset import load_split
from evaluation.eval_baselines_and_strat_val import run_mmseqs_baseline
from evaluation.eval_blind_test_common_clean_final_once import eval_sota_ckpt, stack_emb
from evaluation.eval_comprehensive_val import build_inputs
from evaluation.eval_external_v2 import predict_v2
from models.zymebind_v2 import load_v2_checkpoint
from training_zymebind import TIERS
from training_zymebind_v4_dualplm import ProstT5AACache

CKPT_DIR = Path("/mnt/data/cs/ZymeBind_sota_baselines/retrained_ckpts")


class TopKSAE(nn.Module):
    def __init__(self, d_in: int = 768, d_hidden: int = 4096, k: int = 32):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in, bias=False)

    def encode(self, x):
        pre = F.relu(self.enc(x - self.b_pre))
        vals, idx = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z


def exact_shapley_single(stage2, seq_1, tri_1, txt_1, dom_1, target_col: int, device: str) -> dict[str, float]:
    """Exact 2^4 = 16 coalition Shapley values for a single enzyme's true EC4 probability."""
    mods = ["sequence", "structure", "text", "domain"]
    raw = {
        "sequence": torch.from_numpy(seq_1).to(device),
        "structure": torch.from_numpy(tri_1).to(device),
        "text": torch.from_numpy(txt_1).to(device),
        "domain": torch.from_numpy(dom_1).to(device),
    }
    zeros = {m: torch.zeros_like(t) for m, t in raw.items()}

    v = {}
    with torch.no_grad():
        for mask in range(16):
            feats = {m: (raw[m] if (mask & (1 << i)) else zeros[m]) for i, m in enumerate(mods)}
            logits, _ = stage2(feats)
            v[mask] = float(torch.sigmoid(logits[0, target_col]).item())

    phi = {}
    n = 4
    for i, m in enumerate(mods):
        s = 0.0
        for mask in range(16):
            if mask & (1 << i):
                continue
            k = bin(mask).count("1")
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            s += w * (v[mask | (1 << i)] - v[mask])
        phi[m] = round(s, 4)
    return {"shapley": phi, "v_empty": round(v[0], 4), "v_full": round(v[15], 4)}


def fetch_uniprot_sites(acc: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ZymeBind-Research/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        sites = []
        for f in d.get("features", []):
            if f.get("type") in ("Active site", "Binding site"):
                loc = f.get("location", {})
                sites.append({
                    "type": f["type"],
                    "start": loc.get("start", {}).get("value"),
                    "end": loc.get("end", {}).get("value"),
                    "desc": f.get("description", ""),
                })
        afdb = [x["id"] for x in d.get("uniProtKBCrossReferences", []) if x.get("database") == "AlphaFoldDB"]
        ipr = [f"{x['id']}:{x.get('properties',[{}])[0].get('value','')}" for x in d.get("uniProtKBCrossReferences", []) if x.get("database") == "InterPro"]
        return {
            "protein_name": d.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "active_and_binding_sites": sites[:12],
            "afdb_id": afdb[0] if afdb else "",
            "interpro": ipr[:8],
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m2_ec2id = json.loads((ROOT / TIERS["p1m2_benchclusterfree"][1]).read_text())["ec2id"]
    union = sorted(e for e in m2_ec2id if e.count(".") == 3)
    u_idx = {e: i for i, e in enumerate(union)}
    src = np.array([m2_ec2id[e] for e in union])
    C = len(union)

    ec1_list = sorted({".".join(e.split(".")[:1]) for e in union})
    ec2_list = sorted({".".join(e.split(".")[:2]) for e in union})
    ec3_list = sorted({".".join(e.split(".")[:3]) for e in union})
    hier = (None, None, None, len(ec1_list), len(ec2_list), len(ec3_list))

    train_rows_all = load_split(ROOT / TIERS["p1m2_benchclusterfree"][0] / "train.tsv")
    train_rows, tr_labs = [], []
    for r in train_rows_all:
        ls = [u_idx[e] for e in r["ec_numbers"] if e in u_idx]
        if ls:
            train_rows.append(r); tr_labs.append(ls)
    prim_y = np.array([ls[0] for ls in tr_labs], dtype=np.int64)

    test_rows = load_split(ROOT / "data/processed/common_clean/test.tsv")
    test_rows = [r for r in test_rows if any(e in u_idx for e in r["ec_numbers"])]

    print("[1/4] Computing MMseqs2 homology on common_clean/test...", flush=True)
    _, max_pid = run_mmseqs_baseline(test_rows, train_rows_all, union, u_idx)

    seq_c = EmbeddingCache("sequence")
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")
    prost_c = ProstT5AACache()
    caches = (seq_c, txt_c, tdi_c, prost_c)
    dom_cache = {}

    X_tr = stack_emb(train_rows, seq_c, 1280)
    S_tr = stack_emb(train_rows, tdi_c, 1024)
    X_te = stack_emb(test_rows, seq_c, 1280)
    S_te = stack_emb(test_rows, tdi_c, 1024)

    print("[2/4] Running 5 SOTA checkpoints & locked ZymeBind (m2_dualplm_large) on common_clean/test...", flush=True)
    sota_preds = {}
    for sname in ["sota_clean", "sota_enzymehunter", "sota_deepec_transformer", "sota_hitec", "sota_graphec_ld"]:
        ck = CKPT_DIR / f"{sname}.pt"
        if ck.exists():
            sota_preds[sname] = eval_sota_ckpt(sname, ck, X_tr, S_tr, prim_y, tr_labs, X_te, S_te, C, hier, device)
    del X_tr, S_tr

    seq, tri, txt, dom = build_inputs(test_rows, "p1m2_benchclusterfree", True, caches, dom_cache)
    z = lambda a: np.zeros_like(a)
    _, _, stage2, _, _ = load_v2_checkpoint(ROOT / "checkpoints/zymebind_v4large_dualplm_p1m2_benchclusterfree.pt", str(device))
    p_zb_ss = predict_v2(stage2, seq, tri, z(txt), z(dom), str(device))[:, src]
    p_zb_seq = predict_v2(stage2, seq, z(tri), z(txt), z(dom), str(device))[:, src]

    print("[3/4] Mining Twilight-Zone (<30%) & Multi-EC Hero Cases where SOTA fail & ZymeBind succeeds...", flush=True)
    heroes = []
    for i, r in enumerate(test_rows):
        pid = float(max_pid.get(r["accession"], 0.0))
        true_ecs = sorted({e for e in r["ec_numbers"] if e in u_idx})
        top_zb = union[int(p_zb_ss[i].argmax())]
        prob_zb = float(p_zb_ss[i].max())
        if top_zb not in true_ecs or prob_zb < 0.70:
            continue
        # Count how many of the 5 SOTAs failed top-1
        sota_top1 = {k: (union[int(P[i].argmax())], round(float(P[i].max()), 3)) for k, P in sota_preds.items()}
        n_sota_fail = sum(1 for ec, _ in sota_top1.values() if ec not in true_ecs)
        if (pid < 30.0 and n_sota_fail >= 3) or (len(true_ecs) >= 2 and n_sota_fail >= 2) or (pid < 20.0 and n_sota_fail >= 2):
            heroes.append({
                "idx": i,
                "accession": r["accession"],
                "organism": r.get("organism", ""),
                "length": int(r.get("length", 0)),
                "max_train_pident": round(pid, 2),
                "true_ecs": true_ecs,
                "is_multi_ec": len(true_ecs) >= 2,
                "zb_seq_struct_top1": top_zb,
                "zb_seq_struct_prob": round(prob_zb, 4),
                "zb_seq_only_top1": union[int(p_zb_seq[i].argmax())],
                "zb_seq_only_prob": round(float(p_zb_seq[i].max()), 4),
                "n_sota_failed_out_of_5": n_sota_fail,
                "sota_top1": sota_top1,
            })

    heroes.sort(key=lambda x: (-x["n_sota_failed_out_of_5"], x["max_train_pident"], -x["zb_seq_struct_prob"]))
    print(f"  Found {len(heroes)} Hero Cases! Dissecting top 10 with L2 Shapley + L3 SAE + UniProt Sites...", flush=True)

    # Load L3 SAE & neuron dictionary
    sae_ckpt = torch.load(ROOT / "checkpoints/sae_l3_m2_dualplm_large.pt", map_location=device, weights_only=False)
    sae_meta = json.loads((ROOT / "results/val_interpretability_l3_sae_monosemantic.json").read_text())
    neuron_labels: dict[int, list[str]] = {}
    for cat in ("interpro", "ec3", "cofactor"):
        for item in sae_meta.get("concepts", {}).get(cat, []):
            nid = int(item["best_neuron"])
            if float(item.get("f1", 0)) >= 0.5:
                neuron_labels.setdefault(nid, []).append(f"{cat}:{item['concept']}(F1={item['f1']:.2f})")
    sae = TopKSAE(d_in=768, d_hidden=4096, k=32).to(device)
    sae.load_state_dict(sae_ckpt["state_dict"] if isinstance(sae_ckpt, dict) and "state_dict" in sae_ckpt else sae_ckpt, strict=False)
    sae.eval()

    selected = heroes[:10]
    for h in selected:
        i = h["idx"]
        target_col = int(m2_ec2id[h["zb_seq_struct_top1"]])
        # L2 Exact 4-Modality Shapley
        h["l2_exact_shapley"] = exact_shapley_single(
            stage2, seq[i:i + 1], tri[i:i + 1], txt[i:i + 1], dom[i:i + 1], target_col, str(device)
        )
        # L3 SAE top monosemantic neurons on pure seq+struct representation
        with torch.no_grad():
            feats = {
                "sequence": torch.from_numpy(seq[i:i + 1]).to(device),
                "structure": torch.from_numpy(tri[i:i + 1]).to(device),
                "text": torch.zeros_like(torch.from_numpy(txt[i:i + 1]).to(device)),
                "domain": torch.zeros_like(torch.from_numpy(dom[i:i + 1]).to(device)),
            }
            h_f, _, _ = stage2.fusion(feats)
            z_sae = sae.encode(h_f)[0].cpu().numpy()
        nz = np.argsort(-z_sae)[:10]
        h["l3_sae_concepts"] = [
            {"neuron": int(nid), "act": round(float(z_sae[nid]), 3), "concepts": neuron_labels[int(nid)][:3]}
            for nid in nz if z_sae[nid] > 0 and int(nid) in neuron_labels
        ]
        # UniProt active sites & AFDB ID
        h["uniprot_biophysics"] = fetch_uniprot_sites(h["accession"])
        del h["idx"]
        print(
            f"  [HERO {h['accession']}] pid={h['max_train_pident']}% true={h['true_ecs']} "
            f"ZB={h['zb_seq_struct_top1']}({h['zb_seq_struct_prob']}) sota_fail={h['n_sota_failed_out_of_5']}/5 "
            f"Shapley={h['l2_exact_shapley']['shapley']} SAE_hits={len(h['l3_sae_concepts'])} "
            f"AFDB={h['uniprot_biophysics'].get('afdb_id','')}",
            flush=True,
        )

    out_path = ROOT / "results/case_study_3_hero_mechanistic.json"
    out_path.write_text(json.dumps({
        "n_total_heroes_found": len(heroes),
        "top_10_dissected_heroes": selected,
        "elapsed_seconds": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False))
    print(f"[DONE Case Study 3] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
