#!/usr/bin/env python
"""ONE-SHOT EXTERNAL BENCHMARK EVALUATION (NEW-392 + Price-149, including Price-149 <40% subset).

Evaluates:
  - Locked ZymeBind single (m2_dualplm_large) & 4-ensemble (ens_dualplm_4) across NEW-392 contexts & Price-149
  - Same-split retrained SOTA checkpoints (CLEAN, EnzymeHunter, DeepECtransformer, HIT-EC, GraphEC-LD, ECRECer)
Strictly uses val-locked thresholds tau* from results/val_locked_thresholds_for_blind_test.json
and reports BOTH CAFA/EC-Bench and CLEAN/EnzymeHunter Dual-Protocol Metrics.
Refuses to overwrite results/blind_test_external_new392_price149_final_once.json if it exists.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/data/cs/ZymeBind")
sys.path.insert(0, str(ROOT))

from data.cached_dataset import EmbeddingCache
from data.dataset import build_domain_vocab, load_split
from evaluation.ec_dual_metrics import compute_dual_ec_metrics
from evaluation.eval_blind_test_common_clean_final_once import eval_sota_ckpt, stack_emb
from evaluation.eval_external_benchmarks import (
    build_features,
    domain_multihot,
    load_benchmarks,
    load_main_tables,
)
from evaluation.eval_external_v2 import FROZEN_TABLE, load_obs3di_matrix, load_table_dict, predict_v2
from models.zymebind_v2 import load_v2_checkpoint
from training_zymebind import TIERS
from training_zymebind_v4_dualplm import ProstT5AACache

OUT_JSON = ROOT / "results/blind_test_external_new392_price149_final_once.json"
CKPT_DIR = Path("/mnt/data/cs/ZymeBind_sota_baselines/retrained_ckpts")


def build_label_matrix(rows: list[dict], u_idx: dict[str, int], C: int):
    Y = np.zeros((len(rows), C), dtype=np.int8)
    keep = []
    for i, r in enumerate(rows):
        ecs = [e.strip() for e in (r.get("ec") or "").replace(";", ",").split(",") if e.strip()]
        hit = False
        for e in ecs:
            if e in u_idx:
                Y[i, u_idx[e]] = 1
                hit = True
        if hit:
            keep.append(i)
    return Y, np.array(keep, dtype=np.int64)


def main():
    if OUT_JSON.exists():
        raise RuntimeError(f"REFUSING TO OVERWRITE ONE-SHOT EXTERNAL RESULT: {OUT_JSON}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    locked_taus = json.loads((ROOT / "results/val_locked_thresholds_for_blind_test.json").read_text())

    split_rel, vocab_rel, _ = TIERS["p1m2_benchclusterfree"]
    m2_ec2id = json.loads((ROOT / vocab_rel).read_text())["ec2id"]
    union = sorted(e for e in m2_ec2id if e.count(".") == 3)
    u_idx = {e: i for i, e in enumerate(union)}
    src = np.array([m2_ec2id[e] for e in union])
    C = len(union)

    ec1_list = sorted({".".join(e.split(".")[:1]) for e in union})
    ec2_list = sorted({".".join(e.split(".")[:2]) for e in union})
    ec3_list = sorted({".".join(e.split(".")[:3]) for e in union})
    hier = (None, None, None, len(ec1_list), len(ec2_list), len(ec3_list))

    train_rows_all = load_split(ROOT / split_rel / "train.tsv")
    domain2id = build_domain_vocab(train_rows_all)
    train_rows, tr_labs = [], []
    for r in train_rows_all:
        ls = [u_idx[e] for e in r["ec_numbers"] if e in u_idx]
        if ls:
            train_rows.append(r); tr_labs.append(ls)
    prim_y = np.array([ls[0] for ls in tr_labs], dtype=np.int64)
    tr_cnt = dict(Counter(union[j] for ls in tr_labs for j in ls))

    benchmarks = dict(load_benchmarks())
    full_meta = load_main_tables()
    frozen_meta = load_table_dict(FROZEN_TABLE)
    obs3di_cache = EmbeddingCache("threedi_new392_observed")
    prost_c = ProstT5AACache()

    all_rows = [r for rows in benchmarks.values() for r in rows]
    print("[ext] Building ESM-2 & BioBERT features for NEW-392 & Price-149...", flush=True)
    feats_full = build_features(all_rows, full_meta, str(device))
    feats_frozen = build_features(all_rows, frozen_meta, str(device))

    # Load Price-149 <40% homology audit ids
    price_hom = json.loads((ROOT / "results/v2_price149_p1m1_homology.json").read_text())
    low_hom_ids = set(price_hom.get("lt40_ids", []))

    # Load M2 train embeddings for SOTA centroid/LD inference
    seq_c = EmbeddingCache("sequence")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")
    X_tr = stack_emb(train_rows, seq_c, 1280)
    S_tr = stack_emb(train_rows, tdi_c, 1024)

    ckpts = {
        "m2_dualplm": "checkpoints/zymebind_v4_dualplm_p1m2_benchclusterfree.pt",
        "m2_dualplm_large": "checkpoints/zymebind_v4large_dualplm_p1m2_benchclusterfree.pt",
        "m2_v45_s42": "checkpoints/zymebind_v45_m2_s42.pt",
        "m2_v45_s2026": "checkpoints/zymebind_v45_m2_s2026.pt",
    }

    results = {
        "protocol": "ONE-SHOT LOCKED EXTERNAL BENCHMARK EVALUATION (val-locked models & tau*)",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure": "See docs/ZymeBind_v2+_外部基准使用登记与最终评测协议.md for prior exploratory runs; this file is the single locked final_once pass.",
        "benchmarks": {},
    }

    for bname, rows in benchmarks.items():
        print(f"\n=== External Benchmark: {bname} (raw n={len(rows)}) ===", flush=True)
        Y_all, keep = build_label_matrix(rows, u_idx, C)
        Y_eval = Y_all[keep]

        esm_np = np.stack([feats_full["sequence"][r["accession"]] for r in rows]).astype(np.float32)
        prost_np = np.stack([
            prost_c.get(r["accession"]).numpy() if prost_c.get(r["accession"]) is not None else np.zeros(1024, dtype=np.float32)
            for r in rows
        ]).astype(np.float32)
        dual_seq = np.concatenate([esm_np, prost_np], axis=1).astype(np.float32)

        zero_tri = np.zeros((len(rows), 1024), dtype=np.float32)
        obs_tri = load_obs3di_matrix(rows, obs3di_cache)
        zero_txt = np.zeros((len(rows), 768), dtype=np.float32)
        zero_dom = np.zeros((len(rows), len(domain2id)), dtype=np.float32)
        full_txt = np.stack([feats_full["text"][r["accession"]] for r in rows]).astype(np.float32)
        full_dom = np.stack([domain_multihot(r["accession"], full_meta, domain2id) for r in rows]).astype(np.float32)
        froz_txt = np.stack([feats_frozen["text"][r["accession"]] for r in rows]).astype(np.float32)
        froz_dom = np.stack([domain_multihot(r["accession"], frozen_meta, domain2id) for r in rows]).astype(np.float32)

        b_res = {"n_raw": len(rows), "n_in_m2_vocab": int(len(keep)), "sota_same_split": {}, "zymebind_locked": {}}

        # 1. Same-split SOTA models
        for sota_name in ["sota_clean", "sota_enzymehunter", "sota_deepec_transformer", "sota_ecrecer", "sota_hitec", "sota_graphec_ld"]:
            ck = CKPT_DIR / f"{sota_name}.pt"
            if not ck.exists():
                continue
            S_in = obs_tri if bname == "new392" else zero_tri
            P = eval_sota_ckpt(sota_name, ck, X_tr, S_tr, prim_y, tr_labs, esm_np, S_in, C, hier, device)
            tau_lock = locked_taus.get(sota_name, 0.5)
            m = compute_dual_ec_metrics(P[keep], Y_eval, union, train_counts=tr_cnt, locked_threshold=tau_lock)
            b_res["sota_same_split"][sota_name] = m
            print(f"  [{bname} | {sota_name:24s}] Fmax={m['fmax']:.4f} wF1={m['weighted_f1']:.4f} wAUC={m['weighted_auc']:.4f} AUPR-M={m['aupr_macro']:.4f}", flush=True)
            if bname == "price149" and low_hom_ids:
                sub = [idx for idx, orig_i in enumerate(keep) if rows[orig_i]["accession"] in low_hom_ids]
                if sub:
                    ms = compute_dual_ec_metrics(P[keep][sub], Y_eval[sub], union, train_counts=tr_cnt, locked_threshold=tau_lock)
                    b_res.setdefault("price149_lt40", {})[sota_name] = ms

        # 2. Locked ZymeBind models
        contexts = {
            "seq_only": (dual_seq, zero_tri, zero_txt, zero_dom),
            "observed3di": (dual_seq, obs_tri, zero_txt, zero_dom),
            "frozen_2022_04": (dual_seq, zero_tri, froz_txt, froz_dom),
            "frozen_2022_04_obs3di": (dual_seq, obs_tri, froz_txt, froz_dom),
            "full_context": (dual_seq, zero_tri, full_txt, full_dom),
            "full_context_obs3di": (dual_seq, obs_tri, full_txt, full_dom),
        }
        if bname == "price149":
            contexts = {"seq_only": contexts["seq_only"]}

        zb_preds = {k: {} for k in ckpts}
        for k, ck in ckpts.items():
            _, _, stage2, _, _ = load_v2_checkpoint(ROOT / ck, str(device))
            for cname, (sq, tr, tx, dm) in contexts.items():
                zb_preds[k][cname] = predict_v2(stage2, sq, tr, tx, dm, str(device))[:, src]
            del stage2
            torch.cuda.empty_cache()

        for cname in contexts:
            P_single = zb_preds["m2_dualplm_large"][cname]
            P_ens = sum(zb_preds[k][cname] for k in ckpts) / len(ckpts)
            tau_key_s = "zb_single_seq_only" if cname == "seq_only" else ("zb_single_seq_struct" if cname == "observed3di" else "zb_single_full")
            tau_key_e = "zb_ens4_seq_only" if cname == "seq_only" else ("zb_ens4_seq_struct" if cname == "observed3di" else "zb_ens4_full")

            ms = compute_dual_ec_metrics(P_single[keep], Y_eval, union, train_counts=tr_cnt, locked_threshold=locked_taus.get(tau_key_s, 0.5))
            me = compute_dual_ec_metrics(P_ens[keep], Y_eval, union, train_counts=tr_cnt, locked_threshold=locked_taus.get(tau_key_e, 0.5))
            b_res["zymebind_locked"][f"single_{cname}"] = ms
            b_res["zymebind_locked"][f"ens4_{cname}"] = me
            print(f"  [{bname} | single_{cname:22s}] Fmax={ms['fmax']:.4f} wF1={ms['weighted_f1']:.4f} wAUC={ms['weighted_auc']:.4f} AUPR-M={ms['aupr_macro']:.4f}", flush=True)
            print(f"  [{bname} | ens4_{cname:24s}] Fmax={me['fmax']:.4f} wF1={me['weighted_f1']:.4f} wAUC={me['weighted_auc']:.4f} AUPR-M={me['aupr_macro']:.4f}", flush=True)

            if bname == "price149" and low_hom_ids:
                sub = [idx for idx, orig_i in enumerate(keep) if rows[orig_i]["accession"] in low_hom_ids]
                if sub:
                    b_res.setdefault("price149_lt40", {})[f"single_{cname}"] = compute_dual_ec_metrics(P_single[keep][sub], Y_eval[sub], union, train_counts=tr_cnt, locked_threshold=locked_taus.get(tau_key_s, 0.5))
                    b_res.setdefault("price149_lt40", {})[f"ens4_{cname}"] = compute_dual_ec_metrics(P_ens[keep][sub], Y_eval[sub], union, train_counts=tr_cnt, locked_threshold=locked_taus.get(tau_key_e, 0.5))

        results["benchmarks"][bname] = b_res

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[DONE ONE-SHOT EXTERNAL] Saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
