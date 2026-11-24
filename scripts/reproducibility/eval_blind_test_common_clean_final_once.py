#!/usr/bin/env python
"""ONE-SHOT INDEPENDENT BLIND TEST on data/processed/common_clean/test.tsv (n=2,841, 673 clusters).

Strictly uses:
  - Locked checkpoints from results/final_selection_locked_v45.json
  - Val-locked thresholds tau* from results/val_locked_thresholds_for_blind_test.json
  - Same-split retrained SOTA checkpoints from /mnt/data/cs/ZymeBind_sota_baselines/retrained_ckpts/
Refuses to overwrite results/blind_test_common_clean_final_once.json if it already exists.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/mnt/data/cs/ZymeBind")
sys.path.insert(0, str(ROOT))

from data.cached_dataset import EmbeddingCache
from data.dataset import load_split
from evaluation.ec_dual_metrics import compute_dual_ec_metrics
from evaluation.eval_ablation_and_stats_valonly import load_acc2rep
from evaluation.eval_baselines_and_strat_val import (
    calibration_metrics,
    run_knn_baseline,
    run_mmseqs_baseline,
)
from evaluation.eval_comprehensive_val import build_inputs
from evaluation.eval_external_v2 import predict_v2
from evaluation.summarize_sota_pk_val import cluster_bootstrap_diff
from evaluation.train_eval_sota_baselines_val import (
    CLEANLayerNormNet,
    DeepECTransformerNet,
    ECRECerHDMLFNet,
    EnzymeHunterNet,
    GraphECStructEncoder,
    HITECNet,
)
from models.zymebind_v2 import load_v2_checkpoint
from training_zymebind import TIERS
from training_zymebind_v4_dualplm import ProstT5AACache

OUT_JSON = ROOT / "results/blind_test_common_clean_final_once.json"
CKPT_DIR = Path("/mnt/data/cs/ZymeBind_sota_baselines/retrained_ckpts")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stack_emb(rows, cache, dim):
    arr = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        v = cache.get(r["accession"])
        if v is not None:
            arr[i] = v.numpy()
    return arr


@torch.no_grad()
def eval_sota_ckpt(name: str, ckpt_path: Path, X_tr, S_tr, prim_y, tr_labs, X_te, S_te, C, hier, device):
    state = torch.load(ckpt_path, map_location=device)
    _, _, _, n1, n2, n3 = hier

    if name == "sota_clean":
        model = CLEANLayerNormNet(1280, 512, 256, 0.1).to(device)
        model.load_state_dict(state); model.eval()
        cents = torch.zeros((C, 256), device=device)
        cnts = torch.zeros((C, 1), device=device)
        for i in range(0, len(X_tr), 4096):
            zb = model(torch.from_numpy(X_tr[i:i+4096]).to(device))
            yb = torch.from_numpy(prim_y[i:i+4096]).to(device)
            cents.index_add_(0, yb, zb)
            cnts.index_add_(0, yb, torch.ones((len(yb), 1), device=device))
        cents = F.normalize(cents / cnts.clamp_min(1.0), dim=-1)
        has_cent = cnts.squeeze(-1) > 0
        out = np.zeros((len(X_te), C), dtype=np.float32)
        for i in range(0, len(X_te), 1024):
            zb = model(torch.from_numpy(X_te[i:i+1024]).to(device))
            sim = zb @ cents.T
            sim[:, ~has_cent] = -2.0
            out[i:i+1024] = F.softmax(sim / 0.05, dim=-1).cpu().numpy()
        return out

    if name == "sota_enzymehunter":
        model = EnzymeHunterNet(1280, 1024, 512, 256, C, drop=0.1).to(device)
        model.load_state_dict(state); model.eval()
        cents = torch.zeros((C, 256), device=device)
        cnts = torch.zeros((C, 1), device=device)
        for i in range(0, len(X_tr), 4096):
            zb, _ = model(torch.from_numpy(X_tr[i:i+4096]).to(device), torch.from_numpy(S_tr[i:i+4096]).to(device))
            zb = F.normalize(zb, dim=-1)
            yb = torch.from_numpy(prim_y[i:i+4096]).to(device)
            cents.index_add_(0, yb, zb)
            cnts.index_add_(0, yb, torch.ones((len(yb), 1), device=device))
        cents = F.normalize(cents / cnts.clamp_min(1.0), dim=-1)
        has_cent = cnts.squeeze(-1) > 0
        out = np.zeros((len(X_te), C), dtype=np.float32)
        for i in range(0, len(X_te), 1024):
            zb, _ = model(torch.from_numpy(X_te[i:i+1024]).to(device), torch.from_numpy(S_te[i:i+1024]).to(device))
            zb = F.normalize(zb, dim=-1)
            sim = zb @ cents.T
            sim[:, ~has_cent] = -2.0
            out[i:i+1024] = F.softmax(sim / 0.05, dim=-1).cpu().numpy()
        return out

    if name == "sota_deepec_transformer":
        model = DeepECTransformerNet(1280, 256, 8, C).to(device)
        model.load_state_dict(state); model.eval()
        out = np.zeros((len(X_te), C), dtype=np.float32)
        for i in range(0, len(X_te), 1024):
            out[i:i+1024] = torch.sigmoid(model(torch.from_numpy(X_te[i:i+1024]).to(device))).cpu().numpy()
        return out

    if name == "sota_ecrecer":
        model = ECRECerHDMLFNet(1280, n1, n2, n3, C, h_size=512).to(device)
        model.load_state_dict(state); model.eval()
        out = np.zeros((len(X_te), C), dtype=np.float32)
        for i in range(0, len(X_te), 1024):
            _, _, _, l4 = model(torch.from_numpy(X_te[i:i+1024]).to(device))
            out[i:i+1024] = torch.sigmoid(l4).cpu().numpy()
        return out

    if name == "sota_hitec":
        model = HITECNet(1280, n1, n2, n3, C, d_model=256).to(device)
        model.load_state_dict(state); model.eval()
        out = np.zeros((len(X_te), C), dtype=np.float32)
        for i in range(0, len(X_te), 1024):
            _, _, _, a, b = model(torch.from_numpy(X_te[i:i+1024]).to(device))
            out[i:i+1024] = (a / (a + b)).cpu().numpy()
        return out

    if name == "sota_graphec_ld":
        model = GraphECStructEncoder(1280, 1024, 512, C).to(device)
        model.load_state_dict(state); model.eval()
        tr_embs = np.zeros((len(X_tr), 512), dtype=np.float32)
        for i in range(0, len(X_tr), 2048):
            z, _ = model(torch.from_numpy(X_tr[i:i+2048]).to(device), torch.from_numpy(S_tr[i:i+2048]).to(device))
            tr_embs[i:i+2048] = z.cpu().numpy()
        te_embs = np.zeros((len(X_te), 512), dtype=np.float32)
        te_probs = np.zeros((len(X_te), C), dtype=np.float32)
        for i in range(0, len(X_te), 1024):
            z, lg = model(torch.from_numpy(X_te[i:i+1024]).to(device), torch.from_numpy(S_te[i:i+1024]).to(device))
            te_embs[i:i+1024] = z.cpu().numpy()
            te_probs[i:i+1024] = torch.sigmoid(lg).cpu().numpy()
        tr_t = torch.from_numpy(tr_embs).to(device)
        te_t = torch.from_numpy(te_embs).to(device)
        ld_probs = np.zeros_like(te_probs)
        for i in range(0, len(te_embs), 512):
            vals, idxs = torch.topk(te_t[i:i+512] @ tr_t.T, k=15, dim=1)
            wts = F.softmax(vals * 10.0, dim=1).cpu().numpy()
            idxs_np = idxs.cpu().numpy()
            for r_i in range(len(wts)):
                for m_i in range(15):
                    for c in tr_labs[idxs_np[r_i, m_i]]:
                        ld_probs[i + r_i, c] += wts[r_i, m_i]
        return 0.75 * te_probs + 0.25 * ld_probs
    raise ValueError(name)


def main():
    if OUT_JSON.exists():
        raise RuntimeError(f"REFUSING TO OVERWRITE ONE-SHOT BLIND TEST RESULT: {OUT_JSON}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    locked_taus = json.loads((ROOT / "results/val_locked_thresholds_for_blind_test.json").read_text())

    m2_ec2id = json.loads((ROOT / TIERS["p1m2_benchclusterfree"][1]).read_text())["ec2id"]
    union = sorted(e for e in m2_ec2id if e.count(".") == 3)
    u_idx = {e: i for i, e in enumerate(union)}
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
    tr_cnt = dict(Counter(union[j] for ls in tr_labs for j in ls))

    test_path = ROOT / "data/processed/common_clean/test.tsv"
    test_rows = load_split(test_path)
    labs = [{e for e in r["ec_numbers"] if e in u_idx} for r in test_rows]
    keep = [i for i, l in enumerate(labs) if l]
    test_rows = [test_rows[i] for i in keep]
    labs = [labs[i] for i in keep]

    Y = np.zeros((len(test_rows), C), dtype=np.int8)
    for i, l in enumerate(labs):
        for e in l:
            Y[i, u_idx[e]] = 1

    acc2rep = load_acc2rep()
    clusters = [acc2rep.get(r["accession"], r["accession"]) for r in test_rows]

    print(f"[BLIND TEST ONE-SHOT] common_clean/test n={len(test_rows)}, clusters={len(set(clusters))}, C={C}", flush=True)

    print("[1/4] Running MMseqs2 + kNN baselines...", flush=True)
    P_mmseqs, max_pid = run_mmseqs_baseline(test_rows, train_rows_all, union, u_idx)
    hom_buckets = {"lt20": [], "20_30": [], "30_40": [], "ge40": []}
    for i, r in enumerate(test_rows):
        pid = max_pid.get(r["accession"], 0.0)
        b = "lt20" if pid < 20.0 else ("20_30" if pid < 30.0 else ("30_40" if pid < 40.0 else "ge40"))
        hom_buckets[b].append(i)
    print("  Homology buckets (test vs M2 train):", {k: len(v) for k, v in hom_buckets.items()}, flush=True)

    seq_c = EmbeddingCache("sequence")
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")
    prost_c = ProstT5AACache()

    X_tr = stack_emb(train_rows, seq_c, 1280)
    S_tr = stack_emb(train_rows, tdi_c, 1024)
    X_te = stack_emb(test_rows, seq_c, 1280)
    S_te = stack_emb(test_rows, tdi_c, 1024)

    tr_prost = stack_emb(train_rows, prost_c, 1024)
    te_prost = stack_emb(test_rows, prost_c, 1024)
    tr_dual = np.concatenate([
        X_tr / np.maximum(np.linalg.norm(X_tr, axis=1, keepdims=True), 1e-6),
        tr_prost / np.maximum(np.linalg.norm(tr_prost, axis=1, keepdims=True), 1e-6),
    ], axis=1).astype(np.float32)
    te_dual = np.concatenate([
        X_te / np.maximum(np.linalg.norm(X_te, axis=1, keepdims=True), 1e-6),
        te_prost / np.maximum(np.linalg.norm(te_prost, axis=1, keepdims=True), 1e-6),
    ], axis=1).astype(np.float32)
    P_knn_dual = run_knn_baseline(te_dual, tr_dual, tr_labs, C, str(device))
    del tr_prost, te_prost, tr_dual, te_dual

    all_preds = {
        "mmseqs2_homology": P_mmseqs,
        "knn_dualplm": P_knn_dual,
    }

    print("[2/4] Evaluating saved same-split SOTA checkpoints...", flush=True)
    for sota_name in ["sota_clean", "sota_enzymehunter", "sota_deepec_transformer", "sota_ecrecer", "sota_hitec", "sota_graphec_ld"]:
        ck = CKPT_DIR / f"{sota_name}.pt"
        if ck.exists():
            print(f"  Evaluating {sota_name} from {ck}...", flush=True)
            all_preds[sota_name] = eval_sota_ckpt(sota_name, ck, X_tr, S_tr, prim_y, tr_labs, X_te, S_te, C, hier, device)
        else:
            print(f"  [WARN] Missing checkpoint {ck}, skipping", flush=True)
    del X_tr, S_tr

    print("[3/4] Evaluating locked ZymeBind models (m2_dualplm_large & ens_dualplm_4)...", flush=True)
    caches = (seq_c, txt_c, tdi_c, prost_c)
    dom_cache = {}
    ckpts = {
        "m2_dualplm": "checkpoints/zymebind_v4_dualplm_p1m2_benchclusterfree.pt",
        "m2_dualplm_large": "checkpoints/zymebind_v4large_dualplm_p1m2_benchclusterfree.pt",
        "m2_v45_s42": "checkpoints/zymebind_v45_m2_s42.pt",
        "m2_v45_s2026": "checkpoints/zymebind_v45_m2_s2026.pt",
    }
    src = np.array([m2_ec2id[e] for e in union])
    seq, tri, txt, dom = build_inputs(test_rows, "p1m2_benchclusterfree", True, caches, dom_cache)
    z = lambda a: np.zeros_like(a)
    views = {"full": (tri, txt, dom), "seq_struct": (tri, z(txt), z(dom)), "seq_only": (z(tri), z(txt), z(dom))}

    zb_preds = {}
    for name, ck in ckpts.items():
        _, _, stage2, _, _ = load_v2_checkpoint(ROOT / ck, str(device))
        zb_preds[name] = {v: predict_v2(stage2, seq, a, b, c, str(device))[:, src] for v, (a, b, c) in views.items()}
        del stage2
        torch.cuda.empty_cache()

    P_ens = {v: sum(zb_preds[k][v] for k in ckpts) / len(ckpts) for v in views}
    P_single = zb_preds["m2_dualplm_large"]
    all_preds["zb_single_seq_only"] = P_single["seq_only"]
    all_preds["zb_single_seq_struct"] = P_single["seq_struct"]
    all_preds["zb_single_full"] = P_single["full"]
    all_preds["zb_ens4_seq_only"] = P_ens["seq_only"]
    all_preds["zb_ens4_seq_struct"] = P_ens["seq_struct"]
    all_preds["zb_ens4_full"] = P_ens["full"]

    print("[4/4] Computing Dual-Protocol Metrics (with val-locked tau*), Homology Curve, Calibration & Significance...", flush=True)
    metrics = {}
    for name, P in all_preds.items():
        tau_lock = locked_taus.get(name, 0.5)
        m = compute_dual_ec_metrics(P, Y, union, train_counts=tr_cnt, locked_threshold=tau_lock)
        metrics[name] = m
        print(
            f"  [{name:24s}] Fmax={m['fmax']:.4f} AUPR-M={m['aupr_macro']:.4f} "
            f"| LOCKED(tau={tau_lock:.2f}) wF1={m['weighted_f1']:.4f} wAUC={m['weighted_auc']:.4f} "
            f"mF1={m['micro_f1']:.4f} MF1={m['macro_f1_L4']:.4f} Rare={m['rare_fmax']}",
            flush=True,
        )

    hom_curve = {}
    for bname in ("lt20", "20_30", "30_40", "ge40"):
        idx = np.array(hom_buckets[bname], dtype=np.int64)
        hom_curve[bname] = {"n": int(len(idx)), "fmax": {}}
        if len(idx) == 0:
            continue
        for name, P in all_preds.items():
            Ps, Ys = P[idx], Y[idx]
            tc = Ys.sum(axis=1).astype(np.float32)
            bf = 0.0
            for t in np.linspace(0.02, 0.98, 49):
                pr_m = Ps >= t
                pc = pr_m.sum(axis=1)
                h = (pr_m & (Ys > 0)).sum(axis=1).astype(np.float32)
                rc = float((h / tc).mean())
                cov = pc > 0
                pr = float((h[cov] / pc[cov]).mean()) if cov.any() else 0.0
                f = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
                if f > bf:
                    bf = f
            hom_curve[bname]["fmax"][name] = round(bf, 4)

    calib = {}
    for k in ("zb_single_seq_only", "zb_single_full", "zb_ens4_seq_only", "zb_ens4_full"):
        calib[k] = calibration_metrics(all_preds[k], Y.astype(bool))

    sig = {}
    comparisons = [
        ("zb_ens4_seq_only", "mmseqs2_homology"),
        ("zb_ens4_seq_only", "knn_dualplm"),
        ("zb_ens4_seq_only", "sota_clean"),
        ("zb_ens4_seq_only", "sota_deepec_transformer"),
        ("zb_ens4_seq_only", "sota_ecrecer"),
        ("zb_ens4_seq_only", "sota_hitec"),
        ("zb_ens4_seq_struct", "sota_enzymehunter"),
        ("zb_ens4_seq_struct", "sota_graphec_ld"),
        ("zb_ens4_full", "sota_graphec_ld"),
        ("zb_ens4_full", "sota_deepec_transformer"),
        ("zb_ens4_full", "sota_clean"),
        ("zb_ens4_full", "sota_enzymehunter"),
        ("zb_ens4_full", "mmseqs2_homology"),
    ]
    for a, b in comparisons:
        if a in all_preds and b in all_preds:
            res = cluster_bootstrap_diff(
                all_preds[a], all_preds[b], Y, clusters,
                locked_taus.get(a, 0.5), locked_taus.get(b, 0.5), n_boot=400,
            )
            sig[f"{a}_minus_{b}"] = res
            print(f"  [Sig {a} - {b}] diff={res['diff']:+.4f} CI={res['ci95']} p={res['p_value']:.4f}", flush=True)

    payload = {
        "protocol": "ONE-SHOT INDEPENDENT BLIND TEST (val-locked models & val-locked thresholds)",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set": str(test_path.relative_to(ROOT)),
        "test_set_sha256": sha256_file(test_path),
        "n": int(len(test_rows)),
        "n_clusters": int(len(set(clusters))),
        "val_locked_thresholds": locked_taus,
        "models": metrics,
        "homology_decay_curve": hom_curve,
        "calibration": calib,
        "significance_bootstrap_40pct_cluster": sig,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[DONE ONE-SHOT BLIND TEST] Saved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
