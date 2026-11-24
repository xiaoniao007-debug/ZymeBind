#!/usr/bin/env python3
"""Workstream 2: In Silico Deep Mutational Scanning (DMS) vs Experimental ProteinGym Fitness.

Evaluates locked ZymeBind (ens_dualplm_4, Seq-only) on 4 classic ProteinGym v1.1 enzyme DMS
benchmarks spanning EC1 (Oxidoreductase), EC2 (Transferase), and EC3 (Hydrolase):
  1. BLAT_ECOLX_Stiffler_2015          (TEM-1 beta-lactamase, EC 3.5.2.6)
  2. MTH3_HAEAE_RockahShmuel_2015      (DNA methyltransferase M.HaeIII, EC 2.1.1.37)
  3. KKA2_KLEPN_Melnikov_2014          (Aminoglycoside 3'-phosphotransferase, EC 2.7.1.95)
  4. OXDA_RHOTO_Vanella_2023_activity  (D-amino acid oxidase, EC 1.4.3.3)

Computes:
  - Spearman rho (mutant-level & per-residue mean) between ZymeBind delta_logit(true EC4)
    and experimental DMS catalytic activity/fitness.
  - AUROC for classifying experimentally deleterious (DMS_score_bin==0) vs functional mutants.
  - Catalytic active-site enrichment (Mann-Whitney U test + fold enrichment of |delta_logit|).
  - Full per-residue sensitivity profile for TEM-1 (for 3D mapping onto 1BTL in PyMOL).

Output: results/ext2_dms_catalytic_landscape.json
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

ROOT = Path("/mnt/data/cs/ZymeBind")
sys.path.insert(0, str(ROOT))

from evaluation.eval_external_v2 import predict_v2
from models.zymebind_v2 import load_v2_checkpoint
from training_zymebind import TIERS

DMS_DIR = Path("/tmp/proteingym_dms")
OUT_PATH = ROOT / "results/ext2_dms_catalytic_landscape.json"

BENCHMARKS = [
    {
        "id": "BLAT_ECOLX_Stiffler_2015",
        "csv": "BLAT_ECOLX_Stiffler_2015.csv",
        "name": "TEM-1 beta-lactamase",
        "organism": "Escherichia coli",
        "ec4": "3.5.2.6",
        "pdb_id": "1BTL",
        "max_mutants": 1600,
        # Active site motifs in TEM-1: SxxK (S68, K71), SDN (S128, D129, N130), E164 (EPELN), KTG (K232, T233, G234)
        "active_motifs": [("STFK", [0, 3]), ("SDN", [0, 1, 2]), ("EPELN", [0]), ("KTG", [0, 1, 2])],
    },
    {
        "id": "MTH3_HAEAE_RockahShmuel_2015",
        "csv": "MTH3_HAEAE_RockahShmuel_2015.csv",
        "name": "DNA methyltransferase M.HaeIII",
        "organism": "Haemophilus aegyptius",
        "ec4": "2.1.1.37",
        "pdb_id": "1DCT",
        "max_mutants": 1800,
        # Catalytic PCQ (Cys nucleophile), ENV (Glu proton donor), FAG (AdoMet binding)
        "active_motifs": [("PCQ", [0, 1, 2]), ("ENV", [0, 1]), ("FAG", [0, 1, 2])],
    },
    {
        "id": "KKA2_KLEPN_Melnikov_2014",
        "csv": "KKA2_KLEPN_Melnikov_2014.csv",
        "name": "Aminoglycoside kinase APH(3')-II",
        "organism": "Klebsiella pneumoniae",
        "ec4": "2.7.1.95",
        "pdb_id": "1ND4",
        "max_mutants": 1300,
        # Catalytic HGD (Asp190 base), DFG/D208 (Mg2+ binding), K50 (ATP binding)
        "active_motifs": [("HGD", [0, 1, 2]), ("LID", [2]), ("VAVK", [3])],
    },
    {
        "id": "OXDA_RHOTO_Vanella_2023_activity",
        "csv": "OXDA_RHOTO_Vanella_2023_activity.csv",
        "name": "D-amino acid oxidase (DAOx)",
        "organism": "Rhodotorula toruloides",
        "ec4": "1.4.3.3",
        "pdb_id": "1C0P",
        "max_mutants": 1300,
        # Active site FAD-binding GxGxxG + catalytic Y223, Y238, R285
        "active_motifs": [("GAGVIG", [0, 2, 5]), ("YGG", [0]), ("RY", [0, 1])],
    },
]


def reconstruct_wt(rows: list[dict]) -> str:
    """Reconstruct wild-type sequence from single-point mutant records."""
    r0 = rows[0]
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", r0["mutant"].strip())
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    seq = list(r0["mutated_sequence"].strip())
    assert seq[pos - 1] == mut_aa, f"Mismatch at {pos}: {seq[pos-1]} != {mut_aa}"
    seq[pos - 1] = wt_aa
    return "".join(seq)


def find_active_positions(wt_seq: str, motifs: list[tuple[str, list[int]]]) -> list[int]:
    pos_set = set()
    for motif, offsets in motifs:
        idx = wt_seq.find(motif)
        if idx >= 0:
            for off in offsets:
                pos_set.add(idx + off + 1)  # 1-based
    return sorted(pos_set)


def stratified_sample_mutants(rows: list[dict], max_n: int) -> list[dict]:
    """Keep all single mutants if <= max_n; otherwise stratify evenly across ALL residue positions."""
    singles = []
    for r in rows:
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", r["mutant"].strip())
        if m:
            singles.append({
                "mutant": r["mutant"].strip(),
                "pos": int(m.group(2)),
                "wt_aa": m.group(1),
                "mut_aa": m.group(3),
                "seq": r["mutated_sequence"].strip(),
                "dms_score": float(r["DMS_score"]),
                "dms_bin": int(r["DMS_score_bin"]),
            })
    if len(singles) <= max_n:
        return singles

    by_pos = defaultdict(list)
    for s in singles:
        by_pos[s["pos"]].append(s)

    rng = np.random.default_rng(42)
    per_pos = max(1, max_n // len(by_pos))
    chosen = []
    for pos in sorted(by_pos.keys()):
        lst = sorted(by_pos[pos], key=lambda x: x["dms_score"])
        # Always include Alanine if present, plus evenly spaced quantiles across DMS_score
        if len(lst) <= per_pos:
            chosen.extend(lst)
        else:
            idxs = np.linspace(0, len(lst) - 1, per_pos, dtype=int)
            picked = {int(i) for i in idxs}
            for i, item in enumerate(lst):
                if item["mut_aa"] == "A":
                    picked.add(i)
                    break
            chosen.extend(lst[i] for i in sorted(picked))
    return chosen[:max_n]


@torch.no_grad()
def compute_esm2_emb(seqs: list[str], tok, mod, device: str, bs: int = 32) -> np.ndarray:
    out = np.zeros((len(seqs), 1280), dtype=np.float32)
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        hs = mod(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mask[:, 0, :] = 0.0
        lengths = enc["attention_mask"].sum(dim=1)
        for b_i, L in enumerate(lengths):
            if L > 1:
                mask[b_i, L - 1, :] = 0.0
        out[i:i + bs] = ((hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).cpu().numpy()
    return out


@torch.no_grad()
def compute_prostt5_aa_emb(seqs: list[str], tok, mod, device: str, bs: int = 24) -> np.ndarray:
    out = np.zeros((len(seqs), 1024), dtype=np.float32)
    for i in range(0, len(seqs), bs):
        batch = ["<AA2fold> " + " ".join(list(s)) for s in seqs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        hs = mod(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mask[:, 0, :] = 0.0
        out[i:i + bs] = ((hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).cpu().numpy()
    return out


def safe_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def main():
    t0 = time.time()
    device = "cuda:0"

    # Load EC4 vocabulary
    m2_ec2id = json.loads((ROOT / TIERS["p1m2_benchclusterfree"][1]).read_text())["ec2id"]
    union = sorted(e for e in m2_ec2id if e.count(".") == 3)
    ec_to_u = {e: i for i, e in enumerate(union)}
    src = np.array([m2_ec2id[e] for e in union])

    # Load datasets & prepare all sequences
    datasets = []
    all_seqs = []
    for cfg in BENCHMARKS:
        with open(DMS_DIR / cfg["csv"]) as f:
            raw_rows = list(csv.DictReader(f))
        wt_seq = reconstruct_wt(raw_rows)
        muts = stratified_sample_mutants(raw_rows, cfg["max_mutants"])
        act_pos = find_active_positions(wt_seq, cfg["active_motifs"])
        start_idx = len(all_seqs)
        all_seqs.append(wt_seq)
        for m in muts:
            all_seqs.append(m["seq"])
        end_idx = len(all_seqs)
        datasets.append({
            "cfg": cfg,
            "wt_seq": wt_seq,
            "muts": muts,
            "act_pos": act_pos,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })
        print(
            f"[Ext2] Loaded {cfg['id']}: len={len(wt_seq)}, n_muts={len(muts)}, "
            f"ec4={cfg['ec4']} (in_vocab={cfg['ec4'] in ec_to_u}), active_pos={act_pos}",
            flush=True,
        )

    print(f"[Ext2] Computing ESM-2 650M embeddings for {len(all_seqs)} sequences on {device}...", flush=True)
    esm_tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_mod = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float16).to(device).eval()
    esm_emb = compute_esm2_emb(all_seqs, esm_tok, esm_mod, device, bs=48)
    del esm_mod
    torch.cuda.empty_cache()

    print(f"[Ext2] Computing ProstT5-3B AA embeddings for {len(all_seqs)} sequences on {device}...", flush=True)
    pt5_tok = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    pt5_mod = T5EncoderModel.from_pretrained("Rostlab/ProstT5", torch_dtype=torch.float16).to(device).eval()
    pt5_emb = compute_prostt5_aa_emb(all_seqs, pt5_tok, pt5_mod, device, bs=32)
    del pt5_mod
    torch.cuda.empty_cache()

    dual_seq = np.concatenate([esm_emb, pt5_emb], axis=1).astype(np.float32)

    print("[Ext2] Running locked ZymeBind ens_dualplm_4 (Seq-only)...", flush=True)
    ckpts = [
        "checkpoints/zymebind_v4_dualplm_p1m2_benchclusterfree.pt",
        "checkpoints/zymebind_v4large_dualplm_p1m2_benchclusterfree.pt",
        "checkpoints/zymebind_v45_m2_s42.pt",
        "checkpoints/zymebind_v45_m2_s2026.pt",
    ]
    z_tri = np.zeros((len(all_seqs), 1024), dtype=np.float32)
    z_txt = np.zeros((len(all_seqs), 768), dtype=np.float32)
    probs_sum = np.zeros((len(all_seqs), len(union)), dtype=np.float32)

    for ck in ckpts:
        _, _, stage2, _, ck_data = load_v2_checkpoint(ROOT / ck, device)
        z_dom = np.zeros((len(all_seqs), int(ck_data["in_dims"]["domain"])), dtype=np.float32)
        probs_sum += predict_v2(stage2, dual_seq, z_tri, z_txt, z_dom, device)[:, src] / len(ckpts)
        del stage2
        torch.cuda.empty_cache()

    # Evaluate each benchmark
    out_benchmarks = []
    for ds in datasets:
        cfg = ds["cfg"]
        ec_idx = ec_to_u[cfg["ec4"]]
        p_slice = probs_sum[ds["start_idx"]:ds["end_idx"], ec_idx]
        wt_prob = float(p_slice[0])
        wt_logit = float(safe_logit(np.array([wt_prob]))[0])
        mut_probs = p_slice[1:]
        mut_logits = safe_logit(mut_probs)
        delta_prob = mut_probs - wt_prob
        delta_logit = mut_logits - wt_logit

        dms_scores = np.array([m["dms_score"] for m in ds["muts"]], dtype=np.float64)
        dms_bins = np.array([m["dms_bin"] for m in ds["muts"]], dtype=np.int32)
        positions = np.array([m["pos"] for m in ds["muts"]], dtype=np.int32)

        rho_all, p_rho_all = spearmanr(delta_logit, dms_scores)
        auroc_lof = float(roc_auc_score(dms_bins, delta_logit))

        # Per-residue mean sensitivity vs per-residue mean DMS fitness
        pos_groups = defaultdict(list)
        for i, pos in enumerate(positions):
            pos_groups[int(pos)].append((float(delta_logit[i]), float(delta_prob[i]), float(dms_scores[i])))

        per_res = []
        for pos in sorted(pos_groups.keys()):
            vals = pos_groups[pos]
            per_res.append({
                "pos": pos,
                "wt_aa": ds["wt_seq"][pos - 1],
                "is_active": pos in ds["act_pos"],
                "mean_delta_logit": round(float(np.mean([v[0] for v in vals])), 4),
                "mean_abs_delta_logit": round(float(np.mean([abs(v[0]) for v in vals])), 4),
                "mean_delta_prob": round(float(np.mean([v[1] for v in vals])), 5),
                "mean_dms_score": round(float(np.mean([v[2] for v in vals])), 4),
                "n_muts": len(vals),
            })
        rho_res, p_rho_res = spearmanr(
            [r["mean_delta_logit"] for r in per_res],
            [r["mean_dms_score"] for r in per_res],
        )

        # Active-site vs non-active-site sensitivity enrichment
        act_mask = np.isin(positions, ds["act_pos"])
        act_sens = np.abs(delta_logit[act_mask])
        bg_sens = np.abs(delta_logit[~act_mask])
        if len(act_sens) > 0 and len(bg_sens) > 0:
            _, p_act = mannwhitneyu(act_sens, bg_sens, alternative="greater")
            act_fold = float(np.mean(act_sens) / max(1e-8, np.mean(bg_sens)))
        else:
            p_act, act_fold = 1.0, 1.0

        # Sample 300 points for scatter plotting in R
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(dms_scores), size=min(400, len(dms_scores)), replace=False)
        scatter_pts = [
            {
                "mutant": ds["muts"][i]["mutant"],
                "pos": int(positions[i]),
                "is_active": bool(act_mask[i]),
                "delta_logit": round(float(delta_logit[i]), 4),
                "dms_score": round(float(dms_scores[i]), 4),
                "dms_bin": int(dms_bins[i]),
            }
            for i in sample_idx
        ]

        rec = {
            "id": cfg["id"],
            "name": cfg["name"],
            "organism": cfg["organism"],
            "ec4": cfg["ec4"],
            "pdb_id": cfg["pdb_id"],
            "length": len(ds["wt_seq"]),
            "n_mutants_eval": len(ds["muts"]),
            "wt_prob": round(wt_prob, 5),
            "wt_logit": round(wt_logit, 4),
            "spearman_rho_all": round(float(rho_all), 4),
            "spearman_p_all": float(p_rho_all),
            "spearman_rho_per_residue": round(float(rho_res), 4),
            "spearman_p_per_residue": float(p_rho_res),
            "auroc_lof": round(auroc_lof, 4),
            "active_positions": ds["act_pos"],
            "active_mean_abs_dlogit": round(float(np.mean(act_sens)) if len(act_sens) else 0.0, 4),
            "bg_mean_abs_dlogit": round(float(np.mean(bg_sens)) if len(bg_sens) else 0.0, 4),
            "active_fold_enrichment": round(act_fold, 2),
            "active_mwu_p": float(p_act),
            "per_residue": per_res,
            "scatter_sample": scatter_pts,
        }
        out_benchmarks.append(rec)
        print(
            f"  [RESULT] {cfg['name']} ({cfg['ec4']}): WT_prob={wt_prob:.4f} | "
            f"Spearman(all)={rho_all:.3f} (p={p_rho_all:.2e}) | "
            f"Spearman(pos)={rho_res:.3f} | AUROC={auroc_lof:.3f} | "
            f"ActiveSite fold={act_fold:.2f}x (p={p_act:.2e})",
            flush=True,
        )

    out = {
        "n_benchmarks": len(out_benchmarks),
        "total_mutants_evaluated": sum(b["n_mutants_eval"] for b in out_benchmarks),
        "mean_spearman_all": round(float(np.mean([b["spearman_rho_all"] for b in out_benchmarks])), 4),
        "mean_spearman_per_residue": round(float(np.mean([b["spearman_rho_per_residue"] for b in out_benchmarks])), 4),
        "mean_auroc_lof": round(float(np.mean([b["auroc_lof"] for b in out_benchmarks])), 4),
        "elapsed_seconds": round(time.time() - t0, 1),
        "benchmarks": out_benchmarks,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[DONE Ext2] Saved {OUT_PATH} in {out['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
