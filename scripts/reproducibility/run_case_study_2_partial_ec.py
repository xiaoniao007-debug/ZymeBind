#!/usr/bin/env python
"""Case Study 2: Incomplete EC (x.x.x.-) Precision Completion & Text-Shortcut Audit.

Part A — Held-Out Incomplete EC Completion (n=3,245 in p1m2/test with ONLY partial EC):
  1. Runs locked ens_dualplm_4 (Seq+3Di and All-4) on all 3,245 held-out partial-EC enzymes.
  2. Measures hierarchical consistency: when ZymeBind predicts a complete 4-digit EC4 with
     high confidence (prob >= 0.70 / 0.85), what % strictly preserves the known EC1.EC2.EC3 prefix?
  3. Queries live UniProt REST API for top-60 high-confidence completions to check whether
     UniProt's latest release has since curated a complete 4-digit EC4 (prospective validation!).

Part B — Text-Shortcut / Misannotation Audit on common_clean/test:
  Identifies concrete real enzymes where generic superfamily text in UniProt misleads All-4
  toward a sibling EC4 while ZymeBind's biophysical branch (Seq+3Di) predicts the exact true EC4.

Output: results/case_study_2_partial_ec_and_audit.json
"""
from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/data/cs/ZymeBind")
sys.path.insert(0, str(ROOT))

from data.cached_dataset import EmbeddingCache
from data.dataset import load_split
from evaluation.eval_comprehensive_val import build_inputs
from evaluation.eval_external_v2 import predict_v2
from models.zymebind_v2 import load_v2_checkpoint
from training_zymebind import TIERS
from training_zymebind_v4_dualplm import ProstT5AACache


def fetch_uniprot_live_ec(accs: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(accs), 25):
        chunk = accs[i:i + 25]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = (
            "https://rest.uniprot.org/uniprotkb/search?"
            + urllib.parse.urlencode({
                "query": q, "format": "tsv",
                "fields": "accession,protein_name,ec,cc_catalytic_activity,xref_interpro",
                "size": 50,
            })
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ZymeBind-Research/1.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                text = resp.read().decode("utf-8")
            for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
                ecs = [x.strip() for x in (r.get("EC number") or "").split(";") if x.strip()]
                out[r["Entry"]] = {
                    "protein_name": r.get("Protein names", ""),
                    "live_ec": ecs,
                    "catalytic_activity": (r.get("Catalytic activity") or "")[:300],
                    "interpro": [x.strip() for x in (r.get("InterPro") or "").split(";") if x.strip()],
                }
        except Exception as e:
            print(f"  [WARN UniProt batch {i}] {e}", flush=True)
    return out


def main():
    t0 = time.time()
    device = "cuda:0"
    m2_ec2id = json.loads((ROOT / TIERS["p1m2_benchclusterfree"][1]).read_text())["ec2id"]
    union = sorted(e for e in m2_ec2id if e.count(".") == 3)
    src = np.array([m2_ec2id[e] for e in union])

    print("[1/3] Loading 3,245 held-out partial-EC enzymes from p1m2/test...", flush=True)
    test_all = load_split(ROOT / TIERS["p1m2_benchclusterfree"][0] / "test.tsv")
    part_rows = [r for r in test_all if r.get("ec_partial") and not r.get("ec_numbers")]
    print(f"  Found {len(part_rows)} partial-only enzymes in held-out p1m2/test", flush=True)

    seq_c = EmbeddingCache("sequence")
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")
    prost_c = ProstT5AACache()
    caches = (seq_c, txt_c, tdi_c, prost_c)
    dom_cache = {}

    seq, tri, txt, dom = build_inputs(part_rows, "p1m2_benchclusterfree", True, caches, dom_cache)
    z = lambda a: np.zeros_like(a)

    ckpts = {
        "m2_dualplm": "checkpoints/zymebind_v4_dualplm_p1m2_benchclusterfree.pt",
        "m2_dualplm_large": "checkpoints/zymebind_v4large_dualplm_p1m2_benchclusterfree.pt",
        "m2_v45_s42": "checkpoints/zymebind_v45_m2_s42.pt",
        "m2_v45_s2026": "checkpoints/zymebind_v45_m2_s2026.pt",
    }

    p_ss = np.zeros((len(part_rows), len(union)), dtype=np.float32)
    p_full = np.zeros((len(part_rows), len(union)), dtype=np.float32)
    for _, ck in ckpts.items():
        _, _, stage2, _, _ = load_v2_checkpoint(ROOT / ck, device)
        p_ss += predict_v2(stage2, seq, tri, z(txt), z(dom), device)[:, src] / len(ckpts)
        p_full += predict_v2(stage2, seq, tri, txt, dom, device)[:, src] / len(ckpts)
        del stage2
        torch.cuda.empty_cache()

    print("[2/3] Evaluating hierarchical EC3 consistency & prospective UniProt live lookup...", flush=True)
    consistency = {}
    candidates = []
    for mode_name, P in [("seq_struct", p_ss), ("all_4", p_full)]:
        stats = {}
        for thr in (0.50, 0.70, 0.85):
            n_ec3_known = n_match_ec3 = n_ec2_match = 0
            for i, r in enumerate(part_rows):
                ec3_known = {".".join(p.split(".")[:3]) for p in r["ec_partial"] if p.count(".") == 3 and p.endswith(".-") and not p.endswith(".-.-")}
                if not ec3_known:
                    continue
                j = int(P[i].argmax())
                prob = float(P[i, j])
                if prob < thr:
                    continue
                n_ec3_known += 1
                pred_ec = union[j]
                if ".".join(pred_ec.split(".")[:3]) in ec3_known:
                    n_match_ec3 += 1
                if any(".".join(pred_ec.split(".")[:2]) == ".".join(k.split(".")[:2]) for k in ec3_known):
                    n_ec2_match += 1
                if mode_name == "seq_struct" and thr == 0.85 and ".".join(pred_ec.split(".")[:3]) in ec3_known:
                    candidates.append({
                        "accession": r["accession"],
                        "organism": r.get("organism", ""),
                        "ec_partial": r["ec_partial"],
                        "pred_ec4_ss": pred_ec,
                        "prob_ss": round(prob, 4),
                        "pred_ec4_full": union[int(p_full[i].argmax())],
                        "prob_full": round(float(p_full[i].max()), 4),
                        "function_text": (r.get("function_text") or "")[:240],
                    })
            stats[f"prob_ge_{thr:.2f}"] = {
                "n_completed": n_ec3_known,
                "ec3_exact_consistency": round(n_match_ec3 / max(n_ec3_known, 1), 4),
                "ec2_consistency": round(n_ec2_match / max(n_ec3_known, 1), 4),
            }
        consistency[mode_name] = stats
        print(f"  [{mode_name}] {stats}", flush=True)

    candidates.sort(key=lambda x: -x["prob_ss"])
    top_check = candidates[:60]
    live_info = fetch_uniprot_live_ec([c["accession"] for c in top_check])
    n_upgraded = 0
    n_exact_hit = 0
    prospective_hits = []
    for c in top_check:
        info = live_info.get(c["accession"], {})
        live_full_ec4 = [e for e in info.get("live_ec", []) if "-" not in e and e.count(".") == 3]
        c["live_uniprot_ec"] = info.get("live_ec", [])
        c["live_protein_name"] = info.get("protein_name", "")
        if live_full_ec4:
            n_upgraded += 1
            hit = c["pred_ec4_ss"] in live_full_ec4
            if hit:
                n_exact_hit += 1
            c["prospective_exact_match"] = hit
            prospective_hits.append(c)

    print(f"  Prospective UniProt check (top {len(top_check)}): upgraded_to_full_EC4={n_upgraded}, exact_match={n_exact_hit}", flush=True)

    print("[3/3] Auditing Text-Shortcut Disagreements on common_clean/test...", flush=True)
    cc_rows = load_split(ROOT / "data/processed/common_clean/test.tsv")
    u_idx = {e: i for i, e in enumerate(union)}
    cc_rows = [r for r in cc_rows if any(e in u_idx for e in r["ec_numbers"])]
    seq_c2, tri_c2, txt_c2, dom_c2 = build_inputs(cc_rows, "p1m2_benchclusterfree", True, caches, dom_cache)

    pc_ss = np.zeros((len(cc_rows), len(union)), dtype=np.float32)
    pc_full = np.zeros((len(cc_rows), len(union)), dtype=np.float32)
    for _, ck in ckpts.items():
        _, _, stage2, _, _ = load_v2_checkpoint(ROOT / ck, device)
        pc_ss += predict_v2(stage2, seq_c2, tri_c2, z(txt_c2), z(dom_c2), device)[:, src] / len(ckpts)
        pc_full += predict_v2(stage2, seq_c2, tri_c2, txt_c2, dom_c2, device)[:, src] / len(ckpts)
        del stage2
        torch.cuda.empty_cache()

    biophysics_rescues = []
    for i, r in enumerate(cc_rows):
        true_set = {e for e in r["ec_numbers"] if e in u_idx}
        j_ss, j_fl = int(pc_ss[i].argmax()), int(pc_full[i].argmax())
        ec_ss, ec_fl = union[j_ss], union[j_fl]
        if ec_ss in true_set and ec_fl not in true_set and pc_ss[i, j_ss] >= 0.70:
            biophysics_rescues.append({
                "accession": r["accession"],
                "organism": r.get("organism", ""),
                "true_ec": sorted(true_set),
                "pred_seq_struct": ec_ss,
                "prob_seq_struct": round(float(pc_ss[i, j_ss]), 4),
                "pred_all4": ec_fl,
                "prob_all4": round(float(pc_full[i, j_fl]), 4),
                "function_text": (r.get("function_text") or "")[:240],
            })
    biophysics_rescues.sort(key=lambda x: -x["prob_seq_struct"])
    print(f"  Found {len(biophysics_rescues)} cases on common_clean/test where pure Seq+3Di rescues text-misled All-4!", flush=True)

    out = {
        "n_heldout_partial_ec_enzymes": len(part_rows),
        "hierarchical_consistency": consistency,
        "prospective_uniprot_check": {
            "n_checked": len(top_check),
            "n_upgraded_in_live_uniprot": n_upgraded,
            "n_exact_ec4_match": n_exact_hit,
            "prospective_hits": prospective_hits[:15],
        },
        "top_20_partial_completions": top_check[:20],
        "text_shortcut_biophysics_rescues_count": len(biophysics_rescues),
        "top_15_biophysics_rescues": biophysics_rescues[:15],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = ROOT / "results/case_study_2_partial_ec_and_audit.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[DONE Case Study 2] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
