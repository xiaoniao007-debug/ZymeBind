#!/usr/bin/env python
"""Enhancement 2: Large-Scale Metagenomic Dark Proteome Discovery (5,000+ zero-EC proteins).

1. Fetches 5,000+ real unreviewed environmental & marine metagenomic uncharacterized
   proteins (ZERO EC annotation, length 150-500 aa) via UniProt REST API cursor pagination.
2. Filters via MMseqs2 against all 218,602 M2 training enzymes to isolate true
   dark/twilight-zone proteins (max_pident < 30% and zero-hit pid = 0.0%).
3. Computes ESM-2 650M + ProstT5-3B <AA2fold> embeddings on GPU, runs locked
   ens_dualplm_4 (Seq-only, val-locked threshold tau=0.62) + L3 Top-K SAE.
4. Performs blind EBI InterPro corroboration on top dark discoveries (including
   counting how many zero-hit dark discoveries match catalytic InterPro superfamilies
   AND how many are ultra-dark IPR=[] de novo candidates, plus AlphaFoldDB IDs).

Output: results/case_study_1_metagenome_5000_discovery.json
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

ROOT = Path("/mnt/data/cs/ZymeBind")
sys.path.insert(0, str(ROOT))

from data.dataset import load_split
from evaluation.eval_external_v2 import predict_v2
from models.zymebind_v2 import load_v2_checkpoint
from training_zymebind import TIERS


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


def fetch_uniprot_paginated(query: str, target_n: int = 5000) -> list[dict]:
    fields = "accession,id,protein_name,organism_name,length,sequence,xref_interpro,xref_pfam,xref_alphafolddb"
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        + urllib.parse.urlencode({"query": query, "format": "tsv", "fields": fields, "size": 500})
    )
    rows = []
    seen = set()
    page = 0
    while url and len(rows) < target_n and page < 15:
        page += 1
        req = urllib.request.Request(url, headers={"User-Agent": "ZymeBind-Research/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
            link = resp.headers.get("Link", "")
        for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
            acc = r["Entry"]
            if acc in seen:
                continue
            seq = (r.get("Sequence") or "").strip().upper()
            if 150 <= len(seq) <= 500 and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in seq):
                seen.add(acc)
                rows.append({
                    "accession": acc,
                    "entry_name": r["Entry Name"],
                    "protein_name": r["Protein names"],
                    "organism": r["Organism"],
                    "length": int(r["Length"]),
                    "sequence": seq,
                    "interpro": [x for x in (r.get("InterPro") or "").split(";") if x.strip()],
                    "pfam": [x for x in (r.get("Pfam") or "").split(";") if x.strip()],
                    "afdb": (r.get("AlphaFoldDB") or "").strip().rstrip(";"),
                })
                if len(rows) >= target_n:
                    break
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        print(f"  [page {page:2d}] total collected={len(rows)}", flush=True)
    return rows


def run_mmseqs_max_pident(query_rows: list[dict], train_rows: list[dict]) -> dict[str, tuple[float, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        q_fa = tmp_p / "q.faa"
        t_fa = tmp_p / "t.faa"
        out_tsv = tmp_p / "res.tsv"
        with open(q_fa, "w") as f:
            for r in query_rows:
                f.write(f">{r['accession']}\n{r['sequence']}\n")
        with open(t_fa, "w") as f:
            for r in train_rows:
                if r.get("sequence"):
                    f.write(f">{r['accession']}\n{r['sequence']}\n")
        cmd = [
            "mmseqs", "easy-search", str(q_fa), str(t_fa), str(out_tsv), str(tmp_p / "work"),
            "--format-output", "query,target,pident,evalue",
            "-s", "5.7", "--threads", "16", "-v", "0",
        ]
        subprocess.run(cmd, check=True)
        best: dict[str, tuple[float, str]] = {}
        if out_tsv.exists():
            with open(out_tsv) as f:
                for line in f:
                    q, t, pid, _ = line.strip().split("\t")
                    pid_f = float(pid)
                    if q not in best or pid_f > best[q][0]:
                        best[q] = (pid_f, t)
    return best


@torch.no_grad()
def compute_esm2_emb(seqs: list[str], device: str, bs: int = 32) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    mod = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D", torch_dtype=torch.float16).to(device).eval()
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
        if (i // bs) % 40 == 0:
            print(f"  [ESM-2] {min(i+bs, len(seqs))}/{len(seqs)}", flush=True)
    del mod
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def compute_prostt5_aa_emb(seqs: list[str], device: str, bs: int = 24) -> np.ndarray:
    tok = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    mod = T5EncoderModel.from_pretrained("Rostlab/ProstT5", torch_dtype=torch.float16).to(device).eval()
    out = np.zeros((len(seqs), 1024), dtype=np.float32)
    for i in range(0, len(seqs), bs):
        batch = ["<AA2fold> " + " ".join(list(s)) for s in seqs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        hs = mod(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mask[:, 0, :] = 0.0
        out[i:i + bs] = ((hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).cpu().numpy()
        if (i // bs) % 50 == 0:
            print(f"  [ProstT5] {min(i+bs, len(seqs))}/{len(seqs)}", flush=True)
    del mod
    torch.cuda.empty_cache()
    return out


def lookup_interpro_names(iprs: list[str]) -> dict[str, str]:
    out = {}
    for ipr in iprs:
        try:
            url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{ipr}?format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "ZymeBind/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                d = json.loads(r.read().decode())
                out[ipr] = d.get("metadata", {}).get("name", {}).get("name", "")
        except Exception:
            pass
    return out


def main():
    t0 = time.time()
    device = "cuda:0"
    print("[1/5] Fetching 5,000 real zero-EC metagenomic uncharacterized proteins from UniProt...", flush=True)
    q = 'metagenome AND reviewed:false AND NOT ec:* AND protein_name:"Uncharacterized protein" AND length:[150 TO 500]'
    rows = fetch_uniprot_paginated(q, target_n=5000)
    print(f"  Collected {len(rows)} unique metagenomic uncharacterized proteins (ZERO EC)", flush=True)

    print("[2/5] Running MMseqs2 against 218,602 M2 training enzymes...", flush=True)
    train_rows = load_split(ROOT / TIERS["p1m2_benchclusterfree"][0] / "train.tsv")
    best_hom = run_mmseqs_max_pident(rows, train_rows)
    for r in rows:
        pid, tgt = best_hom.get(r["accession"], (0.0, ""))
        r["max_train_pident"] = round(pid, 2)
        r["nearest_train_acc"] = tgt

    n_nohit = sum(1 for r in rows if r["max_train_pident"] == 0.0)
    n_lt30 = sum(1 for r in rows if r["max_train_pident"] < 30.0)
    print(f"  MMseqs2 dark stats: no_hit={n_nohit}/{len(rows)} ({100*n_nohit/len(rows):.1f}%), pid<30%={n_lt30}/{len(rows)}", flush=True)

    print("[3/5] Computing ESM-2 650M + ProstT5-3B <AA2fold> embeddings on GPU...", flush=True)
    seqs = [r["sequence"] for r in rows]
    esm_emb = compute_esm2_emb(seqs, device)
    prost_emb = compute_prostt5_aa_emb(seqs, device)
    dual_seq = np.concatenate([esm_emb, prost_emb], axis=1).astype(np.float32)

    print("[4/5] Running locked ZymeBind ens_dualplm_4 (Seq-only, tau=0.62)...", flush=True)
    m2_ec2id = json.loads((ROOT / TIERS["p1m2_benchclusterfree"][1]).read_text())["ec2id"]
    union = sorted(e for e in m2_ec2id if e.count(".") == 3)
    src = np.array([m2_ec2id[e] for e in union])

    ckpts = [
        "checkpoints/zymebind_v4_dualplm_p1m2_benchclusterfree.pt",
        "checkpoints/zymebind_v4large_dualplm_p1m2_benchclusterfree.pt",
        "checkpoints/zymebind_v45_m2_s42.pt",
        "checkpoints/zymebind_v45_m2_s2026.pt",
    ]
    z_tri = np.zeros((len(rows), 1024), dtype=np.float32)
    z_txt = np.zeros((len(rows), 768), dtype=np.float32)
    probs_sum = np.zeros((len(rows), len(union)), dtype=np.float32)

    for ck in ckpts:
        _, _, stage2, _, ck_data = load_v2_checkpoint(ROOT / ck, device)
        z_dom = np.zeros((len(rows), int(ck_data["in_dims"]["domain"])), dtype=np.float32)
        probs_sum += predict_v2(stage2, dual_seq, z_tri, z_txt, z_dom, device)[:, src] / len(ckpts)
        del stage2
        torch.cuda.empty_cache()

    print("[5/5] Aggregating discoveries & EBI InterPro blind corroboration...", flush=True)
    locked_tau = 0.62
    ec1_all = Counter()
    ec1_dark = Counter()
    ec3_dark = Counter()
    all_pred = []
    dark_lt30 = []
    dark_nohit = []
    ultra_dark_noipr = []

    for i, r in enumerate(rows):
        p = probs_sum[i]
        top_idx = int(p.argmax())
        top_prob = float(p[top_idx])
        top_ec = union[top_idx]
        rec = {k: v for k, v in r.items() if k != "sequence"}
        rec["pred_ec4"] = top_ec
        rec["pred_ec3"] = ".".join(top_ec.split(".")[:3])
        rec["pred_ec1"] = "EC" + top_ec.split(".")[0]
        rec["pred_prob"] = round(top_prob, 4)
        if top_prob >= locked_tau:
            ec1_all[rec["pred_ec1"]] += 1
            all_pred.append(rec)
            if rec["max_train_pident"] < 30.0:
                ec1_dark[rec["pred_ec1"]] += 1
                ec3_dark[rec["pred_ec3"]] += 1
                dark_lt30.append(rec)
                if rec["max_train_pident"] == 0.0:
                    dark_nohit.append(rec)
                    if not rec["interpro"] and not rec["pfam"]:
                        ultra_dark_noipr.append(rec)

    dark_lt30.sort(key=lambda x: (-x["pred_prob"], x["max_train_pident"]))
    dark_nohit.sort(key=lambda x: -x["pred_prob"])
    ultra_dark_noipr.sort(key=lambda x: -x["pred_prob"])

    # Lookup InterPro names for top 30 zero-hit dark discoveries
    top_iprs = sorted({ipr for d in dark_nohit[:30] for ipr in d["interpro"][:3]})[:40]
    ipr_names = lookup_interpro_names(top_iprs)
    for d in dark_nohit[:30]:
        d["interpro_named"] = [f"{x} ({ipr_names.get(x, '')})" for x in d["interpro"][:3]]

    # Homology distribution of predicted enzymes
    hom_dist_pred = {
        "pid_0_nohit": sum(1 for d in all_pred if d["max_train_pident"] == 0.0),
        "pid_0_30": sum(1 for d in all_pred if 0.0 < d["max_train_pident"] < 30.0),
        "pid_30_40": sum(1 for d in all_pred if 30.0 <= d["max_train_pident"] < 40.0),
        "pid_ge40": sum(1 for d in all_pred if d["max_train_pident"] >= 40.0),
    }

    summary = {
        "n_total_scanned": len(rows),
        "n_mmseqs_no_hit": n_nohit,
        "n_mmseqs_lt30": n_lt30,
        "n_pred_enzyme_ge_tau062": len(all_pred),
        "n_dark_lt30_discoveries": len(dark_lt30),
        "n_dark_nohit_discoveries": len(dark_nohit),
        "n_dark_nohit_with_ipr": sum(1 for d in dark_nohit if d["interpro"]),
        "n_ultra_dark_nohit_noipr": len(ultra_dark_noipr),
        "n_dark_nohit_with_afdb": sum(1 for d in dark_nohit if d["afdb"]),
        "homology_distribution_of_predicted_enzymes": hom_dist_pred,
        "ec1_distribution_all_predicted": dict(ec1_all),
        "ec1_distribution_dark_lt30": dict(ec1_dark),
        "top_15_ec3_dark_lt30": ec3_dark.most_common(15),
        "top_30_nohit_discoveries": dark_nohit[:30],
        "top_20_ultra_dark_noipr": ultra_dark_noipr[:20],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_path = ROOT / "results/case_study_1_metagenome_5000_discovery.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"[DONE Enhancement 2] ({summary['elapsed_seconds']}s) scanned={len(rows)}, "
        f"pred={len(all_pred)}, dark(<30%)={len(dark_lt30)}, nohit={len(dark_nohit)}, "
        f"ultra_dark(noIPR)={len(ultra_dark_noipr)}, with_AFDB={summary['n_dark_nohit_with_afdb']}",
        flush=True,
    )
    print(f"  EC1 dark(<30%): {dict(ec1_dark)}", flush=True)
    for d in dark_nohit[:10]:
        print(f"  {d['accession']} | pred={d['pred_ec4']} ({d['pred_prob']}) | org={d['organism'][:25]} | AFDB={bool(d['afdb'])} | IPR={d.get('interpro_named',[])[:2]}", flush=True)


if __name__ == "__main__":
    main()
