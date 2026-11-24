#!/usr/bin/env python3
"""Workstream 3: Extremophile Whole-Proteome Metabolic Pathway Hole Filling.

Scans the complete reference proteome of the deep-sea hydrothermal vent hyperthermophilic
archaeon Methanocaldococcus jannaschii DSM 2661 (UP000000805, ~1,787 proteins):
  1. Fetches complete proteome from UniProt REST API (including current EC, InterPro, AFDB, KEGG IDs).
  2. Fetches official KEGG organism pathways (mja), currently annotated KEGG ECs (mja),
     and reference KEGG pathway-to-EC catalog via KEGG REST API.
  3. Runs MMseqs2 against all 218,602 M2 training enzymes + locked ZymeBind (ens_dualplm_4, Seq-only).
  4. Quantifies:
       - Sanity-check accuracy on already-annotated M. jannaschii enzymes.
       - Unannotated / hypothetical proteins (zero EC in UniProt & zero EC in KEGG) where
         ZymeBind fills a MISSING enzymatic step ("Pathway Hole") in an active M. jannaschii
         KEGG metabolic pathway!

Output: results/ext3_extremophile_pathway_holes.json
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

ROOT = Path("/mnt/data/cs/ZymeBind")
sys.path.insert(0, str(ROOT))

from data.dataset import load_split
from evaluation.eval_external_v2 import predict_v2
from models.zymebind_v2 import load_v2_checkpoint
from training_zymebind import TIERS

OUT_PATH = ROOT / "results/ext3_extremophile_pathway_holes.json"
UA = {"User-Agent": "ZymeBind-Research/1.0"}


def http_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def fetch_mja_proteome() -> list[dict]:
    fields = "accession,id,protein_name,ec,length,sequence,xref_interpro,xref_pfam,xref_alphafolddb,xref_kegg"
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        + urllib.parse.urlencode({"query": "proteome:UP000000805", "format": "tsv", "fields": fields, "size": 500})
    )
    rows = []
    seen = set()
    while url:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
            link = resp.headers.get("Link", "")
        for r in csv.DictReader(io.StringIO(text), delimiter="\t"):
            acc = r["Entry"]
            if acc in seen:
                continue
            seq = (r.get("Sequence") or "").strip().upper()
            if len(seq) < 30 or not all(c in "ACDEFGHIKLMNPQRSTVWYBXZJUO" for c in seq):
                continue
            seq_clean = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq)
            ecs = [x.strip() for x in (r.get("EC number") or "").split(";") if x.strip()]
            keggs = [x.strip() for x in (r.get("KEGG") or "").split(";") if x.strip()]
            seen.add(acc)
            rows.append({
                "accession": acc,
                "entry_name": r["Entry Name"],
                "protein_name": r["Protein names"],
                "length": int(r["Length"]),
                "sequence": seq_clean,
                "uniprot_ecs": ecs,
                "kegg_ids": keggs,
                "interpro": [x.strip() for x in (r.get("InterPro") or "").split(";") if x.strip()],
                "pfam": [x.strip() for x in (r.get("Pfam") or "").split(";") if x.strip()],
                "afdb": (r.get("AlphaFoldDB") or "").strip().rstrip(";"),
            })
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return rows


def fetch_kegg_mja_metabolic_network():
    kdir = Path("/tmp/kegg_mja")
    # 1. Organism gene -> pathway
    gene_pw_txt = (kdir / "link_pathway_mja.txt").read_text() if (kdir / "link_pathway_mja.txt").exists() else http_text("https://rest.kegg.jp/link/pathway/mja")
    mja_pathways = set()
    gene_to_pws = defaultdict(set)
    for line in gene_pw_txt.splitlines():
        if "\t" not in line:
            continue
        g, pw = line.strip().split("\t")
        pw_num = pw.replace("path:mja", "")
        if pw_num in ("01100", "01110", "01120", "01200", "01210", "01212", "01230", "01232", "01250"):
            continue  # Skip global overview maps; keep specific metabolic pathways!
        mja_pathways.add(pw_num)
        gene_to_pws[g].add(pw_num)

    # 2. Organism gene -> EC
    gene_ec_txt = (kdir / "link_ec_mja.txt").read_text() if (kdir / "link_ec_mja.txt").exists() else http_text("https://rest.kegg.jp/link/ec/mja")
    mja_kegg_ecs = set()
    gene_to_ecs = defaultdict(set)
    for line in gene_ec_txt.splitlines():
        if "\t" not in line:
            continue
        g, ec = line.strip().split("\t")
        ec_num = ec.replace("ec:", "")
        mja_kegg_ecs.add(ec_num)
        gene_to_ecs[g].add(ec_num)

    # 3. Reference pathway -> EC
    pw_ec_txt = (kdir / "link_ec_pathway.txt").read_text() if (kdir / "link_ec_pathway.txt").exists() else http_text("https://rest.kegg.jp/link/ec/pathway")
    ref_pw_to_ecs = defaultdict(set)
    for line in pw_ec_txt.splitlines():
        if "\t" not in line:
            continue
        pw, ec = line.strip().split("\t")
        if not pw.startswith("path:map"):
            continue
        pw_num = pw.replace("path:map", "")
        ec_num = ec.replace("ec:", "")
        if pw_num in mja_pathways:
            ref_pw_to_ecs[pw_num].add(ec_num)

    # 4. Pathway names
    pw_list_txt = (kdir / "list_pathway.txt").read_text() if (kdir / "list_pathway.txt").exists() else http_text("https://rest.kegg.jp/list/pathway")
    pw_names = {}
    for line in pw_list_txt.splitlines():
        if "\t" not in line:
            continue
        pw, name = line.strip().split("\t", 1)
        pw_names[pw.replace("map", "")] = name

    # Compute currently present vs missing (holes) ECs per active M. jannaschii pathway
    ec_to_missing_pws = defaultdict(list)
    for pw_num in sorted(mja_pathways):
        ref_ecs = ref_pw_to_ecs.get(pw_num, set())
        if len(ref_ecs) < 5:
            continue
        present = ref_ecs & mja_kegg_ecs
        missing = ref_ecs - mja_kegg_ecs
        if len(present) >= 2:  # Active metabolic pathway in M. jannaschii
            for ec in missing:
                ec_to_missing_pws[ec].append({
                    "pathway_id": f"map{pw_num}",
                    "pathway_name": pw_names.get(pw_num, f"map{pw_num}"),
                    "n_present_kegg": len(present),
                    "n_ref_total": len(ref_ecs),
                })

    return mja_pathways, mja_kegg_ecs, gene_to_ecs, ref_pw_to_ecs, pw_names, ec_to_missing_pws


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
        best = {}
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
    del mod
    torch.cuda.empty_cache()
    return out


def main():
    t0 = time.time()
    device = "cuda:1"

    print("[Ext3 1/5] Fetching complete Methanocaldococcus jannaschii DSM 2661 proteome...", flush=True)
    rows = fetch_mja_proteome()
    print(f"  Total M. jannaschii proteins: {len(rows)}", flush=True)

    print("[Ext3 2/5] Fetching KEGG mja metabolic network & pathway holes...", flush=True)
    mja_pws, mja_kegg_ecs, gene_to_ecs, ref_pw_to_ecs, pw_names, ec_to_missing_pws = fetch_kegg_mja_metabolic_network()
    print(f"  Active KEGG pathways: {len(mja_pws)}, Annotated ECs: {len(mja_kegg_ecs)}, Distinct missing pathway-hole ECs: {len(ec_to_missing_pws)}", flush=True)

    print("[Ext3 3/5] Running MMseqs2 against 218,602 M2 training enzymes...", flush=True)
    train_rows = load_split(ROOT / TIERS["p1m2_benchclusterfree"][0] / "train.tsv")
    best_hom = run_mmseqs_max_pident(rows, train_rows)
    for r in rows:
        pid, tgt = best_hom.get(r["accession"], (0.0, ""))
        r["max_train_pident"] = round(pid, 2)
        r["nearest_train_acc"] = tgt
        r["kegg_ecs"] = sorted({ec for kid in r["kegg_ids"] for ec in gene_to_ecs.get(kid, set())})
        r["is_unannotated_no_ec"] = (len(r["uniprot_ecs"]) == 0 and len(r["kegg_ecs"]) == 0)

    print(f"[Ext3 4/5] Computing Dual-PLM embeddings & running locked ZymeBind (Seq-only) on {device}...", flush=True)
    seqs = [r["sequence"] for r in rows]
    esm_emb = compute_esm2_emb(seqs, device)
    pt5_emb = compute_prostt5_aa_emb(seqs, device)
    dual_seq = np.concatenate([esm_emb, pt5_emb], axis=1).astype(np.float32)

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

    print("[Ext3 5/5] Analyzing proteome annotation & KEGG pathway hole filling...", flush=True)
    locked_tau = 0.62
    n_known_ec4 = 0
    known_ec4_match = 0
    known_ec3_match = 0
    unannotated_total = sum(1 for r in rows if r["is_unannotated_no_ec"])
    unannotated_pred = []
    pathway_hole_fills = []
    pathway_hole_fills_dark = []
    improved_pathways = defaultdict(list)

    for i, r in enumerate(rows):
        p = probs_sum[i]
        top_idx = int(p.argmax())
        top_prob = float(p[top_idx])
        top_ec = union[top_idx]

        # Sanity check on known complete EC4 enzymes
        known_ec4s = [e for e in r["uniprot_ecs"] if e.count(".") == 3 and "-" not in e]
        if known_ec4s:
            n_known_ec4 += 1
            if top_ec in known_ec4s:
                known_ec4_match += 1
            if any(".".join(top_ec.split(".")[:3]) == ".".join(e.split(".")[:3]) for e in known_ec4s):
                known_ec3_match += 1

        if r["is_unannotated_no_ec"] and top_prob >= locked_tau:
            rec = {k: v for k, v in r.items() if k != "sequence"}
            rec["pred_ec4"] = top_ec
            rec["pred_prob"] = round(top_prob, 4)
            rec["fills_kegg_hole"] = top_ec in ec_to_missing_pws
            rec["filled_pathways"] = ec_to_missing_pws.get(top_ec, [])
            unannotated_pred.append(rec)
            if top_ec in ec_to_missing_pws:
                pathway_hole_fills.append(rec)
                if r["max_train_pident"] < 30.0:
                    pathway_hole_fills_dark.append(rec)
                for pw in ec_to_missing_pws[top_ec]:
                    improved_pathways[pw["pathway_id"]].append({
                        "pathway_name": pw["pathway_name"],
                        "n_present_kegg": pw["n_present_kegg"],
                        "n_ref_total": pw["n_ref_total"],
                        "accession": r["accession"],
                        "protein_name": r["protein_name"],
                        "pred_ec4": top_ec,
                        "pred_prob": round(top_prob, 4),
                        "max_train_pident": r["max_train_pident"],
                    })

    pathway_hole_fills.sort(key=lambda x: (-x["pred_prob"], x["max_train_pident"]))
    pathway_hole_fills_dark.sort(key=lambda x: (-x["pred_prob"], x["max_train_pident"]))

    # Summarize improved pathways
    pw_summary = []
    for pw_id, fills in improved_pathways.items():
        unique_ecs = sorted({f["pred_ec4"] for f in fills})
        pw_summary.append({
            "pathway_id": pw_id,
            "pathway_name": fills[0]["pathway_name"],
            "n_present_kegg": fills[0]["n_present_kegg"],
            "n_ref_total": fills[0]["n_ref_total"],
            "n_new_ecs_filled": len(unique_ecs),
            "new_ecs_filled": unique_ecs,
            "n_dark_proteins": sum(1 for f in fills if f["max_train_pident"] < 30.0),
            "top_protein": max(fills, key=lambda x: x["pred_prob"]),
        })
    pw_summary.sort(key=lambda x: (-x["n_new_ecs_filled"], -x["top_protein"]["pred_prob"]))

    out = {
        "organism": "Methanocaldococcus jannaschii DSM 2661",
        "proteome_id": "UP000000805",
        "kegg_org": "mja",
        "n_total_proteins": len(rows),
        "n_known_complete_ec4": n_known_ec4,
        "known_ec4_top1_acc": round(known_ec4_match / max(1, n_known_ec4), 4),
        "known_ec3_top1_acc": round(known_ec3_match / max(1, n_known_ec4), 4),
        "n_unannotated_no_ec": unannotated_total,
        "n_unannotated_pred_enzyme_ge_tau062": len(unannotated_pred),
        "n_pathway_hole_fill_proteins": len(pathway_hole_fills),
        "n_pathway_hole_fill_unique_ecs": len({x["pred_ec4"] for x in pathway_hole_fills}),
        "n_pathway_hole_fill_dark_lt30": len(pathway_hole_fills_dark),
        "n_kegg_pathways_improved": len(pw_summary),
        "elapsed_seconds": round(time.time() - t0, 1),
        "improved_pathways_top20": pw_summary[:20],
        "top_25_pathway_hole_fills": pathway_hole_fills[:25],
        "top_15_dark_pathway_hole_fills": pathway_hole_fills_dark[:15],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(
        f"[DONE Ext3] ({out['elapsed_seconds']}s) total={len(rows)}, known_EC4_acc={out['known_ec4_top1_acc']}, "
        f"unannotated={unannotated_total}, pred_enzymes={len(unannotated_pred)}, "
        f"pathway_hole_proteins={len(pathway_hole_fills)} (unique_ECs={out['n_pathway_hole_fill_unique_ecs']}, "
        f"dark<30%={len(pathway_hole_fills_dark)}), improved_KEGG_pathways={len(pw_summary)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
