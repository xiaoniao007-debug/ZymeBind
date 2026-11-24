#!/usr/bin/env python
"""从本地 ExPASy enzyme.dat 构建 EC 本体节点 BioBERT 语义先验嵌入。

输出保存于 data/processed/ec_semantic/{tier}_ec_biobert.pt（不在 embeddings/ 下，
不影响 input_provenance_audit.json 的 embedding_cache 清单校验）。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ENZYME_DAT = ROOT / "data/raw/enzyme.dat"
OUT_DIR = ROOT / "data/processed/ec_semantic"
BIOBERT_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

LEVEL1_DESC = {
    "1": "EC 1 Oxidoreductases: enzymes catalyzing oxidation-reduction reactions.",
    "2": "EC 2 Transferases: enzymes transferring functional groups from one molecule to another.",
    "3": "EC 3 Hydrolases: enzymes catalyzing the hydrolysis of various bonds.",
    "4": "EC 4 Lyases: enzymes cleaving bonds by means other than hydrolysis and oxidation.",
    "5": "EC 5 Isomerases: enzymes catalyzing geometric or structural changes within a single molecule.",
    "6": "EC 6 Ligases: enzymes joining two molecules with covalent bonds coupled to ATP hydrolysis.",
    "7": "EC 7 Translocases: enzymes catalyzing the movement of ions or molecules across membranes.",
}


def parse_enzyme_dat_rich(path: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    cur = None
    with open(path, errors="replace") as fh:
        for line in fh:
            code, body = line[:2], line[5:].strip()
            if code == "ID":
                cur = {"ec": body.rstrip(";").strip(), "de": [], "an": [], "ca": [], "cf": []}
            elif cur is None:
                continue
            elif code == "DE":
                cur["de"].append(body)
            elif code == "AN":
                cur["an"].append(body.rstrip("."))
            elif code == "CA":
                cur["ca"].append(body)
            elif code == "CF":
                cur["cf"].append(body)
            elif code == "//":
                entries[cur["ec"]] = cur
                cur = None
    return entries


def build_text(ec: str, entries: dict[str, dict]) -> str | None:
    if ec in LEVEL1_DESC:
        return LEVEL1_DESC[ec]
    e = entries.get(ec)
    if not e:
        return None
    parts = [f"EC {ec}: {' '.join(e['de']).strip()}"]
    if e["an"]:
        parts.append("Alternate names: " + "; ".join(e["an"][:5]))
    if e["ca"]:
        parts.append("Catalytic activity: " + " ".join(e["ca"]))
    if e["cf"]:
        parts.append("Cofactor: " + " ".join(e["cf"]))
    return " ".join(parts)


@torch.no_grad()
def encode_texts(texts: list[str], tok, model, device: str, bs: int = 128) -> torch.Tensor:
    outs = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        outs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outs, dim=0)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = parse_enzyme_dat_rich(ENZYME_DAT)
    print(f"[enzyme.dat] parsed {len(entries)} EC entries")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BIOBERT_NAME, local_files_only=True)
    model = AutoModel.from_pretrained(BIOBERT_NAME, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()

    vocabs = {
        "m0": ROOT / "data/processed/ec_vocab.json",
        "p1m1": ROOT / "data/processed/p1m1/ec_vocab.json",
        "p1m2": ROOT / "data/processed/p1m2/ec_vocab.json",
    }

    all_ecs = set()
    for vp in vocabs.values():
        all_ecs.update(json.loads(vp.read_text())["ec2id"].keys())

    ec_with_text = {}
    for ec in sorted(all_ecs):
        t = build_text(ec, entries)
        if t:
            ec_with_text[ec] = t

    keys = sorted(ec_with_text.keys())
    embs_raw = encode_texts([ec_with_text[k] for k in keys], tok, model, device)
    ec2emb = {k: embs_raw[i] for i, k in enumerate(keys)}
    print(f"[BioBERT] encoded {len(ec2emb)}/{len(all_ecs)} EC nodes with direct text descriptions")

    dim = embs_raw.shape[1]
    for tier, vp in vocabs.items():
        ec2id = json.loads(vp.read_text())["ec2id"]
        n = len(ec2id)
        children_by_prefix = defaultdict(list)
        for ec in ec2id:
            if ec in ec2emb:
                parts = ec.split(".")
                for lvl in range(1, len(parts)):
                    children_by_prefix[".".join(parts[:lvl])].append(ec2emb[ec])

        mat = torch.zeros(n, dim, dtype=torch.float32)
        direct_hit = 0
        agg_hit = 0
        fallback = 0
        for ec, idx in ec2id.items():
            if ec in ec2emb:
                mat[idx] = ec2emb[ec]
                direct_hit += 1
            elif ec in children_by_prefix:
                mat[idx] = F.normalize(torch.stack(children_by_prefix[ec]).mean(0), dim=-1)
                agg_hit += 1
            else:
                parts = ec.split(".")
                found = False
                for lvl in range(len(parts) - 1, 0, -1):
                    p = ".".join(parts[:lvl])
                    if p in ec2emb:
                        mat[idx] = ec2emb[p]
                        found = True
                        break
                    if p in children_by_prefix:
                        mat[idx] = F.normalize(torch.stack(children_by_prefix[p]).mean(0), dim=-1)
                        found = True
                        break
                if not found:
                    fallback += 1
        out_path = OUT_DIR / f"{tier}_ec_biobert.pt"
        torch.save({"ec2id": ec2id, "embeddings": mat, "model": BIOBERT_NAME}, out_path)
        print(f"  [{tier}] saved {out_path.name}: shape={tuple(mat.shape)} direct={direct_hit} agg={agg_hit} fallback={fallback}")


if __name__ == "__main__":
    main()
