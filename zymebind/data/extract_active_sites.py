#!/usr/bin/env python
"""从本地 data/raw/uniprot_rest_tsv_p1/*.tsv 提取 Active site 与 Binding site 残基坐标。

输出：data/processed/active_sites_eval.json（不在 embeddings/ 下，不影响 provenance 校验）。
覆盖 p1m1_benchclusterfree/test.tsv 与 NEW-392 中的酶，供 L1 残基级可解释性定量评测。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data/processed/active_sites_eval.json"

ACT_RE = re.compile(r"ACT_SITE\s+(\d+)(?:\.\.(\d+))?")
BIND_RE = re.compile(r"BINDING\s+(\d+)(?:\.\.(\d+))?")


def load_target_accs() -> tuple[set[str], set[str]]:
    test_accs, new392_accs = set(), set()
    with open(ROOT / "data/splits_p1m1_benchclusterfree/test.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            test_accs.add(r["accession"])
    with open(ROOT / "evaluation/benchmarks/new392.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            new392_accs.add(r["accession"])
    return test_accs, new392_accs


def parse_sites(text: str, pat: re.Pattern) -> list[int]:
    pos: set[int] = set()
    for m in pat.finditer(text or ""):
        s = int(m.group(1))
        e = int(m.group(2)) if m.group(2) else s
        if e - s <= 25:
            pos.update(range(s, e + 1))
    return sorted(pos)


def main():
    test_accs, new392_accs = load_target_accs()
    targets = test_accs | new392_accs
    annots: dict[str, dict[str, list[int]]] = {}

    for tsv_dir in (ROOT / "data/raw/uniprot_rest_tsv_p1", ROOT / "data/raw/uniprot_rest_tsv"):
        if not tsv_dir.exists():
            continue
        for p in sorted(tsv_dir.glob("*.tsv")):
            with p.open() as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for r in reader:
                    acc = r.get("Entry", "")
                    if acc not in targets or acc in annots:
                        continue
                    act = parse_sites(r.get("Active site", ""), ACT_RE)
                    bind = parse_sites(r.get("Binding site", ""), BIND_RE)
                    if act or bind:
                        annots[acc] = {"act_site": act, "binding": bind}

    out = {
        "source": "data/raw/uniprot_rest_tsv_p1/*.tsv",
        "counts": {
            "test_with_act_site": sum(1 for a in test_accs if a in annots and annots[a]["act_site"]),
            "test_with_act_or_bind": sum(1 for a in test_accs if a in annots),
            "new392_with_act_site": sum(1 for a in new392_accs if a in annots and annots[a]["act_site"]),
            "new392_with_act_or_bind": sum(1 for a in new392_accs if a in annots),
        },
        "annotations": annots,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[DONE] saved {OUT_PATH}: {out['counts']}")


if __name__ == "__main__":
    main()
