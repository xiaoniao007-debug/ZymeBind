#!/usr/bin/env python
"""构建 EC 编号层级本体（4 层 DAG）并导出图结构。

对应研究方案 3.2（EC 本体模态）/ 4.2 第 4 步：
- 解析 EXPASY enzyme.dat，得到 EC -> 名称/别名；
- 由 EC 编号的层级结构（1 / 1.1 / 1.1.1 / 1.1.1.1）构建 DAG；
- 输出 nodes.tsv（ec, level, name）、edges.tsv（parent -> child）、
  以及 ec_vocab.json（id 映射，供图嵌入使用）。

用法:
    python data/ec_ontology.py --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from data.download_uniprot import download, load_config
except ImportError:  # 直接以脚本方式运行 (python data/ec_ontology.py)
    from download_uniprot import download, load_config


def parse_enzyme_dat(path: Path) -> dict[str, dict]:
    """解析 enzyme.dat: ID/DE/AN 三类记录。"""
    entries: dict[str, dict] = {}
    cur = None
    with open(path, errors="replace") as fh:
        for line in fh:
            code, body = line[:2], line[5:].strip()
            if code == "ID":
                cur = {"ec": body.rstrip(";").strip(), "name": "", "alt_names": []}
            elif cur is None:
                continue
            elif code == "DE":
                cur["name"] = (cur["name"] + " " + body).strip()
            elif code == "AN":
                cur["alt_names"].append(body.rstrip(";").strip())
            elif code == "//":
                entries[cur["ec"]] = cur
                cur = None
    return entries


def build_hierarchy(ec_set: set[str], level1_names: dict[str, str]):
    """由 EC 编号集合构建 1->2->3->4 层 DAG（缺失中间节点自动补全）。"""
    nodes: dict[str, dict] = {}

    def add(ec: str):
        if ec in nodes:
            return
        level = ec.count(".") + 1
        nodes[ec] = {"ec": ec, "level": level, "name": ""}

    for ec in ec_set:
        parts = ec.split(".")
        for i in range(1, len(parts) + 1):
            add(".".join(parts[:i]))

    edges = []
    for ec in list(nodes):
        if nodes[ec]["level"] > 1:
            parent = ec.rsplit(".", 1)[0]
            edges.append((parent, ec))

    for ec, meta in nodes.items():
        if meta["level"] == 1:
            meta["name"] = level1_names.get(ec, "")
    return nodes, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--annotated-tsv", default=None,
                    help="uniprot_enzymes.tsv，用于只保留出现的 EC 编号")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir, out_dir = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dat = raw_dir / Path(cfg["ec_ontology"]["enzyme_dat_url"]).name
    if not dat.exists():
        download(cfg["ec_ontology"]["enzyme_dat_url"], dat)

    enzyme = parse_enzyme_dat(dat)
    print(f"[ok] enzyme.dat 解析 {len(enzyme)} 条 EC 记录")

    annotated: set[str] = set()
    tsv = Path(args.annotated_tsv) if args.annotated_tsv else out_dir / "uniprot_enzymes.tsv"
    if tsv.exists():
        import csv
        with open(tsv) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                annotated.update(json.loads(row["ec_numbers"]))
        print(f"[ok] 训练集覆盖 {len(annotated)} 个 EC 编号")

    ec_set = annotated or set(enzyme)
    nodes, edges = build_hierarchy(ec_set, cfg["ec_ontology"]["level1_names"])

    # 用 enzyme.dat 补全名称
    for ec, meta in nodes.items():
        if ec in enzyme and enzyme[ec]["name"]:
            meta["name"] = enzyme[ec]["name"]

    nodes_sorted = sorted(nodes.values(), key=lambda x: (x["level"], x["ec"]))
    with open(out_dir / "ec_nodes.tsv", "w") as fh:
        fh.write("ec\tlevel\tname\n")
        for n in nodes_sorted:
            fh.write(f"{n['ec']}\t{n['level']}\t{n['name']}\n")

    with open(out_dir / "ec_edges.tsv", "w") as fh:
        fh.write("parent\tchild\n")
        for p, c in edges:
            fh.write(f"{p}\t{c}\n")

    vocab = {n["ec"]: i for i, n in enumerate(nodes_sorted)}
    with open(out_dir / "ec_vocab.json", "w") as fh:
        json.dump({"ec2id": vocab, "id2ec": {v: k for k, v in vocab.items()},
                   "level1_names": cfg["ec_ontology"]["level1_names"]}, fh, indent=2)

    dist = defaultdict(int)
    for n in nodes_sorted:
        dist[n["level"]] += 1
    print(f"[ok] EC 图: {len(nodes_sorted)} 节点 / {len(edges)} 边; 层级分布 {dict(dist)}")
    print(f"[ok] -> {out_dir/'ec_nodes.tsv'}, {out_dir/'ec_edges.tsv'}, {out_dir/'ec_vocab.json'}")


if __name__ == "__main__":
    main()
