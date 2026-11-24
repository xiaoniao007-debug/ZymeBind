"""缓存模式数据集：直接读离线嵌入，跳过 PLM 前向（闭环验证/消融用）。

与 data/dataset.py 的端到端模式共用同一套标签构造规则（label_and_weight），
保证多功能酶降权、部分 EC 层级监督在两种模式下完全一致。

输入特征（4 个 tensor = 3 类离线嵌入缓存 + 1 个运行时多热向量）:
    sequence : ESM-2 650M pooled (1280)，离线缓存
    text     : BioBERT pooled (768)，离线缓存；无功能描述为 0 向量
    threedi  : foldseek 观测 3Di → ProstT5 encoder pooled (1024)，离线缓存；可选，缺结构为 0 向量
    domain   : InterPro/Pfam/... multi-hot (n_domains)，训练时由划分 TSV 现算，不是嵌入缓存
    ec_ontology 未接入当前模型；结构以 3Di 嵌入形式进入，不是坐标/图。

术语「三类嵌入 vs 四模态」的唯一定义见 docs/ZymeBind_数据集构造与版本命名.md 第 12 节。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import evidence_sample_weight, label_and_weight


class EmbeddingCache:
    """按 accession 索引分片 .npy（mmap，按需加载）。"""

    def __init__(self, modality: str, root: str | Path = "data/processed/embeddings"):
        root = Path(root)
        idx = root / f"index_{modality}.tsv"
        self.index: dict[str, tuple[int, int]] = {}
        with open(idx) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                self.index[r["accession"]] = (int(r["shard"]), int(r["row"]))
        self.shards = {}
        for s in {sh for sh, _ in self.index.values()}:
            self.shards[s] = np.load(root / modality / f"shard{s}.npy", mmap_mode="r")
        self.dim = int(next(iter(self.shards.values())).shape[1])
        print(f"[cache] {modality}: {len(self.index)} 条, dim={self.dim}, "
              f"{len(self.shards)} shards")

    def get(self, acc: str) -> torch.Tensor:
        if acc not in self.index:
            return torch.zeros(self.dim)
        sh, row = self.index[acc]
        return torch.from_numpy(np.asarray(self.shards[sh][row], dtype=np.float32))


class CachedFeatureDataset(Dataset):
    def __init__(self, rows: list[dict], ec2id: dict[str, int], domain2id: dict[str, int],
                 seq_cache: EmbeddingCache, text_cache: EmbeddingCache | None = None,
                 anc_soft: float = 0.0, threedi_cache: EmbeddingCache | None = None,
                 evidence_weight: float = 0.0):
        self.evidence_weight = evidence_weight  # S5：0=关闭（M0/M1 默认）
        self.rows = rows
        self.ec2id = ec2id
        self.domain2id = domain2id
        self.n_domains = len(domain2id)
        self.seq = seq_cache
        self.text = text_cache
        self.threedi = threedi_cache  # ProstT5 3Di 嵌入（结构模态，可选）
        self.anc_soft = anc_soft  # 祖先闭包软标签强度（0=关闭）

    def __len__(self):
        return len(self.rows)

    def _domain_multihot(self, r: dict) -> torch.Tensor:
        v = torch.zeros(self.n_domains)
        for db, items in (r.get("domains") or {}).items():
            for it in items:
                j = self.domain2id.get(f"{db}:{it}")
                if j is not None:
                    v[j] = 1.0
        return v

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        feats = {"sequence": self.seq.get(r["accession"])}
        if self.text is not None:
            feats["text"] = self.text.get(r["accession"])
        if self.threedi is not None:
            feats["threedi"] = self.threedi.get(r["accession"])
        feats["domain"] = self._domain_multihot(r)
        y, w, primary = label_and_weight(self.ec2id, r["ec_numbers"],
                                         r.get("ec_partial") or [],
                                         anc_soft=self.anc_soft)
        return {
            "accession": r["accession"],
            "feats": feats,
            "labels": y,
            "label_weights": w,
            "sample_weight": evidence_sample_weight(
                r.get("protein_existence", ""), self.evidence_weight),
            "primary_ec": primary,
            "is_multi_ec": len(r["ec_numbers"]) > 1,
            "ec_numbers": r["ec_numbers"],
        }


def cached_collate(batch: list[dict]) -> dict:
    keys = batch[0]["feats"].keys()
    return {
        "feats": {k: torch.stack([b["feats"][k] for b in batch]) for k in keys},
        "labels": torch.stack([b["labels"] for b in batch]),
        "label_weights": torch.stack([b["label_weights"] for b in batch]),
        "sample_weights": torch.tensor([b["sample_weight"] for b in batch],
                                       dtype=torch.float32),
        "is_multi_ec": torch.tensor([b["is_multi_ec"] for b in batch]),
        "primary_ec": [b["primary_ec"] for b in batch],
        "accession": [b["accession"] for b in batch],
        "ec_numbers": [b["ec_numbers"] for b in batch],
    }
