"""多模态数据集：读取 splits/*.tsv，产出五模态输入（方案 3.2）。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

# 次级 EC 标签（与主 EC 仅第 4 位不同）的损失权重，其余标签权重 1.0。
# 依据：多功能酶中 61% 的多个 EC 只差第 4 位，多为注释粒度差异而非真双功能。
GRANULARITY_WEIGHT = 0.5

# S5 evidence-aware 加权：UniProt "Protein existence" 证据层级 -> 样本损失权重。
# M2 全量中 77% 为同源推断注释，标签噪声高；M1/M0 默认 strength=0 不受影响。
EVIDENCE_WEIGHT = {
    "Evidence at protein level": 1.0,
    "Evidence at transcript level": 0.9,
    "Inferred from homology": 0.5,
    "Predicted": 0.3,
    "Uncertain": 0.2,
    "": 1.0,           # postp0 无该字段：不做降权，保证旧实验完全可复现
}


def evidence_sample_weight(protein_existence: str, strength: float = 0.0) -> float:
    """strength=0 → 恒为 1.0（关闭）；strength=1 → 完全按证据层级降权。"""
    raw = EVIDENCE_WEIGHT.get(protein_existence or "", 1.0)
    return 1.0 - strength + strength * raw


def partial_node(ec2id: dict[str, int], ec: str) -> str | None:
    """部分 EC（如 3.4.24.-）→ 词表中最深的已定义前缀节点（P0 层级监督）。

    词表 6,110 节点包含 347 个层级中间节点（EC1-EC3），部分编号映射过去
    后作为"确定的家族归属"标签（权重 1.0），同时天然支持 EC-Bench 的
    completion 任务评测。
    """
    for cut in (3, 2, 1):
        node = ".".join(ec.split(".")[:cut])
        if node in ec2id:
            return node
    return None


def label_and_weight(ec2id: dict[str, int], ecs: list[str], partial: list[str] = (),
                     anc_soft: float = 0.0
                     ) -> tuple[torch.Tensor, torch.Tensor, str | None]:
    """多热标签 + 逐标签损失权重（端到端与缓存模式共用）。

    规则（方案讨论定稿）：
      - 完整 EC：列表首个（暂定主 EC）权重 1.0；仅第 4 位不同的次级 → 0.5；
        其余（跨前三层/跨一级类，属真多功能）→ 1.0；
      - 部分 EC（x.x.x.-）：映射到最深已定义前缀节点，权重 1.0；
      - anc_soft > 0 时：完整 EC 的 EC1-3 祖先闭包作为弱正例（软目标 anc_soft，
        max 保护不被覆盖已有 1.0；BCE/Focal 对连续目标天然支持）。
        动机：6,110 类长尾下稀有 EC4 表征弱（aupr_macro 落后基线），
        祖先监督把"家族归属"信号传播给浅层共享节点。
    """
    y = torch.zeros(len(ec2id))
    w = torch.ones(len(ec2id))
    ids = [e for e in ecs if e in ec2id]
    primary = ids[0] if ids else None
    p3 = primary.rsplit(".", 1)[0] if primary else None
    for i, e in enumerate(ids):
        idx = ec2id[e]
        y[idx] = 1.0
        if i > 0 and e.rsplit(".", 1)[0] == p3:
            w[idx] = GRANULARITY_WEIGHT
    for e in partial:
        node = partial_node(ec2id, e)
        if node:
            y[ec2id[node]] = 1.0
    if anc_soft > 0:
        for e in ids:
            parts = e.split(".")
            for cut in (1, 2, 3):
                node = ".".join(parts[:cut])
                if node == e:
                    continue
                idx = ec2id.get(node)
                if idx is not None:
                    y[idx] = max(float(y[idx]), anc_soft)
    return y, w, primary


def compute_pos_weight(rows: list[dict], ec2id: dict[str, int],
                       cap: float = 10.0) -> torch.Tensor:
    """逐类别正例权重 = min(cap, sqrt(n_neg / n_pos))，按训练集频率统计。

    6,110 类中 98% 以上是负例；不加此项时 Focal Loss 会被负类主导，
    模型退化成"全预测 0"（train loss→0 但 Fmax≈0）。
    端到端训练与缓存模式共用，保证两种模式口径一致。
    """
    from collections import Counter
    freq: Counter = Counter()
    for r in rows:
        freq.update(e for e in r["ec_numbers"] if e in ec2id)
        for p in (r.get("ec_partial") or []):
            for cut in (3, 2, 1):
                node = ".".join(p.split(".")[:cut])
                if node in ec2id:
                    freq[node] += 1
                    break
    n = len(rows)
    pw = torch.ones(len(ec2id))
    for ec, j in ec2id.items():
        f = max(freq.get(ec, 0), 1)
        # 下界 1.0：高频类（neg < pos）不应被压低，只放大稀有类
        pw[j] = max(1.0, min(cap, ((n - f) / f) ** 0.5))
    return pw


def load_split(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            r["ec_numbers"] = json.loads(r["ec_numbers"])
            r["ec_partial"] = json.loads(r["ec_partial"]) if r.get("ec_partial") else []
            r["ec_evidence"] = json.loads(r.get("ec_evidence") or "{}")
            r["domains"] = json.loads(r["domains"]) if r.get("domains") else {}
            r["protein_existence"] = r.get("protein_existence", "")
            rows.append(r)
    return rows


class MultiModalEnzymeDataset(Dataset):
    """返回 dict: sequence-text/ids、3Di 序列、域 id 列表、EC 多标签。

    3Di 序列由 data/build_3di.py 预计算并缓存为 {acc}.3di 文本文件。
    """

    def __init__(self, rows: list[dict], ec2id: dict[str, int], domain2id: dict[str, int],
                 seq_tokenizer, text_tokenizer, max_seq_len: int = 1024,
                 max_text_len: int = 512, threedi_dir: str | Path | None = None):
        self.rows = rows
        self.ec2id = ec2id
        self.domain2id = domain2id
        self.seq_tok = seq_tokenizer
        self.text_tok = text_tokenizer
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.threedi_dir = Path(threedi_dir) if threedi_dir else None

    def __len__(self):
        return len(self.rows)

    def _label(self, ecs: list[str]) -> torch.Tensor:
        y = torch.zeros(len(self.ec2id))
        for e in ecs:
            if e in self.ec2id:
                y[self.ec2id[e]] = 1.0
        return y

    def _partial_node(self, ec: str) -> str | None:
        return partial_node(self.ec2id, ec)

    def _label_and_weight(self, ecs: list[str], partial: list[str] = ()
                          ) -> tuple[torch.Tensor, torch.Tensor, str | None]:
        return label_and_weight(self.ec2id, ecs, partial)

    def _domains(self, r: dict) -> list[int]:
        ids = []
        for db, items in r["domains"].items():
            for it in items:
                key = f"{db}:{it}"
                if key in self.domain2id:
                    ids.append(self.domain2id[key])
        return ids or [0]

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        seq = r["sequence"][: self.max_seq_len]
        s = self.seq_tok(seq, truncation=True, max_length=self.max_seq_len,
                         padding="max_length", return_tensors="pt")
        text = r.get("function_text") or r.get("protein_name") or "enzyme"
        t = self.text_tok(text, truncation=True, max_length=self.max_text_len,
                          padding="max_length", return_tensors="pt")
        threedi = ""
        if self.threedi_dir:
            p = self.threedi_dir / f"{r['accession']}.3di"
            if p.exists():
                threedi = p.read_text().strip()
                if "\t" in threedi:  # 兼容 foldseek 原始输出格式
                    threedi = threedi.rsplit("\t", 1)[1].strip()
        ecs = r["ec_numbers"]
        y, w, primary = self._label_and_weight(ecs, r.get("ec_partial") or [])
        return {
            "accession": r["accession"],
            "sequence": seq,
            "threedi": threedi,
            "input_ids": s.input_ids.squeeze(0),
            "attention_mask": s.attention_mask.squeeze(0),
            "text_ids": t.input_ids.squeeze(0),
            "text_mask": t.attention_mask.squeeze(0),
            "domain_ids": torch.tensor(self._domains(r), dtype=torch.long),
            "labels": y,
            "label_weights": w,
            "primary_ec": primary,
            "n_ec": len(ecs),
            "is_multi_ec": len(ecs) > 1,
            "ec_numbers": ecs,
        }


def collate(batch: list[dict]) -> dict:
    """结构域子图需要动态拼接，单独在 trainer 中构建 edge_index。"""
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "label_weights": torch.stack([b["label_weights"] for b in batch]),
        "is_multi_ec": torch.tensor([b["is_multi_ec"] for b in batch]),
        "primary_ec": [b["primary_ec"] for b in batch],
        "domain_lists": [b["domain_ids"] for b in batch],
        "threedi": [b["threedi"] for b in batch],
        "accession": [b["accession"] for b in batch],
        "ec_numbers": [b["ec_numbers"] for b in batch],
    }
    return out


def build_domain_vocab(rows: list[dict], min_count: int = 3) -> dict[str, int]:
    """统计 InterPro/Pfam 等域出现频次，过滤低频后建立 id（0 保留给 padding）。"""
    from collections import Counter
    cnt = Counter()
    for r in rows:
        for db, items in r["domains"].items():
            for it in items:
                cnt[f"{db}:{it}"] += 1
    vocab = {"<pad>": 0}
    for k, c in cnt.most_common():
        if c < min_count:
            continue
        vocab[k] = len(vocab)
    return vocab
