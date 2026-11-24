#!/usr/bin/env python
"""Mode 1 零样本检索评测（研究计划 3.4 / 5.2）。

在 val/test 中「训练未见 EC」的样本上，统计全词表检索的 Top-K 召回与真实排名，
不使用任何标签、不使用分类阈值、不做 top-K 截断。

两条通路（计划 3.1 双模式）：
  --path mode1  : 阶段一冻结快照（对比投影头 + EC 本体编码器），标准零样本检索
  --path mode2  : 阶段二监督模型（融合嵌入 + sigmoid(logits)），仅作对照

同时给出「分类分数 top-100 截断」口径，用于区分
「评测截断导致 zeroshot_recall@10 恒为 0」与「模型本身不具备零样本检索能力」。

--zero-structure 为计划 5.4 的「移除结构模态」消融，同时用于模拟外部基准
（NEW-392 / Price-149 不在 3Di 缓存中，评测时 3Di 一律零向量）的特征缺失。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import CachedFeatureDataset, EmbeddingCache, cached_collate  # noqa: E402
from data.dataset import build_domain_vocab, load_split  # noqa: E402
from models.contrastive import ZymeBindContrastive  # noqa: E402
from models.encoders.ec_ontology_gnn import ECOntologyEncoder  # noqa: E402
from training_cached import load_cfg  # noqa: E402
from training_zymebind import (CachedEncoder, MODALITIES, TIERS, ZymeBindModel,  # noqa: E402
                               make_edge_index, ordered_feats)


def load_checkpoint(ckpt_path: Path, tier: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vocab_path = ROOT / str(ckpt.get("vocab", TIERS[tier][1]))
    vocab = json.loads(vocab_path.read_text())
    ec2id = vocab["ec2id"]
    edge_index = make_edge_index(vocab_path, ec2id, device)
    return ckpt, vocab_path, ec2id, edge_index


def build_mode1(ckpt, vocab_path, edge_index, device, cfg):
    if "mode1" not in ckpt:
        raise RuntimeError("checkpoint 无 Mode 1 快照，请用双模式版 training_zymebind.py 重训")
    contrastive = ZymeBindContrastive(
        {k: CachedEncoder() for k in MODALITIES}, ckpt["in_dims"],
        cfg["model"]["projection_dim"], cfg["model"]["temperature"],
        n_experts=int(ckpt.get("moe_experts", 1)),
        diversity_weight=float(ckpt.get("diversity_weight", 0.0))).to(device)
    contrastive.load_state_dict(ckpt["mode1"]["contrastive"])
    ec_encoder = ECOntologyEncoder(str(vocab_path),
                                   dim=cfg["model"]["projection_dim"],
                                   layers=2).to(device)
    ec_encoder.load_state_dict(ckpt["mode1"]["ec_encoder"])
    contrastive.eval()
    ec_encoder.eval()
    return contrastive, ec_encoder


def build_mode2(ckpt, vocab_path, edge_index, ec2id, device, cfg):
    ec_encoder = ECOntologyEncoder(str(vocab_path),
                                   dim=cfg["model"]["projection_dim"],
                                   layers=2).to(device)
    fusion = cfg["model"]["fusion"]
    model = ZymeBindModel(ckpt["in_dims"], ec_encoder, edge_index, len(ec2id),
                          fusion["out_dim"], fusion["n_heads"], fusion["dropout"],
                          cfg["model"]["projection_dim"],
                          bool(cfg["finetune"]["use_concept_bottleneck"]),
                          head_type=ckpt.get("head_type", "dot")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def recall_at(ranks: list[int], k: int) -> float:
    return sum(r <= k for r in ranks) / max(len(ranks), 1)


def rank_block(ranks: list[int]) -> dict:
    return {
        "recall@1": recall_at(ranks, 1),
        "recall@5": recall_at(ranks, 5),
        "recall@10": recall_at(ranks, 10),
        "recall@100": recall_at(ranks, 100),
        "median_rank": statistics.median(ranks) if ranks else None,
        "mean_rank": sum(ranks) / max(len(ranks), 1),
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), default="p1m1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--path", choices=("mode1", "mode2"), default="mode1")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke 用：限制评测蛋白数")
    ap.add_argument("--zero-structure", action="store_true",
                    help="消融：把 3Di 结构模态置零（模拟外部基准特征缺失）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg(str(ROOT / "configs/pretrain.yaml"))
    ckpt_path = Path(args.ckpt or ROOT / f"checkpoints/zymebind_{args.tier}_hierprior.pt")
    ckpt, vocab_path, ec2id, edge_index = load_checkpoint(ckpt_path, args.tier, device)
    contrastive = ec_encoder = model = None
    if args.path == "mode1":
        contrastive, ec_encoder = build_mode1(ckpt, vocab_path, edge_index, device, cfg)
    else:
        model = build_mode2(ckpt, vocab_path, edge_index, ec2id, device, cfg)

    splits_dir = ROOT / TIERS[args.tier][0]
    train_rows = load_split(splits_dir / "train.tsv")
    rows = load_split(splits_dir / f"{args.split}.tsv")
    seen = {e for r in train_rows for e in r["ec_numbers"]}
    unseen = {e for r in rows for e in r["ec_numbers"]} - seen
    targets = [r for r in rows if any(e in unseen for e in r["ec_numbers"])]
    if args.limit:
        targets = targets[:args.limit]
    print(f"[info] tier={args.tier} path={args.path} split={args.split} "
          f"n_eval={len(targets)} n_unseen_ec={len(unseen)} "
          f"n_classes={len(ec2id)} zero_structure={args.zero_structure}", flush=True)

    domain2id = build_domain_vocab(train_rows)
    seq_cache = EmbeddingCache("sequence")
    text_cache = EmbeddingCache("text")
    threedi_cache = EmbeddingCache("threedi_observed_afdbv6")
    ds = CachedFeatureDataset(targets, ec2id, domain2id, seq_cache, text_cache,
                              threedi_cache=threedi_cache)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, collate_fn=cached_collate)

    ranks_ret: list[int] = []
    ranks_cls: list[int] = []
    in_top100 = 0
    n_assign = 0
    for batch in dl:
        feats = ordered_feats(batch, device)
        if args.zero_structure:
            feats["structure"] = torch.zeros_like(feats["structure"])
        if args.path == "mode1":
            embs = contrastive({k: {"x": v} for k, v in feats.items()})
            z = F.normalize(torch.stack([embs[k] for k in MODALITIES], 0).mean(0), dim=-1)
            ec = F.normalize(ec_encoder(edge_index), dim=-1)
            cls_scores = z @ ec.t()
        else:
            logits, z = model(feats)
            ec = F.normalize(model.ec_encoder(model.edge_index), dim=-1)
            cls_scores = torch.sigmoid(logits)
        sim_ret = z @ ec.t()
        rank_ret = sim_ret.argsort(dim=1, descending=True).argsort(dim=1)
        order_cls = cls_scores.argsort(dim=1, descending=True)
        rank_cls = order_cls.argsort(dim=1)
        for i, ecs in enumerate(batch["ec_numbers"]):
            top100 = set(order_cls[i, :100].tolist())
            for e in ecs:
                j = ec2id.get(e)
                if e not in unseen or j is None:
                    continue
                ranks_ret.append(int(rank_ret[i, j]) + 1)
                ranks_cls.append(int(rank_cls[i, j]) + 1)
                in_top100 += int(j in top100)
                n_assign += 1

    res = {
        "tier": args.tier,
        "split": args.split,
        "path": args.path,
        "zero_structure": args.zero_structure,
        "ckpt": str(ckpt_path.relative_to(ROOT)) if ckpt_path.is_relative_to(ROOT)
                else str(ckpt_path),
        "n_eval_proteins": len(targets),
        "n_unseen_ec": len(unseen),
        "n_unseen_assignments": n_assign,
        "n_classes": len(ec2id),
        "mode1_cosine_retrieval": rank_block(ranks_ret),
        "classifier_score_ranking": rank_block(ranks_cls),
        "classifier_top100_coverage": in_top100 / max(n_assign, 1),
    }
    out = Path(args.out or ROOT /
               f"results/zeroshot_{args.tier}_{args.split}_{args.path}_hierprior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    print(f"[ok] -> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}",
          flush=True)


if __name__ == "__main__":
    main()
