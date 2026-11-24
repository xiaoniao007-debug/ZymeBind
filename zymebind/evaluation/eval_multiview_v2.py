#!/usr/bin/env python
"""多视角内部验证/测试评测脚本（Full / Seq+3Di / Seq-only + Mode 1 零样本）。

统一采用 full_scores（min_score=0.01）CAFA 口径，不截断 top-K。
支持评测：
  - baseline: ECClassifier (cached_full_*.pt)
  - v1: ZymeBindModel (zymebind_*_hierprior.pt 等)
  - v2: ZymeBindV2Model (zymebind_v2_*.pt)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import CachedFeatureDataset, EmbeddingCache, cached_collate  # noqa: E402
from data.dataset import build_domain_vocab, load_split  # noqa: E402
from evaluation.cafa_eval import evaluate_all  # noqa: E402
from evaluation.zeroshot_eval import build_mode1, build_mode2, load_checkpoint, rank_block  # noqa: E402
from models.classifier import ECClassifier  # noqa: E402
from training_cached import load_cfg  # noqa: E402
from training_zymebind import MODALITIES, TIERS, ordered_feats  # noqa: E402

VIEWS = ("full", "seq_struct", "seq_only")


def mask_view(feats: dict[str, torch.Tensor], view: str, key_map: str = "zymebind") -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in feats.items()}
    s_key = "structure" if key_map == "zymebind" else "threedi"
    if view == "seq_struct":
        out["text"].zero_()
        out["domain"].zero_()
    elif view == "seq_only":
        out[s_key].zero_()
        out["text"].zero_()
        out["domain"].zero_()
    return out


def probs_to_cafa_scores(probs_list: list[np.ndarray], labels_list: list[set[str]],
                         id2ec: dict[int, str], min_score: float = 0.01) -> tuple[list[dict[str, float]], list[set[str]]]:
    scores, labels = [], []
    probs = np.concatenate(probs_list, axis=0)
    for i, lab in enumerate(labels_list):
        if not lab:
            continue
        p = probs[i]
        idx = np.where(p >= min_score)[0]
        row = {id2ec[int(j)]: float(p[j]) for j in idx}
        scores.append(row)
        labels.append(lab)
    return scores, labels


@torch.no_grad()
def eval_supervised_views(model, loader, id2ec: dict[int, str], unseen_ecs: set[str],
                          device: str, kind: str = "v1", min_score: float = 0.01) -> dict:
    model.eval()
    res = {}
    for view in VIEWS:
        probs_list, labels_list = [], []
        for batch in loader:
            if kind == "baseline":
                feats = {k: v.to(device) for k, v in batch["feats"].items()}
                feats = mask_view(feats, view, "cached")
                logits = model(feats)
            elif kind == "v1":
                feats = mask_view(ordered_feats(batch, device), view, "zymebind")
                logits, _ = model(feats)
            elif kind == "v2":
                feats = mask_view(ordered_feats(batch, device), view, "zymebind")
                logits, _ = model(feats)
            else:
                raise ValueError(kind)
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
            labels_list.extend(set(x) for x in batch["ec_numbers"])
        scores, labels = probs_to_cafa_scores(probs_list, labels_list, id2ec, min_score=min_score)
        m = evaluate_all(scores, labels, unseen_ecs=unseen_ecs)
        res[view] = {
            "fmax": round(float(m["fmax"]), 4),
            "threshold": round(float(m.get("fmax_threshold", m.get("threshold", 0.0))), 4),
            "aupr_macro": round(float(m["aupr_macro"]), 4),
            "f1_macro_L4": round(float(m.get("f1_macro@L4", 0.0)), 4),
            "recall@1": round(float(m.get("recall@1", 0.0)), 4),
            "n_eval": int(m.get("n_samples", len(labels))),
        }
    return res


@torch.no_grad()
def eval_mode1_views(contrastive, ec_enc, edge_index, loader, ec2id: dict[str, int],
                     unseen_ec4_ids: set[int], device: str, is_v2: bool = False) -> dict:
    contrastive.eval()
    ec_enc.eval()
    ec_emb = F.normalize(ec_enc(edge_index), dim=-1)
    res = {}
    for view in VIEWS:
        ranks = []
        for batch in loader:
            feats = mask_view(ordered_feats(batch, device), view, "zymebind")
            if is_v2:
                z = contrastive.encode_query(feats)
            else:
                embs = contrastive({k: {"x": feats[k]} for k in MODALITIES})
                if view == "full":
                    present = {k: (feats[k].abs().sum(-1, keepdim=True) > 1e-6).float() for k in MODALITIES}
                    num = sum(embs[k] * present[k] for k in MODALITIES)
                    den = sum(present.values()).clamp(min=1.0)
                    z = F.normalize(num / den, dim=-1)
                elif view == "seq_struct":
                    present_s = (feats["structure"].abs().sum(-1, keepdim=True) > 1e-6).float()
                    z = F.normalize((embs["sequence"] + embs["structure"] * present_s) / (1.0 + present_s), dim=-1)
                else:
                    z = F.normalize(embs["sequence"], dim=-1)
            sims = z @ ec_emb.t()
            for i, ecs in enumerate(batch["ec_numbers"]):
                for e in ecs:
                    cid = ec2id.get(e)
                    if cid is not None and cid in unseen_ec4_ids:
                        r = int((sims[i] > sims[i, cid]).sum().item()) + 1
                        ranks.append(r)
        rb = rank_block(ranks)
        res[view] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rb.items()}
        res[view]["n_pairs"] = len(ranks)
    return res


@torch.no_grad()
def eval_supervised_views_fast(model, loader, id2ec: dict[int, str],
                               device: str) -> dict:
    """向量化 GPU 精确 CAFA leaf-EC4 Fmax，供训练期逐轮筛选（M2 3 万验证集 <1 秒）。"""
    model.eval()
    n_classes = len(id2ec)
    ec4_idx = torch.tensor(
        [j for j in range(n_classes) if id2ec[j].count(".") == 3],
        dtype=torch.long, device=device,
    )
    ec2id = {v: k for k, v in id2ec.items()}
    thresholds = [0.05 + 0.025 * i for i in range(37)]  # 0.05 .. 0.95

    res = {}
    for view in VIEWS:
        tp_sum = torch.zeros(len(thresholds), device=device)
        prec_sum = torch.zeros(len(thresholds), device=device)
        prec_cnt = torch.zeros(len(thresholds), device=device)
        rec_sum = torch.zeros(len(thresholds), device=device)
        n_total = 0

        for batch in loader:
            feats = mask_view(ordered_feats(batch, device), view, "zymebind")
            logits, _ = model(feats)
            probs = torch.sigmoid(logits)[:, ec4_idx]  # (B, C4)

            # Build exact leaf-EC4 boolean target matrix
            B = probs.size(0)
            y_full = torch.zeros(B, n_classes, dtype=torch.bool, device=device)
            valid_rows = []
            for i, ecs in enumerate(batch["ec_numbers"]):
                hits = [ec2id[e] for e in ecs if e in ec2id and e.count(".") == 3]
                if hits:
                    y_full[i, hits] = True
                    valid_rows.append(i)
            if not valid_rows:
                continue
            v_idx = torch.tensor(valid_rows, dtype=torch.long, device=device)
            p = probs[v_idx]
            y = y_full[v_idx][:, ec4_idx]
            true_cnt = y.sum(dim=1).float().clamp(min=1.0)
            n_total += len(valid_rows)

            for ti, t in enumerate(thresholds):
                pred = p >= t
                tp = (pred & y).sum(dim=1).float()
                pc = pred.sum(dim=1).float()
                has_p = pc > 0
                if has_p.any():
                    prec_sum[ti] += (tp[has_p] / pc[has_p]).sum()
                    prec_cnt[ti] += has_p.sum().float()
                rec_sum[ti] += (tp / true_cnt).sum()

        best_f, best_t = 0.0, 0.5
        for ti, t in enumerate(thresholds):
            pr = float((prec_sum[ti] / prec_cnt[ti].clamp(min=1.0)).item()) if prec_cnt[ti] > 0 else 0.0
            rc = float((rec_sum[ti] / max(n_total, 1)).item())
            f1 = 2.0 * pr * rc / (pr + rc + 1e-12)
            if f1 > best_f:
                best_f, best_t = f1, t
        res[view] = {"fmax": round(best_f, 4), "threshold": round(best_t, 4), "n_eval": n_total}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="p1m1_benchclusterfree")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--kind", choices=("baseline", "v1", "v2"), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_cfg("configs/pretrain.yaml")
    sdir = ROOT / TIERS[args.tier][0]
    vpath = ROOT / TIERS[args.tier][1]
    vocab = json.loads(vpath.read_text())
    ec2id = vocab["ec2id"]
    id2ec = {int(v): k for k, v in ec2id.items()}

    train_rows = load_split(sdir / "train.tsv")
    eval_rows = load_split(sdir / f"{args.split}.tsv")
    domain2id = build_domain_vocab(train_rows)

    seq_c = EmbeddingCache("sequence")
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")
    ds = CachedFeatureDataset(eval_rows, ec2id, domain2id, seq_c, txt_c, anc_soft=0.0, threedi_cache=tdi_c)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=cached_collate)

    seen_ec = {e for r in train_rows for e in r["ec_numbers"] if e in ec2id}
    unseen_ecs = {e for r in eval_rows for e in r["ec_numbers"] if e in ec2id and e not in seen_ec and e.count(".") == 3}
    unseen_ec4_ids = {ec2id[e] for e in unseen_ecs}

    out = {"tier": args.tier, "split": args.split, "kind": args.kind, "ckpt": args.ckpt,
           "n_eval_rows": len(eval_rows), "n_unseen_ec4": len(unseen_ecs)}

    if args.kind == "baseline":
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        model = ECClassifier(ckpt["in_dims"], len(ec2id),
                             cfg["model"]["fusion"]["out_dim"],
                             cfg["model"]["fusion"]["n_heads"],
                             cfg["model"]["fusion"]["dropout"]).to(args.device)
        model.load_state_dict(ckpt["model"])
        out["supervised"] = eval_supervised_views(model, loader, id2ec, unseen_ecs, args.device, kind="baseline")
    elif args.kind == "v1":
        ckpt, vp, _, edge_index = load_checkpoint(Path(args.ckpt), args.tier, args.device)
        m2 = build_mode2(ckpt, vp, edge_index, ec2id, args.device, cfg)
        out["supervised"] = eval_supervised_views(m2, loader, id2ec, unseen_ecs, args.device, kind="v1")
        contrastive, ec_enc = build_mode1(ckpt, vp, edge_index, args.device, cfg)
        out["mode1_zeroshot"] = eval_mode1_views(contrastive, ec_enc, edge_index, loader, ec2id, unseen_ec4_ids, args.device, is_v2=False)
    elif args.kind == "v2":
        from models.zymebind_v2 import load_v2_checkpoint
        m1_enc, m1_ec, m2, edge_index, _ = load_v2_checkpoint(Path(args.ckpt), args.device)
        out["supervised"] = eval_supervised_views(m2, loader, id2ec, unseen_ecs, args.device, kind="v2")
        out["mode1_zeroshot"] = eval_mode1_views(m1_enc, m1_ec, edge_index, loader, ec2id, unseen_ec4_ids, args.device, is_v2=True)

    txt = json.dumps(out, indent=2, ensure_ascii=False)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
