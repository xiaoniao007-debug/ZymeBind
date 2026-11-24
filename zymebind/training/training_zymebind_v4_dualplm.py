#!/usr/bin/env python
"""ZymeBind v4 (Dual-PLM): ESM-2 650M (1280d) + ProstT5-3B AA (1024d) = 2304d 联合序列表示。

复用 ZymeBind v2 的完整架构（SemanticECOntologyEncoder + PresenceAwareModalityFusion +
独立 PerMod 监督头 + 缺失感知共识 + 跨模态聚焦蒸馏 + ASL），训练后自动完成：
  1. 内部 test (full / seq_struct / seq_only + Mode 1 零样本) full_scores 评测；
  2. 外部 NEW-392 (6 语境) 与 Price-149 (总分 + <40% 低同源分层) full_scores 评测。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import EmbeddingCache, cached_collate  # noqa: E402
from data.dataset import build_domain_vocab, evidence_sample_weight, label_and_weight, load_split  # noqa: E402
from evaluation.cafa_eval import evaluate_all  # noqa: E402
from evaluation.eval_external_benchmarks import (  # noqa: E402
    build_features,
    domain_multihot,
    load_benchmarks,
    load_main_tables,
)
from evaluation.eval_external_v2 import FROZEN_TABLE, load_obs3di_matrix, load_table_dict, predict_v2  # noqa: E402
from evaluation.eval_multiview_v2 import eval_mode1_views, eval_supervised_views  # noqa: E402
from models.zymebind_v2 import SemanticECOntologyEncoder, ZymeBindV2Stage1, ZymeBindV2Stage2  # noqa: E402
from training_zymebind import ANC_SOFT, TIERS, make_edge_index  # noqa: E402
from training_zymebind_v2 import (  # noqa: E402
    make_seen_mask,
    set_seed,
    tier_to_semantic_path,
    train_stage1,
    train_stage2,
)


class ProstT5AACache:
    """加载 data/processed/prostt5_aa/shard{0,1}.npy + ids_{0,1}.txt。"""

    def __init__(self, root: Path = ROOT / "data/processed/prostt5_aa"):
        self.index: dict[str, tuple[int, int]] = {}
        self.shards: dict[int, np.ndarray] = {}
        for sh_path in sorted(root.glob("shard*.npy")):
            sh = int(sh_path.stem.replace("shard", ""))
            self.shards[sh] = np.load(sh_path, mmap_mode="r")
            ids_path = root / f"ids_{sh}.txt"
            for row, acc in enumerate(ids_path.read_text().splitlines()):
                if acc:
                    self.index[acc] = (sh, row)
        self.dim = int(next(iter(self.shards.values())).shape[1])
        print(f"[cache] prostt5_aa: {len(self.index)} 条, dim={self.dim}, {len(self.shards)} shards", flush=True)

    def get(self, acc: str) -> torch.Tensor:
        if acc not in self.index:
            return torch.zeros(self.dim, dtype=torch.float32)
        sh, row = self.index[acc]
        return torch.from_numpy(np.asarray(self.shards[sh][row], dtype=np.float32))


class DualPLMCachedDataset(Dataset):
    """sequence = concat(ESM-2 1280d, ProstT5-3B AA 1024d) -> 2304d；其余 3 模态不变。"""

    def __init__(self, rows: list[dict], ec2id: dict[str, int], domain2id: dict[str, int],
                 esm_cache: EmbeddingCache, prost_cache: ProstT5AACache,
                 txt_cache: EmbeddingCache, tdi_cache: EmbeddingCache,
                 anc_soft: float = 0.0, evidence_weight: float = 0.0):
        self.rows = rows
        self.ec2id = ec2id
        self.domain2id = domain2id
        self.n_domains = len(domain2id)
        self.esm = esm_cache
        self.prost = prost_cache
        self.txt = txt_cache
        self.tdi = tdi_cache
        self.anc_soft = anc_soft
        self.ev_w = evidence_weight

    def __len__(self) -> int:
        return len(self.rows)

    def _domain_multihot(self, r: dict) -> torch.Tensor:
        v = torch.zeros(self.n_domains, dtype=torch.float32)
        for db, items in (r.get("domains") or {}).items():
            for it in items:
                j = self.domain2id.get(f"{db}:{it}")
                if j is not None:
                    v[j] = 1.0
        return v

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        acc = r["accession"]
        seq_dual = torch.cat([self.esm.get(acc), self.prost.get(acc)], dim=0)
        feats = {
            "sequence": seq_dual,
            "text": self.txt.get(acc),
            "threedi": self.tdi.get(acc),
            "domain": self._domain_multihot(r),
        }
        y, w, primary = label_and_weight(
            self.ec2id, r["ec_numbers"], r.get("ec_partial") or [], anc_soft=self.anc_soft
        )
        return {
            "accession": acc,
            "feats": feats,
            "labels": y,
            "label_weights": w,
            "sample_weight": evidence_sample_weight(r.get("protein_existence", ""), self.ev_w),
            "primary_ec": primary,
            "is_multi_ec": len(r["ec_numbers"]) > 1,
            "ec_numbers": r["ec_numbers"],
        }


def eval_prob_matrix(probs: np.ndarray, labs: list[set[str]], id2ec: dict[int, str],
                     unseen_ecs: set[str], min_score: float = 0.01) -> dict:
    scores, labels = [], []
    for i, lab in enumerate(labs):
        if not lab:
            continue
        p = probs[i]
        idx = np.where(p >= min_score)[0]
        scores.append({id2ec[int(j)]: float(p[j]) for j in idx})
        labels.append(lab)
    m = evaluate_all(scores, labels, unseen_ecs=unseen_ecs)
    return {
        "fmax": round(float(m["fmax"]), 4),
        "aupr_macro": round(float(m["aupr_macro"]), 4),
        "f1_macro_L4": round(float(m.get("f1_macro@L4", 0.0)), 4),
        "recall@1": round(float(m.get("recall@1", 0.0)), 4),
        "n_eval": int(m.get("n_samples", len(labels))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="p1m1_benchclusterfree")
    ap.add_argument("--pretrain-epochs", type=int, default=5)
    ap.add_argument("--pretrain-eval-every", type=int, default=1)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--fusion-dim", type=int, default=768)
    ap.add_argument("--hidden-dim", type=int, default=1536)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--gcn-layers", type=int, default=2)
    ap.add_argument("--moe-experts", type=int, default=3)
    ap.add_argument("--diversity-weight", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--gamma-neg", type=float, default=4.0)
    ap.add_argument("--asl-clip", type=float, default=0.05)
    ap.add_argument("--permod-weight", type=float, default=0.5)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--feat-kd-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/zymebind_v4_dualplm_p1m1_benchclusterfree.json")
    ap.add_argument("--ckpt", default="checkpoints/zymebind_v4_dualplm_p1m1_benchclusterfree.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    split_rel, vocab_rel, ev_w = TIERS[args.tier]
    sdir = ROOT / split_rel
    vpath = ROOT / vocab_rel
    vocab = json.loads(vpath.read_text())
    ec2id = vocab["ec2id"]
    id2ec = {int(v): k for k, v in ec2id.items()}

    train_rows = load_split(sdir / "train.tsv")
    val_rows = load_split(sdir / "val.tsv")
    test_rows = load_split(sdir / "test.tsv")
    domain2id = build_domain_vocab(train_rows)

    esm_c = EmbeddingCache("sequence")
    prost_c = ProstT5AACache()
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")

    tr_ds_s1 = DualPLMCachedDataset(train_rows, ec2id, domain2id, esm_c, prost_c, txt_c, tdi_c, anc_soft=ANC_SOFT, evidence_weight=ev_w)
    tr_ds_s2 = DualPLMCachedDataset(train_rows, ec2id, domain2id, esm_c, prost_c, txt_c, tdi_c, anc_soft=0.0, evidence_weight=ev_w)
    va_ds = DualPLMCachedDataset(val_rows, ec2id, domain2id, esm_c, prost_c, txt_c, tdi_c, anc_soft=0.0)
    te_ds = DualPLMCachedDataset(test_rows, ec2id, domain2id, esm_c, prost_c, txt_c, tdi_c, anc_soft=0.0)

    tr_dl_s1 = DataLoader(tr_ds_s1, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=cached_collate)
    tr_dl_s2 = DataLoader(tr_ds_s2, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=cached_collate)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=cached_collate)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=cached_collate)

    in_dims = {
        "sequence": esm_c.dim + prost_c.dim,  # 1280 + 1024 = 2304
        "structure": tdi_c.dim,
        "text": txt_c.dim,
        "domain": len(domain2id),
    }
    edge_index = make_edge_index(vpath, ec2id, args.device)
    seen_mask = make_seen_mask(train_rows, ec2id, args.device)
    ec4_mask = torch.tensor([id2ec[i].count(".") == 3 for i in range(len(ec2id))], dtype=torch.bool, device=args.device)

    seen_ec = {e for r in train_rows for e in r["ec_numbers"] if e in ec2id}
    val_unseen = {e for r in val_rows for e in r["ec_numbers"] if e in ec2id and e not in seen_ec and e.count(".") == 3}
    val_unseen_ids = {ec2id[e] for e in val_unseen}
    test_unseen = {e for r in test_rows for e in r["ec_numbers"] if e in ec2id and e not in seen_ec and e.count(".") == 3}
    test_unseen_ids = {ec2id[e] for e in test_unseen}

    sem_emb = torch.load(tier_to_semantic_path(args.tier), map_location="cpu", weights_only=False)["embeddings"]
    print(f"[ZymeBind v4 Dual-PLM] tier={args.tier} in_dims={in_dims}", flush=True)

    stage1 = ZymeBindV2Stage1(
        in_dims, proj_dim=args.proj_dim, n_experts=args.moe_experts,
        temperature=args.temperature, dropout=args.dropout, diversity_weight=args.diversity_weight,
    ).to(args.device)
    ec_encoder = SemanticECOntologyEncoder(
        str(vpath), sem_emb, dim=args.proj_dim, layers=args.gcn_layers, dropout=args.dropout,
    ).to(args.device)

    s1_best, s1_hist = train_stage1(
        stage1, ec_encoder, edge_index, tr_dl_s1, va_dl,
        ec2id, seen_mask, ec4_mask, val_unseen_ids, args, args.device,
    )

    stage2 = ZymeBindV2Stage2(
        in_dims, len(ec2id), fusion_dim=args.fusion_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout, hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
    ).to(args.device)
    s2_best, s2_hist = train_stage2(stage2, tr_dl_s2, va_dl, id2ec, val_unseen, args, args.device)

    stage1.load_state_dict(s1_best["stage1"])
    ec_encoder.load_state_dict(s1_best["ec_encoder"])
    stage2.load_state_dict(s2_best["state_dict"])

    v2_cfg = {
        "proj_dim": args.proj_dim, "fusion_dim": args.fusion_dim, "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads, "n_layers": args.n_layers, "gcn_layers": args.gcn_layers,
        "moe_experts": args.moe_experts, "diversity_weight": args.diversity_weight,
        "temperature": args.temperature, "dropout": args.dropout, "use_ec_delta": True,
        "dual_plm": "esm2_650m+prostt5_3b_aa",
    }
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "zymebind_v4_dualplm", "tier": args.tier, "vocab": vocab_rel,
        "in_dims": in_dims, "v2_cfg": v2_cfg,
        "mode1": s1_best, "model": s2_best["state_dict"],
        "stage2_best_epoch": s2_best["best_epoch"],
    }, args.ckpt)

    # 1. 内部 test 全口径评测
    test_sup = eval_supervised_views(stage2, te_dl, id2ec, test_unseen, args.device, kind="v2", min_score=0.01)
    test_m1 = eval_mode1_views(stage1, ec_encoder, edge_index, te_dl, ec2id, test_unseen_ids, args.device, is_v2=True)
    print(f"[v4 test] full={test_sup['full']['fmax']:.4f} ss={test_sup['seq_struct']['fmax']:.4f} seq={test_sup['seq_only']['fmax']:.4f}", flush=True)
    print(f"[v4 test Mode1] full R@10={test_m1['full']['recall@10']:.4f}(med={test_m1['full']['median_rank']}) "
          f"ss={test_m1['seq_struct']['recall@10']:.4f} seq={test_m1['seq_only']['recall@10']:.4f}", flush=True)

    # 2. 外部 NEW-392 & Price-149 评测
    benchmarks = dict(load_benchmarks())
    full_meta = load_main_tables()
    frozen_meta = load_table_dict(FROZEN_TABLE)
    obs3di_cache = EmbeddingCache("threedi_new392_observed")
    all_rows = [r for rows in benchmarks.values() for r in rows]
    feats_full = build_features(all_rows, full_meta, args.device)
    feats_frozen = build_features(all_rows, frozen_meta, args.device)

    hom_path = ROOT / "results/v2_price149_p1m1_homology.json"
    p149_hom = json.loads(hom_path.read_text())["max_pident"] if hom_path.exists() else {}

    ext_res = {}
    for bname, rows in benchmarks.items():
        esm_np = np.stack([feats_full["sequence"][r["accession"]] for r in rows])
        prost_np = np.stack([prost_c.get(r["accession"]).numpy() for r in rows])
        seq_np = np.concatenate([esm_np, prost_np], axis=1)
        zero_tri = np.zeros((len(rows), 1024), dtype=np.float32)
        obs_tri = load_obs3di_matrix(rows, obs3di_cache)
        zero_txt = np.zeros((len(rows), 768), dtype=np.float32)
        zero_dom = np.zeros((len(rows), len(domain2id)), dtype=np.float32)
        full_txt = np.stack([feats_full["text"][r["accession"]] for r in rows])
        full_dom = np.stack([domain_multihot(r["accession"], full_meta, domain2id) for r in rows])
        froz_txt = np.stack([feats_frozen["text"][r["accession"]] for r in rows])
        froz_dom = np.stack([domain_multihot(r["accession"], frozen_meta, domain2id) for r in rows])

        labs = [set(e.strip() for e in (r.get("ec") or "").replace(";", ",").split(",") if e.strip()) for r in rows]
        b_unseen = {e for s in labs for e in s if e in ec2id and e not in seen_ec and e.count(".") == 3}
        contexts = {
            "seq_only": (seq_np, zero_tri, zero_txt, zero_dom),
            "observed3di": (seq_np, obs_tri, zero_txt, zero_dom),
            "frozen_2022_04": (seq_np, zero_tri, froz_txt, froz_dom),
            "frozen_2022_04_obs3di": (seq_np, obs_tri, froz_txt, froz_dom),
            "full_context": (seq_np, zero_tri, full_txt, full_dom),
            "full_context_obs3di": (seq_np, obs_tri, full_txt, full_dom),
        }
        if bname == "price149":
            contexts = {"seq_only": contexts["seq_only"]}
        ext_res[bname] = {}
        for cname, (sq, tr, tx, dm) in contexts.items():
            probs = predict_v2(stage2, sq, tr, tx, dm, args.device)
            m = eval_prob_matrix(probs, labs, id2ec, b_unseen, min_score=0.01)
            if bname == "price149" and p149_hom:
                lt40 = [i for i, r in enumerate(rows) if p149_hom.get(r["accession"], 0.0) < 40.0]
                ge40 = [i for i, r in enumerate(rows) if p149_hom.get(r["accession"], 0.0) >= 40.0]
                m["lt40"] = eval_prob_matrix(probs[lt40], [labs[i] for i in lt40], id2ec, b_unseen, min_score=0.01)
                m["ge40"] = eval_prob_matrix(probs[ge40], [labs[i] for i in ge40], id2ec, b_unseen, min_score=0.01)
            ext_res[bname][cname] = m
            print(f"  [v4 ext {bname:8s} | {cname:22s}] Fmax={m['fmax']:.4f} AUPR={m['aupr_macro']:.4f} "
                  f"F1@L4={m['f1_macro_L4']:.4f} R@1={m['recall@1']:.4f}", flush=True)
            if bname == "price149" and "lt40" in m:
                print(f"    Price-149 <40%: Fmax={m['lt40']['fmax']:.4f} | >=40%: Fmax={m['ge40']['fmax']:.4f}", flush=True)

    out_obj = {
        "arch": "zymebind_v4_dualplm", "tier": args.tier, "ckpt": args.ckpt,
        "in_dims": in_dims, "v2_cfg": v2_cfg,
        "stage1_best_epoch": s1_best["best_epoch"], "stage2_best_epoch": s2_best["best_epoch"],
        "test_supervised": test_sup, "test_mode1": test_m1, "external": ext_res,
    }
    Path(args.out).write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n")
    print(f"[DONE] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
