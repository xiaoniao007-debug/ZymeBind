#!/usr/bin/env python
"""ZymeBind v2 两阶段训练脚本（缺失模态鲁棒 + 语义本体先验 + EC-SupCon + 跨模态蒸馏）。

严守口径纪律：
- 只读既有 splits / vocab / 嵌入缓存，零修改任何 pinned 文件；
- 内部验证同时监控 Full / Seq+3Di / Seq-only 三视角 Fmax 与 Mode 1 零样本检索；
- 严禁在训练期触碰 NEW-392 / Price-149 外部基准。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import CachedFeatureDataset, EmbeddingCache, cached_collate  # noqa: E402
from data.dataset import build_domain_vocab, load_split, partial_node  # noqa: E402
from evaluation.eval_multiview_v2 import eval_mode1_views, eval_supervised_views_fast  # noqa: E402
from models.zymebind_v2 import (  # noqa: E402
    MODALITIES,
    AsymmetricLoss,
    SemanticECOntologyEncoder,
    ZymeBindV2Stage1,
    ZymeBindV2Stage2,
    detect_presence,
)
from training_zymebind import (  # noqa: E402
    ANC_SOFT,
    TIERS,
    ec_alignment_loss,
    make_edge_index,
    ordered_feats,
)


def make_seen_mask(train_rows: list[dict], ec2id: dict[str, int], device: str) -> torch.Tensor:
    nodes = set()
    for r in train_rows:
        for ec in r["ec_numbers"]:
            parts = ec.split(".")
            for i in range(1, len(parts) + 1):
                anc = ".".join(parts[:i])
                if anc in ec2id:
                    nodes.add(anc)
        for p in (r.get("ec_partial") or []):
            n = partial_node(ec2id, p)
            if n and n in ec2id:
                parts = n.split(".")
                for i in range(1, len(parts) + 1):
                    anc = ".".join(parts[:i])
                    if anc in ec2id:
                        nodes.add(anc)
    mask = torch.zeros(len(ec2id), dtype=torch.bool, device=device)
    for ec in nodes:
        mask[ec2id[ec]] = True
    return mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tier_to_semantic_path(tier: str) -> Path:
    if tier.startswith("m0"):
        key = "m0"
    elif tier.startswith("p1m1"):
        key = "p1m1"
    elif tier.startswith("p1m2"):
        key = "p1m2"
    else:
        raise ValueError(f"Unknown tier: {tier}")
    return ROOT / f"data/processed/ec_semantic/{key}_ec_biobert.pt"


def masked_info_nce(anchor: torch.Tensor, other: torch.Tensor,
                    valid: torch.Tensor, tau: float) -> torch.Tensor:
    idx = valid.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() < 4:
        return torch.tensor(0.0, device=anchor.device)
    a = anchor[idx]
    b = other[idx]
    logits = (a @ b.t()) / tau
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def ec_supcon_loss(z: torch.Tensor, labels: torch.Tensor,
                   ec4_mask: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
    """EC 监督对比学习：同 batch 内共享至少一个 EC4 的蛋白互为正例。"""
    y4 = (labels[:, ec4_mask] >= 0.99).float()  # (B, C4)
    pos_mat = (y4 @ y4.t()) > 0                 # (B, B)
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mat = pos_mat & (~eye)
    has_pos = pos_mat.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=z.device)
    sim = (z @ z.t()) / tau
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = F.log_softmax(sim, dim=-1)
    pos_f = pos_mat.float()
    loss_per = -(log_prob.masked_fill(~pos_mat, 0.0) * pos_f).sum(dim=-1) / pos_f.sum(dim=-1).clamp(min=1.0)
    return loss_per[has_pos].mean()


def sample_structured_dropout(feats: dict[str, torch.Tensor],
                              natural_present: dict[str, torch.Tensor]
                              ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """结构化模态 Dropout：覆盖纯序列、序列+结构、部分元数据与随机丢弃。"""
    B = next(iter(feats.values())).size(0)
    device = next(iter(feats.values())).device
    r = torch.rand(B, device=device)

    # 35% seq_only, 35% seq+struct, 10% seq+domain, 10% seq+text, 10% seq+struct+domain
    keep = {
        "sequence": natural_present["sequence"].clone(),
        "structure": natural_present["structure"] & ((r >= 0.35) & (r < 0.70) | (r >= 0.90)),
        "domain": natural_present["domain"] & (((r >= 0.70) & (r < 0.80)) | (r >= 0.90)),
        "text": natural_present["text"] & ((r >= 0.80) & (r < 0.90)),
    }
    dropped_feats = {
        k: feats[k] * keep[k].float().unsqueeze(-1) for k in MODALITIES
    }
    return dropped_feats, keep


def focused_logit_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     targets: torch.Tensor, top_k: int = 64, temp: float = 2.0) -> torch.Tensor:
    """聚焦蒸馏：只在真实正类 ∪ Teacher Top-K 候选类上对齐软分布，避免被 6100+ 零类稀释。"""
    with torch.no_grad():
        _, top_idx = teacher_logits.topk(min(top_k, teacher_logits.size(1)), dim=1)
        focus = torch.zeros_like(teacher_logits, dtype=torch.bool)
        focus.scatter_(1, top_idx, True)
        focus |= (targets > 0.0)
        t_prob = torch.sigmoid(teacher_logits / temp)
    bce = F.binary_cross_entropy_with_logits(student_logits / temp, t_prob, reduction="none")
    return (bce * focus.float()).sum(dim=1).mean() * (temp * temp) / float(top_k)


def train_stage1(stage1: ZymeBindV2Stage1, ec_encoder: SemanticECOntologyEncoder,
                 edge_index: torch.Tensor, train_dl: DataLoader, val_dl: DataLoader,
                 ec2id: dict[str, int], seen_mask: torch.Tensor, ec4_mask: torch.Tensor,
                 unseen_ec4_ids: set[int], args, device: str) -> tuple[dict, list[dict]]:
    opt = torch.optim.AdamW(
        list(stage1.parameters()) + list(ec_encoder.parameters()),
        lr=args.pretrain_lr, weight_decay=1e-2,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.pretrain_epochs, 1))
    history = []
    best_score = -1.0
    best_snap = None

    for ep in range(1, args.pretrain_epochs + 1):
        t0 = time.time()
        stage1.train()
        ec_encoder.train()
        tot_loss, n_steps = 0.0, 0
        for batch in train_dl:
            feats = ordered_feats(batch, device)
            labels = batch["labels"].to(device)
            present = detect_presence(feats)
            tau = stage1.log_tau.exp().clamp(0.05, 0.5)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                embs, div_loss = stage1.project_all(feats)
                ec_emb = F.normalize(ec_encoder(edge_index), dim=-1)

                # 1. 跨模态 InfoNCE（以序列为锚，仅在非零样本上计算）
                l_cm = (
                    masked_info_nce(embs["sequence"], embs["structure"], present["structure"], tau)
                    + masked_info_nce(embs["sequence"], embs["text"], present["text"], tau)
                    + masked_info_nce(embs["sequence"], embs["domain"], present["domain"], tau)
                ) / 3.0

                # 2. 多视角酶查询向量
                z_full = stage1.encode_query(feats, present)
                mask_ss = {
                    "sequence": present["sequence"],
                    "structure": present["structure"],
                    "text": torch.zeros_like(present["text"]),
                    "domain": torch.zeros_like(present["domain"]),
                }
                z_ss = stage1.encode_query(feats, mask_ss)
                z_seq = embs["sequence"]

                # 3. 多视角 EC 本体对齐（严格 seen_mask 保护未见类）
                l_ec = (
                    ec_alignment_loss(z_full, ec_emb, labels, tau=float(tau.item()), seen_mask=seen_mask)
                    + ec_alignment_loss(z_ss, ec_emb, labels, tau=float(tau.item()), seen_mask=seen_mask)
                    + ec_alignment_loss(z_seq, ec_emb, labels, tau=float(tau.item()), seen_mask=seen_mask)
                ) / 3.0

                # 4. EC-SupCon（在 seq+struct 与 full 空间拉近同 EC4 蛋白）
                l_supcon = 0.5 * (
                    ec_supcon_loss(z_ss, labels, ec4_mask, tau=0.15)
                    + ec_supcon_loss(z_full, labels, ec4_mask, tau=0.15)
                )

                loss = l_ec + 0.5 * l_cm + 0.3 * l_supcon + stage1.diversity_weight * div_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(stage1.parameters()) + list(ec_encoder.parameters()), 1.0
            )
            opt.step()
            tot_loss += float(loss.item())
            n_steps += 1
        sched.step()

        row = {"epoch": ep, "loss": round(tot_loss / max(n_steps, 1), 4), "sec": round(time.time() - t0, 1)}
        if ep % args.pretrain_eval_every == 0 or ep == args.pretrain_epochs:
            m1_val = eval_mode1_views(stage1, ec_encoder, edge_index, val_dl, ec2id, unseen_ec4_ids, device, is_v2=True)
            score = (
                0.4 * m1_val["full"]["recall@10"]
                + 0.35 * m1_val["seq_struct"]["recall@10"]
                + 0.25 * m1_val["seq_only"]["recall@10"]
            )
            row["val_mode1"] = m1_val
            row["composite_r10"] = round(score, 4)
            print(
                f"[Stage1 ep {ep:02d}/{args.pretrain_epochs}] loss={row['loss']:.4f} ({row['sec']}s) "
                f"R@10 full={m1_val['full']['recall@10']:.4f}(med={m1_val['full']['median_rank']}) "
                f"ss={m1_val['seq_struct']['recall@10']:.4f}(med={m1_val['seq_struct']['median_rank']}) "
                f"seq={m1_val['seq_only']['recall@10']:.4f}(med={m1_val['seq_only']['median_rank']})",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_snap = {
                    "stage1": copy.deepcopy(stage1.state_dict()),
                    "ec_encoder": copy.deepcopy(ec_encoder.state_dict()),
                    "best_epoch": ep,
                    "best_composite_r10": round(best_score, 4),
                }
        else:
            print(f"[Stage1 ep {ep:02d}/{args.pretrain_epochs}] loss={row['loss']:.4f} ({row['sec']}s)", flush=True)
        history.append(row)

    if best_snap is None:
        best_snap = {
            "stage1": copy.deepcopy(stage1.state_dict()),
            "ec_encoder": copy.deepcopy(ec_encoder.state_dict()),
            "best_epoch": args.pretrain_epochs,
            "best_composite_r10": 0.0,
        }
    return best_snap, history


def train_stage2(stage2: ZymeBindV2Stage2, train_dl: DataLoader, val_dl: DataLoader,
                 id2ec: dict[int, str], unseen_ecs: set[str], args, device: str) -> tuple[dict, list[dict]]:
    asl = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=0.0, clip=args.asl_clip)
    opt = torch.optim.AdamW(stage2.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    history = []
    best_score = -1.0
    best_state = None
    best_ep = 0
    best_val = None
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        stage2.train()
        tot_loss, n_steps = 0.0, 0

        for batch in train_dl:
            feats = ordered_feats(batch, device)
            labels = batch["labels"].to(device)
            lw = batch["label_weights"].to(device)
            sw = batch["sample_weights"].to(device)
            present = detect_presence(feats)

            s_feats, s_present = sample_structured_dropout(feats, present)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t_out = stage2.forward_detailed(feats, present_mask=present)
                s_out = stage2.forward_detailed(s_feats, present_mask=s_present)

                # 1. Teacher（全模态）监督损失
                l_teacher = asl(t_out["logits"], labels, lw, sw) + 0.5 * asl(t_out["fused_logits"], labels, lw, sw)

                # 2. Student（结构化缺失模态）监督损失
                l_student = asl(s_out["logits"], labels, lw, sw) + 0.5 * asl(s_out["fused_logits"], labels, lw, sw)

                # 3. 四模态独立监督损失（仅在各模态存在的样本上计算，彻底打破 text/domain 梯度垄断）
                l_permod = torch.tensor(0.0, device=device)
                n_pm = 0
                for k in MODALITIES:
                    idx = present[k].nonzero(as_tuple=False).squeeze(-1)
                    if idx.numel() > 0:
                        l_permod = l_permod + asl(
                            t_out["permod_logits"][k][idx], labels[idx], lw[idx], sw[idx]
                        )
                        n_pm += 1
                if n_pm > 0:
                    l_permod = l_permod / n_pm

                # 4. 跨模态知识蒸馏（Teacher -> Student: 聚焦 Logit KD + 融合表征余弦对齐）
                l_kd_logit = focused_logit_kd(s_out["logits"], t_out["logits"].detach(), labels, top_k=64, temp=2.0)
                l_kd_feat = (1.0 - F.cosine_similarity(s_out["fused"], t_out["fused"].detach(), dim=-1)).mean()

                loss = (
                    l_teacher
                    + l_student
                    + args.permod_weight * l_permod
                    + args.kd_weight * l_kd_logit
                    + args.feat_kd_weight * l_kd_feat
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage2.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.item())
            n_steps += 1
        sched.step()

        row = {"epoch": ep, "loss": round(tot_loss / max(n_steps, 1), 4), "sec": round(time.time() - t0, 1)}
        if ep % args.eval_every == 0 or ep == args.epochs:
            val_m = eval_supervised_views_fast(stage2, val_dl, id2ec, device)
            comp = (
                0.40 * val_m["full"]["fmax"]
                + 0.35 * val_m["seq_struct"]["fmax"]
                + 0.25 * val_m["seq_only"]["fmax"]
            )
            row["val"] = val_m
            row["composite_fmax"] = round(comp, 4)
            print(
                f"[Stage2 ep {ep:02d}/{args.epochs}] loss={row['loss']:.4f} ({row['sec']}s) "
                f"Fmax full={val_m['full']['fmax']:.4f} ss={val_m['seq_struct']['fmax']:.4f} "
                f"seq={val_m['seq_only']['fmax']:.4f} comp={comp:.4f}",
                flush=True,
            )
            if comp > best_score:
                best_score = comp
                best_state = copy.deepcopy(stage2.state_dict())
                best_ep = ep
                best_val = val_m
                no_improve = 0
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    print(f"[Stage2] early stop at ep {ep} (best ep {best_ep}, comp={best_score:.4f})", flush=True)
                    history.append(row)
                    break
        else:
            print(f"[Stage2 ep {ep:02d}/{args.epochs}] loss={row['loss']:.4f} ({row['sec']}s)", flush=True)
        history.append(row)

    if best_state is None:
        best_state = copy.deepcopy(stage2.state_dict())
        best_ep = args.epochs
    return {"state_dict": best_state, "best_epoch": best_ep, "best_composite_fmax": round(best_score, 4), "best_val": best_val}, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), default="p1m1_benchclusterfree")
    ap.add_argument("--threedi", default="threedi_observed_afdbv6")
    ap.add_argument("--pretrain-epochs", type=int, default=12)
    ap.add_argument("--pretrain-eval-every", type=int, default=3)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--fusion-dim", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=1024)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--gcn-layers", type=int, default=2)
    ap.add_argument("--moe-experts", type=int, default=3)
    ap.add_argument("--diversity-weight", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--gamma-neg", type=float, default=4.0)
    ap.add_argument("--asl-clip", type=float, default=0.05)
    ap.add_argument("--permod-weight", type=float, default=0.5)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--feat-kd-weight", type=float, default=0.5)
    ap.add_argument("--no-ec-delta", action="store_true",
                    help="禁用 EC 节点自由 delta 参数，防止已见 EC 漂离 BioBERT 语义流形")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
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
    domain2id = build_domain_vocab(train_rows)

    seq_c = EmbeddingCache("sequence")
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache(args.threedi)

    # Stage 1 使用 anc_soft=0.3（层级祖先软正例传播）
    tr_ds_s1 = CachedFeatureDataset(train_rows, ec2id, domain2id, seq_c, txt_c,
                                    anc_soft=ANC_SOFT, threedi_cache=tdi_c, evidence_weight=ev_w)
    # Stage 2 使用硬标签（与现有口径一致）
    tr_ds_s2 = CachedFeatureDataset(train_rows, ec2id, domain2id, seq_c, txt_c,
                                    anc_soft=0.0, threedi_cache=tdi_c, evidence_weight=ev_w)
    va_ds = CachedFeatureDataset(val_rows, ec2id, domain2id, seq_c, txt_c,
                                 anc_soft=0.0, threedi_cache=tdi_c)

    tr_dl_s1 = DataLoader(tr_ds_s1, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True, collate_fn=cached_collate)
    tr_dl_s2 = DataLoader(tr_ds_s2, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True, collate_fn=cached_collate)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True, collate_fn=cached_collate)

    in_dims = {
        "sequence": seq_c.dim,
        "structure": tdi_c.dim,
        "text": txt_c.dim,
        "domain": len(domain2id),
    }
    edge_index = make_edge_index(vpath, ec2id, args.device)
    seen_mask = make_seen_mask(train_rows, ec2id, args.device)
    ec4_mask = torch.tensor([ec.count(".") == 3 for ec in [id2ec[i] for i in range(len(ec2id))]],
                            dtype=torch.bool, device=args.device)

    seen_ec = {e for r in train_rows for e in r["ec_numbers"] if e in ec2id}
    unseen_ecs = {e for r in val_rows for e in r["ec_numbers"] if e in ec2id and e not in seen_ec and e.count(".") == 3}
    unseen_ec4_ids = {ec2id[e] for e in unseen_ecs}

    sem_data = torch.load(tier_to_semantic_path(args.tier), map_location="cpu", weights_only=False)
    sem_emb = sem_data["embeddings"]

    print(f"[ZymeBind v2] tier={args.tier} train={len(train_rows)} val={len(val_rows)} "
          f"n_classes={len(ec2id)} in_dims={in_dims} unseen_val_ec4={len(unseen_ecs)}", flush=True)

    stage1 = ZymeBindV2Stage1(
        in_dims, proj_dim=args.proj_dim, n_experts=args.moe_experts,
        temperature=args.temperature, dropout=args.dropout,
        diversity_weight=args.diversity_weight,
    ).to(args.device)
    ec_encoder = SemanticECOntologyEncoder(
        str(vpath), sem_emb, dim=args.proj_dim, layers=args.gcn_layers,
        dropout=args.dropout, use_delta=not args.no_ec_delta,
    ).to(args.device)

    s1_best, s1_hist = train_stage1(
        stage1, ec_encoder, edge_index, tr_dl_s1, va_dl,
        ec2id, seen_mask, ec4_mask, unseen_ec4_ids, args, args.device,
    )

    stage2 = ZymeBindV2Stage2(
        in_dims, len(ec2id), fusion_dim=args.fusion_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout, hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
    ).to(args.device)

    s2_best, s2_hist = train_stage2(
        stage2, tr_dl_s2, va_dl, id2ec, unseen_ecs, args, args.device,
    )

    v2_cfg = {
        "proj_dim": args.proj_dim,
        "fusion_dim": args.fusion_dim,
        "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "gcn_layers": args.gcn_layers,
        "moe_experts": args.moe_experts,
        "diversity_weight": args.diversity_weight,
        "temperature": args.temperature,
        "dropout": args.dropout,
        "use_ec_delta": not args.no_ec_delta,
    }
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "zymebind_v2",
        "tier": args.tier,
        "vocab": vocab_rel,
        "in_dims": in_dims,
        "v2_cfg": v2_cfg,
        "mode1": {
            "stage1": s1_best["stage1"],
            "ec_encoder": s1_best["ec_encoder"],
            "best_epoch": s1_best["best_epoch"],
            "best_composite_r10": s1_best["best_composite_r10"],
        },
        "model": s2_best["state_dict"],
        "stage2_best_epoch": s2_best["best_epoch"],
        "stage2_best_composite_fmax": s2_best["best_composite_fmax"],
        "args": vars(args),
    }, args.ckpt)

    summary = {
        "arch": "zymebind_v2",
        "tier": args.tier,
        "ckpt": args.ckpt,
        "in_dims": in_dims,
        "v2_cfg": v2_cfg,
        "args": vars(args),
        "stage1_best_epoch": s1_best["best_epoch"],
        "stage1_best_composite_r10": s1_best["best_composite_r10"],
        "stage2_best_epoch": s2_best["best_epoch"],
        "stage2_best_composite_fmax": s2_best["best_composite_fmax"],
        "stage2_best_val": s2_best["best_val"],
        "stage1_history": s1_hist,
        "stage2_history": s2_hist,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[DONE] saved ckpt={args.ckpt} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
