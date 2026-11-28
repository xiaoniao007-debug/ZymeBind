#!/usr/bin/env python
"""ZymeBind v4.5 训练与消融脚本（严格只看 val，绝不触碰 test / 外部基准）。

新增改进（相对 v4）：
  1. EMA (Exponential Moving Average) 权重平滑，提升长尾 AUPR-macro 与 F1@L4；
  2. Teacher 通路可选 domain-drop（随机屏蔽部分 domain multi-hot），缓解全模态下
     domain 捷径压制罕见 EC4（Rare 1-5）的现象；
  3. 完整消融开关（单变量控制）：--no-moddrop / --loss {asl,focal} /
     --kd-weight / --feat-kd-weight / --permod-weight / --no-prostt5。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import CachedFeatureDataset, EmbeddingCache, cached_collate  # noqa: E402
from data.dataset import build_domain_vocab, compute_pos_weight, load_split  # noqa: E402
from evaluation.eval_multiview_v2 import eval_supervised_views_fast  # noqa: E402
from models.classifier import FocalLoss  # noqa: E402
from models.zymebind_v2 import (  # noqa: E402
    MODALITIES,
    AsymmetricLoss,
    SemanticECOntologyEncoder,
    ZymeBindV2Stage1,
    ZymeBindV2Stage2,
    detect_presence,
)
from training_zymebind import ANC_SOFT, TIERS, make_edge_index  # noqa: E402
from training_zymebind_v2 import (  # noqa: E402
    focused_logit_kd,
    make_seen_mask,
    sample_structured_dropout,
    set_seed,
    tier_to_semantic_path,
    train_stage1,
)
from training_zymebind_v4_dualplm import DualPLMCachedDataset, ProstT5AACache  # noqa: E402


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k], alpha=1.0 - d)
            else:
                v.copy_(msd[k])


def train_stage2_v45(stage2: ZymeBindV2Stage2, train_dl: DataLoader, val_dl: DataLoader,
                     id2ec: dict[int, str], pos_weight: torch.Tensor | None, args, device: str):
    if args.loss == "asl":
        crit = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=0.0, clip=args.asl_clip)
    else:
        crit = FocalLoss(gamma=2.0, alpha=0.25)

    def calc_loss(logits, y, lw, sw):
        if args.loss == "asl":
            return crit(logits, y, lw, sw)
        # FocalLoss 兼容（按 sample_weight 加权）
        w = lw * sw.unsqueeze(1) if sw is not None else lw
        return crit(logits, y, weights=w, pos_weight=pos_weight) * float(logits.size(1)) * 0.01

    opt = torch.optim.AdamW(stage2.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    ema = ModelEMA(stage2, decay=args.ema_decay) if args.ema_decay > 0 else None

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
            raw = batch["feats"]
            feats = {
                "sequence": raw["sequence"].to(device, non_blocking=True),
                "structure": raw["threedi"].to(device, non_blocking=True),
                "text": raw["text"].to(device, non_blocking=True),
                "domain": raw["domain"].to(device, non_blocking=True),
            }
            labels = batch["labels"].to(device, non_blocking=True)
            lw = batch["label_weights"].to(device, non_blocking=True)
            sw = batch["sample_weights"].to(device, non_blocking=True)
            present = detect_presence(feats)

            # Teacher 通路可选随机 domain 位点 dropout（防稀有 EC 被高频同源 domain 掩盖）
            if args.domain_drop > 0.0:
                keep_d = (torch.rand_like(feats["domain"]) > args.domain_drop).float()
                t_feats = {**feats, "domain": feats["domain"] * keep_d}
            else:
                t_feats = feats

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                t_out = stage2.forward_detailed(t_feats, present_mask=present)
                l_teacher = calc_loss(t_out["logits"], labels, lw, sw) + 0.5 * calc_loss(t_out["fused_logits"], labels, lw, sw)

                loss = l_teacher

                if not args.no_moddrop:
                    s_feats, s_present = sample_structured_dropout(feats, present)
                    s_out = stage2.forward_detailed(s_feats, present_mask=s_present)
                    l_student = calc_loss(s_out["logits"], labels, lw, sw) + 0.5 * calc_loss(s_out["fused_logits"], labels, lw, sw)
                    loss = loss + l_student

                    if args.kd_weight > 0.0:
                        l_kd_logit = focused_logit_kd(s_out["logits"], t_out["logits"].detach(), labels, top_k=64, temp=2.0)
                        loss = loss + args.kd_weight * l_kd_logit
                    if args.feat_kd_weight > 0.0:
                        l_kd_feat = (1.0 - F.cosine_similarity(s_out["fused"], t_out["fused"].detach(), dim=-1)).mean()
                        loss = loss + args.feat_kd_weight * l_kd_feat

                if args.permod_weight > 0.0:
                    l_permod = torch.tensor(0.0, device=device)
                    n_pm = 0
                    for k in MODALITIES:
                        idx = present[k].nonzero(as_tuple=False).squeeze(-1)
                        if idx.numel() > 0:
                            l_permod = l_permod + calc_loss(t_out["permod_logits"][k][idx], labels[idx], lw[idx], sw[idx])
                            n_pm += 1
                    if n_pm > 0:
                        loss = loss + args.permod_weight * (l_permod / n_pm)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage2.parameters(), 1.0)
            opt.step()
            if ema is not None:
                ema.update(stage2)
            tot_loss += float(loss.item())
            n_steps += 1
        sched.step()

        row = {"epoch": ep, "loss": round(tot_loss / max(n_steps, 1), 4), "sec": round(time.time() - t0, 1)}
        if ep % args.eval_every == 0 or ep == args.epochs:
            eval_mod = ema.ema if ema is not None else stage2
            val_m = eval_supervised_views_fast(eval_mod, val_dl, id2ec, device)
            comp = 0.40 * val_m["full"]["fmax"] + 0.35 * val_m["seq_struct"]["fmax"] + 0.25 * val_m["seq_only"]["fmax"]
            row["val"] = val_m
            row["composite_fmax"] = round(comp, 4)
            print(
                f"[v4.5 ep {ep:02d}/{args.epochs}] loss={row['loss']:.4f} ({row['sec']}s) "
                f"Fmax full={val_m['full']['fmax']:.4f} ss={val_m['seq_struct']['fmax']:.4f} "
                f"seq={val_m['seq_only']['fmax']:.4f} comp={comp:.4f}",
                flush=True,
            )
            if comp > best_score:
                best_score = comp
                best_state = copy.deepcopy(eval_mod.state_dict())
                best_ep = ep
                best_val = val_m
                no_improve = 0
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    print(f"[v4.5] early stop at ep {ep} (best ep {best_ep}, comp={best_score:.4f})", flush=True)
                    history.append(row)
                    break
        else:
            print(f"[v4.5 ep {ep:02d}/{args.epochs}] loss={row['loss']:.4f} ({row['sec']}s)", flush=True)
        history.append(row)

    return {"state_dict": best_state, "best_epoch": best_ep, "best_composite_fmax": round(best_score, 4), "best_val": best_val}, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="p1m2_benchclusterfree")
    ap.add_argument("--no-prostt5", action="store_true")
    ap.add_argument("--skip-stage1", action="store_true", help="消融实验只测 Stage2 时跳过 Stage1 省时间")
    ap.add_argument("--pretrain-epochs", type=int, default=4)
    ap.add_argument("--pretrain-eval-every", type=int, default=1)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--fusion-dim", type=int, default=768)
    ap.add_argument("--hidden-dim", type=int, default=1536)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--gcn-layers", type=int, default=0)
    ap.add_argument("--moe-experts", type=int, default=3)
    ap.add_argument("--diversity-weight", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss", choices=("asl", "focal"), default="asl")
    ap.add_argument("--gamma-neg", type=float, default=4.0)
    ap.add_argument("--asl-clip", type=float, default=0.05)
    ap.add_argument("--permod-weight", type=float, default=0.5)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--feat-kd-weight", type=float, default=0.5)
    ap.add_argument("--no-moddrop", action="store_true")
    ap.add_argument("--domain-drop", type=float, default=0.25)
    ap.add_argument("--ema-decay", type=float, default=0.995)
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

    esm_c = EmbeddingCache("sequence")
    txt_c = EmbeddingCache("text")
    tdi_c = EmbeddingCache("threedi_observed_afdbv6")

    if args.no_prostt5:
        seq_dim = esm_c.dim
        tr_ds_s1 = CachedFeatureDataset(train_rows, ec2id, domain2id, esm_c, txt_c, anc_soft=ANC_SOFT, threedi_cache=tdi_c, evidence_weight=ev_w)
        tr_ds_s2 = CachedFeatureDataset(train_rows, ec2id, domain2id, esm_c, txt_c, anc_soft=0.0, threedi_cache=tdi_c, evidence_weight=ev_w)
        va_ds = CachedFeatureDataset(val_rows, ec2id, domain2id, esm_c, txt_c, anc_soft=0.0, threedi_cache=tdi_c)
    else:
        prost_c = ProstT5AACache()
        seq_dim = esm_c.dim + prost_c.dim
        tr_ds_s1 = DualPLMCachedDataset(train_rows, ec2id, domain2id, esm_c, prost_c, txt_c, tdi_c, anc_soft=ANC_SOFT, evidence_weight=ev_w)
        tr_ds_s2 = DualPLMCachedDataset(train_rows, ec2id, domain2id, esm_c, prost_c, txt_c, tdi_c, anc_soft=0.0, evidence_weight=ev_w)
        va_ds = DualPLMCachedDataset(val_rows, ec2id, domain2id, esm_c, prost_c, txt_c, tdi_c, anc_soft=0.0)

    tr_dl_s1 = DataLoader(tr_ds_s1, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=cached_collate)
    tr_dl_s2 = DataLoader(tr_ds_s2, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=cached_collate)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=cached_collate)

    in_dims = {"sequence": seq_dim, "structure": tdi_c.dim, "text": txt_c.dim, "domain": len(domain2id)}
    edge_index = make_edge_index(vpath, ec2id, args.device)
    sem_emb = torch.load(tier_to_semantic_path(args.tier), map_location="cpu", weights_only=False)["embeddings"]

    stage1 = ZymeBindV2Stage1(
        in_dims, proj_dim=args.proj_dim, n_experts=args.moe_experts,
        temperature=args.temperature, dropout=args.dropout, diversity_weight=args.diversity_weight,
    ).to(args.device)
    ec_encoder = SemanticECOntologyEncoder(
        str(vpath), sem_emb, dim=args.proj_dim, layers=args.gcn_layers, dropout=args.dropout,
    ).to(args.device)

    if args.skip_stage1:
        s1_best = {"stage1": stage1.state_dict(), "ec_encoder": ec_encoder.state_dict(), "best_epoch": 0, "best_composite_r10": 0.0}
        s1_hist = []
    else:
        seen_mask = make_seen_mask(train_rows, ec2id, args.device)
        ec4_mask = torch.tensor([id2ec[i].count(".") == 3 for i in range(len(ec2id))], dtype=torch.bool, device=args.device)
        seen_ec = {e for r in train_rows for e in r["ec_numbers"] if e in ec2id}
        val_unseen = {e for r in val_rows for e in r["ec_numbers"] if e in ec2id and e not in seen_ec and e.count(".") == 3}
        val_unseen_ids = {ec2id[e] for e in val_unseen}
        s1_best, s1_hist = train_stage1(
            stage1, ec_encoder, edge_index, tr_dl_s1, va_dl,
            ec2id, seen_mask, ec4_mask, val_unseen_ids, args, args.device,
        )

    pos_weight = compute_pos_weight(train_rows, ec2id).to(args.device) if args.loss == "focal" else None
    stage2 = ZymeBindV2Stage2(
        in_dims, len(ec2id), fusion_dim=args.fusion_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout, hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
    ).to(args.device)
    s2_best, s2_hist = train_stage2_v45(stage2, tr_dl_s2, va_dl, id2ec, pos_weight, args, args.device)

    v2_cfg = {
        "proj_dim": args.proj_dim, "fusion_dim": args.fusion_dim, "hidden_dim": args.hidden_dim,
        "n_heads": args.n_heads, "n_layers": args.n_layers, "gcn_layers": args.gcn_layers,
        "moe_experts": args.moe_experts, "diversity_weight": args.diversity_weight,
        "temperature": args.temperature, "dropout": args.dropout, "use_ec_delta": True,
        "dual_plm": "esm2_650m" if args.no_prostt5 else "esm2_650m+prostt5_3b_aa",
        "ema_decay": args.ema_decay, "domain_drop": args.domain_drop, "loss": args.loss,
    }
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": "zymebind_v45", "tier": args.tier, "vocab": vocab_rel,
        "in_dims": in_dims, "v2_cfg": v2_cfg,
        "mode1": s1_best, "model": s2_best["state_dict"],
        "stage2_best_epoch": s2_best["best_epoch"], "args": vars(args),
    }, args.ckpt)
    out = {
        "arch": "zymebind_v45", "tier": args.tier, "ckpt": args.ckpt,
        "in_dims": in_dims, "v2_cfg": v2_cfg, "args": vars(args),
        "stage1_best_epoch": s1_best["best_epoch"], "stage1_best_composite_r10": s1_best["best_composite_r10"],
        "stage2_best_epoch": s2_best["best_epoch"], "stage2_best_composite_fmax": s2_best["best_composite_fmax"],
        "stage2_best_val": s2_best["best_val"],
        "stage1_history": s1_hist, "stage2_history": s2_hist,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"[DONE v4.5 val-only] saved {args.ckpt} & {args.out}", flush=True)


if __name__ == "__main__":
    main()
