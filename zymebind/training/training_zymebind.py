#!/usr/bin/env python
"""ZymeBind 计划架构主线：阶段一对比预训练 + 阶段二监督微调 + 零样本检索。

对应 docs/ZymeBind_研究计划与实施方案.md 3.1 / 3.3 / 3.4 / 3.5 / 5.1：

阶段一（自监督对比预训练）
  - 以序列模态为锚，用 InfoNCE 对齐 structure(3Di) / text / domain（models/contrastive.py）
  - 酶融合嵌入 ↔ EC 本体 GCN 嵌入做多正例对比（EC 层级 DAG，models/encoders/ec_ontology_gnn.py）
  - EC1-3 祖先闭包作软正例，把家族归属传播给训练未见 EC4 的父节点
  - EC 本体节点用确定性层级先验 + GCN 残差连接，避免所有类嵌入塌缩
  - 训练未见 EC 类从 InfoNCE 负例分母排除，避免被系统性推远
  - 结束后冻结一份快照（对比投影头 + EC 本体编码器）专供 Mode 1
阶段二（监督微调）
  - 四酶模态经 ModalityFusion（Transformer + [CLS]）融合
  - --head mlp : 计划 3.5 的 MLP 多标签分类头（Mode 2，判别性优先）
  - --head dot : 点积检索 + 概念瓶颈残差（早期实现，保留作消融对照）
  - Focal Loss + pos_weight + label_weights + evidence-aware 样本权重
Mode 1（零样本，计划 3.4）
  - 独立通路：阶段一冻结快照上做全词表余弦检索，不经过阶段二监督头
  - 见 evaluation/zeroshot_eval.py --path mode1；阶段二不得写回该快照

计划 3.1 要求两条模式并列。早期版本把阶段二微调后的 EC 本体嵌入同时用于
零样本检索，导致未见 EC 被压制；本版分离，并修正随机本体嵌入塌缩与
未见类被当作负例推远两个根因。`--head mlp` 进一步把 Mode 2 与检索头解耦。

技术路线沿用项目既定「冻结编码器 + 离线嵌入缓存」（data/cache_embeddings*.py），
故端到端编码器 LoRA 微调未启用；这是与计划 3.2 的唯一偏差，见文档第 15.1 节。

用法:
    python training_zymebind.py --tier p1m1 --head mlp \
        --pretrain-epochs 3 --epochs 6 \
        --out results/zymebind_p1m1_mlphead.json \
        --ckpt checkpoints/zymebind_p1m1_mlphead.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.cached_dataset import CachedFeatureDataset, EmbeddingCache, cached_collate  # noqa: E402
from data.dataset import (build_domain_vocab, compute_pos_weight, load_split,  # noqa: E402
                          partial_node)
from evaluation.cafa_eval import evaluate_all, main_table_row  # noqa: E402
from models.classifier import FocalLoss, ModalityFusion  # noqa: E402
from models.concept_bottleneck import ConceptBottleneck  # noqa: E402
from models.contrastive import ZymeBindContrastive  # noqa: E402
from models.encoders.ec_ontology_gnn import ECOntologyEncoder  # noqa: E402
from training_cached import load_cfg  # noqa: E402

# 计划术语：structure = foldseek 观测 3Di 的 ProstT5 嵌入
MODALITIES = ("sequence", "structure", "text", "domain")
TIERS = {
    "m0_benchfree": ("data/splits_m0_benchfree", "data/processed/ec_vocab.json", 0.0),
    "p1m1": ("data/splits_p1m1", "data/processed/p1m1/ec_vocab.json", 0.0),
    "p1m2": ("data/splits_p1m2", "data/processed/p1m2/ec_vocab.json", 1.0),
    # 40% 同簇重排除后的干净划分（18.17 评估 / 2026-11-08 批准）：只换 splits，
    # EC 词表沿用既有文件（保证 NEW-392 的 177 个评测标签全部可映射），
    # evidence_weight 与对应原档位一致，从而只改变「划分」这一个变量。
    "m0_benchclusterfree": (
        "data/splits_m0_benchclusterfree", "data/processed/ec_vocab.json", 0.0),
    "p1m1_benchclusterfree": (
        "data/splits_p1m1_benchclusterfree",
        "data/processed/p1m1/ec_vocab.json", 0.0),
    "p1m2_benchclusterfree": (
        "data/splits_p1m2_benchclusterfree",
        "data/processed/p1m2/ec_vocab.json", 1.0),
}
# 阶段一祖先闭包软标签强度：令酶嵌入同时对齐 EC1-3 层级节点，使训练未见
# EC4 可经共享父节点获得语义（data/dataset.py::label_and_weight）。
# 阶段二仍用硬标签，保证 Mode 2 指标口径与既有结果一致。
ANC_SOFT = 0.3


class CachedEncoder(nn.Module):
    """离线 pooled 嵌入直通（投影由 ZymeBindContrastive 的 ProjectionHead 完成）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ZymeBindModel(nn.Module):
    """阶段二：多模态融合 + 可切换分类头。

    head_type='mlp' 对应计划 3.5（融合特征 → MLP 多标签分类）；
    head_type='permod' 借鉴 FunBind：每个模态各一个分类头，logits 平均为 consensus；
    head_type='dot' 为早期点积检索 + 概念瓶颈残差，仅作消融对照。
    三种头都返回 (logits, z)，z 供零样本对照评测使用。
    """

    def __init__(self, in_dims: dict[str, int], ec_encoder: ECOntologyEncoder,
                 edge_index: torch.Tensor, n_classes: int, fusion_dim: int,
                 n_heads: int, dropout: float, proj_dim: int,
                 use_concept_bottleneck: bool, head_type: str = "dot"):
        super().__init__()
        self.head_type = head_type
        self.fusion = ModalityFusion(in_dims, fusion_dim, n_heads, dropout)
        self.z_proj = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim))
        self.logit_scale = nn.Parameter(torch.tensor(10.0).log())
        self.ec_encoder = ec_encoder
        self.register_buffer("edge_index", edge_index)
        self.use_cbm = use_concept_bottleneck and head_type == "dot"
        if head_type == "mlp":
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(fusion_dim), nn.Dropout(dropout),
                nn.Linear(fusion_dim, 1024), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(1024, n_classes))
        if head_type == "permod":
            # FunBind 式：每个非功能模态各挂一个分类头，推理时 logits 平均为 consensus
            self.permod_heads = nn.ModuleDict({
                k: nn.Sequential(
                    nn.LayerNorm(fusion_dim), nn.Dropout(dropout),
                    nn.Linear(fusion_dim, 1024), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(1024, n_classes))
                for k in in_dims})
        if self.use_cbm:
            self.cbm = ConceptBottleneck(fusion_dim, n_concepts=64,
                                         n_classes=n_classes, dropout=dropout)
            # 零初始化 MLP 分支：训练初期等价于纯本体检索，之后学习已见类修正
            nn.init.zeros_(self.cbm.ec_head[-1].weight)
            nn.init.zeros_(self.cbm.ec_head[-1].bias)

    def forward(self, feats: dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion(feats)
        z = F.normalize(self.z_proj(fused), dim=-1)
        if self.head_type == "mlp":
            return self.mlp_head(fused), z
        if self.head_type == "permod":
            tokens = {k: self.fusion.proj[k](v) for k, v in feats.items()}
            logits = torch.stack([self.permod_heads[k](t)
                                  for k, t in tokens.items()], dim=0).mean(0)
            return logits, z
        ec = F.normalize(self.ec_encoder(self.edge_index), dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 30.0)
        logits = scale * (z @ ec.t())
        if self.use_cbm:
            cbm_logits, _, _ = self.cbm(fused)
            logits = logits + cbm_logits
        return logits, z


def ordered_feats(batch: dict, device: str) -> dict[str, torch.Tensor]:
    """把 cached_collate 的 threedi 键映射为计划术语 structure，并固定模态顺序。"""
    raw = batch["feats"]
    src = {
        "sequence": raw["sequence"],
        "structure": raw["threedi"],
        "text": raw["text"],
        "domain": raw["domain"],
    }
    return {k: src[k].to(device) for k in MODALITIES}


def ec_alignment_loss(z: torch.Tensor, ec_emb: torch.Tensor,
                      labels: torch.Tensor, tau: float = 0.2,
                      seen_mask: torch.Tensor | None = None) -> torch.Tensor:
    """多正例 InfoNCE：酶嵌入与该蛋白全部真实 EC（含软祖先）的本体嵌入对齐。

    seen_mask 把训练中从未作为正例出现的 EC 类排除出 softmax 分母；
    否则这些类会被当作负例系统性推远，Mode 1 零样本检索必然失败。
    """
    sim = z @ ec_emb.t() / tau
    if seen_mask is not None:
        # 未见类不进入 softmax 分母；softmax 后这些位置的 log_prob 为 -inf，
        # 与 target=0 相乘会得到 NaN，故显式置 0，并把多正例目标限制在 seen 子空间。
        sim = sim.masked_fill(~seen_mask.unsqueeze(0), float("-inf"))
    log_prob = F.log_softmax(sim, dim=-1)
    if seen_mask is not None:
        log_prob = log_prob.masked_fill(~seen_mask.unsqueeze(0), 0.0)
        labels = labels * seen_mask.unsqueeze(0).to(labels.dtype)
    has_label = labels.sum(1) > 0
    if not has_label.any():
        return z.new_zeros(())
    target = labels / labels.sum(1, keepdim=True).clamp(min=1)
    return -(log_prob * target).sum(-1)[has_label].mean()


def make_edge_index(vocab_path: Path, ec2id: dict[str, int], device: str) -> torch.Tensor:
    edges = vocab_path.with_name("ec_edges.tsv")
    if not edges.exists():
        raise FileNotFoundError(f"缺少 EC 本体边文件: {edges}")
    return ECOntologyEncoder.load_edge_index(str(edges), ec2id).to(device)


def build_seen_mask(train_rows: list[dict], ec2id: dict[str, int],
                    device: str) -> torch.Tensor:
    """阶段一 labels（完整 EC + anc_soft 祖先 + partial 前缀）中出现的全部节点。

    与 CachedFeatureDataset(anc_soft=ANC_SOFT)::label_and_weight 的正例集合一致；
    其余节点不进入 InfoNCE 分母，从而在阶段一保持其层级先验不被推远。
    """
    nodes: set[str] = set()
    for r in train_rows:
        for ec in r["ec_numbers"]:
            if ec not in ec2id:
                continue
            nodes.add(ec)
            parts = ec.split(".")
            for cut in (1, 2, 3):
                node = ".".join(parts[:cut])
                if node != ec and node in ec2id:
                    nodes.add(node)
        for p in (r.get("ec_partial") or []):
            n = partial_node(ec2id, p)
            if n:
                nodes.add(n)
    mask = torch.zeros(len(ec2id), dtype=torch.bool, device=device)
    for ec in nodes:
        mask[ec2id[ec]] = True
    return mask


@torch.no_grad()
def evaluate(model, dl, id2ec: dict[int, str], ec2id: dict[str, int],
             device: str, unseen_ecs: set[str], top_k: int = 100):
    """Mode 2 验证指标。

    常规分数只保留 top_k 以控制 evaluate_all 内存；但零样本类若被截断，
    zeroshot_recall@10 会恒为 0（假零），故把全部未见 EC 的分数一并保留。
    """
    model.eval()
    scores, labels = [], []
    unseen_ids = [j for e in unseen_ecs if (j := ec2id.get(e)) is not None and j in id2ec]
    for batch in dl:
        logits, _ = model(ordered_feats(batch, device))
        p = torch.sigmoid(logits)
        vals, idx = p.topk(min(top_k, p.size(1)), dim=1)
        for i in range(len(idx)):
            lab = set(batch["ec_numbers"][i])
            if not lab:
                # 仅有部分 EC 的样本没有 exact-EC4 真值，不能作为假阳性进入 Fmax 分母
                continue
            row = {id2ec[int(j)]: float(v) for v, j in zip(vals[i], idx[i])}
            for j in unseen_ids:
                ec = id2ec[j]
                if ec not in row:
                    row[ec] = float(p[i, j])
            scores.append(row)
            labels.append(lab)
    model.train()
    return evaluate_all(scores, labels, unseen_ecs=unseen_ecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--tier", choices=list(TIERS), default="p1m1")
    ap.add_argument("--threedi", default="threedi_observed_afdbv6")
    ap.add_argument("--head", choices=("dot", "mlp", "permod"), default="dot",
                    help="Mode 2 分类头：mlp=计划 3.5，permod=FunBind 式每模态头+平均，dot=早期消融")
    ap.add_argument("--pretrain-epochs", type=int, default=3,
                    help="固定阶段一预算；自适应模式下由 --pretrain-patience 接管")
    ap.add_argument("--epochs", type=int, default=6,
                    help="固定阶段二预算；自适应模式下由 --early-stop-patience 接管")
    ap.add_argument("--moe-experts", type=int, default=1,
                    help="FunBind 式 MoE 投影专家数；1=退化为单 MLP（兼容旧 checkpoint）")
    ap.add_argument("--diversity-weight", type=float, default=0.0,
                    help="专家多样性损失权重（FunBind L_diversity）")
    ap.add_argument("--max-epochs", type=int, default=60,
                    help="自适应模式下阶段二预算上限（不是固定训练轮数）")
    ap.add_argument("--min-epochs", type=int, default=3,
                    help="早停生效前至少训练的 epoch 数")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help=">0 时按 val Fmax 早停：连续 N 个评估点无提升即停；0=固定预算")
    ap.add_argument("--pretrain-max-epochs", type=int, default=30,
                    help="自适应模式下阶段一预算上限")
    ap.add_argument("--pretrain-min-epochs", type=int, default=2)
    ap.add_argument("--pretrain-patience", type=int, default=0,
                    help=">0 时按阶段一对齐 loss 早停；0=固定预算")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke 用：每阶段最大步数")
    ap.add_argument("--save-best", action="store_true",
                    help="按 val Fmax 保存最佳 epoch 权重（默认关闭，保持既有口径）")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg(args.config)
    fusion = cfg["model"]["fusion"]
    proj_dim = cfg["model"]["projection_dim"]
    splits_dir, vocab_path, evidence_weight = TIERS[args.tier]
    vocab_path = ROOT / vocab_path
    vocab = json.loads(vocab_path.read_text())
    ec2id = vocab["ec2id"]
    id2ec = {int(k): v for k, v in vocab["id2ec"].items()}
    n_classes = len(ec2id)

    train_rows = load_split(ROOT / splits_dir / "train.tsv")
    val_rows = load_split(ROOT / splits_dir / "val.tsv")
    if args.max_val:
        val_rows = val_rows[:args.max_val]
    domain2id = build_domain_vocab(train_rows)
    unseen_ecs = ({e for r in val_rows for e in r["ec_numbers"]}
                  - {e for r in train_rows for e in r["ec_numbers"]})
    seen_mask = build_seen_mask(train_rows, ec2id, device)
    print(f"[info] tier={args.tier} head={args.head} train={len(train_rows)} "
          f"val={len(val_rows)} domains={len(domain2id)} classes={n_classes} "
          f"unseen_ec_in_val={len(unseen_ecs)} anc_soft={ANC_SOFT} "
          f"seen_nodes={int(seen_mask.sum())}", flush=True)

    seq_cache = EmbeddingCache("sequence")
    text_cache = EmbeddingCache("text")
    threedi_cache = EmbeddingCache(args.threedi)
    ds = CachedFeatureDataset(train_rows, ec2id, domain2id, seq_cache, text_cache,
                              threedi_cache=threedi_cache,
                              evidence_weight=evidence_weight)
    # 阶段一专用：祖先闭包软标签，仅影响 ec_alignment_loss 的多正例目标
    pretrain_ds = CachedFeatureDataset(train_rows, ec2id, domain2id, seq_cache,
                                       text_cache, threedi_cache=threedi_cache,
                                       anc_soft=ANC_SOFT)
    val_ds = CachedFeatureDataset(val_rows, ec2id, domain2id, seq_cache, text_cache,
                                  threedi_cache=threedi_cache)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=cached_collate, drop_last=True)
    pretrain_dl = DataLoader(pretrain_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=cached_collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=cached_collate)

    sample = cached_collate([ds[0]])
    raw_dims = {k: v.shape[1] for k, v in sample["feats"].items()}
    in_dims = {"sequence": raw_dims["sequence"], "structure": raw_dims["threedi"],
               "text": raw_dims["text"], "domain": raw_dims["domain"]}
    print(f"[info] in_dims={in_dims} proj_dim={proj_dim}", flush=True)

    edge_index = make_edge_index(vocab_path, ec2id, device)
    ec_encoder = ECOntologyEncoder(str(vocab_path), dim=proj_dim, layers=2).to(device)
    # 训练长度由验证/对齐曲线决定，epoch 数只是预算上限而非固定超参。
    # 显式 --pretrain-epochs / --epochs 仍作为固定预算，用于复现既有结果。
    pretrain_budget = (args.pretrain_max_epochs if args.pretrain_patience > 0
                       else args.pretrain_epochs)
    epoch_budget = args.max_epochs if args.early_stop_patience > 0 else args.epochs
    contrastive = ZymeBindContrastive(
        {k: CachedEncoder() for k in MODALITIES}, in_dims,
        proj_dim, cfg["model"]["temperature"],
        n_experts=args.moe_experts,
        diversity_weight=args.diversity_weight).to(device)
    print(f"[info] moe_experts={args.moe_experts} "
          f"diversity_weight={args.diversity_weight} "
          f"pretrain_budget={pretrain_budget} budget={epoch_budget}", flush=True)

    # ---------- 阶段一：跨模态对比预训练 ----------
    opt1 = torch.optim.AdamW(
        [p for p in list(contrastive.parameters()) + list(ec_encoder.parameters())
         if p.requires_grad], lr=args.pretrain_lr,
        weight_decay=cfg["pretrain"]["weight_decay"])
    pretrain_history = []
    pre_best, pre_wait, pre_stopped, last_logs = float("inf"), 0, None, {}
    for epoch in range(pretrain_budget):
        contrastive.train(); ec_encoder.train()
        total, n = 0.0, 0
        for step, batch in enumerate(pretrain_dl):
            feats = ordered_feats(batch, device)
            embs = contrastive({k: {"x": v} for k, v in feats.items()})
            loss_align, last_logs = contrastive.loss(embs)
            z = F.normalize(torch.stack([embs[k] for k in MODALITIES], 0).mean(0), dim=-1)
            ec_emb = F.normalize(ec_encoder(edge_index), dim=-1)
            loss_ec = ec_alignment_loss(z, ec_emb, batch["labels"].to(device),
                                        seen_mask=seen_mask)
            loss = loss_align + loss_ec
            opt1.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(opt1.param_groups[0]["params"], 1.0)
            opt1.step()
            total += float(loss); n += 1
            if args.max_steps and step + 1 >= args.max_steps:
                break
        avg = total / max(n, 1)
        pretrain_history.append({"epoch": epoch, "loss": avg, "logs": last_logs})
        print(f"[pretrain ep{epoch}] loss={avg:.4f}", flush=True)
        if args.max_steps:
            break
        if args.pretrain_patience > 0:
            pre_wait = 0 if avg < pre_best - 1e-4 else pre_wait + 1
            pre_best = min(pre_best, avg)
            if epoch + 1 >= args.pretrain_min_epochs and pre_wait >= args.pretrain_patience:
                pre_stopped = epoch
                print(f"[pretrain stop] epoch={epoch} loss={avg:.4f} "
                      f"patience={args.pretrain_patience}", flush=True)
                break

    # Mode 1 快照：阶段二会继续训练 ec_encoder，零样本检索必须用阶段一对齐后的版本
    mode1_state = {
        "contrastive": {k: v.detach().cpu().clone()
                        for k, v in contrastive.state_dict().items()},
        "ec_encoder": {k: v.detach().cpu().clone()
                       for k, v in ec_encoder.state_dict().items()},
    }

    # ---------- 阶段二：监督微调 ----------
    model = ZymeBindModel(in_dims, ec_encoder, edge_index, n_classes,
                          fusion["out_dim"], fusion["n_heads"], fusion["dropout"],
                          proj_dim, bool(cfg["finetune"]["use_concept_bottleneck"]),
                          head_type=args.head).to(device)
    # 复用阶段一学到的 EC 本体嵌入与模态投影知识：把对比投影头参数迁移到融合层
    with torch.no_grad():
        for k in MODALITIES:
            src = contrastive.heads[k].first_linear  # 单 MLP 或 MoE 首个专家的首层
            dst = model.fusion.proj[k]               # Linear(in_dim, fusion_dim)
            if src.out_features == dst.out_features:
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
    crit = FocalLoss(cfg["finetune"]["focal_gamma"], cfg["finetune"]["focal_alpha"])
    pos_weight = compute_pos_weight(train_rows, ec2id).to(device)
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=cfg["pretrain"]["weight_decay"])
    history = []
    best_fmax, best_epoch, best_state = -1.0, -1, None
    es_best, es_wait = -1.0, 0
    stopped_epoch, stop_reason = -1, "budget"
    print("| Method | Fmax | AUPR-macro | Recall@1 | single-EC Fmax | multi-EC hit@5 | multi-EC Fmax |", flush=True)
    for epoch in range(epoch_budget):
        stopped_epoch = epoch
        model.train()
        total, n = 0.0, 0
        for step, batch in enumerate(dl):
            logits, _ = model(ordered_feats(batch, device))
            lw = batch["label_weights"].to(device)
            if evidence_weight > 0:
                lw = lw * batch["sample_weights"].to(device).unsqueeze(1)
            loss = crit(logits, batch["labels"].to(device), lw, pos_weight)
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            total += float(loss); n += 1
            if args.max_steps and step + 1 >= args.max_steps:
                break
        entry = {"epoch": epoch, "train_loss": total / max(n, 1)}
        print(f"[ep{epoch}] train_loss={total / max(n, 1):.4f}", flush=True)
        if (epoch + 1) % args.eval_every == 0 or epoch == epoch_budget - 1:
            res = evaluate(model, val_dl, id2ec, ec2id, device, unseen_ecs)
            print(main_table_row(res, f"zymebind_{args.tier}_ep{epoch}"), flush=True)
            entry["metrics"] = res
            if args.save_best and float(res["fmax"]) > best_fmax:
                best_fmax, best_epoch = float(res["fmax"]), epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if args.early_stop_patience > 0:
                fmax = float(res["fmax"])
                es_wait = 0 if fmax > es_best + 1e-4 else es_wait + 1
                es_best = max(es_best, fmax)
                if epoch >= args.min_epochs and es_wait >= args.early_stop_patience:
                    stop_reason = "early_stop"
                    print(f"[early stop] epoch={epoch} fmax={fmax:.4f} "
                          f"patience={args.early_stop_patience}", flush=True)
        history.append(entry)
        if stop_reason == "early_stop" or args.max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[best] epoch={best_epoch} val_fmax={best_fmax:.4f}", flush=True)
    out = Path(args.out or f"results/zymebind_{args.tier}_hierprior.json")
    ckpt = Path(args.ckpt or f"checkpoints/zymebind_{args.tier}_hierprior.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "contrastive": contrastive.state_dict(),
                "mode1": mode1_state, "head_type": args.head,
                "moe_experts": args.moe_experts,
                "diversity_weight": args.diversity_weight,
                "in_dims": in_dims, "tier": args.tier,
                "vocab": str(vocab_path.relative_to(ROOT)), "anc_soft": ANC_SOFT}, ckpt)
    result = {
        "args": vars(args), "anc_soft": ANC_SOFT,
        "ec_ontology": "level_prior_residual", "head_type": args.head,
        "moe_experts": args.moe_experts,
        "diversity_weight": args.diversity_weight,
        "seen_nodes_in_alignment": int(seen_mask.sum()),
        "pretrain_history": pretrain_history, "history": history,
        "pretrain_budget": pretrain_budget,
        "pretrain_stopped_epoch": pre_stopped,
        "epoch_budget": epoch_budget,
        "stopped_epoch": stopped_epoch,
        "stop_reason": stop_reason,
        "n_unseen_ec_in_val": len(unseen_ecs),
    }
    if args.save_best:
        result["best_epoch"] = best_epoch
        result["best_val_fmax"] = best_fmax
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[ok] -> {out} / {ckpt}", flush=True)


if __name__ == "__main__":
    main()
