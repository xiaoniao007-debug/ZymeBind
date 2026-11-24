"""ZymeBind v2: 缺失模态鲁棒 + 语义本体先验 + EC-SupCon + 跨模态蒸馏架构。

核心改进：
1. SemanticECOntologyEncoder: 用 BioBERT 编码的 ExPASy enzyme.dat 反应描述作为
   EC 本体节点语义先验，替代随机 MD5 哈希，经 2 层双向 GCN 在 EC 层级 DAG 上传播。
2. PresenceAwareModalityFusion: 显式检测非零模态掩码（presence_mask），通过
   Transformer src_key_padding_mask 完全屏蔽缺失模态注意力，彻底杜绝零向量污染。
3. ZymeBindV2Stage1 (Mode 1): MoE 投影 + 多视角对齐（full / seq+3di / seq_only）+
   EC-SupCon + 语义本体对齐（严格保留 seen_mask 保护未见类）。
4. ZymeBindV2Stage2 (Mode 2): 融合主头 + 四模态独立监督辅助头 + 缺失感知加权共识，
   配合结构化 Modality Dropout、跨模态知识蒸馏与 Asymmetric Loss (ASL)。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None

try:
    from zymebind.models.contrastive import MoEProjectionHead, ProjectionHead
    from zymebind.models.encoders.ec_ontology_gnn import level_prior
except ImportError:
    from models.contrastive import MoEProjectionHead, ProjectionHead
    from models.encoders.ec_ontology_gnn import level_prior

MODALITIES = ("sequence", "structure", "text", "domain")


class AsymmetricLoss(nn.Module):
    """多标签长尾 Asymmetric Loss (Ridnik et al., ICCV 2021)。

    在 6,100+ 类极端长尾（负例占比 >99.95%）下，对易分负例施加强抑制
    （gamma_neg=4, clip=0.05）而不压制正例梯度（gamma_pos=0）。
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                label_weights: torch.Tensor | None = None,
                sample_weights: torch.Tensor | None = None) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = -(los_pos + los_neg) * one_sided_w
        else:
            loss = -(los_pos + los_neg)

        if label_weights is not None:
            loss = loss * label_weights
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        # 按样本求和后取 batch 均值（比 .mean() 梯度尺度更稳定，不受 6162 类稀释）
        return loss.sum(dim=1).mean()


class SemanticECOntologyEncoder(nn.Module):
    """BioBERT 反应描述语义先验 + 层级结构先验 + 双向 EC DAG GCN。"""

    def __init__(self, vocab_path: str, semantic_emb: torch.Tensor,
                 dim: int = 256, layers: int = 2, dropout: float = 0.1,
                 use_delta: bool = True):
        super().__init__()
        vocab = json.loads(Path(vocab_path).read_text())
        self.ec2id: dict[str, int] = vocab["ec2id"]
        self.n_nodes = len(self.ec2id)
        self.use_delta = use_delta
        sem_dim = int(semantic_emb.shape[1])
        self.register_buffer("sem_prior", F.normalize(semantic_emb.float(), dim=-1))
        self.register_buffer("struct_prior", level_prior(self.ec2id, dim))
        self.sem_proj = nn.Sequential(
            nn.Linear(sem_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.2))  # 结构先验残差权重
        if use_delta:
            self.delta = nn.Parameter(torch.zeros(self.n_nodes, dim))
        else:
            self.register_buffer("delta", torch.zeros(self.n_nodes, dim))
        self.convs = nn.ModuleList([GCNConv(dim, dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = dim

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        s = F.normalize(self.sem_proj(self.sem_prior), dim=-1)
        x = F.normalize(s + self.alpha * self.struct_prior + self.delta, dim=-1)
        for conv, norm in zip(self.convs, self.norms):
            neigh = self.dropout(norm(F.gelu(conv(x, edge_index))))
            x = F.normalize(x + neigh, dim=-1)
        return x


def detect_presence(feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """检测每个样本各模态是否非零（返回 bool tensor [B]）。"""
    return {k: (v.abs().sum(dim=-1) > 1e-6) for k, v in feats.items()}


class ModalityEncoderBlock(nn.Module):
    """单模态 2 层残差编码块。"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PresenceAwareModalityFusion(nn.Module):
    """显式屏蔽缺失模态的 Transformer 多模态融合器。"""

    def __init__(self, in_dims: dict[str, int], out_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.modalities = tuple(in_dims.keys())
        self.encoders = nn.ModuleDict({
            k: ModalityEncoderBlock(d, out_dim, dropout) for k, d in in_dims.items()
        })
        self.mod_type = nn.ParameterDict({
            k: nn.Parameter(torch.randn(1, 1, out_dim) * 0.02) for k in in_dims
        })
        self.cls = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                out_dim, n_heads, out_dim * 4, dropout,
                activation="gelu", batch_first=True, norm_first=True
            ),
            num_layers=n_layers,
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, feats: dict[str, torch.Tensor],
                present_mask: dict[str, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if present_mask is None:
            present_mask = detect_presence(feats)
        B = next(iter(feats.values())).size(0)
        device = next(iter(feats.values())).device

        mod_tokens: dict[str, torch.Tensor] = {}
        tok_list = [self.cls.expand(B, -1, -1)]
        pad_masks = [torch.zeros(B, 1, dtype=torch.bool, device=device)]  # CLS never masked

        for k in self.modalities:
            h = self.encoders[k](feats[k])
            mod_tokens[k] = h
            tok_list.append(h.unsqueeze(1) + self.mod_type[k])
            # True = ignored by Transformer attention
            pad_masks.append(~present_mask[k].unsqueeze(1))

        tokens = torch.cat(tok_list, dim=1)          # (B, 1 + M, D)
        key_pad = torch.cat(pad_masks, dim=1)        # (B, 1 + M)
        # 安全兜底：如果某样本极端情况下所有模态全缺，不屏蔽 sequence 防止 NaN
        all_missing = key_pad[:, 1:].all(dim=1)
        if all_missing.any():
            key_pad = key_pad.clone()
            key_pad[all_missing, 1] = False

        enc = self.transformer(tokens, src_key_padding_mask=key_pad)
        cls_out = enc[:, 0]

        # 仅对存在模态做平均池化残差，保证单模态下梯度直通
        valid_f = (~key_pad[:, 1:]).float().unsqueeze(-1)  # (B, M, 1)
        mod_out = enc[:, 1:]
        mean_out = (mod_out * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        fused = self.out_norm(0.5 * cls_out + 0.5 * mean_out)
        return fused, mod_tokens, present_mask


class ZymeBindV2Stage1(nn.Module):
    """阶段一：MoE 对比投影 + 多视角查询编码（供 Mode 1 零样本检索）。"""

    def __init__(self, in_dims: dict[str, int], proj_dim: int = 256,
                 n_experts: int = 3, temperature: float = 0.15,
                 dropout: float = 0.1, diversity_weight: float = 0.05):
        super().__init__()
        self.modalities = tuple(in_dims.keys())
        if n_experts > 1:
            self.heads = nn.ModuleDict({
                k: MoEProjectionHead(d, proj_dim, n_experts=n_experts, dropout=dropout)
                for k, d in in_dims.items()
            })
        else:
            self.heads = nn.ModuleDict({
                k: ProjectionHead(d, proj_dim, dropout=dropout)
                for k, d in in_dims.items()
            })
        self.mod_logits = nn.Parameter(torch.zeros(len(self.modalities)))
        self.log_tau = nn.Parameter(torch.tensor(temperature).log())
        self.diversity_weight = diversity_weight

    def project_all(self, feats: dict[str, torch.Tensor]
                    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        embs = {}
        div_loss = torch.tensor(0.0, device=next(iter(feats.values())).device)
        n_div = 0
        for k in self.modalities:
            h = self.heads[k]
            if isinstance(h, MoEProjectionHead):
                e, exp_out = h(feats[k], return_experts=True)
                embs[k] = e
                div_loss = div_loss + MoEProjectionHead.diversity_loss(exp_out)
                n_div += 1
            else:
                embs[k] = h(feats[k])
        if n_div > 0:
            div_loss = div_loss / n_div
        return embs, div_loss

    def encode_query(self, feats: dict[str, torch.Tensor],
                     present_mask: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """按非零模态存在掩码做加权融合，返回归一化酶查询向量 z。"""
        if present_mask is None:
            present_mask = detect_presence(feats)
        embs, _ = self.project_all(feats)
        weights = F.softplus(self.mod_logits)  # 正权重
        num = 0.0
        den = 0.0
        for i, k in enumerate(self.modalities):
            m = present_mask[k].float().unsqueeze(-1) * weights[i]
            num = num + embs[k] * m
            den = den + m
        return F.normalize(num / den.clamp(min=1e-6), dim=-1)


class ZymeBindV2Stage2(nn.Module):
    """阶段二：缺失模态感知融合主头 + 四模态独立监督辅助头 + 缺失感知共识。"""

    def __init__(self, in_dims: dict[str, int], n_classes: int,
                 fusion_dim: int = 512, n_heads: int = 8, n_layers: int = 2,
                 dropout: float = 0.15, hidden_dim: int = 1024, proj_dim: int = 256):
        super().__init__()
        self.modalities = tuple(in_dims.keys())
        self.fusion = PresenceAwareModalityFusion(
            in_dims, fusion_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        )
        self.fused_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        # 每个模态拥有独立监督头，防止 text/domain 垄断序列/结构梯度
        self.permod_heads = nn.ModuleDict({
            k: nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_classes),
            )
            for k in self.modalities
        })
        # 可学习的共识权重：[fused, seq, struct, text, domain]
        self.consensus_logits = nn.Parameter(torch.tensor([1.0, 0.5, 0.4, 0.5, 0.5]))
        self.z_proj = nn.Sequential(
            nn.Linear(fusion_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward_detailed(self, feats: dict[str, torch.Tensor],
                         present_mask: dict[str, torch.Tensor] | None = None
                         ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        fused, mod_tokens, present = self.fusion(feats, present_mask=present_mask)
        fused_logits = self.fused_head(fused)
        permod_logits = {k: self.permod_heads[k](mod_tokens[k]) for k in self.modalities}

        # 缺失感知加权共识：仅融合存在模态的头 + fused 主头
        B = fused.size(0)
        device = fused.device
        w_raw = self.consensus_logits.unsqueeze(0).expand(B, -1)  # (B, 1+M)
        avail = torch.stack(
            [torch.ones(B, dtype=torch.bool, device=device)] + [present[k] for k in self.modalities],
            dim=1,
        )  # (B, 1+M)
        w = F.softmax(w_raw.masked_fill(~avail, float("-inf")), dim=1)  # (B, 1+M)

        all_logits = torch.stack([fused_logits] + [permod_logits[k] for k in self.modalities], dim=1)
        consensus_logits = (all_logits * w.unsqueeze(-1)).sum(dim=1)
        z = F.normalize(self.z_proj(fused), dim=-1)
        return {
            "logits": consensus_logits,
            "fused_logits": fused_logits,
            "permod_logits": permod_logits,
            "fused": fused,
            "z": z,
            "present": present,
        }

    def forward(self, feats: dict[str, torch.Tensor],
                present_mask: dict[str, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward_detailed(feats, present_mask=present_mask)
        return out["logits"], out["z"]


def load_v2_checkpoint(ckpt_path: Path, device: str):
    """加载 ZymeBind v2 checkpoint，返回 (stage1, ec_encoder, stage2, edge_index, ckpt)。"""
    from training_zymebind import make_edge_index
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    root = Path(__file__).resolve().parent.parent
    vocab_path = root / str(ckpt["vocab"])
    vocab = json.loads(vocab_path.read_text())
    ec2id = vocab["ec2id"]
    edge_index = make_edge_index(vocab_path, ec2id, device)

    mcfg = ckpt["v2_cfg"]
    in_dims = ckpt["in_dims"]
    sem_shape = ckpt["mode1"]["ec_encoder"]["sem_prior"].shape
    dummy_sem = torch.zeros(sem_shape)

    stage1 = ZymeBindV2Stage1(
        in_dims, proj_dim=mcfg["proj_dim"], n_experts=mcfg["moe_experts"],
        temperature=mcfg["temperature"], dropout=mcfg["dropout"],
        diversity_weight=mcfg["diversity_weight"],
    ).to(device)
    stage1.load_state_dict(ckpt["mode1"]["stage1"])
    stage1.eval()

    ec_encoder = SemanticECOntologyEncoder(
        str(vocab_path), dummy_sem, dim=mcfg["proj_dim"],
        layers=mcfg["gcn_layers"], dropout=mcfg["dropout"],
        use_delta=bool(mcfg.get("use_ec_delta", True)),
    ).to(device)
    ec_encoder.load_state_dict(ckpt["mode1"]["ec_encoder"])
    ec_encoder.eval()

    stage2 = ZymeBindV2Stage2(
        in_dims, len(ec2id), fusion_dim=mcfg["fusion_dim"],
        n_heads=mcfg["n_heads"], n_layers=mcfg["n_layers"],
        dropout=mcfg["dropout"], hidden_dim=mcfg["hidden_dim"],
        proj_dim=mcfg["proj_dim"],
    ).to(device)
    stage2.load_state_dict(ckpt["model"])
    stage2.eval()
    return stage1, ec_encoder, stage2, edge_index, ckpt
