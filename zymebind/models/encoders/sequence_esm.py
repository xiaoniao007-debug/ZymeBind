"""序列模态编码器：ESM-2 650M（方案 3.2）。

输出: 蛋白级嵌入 (B, 1280) 与残基级嵌入 (B, L, 1280)（L1 归因需要）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SequenceESMEncoder(nn.Module):
    def __init__(self, name: str = "facebook/esm2_t33_650M_UR50D", dim: int = 1280,
                 freeze_layers: int = 24, lora_r: int = 0, lora_alpha: int = 16,
                 grad_checkpointing: bool = True):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        if grad_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.dim = dim

        # 冻结底部若干层，降低显存与算力需求（L20 适配）
        for i, layer in enumerate(self.model.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        for p in self.model.embeddings.parameters():
            p.requires_grad = False

        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                             target_modules=["query", "key", "value"],
                             bias="none")
            self.model = get_peft_model(self.model, cfg)

    def forward(self, input_ids, attention_mask, return_residue: bool = False):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=return_residue)
        if return_residue:
            h = out.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return pooled, h
        h = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
