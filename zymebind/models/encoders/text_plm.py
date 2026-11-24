"""文本模态编码器：UniProt 功能描述 -> PubMedBERT/BioBERT（方案 3.2）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 dim: int = 768, freeze: bool = True, max_len: int = 512):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.max_len = max_len
        self.dim = dim
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, texts: list[str], device: str = "cuda"):
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_len,
                             padding="longest", return_tensors="pt").to(device)
        with torch.no_grad():
            h = self.model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
