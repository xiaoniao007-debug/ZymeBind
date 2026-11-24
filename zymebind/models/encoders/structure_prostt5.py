"""结构模态编码器：3Di 序列 -> ProstT5 编码（方案 3.2）。

流程: AFDB/ESMFold 结构 -> Foldseek 转 3Di 序列 -> ProstT5 encoder 编码。
ProstT5 全程冻结，仅作特征提取（单卡 48GB 可处理 batch 8 × 1024 aa）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

AA3DI_PREFIX = "<AA2fold>"


class ProstT5Encoder(nn.Module):
    def __init__(self, name: str = "Rostlab/ProstT5", dim: int = 1024,
                 freeze: bool = True, max_len: int = 1024):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(name)
        self.max_len = max_len
        self.dim = dim
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @staticmethod
    def seq_to_t5(seq: str) -> str:
        return " ".join(list(seq.replace("U", "X").replace("Z", "X").replace("O", "X")))

    def forward(self, three_di_seqs: list[str], device: str = "cuda"):
        seqs = [AA3DI_PREFIX + self.seq_to_t5(s[: self.max_len]) for s in three_di_seqs]
        enc = self.tokenizer(seqs, add_special_tokens=True, padding="longest",
                             return_tensors="pt").to(device)
        with torch.no_grad():
            h = self.model(input_ids=enc.input_ids,
                           attention_mask=enc.attention_mask).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
