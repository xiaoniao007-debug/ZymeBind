"""End-to-end supervised EC prediction pipeline for ZymeBind.

Supports sequence-only (`ESM-2 650M + ProstT5-3B` dual-PLM representations),
sequence + Foldseek 3Di structure (`ZymeBind (Seq + Structure)`), and full
multimodal inference with native missing-modality masking.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from zymebind.data.fasta_io import FastaRecord, read_3di_map, read_fasta, write_tsv_predictions


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for model inference. Install GPU/CPU dependencies with:\n"
            "  pip install -e '.[gpu]'  # or see README.md for CUDA wheel installation"
        ) from exc


def _resolve_device(device: str) -> str:
    torch = _require_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class DualPLMEncoder:
    """Extracts concatenated ESM-2 650M (1280-d) + ProstT5-3B AA (1024-d) embeddings,
    plus optional ProstT5-3B Foldseek 3Di structural embeddings (1024-d).
    """

    def __init__(
        self,
        device: str = "cpu",
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        prostt5_model_name: str = "Rostlab/ProstT5",
        use_prostt5: bool = True,
        max_length: int = 1024,
    ):
        self.device = device
        self.esm_model_name = esm_model_name
        self.prostt5_model_name = prostt5_model_name
        self.use_prostt5 = use_prostt5
        self.max_length = max_length
        self._esm_tok = None
        self._esm_mod = None
        self._pt5_tok = None
        self._pt5_mod = None

    def _load_esm(self):
        if self._esm_mod is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._esm_tok = AutoTokenizer.from_pretrained(self.esm_model_name)
        self._esm_mod = AutoModel.from_pretrained(self.esm_model_name).to(self.device).eval()

    def _load_prostt5(self):
        if self._pt5_mod is not None:
            return
        from transformers import T5EncoderModel, T5Tokenizer

        self._pt5_tok = T5Tokenizer.from_pretrained(self.prostt5_model_name, do_lower_case=False)
        self._pt5_mod = T5EncoderModel.from_pretrained(self.prostt5_model_name).to(self.device).eval()

    def encode_sequences(self, sequences: list[str], batch_size: int = 8):
        torch = _require_torch()
        self._load_esm()
        esm_vecs = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = [s[: self.max_length] for s in sequences[i : i + batch_size]]
                enc = self._esm_tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length + 2,
                ).to(self.device)
                out = self._esm_mod(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                esm_vecs.append(pooled.cpu())
        esm_mat = torch.cat(esm_vecs, dim=0)

        if not self.use_prostt5:
            return esm_mat

        self._load_prostt5()
        pt5_vecs = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = [
                    "<AA2fold> " + " ".join(list(re.sub(r"[UZOB]", "X", s[: self.max_length])))
                    for s in sequences[i : i + batch_size]
                ]
                enc = self._pt5_tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length + 4,
                ).to(self.device)
                out = self._pt5_mod(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                ).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pt5_vecs.append(pooled.cpu())
        pt5_mat = torch.cat(pt5_vecs, dim=0)
        return torch.cat([esm_mat, pt5_mat], dim=-1)

    def encode_3di(self, sequences_3di: list[str], batch_size: int = 8):
        torch = _require_torch()
        self._load_prostt5()
        vecs = []
        with torch.no_grad():
            for i in range(0, len(sequences_3di), batch_size):
                batch = [
                    "<fold2AA> " + " ".join(list(s[: self.max_length].lower()))
                    for s in sequences_3di[i : i + batch_size]
                ]
                enc = self._pt5_tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length + 4,
                ).to(self.device)
                out = self._pt5_mod(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                ).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                vecs.append(pooled.cpu())
        return torch.cat(vecs, dim=0)


def load_zymebind_checkpoint(
    ckpt_path: str | Path,
    vocab_path: str | Path | None = None,
    device: str = "auto",
) -> tuple[Any, dict[int, str], dict[str, int], dict[str, Any]]:
    """Load a trained ZymeBind Stage 2 checkpoint for inference."""
    torch = _require_torch()
    dev = _resolve_device(device)
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location=dev, weights_only=False)

    if vocab_path is not None:
        vpath = Path(vocab_path)
    else:
        vpath = Path(str(ckpt.get("vocab", "data/processed/ec_vocab.json")))
        if not vpath.is_file():
            vpath = ckpt_file.parent / "ec_vocab.json"
    if not vpath.is_file():
        raise FileNotFoundError(
            f"EC vocabulary JSON not found at {vpath}. Pass --vocab explicitly."
        )

    vocab = json.loads(vpath.read_text(encoding="utf-8"))
    ec2id: dict[str, int] = vocab["ec2id"]
    id2ec: dict[int, str] = {int(v): k for k, v in ec2id.items()}

    from zymebind.models.zymebind_v2 import ZymeBindV2Stage2

    mcfg = ckpt["v2_cfg"]
    in_dims: dict[str, int] = ckpt["in_dims"]
    model = ZymeBindV2Stage2(
        in_dims=in_dims,
        n_classes=len(ec2id),
        fusion_dim=mcfg["fusion_dim"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        dropout=mcfg["dropout"],
        hidden_dim=mcfg["hidden_dim"],
        proj_dim=mcfg["proj_dim"],
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, id2ec, in_dims, ckpt


def run_predict(
    fasta_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    vocab_path: str | Path | None = None,
    threedi_path: str | Path | None = None,
    mode: str = "sequence",
    threshold: float = 0.62,
    topk: int = 5,
    batch_size: int = 8,
    device: str = "auto",
    output_format: str = "tsv",
) -> list[dict[str, Any]]:
    """Run supervised ZymeBind EC prediction on a FASTA file."""
    torch = _require_torch()
    dev = _resolve_device(device)
    records: list[FastaRecord] = read_fasta(fasta_path)
    model, id2ec, in_dims, _ = load_zymebind_checkpoint(
        checkpoint_path, vocab_path=vocab_path, device=dev
    )

    use_prostt5 = in_dims.get("sequence", 2304) > 1280
    encoder = DualPLMEncoder(device=dev, use_prostt5=use_prostt5)
    seq_emb = encoder.encode_sequences([r.sequence for r in records], batch_size=batch_size)

    n_samples = len(records)
    struct_emb = torch.zeros(n_samples, in_dims["structure"], dtype=torch.float32)
    text_emb = torch.zeros(n_samples, in_dims["text"], dtype=torch.float32)
    domain_emb = torch.zeros(n_samples, in_dims["domain"], dtype=torch.float32)

    active_mod_label = "sequence"
    if mode == "seq_struct" or threedi_path is not None:
        if threedi_path is None:
            raise ValueError("Mode 'seq_struct' requires --threedi <3di_fasta_or_tsv>.")
        threedi_map = read_3di_map(threedi_path)
        present_idx = [i for i, r in enumerate(records) if r.protein_id in threedi_map]
        if present_idx:
            sub_3di = [threedi_map[records[i].protein_id] for i in present_idx]
            encoded_3di = encoder.encode_3di(sub_3di, batch_size=batch_size)
            for j, idx in enumerate(present_idx):
                struct_emb[idx] = encoded_3di[j]
            active_mod_label = "sequence+structure"

    feats = {
        "sequence": seq_emb.to(dev),
        "structure": struct_emb.to(dev),
        "text": text_emb.to(dev),
        "domain": domain_emb.to(dev),
    }

    with torch.no_grad():
        logits, _ = model(feats)
        probs = torch.sigmoid(logits).cpu()

    rows: list[dict[str, Any]] = []
    json_payload: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        p = probs[i]
        top_vals, top_idx = torch.topk(p, k=min(topk, p.numel()))
        preds_for_protein = []
        for rank, (score, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
            ec_code = id2ec[idx]
            is_pos = score >= threshold
            row = {
                "protein_id": rec.protein_id,
                "rank": rank,
                "ec_number": ec_code,
                "probability": f"{score:.4f}",
                "above_threshold": str(is_pos).lower(),
                "active_modalities": active_mod_label,
            }
            rows.append(row)
            preds_for_protein.append(
                {
                    "rank": rank,
                    "ec_number": ec_code,
                    "probability": round(score, 6),
                    "above_threshold": is_pos,
                }
            )
        json_payload.append(
            {
                "protein_id": rec.protein_id,
                "length": len(rec.sequence),
                "active_modalities": active_mod_label,
                "predictions": preds_for_protein,
            }
        )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json" or out_file.suffix.lower() == ".json":
        out_file.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    else:
        write_tsv_predictions(rows, out_file)

    return rows
