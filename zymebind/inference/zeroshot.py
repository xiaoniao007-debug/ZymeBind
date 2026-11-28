"""Zero-shot unseen-EC4 retrieval on ZymeBind's L2-normalized ontology hypersphere."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zymebind.data.fasta_io import read_3di_map, read_fasta
from zymebind.inference.predictor import DualPLMEncoder, _require_torch, _resolve_device


def run_zeroshot(
    fasta_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    vocab_path: str | Path | None = None,
    threedi_path: str | Path | None = None,
    topk: int = 10,
    batch_size: int = 8,
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Retrieve nearest EC4 ontology prototypes for query enzymes in Stage 1 space."""
    torch = _require_torch()
    import torch.nn.functional as F

    dev = _resolve_device(device)
    records = read_fasta(fasta_path)
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    ckpt = torch.load(ckpt_file, map_location=dev, weights_only=False)
    vpath = Path(vocab_path) if vocab_path else Path(str(ckpt.get("vocab", "data/processed/ec_vocab.json")))
    if not vpath.is_file():
        raise FileNotFoundError(f"EC vocabulary JSON not found: {vpath}")

    vocab = json.loads(vpath.read_text(encoding="utf-8"))
    ec2id: dict[str, int] = vocab["ec2id"]
    id2ec: dict[int, str] = {int(v): k for k, v in ec2id.items()}

    from zymebind.models.zymebind_v2 import SemanticECOntologyEncoder, ZymeBindV2Stage1
    from zymebind.training.training_zymebind import make_edge_index

    mcfg = ckpt["v2_cfg"]
    in_dims: dict[str, int] = ckpt["in_dims"]
    edge_index = make_edge_index(vpath, ec2id, dev)
    sem_shape = ckpt["mode1"]["ec_encoder"]["sem_prior"].shape

    stage1 = ZymeBindV2Stage1(
        in_dims,
        proj_dim=mcfg["proj_dim"],
        n_experts=mcfg["moe_experts"],
        temperature=mcfg["temperature"],
        dropout=mcfg["dropout"],
        diversity_weight=mcfg["diversity_weight"],
    ).to(dev)
    stage1.load_state_dict(ckpt["mode1"]["stage1"])
    stage1.eval()

    ec_encoder = SemanticECOntologyEncoder(
        str(vpath),
        torch.zeros(sem_shape),
        dim=mcfg["proj_dim"],
        layers=mcfg["gcn_layers"],
        dropout=mcfg["dropout"],
        use_delta=bool(mcfg.get("use_ec_delta", True)),
    ).to(dev)
    ec_encoder.load_state_dict(ckpt["mode1"]["ec_encoder"])
    ec_encoder.eval()

    use_prostt5 = in_dims.get("sequence", 2304) > 1280
    encoder = DualPLMEncoder(device=dev, use_prostt5=use_prostt5)
    seq_emb = encoder.encode_sequences([r.sequence for r in records], batch_size=batch_size)

    n = len(records)
    struct_emb = torch.zeros(n, in_dims["structure"], dtype=torch.float32)
    if threedi_path is not None:
        tmap = read_3di_map(threedi_path)
        idx_list = [i for i, r in enumerate(records) if r.protein_id in tmap]
        if idx_list:
            enc_3di = encoder.encode_3di([tmap[records[i].protein_id] for i in idx_list], batch_size=batch_size)
            for j, idx in enumerate(idx_list):
                struct_emb[idx] = enc_3di[j]

    feats = {
        "sequence": seq_emb.to(dev),
        "structure": struct_emb.to(dev),
        "text": torch.zeros(n, in_dims["text"], device=dev),
        "domain": torch.zeros(n, in_dims["domain"], device=dev),
    }

    with torch.no_grad():
        ec_emb = F.normalize(ec_encoder(edge_index), dim=-1)
        z_prot = F.normalize(stage1.encode_query(feats), dim=-1)
        sims = (z_prot @ ec_emb.T).cpu()

    # Restrict retrieval ranking to 4th-level EC4 nodes (a.b.c.d with no wildcards)
    ec4_ids = [idx for idx, code in id2ec.items() if code.count(".") == 3 and "-" not in code]
    ec4_tensor = torch.tensor(ec4_ids, dtype=torch.long)

    results: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        sub_sims = sims[i, ec4_tensor]
        vals, rel_idx = torch.topk(sub_sims, k=min(topk, sub_sims.numel()))
        hits = [
            {
                "rank": rank,
                "ec_number": id2ec[int(ec4_tensor[int(r)])],
                "cosine_similarity": round(float(v), 6),
            }
            for rank, (v, r) in enumerate(zip(vals, rel_idx), start=1)
        ]
        results.append({"protein_id": rec.protein_id, "zero_shot_ec4_hits": hits})

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
