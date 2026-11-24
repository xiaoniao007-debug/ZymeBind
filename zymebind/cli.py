"""Command-line interface for ZymeBind.

Usage examples:
    zymebind info
    zymebind predict --fasta examples/sample_enzymes.fasta --ckpt checkpoints/zymebind.pt --out preds.tsv
    zymebind zeroshot --fasta examples/sample_enzymes.fasta --ckpt checkpoints/zymebind.pt --out zs_hits.json
    zymebind explain --level l2 --ckpt checkpoints/zymebind.pt --out shapley.json
    zymebind train --help
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from zymebind import __authors__, __version__


def _cmd_info(_args: argparse.Namespace) -> int:
    print(f"ZymeBind v{__version__}")
    print("Presence-Aware Multimodal Enzyme EC Prediction & Mechanistic Interpretability")
    print(f"Authors: {', '.join(__authors__)}")
    print(f"Python:  {platform.python_version()} ({platform.platform()})")
    try:
        import torch

        cuda_str = (
            f"CUDA available ({torch.cuda.get_device_name(0)})"
            if torch.cuda.is_available()
            else "CPU only"
        )
        print(f"PyTorch: {torch.__version__} | {cuda_str}")
    except ImportError:
        print("PyTorch: not installed (install with: pip install -e '.[gpu]')")
    print("Default operating threshold: 0.62 (validation-tuned)")
    print("Supported modes: sequence (ESM-2 650M + ProstT5-3B), seq_struct (+ Foldseek 3Di), full")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from zymebind.inference.predictor import run_predict

    rows = run_predict(
        fasta_path=args.fasta,
        checkpoint_path=args.ckpt,
        output_path=args.out,
        vocab_path=args.vocab,
        threedi_path=args.threedi,
        mode=args.mode,
        threshold=args.threshold,
        topk=args.topk,
        batch_size=args.batch_size,
        device=args.device,
        output_format=args.format,
    )
    print(f"[zymebind predict] Wrote {len(rows)} ranked EC predictions to {args.out}")
    return 0


def _cmd_zeroshot(args: argparse.Namespace) -> int:
    from zymebind.inference.zeroshot import run_zeroshot

    results = run_zeroshot(
        fasta_path=args.fasta,
        checkpoint_path=args.ckpt,
        output_path=args.out,
        vocab_path=args.vocab,
        threedi_path=args.threedi,
        topk=args.topk,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"[zymebind zeroshot] Retrieved top-{args.topk} EC4 candidates for {len(results)} proteins -> {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parent / "training" / "training_zymebind_v45_valonly.py"
    cmd = [sys.executable, str(script)] + args.train_args
    print(f"[zymebind train] Executing: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _cmd_explain(args: argparse.Namespace) -> int:
    interp_dir = Path(__file__).resolve().parent / "interpretability"
    script_map = {
        "l1_masking": interp_dir / "interp_l1_site_masking.py",
        "l1_ig": interp_dir / "interp_l1_ig.py",
        "l2_shapley": interp_dir / "run_shapley_l2.py",
        "l3_sae": interp_dir / "train_l3_sae_val.py",
    }
    script = script_map[args.level]
    cmd = [sys.executable, str(script)] + args.explain_args
    print(f"[zymebind explain:{args.level}] Executing: {' '.join(cmd)}")
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zymebind",
        description=(
            "ZymeBind CLI: presence-aware multimodal enzyme EC prediction, "
            "zero-shot unseen-EC4 retrieval, and L1/L2/L3 mechanistic interpretability."
        ),
    )
    parser.add_argument("--version", action="version", version=f"zymebind {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="Display ZymeBind version, environment, and default settings.")
    p_info.set_defaults(func=_cmd_info)

    # predict
    p_pred = sub.add_parser("predict", help="Predict EC numbers for enzyme sequences in a FASTA file.")
    p_pred.add_argument("--fasta", required=True, help="Input protein FASTA file.")
    p_pred.add_argument("--ckpt", required=True, help="Path to trained ZymeBind checkpoint (.pt).")
    p_pred.add_argument("--vocab", default=None, help="Path to ec_vocab.json (inferred from checkpoint if omitted).")
    p_pred.add_argument("--threedi", default=None, help="Optional Foldseek 3Di sequences (FASTA or TSV) for Seq+Structure mode.")
    p_pred.add_argument(
        "--mode",
        choices=["sequence", "seq_struct"],
        default="sequence",
        help="Inference modality configuration (default: sequence).",
    )
    p_pred.add_argument("--threshold", type=float, default=0.62, help="Probability threshold for positive EC calls (default: 0.62).")
    p_pred.add_argument("--topk", type=int, default=5, help="Number of top-ranked EC predictions per protein (default: 5).")
    p_pred.add_argument("--batch-size", type=int, default=8, help="Batch size for PLM feature extraction (default: 8).")
    p_pred.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device (default: auto).")
    p_pred.add_argument("--format", choices=["tsv", "json"], default="tsv", help="Output format (default: tsv).")
    p_pred.add_argument("--out", default="zymebind_predictions.tsv", help="Output file path.")
    p_pred.set_defaults(func=_cmd_predict)

    # zeroshot
    p_zs = sub.add_parser("zeroshot", help="Zero-shot unseen-EC4 retrieval via Stage 1 semantic ontology alignment.")
    p_zs.add_argument("--fasta", required=True, help="Input protein FASTA file.")
    p_zs.add_argument("--ckpt", required=True, help="Path to trained ZymeBind checkpoint (.pt) containing Stage 1 weights.")
    p_zs.add_argument("--vocab", default=None, help="Path to ec_vocab.json.")
    p_zs.add_argument("--threedi", default=None, help="Optional Foldseek 3Di sequences (FASTA or TSV).")
    p_zs.add_argument("--topk", type=int, default=10, help="Number of nearest EC4 ontology prototypes to return (default: 10).")
    p_zs.add_argument("--batch-size", type=int, default=8, help="Batch size for PLM feature extraction (default: 8).")
    p_zs.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device.")
    p_zs.add_argument("--out", default="zymebind_zeroshot.json", help="Output JSON file path.")
    p_zs.set_defaults(func=_cmd_zeroshot)

    # train
    p_train = sub.add_parser("train", help="Launch two-stage ZymeBind training (forwards flags to the training script).")
    p_train.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed directly to the training runner.")
    p_train.set_defaults(func=_cmd_train)

    # explain
    p_exp = sub.add_parser("explain", help="Run Level 1 (active-site), Level 2 (Shapley), or Level 3 (SAE) interpretability.")
    p_exp.add_argument(
        "--level",
        required=True,
        choices=["l1_masking", "l1_ig", "l2_shapley", "l3_sae"],
        help="Interpretability level to execute.",
    )
    p_exp.add_argument("explain_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the interpretability module.")
    p_exp.set_defaults(func=_cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
