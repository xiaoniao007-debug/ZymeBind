"""Lightweight unit and smoke tests for ZymeBind CLI, I/O, and architecture."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from zymebind import __version__
from zymebind.cli import build_parser
from zymebind.data.fasta_io import read_3di_map, read_fasta, write_tsv_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FASTA = REPO_ROOT / "examples" / "sample_enzymes.fasta"


def test_version_and_cli_parser():
    assert __version__ == "1.0.0"
    parser = build_parser()
    args = parser.parse_args(["info"])
    assert args.command == "info"

    pred_args = parser.parse_args(
        [
            "predict",
            "--fasta",
            str(SAMPLE_FASTA),
            "--ckpt",
            "dummy.pt",
            "--mode",
            "sequence",
            "--threshold",
            "0.62",
            "--topk",
            "5",
        ]
    )
    assert pred_args.command == "predict"
    assert pred_args.threshold == 0.62
    assert pred_args.topk == 5


def test_cli_info_subprocess():
    res = subprocess.run(
        [sys.executable, "-m", "zymebind.cli", "info"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ZymeBind v1.0.0" in res.stdout
    assert "Chen Chen" in res.stdout


def test_read_sample_fasta_and_write_tsv(tmp_path: Path):
    records = read_fasta(SAMPLE_FASTA)
    assert len(records) == 3
    assert records[0].protein_id.startswith("P62593")
    assert len(records[0].sequence) == 286

    tsv_out = tmp_path / "preds.tsv"
    write_tsv_predictions(
        [
            {
                "protein_id": records[0].protein_id,
                "rank": 1,
                "ec_number": "3.5.2.6",
                "probability": "0.9912",
                "above_threshold": "true",
                "active_modalities": "sequence",
            }
        ],
        tsv_out,
    )
    lines = tsv_out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "3.5.2.6" in lines[1]


def test_read_3di_map(tmp_path: Path):
    p = tmp_path / "sample.3di.tsv"
    p.write_text("protein_id\tthreedi\nP62593\tddddvvvvvvvv\n", encoding="utf-8")
    m = read_3di_map(p)
    assert m["P62593"] == "ddddvvvvvvvv"
