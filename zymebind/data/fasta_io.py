"""FASTA and optional auxiliary modality I/O utilities for ZymeBind CLI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FastaRecord:
    """Single protein record parsed from a FASTA file."""

    protein_id: str
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    """Read protein sequences from a FASTA file.

    Empty lines are ignored and whitespace within sequences is stripped.
    Raises ValueError if the file contains no valid sequences.
    """
    fasta_path = Path(path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records: list[FastaRecord] = []
    header: str | None = None
    seq_chunks: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(_make_record(header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append("".join(line.split()).upper())

    if header is not None:
        records.append(_make_record(header, "".join(seq_chunks)))

    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return records


def _make_record(header: str, sequence: str) -> FastaRecord:
    parts = header.split(None, 1)
    protein_id = parts[0] if parts else "unknown"
    description = parts[1] if len(parts) > 1 else ""
    if not sequence:
        raise ValueError(f"Empty sequence for FASTA entry '{protein_id}'")
    return FastaRecord(protein_id=protein_id, description=description, sequence=sequence)


def read_3di_map(path: str | Path) -> dict[str, str]:
    """Read optional Foldseek 3Di structural sequences from a FASTA or TSV file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"3Di file not found: {p}")

    first_nonempty = ""
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                first_nonempty = s
                break

    if first_nonempty.startswith(">"):
        return {rec.protein_id: rec.sequence for rec in read_fasta(p)}

    mapping: dict[str, str] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            cols = s.split("\t")
            if len(cols) >= 2 and cols[0].lower() not in {"protein_id", "id", "entry"}:
                mapping[cols[0]] = cols[1].strip()
    return mapping


def write_tsv_predictions(
    rows: Iterable[dict[str, object]],
    out_path: str | Path,
) -> Path:
    """Write EC predictions to a tab-separated file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "protein_id",
        "rank",
        "ec_number",
        "probability",
        "above_threshold",
        "active_modalities",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in header) + "\n")
    return path
