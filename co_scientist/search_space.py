"""Load a bounded candidate universe for constrained research sessions."""

from __future__ import annotations

import csv
from pathlib import Path

_COMMON_HEADERS = {"candidate", "compound", "drug", "entity", "item", "name"}


def load_candidate_universe(path: Path) -> list[str]:
    """Load unique candidates from text, CSV, or TSV while preserving order."""
    if not path.is_file():
        raise ValueError(f"candidate universe file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            values = [
                row[0].strip()
                for row in csv.reader(handle, delimiter=delimiter)
                if row and row[0].strip()
            ]
        if values and values[0].lower() in _COMMON_HEADERS:
            values = values[1:]
    else:
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    unique = list(dict.fromkeys(values))
    if not unique:
        raise ValueError(f"candidate universe is empty: {path}")
    return unique
