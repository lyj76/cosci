"""Reconstruct a paper-style approved-drug universe from Open Targets exports.

This script is intentionally conservative:
- It reads local Open Targets 25.03 `drug_molecule/` and `known_drug/`
  parquet directories.
- It keeps only molecules that look approved by either phase/status fields in
  `drug_molecule` or approval-like rows in `known_drug`.
- It writes a deduplicated list of candidate names suitable for
  `--candidate-universe`.

Because Open Targets release schemas may change, the script prints the observed
schemas and exits with a clear error if it cannot find the expected id/name
columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ID_KEYS = (
    "molecule_chembl_id",
    "chembl_id",
    "molecule_id",
    "drug_id",
    "drugId",
    "id",
)
NAME_KEYS = (
    "pref_name",
    "molecule_name",
    "drug_name",
    "name",
    "label",
)
PHASE_KEYS = (
    "max_phase",
    "phase",
    "highest_phase",
    "clinical_phase",
    "max_phase_for_indication",
)
STATUS_KEYS = (
    "status",
    "clinical_status",
    "approval_status",
    "drug_status",
    "is_approved",
    "approved",
    "approved_indication",
)


@dataclass(frozen=True)
class UniverseRow:
    candidate: str
    source_id: str | None = None
    source_name: str | None = None
    source: str = "drug_molecule"


def _import_pyarrow():
    try:
        import pyarrow.dataset as ds  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guidance only
        raise SystemExit(
            "pyarrow is required for this script. Install it with `pip install -e '.[data]'`."
        ) from exc
    return ds


def _ensure_dataset_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"{label} directory does not exist: {path}\n"
            f"Expected an Open Targets 25.03 parquet export directory such as "
            f"`.../output/{label}`."
        )
    if not path.is_dir():
        raise SystemExit(f"{label} path is not a directory: {path}")


def _dataset_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    ds = _import_pyarrow()
    dataset = ds.dataset(str(path), format="parquet")
    schema_fields = [field.name for field in dataset.schema]
    rows = dataset.to_table().to_pylist()
    return rows, schema_fields


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _string_value(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first_value(row, keys)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def _is_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "approved", "approved_indication", "marketed"}


def _is_approved_like(row: dict[str, Any]) -> bool:
    phase = _first_value(row, PHASE_KEYS)
    if isinstance(phase, (int, float)) and phase >= 4:
        return True
    if isinstance(phase, str):
        try:
            if float(phase) >= 4:
                return True
        except ValueError:
            if "approved" in phase.lower() or "market" in phase.lower():
                return True
    for key in STATUS_KEYS:
        if key in row and _is_truthy(row[key]):
            return True
        if key in row and isinstance(row[key], str):
            txt = row[key].strip().lower()
            if "approved" in txt or "market" in txt:
                return True
    return False


def _canonical_candidate(row: dict[str, Any]) -> UniverseRow | None:
    candidate_id = _string_value(row, ID_KEYS)
    candidate_name = _string_value(row, NAME_KEYS)
    if candidate_name is None:
        return None
    return UniverseRow(
        candidate=candidate_name,
        source_id=candidate_id,
        source_name=candidate_name,
    )


def build_universe(
    drug_molecule_dir: Path,
    known_drug_dir: Path | None = None,
) -> tuple[list[UniverseRow], dict[str, list[str]]]:
    """Return a deduplicated candidate universe and debug schema info."""
    _ensure_dataset_dir(drug_molecule_dir, "drug_molecule")
    drug_rows, drug_schema = _dataset_rows(drug_molecule_dir)
    known_rows: list[dict[str, Any]] = []
    known_schema: list[str] = []
    if known_drug_dir is not None:
        _ensure_dataset_dir(known_drug_dir, "known_drug")
        known_rows, known_schema = _dataset_rows(known_drug_dir)

    approved_ids: set[str] = set()
    for row in known_rows:
        candidate_id = _string_value(row, ID_KEYS)
        if candidate_id and _is_approved_like(row):
            approved_ids.add(candidate_id)

    universe: list[UniverseRow] = []
    seen: set[str] = set()
    for row in drug_rows:
        entry = _canonical_candidate(row)
        if entry is None:
            continue

        candidate_id = entry.source_id
        keep = _is_approved_like(row)
        if approved_ids:
            keep = keep or (candidate_id is not None and candidate_id in approved_ids)
        if not keep:
            continue

        key = entry.candidate.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        universe.append(entry)

    schemas = {
        "drug_molecule": drug_schema,
        "known_drug": known_schema,
    }
    return universe, schemas


def _write_txt(path: Path, rows: list[UniverseRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.candidate + "\n")


def _write_csv(path: Path, rows: list[UniverseRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "source_id", "source_name", "source"])
        for row in rows:
            writer.writerow([row.candidate, row.source_id or "", row.source_name or "", row.source])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-molecule-dir", type=Path, required=True)
    parser.add_argument("--known-drug-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format", choices=("txt", "csv"), default="txt")
    parser.add_argument("--dump-schema", action="store_true")
    args = parser.parse_args()

    rows, schemas = build_universe(args.drug_molecule_dir, args.known_drug_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "txt":
        _write_txt(args.out, rows)
    else:
        _write_csv(args.out, rows)

    print(
        json.dumps(
            {
                "n_candidates": len(rows),
                "out": str(args.out),
                "drug_molecule_fields": schemas["drug_molecule"],
                "known_drug_fields": schemas["known_drug"],
            },
            indent=2,
        )
    )
    if args.dump_schema:
        print("\n# drug_molecule schema")
        print("\n".join(schemas["drug_molecule"]))
        print("\n# known_drug schema")
        print("\n".join(schemas["known_drug"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
