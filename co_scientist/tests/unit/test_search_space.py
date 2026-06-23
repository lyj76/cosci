"""Tests for constrained candidate-universe loading."""

from __future__ import annotations

from co_scientist.search_space import load_candidate_universe


def test_load_candidate_universe_text_deduplicates_and_skips_comments(tmp_path) -> None:
    path = tmp_path / "drugs.txt"
    path.write_text(
        "# approved drugs\nNanvuranlat\nKIRA6\nNanvuranlat\nLeflunomide\n",
        encoding="utf-8",
    )

    assert load_candidate_universe(path) == [
        "Nanvuranlat",
        "KIRA6",
        "Leflunomide",
    ]


def test_load_candidate_universe_csv_uses_first_column_and_skips_header(tmp_path) -> None:
    path = tmp_path / "drugs.csv"
    path.write_text(
        "drug,approved_indication\nKIRA6,research compound\nLeflunomide,RA\n",
        encoding="utf-8",
    )

    assert load_candidate_universe(path) == ["KIRA6", "Leflunomide"]
