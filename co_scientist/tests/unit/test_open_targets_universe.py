"""Tests for Open Targets candidate-universe reconstruction helpers."""

from __future__ import annotations

from scripts.build_open_targets_aml_universe import (
    _canonical_candidate,
    _is_approved_like,
)


def test_is_approved_like_matches_common_phase_and_status_fields() -> None:
    assert _is_approved_like({"max_phase": 4})
    assert _is_approved_like({"phase": "4"})
    assert _is_approved_like({"status": "approved"})
    assert _is_approved_like({"approved": True})
    assert not _is_approved_like({"max_phase": 2})


def test_canonical_candidate_prefers_name_field() -> None:
    row = {"molecule_chembl_id": "CHEMBL1", "pref_name": "Leflunomide"}
    out = _canonical_candidate(row)
    assert out is not None
    assert out.candidate == "Leflunomide"
    assert out.source_id == "CHEMBL1"
