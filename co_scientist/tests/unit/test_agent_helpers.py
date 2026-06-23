"""Tests for agent helper functions that don't require an LLM call."""

from __future__ import annotations

from co_scientist.agents.generation import _filter_to_seen_urls, _render_hypothesis_md
from co_scientist.agents.reflection import _render_review_md
from co_scientist.models import Review
from co_scientist.safety.citation_verifier import summarize_verification


def test_citation_url_filter_keeps_only_seen() -> None:
    citations = [
        {"title": "A", "url": "https://a.example/paper1"},
        {"title": "B", "url": "https://hallucinated.example/paper2"},
        {"title": "C", "url": "https://c.example/paper3"},
        {"no_url": True},
    ]
    seen = {"https://a.example/paper1", "https://c.example/paper3"}
    out = _filter_to_seen_urls(citations, seen)
    urls = {c["url"] for c in out}
    assert urls == seen
    # hallucinated URL is dropped
    assert "https://hallucinated.example/paper2" not in urls


def test_hypothesis_md_renders_sections() -> None:
    md = _render_hypothesis_md(
        {
            "title": "T",
            "statement": "S",
            "mechanism": "M",
            "entities": ["E1", "E2"],
            "anticipated_outcomes": "AO",
            "novelty_argument": "N",
            "citations": [
                {"title": "Paper", "url": "https://example.com/x", "year": 2024}
            ],
        }
    )
    for marker in ("# T", "**Hypothesis.** S", "## Mechanism", "## Entities",
                   "## Anticipated outcomes", "## Novelty", "## Citations",
                   "https://example.com/x"):
        assert marker in md


def test_review_md_renders_sections() -> None:
    md = _render_review_md(
        {
            "verdict": "missing_piece",
            "novelty": 0.7, "correctness": 0.5, "testability": 0.6,
            "assumptions": [
                {"assumption": "A1", "plausibility": "plausible", "rationale": "R1"}
            ],
            "evidence": [
                {"claim": "claim1", "url": "https://e.example/p", "excerpt": "quote"}
            ],
            "notes": "n",
        }
    )
    assert "Verdict" in md
    assert "novelty 0.70" in md
    assert "plausible" in md
    assert "https://e.example/p" in md
    assert "n" in md


def test_verification_verdicts_are_valid_review_values() -> None:
    common = {
        "id": "rev_1",
        "hypothesis_id": "hyp_1",
        "session_id": "ses_1",
        "created_at": "2026-01-01T00:00:00Z",
        "kind": "verification",
        "body": "review",
        "artifact_path": "artifacts/ses_1/reviews/rev_1.json",
    }
    assert Review(**common, verdict="verified").verdict == "verified"
    assert Review(**common, verdict="weak").verdict == "weak"


def test_summarize_verification_counts_and_fraction() -> None:
    summary = summarize_verification({
        "0:https://a": {"status": "ok"},
        "1:https://b": {"status": "unverified"},
        "2:https://c": {"status": "fetch_failed"},
    })
    assert summary == {
        "ok": 1,
        "unverified": 1,
        "fetch_failed": 1,
        "total": 3,
        "verified_fraction": 1 / 3,
    }
