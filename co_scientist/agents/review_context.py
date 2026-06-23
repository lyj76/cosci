"""Build evidence-aware review context for downstream agents."""

from __future__ import annotations

from ..storage.artifacts import read_json
from ..storage.repos import reviews as rev_repo
from .base import AgentDeps

_KIND_ORDER = {
    "verification": 0,
    "full": 1,
    "observation": 2,
    "simulation": 3,
}


async def review_bundle(deps: AgentDeps, hypothesis_id: str) -> str | None:
    """Return all reviews, with citation verification status made explicit."""
    reviews = await rev_repo.list_for_hypothesis(deps.db, hypothesis_id)
    if not reviews:
        return None

    sections: list[str] = []
    for review in sorted(
        reviews,
        key=lambda item: (
            _KIND_ORDER.get(item.kind, 99),
            -(item.scores.correctness or 0),
        ),
    ):
        verification_line = "Evidence verification: unavailable."
        try:
            artifact = await read_json(deps.cfg, review.artifact_path)
            record = artifact.get("record") if isinstance(artifact, dict) else None
            summary = (
                record.get("evidence_verification_summary")
                if isinstance(record, dict)
                else None
            )
            if isinstance(summary, dict):
                verification_line = (
                    "Evidence verification: "
                    f"{summary.get('ok', 0)}/{summary.get('total', 0)} verified; "
                    f"{summary.get('unverified', 0)} unverified; "
                    f"{summary.get('fetch_failed', 0)} fetch failed."
                )
        except Exception:
            pass

        sections.append(
            f"### {review.kind} review (verdict={review.verdict or '?'})\n"
            f"{verification_line}\n\n{review.body}"
        )
    return "\n\n---\n\n".join(sections)
