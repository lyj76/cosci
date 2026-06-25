"""Reflection agent — reviews and verifies a hypothesis."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .. import ids
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.prompts import render
from ..llm.routing import route
from ..llm.tool_loop import ToolLoopExhausted, run_tool_loop
from ..models import Review, ReviewScores, Task, TaskResult
from ..safety.citation_verifier import CitationVerifier, summarize_verification
from ..safety.quoting import quote_hypothesis
from ..storage.artifacts import write_json
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from .base import BaseAgent
from .schemas import RECORD_REVIEW_TOOL


class ReflectionAgent(BaseAgent):
    name = "reflection"

    async def execute(self, task: Task) -> TaskResult:
        kind = task.payload.get("kind", "full")
        hypothesis_id = task.target_id
        if not hypothesis_id:
            raise ValueError("ReflectionAgent.execute requires target_id (hypothesis_id)")

        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")
        h = await hyp_repo.fetch(self.deps.db, hypothesis_id)
        if h is None:
            raise RuntimeError(f"hypothesis {hypothesis_id} missing")

        if kind not in ("full", "verification"):
            raise NotImplementedError(f"reflection kind {kind!r} lands in a later milestone")

        hypothesis_text = quote_hypothesis(h.full_text, id_=h.id)
        if kind == "full":
            prompt = render(
                "reflection.full",
                goal=session.research_plan.objective,
                preferences="; ".join(session.research_plan.preferences),
                hypothesis_id=h.id,
                hypothesis_text=hypothesis_text,
                articles_block=(
                    "Use the available search tools (web_search, pubmed_search, "
                    "arxiv_search, europe_pmc_search, web_fetch) to gather supporting "
                    "and contradicting evidence. Cite URLs that you actually fetched."
                ),
            )
        else:
            prompt = render(
                "reflection.verification",
                goal=session.research_plan.objective,
                hypothesis_id=h.id,
                hypothesis_text=hypothesis_text,
            )

        sys_blocks = [
            CachedBlock(self._system_prompt_header(), cache=True),
            CachedBlock(
                f"# Research goal\n{session.research_goal}\n\n"
                f"# Preferences\n{'; '.join(session.research_plan.preferences)}"
                + await _review_feedback_context(self.deps, session.id),
                cache=True,
            ),
        ]
        user_blocks = [CachedBlock(prompt, cache=False)]

        r = route(self.deps.cfg, "reflection", kind)
        tools = [*self.deps.tools.anthropic_tools_for("reflection"), RECORD_REVIEW_TOOL]

        spec = AgentCallSpec(
            route=r,
            system_blocks=sys_blocks,
            user_blocks=user_blocks,
            tools=tools,
            tool_choice={"type": "auto"},
            max_output_tokens=4096,
        )
        ctx = CallContext(
            session_id=task.session_id, task_id=task.id,
            agent="reflection", action="ReviewHypothesis", mode=kind,
        )

        try:
            loop_result = await run_tool_loop(
                self.deps.llm,
                spec=spec, ctx=ctx,
                registry=self.deps.tools,
                max_iters=self.deps.cfg.tool_loop.reflection_max_iters,
                parallel_cap=self.deps.cfg.tool_loop.parallel_cap,
                tool_timeout_s=self.deps.cfg.tool_loop.tool_timeout_seconds,
                force_terminal_tool="record_review",
            )
        except ToolLoopExhausted as e:
            raise RuntimeError(f"reflection exhausted tool loop: {e}") from e

        record = self._final_tool_use(loop_result.response, "record_review")
        if record is None:
            raise RuntimeError("Reflection did not call record_review")
        _normalize_review_record(record)

        # Drop evidence entries whose URL we never saw — keep the review honest.
        seen = loop_result.seen_urls
        record["evidence"] = [
            e for e in record.get("evidence", [])
            if isinstance(e, dict) and e.get("url") in seen
        ]

        review_id = ids.review_id(h.id, kind, iteration=0)
        verification_results = await CitationVerifier(
            self.deps.cfg
        ).verify_evidence(
            session_id=session.id,
            review_id=review_id,
            evidence=record["evidence"],
            db=self.deps.db,
        )
        record["evidence_verification"] = verification_results
        record["evidence_verification_summary"] = summarize_verification(
            verification_results
        )
        for index, evidence in enumerate(record["evidence"]):
            result = verification_results.get(f"{index}:{evidence.get('url')}")
            evidence["verification_status"] = (
                result.get("status") if result else "not_checked"
            )

        artifact_path = await write_json(
            self.deps.cfg, session.id, "reviews", review_id,
            {"hypothesis_id": h.id, "record": record},
        )
        body_md = _render_review_md(record)
        review = Review(
            id=review_id,
            hypothesis_id=h.id,
            session_id=session.id,
            created_at=datetime.now(UTC),
            kind=kind,
            verdict=record.get("verdict"),       # type: ignore[arg-type]
            scores=ReviewScores(
                novelty=record.get("novelty"),
                correctness=record.get("correctness"),
                testability=record.get("testability"),
                feasibility=record.get("feasibility"),
            ),
            assumptions=record.get("assumptions", []),
            evidence=record.get("evidence", []),
            body=body_md,
            artifact_path=artifact_path,
        )
        await rev_repo.insert(self.deps.db, review)
        # Only promote draft → reviewed. If Reflection re-fires on an
        # already-ranked/evolved/pinned hypothesis we must not drag it back.
        if kind == "full":
            await hyp_repo.set_state_if(
                self.deps.db, h.id, new_state="reviewed", expected_states=("draft",),
            )

        return TaskResult(
            kind="review_completed",
            review_ids=[review_id],
            hypothesis_ids=[h.id],
            extra={"verdict": record.get("verdict"), "review_kind": kind},
        )


_ASSUMPTION_PLAUSIBILITY = {"plausible", "uncertain", "implausible"}
_PLAUSIBILITY_ALIASES = {
    "verified": "plausible",
    "supported": "plausible",
    "support": "plausible",
    "likely": "plausible",
    "true": "plausible",
    "ok": "plausible",
    "unverified": "uncertain",
    "unknown": "uncertain",
    "unclear": "uncertain",
    "mixed": "uncertain",
    "ambiguous": "uncertain",
    "not_checked": "uncertain",
    "not checked": "uncertain",
    "weak": "uncertain",
    "unsupported": "implausible",
    "contradicted": "implausible",
    "false": "implausible",
    "disproved": "implausible",
    "unlikely": "implausible",
}


def _normalize_review_record(record: dict[str, Any]) -> None:
    """Normalize model tool-call drift before strict Review validation.

    OpenAI-compatible providers may not enforce JSON-schema enums strictly.
    Keep the stored Review model canonical while preserving the model's raw
    value in the JSON artifact for audit.
    """
    assumptions = record.get("assumptions")
    if not isinstance(assumptions, list):
        record["assumptions"] = []
        return
    normalized: list[dict[str, Any]] = []
    for assumption in assumptions:
        if not isinstance(assumption, dict):
            continue
        raw = assumption.get("plausibility")
        key = str(raw or "").strip().lower()
        value = key if key in _ASSUMPTION_PLAUSIBILITY else _PLAUSIBILITY_ALIASES.get(key)
        if value is None:
            value = "uncertain"
        if raw != value:
            assumption["original_plausibility"] = raw
            assumption["plausibility"] = value
        normalized.append(assumption)
    record["assumptions"] = normalized


def _render_review_md(record: dict[str, Any]) -> str:
    parts: list[str] = ["# Review"]
    if record.get("verdict"):
        parts.append(f"**Verdict.** {record['verdict']}")
    scores = []
    for s in ("novelty", "correctness", "testability", "feasibility"):
        if record.get(s) is not None:
            scores.append(f"{s} {record[s]:.2f}")
    if scores:
        parts.append("**Scores.** " + " · ".join(scores))
    if record.get("assumptions"):
        parts.append("## Assumptions")
        for a in record["assumptions"]:
            parts.append(
                f"- *{a.get('plausibility','?')}*: {a.get('assumption','')}\n  "
                f"  {a.get('rationale','')}"
            )
    if record.get("evidence"):
        parts.append("## Evidence")
        for e in record["evidence"]:
            status = e.get("verification_status", "not_checked")
            parts.append(
                f"- [{status}] {e.get('claim','')} — {e.get('url','')}\n"
                f"  > {e.get('excerpt','')}"
            )
    summary = record.get("evidence_verification_summary")
    if summary:
        parts.append(
            "## Evidence verification\n"
            f"Verified {summary.get('ok', 0)}/{summary.get('total', 0)}; "
            f"unverified {summary.get('unverified', 0)}; "
            f"fetch failed {summary.get('fetch_failed', 0)}."
        )
    if record.get("notes"):
        parts.append(f"## Notes\n{record['notes']}")
    return "\n\n".join(parts)


async def _review_feedback_context(deps, session_id: str) -> str:
    feedback = await fb_repo.latest_system_feedback(deps.db, session_id)
    if feedback is None:
        return ""
    return (
        "\n\n# Meta-review guidance for this review\n"
        + feedback.text[:8000]
    )
