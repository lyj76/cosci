"""Database-level tests for idle scheduling policy."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from co_scientist import ids
from co_scientist.agents.supervisor import Supervisor
from co_scientist.models import (
    Hypothesis,
    ResearchPlan,
    Session,
    Task,
    TaskResult,
    TournamentMatch,
)
from co_scientist.storage.repos import hypotheses as hyp_repo
from co_scientist.storage.repos import sessions as sess_repo
from co_scientist.storage.repos import tournaments as tourney_repo


def _now() -> datetime:
    return datetime.now(UTC)


async def _session(conn, sid: str = "ses_schedule") -> Session:
    session = Session(
        id=sid,
        created_at=_now(),
        updated_at=_now(),
        status="running",
        research_goal="Find testable scientific hypotheses",
        research_plan=ResearchPlan(objective="Find testable scientific hypotheses"),
        config_snapshot={},
        budget_tokens=100_000,
        budget_usd=10.0,
    )
    await sess_repo.insert(conn, session)
    return session


async def _hypothesis(
    conn,
    session_id: str,
    *,
    index: int,
    matches: int = 0,
    state: str = "in_tournament",
) -> Hypothesis:
    hypothesis = Hypothesis(
        id=f"hyp_schedule_{index}",
        session_id=session_id,
        created_at=_now(),
        created_by="generation",
        strategy="literature",
        title=f"Hypothesis {index}",
        summary=f"Summary {index}",
        full_text=f"Full text {index}",
        artifact_path=f"artifacts/{session_id}/hypotheses/{index}.json",
        elo=1200.0 if state == "in_tournament" else None,
        matches_played=matches,
        state=state,
    )
    await hyp_repo.insert(conn, hypothesis)
    return hypothesis


async def _task_counts(conn, session_id: str) -> dict[tuple[str, str], int]:
    async with conn.execute(
        """SELECT agent, action, COUNT(*) AS n
             FROM tasks WHERE session_id=?
             GROUP BY agent, action""",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return {(row["agent"], row["action"]): row["n"] for row in rows}


@pytest.mark.asyncio
async def test_idle_scheduler_refills_generation_to_configured_batch(conn, tmp_cfg) -> None:
    tmp_cfg.run.max_ideas = 10
    tmp_cfg.run.generation_refill_batch = 4
    session = await _session(conn)

    enqueued = await Supervisor(tmp_cfg)._decide_next_steps(conn, session)

    counts = await _task_counts(conn, session.id)
    assert enqueued == 4
    assert counts[("generation", "CreateInitialHypotheses")] == 4


@pytest.mark.asyncio
async def test_idle_scheduler_respects_generation_attempt_cap(conn, tmp_cfg) -> None:
    tmp_cfg.run.max_ideas = 2
    tmp_cfg.run.generation_refill_batch = 4
    tmp_cfg.run.max_generation_attempts_multiplier = 1
    session = await _session(conn)

    first = await Supervisor(tmp_cfg)._decide_next_steps(conn, session)
    second = await Supervisor(tmp_cfg)._decide_next_steps(conn, session)

    counts = await _task_counts(conn, session.id)
    assert first == 2
    assert second == 0
    assert counts[("generation", "CreateInitialHypotheses")] == 2


@pytest.mark.asyncio
async def test_idle_scheduler_only_ranks_candidates_below_match_cap(conn, tmp_cfg) -> None:
    tmp_cfg.run.max_ideas = 2
    tmp_cfg.run.max_matches_per_idea = 3
    session = await _session(conn)
    await _hypothesis(conn, session.id, index=1, matches=3)
    await _hypothesis(conn, session.id, index=2, matches=2)

    enqueued = await Supervisor(tmp_cfg)._decide_next_steps(conn, session)

    counts = await _task_counts(conn, session.id)
    assert enqueued == 0
    assert ("ranking", "RunTournamentBatch") not in counts


@pytest.mark.asyncio
async def test_idle_scheduler_triggers_evolution_at_configured_threshold(conn, tmp_cfg) -> None:
    tmp_cfg.run.max_ideas = 10
    tmp_cfg.run.generation_refill_batch = 0
    tmp_cfg.run.evolution_min_mature = 2
    tmp_cfg.run.evolution_every_matches = 1
    session = await _session(conn)
    a = await _hypothesis(conn, session.id, index=1, matches=3)
    b = await _hypothesis(conn, session.id, index=2, matches=3)
    await tourney_repo.insert_match(
        conn,
        TournamentMatch(
            id=ids.match_id(a.id, b.id, "seed"),
            session_id=session.id,
            created_at=_now(),
            hyp_a=a.id,
            hyp_b=b.id,
            mode="pairwise",
            winner="a",
            elo_a_before=1200,
            elo_b_before=1200,
            elo_a_after=1216,
            elo_b_after=1184,
            rationale="seed match",
        ),
    )

    enqueued = await Supervisor(tmp_cfg)._decide_next_steps(conn, session)

    counts = await _task_counts(conn, session.id)
    assert enqueued == 2
    assert counts[("ranking", "RunTournamentBatch")] == 1
    assert counts[("evolution", "EvolveTopHypotheses")] == 1


@pytest.mark.asyncio
async def test_refill_scheduler_periodically_uses_debate_generation(conn, tmp_cfg) -> None:
    tmp_cfg.run.max_ideas = 10
    tmp_cfg.run.generation_refill_batch = 1
    tmp_cfg.run.generation_debate_every = 4
    session = await _session(conn)
    await _hypothesis(conn, session.id, index=1, state="draft")
    await _hypothesis(conn, session.id, index=2, state="draft")
    for index in range(4):
        await conn.execute(
            """INSERT INTO tasks(
                   id, session_id, created_at, agent, action, payload,
                   priority, status, attempts, idempotency_key)
               VALUES (?, ?, ?, 'generation', 'CreateInitialHypotheses', '{}',
                       100, 'done', 0, ?)""",
            (
                f"tsk_seed_{index}",
                session.id,
                _now().isoformat(),
                f"{session.id}::seed::{index}",
            ),
        )
    await conn.commit()

    await Supervisor(tmp_cfg)._decide_next_steps(conn, session)

    async with conn.execute(
        """SELECT payload FROM tasks
             WHERE session_id=? AND status='pending' AND agent='generation'""",
        (session.id,),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert '"strategy": "debate"' in row["payload"]


@pytest.mark.asyncio
async def test_full_review_schedules_verification_before_ranking(conn, tmp_cfg) -> None:
    session = await _session(conn)
    hypothesis = await _hypothesis(conn, session.id, index=1, state="reviewed")
    task = Task(
        id="tsk_full_review",
        session_id=session.id,
        created_at=_now(),
        agent="reflection",
        action="ReviewHypothesis",
        target_id=hypothesis.id,
        payload={"kind": "full"},
        status="done",
    )
    result = TaskResult(
        kind="review_completed",
        hypothesis_ids=[hypothesis.id],
        extra={"review_kind": "full"},
    )

    await Supervisor(tmp_cfg)._apply_follow_ups(conn, session, task, result)

    counts = await _task_counts(conn, session.id)
    assert counts[("reflection", "ReviewHypothesis")] == 1
    assert ("ranking", "AddToTournament") not in counts


@pytest.mark.asyncio
async def test_constrained_scheduler_assigns_each_candidate_once(conn, tmp_cfg) -> None:
    tmp_cfg.run.max_ideas = 3
    tmp_cfg.run.generation_refill_batch = 2
    session = await _session(conn)
    session.config_snapshot["search_space_items"] = [
        "Nanvuranlat",
        "KIRA6",
        "Leflunomide",
    ]
    await conn.execute(
        "UPDATE sessions SET config_snapshot=? WHERE id=?",
        (
            '{"search_space_items":["Nanvuranlat","KIRA6","Leflunomide"]}',
            session.id,
        ),
    )
    await conn.commit()

    first = await Supervisor(tmp_cfg)._decide_next_steps(conn, session)
    second = await Supervisor(tmp_cfg)._decide_next_steps(conn, session)

    async with conn.execute(
        """SELECT payload FROM tasks
             WHERE session_id=? AND agent='generation'
             ORDER BY created_at, id""",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()
    payloads = [row["payload"] for row in rows]
    assert first == 2
    assert second == 1
    assert sum("Nanvuranlat" in payload for payload in payloads) == 1
    assert sum("KIRA6" in payload for payload in payloads) == 1
    assert sum("Leflunomide" in payload for payload in payloads) == 1
    assert all('"strategy": "literature"' in payload for payload in payloads)
