"""Persistence for the evidence/mechanism/experiment control graph."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import aiosqlite

from ...models.scientific_graph import ScientificEdge, ScientificGraph, ScientificNode
from ...orchestrator.scientific_controller import ControlDecision


async def upsert_graph(
    conn: aiosqlite.Connection, session_id: str, graph: ScientificGraph
) -> None:
    for node in graph.nodes.values():
        await conn.execute(
            """INSERT INTO scientific_nodes
               (id, session_id, kind, label, description, confidence, status, metadata)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, label=excluded.label,
                 description=excluded.description, confidence=excluded.confidence,
                 status=excluded.status, metadata=excluded.metadata""",
            (node.id, session_id, node.kind, node.label, node.description,
             node.confidence, node.status, json.dumps(node.metadata)),
        )
    for edge in graph.edges:
        await conn.execute(
            """INSERT INTO scientific_edges
               (session_id, source, target, relation, evidence_ids, confidence, unresolved)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(session_id, source, target, relation) DO UPDATE SET
                 evidence_ids=excluded.evidence_ids, confidence=excluded.confidence,
                 unresolved=excluded.unresolved""",
            (session_id, edge.source, edge.target, edge.relation,
             json.dumps(edge.evidence_ids), edge.confidence, int(edge.unresolved)),
        )
    await conn.commit()


async def load_graph(conn: aiosqlite.Connection, session_id: str) -> ScientificGraph:
    async with conn.execute("SELECT * FROM scientific_nodes WHERE session_id=?", (session_id,)) as cur:
        nodes = [dict(row) for row in await cur.fetchall()]
    async with conn.execute("SELECT * FROM scientific_edges WHERE session_id=?", (session_id,)) as cur:
        edges = [dict(row) for row in await cur.fetchall()]
    return ScientificGraph(
        nodes={row["id"]: ScientificNode(
            id=row["id"], kind=row["kind"], label=row["label"],
            description=row["description"], confidence=row["confidence"],
            status=row["status"], metadata=json.loads(row["metadata"]),
        ) for row in nodes},
        edges=[ScientificEdge(
            source=row["source"], target=row["target"], relation=row["relation"],
            evidence_ids=json.loads(row["evidence_ids"]), confidence=row["confidence"],
            unresolved=bool(row["unresolved"]),
        ) for row in edges],
    )


async def record_decision(
    conn: aiosqlite.Connection, session_id: str, decision: ControlDecision,
    *, graph_coverage: float, budget_remaining: float | None = None,
) -> int:
    cur = await conn.execute(
        """INSERT INTO control_decisions
           (session_id, created_at, action, reason, target_edge,
            expected_information_gain, graph_coverage, budget_remaining)
           VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, datetime.now(UTC).isoformat(), decision.action, decision.reason,
         json.dumps(decision.target_edge), decision.expected_information_gain,
         graph_coverage, budget_remaining),
    )
    await conn.commit()
    return int(cur.lastrowid)
