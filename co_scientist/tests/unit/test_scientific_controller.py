from co_scientist.models.scientific_graph import ScientificEdge, ScientificGraph, ScientificNode
from co_scientist.orchestrator.scientific_controller import choose_next_action
from co_scientist.storage.repos import scientific as scientific_repo
from co_scientist.agents.scientific_actions import ScientificActionAgent
from co_scientist.agents.scientific_actions import classify_evidence_relation
from co_scientist.models import Task
from datetime import datetime, UTC
from co_scientist.models.scientific_graph import ScientificEdge


async def _session_id(conn):
    await conn.execute(
        """INSERT INTO sessions
           (id, created_at, updated_at, status, research_goal, research_plan,
            config_snapshot, budget_tokens, budget_usd, budget_used_tokens,
            budget_used_usd, wall_deadline, final_overview)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("sci-test", "2026-01-01", "2026-01-01", "running", "goal", "{}", "{}", 1000, 1.0, 0, 0, None, None),
    )
    await conn.commit()
    return "sci-test"


def test_missing_causal_bridge_prioritizes_retrieval():
    graph = ScientificGraph(
        nodes={
            "a": ScientificNode(id="a", kind="claim", label="A"),
            "b": ScientificNode(id="b", kind="mechanism", label="B"),
        },
        edges=[ScientificEdge(source="a", target="b", relation="bridges")],
    )
    decision = choose_next_action(graph, candidate_count=3, budget_remaining=1.0)
    assert decision.action == "retrieve_missing_evidence"
    assert decision.target_edge == ("a", "b")


def test_grounded_candidates_prioritize_discriminating_experiment():
    graph = ScientificGraph(
        nodes={"m": ScientificNode(id="m", kind="mechanism", label="M")},
        edges=[ScientificEdge(source="e", target="m", relation="supports", evidence_ids=["e"])],
    )
    decision = choose_next_action(graph, candidate_count=2, budget_remaining=1.0)
    assert decision.action == "design_discriminating_experiment"
    assert decision.expected_information_gain > 0


def test_budget_exhaustion_stops_before_graph_actions():
    decision = choose_next_action(ScientificGraph(), candidate_count=10, budget_remaining=0)
    assert decision.action == "stop"


def test_evidence_relation_classifier_is_conservative():
    assert classify_evidence_relation("GD3 engages Siglec-7 and suppresses NK cytotoxicity", "GD3", "Siglec-7") == "supports"
    assert classify_evidence_relation("GD3 and Siglec-7 were measured in OA samples", "GD3", "Siglec-7") == "context_only"
    assert classify_evidence_relation("GD3 had no effect through Siglec-7", "GD3", "Siglec-7") == "contradicts"
    assert classify_evidence_relation("GD3 was elevated in OA", "GD3", "Siglec-7") == "unknown"


async def test_scientific_graph_round_trips_and_decision_is_audited(conn):
    session_id = await _session_id(conn)
    graph = ScientificGraph(
        nodes={"m": ScientificNode(id="m", kind="mechanism", label="M")},
        edges=[ScientificEdge(source="a", target="m", relation="bridges")],
    )
    await scientific_repo.upsert_graph(conn, session_id, graph)
    loaded = await scientific_repo.load_graph(conn, session_id)
    assert len(loaded.unresolved_edges()) == 1
    decision = choose_next_action(loaded, candidate_count=1, budget_remaining=1.0)
    await scientific_repo.record_decision(conn, session_id, decision, graph_coverage=loaded.evidence_coverage())
    async with conn.execute("SELECT action FROM control_decisions WHERE session_id=?", (session_id,)) as cur:
        assert (await cur.fetchone())["action"] == "retrieve_missing_evidence"


async def test_scientific_actions_create_executable_work_items(conn, tmp_cfg):
    session_id = await _session_id(conn)
    from unittest.mock import MagicMock
    deps = MagicMock(db=conn, cfg=tmp_cfg)
    agent = ScientificActionAgent(deps)
    task = Task(id="t-gap", session_id=session_id, created_at=datetime.now(UTC),
                agent="scientific", action="RetrieveEvidenceGap",
                payload={"source_node": "A", "target_node": "B"})
    result = await agent.execute(task)
    assert result.kind == "evidence_gap_queued"
    assert "A B" in result.extra["query"]


async def test_evidence_retrieval_writes_nodes_and_resolves_edge(conn, tmp_cfg):
    tmp_cfg.tools.offline_mode = True
    session_id = await _session_id(conn)
    graph = ScientificGraph(
        nodes={
            "GD3": ScientificNode(id="GD3", kind="claim", label="GD3"),
            "Siglec-7": ScientificNode(id="Siglec-7", kind="mechanism", label="Siglec-7"),
        },
        edges=[ScientificEdge(source="GD3", target="Siglec-7", relation="bridges", unresolved=True)],
    )
    await scientific_repo.upsert_graph(conn, session_id, graph)

    from co_scientist.agents import scientific_actions as actions
    class FakeRetriever:
        async def index_directory(self, path):
            assert path == tmp_cfg.tools.local_corpus_dir
            return 1
        async def query(self, query):
            from co_scientist.rag.paperqa_retriever import EvidenceSnippet, RetrievalResult
            return RetrievalResult(query=query, answer="Evidence", evidence_snippets=[
                EvidenceSnippet("paper.pdf", "paper.pdf pages 3", [3], "GD3 engages Siglec-7 through a receptor", 0.9)
            ])
    old_retriever = actions.PaperQARetriever
    actions.PaperQARetriever = FakeRetriever
    deps = type("Deps", (), {"db": conn, "cfg": tmp_cfg, "tools": object()})()
    agent = ScientificActionAgent(deps)
    task = Task(id="t-resolve", session_id=session_id, created_at=datetime.now(UTC),
                agent="scientific", action="RetrieveEvidenceGap",
                payload={"source_node": "GD3", "target_node": "Siglec-7", "query": "GD3 Siglec-7"})
    try:
        result = await agent.execute(task)
    finally:
        actions.PaperQARetriever = old_retriever
    assert result.kind == "evidence_gap_resolved"
    loaded = await scientific_repo.load_graph(conn, session_id)
    assert loaded.edges[0].unresolved is False
    assert loaded.edges[0].confidence == 0.75
    assert len([n for n in loaded.nodes.values() if n.kind == "evidence"]) == 1
    # Resolving one causal bridge must not be confused with validating the
    # whole hypothesis graph; the remaining coverage gap drives revision.
    assert choose_next_action(loaded, candidate_count=1, budget_remaining=1.0).action == "revise_hypothesis"
