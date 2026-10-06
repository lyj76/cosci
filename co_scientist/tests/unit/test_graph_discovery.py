"""Tests of real orchestration paths with model and retrieval boundaries stubbed."""
import json
import socket
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from co_scientist.agents.evidence_extractor import ExtractedEvidenceBundle, GroundedClaim
from co_scientist.agents.mechanism_generator import MechanisticHypothesis, MechanismStep
from co_scientist.agents.pi_synthesizer import FinalSynthesizedHypothesis
from co_scientist.agents.skeptic import EvidenceAuditItem, SkepticCritique
from co_scientist.orchestrator import graph_discovery as gd
from co_scientist.orchestrator import oa_pipeline as pipeline
from co_scientist.rag.paperqa_retriever import EvidenceSnippet, RetrievalResult


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("offline orchestration tests must not open sockets")
    monkeypatch.setattr(socket.socket, "connect", reject)


def hypothesis(title="h"):
    return MechanisticHypothesis(title, "summary", ["NK"], "GD3", "Siglec-7", "test",
        [MechanismStep(1, "GD3", "NK clearance", "suppresses",
                       "GD3 suppresses NK clearance in OA cartilage", ["GD3 is elevated"])],
        "unknown", "", "")


def evidence():
    return ExtractedEvidenceBundle("OA", "summary", claims=[
        GroundedClaim("GD3 is elevated", "paper", [1], "GD3 is elevated")],
        caveats_and_contradictions=[GroundedClaim("No cartilage rescue", "paper", [2], "No cartilage rescue")])


def review():
    return SkepticCritique("h", "MAJOR_REVISIONS_NEEDED", "OA-specific causality missing",
        audited_links=[EvidenceAuditItem("Step 1", "unsupported_leap", "no causality", "")],
        unsupported_core_claim=True)


@pytest.mark.parametrize("relation,scope,quote,resolved", [
    ("supports", True, "GD3 suppresses NK clearance", True),
    ("supports", False, "GD3 suppresses NK clearance", False),
    ("supports", True, "invented quotation", False),
    ("entity_only", True, "GD3 suppresses NK clearance", False),
    ("contradicts", True, "GD3 suppresses NK clearance", False),
])
def test_scope_and_quote_gate(relation, scope, quote, resolved):
    graph = gd.build_graph(evidence(), [("a", hypothesis())])
    edge = graph.edges[0]
    assert edge.unresolved  # Generator's exact matching citation is not proof.
    snippets = [EvidenceSnippet("p", "p pages 1", [1], "GD3 suppresses NK clearance", 10)]
    records = gd.apply_attributions(graph, edge, snippets, gd.Attributions(items=[
        gd.Attribution(snippet_index=0, relation=relation, scope_matches=scope,
                       quote=quote, rationale="test")]))
    assert (not edge.unresolved) is resolved
    assert graph.nodes[records[0]["evidence_id"]].confidence == 0  # Score 10 isn't probability.
    assert any(n.metadata.get("origin") == "caveat" for n in graph.nodes.values())


def test_conflicting_passages_never_close_gap():
    graph = gd.build_graph(evidence(), [("a", hypothesis())])
    edge = graph.edges[0]
    snippets = [EvidenceSnippet("p", "p pages 1", [1], "effect present"),
                EvidenceSnippet("p", "p pages 2", [2], "effect absent")]
    gd.apply_attributions(graph, edge, snippets, gd.Attributions(items=[
        gd.Attribution(snippet_index=i, relation=r, scope_matches=True, quote=s.text, rationale="test")
        for i, (r, s) in enumerate(zip(["supports", "contradicts"], snippets))]))
    assert edge.unresolved
    assert graph.nodes[edge.target].status == "contradicted"


async def test_no_hit_queries_are_not_repeated_and_resume_is_free():
    graph = gd.build_graph(evidence(), [("a", hypothesis()), ("b", hypothesis())])
    retriever = SimpleNamespace(query=AsyncMock(return_value=RetrievalResult("q", "no evidence")))
    memory = {}
    def cache(action, key, value):
        if action == "save":
            memory[key] = value
        return memory.get(key)
    trace, stop = await gd.run_evidence_loop(graph, retriever, None, "OA",
        [("a", review()), ("b", review())], 2, cache)
    assert retriever.query.await_count == 1  # identical claims share retrieval cache
    assert len({r["target"] for r in trace}) == 2
    assert len(graph.unresolved_edges()) == 2
    assert stop == "query_budget_exhausted"
    await gd.run_evidence_loop(graph, retriever, None, "OA", [], 2, cache, trace)
    assert retriever.query.await_count == 1


def test_revision_does_not_reward_verbosity_or_accept_regression():
    old = review()
    assert not pipeline._review_improved(old, replace(old, overall_skeptical_assessment="much more prose"))
    assert not pipeline._review_improved(old, replace(old, verdict="FATAL_FLAW"))
    assert pipeline._review_improved(old, replace(old, unsupported_core_claim=False))


async def test_pipeline_graph_feedback_and_resume_after_pi_failure(tmp_path, monkeypatch):
    pdf = tmp_path / "fixed.pdf"
    pdf.write_bytes(b"mock PDF; boundary stubbed")
    output = tmp_path / "run"
    retrieval = SimpleNamespace(index_file=AsyncMock(return_value=True),
        query=AsyncMock(return_value=RetrievalResult("q", "answer", [
            EvidenceSnippet("p", "p pages 1", [1], "GD3 is elevated", 10)])))
    extractor = SimpleNamespace(extract_evidence=AsyncMock(return_value=evidence()))
    generator = SimpleNamespace(propose_strategies=AsyncMock(return_value=[("a", "a"), ("b", "b")]),
        generate_hypothesis=AsyncMock(return_value=hypothesis()),
        evolve_hypothesis=AsyncMock(return_value=hypothesis("revised")))
    skeptic = SimpleNamespace(critique=AsyncMock(return_value=review()))
    final = FinalSynthesizedHypothesis("proposal", "summary", "hypothetical", "rationale", {}, [],
        [{"paper": "p", "pages": [1], "quote": "GD3 is elevated"}],
        [{"skeptic_concern": "unknown", "resolution_strategy": "test first"}],
        [{"test": "depletion", "falsifying_result": "no change"}], [], "unknown")
    pi = SimpleNamespace(synthesize=AsyncMock(side_effect=[final, RuntimeError("disconnect")]))
    for name, obj in [("PaperQARetriever", retrieval), ("EvidenceExtractorAgent", extractor),
                      ("MechanismGeneratorAgent", generator), ("SkepticAgent", skeptic), ("PIAgent", pi)]:
        monkeypatch.setattr(pipeline, name, lambda *a, obj=obj: obj)
    async def model_call(agent, prompt, schema, **kwargs):
        if schema is gd.Attributions:
            return gd.Attributions(items=[gd.Attribution(snippet_index=0, relation="context_only",
                scope_matches=False, quote="GD3 is elevated", rationale="correlation only")])
        return gd.Experiment(candidate_ids=["a", "b"], distinction="immune versus direct action",
            intervention="effector depletion", controls=["isotype"],
            predictions={"a": "efficacy lost", "b": "efficacy retained"},
            falsification_rule="efficacy retained refutes immune necessity",
            next_decisions={"retained": "test direct mechanism"})
    monkeypatch.setattr(gd, "structured_call", model_call)
    from co_scientist.storage import exporter
    monkeypatch.setattr(exporter, "persist_as_coscientist_session", AsyncMock(return_value="fake-session"))
    args = dict(paper_dir=tmp_path, paper_files=[pdf.name], output_dir=output,
                max_evolution_rounds=1, graph_evidence_queries=1, parallelism=2)
    with pytest.raises(RuntimeError, match="disconnect"):
        await pipeline.run_oa_pipeline(**args)
    assert pi.synthesize.call_args_list[0].kwargs["evidence"].graph_feedback["discriminating_experiment"]
    assert generator.evolve_hypothesis.call_args_list[0].kwargs["evidence"].graph_feedback["audit"]
    assert any(c.kwargs["evidence"].graph_feedback for c in skeptic.critique.call_args_list)
    calls = (retrieval.query.await_count, generator.evolve_hypothesis.await_count, skeptic.critique.await_count)
    pi.synthesize.side_effect = None
    pi.synthesize.return_value = final
    report = await pipeline.run_oa_pipeline(**args, resume=True)
    assert calls == (retrieval.query.await_count, generator.evolve_hypothesis.await_count, skeptic.critique.await_count)
    assert retrieval.index_file.await_count == 1
    assert pi.synthesize.await_count == 3  # only unfinished branch re-executed
    assert report["graph_control_trace"][0]["attributions"][0]["relation"] == "context_only"
    assert report["evolution_trace"][0]["decisions"][0]["action"] == "retain_parent"
    assert report["discriminating_experiment"]["candidate_ids"] == ["a", "b"]
    assert report["ranking_mode"] == "heuristic_score"
    assert json.loads((output / "pipeline_heartbeat.json").read_text())["phase"] == "pipeline"
    pdf.write_bytes(b"changed")
    with pytest.raises(ValueError, match="signature differs"):
        await pipeline.run_oa_pipeline(**args, resume=True)


def completion(content, finish="stop", reasoning=None, refusal=None):
    return SimpleNamespace(id="response-test", usage=None, choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(
            content=content, reasoning_content=reasoning, refusal=refusal))])


def response_agent(responses):
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = MagicMock(return_value=client)
    return SimpleNamespace(client=client, model="configured-model"), create


VALID_ATTRIBUTIONS = json.dumps({"items": [{"snippet_index": 0, "relation": "unknown",
    "scope_matches": False, "quote": "", "rationale": "scope is not established"}]})


@pytest.mark.parametrize("first,error", [
    (completion(None, "length", "reasoning only"), "output_truncated"),
    (completion("{}"), "schema_validation_failed"),
    (completion(None), "empty_content"),
    (completion('{"items":['), "schema_validation_failed"),
    (completion(VALID_ATTRIBUTIONS, "length"), "output_truncated"),
])
async def test_structured_retry_records_actual_response(first, error):
    agent, create = response_agent([first, completion(VALID_ATTRIBUTIONS)])
    records = []
    result = await gd.structured_call(agent, "prompt", gd.Attributions, on_attempt=records.append)
    assert result.items[0].relation == "unknown"
    assert create.await_count == 2
    assert records[0]["content"] == first.choices[0].message.content
    assert records[0]["error"] == error
    assert records[1]["status"] == "valid"
    assert create.call_args_list[0].kwargs["max_tokens"] == gd._DEFAULT_STRUCTURED_BUDGET
    assert create.call_args_list[1].kwargs["max_tokens"] == (
        gd._RETRY_STRUCTURED_BUDGET if error in {"empty_content", "output_truncated"}
        else gd._DEFAULT_STRUCTURED_BUDGET)
    agent.client.with_options.assert_called_once_with(max_retries=0)


async def test_persistent_invalid_output_fails_closed_after_three_calls():
    agent, create = response_agent([completion("{}"), completion("{}"), completion("{}")])
    records = []
    with pytest.raises(gd.StructuredOutputError, match="attempts=3"):
        await gd.structured_call(agent, "prompt", gd.Attributions, on_attempt=records.append)
    assert create.await_count == len(records) == 3
    assert all(r["status"] == "invalid" for r in records)
    assert records[0]["validation_errors"] == [{"loc": ["items"], "type": "missing"}]


def completion_with_usage(completion_resp, reasoning_tokens=0):
    details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    completion_resp.usage = SimpleNamespace(
        completion_tokens_details=details,
        model_dump=lambda *a, **k: {"completion_tokens_details": {"reasoning_tokens": reasoning_tokens}})
    return completion_resp


async def test_reasoning_tokens_detect_reasoning_model_and_grow_budget():
    first = completion_with_usage(completion('{"items":['), reasoning_tokens=11551)
    agent, create = response_agent([first, completion(VALID_ATTRIBUTIONS)])
    records = []
    result = await gd.structured_call(agent, "prompt", gd.Attributions, on_attempt=records.append)
    assert result.items[0].relation == "unknown"
    assert create.call_args_list[1].kwargs["max_tokens"] == gd._RETRY_STRUCTURED_BUDGET
    assert records[0]["reasoning_tokens"] == 11551  # raw response preserved in diagnostics


async def test_max_tokens_rejection_falls_back_to_default_budget():
    # Attempt 1 truncates (grows budget to 24576), attempt 2 is rejected by the
    # gateway for exceeding its cap, attempt 3 retries at the default budget.
    create = AsyncMock(side_effect=[
        completion(None, "length", "reasoning content"),
        RuntimeError("maximum context length is 16384 tokens; you requested 24576"),
        completion(VALID_ATTRIBUTIONS)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = MagicMock(return_value=client)
    agent = SimpleNamespace(client=client, model="configured-model")
    records = []
    result = await gd.structured_call(agent, "prompt", gd.Attributions, on_attempt=records.append)
    assert result.items[0].relation == "unknown"
    assert create.await_count == 3
    assert records[1]["status"] == "request_failed"
    assert create.call_args_list[0].kwargs["max_tokens"] == gd._DEFAULT_STRUCTURED_BUDGET
    assert create.call_args_list[1].kwargs["max_tokens"] == gd._RETRY_STRUCTURED_BUDGET
    assert create.call_args_list[2].kwargs["max_tokens"] == gd._DEFAULT_STRUCTURED_BUDGET


def completion_with_content(first):
    first.choices[0].message.content = '{"items": [{"snippet_index": 0, "relation": "unknown", "scope_matches": false, "quote": "", "rationale": "truncated"}]'
    first.choices[0].finish_reason = "length"
    return first


async def test_truncated_body_retries_before_failing_closed():
    agent, create = response_agent(
        [completion_with_content(completion(None)), completion(VALID_ATTRIBUTIONS)])
    records = []
    result = await gd.structured_call(agent, "prompt", gd.Attributions, on_attempt=records.append)
    assert result.items[0].relation == "unknown"
    assert create.await_count == 2
    assert create.call_args_list[1].kwargs["max_tokens"] == gd._RETRY_STRUCTURED_BUDGET


def _compact_prompt_capture(captured):
    async def capture(agent, prompt, schema, **kwargs):
        captured.append(prompt)
        return gd.Experiment(
            candidate_ids=["a", "b"],
            distinction="immune versus non-immune effect",
            intervention="effector depletion", controls=["isotype"],
            predictions={"a": "loss", "b": "retained"},
            falsification_rule="retained outcome refutes immune requirement",
            next_decisions={"loss": "pursue direct effect"})
    return capture


def _compact_hypothesis():
    return MechanismStep(1, "GD3", "NK clearance", "suppresses",
                         "GD3 suppresses NK clearance in OA cartilage", ["GD3 is elevated"])


async def test_design_experiment_prompt_is_compact(monkeypatch):
    hyp = MechanisticHypothesis("h", "summary", ["NK"], "GD3", "Siglec-7", "test",
        [_compact_hypothesis()], "phenotype", "", "")
    captured = []
    monkeypatch.setattr(gd, "structured_call", _compact_prompt_capture(captured))
    graph = gd.build_graph(evidence(), [("a", hyp), ("b", hyp)])
    result = await gd.design_experiment(None, [("a", hyp), ("b", hyp)],
                                        [("a", review()), ("b", review())], graph)
    assert result["candidate_ids"] == ["a", "b"]
    prompt = captured[0]
    assert "criticism" not in prompt                       # review prose trimmed
    assert "falsification_criteria" not in prompt
    assert "improvement_recommendations" not in prompt
    assert "supporting_evidence_claims" not in prompt      # mechanism prose trimmed
    assert "h" in prompt
    assert "summary" in prompt


async def test_refusal_does_not_retry():
    agent, create = response_agent([completion(None, refusal="refused")])
    with pytest.raises(gd.StructuredOutputError, match="refused_or_filtered"):
        await gd.structured_call(agent, "prompt", gd.Attributions)
    assert create.await_count == 1


async def test_failed_attribution_preserves_retrieval_and_leaves_graph_open():
    graph = gd.build_graph(evidence(), [("a", hypothesis())])
    original = graph.model_dump()
    retriever = SimpleNamespace(query=AsyncMock(return_value=RetrievalResult("q", "answer", [
        EvidenceSnippet("p", "p pages 1", [1], "GD3 is elevated", 10)])))
    memory = {}
    def cache(action, key, value):
        if action == "save":
            memory[key] = value
        return memory.get(key)
    agent, create = response_agent(
        [completion(None), completion("{}"), completion("{}"), completion(VALID_ATTRIBUTIONS)])
    with pytest.raises(gd.StructuredOutputError):
        await gd.run_evidence_loop(graph, retriever, agent, "OA", [("a", review())], 1, cache)
    assert graph.model_dump() == original
    assert "graph_episode" not in memory
    assert any(k.startswith("structured_diagnostics_") for k in memory)
    await gd.run_evidence_loop(graph, retriever, agent, "OA", [("a", review())], 1, cache)
    assert create.await_count == 4
    assert retriever.query.await_count == 1
    assert graph.unresolved_edges()


def test_compatible_resume_is_specific_to_known_code_and_upstream_stages():
    old_spec = {"objective": "same goal", "corpus": [{"sha256": "original_pdf_hash"}], "code": {
        "graph_discovery.py": "efb9d7aab7c5356b7e3f9e2df775cc55a3f88d5a6c9e43b63ffbf88db031f441",
        "oa_pipeline.py": "25eec7a984be8a6368d5b955632cc829f1466292acf3f85d42ccba19dafbd158",
        "skeptic.py": "original_agent_hash"}}
    stages = ["evidence_bundle", "round1_critiques", "retrieval-0123456789abcdef"]
    manifest = {"input_signature": pipeline._signature(old_spec), "completed": stages}
    new_spec = {**old_spec, "code": {**old_spec["code"], "graph_discovery.py": "fixed", "oa_pipeline.py": "fixed"}}
    assert pipeline._compatible_pre_graph_resume(manifest, new_spec, stages)
    assert not pipeline._compatible_pre_graph_resume(manifest, {**new_spec, "objective": "changed"}, stages)
    assert not pipeline._compatible_pre_graph_resume(manifest, {**new_spec, "corpus": []}, stages)
    assert not pipeline._compatible_pre_graph_resume(manifest, {**new_spec, "code": {
        **new_spec["code"], "skeptic.py": "changed"}}, stages)
    assert not pipeline._compatible_pre_graph_resume(manifest, new_spec, stages + ["graph_episode"])
    assert not pipeline._compatible_pre_graph_resume({**manifest, "completed": stages + ["graph_episode"]},
                                                    new_spec, stages + ["graph_episode"])
