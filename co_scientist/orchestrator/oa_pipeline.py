"""End-to-end Multi-Agent Scientific Discovery Pipeline.

Orchestrates:
1. PaperQA2-backed local evidence retrieval over literature corpus.
2. Evidence Extractor Agent -> Dynamic sub-queries, affirmative claims & critical caveats.
3. Mechanism Generator Agent -> Dynamic strategy generation & causal pathway chains.
4. Skeptic Agent (Round 1) -> Adversarial audit, contradictions, falsification criteria.
5. Evolution Loop (Round 2) -> Mutation & refinement of hypotheses to overcome Skeptic critique.
6. Skeptic Agent (Round 2) -> Re-audit of evolved hypotheses.
7. PI Agent -> Comprehensive hypothesis synthesis with 3-phase experimental validation plans.
8. Scientific Gatekeeper -> Hard-threshold elimination of unfalsifiable or ungrounded proposals.
9. Elo Tournament Engine -> Position-swapped pairwise scientific debates and dynamic Elo leaderboard.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..agents.evidence_extractor import EvidenceExtractorAgent
from ..agents.mechanism_generator import MechanismGeneratorAgent
from ..agents.pi_synthesizer import PIAgent
from ..agents.ranking_rules import EloTournamentEngine, ScientificRankingEngine
from ..rag.paperqa_retriever import PaperQARetriever
from ..agents.skeptic import SkepticAgent
from ..models.scientific_graph import ScientificEdge, ScientificGraph, ScientificNode
from .graph_discovery import (
    build_graph as _build_scientific_graph,
    run_evidence_loop, feedback_for, design_experiment,
)


class PipelineTimeoutError(TimeoutError):
    """Raised when a bounded pipeline phase exceeds its watchdog deadline."""


def _critique_quality(critique: Any) -> float:
    """Return a small, deterministic proxy for scientific critique quality.

    This is deliberately not a truth score. It measures whether the critique
    supplied actionable falsification and made the remaining uncertainty
    visible, which is sufficient for deciding whether another evolution pass
    is worth its cost.
    """
    verdict_score = {
        "FATAL_FLAW": 0.0,
        "MAJOR_REVISIONS_NEEDED": 0.45,
        "PASS_WITH_RESERVATIONS": 0.8,
        "PASS": 0.9,
    }.get(str(getattr(critique, "verdict", "")).upper(), 0.25)
    falsification = min(len(getattr(critique, "falsification_criteria", [])), 3) / 3
    audited = getattr(critique, "audited_links", [])
    supported = sum(
        getattr(item, "grounding_status", "") in {"fully_supported", "partially_supported"}
        for item in audited
    )
    grounding = supported / max(len(audited), 1)
    return round(0.5 * verdict_score + 0.3 * falsification + 0.2 * grounding, 4)


def _needs_evolution(previous: Any, current: Any, *, min_gain: float = 0.05) -> bool:
    """Continue only while the adversarial loop has unresolved value."""
    if str(getattr(current, "verdict", "")).upper() == "FATAL_FLAW":
        return True
    return _critique_quality(current) - _critique_quality(previous) < min_gain


def _review_status(evaluation, critique) -> str:
    """Report unresolved review findings; a PI response is not a new review."""
    if evaluation.is_eliminated:
        return "ELIMINATED"
    if critique.unsupported_core_claim or critique.verdict in {
        "FATAL_FLAW", "MAJOR_REVISIONS_NEEDED",
    }:
        return "REVIEW_REQUIRED"
    return "PROVISIONAL"


def _review_burden(critique) -> list[int]:
    return [
        int(critique.unsupported_core_claim),
        {"PASS": 0, "PASS_WITH_RESERVATIONS": 1, "MAJOR_REVISIONS_NEEDED": 2,
         "FATAL_FLAW": 3}.get(critique.verdict, 3),
        sum(link.grounding_status in {"unsupported_leap", "contradicted"}
            for link in critique.audited_links),
        len(critique.identified_contradictions),
    ]


def _review_improved(before, after) -> bool:
    """Conservative Pareto rule; more prose/falsification items earns no credit."""
    old, new = _review_burden(before), _review_burden(after)
    return bool(after.audited_links) and all(b <= a for a, b in zip(old, new)) and any(
        b < a for a, b in zip(old, new)
    )


def _signature(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _compatible_pre_graph_resume(manifest, spec, existing_stages):
    """Allow only the known structured-output recovery migration.

    The run may already contain downstream graph/evolution checkpoints because
    the old code failed after those stages. Reuse them only when the corpus,
    objective, run parameters, and every code file except the two recovery
    files are unchanged. This is deliberately narrower than a force-resume.
    """
    allowed = {
        "evidence_bundle", "strategies", "initial_mechanisms", "round1_critiques",
        "graph_episode", "graph_control_trace", "scientific_graph", "evolved_state",
        "evolution_round_1", "structured_diagnostics_experiment",
        "structured_diagnostics_retrieval-01f92c1d684131a7",
        "retrieval-01f92c1d684131a7",
        "post_retrieval_anti_gd3_immunotherapy",
        "post_retrieval_immune_clearance_checkpoint_modulation",
        "post_retrieval_st8sia1_enzymatic_inhibition",
    }
    completed = set(manifest.get("completed", []))
    if completed != set(existing_stages) or not completed or any(
        name not in allowed and not re.fullmatch(r"retrieval-[0-9a-f]{16}", name)
        for name in completed
    ):
        return False
    old_spec = manifest.get("input_spec")
    if not isinstance(old_spec, dict):
        # Compatibility with the original v1 manifest, which stored only the
        # signature. Keep this legacy path restricted to pre-graph stages.
        upstream = {"evidence_bundle", "strategies", "initial_mechanisms", "round1_critiques"}
        if completed != set(existing_stages) or not completed or any(
            name not in upstream and not re.fullmatch(r"retrieval-[0-9a-f]{16}", name)
            for name in completed
        ):
            return False
        legacy_spec = {**spec, "code": {**spec["code"],
            "graph_discovery.py": "efb9d7aab7c5356b7e3f9e2df775cc55a3f88d5a6c9e43b63ffbf88db031f441",
            "oa_pipeline.py": "25eec7a984be8a6368d5b955632cc829f1466292acf3f85d42ccba19dafbd158",
        }}
        return manifest.get("input_signature") == _signature(legacy_spec)
    for key in ("version", "corpus", "objective", "paper_dir", "paper_files",
                "max_evolution_rounds", "parallelism", "graph_evidence_queries",
                "run_tournament"):
        if old_spec.get(key) != spec.get(key):
            return False
    old_code = old_spec.get("code", {})
    current_code = spec.get("code", {})
    mutable = {"graph_discovery.py", "oa_pipeline.py"}
    if any(old_code.get(name) != current_code.get(name)
           for name in current_code if name not in mutable):
        return False
    return all(name in old_code for name in current_code)


async def run_oa_pipeline(
    research_objective: str = "基于这些pdf，对于骨关节炎，有什么从免疫层面干预测的方法，给出完整的推理路径",
    paper_dir: str | Path = "paper",
    paper_files: list[str] | None = None,
    output_dir: str | Path = "data/runs/oa_discovery",
    max_evolution_rounds: int = 2,
    parallelism: int = 3,
    phase_timeout_seconds: int = 900,
    overall_timeout_seconds: int = 3600,
    run_tournament: bool = False,
    resume: bool = False,
    graph_evidence_queries: int = 1,
    preflight_only: bool = False,
) -> dict:
    if min(phase_timeout_seconds, overall_timeout_seconds, parallelism) <= 0:
        raise ValueError("timeouts and parallelism must be positive")
    if min(max_evolution_rounds, graph_evidence_queries) < 0:
        raise ValueError("round and query budgets must be nonnegative")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    heartbeat_path = out_path / "pipeline_heartbeat.json"
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    selected_files = ([Path(paper_dir) / name for name in paper_files]
                      if paper_files is not None else sorted(Path(paper_dir).glob("*.pdf")))
    if not selected_files or any(not p.is_file() for p in selected_files):
        raise ValueError("every selected PDF must exist; empty corpus is not allowed")
    corpus_manifest = [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in selected_files]
    # Prompts live in agent source; invalidate downstream cached reasoning on
    # code/corpus changes rather than silently mixing experiment versions.
    code_files = sorted((Path(__file__).parents[1] / "agents").glob("*.py")) + [
        Path(__file__), Path(__file__).with_name("graph_discovery.py"),
        Path(__file__).parents[1] / "rag" / "paperqa_retriever.py",
    ]
    input_spec = {
        "version": "graph-episode-v1",
        "corpus": corpus_manifest,
        "code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
        "objective": research_objective,
        "paper_dir": str(paper_dir),
        "paper_files": paper_files,
        "max_evolution_rounds": max_evolution_rounds,
        "parallelism": parallelism,
        "graph_evidence_queries": graph_evidence_queries,
        "run_tournament": run_tournament,
    }
    input_signature = _signature(input_spec)
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = {}
    compatible_upgrade = False
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_signature") != input_signature:
            compatible_upgrade = _compatible_pre_graph_resume(
                manifest, input_spec, [p.stem for p in checkpoint_dir.glob("*.pkl")])
            if not compatible_upgrade:
                raise ValueError("checkpoint input signature differs; use a new output directory")
    elif resume and not manifest_path.exists():
        raise FileNotFoundError(f"no resumable checkpoint manifest: {manifest_path}")
    elif not resume and any(checkpoint_dir.iterdir()):
        raise ValueError("output has checkpoints; use --resume or a new output directory")

    if preflight_only:
        return {"resume_valid": resume, "compatible_upgrade": compatible_upgrade,
                "completed": manifest.get("completed", []), "model_calls": 0}

    def atomic_json(path, value):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    if compatible_upgrade:
        atomic_json(checkpoint_dir / "manifest.before-structured-output-fix.json", manifest)
        manifest.setdefault("resume_history", []).append({
            "from_signature": manifest["input_signature"], "to_signature": input_signature,
            "reason": "structured output handling fixed; only pre-attribution stages reused",
            "reused_stages": manifest["completed"],
        })
        manifest.update(input_signature=input_signature, input_spec=input_spec)
        atomic_json(manifest_path, manifest)
        print("  [resume] compatible structured-output fix; keeping upstream stages and cached retrieval", flush=True)
    if not resume:
        atomic_json(manifest_path, {"input_signature": input_signature, "corpus": corpus_manifest,
                                    "input_spec": input_spec, "completed": []})

    def checkpoint(name: str, value: Any) -> None:
        """Persist completed stage output for audit/review and future resume."""
        def encode(item):
            if hasattr(item, "model_dump"):
                return item.model_dump(mode="json")
            if hasattr(item, "__dataclass_fields__"):
                return asdict(item)
            if isinstance(item, (list, tuple)):
                return [encode(v) for v in item]
            if isinstance(item, dict):
                return {str(k): encode(v) for k, v in item.items()}
            return item
        encoded = encode(value)
        tmp = checkpoint_dir / f".{name}.pkl.tmp"
        tmp.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(checkpoint_dir / f"{name}.pkl")
        atomic_json(checkpoint_dir / f"{name}.json", encoded)
        atomic_json(manifest_path, {
            **manifest,
            "input_signature": input_signature,
            "input_spec": input_spec,
            "corpus": corpus_manifest,
            "completed": sorted(p.stem for p in checkpoint_dir.glob("*.pkl")),
        })

    def restore(name: str):
        path = checkpoint_dir / f"{name}.pkl"
        if not resume or not path.exists():
            return None
        print(f"  [resume] {name}", flush=True)
        with path.open("rb") as fh:
            return pickle.load(fh)

    def heartbeat(phase: str, status: str, error: str | None = None) -> None:
        atomic_json(heartbeat_path, {
            "phase": phase, "status": status,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - t_start, 2),
            "error": error,
        })

    async def phase(name: str, operation):
        if time.time() - t_start >= overall_timeout_seconds:
            if asyncio.iscoroutine(operation):
                operation.close()
            elif isinstance(operation, asyncio.Future):
                operation.cancel()
            heartbeat(name, "timeout", f"overall limit exceeded {overall_timeout_seconds}s")
            raise PipelineTimeoutError(
                f"pipeline exceeded {overall_timeout_seconds}s; checkpoint: {heartbeat_path}"
            )
        heartbeat(name, "running")
        print(f"  [start] {name}", flush=True)
        async def pulse():
            while True:
                await asyncio.sleep(30)
                heartbeat(name, "running")
                print(f"  [running] {name}; total {time.time() - t_start:.0f}s", flush=True)
        pulse_task = asyncio.create_task(pulse())
        completed = False
        try:
            result = await asyncio.wait_for(operation, timeout=min(
                phase_timeout_seconds,
                max(1, overall_timeout_seconds - int(time.time() - t_start)),
            ))
            completed = True
            return result
        except asyncio.TimeoutError as exc:
            heartbeat(name, "timeout", f"phase/remaining overall deadline reached ({phase_timeout_seconds}s / {overall_timeout_seconds}s)")
            raise PipelineTimeoutError(
                f"phase {name!r} exceeded {phase_timeout_seconds}s; "
                f"checkpoint: {heartbeat_path}"
            ) from exc
        except Exception as exc:
            heartbeat(name, "failed", repr(exc))
            raise
        finally:
            pulse_task.cancel()
            await asyncio.gather(pulse_task, return_exceptions=True)
            if completed and heartbeat_path.exists():
                heartbeat(name, "completed")
                print(f"  [done] {name}", flush=True)

    print("=" * 75)
    print("CO-SCIENTIST AUTONOMOUS DISCOVERY: EVOLUTIONARY MULTI-AGENT SYNTHESIS")
    print(f"Objective: {research_objective}")
    print("=" * 75)

    # Index lazily: a resumed run with all retrievals cached needs no API index.
    print("\n[Phase 1] Initializing PaperQA2 Local Evidence Retriever...")
    retriever = PaperQARetriever()
    class LocalCorpus:
        indexed = False

        async def query(self, query):
            if not self.indexed:
                for path in selected_files:
                    if not await retriever.index_file(path):
                        raise RuntimeError(f"index failed: {path.name}")
                self.indexed = True
                print(f"  Indexed {len(selected_files)} fixed PDFs", flush=True)
            return await retriever.query(query)
    local_corpus = LocalCorpus()

    # 2. Stage 1: Evidence Extractor Agent (Dynamic Sub-Queries)
    print("\n[Phase 2] Running Evidence Extractor Agent (Dynamic Decomposition)...")
    extractor = EvidenceExtractorAgent(local_corpus)
    evidence_bundle = restore("evidence_bundle")
    if evidence_bundle is None:
        evidence_bundle = await phase("extract_evidence", extractor.extract_evidence(query=research_objective))
        checkpoint("evidence_bundle", evidence_bundle)
    print(f"  Extracted {len(evidence_bundle.claims)} affirmative claims.")
    for i, c in enumerate(evidence_bundle.claims[:4]):
        print(f"    Claim {i+1}: {c.claim[:90]}... [Source: {c.source_paper}, pp. {c.pages}]")
    print(f"  Extracted {len(evidence_bundle.caveats_and_contradictions)} critical literature caveats / negative findings.")
    for i, c in enumerate(evidence_bundle.caveats_and_contradictions[:3]):
        print(f"    Caveat {i+1}: {c.claim[:90]}... [Source: {c.source_paper}, pp. {c.pages}]")

    # 3. Stage 2: Dynamic Strategy Discovery & Initial Mechanism Generation
    print("\n[Phase 3] Dynamically Discovering Intervention Strategies & Generating Causal Chains...")
    mechanism_agent = MechanismGeneratorAgent()
    strategies = restore("strategies")
    if strategies is None:
        strategies = await phase("propose_strategies", mechanism_agent.propose_strategies(
            research_objective=research_objective, evidence=evidence_bundle, n=3,
        ))
        checkpoint("strategies", strategies)
    print(f"  Discovered {len(strategies)} orthogonal intervention strategies:")
    if not strategies or len({key for key, _ in strategies}) != len(strategies) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key, _ in strategies
    ):
        raise ValueError("strategies must have unique nonempty alphanumeric slug keys")
    for slug, desc in strategies:
        print(f"    - [{slug}]: {desc}")

    semaphore = asyncio.Semaphore(max(1, parallelism))

    async def bounded(call):
        async with semaphore:
            return await call()

    async def generate_one(item):
        strat_key, strat_desc = item
        hyp = await bounded(lambda: mechanism_agent.generate_hypothesis(
            research_objective=research_objective,
            evidence=evidence_bundle,
            focus_strategy=strat_desc,
        ))
        return strat_key, hyp

    initial_mechanisms = restore("initial_mechanisms")
    if initial_mechanisms is None:
        initial_mechanisms = await phase("generate_mechanisms", asyncio.gather(*(generate_one(item) for item in strategies)))
        checkpoint("initial_mechanisms", initial_mechanisms)
    for strat_key, hyp in initial_mechanisms:
        print(f"    Generated '{strat_key}': '{hyp.title}' (Chain length: {len(hyp.causal_chain)} steps)")

    # 4. Stage 3: Skeptic Adversarial Audit (Round 1)
    print("\n[Phase 4] Skeptic Agent: Round 1 Adversarial Audit & Falsification Demarcation...")
    skeptic_agent = SkepticAgent()
    async def critique_one(item):
        strat_key, hyp = item
        critique = await bounded(lambda: skeptic_agent.critique(hypothesis=hyp, evidence=evidence_bundle))
        return strat_key, critique

    round1_critiques = restore("round1_critiques")
    if round1_critiques is None:
        round1_critiques = await phase("initial_skeptic_review", asyncio.gather(*(critique_one(item) for item in initial_mechanisms)))
        checkpoint("round1_critiques", round1_critiques)

    print("\n[Phase 4b] Skeptic → graph → local evidence → re-audit", flush=True)
    episode = restore("graph_episode")
    scientific_graph = episode["graph"] if episode else _build_scientific_graph(evidence_bundle, initial_mechanisms)

    episode_cache = {}
    def graph_cache(action, key, value):
        if action == "load":
            return episode_cache[key] if key in episode_cache else restore(key)
        episode_cache[key] = value
        checkpoint(key, value)

    graph_trace, graph_stop = await phase("graph_evidence_loop", run_evidence_loop(
        scientific_graph, local_corpus, mechanism_agent, research_objective,
        round1_critiques, graph_evidence_queries, graph_cache,
        restored=episode["trace"] if episode else None,
    ))
    checkpoint("scientific_graph", scientific_graph)
    checkpoint("graph_control_trace", {"trace": graph_trace, "stop_reason": graph_stop})
    branch_evidence = {
        key: replace(evidence_bundle, graph_feedback=feedback_for(scientific_graph, graph_trace, key))
        for key, _ in initial_mechanisms
    }
    # Only re-review branches that actually received new evidence.
    updated_critiques = []
    for (key, hyp), (_, old) in zip(initial_mechanisms, round1_critiques):
        new = restore(f"post_retrieval_{key}")
        if new is None:
            if any(row["strategy"] == key and row["attributions"] for row in graph_trace):
                new = await phase(f"post_retrieval_{key}", skeptic_agent.critique(
                    hypothesis=hyp, evidence=branch_evidence[key]))
            else:
                new = old
            checkpoint(f"post_retrieval_{key}", new)
        updated_critiques.append((key, new))

    # 5. Stage 4: Evolution & Refinement Loop (Round 2 Evolution)
    print("\n[Phase 5] Evolution Loop: Mutating & Refining Hypotheses to Resolve Skeptic Objections...")
    current_mechanisms = initial_mechanisms
    current_critiques = updated_critiques
    evolution_trace = []
    for round_no in range(1, max(0, max_evolution_rounds) + 1):
        async def evolve_one(item):
            (strat_key, hyp), (_, critique) = item
            if not critique.unsupported_core_claim and critique.verdict in {"PASS", "PASS_WITH_RESERVATIONS"}:
                return strat_key, hyp, critique
            evolved = await bounded(lambda: mechanism_agent.evolve_hypothesis(
                hypothesis=hyp, critique=critique, evidence=branch_evidence[strat_key],
                mode="address_skeptic_critique",
            ))
            revised = await bounded(lambda: skeptic_agent.critique(
                hypothesis=evolved, evidence=branch_evidence[strat_key]
            ))
            return strat_key, evolved, revised

        results = restore(f"evolution_round_{round_no}")
        if results is None:
            results = await phase(f"evolution_round_{round_no}", asyncio.gather(*(
                evolve_one(item) for item in zip(current_mechanisms, current_critiques)
            )))
            checkpoint(f"evolution_round_{round_no}", results)
        accepted = []
        decisions = []
        for (key, old_hyp), (_, old_review), (_, new_hyp, new_review) in zip(current_mechanisms, current_critiques, results):
            accept = _review_improved(old_review, new_review)
            accepted.append((key, new_hyp if accept else old_hyp, new_review if accept else old_review))
            decisions.append({"strategy": key, "action": "accept_revision" if accept else "retain_parent",
                              "before": _review_burden(old_review), "after": _review_burden(new_review),
                              "basis": "review proxy, not measured scientific truth"})
        evolution_trace.append({
            "round": round_no,
            "quality": {key: _critique_quality(crit) for key, _, crit in results},
            "decisions": decisions,
        })
        improved = [
            _needs_evolution(old_crit, new_crit)
            for (_, old_crit), (_, _, new_crit) in zip(current_critiques, results)
        ]
        current_mechanisms = [(key, hyp) for key, hyp, _ in accepted]
        current_critiques = [(key, crit) for key, _, crit in accepted]
        print(f"    Evolution round {round_no}: {sum(improved)}/{len(improved)} branches still need work")
        if not any(d["action"] == "accept_revision" for d in decisions):
            break

    evolved_mechanisms = current_mechanisms
    evolved_critiques = current_critiques
    checkpoint("evolved_state", {"mechanisms": evolved_mechanisms, "critiques": evolved_critiques})

    # Rebuild for the selected version. Do not attach an old edge's support to
    # a rewritten mechanism. Preserve the audited initial graph separately.
    selected_graph = _build_scientific_graph(evidence_bundle, evolved_mechanisms)
    selected_targets = {e.target for e in selected_graph.edges}
    audited_edges = {e.target: e for e in scientific_graph.edges if e.relation == "bridges"}
    selected_graph.edges = [audited_edges[e.target].model_copy(deep=True)
                            if e.target in audited_edges else e for e in selected_graph.edges]
    for edge in scientific_graph.edges:
        if edge.target in selected_targets and scientific_graph.nodes[edge.source].kind == "evidence":
            selected_graph.nodes[edge.source] = scientific_graph.nodes[edge.source].model_copy(deep=True)
            selected_graph.edges.append(edge.model_copy(deep=True))
    for target in selected_targets & audited_edges.keys():
        selected_graph.nodes[target].status = scientific_graph.nodes[target].status
    experiment = restore("discriminating_experiment")
    if graph_evidence_queries and experiment is None and selected_graph.unresolved_edges():
        experiment_diagnostics = restore("structured_diagnostics_experiment") or []
        def log_experiment_attempt(record):
            experiment_diagnostics.append(record)
            checkpoint("structured_diagnostics_experiment", experiment_diagnostics)
        experiment = await phase("design_discriminating_experiment", design_experiment(
            mechanism_agent, evolved_mechanisms, evolved_critiques, selected_graph,
            on_attempt=log_experiment_attempt))
        checkpoint("discriminating_experiment", experiment)
    if experiment:
        eid = "experiment-discriminating-1"
        selected_graph.nodes[eid] = ScientificNode(
            id=eid, kind="experiment", label=experiment["distinction"],
            description=json.dumps(experiment, ensure_ascii=False), metadata={"status": "proposed_not_executed"})
        for edge in list(selected_graph.unresolved_edges()):
            if selected_graph.nodes[edge.target].metadata["strategy"] in experiment["candidate_ids"]:
                selected_graph.edges.append(ScientificEdge(source=eid, target=edge.target, relation="tests"))
    checkpoint("selected_version_graph", selected_graph)

    # 6. Stage 5: Principal Investigator (PI) Synthesis
    print("\n[Phase 6] PI Synthesis Agent: Constructing Watertight Proposals & Validation Plans...")
    pi_agent = PIAgent()
    final_proposals = []
    for (strat_key, hyp), (_, critique) in zip(evolved_mechanisms, evolved_critiques):
        print(f"  Synthesizing final proposal for {strat_key}...")
        final_hyp = restore(f"pi_{strat_key}")
        if final_hyp is None:
            pi_evidence = replace(branch_evidence[strat_key], graph_feedback={
                **branch_evidence[strat_key].graph_feedback,
                "version_note": "retrieval audit refers to initial mechanism; revalidate changed edges",
                "discriminating_experiment": experiment,
            })
            final_hyp = await phase(f"pi_{strat_key}", pi_agent.synthesize(
            research_objective=research_objective,
            evidence=pi_evidence,
            mechanism=hyp,
            critique=critique,
            ))
            checkpoint(f"pi_{strat_key}", final_hyp)
        final_proposals.append((strat_key, final_hyp, critique))
        print(f"    Synthesized: '{final_hyp.title}' with {len(final_hyp.experimental_validation_plan)} validation protocols.")
    checkpoint("final_proposals", final_proposals)

    # 7. Stage 6: Scientific Gatekeeper (Hard Elimination Thresholds)
    print("\n[Phase 7] Scientific Gatekeeper: Filtering Unfalsifiable or Ungrounded Proposals...")
    gatekeeper = ScientificRankingEngine()
    candidates_for_tournament = []
    for strat_key, final_hyp, critique in final_proposals:
        eval_result = gatekeeper.evaluate_hypothesis(
            hypothesis=final_hyp,
            evidence=evidence_bundle,
            critique=critique,
        )
        status_tag = _review_status(eval_result, critique)
        print(f"  {status_tag} '{final_hyp.title[:60]}' - Support: {eval_result.evidence_support:.2f}, Falsifiable: {eval_result.testability:.2f}")
        if eval_result.elimination_reason:
            print(f"    Elimination Reason: {eval_result.elimination_reason}")
        candidates_for_tournament.append({
            "strategy": strat_key,
            "hypothesis": final_hyp,
            "critique": critique,
            "gatekeeper_eval": eval_result,
        })

    # 8. Stage 7: Elo Tournament Engine (Position-Swapped Pairwise Debates)
    if run_tournament:
        print("\n[Phase 8] Running optional Head-to-Head Pairwise Elo Tournament...")
        tournament_engine = EloTournamentEngine()
        leaderboard, matches = await phase("elo_tournament", tournament_engine.run_tournament(
            research_objective=research_objective,
            candidates=candidates_for_tournament,
            k_factor=32.0,
        ))
    else:
        print("\n[Phase 8] Elo skipped (scientific controlled mode).")
        # Deterministic scientific ordering: evidence and testability first;
        # Elo remains an optional presentation/ranking layer.
        candidates_for_tournament.sort(
            key=lambda c: (
                c["gatekeeper_eval"].is_eliminated,
                -c["gatekeeper_eval"].final_score,
            )
        )
        from ..agents.ranking_rules import TournamentRankedEntry
        leaderboard = [TournamentRankedEntry(
            rank=i + 1, title=c["hypothesis"].title, strategy=c["strategy"],
            elo_score=c["gatekeeper_eval"].final_score * 1000,
            wins=0, losses=0, ties=0, matches_played=0,
            hypothesis=c["hypothesis"], critique=c["critique"],
            gatekeeper_eval=c["gatekeeper_eval"],
        ) for i, c in enumerate(candidates_for_tournament)]
        matches = []
        checkpoint("gatekeeper_results", leaderboard)

    if not leaderboard:
        raise RuntimeError(
            "No hypothesis passed the scientific gatekeeper; "
            "the evidence/strategy budget produced no rankable candidate."
        )

    print("\n" + "=" * 75)
    score_label = "Elo" if run_tournament else "Heuristic score / 1000"
    print(f"CANDIDATE RANKING ({score_label})")
    print("=" * 75)
    for entry in leaderboard:
        status_label = _review_status(entry.gatekeeper_eval, entry.critique)
        print(f"Rank {entry.rank}: {entry.title}")
        print(f"  Strategy: {entry.strategy} | Status: {status_label} | {score_label}: {entry.elo_score:.1f}")
        if entry.gatekeeper_eval.elimination_reason:
            print(f"  Elimination Reason: {entry.gatekeeper_eval.elimination_reason}")

    print("\n[Tournament Matches Played]")
    for idx, m in enumerate(matches, 1):
        print(f"  Match {idx}: '{m.hyp_a_title[:35]}' vs '{m.hyp_b_title[:35]}'")
        print(f"    Result: Winner = {m.winner} | Elo Δ: A({m.elo_a_before:.1f} -> {m.elo_a_after:.1f}), B({m.elo_b_before:.1f} -> {m.elo_b_after:.1f})")
        print(f"    Deciding Factor: {m.deciding_factor}")

    # 9. Save Detailed JSON Report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "research_objective": research_objective,
        "elapsed_seconds": round(time.time() - t_start, 2),
        "evidence_bundle": {
            "summary": evidence_bundle.summary,
            "claims": [asdict(c) for c in evidence_bundle.claims],
            "caveats_and_contradictions": [asdict(c) for c in evidence_bundle.caveats_and_contradictions],
        },
        "tournament_leaderboard": [
            {
                "rank": entry.rank,
                "strategy": entry.strategy,
                "title": entry.title,
                "elo_score": entry.elo_score,
                "wins": entry.wins,
                "losses": entry.losses,
                "ties": entry.ties,
                "matches_played": entry.matches_played,
                "is_eliminated": entry.gatekeeper_eval.is_eliminated,
                "elimination_reason": entry.gatekeeper_eval.elimination_reason,
                "gatekeeper_evaluation": asdict(entry.gatekeeper_eval),
                "review_status": _review_status(entry.gatekeeper_eval, entry.critique),
                "hypothesis": asdict(entry.hypothesis),
                "critique": asdict(entry.critique),
            }
            for entry in leaderboard
        ],
        "tournament_matches": [asdict(m) for m in matches],
        "evolution_trace": evolution_trace,
        "ranking_mode": "elo" if run_tournament else "heuristic_score",
        "scientific_graph": scientific_graph.model_dump(mode="json"),
        "graph_control_trace": graph_trace,
        "graph_stop_reason": graph_stop,
        "selected_version_graph": selected_graph.model_dump(mode="json"),
        "discriminating_experiment": experiment,
        "corpus_manifest": corpus_manifest,
        "review_status_by_strategy": {
            entry.strategy: _review_status(entry.gatekeeper_eval, entry.critique)
            for entry in leaderboard
        },
        # Compatibility field for exporter
        "ranked_hypotheses": [
            {
                "rank": entry.rank,
                "strategy": entry.strategy,
                "score": entry.elo_score / 1500.0,
                "is_eliminated": entry.gatekeeper_eval.is_eliminated,
                "elimination_reason": entry.gatekeeper_eval.elimination_reason,
                "evaluation": asdict(entry.gatekeeper_eval),
                "hypothesis": asdict(entry.hypothesis),
                "critique": asdict(entry.critique),
            }
            for entry in leaderboard
        ],
    }

    report_json_path = out_path / "oa_discovery_report.json"
    report_json_path.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDetailed JSON report saved to: {report_json_path}")

    # 10. Generate Markdown Executive Summary
    top_entry = next(
        (entry for entry in leaderboard if not entry.gatekeeper_eval.is_eliminated),
        leaderboard[0],
    )
    top_hyp = top_entry.hypothesis
    top_gate = top_entry.gatekeeper_eval
    top_critique = top_entry.critique

    md_lines = [
        f"# 基于文献的免疫干预假说与完整因果推理路径报告",
        f"\n**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**模型**: deepseek-v4-flash-0731 | **RAG基础设施**: PaperQA2 (严格页码引用 + 阴性结果检索)",
        f"**评审架构**: 最多 {max_evolution_rounds} 轮修订与复审；排序：{score_label}。PI 综合后尚未重新审查。",
        f"**研究问题**: {research_objective}\n",
        f"## 1. 候选排序（分数不是科学正确率）",
        f"| 排名 | 假说标题 | 策略模式 | {score_label} | 胜/负/平 | 审查状态 | 证据支撑代理 |",
        f"|---|---|---|---|---|---|---|",
    ]

    for entry in leaderboard:
        status_str = _review_status(entry.gatekeeper_eval, entry.critique)
        md_lines.append(
            f"| {entry.rank} | **{entry.title}** | `{entry.strategy}` | **{entry.elo_score:.1f}** | {entry.wins}W-{entry.losses}L-{entry.ties}T | {status_str} | {entry.gatekeeper_eval.evidence_support:.2f} |"
        )

    md_lines.extend([
        f"\n## 2. 当前最高分候选（未验证方案）",
        f"**标题**: {top_hyp.title}",
        f"**{score_label}**: `{top_entry.elo_score:.1f}` | **策略**: `{top_entry.strategy}`",
        f"\n### 核心假说陈述 (Primary Hypothesis Statement)",
        f"{top_hyp.primary_hypothesis_statement}\n",
        f"### 目标轴线 (Target Axis)",
        f"- **分子靶点**: {top_hyp.target_axis.get('molecular_target', '')}",
        f"- **检查点 / 通路受体**: {top_hyp.target_axis.get('checkpoint_receptor', '')}",
        f"- **效应细胞群**: {top_hyp.target_axis.get('effector_immune_cells', '')}",
        f"- **目标组织与微环境**: {top_hyp.target_axis.get('target_joint_compartment', '')}\n",
        f"### 完整因果推理路径 (Complete Causal Reasoning Trajectory)",
    ])

    for step in top_hyp.complete_reasoning_path:
        md_lines.append(f"1. **{step.get('step', '')}**")
        md_lines.append(f"   - **机制推演**: {step.get('mechanism', '')}")
        md_lines.append(f"   - **文献证据支撑**: {step.get('evidence_support', '')}")

    md_lines.extend([
        f"\n### 文献佐证表 (Grounded Evidence Table)",
        f"| 来源文献 | 页码 | 提取原文证据 | 支撑论点 |",
        f"|---|---|---|---|",
    ])
    for row in top_hyp.evidence_grounding_table[:6]:
        quote_snippet = row.get("quote", "").replace("\n", " ")[:100]
        md_lines.append(f"| {row.get('paper', '')} | {row.get('pages', [])} | \"{quote_snippet}...\" | {row.get('supported_claim', '')} |")

    md_lines.extend([
        f"\n### 进化闭环：Skeptic 批判与反证化解策略 (Critique & Evolution Mitigation)",
    ])
    for item in top_hyp.contradiction_and_risk_mitigation:
        md_lines.append(f"- **Skeptic质疑/阴性结果风险**: {item.get('skeptic_concern', '')}")
        md_lines.append(f"  - **进化化解策略**: {item.get('resolution_strategy', '')}")

    md_lines.extend([
        f"\n### 波普尔式可证伪性判定标准 (Falsification Demarcation Criteria)",
    ])
    for idx, f_crit in enumerate(top_hyp.falsification_criteria, 1):
        md_lines.append(f"{idx}. **检验实验**: {f_crit.get('test', '')}")
        md_lines.append(f"   - **证伪判据 (Failure/Falsifying Observation)**: {f_crit.get('falsifying_result', '')}")

    md_lines.extend([
        f"\n### 三阶段实验验证方案 (Three-Phase Validation Protocols)",
    ])
    for p in top_hyp.experimental_validation_plan:
        md_lines.append(f"#### [{p.phase}] {p.title}")
        md_lines.append(f"- **实验模型体系**: {p.model_system}")
        md_lines.append(f"- **具体干预措施**: {p.intervention}")
        md_lines.append(f"- **核心读数指标**: {', '.join(p.readouts_and_assays)}")
        md_lines.append(f"- **成功判定标准**: {p.success_criteria}")
        md_lines.append(f"- **证伪/失败标准**: {p.failure_criteria}\n")

    if matches:
        md_lines.extend([
            f"## 3. 锦标赛对决详情 (Tournament Match Transcripts)",
            f"| 对决场次 | 候选 A vs 候选 B | 胜者 | A Elo 变动 | B Elo 变动 | 决定性裁决依据 |",
            f"|---|---|---|---|---|---|",
        ])
        for idx, m in enumerate(matches, 1):
            md_lines.append(
                f"| Match {idx} | {m.hyp_a_title[:28]}... vs {m.hyp_b_title[:28]}... | **{m.winner}** | {m.elo_a_before:.1f} -> {m.elo_a_after:.1f} | {m.elo_b_before:.1f} -> {m.elo_b_after:.1f} | {m.deciding_factor} |"
            )

    report_md_path = out_path / "oa_discovery_report.md"
    md_lines.extend([
        "\n## 4. Graph-driven 控制轨迹",
        f"- 检索预算：{graph_evidence_queries}；执行：{len(graph_trace)}；停止原因：{graph_stop}",
        "- 初始图审计与修订版本分开保存；检索命中或模型归因不代表生物学证明。",
    ])
    for row in graph_trace:
        md_lines.append(f"- **{row['strategy']}**：{row['claim']}")
        md_lines.append(f"  - 未解决边：{row['unresolved_before']} → {row['unresolved_after']}")
        for attribution in row["attributions"]:
            md_lines.append(f"  - {attribution['relation']}: {attribution['rationale']}")
    for round_trace in evolution_trace:
        for decision in round_trace.get("decisions", []):
            md_lines.append(f"- 修订 {decision['strategy']}：{decision['action']}，审查负担 {decision['before']} → {decision['after']}")
    if experiment:
        md_lines.extend(["\n## 5. 优先区分性实验（尚未执行）",
                         "```json", json.dumps(experiment, ensure_ascii=False, indent=2), "```"])
    report_md_text = "\n".join(md_lines)
    report_md_path.write_text(report_md_text, encoding="utf-8")
    print(f"Executive Markdown report saved to: {report_md_path}")

    # Persist into standard Co-Scientist session storage
    from ..storage.exporter import persist_as_coscientist_session
    try:
        sid = await persist_as_coscientist_session(report_data, report_md_text)
        print(f"Persisted to Co-Scientist session: {sid} (data/artifacts/{sid})")
    except Exception as e:
        print(f"Note: Standard session export warning: {e}")

    heartbeat("pipeline", "completed")
    print(f"All operations finished in {time.time() - t_start:.1f}s.", flush=True)
    return report_data


if __name__ == "__main__":
    asyncio.run(run_oa_pipeline())
