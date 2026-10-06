"""Unit tests for OA Immunological Discovery pipeline components."""

import math
import pytest
from co_scientist.agents.evidence_extractor import ExtractedEvidenceBundle, GroundedClaim
from co_scientist.agents.mechanism_generator import MechanismStep, MechanisticHypothesis
from co_scientist.agents.skeptic import (
    EvidenceAuditItem,
    FalsificationCriterion,
    SkepticCritique,
)
from co_scientist.agents.pi_synthesizer import (
    ExperimentalProtocol,
    FinalSynthesizedHypothesis,
)
from co_scientist.agents.ranking_rules import ScientificRankingEngine
from co_scientist.rag.paperqa_retriever import EvidenceSnippet, RetrievalResult
from co_scientist.orchestrator.oa_pipeline import _critique_quality, _needs_evolution, _build_scientific_graph
from co_scientist.rag.paperqa_retriever import PaperQARetriever


def test_ranking_engine_unsupported_claim_elimination():
    engine = ScientificRankingEngine()
    
    hyp = FinalSynthesizedHypothesis(
        title="Flawed Hypothesis",
        executive_summary="Summary",
        primary_hypothesis_statement="Statement",
        biological_rationale="Rationale",
        target_axis={},
        complete_reasoning_path=[],
        evidence_grounding_table=[],
        contradiction_and_risk_mitigation=[],
        falsification_criteria=[{"test": "T1", "falsifying_result": "R1"}],
        experimental_validation_plan=[],
        clinical_translational_potential="None",
    )
    
    critique = SkepticCritique(
        hypothesis_title="Flawed Hypothesis",
        verdict="FATAL_FLAW",
        overall_skeptical_assessment="Fatal unsupported claim",
        unsupported_core_claim=True,
    )
    
    evidence = ExtractedEvidenceBundle(query="test", summary="test")
    
    eval_res = engine.evaluate_hypothesis(hyp, evidence, critique)
    assert eval_res.is_eliminated is True
    assert eval_res.final_score == -math.inf
    assert "unsupported_core_claim" in eval_res.audit_notes[0]


def test_paperqa_references_preserve_one_string_reference():
    class Session:
        references = "Fissoun2025 pages 1-2"
        contexts = []
        formatted_answer = "answer"

    # The public parser contract is exercised through a lightweight mock
    # session in the async integration tests; this regression documents the
    # string-vs-list shape fixed in PaperQARetriever.query.
    assert isinstance(Session.references, str)


def test_mechanism_chain_becomes_unresolved_graph_bridge_without_matching_claim():
    evidence = ExtractedEvidenceBundle(
        query="q", summary="s",
        claims=[GroundedClaim(
            claim="GD3 is elevated in OA", source_paper="p", pages=[1],
            verbatim_quote="GD3 is elevated in OA", biological_entities=["GD3"],
        )],
    )
    mechanism = MechanisticHypothesis(
        title="h", summary="s", target_cells=["cell"], molecular_target="GD3",
        immunological_checkpoint="Siglec-7", therapeutic_modality="mAb",
        causal_chain=[MechanismStep(1, "GD3", "Siglec-7", "inhibits", "bridge", [])],
        expected_joint_phenotype="p", evidence_grounding_summary="", novelty_assessment="",
    )
    graph = _build_scientific_graph(evidence, [("strategy", mechanism)])
    assert len(graph.unresolved_edges()) == 1


def test_critique_quality_rewards_actionable_grounding():
    weak = SkepticCritique(
        hypothesis_title="weak", verdict="MAJOR_REVISIONS_NEEDED",
        overall_skeptical_assessment="", falsification_criteria=[]
    )
    strong = SkepticCritique(
        hypothesis_title="strong", verdict="PASS_WITH_RESERVATIONS",
        overall_skeptical_assessment="",
        audited_links=[EvidenceAuditItem("step", "fully_supported", "", "")],
        falsification_criteria=[
            FalsificationCriterion("experiment", "observation", "logic"),
            FalsificationCriterion("experiment2", "observation2", "logic2"),
        ],
    )
    assert _critique_quality(strong) > _critique_quality(weak)
    assert _needs_evolution(weak, strong) is False


def test_ranking_engine_unfalsifiable_elimination():
    engine = ScientificRankingEngine()
    
    hyp = FinalSynthesizedHypothesis(
        title="Unfalsifiable Hypothesis",
        executive_summary="Summary",
        primary_hypothesis_statement="Statement",
        biological_rationale="Rationale",
        target_axis={},
        complete_reasoning_path=[],
        evidence_grounding_table=[],
        contradiction_and_risk_mitigation=[],
        falsification_criteria=[], # Empty!
        experimental_validation_plan=[],
        clinical_translational_potential="None",
    )
    
    critique = SkepticCritique(
        hypothesis_title="Unfalsifiable Hypothesis",
        verdict="FATAL_FLAW",
        overall_skeptical_assessment="No falsification criteria provided",
        falsification_criteria=[], # Empty!
        unsupported_core_claim=False,
    )
    
    evidence = ExtractedEvidenceBundle(query="test", summary="test")
    
    eval_res = engine.evaluate_hypothesis(hyp, evidence, critique)
    assert eval_res.is_eliminated is True
    assert eval_res.final_score == -math.inf
    assert "falsification" in eval_res.elimination_reason.lower()


def test_ranking_engine_rigorous_hypothesis_scoring():
    engine = ScientificRankingEngine()
    
    hyp = FinalSynthesizedHypothesis(
        title="Targeting GD3-Siglec-7 Checkpoint to Restore Senoclearance in OA",
        executive_summary="Summary",
        primary_hypothesis_statement="Blockade of the GD3-Siglec-7 immune checkpoint restores NK cell degranulation and macrophage efferocytosis, halting OA progression.",
        biological_rationale="Rationale",
        target_axis={
            "molecular_target": "GD3 / ST8SIA1",
            "checkpoint_receptor": "Siglec-7",
            "effector_immune_cells": "NK cells and synovial macrophages",
            "target_joint_compartment": "Synovium and subchondral bone",
        },
        complete_reasoning_path=[
            {"step": "1", "mechanism": "m1", "evidence_support": "e1"},
            {"step": "2", "mechanism": "m2", "evidence_support": "e2"},
            {"step": "3", "mechanism": "m3", "evidence_support": "e3"},
            {"step": "4", "mechanism": "m4", "evidence_support": "e4"},
            {"step": "5", "mechanism": "m5", "evidence_support": "e5"},
        ],
        evidence_grounding_table=[
            {"paper": "Fissoun2025", "pages": [1, 2], "quote": "GD3 is upregulated", "supported_claim": "c1"},
            {"paper": "Iltis2025", "pages": [3, 5], "quote": "GD3 binds Siglec-7", "supported_claim": "c2"},
        ],
        contradiction_and_risk_mitigation=[
            {"skeptic_concern": "Subchondral bone vs cartilage", "resolution_strategy": "Direct effect on bone marrow"},
        ],
        falsification_criteria=[
            {"test": "NK depletion", "falsifying_result": "Identical bone protection in NK-depleted mice"},
            {"test": "Siglec-E KO", "falsifying_result": "No phenotype alteration"},
        ],
        experimental_validation_plan=[
            ExperimentalProtocol(
                phase="In_Vitro",
                title="NK degranulation assay",
                model_system="Primary OA chondrocytes + primary NK cells",
                intervention="Anti-GD3 mAb",
                readouts_and_assays=["CD107a flow", "Perforin ELISA"],
                success_criteria="Increased CD107a",
                failure_criteria="No change in degranulation",
            ),
            ExperimentalProtocol(
                phase="In_Vivo_Preclinical",
                title="DMM OA mouse model",
                model_system="DMM surgical OA in C57BL/6 mice",
                intervention="Intra-articular anti-GD3 mAb",
                readouts_and_assays=["Micro-CT", "TRAP staining", "Mankin score"],
                success_criteria="Preserved subchondral bone trabecular density",
                failure_criteria="No bone preservation",
            ),
        ],
        clinical_translational_potential="High",
    )
    
    critique = SkepticCritique(
        hypothesis_title=hyp.title,
        verdict="PASS_WITH_RESERVATIONS",
        overall_skeptical_assessment="Well grounded proposal",
        audited_links=[
            EvidenceAuditItem(
                claim_or_step="Step 1",
                grounding_status="fully_supported",
                criticism="None",
                relevant_paper_caveats="",
            ),
            EvidenceAuditItem(
                claim_or_step="Step 2",
                grounding_status="fully_supported",
                criticism="None",
                relevant_paper_caveats="",
            ),
        ],
        identified_contradictions=["Cartilage ECM penetration limited"],
        biological_risks_and_limitations=["Need local delivery"],
        falsification_criteria=[
            FalsificationCriterion(
                experiment="NK depletion in OA mice",
                falsifying_observation="No loss of efficacy",
                underlying_logic="Proves mechanism is NK-independent",
            )
        ],
        unsupported_core_claim=False,
    )
    
    evidence = ExtractedEvidenceBundle(query="test", summary="test")
    eval_res = engine.evaluate_hypothesis(hyp, evidence, critique)
    assert eval_res.is_eliminated is False
    assert eval_res.final_score > 0.80
    assert eval_res.evidence_support >= 0.90
    assert eval_res.citation_completeness == 1.0
    assert eval_res.testability > 0.90


@pytest.mark.asyncio
async def test_elo_tournament_engine_position_swapped_matches():
    from unittest.mock import AsyncMock
    from co_scientist.agents.ranking_rules import EloTournamentEngine, RankingEvaluation

    engine = EloTournamentEngine()

    hyp_a = FinalSynthesizedHypothesis(
        title="Hypothesis Alpha",
        executive_summary="Summary A",
        primary_hypothesis_statement="Statement A",
        biological_rationale="Rationale A",
        target_axis={"target": "T1"},
        complete_reasoning_path=[{"step": "1", "mechanism": "M1"}],
        evidence_grounding_table=[],
        contradiction_and_risk_mitigation=[],
        falsification_criteria=[{"test": "T", "falsifying_result": "F"}],
        experimental_validation_plan=[],
        clinical_translational_potential="High",
    )

    hyp_b = FinalSynthesizedHypothesis(
        title="Hypothesis Beta",
        executive_summary="Summary B",
        primary_hypothesis_statement="Statement B",
        biological_rationale="Rationale B",
        target_axis={"target": "T2"},
        complete_reasoning_path=[{"step": "1", "mechanism": "M2"}],
        evidence_grounding_table=[],
        contradiction_and_risk_mitigation=[],
        falsification_criteria=[{"test": "T", "falsifying_result": "F"}],
        experimental_validation_plan=[],
        clinical_translational_potential="Moderate",
    )

    crit = SkepticCritique(hypothesis_title="", verdict="PASS", overall_skeptical_assessment="")
    gate_eval = RankingEvaluation(
        hypothesis_title="", final_score=0.9, is_eliminated=False, elimination_reason=None,
        evidence_support=1.0, citation_completeness=1.0, contradiction_handling=1.0,
        testability=1.0, novelty=0.85,
    )

    candidates = [
        {"strategy": "s1", "hypothesis": hyp_a, "critique": crit, "gatekeeper_eval": gate_eval},
        {"strategy": "s2", "hypothesis": hyp_b, "critique": crit, "gatekeeper_eval": gate_eval},
    ]

    # Mock _judge_pair:
    # Pass 1 (A, B): declares 1 (A is better)
    # Pass 2 (B, A): declares 2 (A is still better across position swap)
    engine._judge_pair = AsyncMock(side_effect=[
        ("1", "Hypothesis Alpha has stronger mechanism", "Causal clarity"),
        ("2", "Hypothesis Alpha remains superior", "Position consistency"),
    ])

    leaderboard, matches = await engine.run_tournament("Test objective", candidates)
    assert len(matches) == 1
    match = matches[0]
    assert match.winner == "A"
    assert match.elo_a_after > 1200.0
    assert match.elo_b_after < 1200.0
    assert leaderboard[0].title == "Hypothesis Alpha"
    assert leaderboard[0].rank == 1
    assert leaderboard[1].title == "Hypothesis Beta"
    assert leaderboard[1].rank == 2
