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
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..agents.evidence_extractor import EvidenceExtractorAgent
from ..agents.mechanism_generator import MechanismGeneratorAgent
from ..agents.pi_synthesizer import PIAgent
from ..agents.ranking_rules import EloTournamentEngine, ScientificRankingEngine
from ..rag.paperqa_retriever import PaperQARetriever
from ..agents.skeptic import SkepticAgent


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


async def run_oa_pipeline(
    research_objective: str = "基于这些pdf，对于骨关节炎，有什么从免疫层面干预测的方法，给出完整的推理路径",
    paper_dir: str | Path = "paper",
    output_dir: str | Path = "data/runs/oa_discovery",
    max_evolution_rounds: int = 2,
    parallelism: int = 3,
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 75)
    print("CO-SCIENTIST AUTONOMOUS DISCOVERY: EVOLUTIONARY MULTI-AGENT SYNTHESIS")
    print(f"Objective: {research_objective}")
    print("=" * 75)

    # 1. Initialize PaperQA2 Retriever
    print("\n[Phase 1] Initializing PaperQA2 Local Evidence Retriever...")
    retriever = PaperQARetriever()
    indexed_count = await retriever.index_directory(paper_dir)
    print(f"  Indexed {indexed_count} PDF documents from {paper_dir}.")

    # 2. Stage 1: Evidence Extractor Agent (Dynamic Sub-Queries)
    print("\n[Phase 2] Running Evidence Extractor Agent (Dynamic Decomposition)...")
    extractor = EvidenceExtractorAgent(retriever)
    evidence_bundle = await extractor.extract_evidence(query=research_objective)
    print(f"  Extracted {len(evidence_bundle.claims)} affirmative claims.")
    for i, c in enumerate(evidence_bundle.claims[:4]):
        print(f"    Claim {i+1}: {c.claim[:90]}... [Source: {c.source_paper}, pp. {c.pages}]")
    print(f"  Extracted {len(evidence_bundle.caveats_and_contradictions)} critical literature caveats / negative findings.")
    for i, c in enumerate(evidence_bundle.caveats_and_contradictions[:3]):
        print(f"    Caveat {i+1}: {c.claim[:90]}... [Source: {c.source_paper}, pp. {c.pages}]")

    # 3. Stage 2: Dynamic Strategy Discovery & Initial Mechanism Generation
    print("\n[Phase 3] Dynamically Discovering Intervention Strategies & Generating Causal Chains...")
    mechanism_agent = MechanismGeneratorAgent()
    strategies = await mechanism_agent.propose_strategies(
        research_objective=research_objective,
        evidence=evidence_bundle,
        n=3,
    )
    print(f"  Discovered {len(strategies)} orthogonal intervention strategies:")
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

    initial_mechanisms = await asyncio.gather(*(generate_one(item) for item in strategies))
    for strat_key, hyp in initial_mechanisms:
        print(f"    Generated '{strat_key}': '{hyp.title}' (Chain length: {len(hyp.causal_chain)} steps)")

    # 4. Stage 3: Skeptic Adversarial Audit (Round 1)
    print("\n[Phase 4] Skeptic Agent: Round 1 Adversarial Audit & Falsification Demarcation...")
    skeptic_agent = SkepticAgent()
    async def critique_one(item):
        strat_key, hyp = item
        critique = await bounded(lambda: skeptic_agent.critique(hypothesis=hyp, evidence=evidence_bundle))
        return strat_key, critique

    round1_critiques = await asyncio.gather(*(critique_one(item) for item in initial_mechanisms))

    # 5. Stage 4: Evolution & Refinement Loop (Round 2 Evolution)
    print("\n[Phase 5] Evolution Loop: Mutating & Refining Hypotheses to Resolve Skeptic Objections...")
    current_mechanisms = initial_mechanisms
    current_critiques = round1_critiques
    evolution_trace = []
    for round_no in range(1, max(0, max_evolution_rounds) + 1):
        async def evolve_one(item):
            (strat_key, hyp), (_, critique) = item
            evolved = await bounded(lambda: mechanism_agent.evolve_hypothesis(
                hypothesis=hyp, critique=critique, evidence=evidence_bundle,
                mode="address_skeptic_critique",
            ))
            revised = await bounded(lambda: skeptic_agent.critique(
                hypothesis=evolved, evidence=evidence_bundle
            ))
            return strat_key, evolved, revised

        results = await asyncio.gather(*(
            evolve_one(item) for item in zip(current_mechanisms, current_critiques)
        ))
        evolution_trace.append({
            "round": round_no,
            "quality": {key: _critique_quality(crit) for key, _, crit in results},
        })
        improved = [
            _needs_evolution(old_crit, new_crit)
            for (_, old_crit), (_, _, new_crit) in zip(current_critiques, results)
        ]
        current_mechanisms = [(key, hyp) for key, hyp, _ in results]
        current_critiques = [(key, crit) for key, _, crit in results]
        print(f"    Evolution round {round_no}: {sum(improved)}/{len(improved)} branches still need work")
        if not any(improved):
            break

    evolved_mechanisms = current_mechanisms
    evolved_critiques = current_critiques

    # 6. Stage 5: Principal Investigator (PI) Synthesis
    print("\n[Phase 6] PI Synthesis Agent: Constructing Watertight Proposals & Validation Plans...")
    pi_agent = PIAgent()
    final_proposals = []
    for (strat_key, hyp), (_, critique) in zip(evolved_mechanisms, evolved_critiques):
        print(f"  Synthesizing final proposal for {strat_key}...")
        final_hyp = await pi_agent.synthesize(
            research_objective=research_objective,
            evidence=evidence_bundle,
            mechanism=hyp,
            critique=critique,
        )
        final_proposals.append((strat_key, final_hyp, critique))
        print(f"    Synthesized: '{final_hyp.title}' with {len(final_hyp.experimental_validation_plan)} validation protocols.")

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
        status_tag = "[ELIMINATED]" if eval_result.is_eliminated else "[QUALIFIED]"
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
    print("\n[Phase 8] Running Head-to-Head Pairwise Elo Tournament...")
    tournament_engine = EloTournamentEngine()
    leaderboard, matches = await tournament_engine.run_tournament(
        research_objective=research_objective,
        candidates=candidates_for_tournament,
        k_factor=32.0,
    )

    if not leaderboard:
        raise RuntimeError(
            "No hypothesis passed the scientific gatekeeper; "
            "the evidence/strategy budget produced no rankable candidate."
        )

    print("\n" + "=" * 75)
    print("TOURNAMENT LEADERBOARD (ELO RATINGS)")
    print("=" * 75)
    for entry in leaderboard:
        status_label = "[DISQUALIFIED]" if entry.gatekeeper_eval.is_eliminated else f"Elo: {entry.elo_score:.1f}"
        print(f"Rank {entry.rank}: {entry.title}")
        print(f"  Strategy: {entry.strategy} | Status: {status_label} | Record: {entry.wins}W - {entry.losses}L - {entry.ties}T")
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
                "hypothesis": asdict(entry.hypothesis),
                "critique": asdict(entry.critique),
            }
            for entry in leaderboard
        ],
        "tournament_matches": [asdict(m) for m in matches],
        "evolution_trace": evolution_trace,
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
    top_entry = leaderboard[0]
    top_hyp = top_entry.hypothesis
    top_gate = top_entry.gatekeeper_eval
    top_critique = top_entry.critique

    md_lines = [
        f"# 基于文献的免疫干预假说与完整因果推理路径报告",
        f"\n**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**模型**: deepseek-v4-flash-0731 | **RAG基础设施**: PaperQA2 (严格页码引用 + 阴性结果检索)",
        f"**评审架构**: 双轮进化回路 (Skeptic Evolution) + 成对位置交换 Elo 锦标赛 (Pairwise Tournament)",
        f"**研究问题**: {research_objective}\n",
        f"## 1. 锦标赛天梯榜 (Tournament Leaderboard)",
        f"| 排名 | 假说标题 | 策略模式 | Elo 积分 | 胜/负/平 | 门禁状态 | 证据支撑度 |",
        f"|---|---|---|---|---|---|---|",
    ]

    for entry in leaderboard:
        status_str = "DISQUALIFIED" if entry.gatekeeper_eval.is_eliminated else "QUALIFIED"
        md_lines.append(
            f"| {entry.rank} | **{entry.title}** | `{entry.strategy}` | **{entry.elo_score:.1f}** | {entry.wins}W-{entry.losses}L-{entry.ties}T | {status_str} | {entry.gatekeeper_eval.evidence_support:.2f} |"
        )

    md_lines.extend([
        f"\n## 2. 最终优胜假说 (Tournament Champion - Rank 1)",
        f"**标题**: {top_hyp.title}",
        f"**Elo Rating**: `{top_entry.elo_score:.1f}` | **策略**: `{top_entry.strategy}`",
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

    print(f"All operations finished in {time.time() - t_start:.1f}s.")
    return report_data


if __name__ == "__main__":
    asyncio.run(run_oa_pipeline())
