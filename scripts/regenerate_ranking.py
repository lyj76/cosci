"""Re-rank and re-generate OA discovery reports using the updated ScientificRankingEngine."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from co_scientist.agents.evidence_extractor import ExtractedEvidenceBundle, GroundedClaim
from co_scientist.agents.pi_synthesizer import ExperimentalProtocol, FinalSynthesizedHypothesis
from co_scientist.agents.ranking_rules import ScientificRankingEngine
from co_scientist.agents.skeptic import EvidenceAuditItem, FalsificationCriterion, SkepticCritique

report_path = Path("data/runs/oa_discovery/oa_discovery_report.json")
with open(report_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Reconstruct evidence bundle
raw_evidence = data["evidence_bundle"]
claims = [
    GroundedClaim(
        claim=c["claim"],
        source_paper=c["source_paper"],
        pages=c["pages"],
        verbatim_quote=c["verbatim_quote"],
        biological_entities=c.get("biological_entities", []),
        confidence=c.get("confidence", "high"),
    )
    for c in raw_evidence["claims"]
]
evidence_bundle = ExtractedEvidenceBundle(
    query=data["research_objective"],
    summary=raw_evidence["summary"],
    claims=claims,
)

ranking_engine = ScientificRankingEngine()
recalculated = []

for item in data["ranked_hypotheses"]:
    strat_key = item["strategy"]
    hyp_dict = item["hypothesis"]
    crit_dict = item["critique"]

    protocols = [
        ExperimentalProtocol(
            phase=p["phase"],
            title=p["title"],
            model_system=p["model_system"],
            intervention=p["intervention"],
            readouts_and_assays=p["readouts_and_assays"],
            success_criteria=p["success_criteria"],
            failure_criteria=p["failure_criteria"],
        )
        for p in hyp_dict.get("experimental_validation_plan", [])
    ]

    final_hyp = FinalSynthesizedHypothesis(
        title=hyp_dict["title"],
        executive_summary=hyp_dict.get("executive_summary", ""),
        primary_hypothesis_statement=hyp_dict.get("primary_hypothesis_statement", ""),
        biological_rationale=hyp_dict.get("biological_rationale", ""),
        target_axis=hyp_dict.get("target_axis", {}),
        complete_reasoning_path=hyp_dict.get("complete_reasoning_path", []),
        evidence_grounding_table=hyp_dict.get("evidence_grounding_table", []),
        contradiction_and_risk_mitigation=hyp_dict.get("contradiction_and_risk_mitigation", []),
        falsification_criteria=hyp_dict.get("falsification_criteria", []),
        experimental_validation_plan=protocols,
        clinical_translational_potential=hyp_dict.get("clinical_translational_potential", ""),
    )

    fals_crit = [
        FalsificationCriterion(
            experiment=fc["experiment"],
            falsifying_observation=fc["falsifying_observation"],
            underlying_logic=fc["underlying_logic"],
        )
        for fc in crit_dict.get("falsification_criteria", [])
    ]

    audited_links = [
        EvidenceAuditItem(
            claim_or_step=al["claim_or_step"],
            grounding_status=al["grounding_status"],
            criticism=al["criticism"],
            relevant_paper_caveats=al.get("relevant_paper_caveats", ""),
        )
        for al in crit_dict.get("audited_links", [])
    ]

    critique = SkepticCritique(
        hypothesis_title=crit_dict["hypothesis_title"],
        verdict=crit_dict["verdict"],
        overall_skeptical_assessment=crit_dict["overall_skeptical_assessment"],
        audited_links=audited_links,
        identified_contradictions=crit_dict.get("identified_contradictions", []),
        biological_risks_and_limitations=crit_dict.get("biological_risks_and_limitations", []),
        falsification_criteria=fals_crit,
        unsupported_core_claim=crit_dict.get("unsupported_core_claim", False),
        improvement_recommendations=crit_dict.get("improvement_recommendations", []),
    )

    evaluation = ranking_engine.evaluate_hypothesis(
        hypothesis=final_hyp,
        evidence=evidence_bundle,
        critique=critique,
    )

    recalculated.append({
        "strategy": strat_key,
        "hypothesis": final_hyp,
        "critique": critique,
        "evaluation": evaluation,
    })

# Sort descending by final score
recalculated.sort(key=lambda x: x["evaluation"].final_score, reverse=True)

print("=" * 70)
print("RE-CALCULATED TOURNAMENT LEADERBOARD & EVALUATION SCORES")
print("=" * 70)
for rank, r in enumerate(recalculated, 1):
    ev = r["evaluation"]
    hyp = r["hypothesis"]
    status = "[ELIMINATED]" if ev.is_eliminated else f"Score: {ev.final_score:.4f}"
    print(f"Rank {rank}: {hyp.title}")
    print(f"  Status: {status} | Support: {ev.evidence_support:.2f} | Citations: {ev.citation_completeness:.2f} | Contradictions: {ev.contradiction_handling:.2f} | Testability: {ev.testability:.2f}")
    if ev.elimination_reason:
        print(f"  Elimination Reason: {ev.elimination_reason}")
    if ev.bonuses:
        print(f"  Bonuses: {', '.join(ev.bonuses)}")
    if ev.penalties:
        print(f"  Penalties: {', '.join(ev.penalties)}")

# Update report data
ranked_data = []
for idx, r in enumerate(recalculated):
    ev = r["evaluation"]
    hyp = r["hypothesis"]
    crit = r["critique"]
    ranked_data.append({
        "rank": idx + 1,
        "strategy": r["strategy"],
        "score": ev.final_score if not math.isinf(ev.final_score) else -999999.0,
        "is_eliminated": ev.is_eliminated,
        "elimination_reason": ev.elimination_reason,
        "evaluation": asdict(ev),
        "hypothesis": {
            **asdict(hyp),
            "experimental_validation_plan": [asdict(p) for p in hyp.experimental_validation_plan],
        },
        "critique": asdict(crit),
    })

data["ranked_hypotheses"] = ranked_data

# Save updated JSON
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Generate updated Markdown
top_entry = recalculated[0]
top_hyp = top_entry["hypothesis"]
top_eval = top_entry["evaluation"]
top_critique = top_entry["critique"]

md_lines = [
    f"# 基于骨关节炎文献的免疫层面干预假说与完整推理路径",
    f"\n**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"**模型**: deepseek-v4-flash-0731 | **RAG基础设施**: PaperQA2 (本地PDF + text-embedding-v3)",
    f"**研究问题**: {data['research_objective']}\n",
    f"## 1. 最终优选假说 (Rank 1)",
    f"**标题**: {top_hyp.title}",
    f"**综合评审得分**: `{top_eval.final_score:.4f}` (证据支持度: {top_eval.evidence_support:.2f}, 引用完整性: {top_eval.citation_completeness:.2f}, 反证与矛盾处理: {top_eval.contradiction_handling:.2f}, 可证伪性: {top_eval.testability:.2f})",
    f"\n### 核心假说陈述 (Primary Hypothesis Statement)",
    f"{top_hyp.primary_hypothesis_statement}\n",
    f"### 目标轴线 (Target Axis)",
    f"- **分子靶点**: {top_hyp.target_axis.get('molecular_target', '')}",
    f"- **免疫检查点受体**: {top_hyp.target_axis.get('checkpoint_receptor', '')}",
    f"- **效应免疫细胞**: {top_hyp.target_axis.get('effector_immune_cells', '')}",
    f"- **作用组织微环境**: {top_hyp.target_axis.get('target_joint_compartment', '')}\n",
    f"### 完整推理路径 (Complete Mechanistic Reasoning Path)",
]

for step in top_hyp.complete_reasoning_path:
    md_lines.append(f"1. **{step.get('step', '')}**")
    md_lines.append(f"   - **机制链**: {step.get('mechanism', '')}")
    md_lines.append(f"   - **文献证据支撑**: {step.get('evidence_support', '')}")

md_lines.append("\n### 文献佐证表 (Grounded Evidence Table)")
md_lines.append("| 来源文献 | 页码 | 提取原文证据 | 支撑论点 |")
md_lines.append("|---|---|---|---|")
for row in top_hyp.evidence_grounding_table:
    quote_snippet = row.get('quote', '').replace('\n', ' ')[:120]
    md_lines.append(f"| {row.get('paper', '')} | {row.get('pages', [])} | \"{quote_snippet}...\" | {row.get('supported_claim', '')} |")

md_lines.append("\n### 矛盾识别与反证风险化解 (Skeptic Critique & Resolution)")
for item in top_hyp.contradiction_and_risk_mitigation:
    md_lines.append(f"- **Skeptic质疑**: {item.get('skeptic_concern', '')}")
    md_lines.append(f"  - **化解与应对策略**: {item.get('resolution_strategy', '')}")

md_lines.append("\n### 可证伪性判定标准 (Falsification Criteria)")
for idx, f_crit in enumerate(top_hyp.falsification_criteria, 1):
    md_lines.append(f"{idx}. **检验实验**: {f_crit.get('test', '')}")
    md_lines.append(f"   - **证伪判据 (Failure Criterion)**: {f_crit.get('falsifying_result', '')}")

md_lines.append("\n### 实验验证方案 (Experimental Validation Protocols)")
for p in top_hyp.experimental_validation_plan:
    md_lines.append(f"#### [{p.phase}] {p.title}")
    md_lines.append(f"- **实验模型**: {p.model_system}")
    md_lines.append(f"- **干预措施**: {p.intervention}")
    md_lines.append(f"- **检测指标**: {', '.join(p.readouts_and_assays)}")
    md_lines.append(f"- **成功标准**: {p.success_criteria}")
    md_lines.append(f"- **证伪/失败标准**: {p.failure_criteria}\n")

md_lines.append("\n### 假说排行榜 (Tournament Leaderboard)")
md_lines.append("| 排名 | 策略 | 得分 | 状态 | 证据支持 | 引用完整 | 矛盾处理 | 可测性 |")
md_lines.append("|---|---|---|---|---|---|---|---|")
for rank, r in enumerate(recalculated, 1):
    ev = r["evaluation"]
    status_str = "QUALIFIED" if not ev.is_eliminated else "ELIMINATED"
    md_lines.append(f"| {rank} | {r['strategy']} | {ev.final_score:.4f} | {status_str} | {ev.evidence_support:.2f} | {ev.citation_completeness:.2f} | {ev.contradiction_handling:.2f} | {ev.testability:.2f} |")

report_md_path = Path("data/runs/oa_discovery/oa_discovery_report.md")
report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
print(f"Successfully re-generated {report_path} and {report_md_path}!")
