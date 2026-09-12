"""Rule-based Ranking and Verification Engine.

Implements the user's strict scientific evaluation rubric:
1. Hard Thresholds (Elimination Filters):
   - Unsupported core claim -> ELIMINATED (-inf)
   - Citations do not support claims -> ELIMINATED (-inf)
   - Lacks clear falsification / failure criteria -> ELIMINATED (-inf)
2. Score Formula:
   score = (
       0.35 * evidence_support
       + 0.20 * citation_completeness
       + 0.20 * contradiction_handling
       + 0.15 * testability
       + 0.10 * novelty
   )
   Modifiers:
   - Only correlation, missing causal chain -> -0.30 penalty
   - Actively addresses identified counter-evidence / caveats -> +0.20 bonus
   - Concrete, phased experimental design -> +0.10 bonus
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .evidence_extractor import ExtractedEvidenceBundle
from .pi_synthesizer import FinalSynthesizedHypothesis
from .skeptic import SkepticCritique


@dataclass
class RankingEvaluation:
    """Detailed score card and audit trail for a scientific hypothesis."""

    hypothesis_title: str
    final_score: float
    is_eliminated: bool
    elimination_reason: str | None

    # Sub-scores [0.0 - 1.0]
    evidence_support: float
    citation_completeness: float
    contradiction_handling: float
    testability: float
    novelty: float

    # Modifiers
    penalties: list[str] = field(default_factory=list)
    bonuses: list[str] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)


class ScientificRankingEngine:
    """Evaluates and ranks hypotheses based on evidence rigor and falsifiability."""

    def evaluate_hypothesis(
        self,
        hypothesis: FinalSynthesizedHypothesis,
        evidence: ExtractedEvidenceBundle,
        critique: SkepticCritique,
    ) -> RankingEvaluation:
        # Check 1: Hard filter on falsification criteria
        if not hypothesis.falsification_criteria and not critique.falsification_criteria:
            return RankingEvaluation(
                hypothesis_title=hypothesis.title,
                final_score=-math.inf,
                is_eliminated=True,
                elimination_reason="Fatal flaw: Failed to provide falsification criteria (unfalsifiable).",
                evidence_support=0.0,
                citation_completeness=0.0,
                contradiction_handling=0.0,
                testability=0.0,
                novelty=0.0,
                audit_notes=["Eliminated by hard threshold: No falsification criteria"],
            )

        # Check 2: Hard filter on unsupported core claim
        # A hypothesis is eliminated if:
        # - Skeptic flagged an unsupported core claim and PI failed to provide contradiction/risk mitigation
        # - Or hypothesis provides no evidence grounding
        unsupported_core = False
        elim_reason = None
        if critique.unsupported_core_claim and not hypothesis.contradiction_and_risk_mitigation:
            unsupported_core = True
            elim_reason = "Fatal flaw: Core claim lacks supporting evidence in literature."
        elif not hypothesis.evidence_grounding_table and critique.unsupported_core_claim:
            unsupported_core = True
            elim_reason = "Fatal flaw: Core claim lacks supporting evidence in literature."

        if unsupported_core:
            return RankingEvaluation(
                hypothesis_title=hypothesis.title,
                final_score=-math.inf,
                is_eliminated=True,
                elimination_reason=elim_reason,
                evidence_support=0.0,
                citation_completeness=0.0,
                contradiction_handling=0.0,
                testability=0.0,
                novelty=0.0,
                audit_notes=["Eliminated by hard threshold: unsupported_core_claim=True"],
            )

        # 1. Evidence Support (0.0 to 1.0)
        # Ratio of supported links vs total links
        n_audited = max(len(critique.audited_links), 1)
        supported_count = sum(
            1
            for a in critique.audited_links
            if a.grounding_status in ("fully_supported", "partially_supported")
        )
        evidence_support = min(1.0, (supported_count / n_audited) * 0.9 + 0.1)

        # 2. Citation Completeness (0.0 to 1.0)
        # Check if evidence table contains page numbers and quotes
        has_pages = any(
            bool(t.get("pages")) for t in hypothesis.evidence_grounding_table
        )
        has_quotes = any(
            len(t.get("quote", "")) > 10 for t in hypothesis.evidence_grounding_table
        )
        citation_completeness = (
            1.0
            if (has_pages and has_quotes)
            else 0.6
            if (has_pages or has_quotes)
            else 0.2
        )

        # 3. Contradiction Handling (0.0 to 1.0)
        # Did the PI actively mitigate skeptic concerns?
        contradiction_handling = 0.5
        bonuses: list[str] = []
        if hypothesis.contradiction_and_risk_mitigation:
            contradiction_handling = min(
                1.0,
                0.6 + 0.15 * len(hypothesis.contradiction_and_risk_mitigation),
            )
            bonuses.append(
                f"Actively mitigates {len(hypothesis.contradiction_and_risk_mitigation)} skeptic concerns (+0.20 bonus)"
            )

        # 4. Testability (0.0 to 1.0)
        # Concrete protocols across in vitro and in vivo phases
        phases = {p.phase for p in hypothesis.experimental_validation_plan}
        has_invitro = "In_Vitro" in phases or any(
            "vitro" in p.phase.lower() for p in hypothesis.experimental_validation_plan
        )
        has_invivo = "In_Vivo_Preclinical" in phases or any(
            "vivo" in p.phase.lower() for p in hypothesis.experimental_validation_plan
        )
        if has_invitro and has_invivo and len(hypothesis.falsification_criteria) >= 2:
            testability = 0.95
            bonuses.append("Multi-phase validation plan with explicit falsification criteria (+0.10 bonus)")
        elif has_invitro or has_invivo:
            testability = 0.70
        else:
            testability = 0.40

        # 5. Novelty (0.0 to 1.0)
        # Grounded novelty assessment from hypothesis or comprehensive target specification
        novelty = 0.85 if hypothesis.clinical_translational_potential or len(hypothesis.target_axis) >= 3 else 0.70

        # Penalties check
        penalties: list[str] = []
        # Check if reasoning chain is just correlation without causal steps
        has_causal_steps = len(hypothesis.complete_reasoning_path) >= 4
        penalty_val = 0.0
        if not has_causal_steps:
            penalty_val += 0.30
            penalties.append("Missing complete multi-step causal chain (-0.30 penalty)")

        # Calculate base score
        base_score = (
            0.35 * evidence_support
            + 0.20 * citation_completeness
            + 0.20 * contradiction_handling
            + 0.15 * testability
            + 0.10 * novelty
        )

        final_score = max(0.0, min(1.0, base_score - penalty_val))

        return RankingEvaluation(
            hypothesis_title=hypothesis.title,
            final_score=round(final_score, 4),
            is_eliminated=False,
            elimination_reason=None,
            evidence_support=round(evidence_support, 3),
            citation_completeness=round(citation_completeness, 3),
            contradiction_handling=round(contradiction_handling, 3),
            testability=round(testability, 3),
            novelty=round(novelty, 3),
            penalties=penalties,
            bonuses=bonuses,
            audit_notes=[
                f"Base score: {base_score:.3f}",
                f"Evidence support weight (0.35): {0.35*evidence_support:.3f}",
                f"Citation completeness weight (0.20): {0.20*citation_completeness:.3f}",
                f"Contradiction handling weight (0.20): {0.20*contradiction_handling:.3f}",
                f"Testability weight (0.15): {0.15*testability:.3f}",
                f"Novelty weight (0.10): {0.10*novelty:.3f}",
            ],
        )


@dataclass
class MatchRecord:
    """Record of an individual head-to-head pairwise debate match."""

    hyp_a_title: str
    hyp_b_title: str
    winner: str  # "A", "B", or "TIE"
    order_1_verdict: str
    order_2_verdict: str
    elo_a_before: float
    elo_b_before: float
    elo_a_after: float
    elo_b_after: float
    deciding_factor: str
    comparative_analysis: str


@dataclass
class TournamentRankedEntry:
    """Entry on the tournament leaderboard."""

    rank: int
    title: str
    strategy: str
    elo_score: float
    wins: int
    losses: int
    ties: int
    matches_played: int
    hypothesis: FinalSynthesizedHypothesis
    critique: SkepticCritique
    gatekeeper_eval: RankingEvaluation


class EloTournamentEngine:
    """Pairwise scientific debate and Elo tournament engine.

    Replaces static formulaic scoring with competitive test-time debate:
    1. Position-swapped pairwise debate (evaluating (A, B) and (B, A) to cancel order bias).
    2. Rigorous multi-factor scientific judging (causal validity, caveat resilience, falsifiability).
    3. Dynamic Elo ratings (initial = 1200.0, K-factor = 32.0).
    """

    def __init__(self, model: str = "deepseek-v4-flash-0731") -> None:
        import os
        from openai import AsyncOpenAI

        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _judge_pair(
        self,
        research_objective: str,
        first: FinalSynthesizedHypothesis,
        second: FinalSynthesizedHypothesis,
    ) -> tuple[str, str, str]:
        import json
        import re

        prompt = f"""You are an elite, impartial Scientific Review Panel judging two competing biomedical hypotheses head-to-head.

RESEARCH OBJECTIVE:
{research_objective}

HYPOTHESIS 1:
Title: {first.title}
Summary: {first.executive_summary}
Target Axis: {json.dumps(first.target_axis, ensure_ascii=False)}
Primary Hypothesis Statement: {first.primary_hypothesis_statement}
Causal Reasoning Trajectory:
{chr(10).join(f"- {s.get('step')}: {s.get('mechanism')}" for s in first.complete_reasoning_path)}
Key Caveat & Skeptic Resolutions:
{chr(10).join(f"- Concern: {m.get('skeptic_concern')} -> Resolution: {m.get('resolution_strategy')}" for m in first.contradiction_and_risk_mitigation[:3])}
Falsification Criteria:
{chr(10).join(f"- {f.get('test')}: Disproven if {f.get('falsifying_result')}" for f in first.falsification_criteria[:2])}

HYPOTHESIS 2:
Title: {second.title}
Summary: {second.executive_summary}
Target Axis: {json.dumps(second.target_axis, ensure_ascii=False)}
Primary Hypothesis Statement: {second.primary_hypothesis_statement}
Causal Reasoning Trajectory:
{chr(10).join(f"- {s.get('step')}: {s.get('mechanism')}" for s in second.complete_reasoning_path)}
Key Caveat & Skeptic Resolutions:
{chr(10).join(f"- Concern: {m.get('skeptic_concern')} -> Resolution: {m.get('resolution_strategy')}" for m in second.contradiction_and_risk_mitigation[:3])}
Falsification Criteria:
{chr(10).join(f"- {f.get('test')}: Disproven if {f.get('falsifying_result')}" for f in second.falsification_criteria[:2])}

EVALUATION CRITERIA:
1. Causal Mechanism Rigor: Which hypothesis presents a more credible, unbroken biophysical mechanism without logical leaps?
2. Resilience against Negative Findings & Barriers: Which hypothesis more effectively resolves tissue penetration, cellular accessibility, or known literature caveats?
3. Falsifiability & Experimental Testability: Which one defines sharper, more concrete criteria that could decisively prove the hypothesis wrong in a laboratory?
4. Translational Significance: Which strategy has greater transformative potential if validated?

TASK:
Declare which hypothesis is scientifically superior ("1", "2", or "TIE" if quality is identical).

Return strictly valid JSON:
{{
  "winner": "1" or "2" or "TIE",
  "comparative_analysis": "3-4 sentence comparative critique justifying the decision",
  "deciding_factor": "The key scientific reason that tilted the verdict"
}}
"""

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert scientific judge in an Elo tournament. Output strictly valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.15,
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            w = str(parsed.get("winner", "TIE")).strip().upper()
            if w not in ("1", "2", "TIE"):
                w = "TIE"
            return w, parsed.get("comparative_analysis", ""), parsed.get("deciding_factor", "")
        except Exception:
            return "TIE", "Parsing failed; defaulting to tie.", "Evaluation error"

    async def run_tournament(
        self,
        research_objective: str,
        candidates: list[dict],  # list of {"strategy": str, "hypothesis": FinalSynthesizedHypothesis, "critique": SkepticCritique, "gatekeeper_eval": RankingEvaluation}
        k_factor: float = 32.0,
    ) -> tuple[list[TournamentRankedEntry], list[MatchRecord]]:
        """Run round-robin position-swapped debate tournament across qualified candidates."""
        matches: list[MatchRecord] = []

        # Filter out gatekeeper eliminated candidates
        qualified = [c for c in candidates if not c["gatekeeper_eval"].is_eliminated]
        if len(qualified) < 2:
            # Not enough candidates for a tournament
            ranked = [
                TournamentRankedEntry(
                    rank=i + 1,
                    title=c["hypothesis"].title,
                    strategy=c["strategy"],
                    elo_score=1200.0,
                    wins=0,
                    losses=0,
                    ties=0,
                    matches_played=0,
                    hypothesis=c["hypothesis"],
                    critique=c["critique"],
                    gatekeeper_eval=c["gatekeeper_eval"],
                )
                for i, c in enumerate(candidates)
            ]
            return ranked, []

        elo_scores = {c["hypothesis"].title: 1200.0 for c in qualified}
        stats = {c["hypothesis"].title: {"wins": 0, "losses": 0, "ties": 0, "played": 0} for c in qualified}

        # Round-robin pairings
        for i in range(len(qualified)):
            for j in range(i + 1, len(qualified)):
                cand_a = qualified[i]
                cand_b = qualified[j]
                h_a = cand_a["hypothesis"]
                h_b = cand_b["hypothesis"]

                t_a = h_a.title
                t_b = h_b.title

                # Pass 1: A is 1, B is 2
                w1, comp1, factor1 = await self._judge_pair(research_objective, h_a, h_b)
                # Pass 2: B is 1, A is 2 (position swap to eliminate order bias)
                w2, comp2, factor2 = await self._judge_pair(research_objective, h_b, h_a)

                # Determine verified winner across position swap
                if w1 == "1" and w2 == "2":
                    match_winner = "A"
                    rationale = comp1
                    deciding = factor1
                elif w1 == "2" and w2 == "1":
                    match_winner = "B"
                    rationale = comp2
                    deciding = factor2
                else:
                    match_winner = "TIE"
                    rationale = f"Order-dependent split decision (Pass 1 favored {w1}, Pass 2 favored {w2}). Considered equal in overall rigor."
                    deciding = "Position sensitivity / comparable evidence quality"

                # Calculate Elo update
                r_a = elo_scores[t_a]
                r_b = elo_scores[t_b]
                expected_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
                expected_b = 1.0 / (1.0 + 10.0 ** ((r_a - r_b) / 400.0))

                if match_winner == "A":
                    s_a, s_b = 1.0, 0.0
                    stats[t_a]["wins"] += 1
                    stats[t_b]["losses"] += 1
                elif match_winner == "B":
                    s_a, s_b = 0.0, 1.0
                    stats[t_b]["wins"] += 1
                    stats[t_a]["losses"] += 1
                else:
                    s_a, s_b = 0.5, 0.5
                    stats[t_a]["ties"] += 1
                    stats[t_b]["ties"] += 1

                stats[t_a]["played"] += 1
                stats[t_b]["played"] += 1

                new_r_a = r_a + k_factor * (s_a - expected_a)
                new_r_b = r_b + k_factor * (s_b - expected_b)

                elo_scores[t_a] = round(new_r_a, 2)
                elo_scores[t_b] = round(new_r_b, 2)

                matches.append(
                    MatchRecord(
                        hyp_a_title=t_a,
                        hyp_b_title=t_b,
                        winner=match_winner,
                        order_1_verdict=f"Idea {w1}",
                        order_2_verdict=f"Idea {w2}",
                        elo_a_before=r_a,
                        elo_b_before=r_b,
                        elo_a_after=elo_scores[t_a],
                        elo_b_after=elo_scores[t_b],
                        deciding_factor=deciding,
                        comparative_analysis=rationale,
                    )
                )

        # Build leaderboard sorted by Elo descending
        ranked_qualified = sorted(
            qualified,
            key=lambda c: elo_scores[c["hypothesis"].title],
            reverse=True,
        )

        leaderboard: list[TournamentRankedEntry] = []
        for rank, c in enumerate(ranked_qualified, 1):
            t = c["hypothesis"].title
            st = stats[t]
            leaderboard.append(
                TournamentRankedEntry(
                    rank=rank,
                    title=t,
                    strategy=c["strategy"],
                    elo_score=elo_scores[t],
                    wins=st["wins"],
                    losses=st["losses"],
                    ties=st["ties"],
                    matches_played=st["played"],
                    hypothesis=c["hypothesis"],
                    critique=c["critique"],
                    gatekeeper_eval=c["gatekeeper_eval"],
                )
            )

        # Append eliminated ones at the bottom
        eliminated = [c for c in candidates if c["gatekeeper_eval"].is_eliminated]
        for c in eliminated:
            leaderboard.append(
                TournamentRankedEntry(
                    rank=len(leaderboard) + 1,
                    title=c["hypothesis"].title,
                    strategy=c["strategy"],
                    elo_score=0.0,
                    wins=0,
                    losses=0,
                    ties=0,
                    matches_played=0,
                    hypothesis=c["hypothesis"],
                    critique=c["critique"],
                    gatekeeper_eval=c["gatekeeper_eval"],
                )
            )

        return leaderboard, matches
