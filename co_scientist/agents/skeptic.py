"""Skeptic Agent.

Performs adversarial critique, identifies contradictions, limitations,
and logic leaps, and formulates strict falsification criteria.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .evidence_extractor import ExtractedEvidenceBundle
from .mechanism_generator import MechanisticHypothesis


@dataclass
class EvidenceAuditItem:
    """Audit of a single claim or link in the mechanism chain."""

    claim_or_step: str
    grounding_status: str  # "fully_supported", "partially_supported", "unsupported_leap", "contradicted"
    criticism: str
    relevant_paper_caveats: str


@dataclass
class FalsificationCriterion:
    """A concrete experimental result that would falsify the hypothesis."""

    experiment: str
    falsifying_observation: str
    underlying_logic: str


@dataclass
class SkepticCritique:
    """Comprehensive adversarial review produced by the Skeptic Agent."""

    hypothesis_title: str
    verdict: str  # "PASS_WITH_RESERVATIONS", "MAJOR_REVISIONS_NEEDED", "FATAL_FLAW"
    overall_skeptical_assessment: str
    audited_links: list[EvidenceAuditItem] = field(default_factory=list)
    identified_contradictions: list[str] = field(default_factory=list)
    biological_risks_and_limitations: list[str] = field(default_factory=list)
    falsification_criteria: list[FalsificationCriterion] = field(default_factory=list)
    unsupported_core_claim: bool = False
    improvement_recommendations: list[str] = field(default_factory=list)


class SkepticAgent:
    """Adversarial critic dedicated to finding flaws, contradictions, and falsification criteria."""

    def __init__(self, model: str = "deepseek-v4-flash-0731") -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def critique(
        self,
        hypothesis: MechanisticHypothesis,
        evidence: ExtractedEvidenceBundle,
    ) -> SkepticCritique:
        evidence_text = "\n".join(
            f"[{i+1}] Claim: {c.claim}\n    Paper: {c.source_paper} (Pages: {c.pages})\n    Quote: \"{c.verbatim_quote}\""
            for i, c in enumerate(evidence.claims)
        )

        chain_text = "\n".join(
            f"Step {s.step_number}: {s.source_entity} -> {s.target_entity} ({s.interaction_type})\n  Details: {s.biological_description}"
            for s in hypothesis.causal_chain
        )

        caveats_text = ""
        if hasattr(evidence, "caveats_and_contradictions") and evidence.caveats_and_contradictions:
            caveats_text = "\n".join(
                f"[{i+1}] Caveat / Negative Finding: {c.claim}\n    Paper: {c.source_paper} (Pages: {c.pages})\n    Quote: \"{c.verbatim_quote}\""
                for i, c in enumerate(evidence.caveats_and_contradictions)
            )
        else:
            caveats_text = "(No explicit negative findings extracted from corpus; evaluate based on biological principles, physical transport/penetration limits, and cell-type specificity)"

        prompt = f"""You are an exceptionally rigorous, adversarial Scientific Peer Reviewer and Skeptic Agent.
Your duty is to challenge the proposed hypothesis, uncover unstated assumptions, detect contradictions with existing literature and negative findings, evaluate delivery feasibility, and design experiments that could DISPROVE it.

Proposed Hypothesis:
Title: {hypothesis.title}
Summary: {hypothesis.summary}
Target Cells: {', '.join(hypothesis.target_cells)}
Molecular Target: {hypothesis.molecular_target}
Immune Checkpoint / Axis: {hypothesis.immunological_checkpoint}
Therapeutic Modality: {hypothesis.therapeutic_modality}
Expected Phenotype: {hypothesis.expected_joint_phenotype}

Proposed Mechanism Chain:
{chain_text}

Available Grounded Literature Evidence (Affirmative):
{evidence_text}

LITERATURE CAVEATS, NEGATIVE FINDINGS & BIOLOGICAL BARRIERS:
{caveats_text}

TASK:
1. Audit each step of the mechanism chain. Flag any unsupported leaps or steps that confuse correlation with causation.
2. Check whether the proposed therapeutic modality physically reaches the target cells (e.g. vascular vs avascular tissue, matrix barrier, off-target toxicity).
3. Identify explicit contradictions or caveats between the hypothesis claims and the literature negative findings.
4. Formulate at least 3 rigorous FALSIFICATION CRITERIA: What concrete experimental outcome would prove this hypothesis WRONG?
5. Decide if there is an unsupported core claim (`unsupported_core_claim: true/false`).
6. Provide actionable recommendations for how the hypothesis can be evolved or refined to be watertight.

Return your response strictly as valid JSON matching this schema:
{{
  "verdict": "PASS_WITH_RESERVATIONS" or "MAJOR_REVISIONS_NEEDED" or "FATAL_FLAW",
  "overall_skeptical_assessment": "Crisp, rigorous 2-3 sentence critique",
  "audited_links": [
    {{
      "claim_or_step": "Step description",
      "grounding_status": "fully_supported" or "partially_supported" or "unsupported_leap" or "contradicted",
      "criticism": "Specific concern or limitation",
      "relevant_paper_caveats": "Exact finding or caveat from paper that tempers this claim"
    }}
  ],
  "identified_contradictions": [
    "Contradiction 1: e.g. claiming broad tissue efficacy when literature reported benefits were restricted to specific compartments."
  ],
  "biological_risks_and_limitations": [
    "Limitation 1: e.g. molecular weight / charge restricts penetration into dense ECM",
    "Limitation 2: e.g. potential systemic immune activation or on-target off-tissue toxicity"
  ],
  "falsification_criteria": [
    {{
      "experiment": "Exact experimental setup (e.g. therapeutic treatment in effector-depleted or receptor-knockout disease models)",
      "falsifying_observation": "Observation that disproves hypothesis (e.g. therapeutic protection persists identically despite mediator ablation)",
      "underlying_logic": "Why this observation demonstrates the hypothesized mechanism is not necessary or true"
    }}
  ],
  "unsupported_core_claim": false,
  "improvement_recommendations": [
    "Actionable recommendation 1 to evolve and resolve this critique",
    "Actionable recommendation 2 to evolve and resolve this critique"
  ]
}}
"""

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an adversarial peer reviewer and skeptic agent. Output strictly valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            parsed = json.loads(m.group(1)) if m else {}

        audited = [
            EvidenceAuditItem(
                claim_or_step=a.get("claim_or_step", ""),
                grounding_status=a.get("grounding_status", "partially_supported"),
                criticism=a.get("criticism", ""),
                relevant_paper_caveats=a.get("relevant_paper_caveats", ""),
            )
            for a in parsed.get("audited_links", [])
        ]

        falsification = [
            FalsificationCriterion(
                experiment=f.get("experiment", ""),
                falsifying_observation=f.get("falsifying_observation", ""),
                underlying_logic=f.get("underlying_logic", ""),
            )
            for f in parsed.get("falsification_criteria", [])
        ]

        return SkepticCritique(
            hypothesis_title=hypothesis.title,
            verdict=parsed.get("verdict", "PASS_WITH_RESERVATIONS"),
            overall_skeptical_assessment=parsed.get(
                "overall_skeptical_assessment", ""
            ),
            audited_links=audited,
            identified_contradictions=parsed.get("identified_contradictions", []),
            biological_risks_and_limitations=parsed.get(
                "biological_risks_and_limitations", []
            ),
            falsification_criteria=falsification,
            unsupported_core_claim=parsed.get("unsupported_core_claim", False),
            improvement_recommendations=parsed.get("improvement_recommendations", []),
        )
