"""Mechanism Generator Agent.

Constructs detailed, causal biological mechanism chains:
Cellular state -> Molecular trigger / Receptor -> Immune Checkpoint ->
Phenotypic consequence -> Targeted Intervention -> Joint outcome in OA.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .evidence_extractor import ExtractedEvidenceBundle, GroundedClaim


@dataclass
class MechanismStep:
    """A single causal step in the biological reasoning pathway."""

    step_number: int
    source_entity: str
    target_entity: str
    interaction_type: str  # e.g., "upregulates", "binds_and_inhibits", "suppresses_clearance"
    biological_description: str
    supporting_evidence_claims: list[str] = field(default_factory=list)


@dataclass
class MechanisticHypothesis:
    """A structured mechanistic hypothesis proposing an immunological intervention in OA."""

    title: str
    summary: str
    target_cells: list[str]
    molecular_target: str
    immunological_checkpoint: str
    therapeutic_modality: str  # e.g. "Intra-articular monoclonal antibody", "Enzymatic inhibitor"
    causal_chain: list[MechanismStep]
    expected_joint_phenotype: str
    evidence_grounding_summary: str
    novelty_assessment: str


class MechanismGeneratorAgent:
    """Generates structured biological hypotheses and causal mechanism chains from grounded evidence."""

    def __init__(self, model: str = "deepseek-v4-flash-0731") -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def propose_strategies(
        self,
        research_objective: str,
        evidence: ExtractedEvidenceBundle,
        n: int = 3,
    ) -> list[tuple[str, str]]:
        """Dynamically propose N distinct, orthogonal scientific intervention strategies grounded in evidence."""
        claims_text = "\n".join(
            f"- {c.claim} (Source: {c.source_paper})"
            for c in evidence.claims[:10]
        )
        prompt = f"""You are a principal scientific strategist in molecular biology and translational therapeutics.
Based on the following research objective and grounded evidence from literature, propose {n} DISTINCT, COMPETING, and ORTHOGONAL intervention strategies.

Research Objective:
{research_objective}

Literature Evidence Summary:
{evidence.summary}

Key Grounded Claims:
{claims_text}

TASK:
Propose {n} diverse therapeutic strategies (e.g., direct receptor blockade, enzymatic synthesis inhibition, compartment-targeted or delivery-enhanced approaches, combination modalities, or phenotypic reprogramming).
Each strategy must have a short slug key (snake_case) and a concise, actionable scientific description.

Return strictly valid JSON:
{{
  "strategies": [
    {{
      "key": "strategy_slug_name",
      "description": "Concise scientific description of this intervention angle"
    }}
  ]
}}
"""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a scientific strategist. Output strictly valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            results = [
                (s.get("key", f"strategy_{i+1}"), s.get("description", ""))
                for i, s in enumerate(parsed.get("strategies", []))
                if s.get("key") and s.get("description")
            ]
            if len(results) >= n:
                return results[:n]
        except Exception:
            pass

        # Robust fallbacks if LLM fails
        return [
            ("targeted_checkpoint_inhibition", "Direct targeting of the primary inhibitory receptor/ligand axis identified in literature"),
            ("upstream_synthesis_inhibition", "Inhibition of key upstream biosynthetic enzymes or regulatory factors"),
            ("compartment_enhanced_combination", "Combined therapeutic targeting addressing tissue penetration barriers and microenvironment"),
        ][:n]

    async def generate_hypothesis(
        self,
        research_objective: str,
        evidence: ExtractedEvidenceBundle,
        focus_strategy: str = "targeted_intervention",
        prior_feedback: str | None = None,
    ) -> MechanisticHypothesis:
        evidence_text = "\n".join(
            f"[{i+1}] Claim: {c.claim}\n    Paper: {c.source_paper} (Pages: {c.pages})\n    Quote: \"{c.verbatim_quote}\"\n    Entities: {', '.join(c.biological_entities)}"
            for i, c in enumerate(evidence.claims)
        )

        caveats_text = ""
        if hasattr(evidence, "caveats_and_contradictions") and evidence.caveats_and_contradictions:
            caveats_text = "\n".join(
                f"- Caveat/Limitation: {c.claim} [Source: {c.source_paper}]"
                for c in evidence.caveats_and_contradictions
            )

        prompt = f"""You are an elite Mechanistic Biology and Translational Medicine AI Agent.
Your goal is to construct a rigorous, testable causal hypothesis proposing an INTERVENTION strictly grounded in the provided literature evidence.

Research Objective:
{research_objective}

Intervention Strategy Focus:
{focus_strategy}

Grounded Literature Evidence:
{evidence_text}

Literature Summary:
{evidence.summary}

{f"Literature Limitations & Caveats to Respect:{chr(10)}{caveats_text}" if caveats_text else ""}
{f"Prior Critique / Feedback to Address:{chr(10)}{prior_feedback}" if prior_feedback else ""}

REQUIREMENTS:
1. Construct an unbroken, step-by-step causal mechanism chain:
   Step 1: Cell state & trigger (disease microenvironment, aberrant expression of target pathway/marker, cellular stress/pathology).
   Step 2: Molecular/immune checkpoint or signaling interaction (ligand-receptor binding, downstream inhibitory/activating cascades).
   Step 3: Pathological persistence (how the untreated axis drives disease chronicity, tissue remodeling, or tissue damage).
   Step 4: Targeted Therapeutic Intervention (specific therapeutic modality, delivery route, and binding/inhibition action).
   Step 5: Effector/cellular consequence (restoration of clearance, pathway modulation, cellular reprogramming).
   Step 6: Tissue/systemic phenotypic outcome (functional preservation, disease resolution, halting pathology).
2. Ground each step in the provided evidence wherever direct evidence exists; if a step is a reasoned bridge, explicitly state the biological rationale.
3. Specify the therapeutic modality (e.g. monoclonal antibody, small molecule, delivery-engineered conjugate) and route.

Return your response strictly as valid JSON matching this schema:
{{
  "title": "Precise, descriptive scientific hypothesis title",
  "summary": "2-3 sentence overview of the hypothesis",
  "target_cells": ["primary cell types involved"],
  "molecular_target": "Specific molecular target(s) / pathway(s)",
  "immunological_checkpoint": "Primary receptor-ligand or signaling axis",
  "therapeutic_modality": "Specific therapeutic agent and delivery route",
  "causal_chain": [
    {{
      "step_number": 1,
      "source_entity": "Upstream trigger or cell state",
      "target_entity": "Target molecule or pathway modulation",
      "interaction_type": "binding / transcriptional_induction / suppression / etc.",
      "biological_description": "Detailed biological mechanism of this step",
      "supporting_evidence_claims": ["Direct claim or logical connection supported by literature"]
    }}
  ],
  "expected_joint_phenotype": "Expected tissue-level phenotypic outcome and disease modification",
  "evidence_grounding_summary": "How this hypothesis directly connects the findings across the papers",
  "novelty_assessment": "How this moves beyond conventional symptomatic or non-specific therapies"
}}
"""

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a causal mechanism generation agent. Output strictly valid JSON.",
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

        steps = [
            MechanismStep(
                step_number=s.get("step_number", i + 1),
                source_entity=s.get("source_entity", ""),
                target_entity=s.get("target_entity", ""),
                interaction_type=s.get("interaction_type", ""),
                biological_description=s.get("biological_description", ""),
                supporting_evidence_claims=s.get("supporting_evidence_claims", []),
            )
            for i, s in enumerate(parsed.get("causal_chain", []))
        ]

        return MechanisticHypothesis(
            title=parsed.get("title", f"Targeted Intervention for {research_objective[:40]}"),
            summary=parsed.get("summary", ""),
            target_cells=parsed.get("target_cells", []),
            molecular_target=parsed.get("molecular_target", ""),
            immunological_checkpoint=parsed.get("immunological_checkpoint", ""),
            therapeutic_modality=parsed.get("therapeutic_modality", ""),
            causal_chain=steps,
            expected_joint_phenotype=parsed.get("expected_joint_phenotype", ""),
            evidence_grounding_summary=parsed.get("evidence_grounding_summary", ""),
            novelty_assessment=parsed.get("novelty_assessment", ""),
        )

    async def evolve_hypothesis(
        self,
        hypothesis: MechanisticHypothesis,
        critique: Any,
        evidence: ExtractedEvidenceBundle,
        mode: str = "address_skeptic_critique",
    ) -> MechanisticHypothesis:
        """Evolve and mutate a hypothesis to directly resolve Skeptic critiques and biological barriers."""
        contradictions = getattr(critique, "identified_contradictions", [])
        risks = getattr(critique, "biological_risks_and_limitations", [])
        recommendations = getattr(critique, "improvement_recommendations", [])
        assessment = getattr(critique, "overall_skeptical_assessment", "")

        chain_text = "\n".join(
            f"Step {s.step_number}: {s.source_entity} -> {s.target_entity} ({s.interaction_type})\n  Details: {s.biological_description}"
            for s in hypothesis.causal_chain
        )

        prompt = f"""You are an elite Scientific Evolution and Hypothesis Refinement Agent.
Your task is to EVOLVE and MUTATE an existing hypothesis to resolve critical vulnerabilities, contradictions, and biological barriers identified by an adversarial Skeptic Agent.

ORIGINAL HYPOTHESIS:
Title: {hypothesis.title}
Summary: {hypothesis.summary}
Target Axis: {hypothesis.molecular_target} via {hypothesis.immunological_checkpoint}
Therapeutic Modality: {hypothesis.therapeutic_modality}
Expected Phenotype: {hypothesis.expected_joint_phenotype}

Original Mechanism Chain:
{chain_text}

SKEPTIC CRITIQUE & OBJECTIONS:
Assessment: {assessment}
Identified Contradictions:
{chr(10).join(f"- {c}" for c in contradictions)}
Biological Risks & Delivery Limitations:
{chr(10).join(f"- {r}" for r in risks)}
Improvement Directives:
{chr(10).join(f"- {rec}" for rec in recommendations)}

EVOLUTION OBJECTIVE ({mode}):
1. Overcome the Skeptic's objections directly (e.g. if tissue penetration was limited, evolve the delivery vehicle or engineer a smaller modality like nanobody/conjugate; if systemic toxicity or off-target risk was raised, engineer cell-type or condition-selective targeting; if correlation was conflated with causation, refine the causal chain).
2. Maintain strong grounding in the available evidence while proposing an upgraded, logically watertight solution.
3. Preserve the core scientific insight while elevating translational and mechanistic rigor.

Return the evolved hypothesis strictly as valid JSON matching this schema:
{{
  "title": "Evolved, upgraded hypothesis title",
  "summary": "Updated 2-3 sentence overview explaining the evolved intervention",
  "target_cells": ["refined target cell types"],
  "molecular_target": "Refined molecular target",
  "immunological_checkpoint": "Refined checkpoint / pathway axis",
  "therapeutic_modality": "Upgraded therapeutic modality, delivery system, or combination",
  "causal_chain": [
    {{
      "step_number": 1,
      "source_entity": "Entity",
      "target_entity": "Entity",
      "interaction_type": "type",
      "biological_description": "Detailed description showing how previous critique is resolved",
      "supporting_evidence_claims": ["Supporting claims"]
    }}
  ],
  "expected_joint_phenotype": "Phenotypic outcome with resolved limitations",
  "evidence_grounding_summary": "How this evolved version harmonizes with all evidence including negative findings",
  "novelty_assessment": "Why this evolved version is superior to the original"
}}
"""

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a scientific evolution agent. Output strictly valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.25,
        )

        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            parsed = json.loads(m.group(1)) if m else {}

        steps = [
            MechanismStep(
                step_number=s.get("step_number", i + 1),
                source_entity=s.get("source_entity", ""),
                target_entity=s.get("target_entity", ""),
                interaction_type=s.get("interaction_type", ""),
                biological_description=s.get("biological_description", ""),
                supporting_evidence_claims=s.get("supporting_evidence_claims", []),
            )
            for i, s in enumerate(parsed.get("causal_chain", []))
        ]

        if not steps:
            # Fallback to original chain if parsing failed
            steps = hypothesis.causal_chain

        return MechanisticHypothesis(
            title=parsed.get("title", f"Evolved: {hypothesis.title}"),
            summary=parsed.get("summary", hypothesis.summary),
            target_cells=parsed.get("target_cells", hypothesis.target_cells),
            molecular_target=parsed.get("molecular_target", hypothesis.molecular_target),
            immunological_checkpoint=parsed.get("immunological_checkpoint", hypothesis.immunological_checkpoint),
            therapeutic_modality=parsed.get("therapeutic_modality", hypothesis.therapeutic_modality),
            causal_chain=steps,
            expected_joint_phenotype=parsed.get("expected_joint_phenotype", hypothesis.expected_joint_phenotype),
            evidence_grounding_summary=parsed.get("evidence_grounding_summary", hypothesis.evidence_grounding_summary),
            novelty_assessment=parsed.get("novelty_assessment", hypothesis.novelty_assessment),
        )
