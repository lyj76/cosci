"""Principal Investigator (PI) Synthesis Agent.

Synthesizes evidence, mechanism, and skeptic critiques into a final,
watertight scientific hypothesis with explicit experimental protocols,
falsification safeguards, and contradiction resolutions.
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
from .skeptic import SkepticCritique


@dataclass
class ExperimentalProtocol:
    """Detailed experimental step to validate or falsify the hypothesis."""

    phase: str  # "In_Vitro", "In_Vivo_Preclinical", "Falsification_Control"
    title: str
    model_system: str
    intervention: str
    readouts_and_assays: list[str]
    success_criteria: str
    failure_criteria: str


@dataclass
class FinalSynthesizedHypothesis:
    """The complete scientific hypothesis ready for ranking and execution."""

    title: str
    executive_summary: str
    primary_hypothesis_statement: str
    biological_rationale: str
    target_axis: dict[str, str]  # target, pathway, immune_cell, tissue
    complete_reasoning_path: list[dict[str, str]]
    evidence_grounding_table: list[dict[str, Any]]
    contradiction_and_risk_mitigation: list[dict[str, str]]
    falsification_criteria: list[dict[str, str]]
    experimental_validation_plan: list[ExperimentalProtocol]
    clinical_translational_potential: str


class PIAgent:
    """Principal Investigator Agent that synthesizes and refines the complete hypothesis."""

    def __init__(self, model: str = "deepseek-v4-flash-0731") -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def synthesize(
        self,
        research_objective: str,
        evidence: ExtractedEvidenceBundle,
        mechanism: MechanisticHypothesis,
        critique: SkepticCritique,
    ) -> FinalSynthesizedHypothesis:
        evidence_summary = "\n".join(
            f"- [{c.source_paper}, pp. {c.pages}]: \"{c.verbatim_quote}\" -> {c.claim}"
            for c in evidence.claims
        )

        critique_summary = f"""Overall: {critique.overall_skeptical_assessment}
Identified Contradictions:
{chr(10).join(f"- {c}" for c in critique.identified_contradictions)}
Risks/Limitations:
{chr(10).join(f"- {r}" for r in critique.biological_risks_and_limitations)}
Falsification Criteria:
{chr(10).join(f"- {f.experiment}: Falsified if {f.falsifying_observation}" for f in critique.falsification_criteria)}
Improvement Directives:
{chr(10).join(f"- {rec}" for rec in critique.improvement_recommendations)}
"""

        prompt = f"""You are the Principal Investigator (PI) leading a cutting-edge scientific laboratory in molecular biology and translational therapeutics.

RESEARCH OBJECTIVE:
{research_objective}

GROUNDED LITERATURE EVIDENCE:
{evidence_summary}

NEGATIVE FINDINGS AND GRAPH AUDIT:
{json.dumps([asdict(c) for c in evidence.caveats_and_contradictions], ensure_ascii=False)}
{json.dumps(getattr(evidence, 'graph_feedback', {}), ensure_ascii=False)}
Preserve unresolved claims and scope restrictions in the primary statement.
Do not turn absent effects in sham joints into proof of disease selectivity,
or a proposed delivery solution into demonstrated accessibility. The selected
discriminating experiment is a proposal, not a result. Prioritize minimal tests
before complex therapeutic constructs; do not introduce new ungrounded targets.

PROPOSED MECHANISM (from Mechanism Generator):
Title: {mechanism.title}
Target: {mechanism.molecular_target}
Pathway / Checkpoint: {mechanism.immunological_checkpoint}
Therapeutic Modality: {mechanism.therapeutic_modality}
Expected Phenotype: {mechanism.expected_joint_phenotype}

CRITIQUE & SKEPTIC SCRUTINY (from Skeptic Agent):
{critique_summary}

YOUR TASK AS PI:
Synthesize a comprehensive, rigorous, and logically watertight scientific hypothesis.
You must:
1. Provide a clear PRIMARY HYPOTHESIS STATEMENT specifying the exact molecular-immunological target, cellular actors, and disease outcome.
2. Formulate the COMPLETE REASONING PATH (unbroken causal trajectory from molecular trigger -> target engagement -> cellular response -> disease modification).
3. Directly RESOLVE the Skeptic's identified contradictions, tissue barriers, and caveats:
   - Provide concrete resolution strategies for each concern (e.g. explaining tissue compartment accessibility, delivery modifications, or combinatorial approaches).
4. Formulate explicit FALSIFICATION CRITERIA: exact conditions and failure readouts that would invalidate the hypothesis.
5. Provide a rigorous 3-phase EXPERIMENTAL VALIDATION PLAN:
   - Phase 1: In_Vitro (primary cell or co-culture systems, target engagement assays, functional cellular response)
   - Phase 2: In_Vivo_Preclinical (relevant in vivo disease model, therapeutic delivery, histology, and functional phenotypes)
   - Phase 3: Falsification_Controls (loss-of-function controls, receptor KO, or effector depletion to prove causality and rule out non-specific artifacts)

Return your response strictly as valid JSON matching this schema:
{{
  "title": "Clear, precise scientific hypothesis title",
  "executive_summary": "Executive summary of the proposal",
  "primary_hypothesis_statement": "Formal scientific hypothesis statement",
  "biological_rationale": "Deep biological rationale grounded in the literature",
  "target_axis": {{
    "molecular_target": "Primary molecular target(s)",
    "checkpoint_receptor": "Primary receptor, pathway, or checkpoint axis",
    "effector_immune_cells": "Effector immune cells or target cell populations",
    "target_joint_compartment": "Target tissue microenvironment or anatomical compartment"
  }},
  "complete_reasoning_path": [
    {{
      "step": "Step 1: Cell state & trigger induction",
      "mechanism": "Causal mechanism step description",
      "evidence_support": "Source paper citation or reasoned bridge"
    }}
  ],
  "evidence_grounding_table": [
    {{
      "paper": "Author et al. Year",
      "pages": [1, 2],
      "quote": "Direct quotation from literature snippet",
      "supported_claim": "Claim supported by this quote"
    }}
  ],
  "contradiction_and_risk_mitigation": [
    {{
      "skeptic_concern": "Specific concern or contradiction raised by Skeptic",
      "resolution_strategy": "Concrete biological or experimental resolution"
    }}
  ],
  "falsification_criteria": [
    {{
      "test": "Concrete falsifying experiment",
      "falsifying_result": "Observation that proves the hypothesis wrong"
    }}
  ],
  "experimental_validation_plan": [
    {{
      "phase": "In_Vitro",
      "title": "Protocol title",
      "model_system": "Cell model / co-culture system",
      "intervention": "Specific intervention agent",
      "readouts_and_assays": ["Assay 1", "Assay 2"],
      "success_criteria": "Criterion indicating positive validation",
      "failure_criteria": "Criterion indicating failure / falsification"
    }},
    {{
      "phase": "In_Vivo_Preclinical",
      "title": "Protocol title",
      "model_system": "In vivo disease model",
      "intervention": "Therapeutic regimen and route",
      "readouts_and_assays": ["Histology", "Micro-CT / Imaging", "Biomarker arrays"],
      "success_criteria": "Therapeutic disease modification criterion",
      "failure_criteria": "Lack of efficacy or disease progression"
    }},
    {{
      "phase": "Falsification_Controls",
      "title": "Protocol title",
      "model_system": "Effector-depleted or receptor-knockout model",
      "intervention": "Same therapeutic regimen compared with wild-type controls",
      "readouts_and_assays": ["Phenotype quantification in knockout vs wildtype"],
      "success_criteria": "Efficacy is abolished in KO/depleted model, proving causal necessity",
      "failure_criteria": "Efficacy persists unchanged, falsifying the proposed axis"
    }}
  ],
  "clinical_translational_potential": "Translational feasibility and clinical significance"
}}
"""

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Principal Investigator research synthesis agent. Output strictly valid JSON.",
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

        protocols = [
            ExperimentalProtocol(
                phase=p.get("phase", "In_Vitro"),
                title=p.get("title", ""),
                model_system=p.get("model_system", ""),
                intervention=p.get("intervention", ""),
                readouts_and_assays=p.get("readouts_and_assays", []),
                success_criteria=p.get("success_criteria", ""),
                failure_criteria=p.get("failure_criteria", ""),
            )
            for p in parsed.get("experimental_validation_plan", [])
        ]

        return FinalSynthesizedHypothesis(
            title=parsed.get("title", "Targeting Ganglioside Immune Checkpoint in Osteoarthritis"),
            executive_summary=parsed.get("executive_summary", ""),
            primary_hypothesis_statement=parsed.get("primary_hypothesis_statement", ""),
            biological_rationale=parsed.get("biological_rationale", ""),
            target_axis=parsed.get("target_axis", {}),
            complete_reasoning_path=parsed.get("complete_reasoning_path", []),
            evidence_grounding_table=parsed.get("evidence_grounding_table", []),
            contradiction_and_risk_mitigation=parsed.get(
                "contradiction_and_risk_mitigation", []
            ),
            falsification_criteria=parsed.get("falsification_criteria", []),
            experimental_validation_plan=protocols,
            clinical_translational_potential=parsed.get(
                "clinical_translational_potential", ""
            ),
        )
