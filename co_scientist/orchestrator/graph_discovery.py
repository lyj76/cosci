"""Bounded local-corpus actions for the OA entry point.

Retrieval relevance is never a causal confidence. A model attribution is kept
with its verbatim passage and scope judgement; it is not experimental proof.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..models.scientific_graph import ScientificEdge, ScientificGraph, ScientificNode


def node_id(kind: str, label: str) -> str:
    return f"{kind}-{hashlib.sha256(label.encode()).hexdigest()[:16]}"


def build_graph(evidence, mechanisms) -> ScientificGraph:
    graph = ScientificGraph()
    for polarity, claims in (("claim", evidence.claims), ("caveat", evidence.caveats_and_contradictions)):
        for claim in claims:
            key = node_id("evidence", json.dumps(asdict(claim), sort_keys=True))
            graph.nodes[key] = ScientificNode(
                id=key, kind="evidence", label=claim.claim, description=claim.verbatim_quote,
                metadata={"paper": claim.source_paper, "pages": str(claim.pages),
                          "origin": polarity, "verification": "extracted_not_edge_verified"},
            )
    for strategy, hyp in mechanisms:
        for index, step in enumerate(hyp.causal_chain):
            # Isolate steps and strategies: equal entity labels do not imply
            # equal causal claims, model systems, or interventions.
            identity = f"{strategy}:{index}:{json.dumps(asdict(step), sort_keys=True)}"
            src, dst = node_id("source", identity), node_id("claim", identity)
            graph.nodes[src] = ScientificNode(id=src, kind="mechanism", label=step.source_entity)
            graph.nodes[dst] = ScientificNode(
                id=dst, kind="claim", label=step.target_entity,
                description=step.biological_description,
                metadata={"strategy": strategy, "step": str(step.step_number),
                          "interaction": step.interaction_type},
            )
            # A generator's citation string is a proposed attribution, not an audit.
            graph.edges.append(ScientificEdge(source=src, target=dst, relation="bridges", unresolved=True))
    return graph


class Attribution(BaseModel):
    snippet_index: int = Field(ge=0)
    relation: Literal["supports", "contradicts", "entity_only", "context_only", "unknown"]
    scope_matches: bool
    quote: str = ""
    rationale: str


class Attributions(BaseModel):
    items: list[Attribution]


class Experiment(BaseModel):
    candidate_ids: list[str] = Field(min_length=2)
    distinction: str = Field(min_length=1)
    intervention: str = Field(min_length=1)
    controls: list[str] = Field(min_length=1)
    predictions: dict[str, str]
    falsification_rule: str = Field(min_length=1)
    next_decisions: dict[str, str]


class StructuredOutputError(RuntimeError):
    """The provider returned no usable structured answer after bounded retries."""


# Reasoning models spend output budget on hidden reasoning tokens before a
# visible body, so the JSON cap must leave headroom. 24576 is an upper target;
# if a gateway rejects it, the caller falls back to the default budget.
_DEFAULT_STRUCTURED_BUDGET = 16384
_RETRY_STRUCTURED_BUDGET = 24576
_MAX_STRUCTURED_ATTEMPTS = 3


async def structured_call(agent, prompt: str, schema, *, on_attempt=None):
    """Validate the actual response; never disguise an empty body as `{}`.

    JSON mode only promises JSON syntax, not the requested schema. Reasoning
    models also use up the output budget before producing a visible body, so a
    `length` finish with no/partial JSON is a capacity problem and must retry
    with a larger budget instead of being trusted as a schema failure. Keep
    diagnostics for every attempt and fail closed only after bounded retries.
    """
    messages = [
            {"role": "system", "content": "Audit only supplied local-corpus evidence. Do not invent sources or experimental results. Return JSON matching the schema."},
            {"role": "user", "content": prompt + "\nSchema:\n" + json.dumps(schema.model_json_schema())},
    ]
    client = agent.client.with_options(max_retries=0)
    budget = _DEFAULT_STRUCTURED_BUDGET
    reasoning_model = False
    for attempt in range(1, _MAX_STRUCTURED_ATTEMPTS + 1):
        record = {"schema": schema.__name__, "model": agent.model, "attempt": attempt,
                  "timestamp": time.time(), "max_tokens": budget,
                  "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            response = await client.chat.completions.create(
                model=agent.model, response_format={"type": "json_object"},
                temperature=0.0, max_tokens=budget, timeout=180, messages=messages,
            )
        except Exception as exc:
            record.update(status="request_failed", error_type=type(exc).__name__)
            if on_attempt:
                on_attempt(record)
            # A gateway that caps max_tokens below the requested value should
            # retry at the default budget rather than abort the whole phase.
            # Raising the budget only ever happens after a truncation, so any
            # request failure at the raised budget warrants one default retry.
            if budget > _DEFAULT_STRUCTURED_BUDGET:
                budget = _DEFAULT_STRUCTURED_BUDGET
                continue
            raise
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        content = getattr(message, "content", None)
        finish = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        detail = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        reasoning_tokens = getattr(detail, "reasoning_tokens", 0) if detail is not None else 0
        if reasoning_tokens:
            reasoning_model = True
        record.update(
            response_id=getattr(response, "id", None), finish_reason=finish,
            content=content, reasoning_tokens=int(reasoning_tokens),
            reasoning_chars=len(getattr(message, "reasoning_content", None) or ""),
            refusal=bool(getattr(message, "refusal", None)),
            usage=usage.model_dump(mode="json") if usage is not None else None,
        )
        parsed = None
        if record["refusal"] or finish == "content_filter":
            error = "refused_or_filtered"
        elif finish == "length":
            error = "output_truncated"
        elif not isinstance(content, str) or not content.strip():
            error = "empty_content"
        else:
            body = content.strip()
            fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.DOTALL)
            if fence:
                body = fence.group(1)
            try:
                parsed = schema.model_validate_json(body)
                error = None
            except ValidationError as exc:
                error = "schema_validation_failed"
                record["validation_errors"] = [
                    {"loc": list(e["loc"]), "type": e["type"]}
                    for e in exc.errors(include_input=False, include_url=False)
                ]
        record.update(status="valid" if parsed is not None else "invalid", error=error)
        if on_attempt:
            on_attempt(record)
        if parsed is not None:
            return parsed
        if error == "refused_or_filtered" or attempt >= _MAX_STRUCTURED_ATTEMPTS:
            raise StructuredOutputError(
                f"{schema.__name__}: {error}; finish_reason={finish}; "
                f"attempts={attempt}. Cached retrieval is retained; inspect structured_diagnostics checkpoints."
            )
        # Truncated/empty bodies usually mean the output cap was hit while the
        # model was still reasoning or writing: raise the budget, never reuse it.
        if reasoning_model or error in {"empty_content", "output_truncated"}:
            budget = _RETRY_STRUCTURED_BUDGET
        print(f"  [structured retry {attempt}/{_MAX_STRUCTURED_ATTEMPTS - 1}] "
              f"{schema.__name__}: {error}; finish={finish}", flush=True)
        messages.append({"role": "user", "content":
            f"The previous response failed validation ({error}). Return only the complete JSON object "
            f"with these required top-level fields: {', '.join(schema.model_fields)}. "
            "Keep rationales and quotes concise, but do not omit items or invent evidence. "
            "Use unknown when the passages do not establish the claim."})


def select_gap(graph, critiques, attempted):
    candidates = []
    for edge in graph.unresolved_edges():
        if edge.target in attempted:
            continue
        strategy = graph.nodes[edge.target].metadata["strategy"]
        critique = critiques.get(strategy)
        step = graph.nodes[edge.target].metadata["step"]
        step_findings = [link.grounding_status for link in critique.audited_links
                         if re.search(rf"\bstep\s+{re.escape(step)}\b", link.claim_or_step, re.I)] if critique else []
        step_priority = max(({"contradicted": 3, "unsupported_leap": 2,
                              "partially_supported": 1}.get(s, 0) for s in step_findings), default=0)
        # Explicit scheduling heuristic, NOT expected information gain.
        priority = (step_priority, int(bool(critique and critique.unsupported_core_claim)),
                    int(bool(critique and critique.verdict == "FATAL_FLAW")))
        candidates.append((priority, edge))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def apply_attributions(graph, edge, snippets, assessment):
    records = []
    used = set()
    for item in assessment.items:
        if item.snippet_index >= len(snippets) or item.snippet_index in used:
            raise ValueError("invalid or duplicate snippet index")
        used.add(item.snippet_index)
    # Omitted passages remain unknown, so a judge cannot quietly discard them.
    by_index = {item.snippet_index: item for item in assessment.items}
    supported, contradicted = [], []
    for index, snippet in enumerate(snippets):
        item = by_index.get(index)
        relation = item.relation if item else "unknown"
        quote = item.quote if item else ""
        verified = bool(quote.strip() and snippet.pages and snippet.citation and
                        " ".join(quote.split()) in " ".join(snippet.text.split()))
        if relation in {"supports", "contradicts"} and not (
            verified and item and item.scope_matches
        ):
            relation = "context_only" if verified else "unknown"
        evidence_id = node_id("evidence", f"{snippet.citation}:{snippet.text}")
        graph.nodes[evidence_id] = ScientificNode(
            id=evidence_id, kind="evidence", label=snippet.citation,
            description=snippet.text, status="unknown",
            metadata={"paper": snippet.paper_name, "pages": json.dumps(snippet.pages),
                      "retrieval_score": str(snippet.relevance_score)},
        )
        link = ScientificEdge(source=evidence_id, target=edge.target, relation=relation,
                              evidence_ids=[evidence_id])
        if link not in graph.edges:
            graph.edges.append(link)
        if relation == "supports":
            supported.append(evidence_id)
        elif relation == "contradicts":
            contradicted.append(evidence_id)
        records.append({"evidence_id": evidence_id, "relation": relation,
                        "quote": quote, "quote_verified": verified,
                        "scope_matches": item.scope_matches if item else False,
                        "rationale": item.rationale if item else "not assessed"})
    edge.evidence_ids = list(dict.fromkeys([*edge.evidence_ids, *supported, *contradicted]))
    # Conflicting evidence always leaves the claim open. Scores stay uncalibrated.
    edge.unresolved = not supported or bool(contradicted)
    graph.nodes[edge.target].status = (
        "contradicted" if contradicted else "inferred" if supported else "unknown"
    )
    return records


async def run_evidence_loop(graph, retriever, agent, objective, critiques, max_queries,
                            checkpoint, restored=None):
    trace = list(restored or [])
    attempted = {row["target"] for row in trace if "target" in row}
    while len(attempted) < max_queries:
        edge = select_gap(graph, dict(critiques), attempted)
        if edge is None:
            break
        target = graph.nodes[edge.target]
        strategy = target.metadata["strategy"]
        critique = dict(critiques).get(strategy)
        query = (
            f"Within the supplied PDFs only, what direct evidence supports OR contradicts this claim: "
            f"{target.description}? Distinguish association from intervention and other diseases, "
            f"species, cell types or compartments from the requested context: {objective}. "
            "Explicitly state if the corpus does not establish this relation."
        )
        print(f"  [graph] query {len(attempted) + 1}/{max_queries}: {strategy} / {target.description}", flush=True)
        cache_key = node_id("retrieval", query)
        # Cache raw PaperQA result before the attribution call: disconnection
        # during the judge step need not repeat a paid retrieval.
        retrieval = checkpoint("load", cache_key, None)
        if retrieval is None:
            retrieval = await retriever.query(query)
            checkpoint("save", cache_key, retrieval)
        snippets = retrieval.evidence_snippets[:5]
        diagnostic_key = f"structured_diagnostics_{cache_key}"
        diagnostics = checkpoint("load", diagnostic_key, None) or []
        def log_attempt(record):
            diagnostics.append(record)
            checkpoint("save", diagnostic_key, diagnostics)
        if snippets:
            assessment = await structured_call(agent,
                f"Objective: {objective}\nTarget claim: {target.description}\n"
                f"Skeptic: {critique.overall_skeptical_assessment if critique else ''}\n"
                "For EACH indexed snippet judge the ENTIRE causal claim, direction and context. "
                "Entity co-occurrence is entity_only, other disease/species/cell context is context_only. "
                "A keyword such as receptor or checkpoint is NOT support. Negative findings only "
                "contradict if they address this exact claim and context. Quote must be verbatim from "
                "the snippet (not the generated answer). Missing scope or causal evidence is unknown.\n"
                + json.dumps([{"index": i, **asdict(s)} for i, s in enumerate(snippets)], ensure_ascii=False),
                Attributions, on_attempt=log_attempt)
        else:
            assessment = Attributions(items=[])
        before = len(graph.unresolved_edges())
        records = apply_attributions(graph, edge, snippets, assessment)
        trace.append({"action": "retrieve_missing_evidence", "strategy": strategy,
                      "target": edge.target, "claim": target.description, "query": query,
                      "reason": "unresolved claim; prioritize unsupported-core/fatal reviews",
                      "attributions": records, "unresolved_before": before,
                      "unresolved_after": len(graph.unresolved_edges()),
                      "resolved_by_model_audit": not edge.unresolved})
        attempted.add(edge.target)
        checkpoint("save", "graph_episode", {"graph": graph, "trace": trace})
        print(f"  [graph] {len(snippets)} passages; gap remains={edge.unresolved}", flush=True)
    stop = "query_budget_exhausted" if len(attempted) >= max_queries else "no_unsearched_gap"
    return trace, stop


def feedback_for(graph, trace, strategy):
    records = [row for row in trace if row.get("strategy") == strategy]
    gaps = [graph.nodes[e.target].description for e in graph.unresolved_edges()
            if graph.nodes[e.target].metadata.get("strategy") == strategy]
    evidence_ids = {a["evidence_id"] for row in records for a in row["attributions"]}
    return {"strategy": strategy, "audit": records, "remaining_claims": gaps,
            "passages": [graph.nodes[key].model_dump() for key in sorted(evidence_ids)],
            "instruction": "These are model attributions, not verified biological truth. Retain negative results, unknown scope and unresolved claims. Do not turn proposed experiments or engineering additions into supporting evidence."}


def _compact_hypothesis(hyp) -> dict:
    """Minimal, decision-relevant view of a mechanism for the experiment planner.

    Full hypothesis and review dumps (16k+ prompt tokens) pushed a reasoning
    model past its output cap before it could emit the experiment JSON. Keep
    only what discriminates predictions between candidates.
    """
    return {
        "title": hyp.title,
        "summary": hyp.summary,
        "target_cells": hyp.target_cells,
        "molecular_target": hyp.molecular_target,
        "immunological_checkpoint": hyp.immunological_checkpoint,
        "therapeutic_modality": hyp.therapeutic_modality,
        "expected_joint_phenotype": hyp.expected_joint_phenotype,
        "causal_chain": [
            {"step": s.step_number, "interaction": s.interaction_type,
             "source_entity": s.source_entity, "target_entity": s.target_entity,
             "description": s.biological_description}
            for s in hyp.causal_chain
        ],
    }


def _compact_review(critique) -> dict:
    return {
        "verdict": critique.verdict,
        "overall_skeptical_assessment": critique.overall_skeptical_assessment,
        "unsupported_core_claim": critique.unsupported_core_claim,
        "audited_links": [
            {"step": a.claim_or_step, "grounding": a.grounding_status}
            for a in critique.audited_links
        ],
        "identified_contradictions": critique.identified_contradictions,
        "biological_risks_and_limitations": critique.biological_risks_and_limitations,
    }


async def design_experiment(agent, mechanisms, critiques, graph, *, on_attempt=None):
    critiques = dict(critiques)
    candidates = {key: {"hypothesis": _compact_hypothesis(hyp),
                        "review": _compact_review(critiques[key])}
                  for key, hyp in mechanisms}
    if len(candidates) < 2:
        return None
    result = await structured_call(agent,
        "Propose ONE minimal discriminating experiment, NOT a therapeutic development programme. "
        "Compare at least two supplied candidate IDs with DIFFERENT predicted observations under "
        "the SAME intervention, controls and readout. Include a non-immune/direct-effect alternative "
        "as a control where appropriate. Give outcome-dependent next decisions and confounders. "
        "If all hypotheses predict the same outcome the experiment is not discriminating. "
        "Do not claim any experiment was performed. No arbitrary numerical information gain.\n"
        + json.dumps({"candidates": candidates, "unresolved": [graph.nodes[e.target].description
                          for e in graph.unresolved_edges()]}, ensure_ascii=False), Experiment,
        on_attempt=on_attempt)
    keys = set(result.candidate_ids)
    if len(keys) < 2 or not keys <= candidates.keys() or not keys <= result.predictions.keys():
        raise ValueError("experiment must compare actual candidate IDs with predictions")
    if len({result.predictions[k].strip().lower() for k in keys}) < 2:
        raise ValueError("experiment has identical predictions")
    if not result.next_decisions:
        raise ValueError("experiment missing outcome-dependent decisions")
    return result.model_dump()
