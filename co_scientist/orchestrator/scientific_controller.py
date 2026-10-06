"""Deterministic controller for the next scientific action.

This is a policy layer, not another reasoning agent.  It makes the feedback
loop inspectable and testable: missing causal evidence takes precedence over
polishing a hypothesis, while a well-grounded graph moves to discriminating
experiment design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..models.scientific_graph import ScientificGraph

Action = Literal[
    "retrieve_missing_evidence",
    "design_discriminating_experiment",
    "revise_hypothesis",
    "stop",
]


@dataclass(frozen=True)
class ControlDecision:
    action: Action
    reason: str
    target_edge: tuple[str, str] | None = None
    expected_information_gain: float = 0.0


def choose_next_action(
    graph: ScientificGraph,
    *,
    candidate_count: int,
    budget_remaining: float,
    minimum_coverage: float = 0.75,
) -> ControlDecision:
    """Choose the next action using observable graph state only."""
    if budget_remaining <= 0:
        return ControlDecision("stop", "budget exhausted")

    unresolved = graph.unresolved_edges()
    if unresolved:
        edge = max(unresolved, key=lambda item: 1.0 - item.confidence)
        return ControlDecision(
            "retrieve_missing_evidence",
            "an unresolved causal bridge has higher value than prose evolution",
            (edge.source, edge.target),
            round(1.0 - edge.confidence, 4),
        )

    if graph.evidence_coverage() < minimum_coverage:
        return ControlDecision(
            "revise_hypothesis",
            "graph coverage is below the minimum grounding threshold",
            expected_information_gain=round(1.0 - graph.evidence_coverage(), 4),
        )

    if candidate_count >= 2:
        return ControlDecision(
            "design_discriminating_experiment",
            "multiple grounded candidates require an experiment that separates them",
            expected_information_gain=0.8,
        )

    return ControlDecision("stop", "one sufficiently grounded candidate remains")
