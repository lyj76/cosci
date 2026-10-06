"""Structured state for evidence-grounded scientific reasoning.

The graph is intentionally storage-agnostic.  Agents may emit these records
as JSON and the Supervisor can use the same representation for control
decisions without asking an LLM to interpret prose a second time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

NodeKind = Literal["evidence", "claim", "mechanism", "experiment", "outcome"]
EdgeRelation = Literal["supports", "contradicts", "bridges", "tests", "predicts", "entity_only", "context_only", "unknown"]
EvidenceStatus = Literal["direct", "inferred", "unknown", "contradicted"]


class ScientificNode(BaseModel):
    id: str
    kind: NodeKind
    label: str
    description: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: EvidenceStatus = "unknown"
    metadata: dict[str, str] = Field(default_factory=dict)


class ScientificEdge(BaseModel):
    source: str
    target: str
    relation: EdgeRelation
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    unresolved: bool = False


class ScientificGraph(BaseModel):
    nodes: dict[str, ScientificNode] = Field(default_factory=dict)
    edges: list[ScientificEdge] = Field(default_factory=list)

    def unresolved_edges(self) -> list[ScientificEdge]:
        """Return causal bridges lacking direct evidence or a test."""
        return [
            edge for edge in self.edges
            if edge.unresolved or (
                edge.relation == "bridges" and not edge.evidence_ids
            )
        ]

    def evidence_coverage(self) -> float:
        """Fraction of non-evidence nodes connected to direct evidence."""
        nodes = [node for node in self.nodes.values() if node.kind != "evidence"]
        if not nodes:
            return 0.0
        grounded = {
            edge.target for edge in self.edges
            if edge.evidence_ids and edge.relation in {"supports", "tests"}
        }
        return round(sum(node.id in grounded for node in nodes) / len(nodes), 4)
