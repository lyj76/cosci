"""Executable actions emitted by the scientific controller.

The first implementation is intentionally deterministic. It creates auditable
work items instead of asking an LLM to silently decide what to do next.
External retrieval and experiment execution can consume these work items.
"""

from __future__ import annotations

import json
import hashlib

from ..models import Task, TaskResult
from ..models.scientific_graph import ScientificEdge, ScientificGraph, ScientificNode
from ..storage.repos import scientific as scientific_repo
from ..tools.base import ToolCtx
from ..rag.paperqa_retriever import PaperQARetriever
from .base import BaseAgent


def classify_evidence_relation(text: str, source: str, target: str) -> str:
    """Conservatively classify one retrieved passage against a causal edge.

    This is a high-precision gate, not a semantic truth oracle.  It refuses to
    call co-occurrence support and requires an explicit relational signal.
    """
    lowered = text.lower()
    source_terms = [term.lower() for term in source.replace("/", " ").split() if len(term) > 2]
    target_terms = [term.lower() for term in target.replace("/", " ").split() if len(term) > 2]
    if not source_terms or not target_terms or not all(t in lowered for t in (*source_terms, *target_terms)):
        return "unknown"
    negative = ("no effect", "did not", "does not", "failed to", "not associated", "independent of")
    relational = ("bind", "engage", "inhibit", "suppress", "activate", "mediate", "regulate",
                  "checkpoint", "receptor", "via", "through", "dependent")
    if any(token in lowered for token in negative):
        return "contradicts"
    if any(token in lowered for token in relational):
        return "supports"
    return "context_only"


class ScientificActionAgent(BaseAgent):
    name = "scientific"

    def __init__(self, deps):
        super().__init__(deps)
        self._local_retriever = None

    async def execute(self, task: Task) -> TaskResult:
        if task.action == "RetrieveEvidenceGap":
            source = task.payload.get("source_node", "")
            target = task.payload.get("target_node", "")
            query = task.payload.get("query") or f"{source} {target} mechanism evidence"
            # Offline mode uses the existing PaperQA2 local corpus path. It is
            # deliberately not represented as a generic network tool.
            search_tools = (
                () if self.deps.cfg.tools.offline_mode
                else ("europe_pmc_search", "pubmed_search")
            )
            if self.deps.cfg.tools.offline_mode:
                if self._local_retriever is None:
                    self._local_retriever = PaperQARetriever()
                    await self._local_retriever.index_directory(self.deps.cfg.tools.local_corpus_dir)
                retrieval = await self._local_retriever.query(query)
                snippets = [
                    {"paper": item.paper_name, "page": item.pages,
                     "text": item.text, "citation": item.citation,
                     "score": item.relevance_score}
                    for item in retrieval.evidence_snippets
                ]
                tool_name = "paperqa2_local"
            else:
                tool_name = next((name for name in search_tools if name in self.deps.tools), None)
                snippets = None
            if tool_name is None:
                return TaskResult(kind="evidence_gap_queued", extra={
                    "query": query, "source_node": source, "target_node": target,
                    "status": "ready_for_retrieval",
                })
            if snippets is None:
                result = await self.deps.tools.call(
                    tool_name, {"query": query, "max_results": 5},
                    ToolCtx(cfg=self.deps.cfg, db=self.deps.db,
                            session_id=task.session_id, task_id=task.id),
                )
                if result.is_error:
                    raise RuntimeError(result.error_message or f"{tool_name} failed")
                if isinstance(result.content, dict) and isinstance(result.content.get("results"), list):
                    snippets = result.content["results"]
                else:
                    snippets = result.content if isinstance(result.content, list) else [result.content]
            graph = await scientific_repo.load_graph(self.deps.db, task.session_id)
            evidence_ids: list[str] = []
            support_ids: list[str] = []
            contradiction_ids: list[str] = []
            for index, snippet in enumerate(snippets[:5]):
                text = snippet if isinstance(snippet, str) else json.dumps(snippet, ensure_ascii=False)
                relation = classify_evidence_relation(text, source, target)
                metadata = {
                    "query": query, "tool": tool_name,
                    "paper": str(snippet.get("paper", "")) if isinstance(snippet, dict) else "",
                    "page": str(snippet.get("page", "")) if isinstance(snippet, dict) else "",
                }
                evidence_id = "evidence-" + hashlib.sha256(
                    f"{task.session_id}:{query}:{index}:{text}".encode()
                ).hexdigest()[:16]
                evidence_ids.append(evidence_id)
                graph.nodes[evidence_id] = ScientificNode(
                    id=evidence_id, kind="evidence", label=f"Literature result {index + 1}",
                    description=text[:2000], confidence=0.6,
                    status="direct", metadata=metadata,
                )
                graph.nodes[evidence_id].metadata["relation"] = relation
                if relation == "supports":
                    support_ids.append(evidence_id)
                elif relation == "contradicts":
                    contradiction_ids.append(evidence_id)
            graph.edges = [
                edge.model_copy(update={
                    "evidence_ids": evidence_ids,
                    "confidence": 0.75 if support_ids else (0.0 if contradiction_ids else edge.confidence),
                    "unresolved": not bool(support_ids),
                })
                if edge.source == source and edge.target == target and edge.relation == "bridges"
                else edge
                for edge in graph.edges
            ]
            await scientific_repo.upsert_graph(self.deps.db, task.session_id, graph)
            return TaskResult(kind="evidence_gap_resolved", extra={
                "query": query, "source_node": source, "target_node": target,
                "status": "evidence_written", "evidence_count": len(evidence_ids),
                "support_count": len(support_ids), "contradiction_count": len(contradiction_ids),
            })
        if task.action == "DesignDiscriminatingExperiment":
            payload = task.payload
            cur = await self.deps.db.execute(
                """INSERT INTO experiment_proposals
                   (session_id, task_id, title, candidates, distinction,
                    intervention, positive_prediction, negative_prediction,
                    falsification_rule, information_gain, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (task.session_id, task.id, payload.get("title", "Discriminating experiment"),
                 json.dumps(payload.get("candidate_ids", [])), payload.get("distinction", ""),
                 payload.get("intervention", ""), payload.get("positive_prediction", ""),
                 payload.get("negative_prediction", ""), payload.get("falsification_rule", ""),
                 float(payload.get("expected_information_gain", 0.0)), "proposed"),
            )
            await self.deps.db.commit()
            return TaskResult(kind="experiment_proposed", extra={"proposal_id": cur.lastrowid})
        return TaskResult(kind="noop", extra={"reason": f"unsupported action {task.action}"})
