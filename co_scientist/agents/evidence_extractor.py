"""Evidence Extractor Agent.

Extracts grounded, verifiable biological claims and exact citations
(paper, page numbers, text spans) from the local literature corpus.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from ..rag.paperqa_retriever import EvidenceSnippet, PaperQARetriever, RetrievalResult


@dataclass
class GroundedClaim:
    """A factual scientific claim strictly grounded in literature."""

    claim: str
    source_paper: str
    pages: list[int]
    verbatim_quote: str
    biological_entities: list[str] = field(default_factory=list)
    confidence: str = "high"  # high, moderate


@dataclass
class ExtractedEvidenceBundle:
    """Collection of grounded claims for a research query."""

    query: str
    summary: str
    claims: list[GroundedClaim] = field(default_factory=list)
    caveats_and_contradictions: list[GroundedClaim] = field(default_factory=list)
    retrieval_contexts: list[dict[str, Any]] = field(default_factory=list)


class EvidenceExtractorAgent:
    """Agent responsible for identifying, validating, and formatting evidence from PDFs."""

    def __init__(
        self,
        retriever: PaperQARetriever,
        model: str = "deepseek-v4-flash-0731",
    ) -> None:
        self.retriever = retriever
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def discover_sub_queries(self, query: str) -> list[str]:
        """Decompose a high-level research goal into targeted domain-specific sub-queries."""
        prompt = f"""You are a senior scientific research strategist.
Given this research objective:
"{query}"

Decompose this objective into 3-4 precise, distinct literature search questions to query a scientific paper corpus.
Your questions should cover:
1. Molecular/cellular disease drivers and aberrant pathway markers.
2. Receptor interactions, immune checkpoints, or signaling axes.
3. Observed therapeutic interventions, biological barriers, negative results, or tissue-specific limitations reported in literature.

Return strictly valid JSON:
{{
  "sub_queries": [
    "Question 1",
    "Question 2",
    "Question 3"
  ]
}}
"""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a scientific research strategist. Output strictly valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            sqs = parsed.get("sub_queries", [])
            if sqs and isinstance(sqs, list):
                return [str(q) for q in sqs if q]
        except Exception:
            pass

        # Fallback queries derived from objective keywords
        return [
            f"What molecular targets and disease mechanisms are identified for: {query}?",
            f"What cellular pathways and immune interactions are involved in: {query}?",
            f"What therapeutic limitations, negative findings, or delivery barriers are reported for: {query}?",
        ]

    async def extract_evidence(
        self, query: str, sub_queries: list[str] | None = None
    ) -> ExtractedEvidenceBundle:
        queries = [query]
        if sub_queries:
            queries.extend(sub_queries)
        else:
            discovered = await self.discover_sub_queries(query)
            queries.extend(discovered)

        all_snippets: list[EvidenceSnippet] = []
        answers: list[str] = []

        for q in queries:
            res = await self.retriever.query(q)
            answers.append(res.answer)
            all_snippets.extend(res.evidence_snippets)

        # Structure and filter the extracted snippets with LLM
        prompt = f"""You are a meticulous scientific Evidence Extractor specializing in biomedicine and molecular mechanisms.
Analyze the following retrieved evidence snippets from peer-reviewed scientific literature.

Research Objective / Query:
{query}

Retrieved Answers & Passages from Literature:
{chr(10).join(f"Query: {q}{chr(10)}Synthesized Answer:{chr(10)}{ans}" for q, ans in zip(queries, answers))}

Available Evidence Snippets:
{chr(10).join(f"- Source: {s.citation} (Paper: {s.paper_name}){chr(10)}  Excerpt: {s.text[:500]}" for s in all_snippets[:14])}

TASK:
1. Extract 4 to 8 DISTINCT affirmative factual claims ('claims') that are directly supported by the text.
2. Extract 2 to 5 CRITICAL CAVEATS, LIMITATIONS, NEGATIVE RESULTS, OR BIOLOGICAL BARRIERS ('caveats_and_contradictions') directly reported in the text (e.g. lack of efficacy in specific tissues, delivery/penetration barriers, lack of correlation vs causation, or adverse effects).

Every claim/caveat MUST:
- Be accompanied by the exact source paper name and page numbers.
- Include a verbatim quote or exact text excerpt from the snippet proving the claim.
- List the biological entities involved (e.g. genes, receptors, cell types, tissues).
- NOT extrapolate beyond what the text explicitly states.

Return your response strictly as valid JSON matching this schema:
{{
  "summary": "High-level objective summary of the literature evidence regarding this query",
  "claims": [
    {{
      "claim": "Clear scientific fact directly stated in the paper",
      "source_paper": "Name or author/year of paper",
      "pages": [1, 2],
      "verbatim_quote": "Exact sentence or excerpt from the paper snippet",
      "biological_entities": ["entity1", "entity2"],
      "confidence": "high"
    }}
  ],
  "caveats_and_contradictions": [
    {{
      "claim": "Concrete limitation, negative finding, delivery barrier, or lack of efficacy reported in the text",
      "source_paper": "Name or author/year of paper",
      "pages": [3, 4],
      "verbatim_quote": "Exact sentence or excerpt describing the limitation or negative finding",
      "biological_entities": ["entity1", "tissue/cell"],
      "confidence": "high"
    }}
  ]
}}
"""

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise evidence extraction agent. Output strictly valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            parsed = json.loads(m.group(1)) if m else {"summary": content, "claims": [], "caveats_and_contradictions": []}

        claims = [
            GroundedClaim(
                claim=c.get("claim", ""),
                source_paper=c.get("source_paper", ""),
                pages=c.get("pages", []),
                verbatim_quote=c.get("verbatim_quote", ""),
                biological_entities=c.get("biological_entities", []),
                confidence=c.get("confidence", "high"),
            )
            for c in parsed.get("claims", [])
        ]

        caveats = [
            GroundedClaim(
                claim=c.get("claim", ""),
                source_paper=c.get("source_paper", ""),
                pages=c.get("pages", []),
                verbatim_quote=c.get("verbatim_quote", ""),
                biological_entities=c.get("biological_entities", []),
                confidence=c.get("confidence", "high"),
            )
            for c in parsed.get("caveats_and_contradictions", [])
        ]

        return ExtractedEvidenceBundle(
            query=query,
            summary=parsed.get("summary", ""),
            claims=claims,
            caveats_and_contradictions=caveats,
            retrieval_contexts=[
                {"citation": s.citation, "paper": s.paper_name, "text": s.text[:300]}
                for s in all_snippets[:8]
            ],
        )
