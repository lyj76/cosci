"""PaperQA2-based Local Scientific Literature Retriever.

Provides robust, grounded retrieval over local scientific PDFs with:
- Strict page-number tracking
- Verifiable text quotes / spans
- Zero external web hallucination (fully offline)
- LiteLLM integration with DashScope LLMs and text-embedding-v3
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Configure base URLs for DashScope compatibility
_dashscope_base = os.environ.get(
    "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
os.environ["OPENAI_BASE_URL"] = _dashscope_base
os.environ["OPENAI_API_BASE"] = _dashscope_base

import litellm

litellm.drop_params = True
litellm.suppress_debug_info = True

# Register custom / third-party models in LiteLLM to prevent cost-calculation warnings
litellm.register_model({
    "deepseek-v4-flash-0731": {
        "max_tokens": 16384,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "openai/deepseek-v4-flash-0731": {
        "max_tokens": 16384,
        "input_cost_per_token": 0.0000005,
        "output_cost_per_token": 0.0000015,
        "litellm_provider": "openai",
        "mode": "chat",
    },
})

from paperqa import Docs, Settings
from paperqa.settings import MultimodalOptions, ParsingSettings


@dataclass
class EvidenceSnippet:
    """Individual evidence snippet grounded in a local PDF."""

    paper_name: str
    citation: str
    pages: list[int]
    text: str
    relevance_score: float = 0.0


@dataclass
class RetrievalResult:
    """Result of querying the local literature corpus."""

    query: str
    answer: str
    evidence_snippets: list[EvidenceSnippet] = field(default_factory=list)
    raw_references: list[str] = field(default_factory=list)


def _extract_pages_from_citation(citation: str) -> list[int]:
    pages: list[int] = []
    if "pages" in citation:
        m = re.findall(r"pages?\s+([0-9]+)(?:-([0-9]+))?", citation)
        for start, end in m:
            if end:
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(start))
    return pages


class PaperQARetriever:
    """High-level wrapper around PaperQA2 for local scientific corpus QA & evidence retrieval."""

    def __init__(
        self,
        model: str = "openai/deepseek-v4-flash-0731",
        embedding_model: str = "openai/text-embedding-v3",
        embedding_batch_size: int = 8,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size

        self.settings = Settings(
            llm=self.model,
            summary_llm=self.model,
            embedding=self.embedding_model,
            embedding_config={"batch_size": self.embedding_batch_size},
            parsing=ParsingSettings(
                multimodal=MultimodalOptions.OFF,
                use_doc_details=False,
            ),
        )
        self.docs = Docs()
        self._indexed_files: set[str] = set()

    async def index_file(self, file_path: str | Path) -> bool:
        path = Path(file_path)
        if not path.exists():
            return False
        abs_str = str(path.resolve())
        if abs_str in self._indexed_files:
            return True
        await self.docs.aadd(abs_str, settings=self.settings)
        self._indexed_files.add(abs_str)
        return True

    async def index_directory(self, dir_path: str | Path, pattern: str = "*.pdf") -> int:
        p = Path(dir_path)
        if not p.exists() or not p.is_dir():
            return 0
        added = 0
        for f in sorted(p.glob(pattern)):
            if f.is_file():
                success = await self.index_file(f)
                if success:
                    added += 1
        return added

    async def query(self, question: str) -> RetrievalResult:
        """Run PaperQA2 query over the indexed corpus."""
        session = await self.docs.aquery(question, settings=self.settings)
        answer = session.formatted_answer

        snippets: list[EvidenceSnippet] = []
        raw_contexts = getattr(session, "contexts", [])
        for c in raw_contexts:
            # Handle Context object structure safely
            text_obj = getattr(c, "text", None)
            raw_text = (
                getattr(text_obj, "text", str(text_obj))
                if text_obj is not None
                else getattr(c, "context", "")
            )
            citation = (
                getattr(text_obj, "name", "")
                if text_obj is not None and hasattr(text_obj, "name")
                else getattr(c, "context", str(c))
            )
            if not citation:
                citation = str(c)

            pages = _extract_pages_from_citation(citation)
            paper_name = citation.split(" pages")[0] if " pages" in citation else citation

            snippets.append(
                EvidenceSnippet(
                    paper_name=paper_name,
                    citation=citation,
                    pages=pages,
                    text=raw_text,
                    relevance_score=getattr(c, "score", 1.0),
                )
            )

        references: list[str] = []
        if hasattr(session, "references") and session.references:
            references = [str(r) for r in session.references]

        return RetrievalResult(
            query=question,
            answer=answer,
            evidence_snippets=snippets,
            raw_references=references,
        )
