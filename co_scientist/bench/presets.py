"""Built-in bench candidate presets.

Curated comparison setups so users can reproduce known benchmarks with one
flag instead of typing N `--candidate` lines.

Substitutions in the "paper" preset
-----------------------------------
The Google Co-Scientist paper compared their system against:

    Gemini 2.0 Flash Thinking Experimental 12-19
    Gemini 2.0 Pro Experimental
    OpenAI o1

across 15 expert-curated research goals. Of those, OpenRouter currently
serves only `openai/o1`; the experimental Gemini 2.0 Thinking and Pro
Experimental branches were retired. We substitute the closest current
analogues and document the swap so reported numbers are interpretable:

    paper-baseline                            current substitute
    ----------------------------------------  --------------------------------
    Gemini 2.0 Flash Thinking Experimental    google/gemini-2.0-flash-001
       (Thinking branch deprecated; production 2.0 Flash is the closest)
    Gemini 2.0 Pro Experimental               google/gemini-2.5-pro
       (2.0 Pro Experimental removed; 2.5 Pro is the current Pro-tier Gemini)
    OpenAI o1                                 openai/o1               (exact)
    Claude Haiku (added by user request)      anthropic/claude-haiku-4.5

Judge model is left configurable via --judge so users can pick a different
referee. The recommended default for this preset is
`openrouter:google/gemini-3-flash-preview`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .goldset import AML_REPURPOSING_PAPER_TOP3, GoldSet
from .runner import BenchCandidate


@dataclass(frozen=True)
class BenchPreset:
    name: str
    description: str
    candidates: tuple[BenchCandidate, ...]
    suggested_judge: str  # "provider:model"
    # Optional preset defaults — let the CLI invoke a preset without forcing
    # the user to retype the goal or attach a gold set.
    default_goal: str | None = None
    goldset: GoldSet | None = None


_PAPER_CANDIDATES: tuple[BenchCandidate, ...] = (
    BenchCandidate(
        label="gemini-2-flash-thinking",
        provider="openrouter",
        model="google/gemini-2.0-flash-001",
    ),
    BenchCandidate(
        label="gemini-2-pro",
        provider="openrouter",
        model="google/gemini-2.5-pro",
    ),
    BenchCandidate(
        label="openai-o1",
        provider="openrouter",
        model="openai/o1",
    ),
    BenchCandidate(
        label="claude-haiku-4.5",
        provider="openrouter",
        model="anthropic/claude-haiku-4.5",
    ),
)


# This goal follows Supplementary Note 5.1. Exact reproduction additionally
# requires the paper's curated ~2300-drug universe, DepMap Q2 2024 data, and
# expert review. Use --candidate-universe to provide the drug list.
_PAPER_AML_GOAL = (
    "Evaluate approved-drug repurposing hypotheses for acute myeloid leukemia "
    "(AML). For each assigned drug candidate: identify its approved indication, "
    "known mechanism of action and affected pathways; search the literature for "
    "prior clinical or preclinical AML evidence, related evidence, and repurposing "
    "challenges; explain a specific "
    "mechanism by which it could affect AML blasts or leukemic stem cells; assess "
    "novelty, correctness, feasibility, and likely selectivity; and propose a "
    "concrete falsifiable in vitro or in vivo experiment. Name the exact compound, "
    "not a generic drug class. Rank the resulting candidate-specific proposals."
)


def _vs_raw(candidates: tuple[BenchCandidate, ...]) -> tuple[BenchCandidate, ...]:
    """Double every candidate: once in pipeline mode, once in direct mode.

    This compares the Generation-only harness against a raw single-shot LM
    call on the same goal. It is not a complete multi-agent session comparison.
    The label gets a `[pipe]` / `[raw]` suffix so the result table makes
    the distinction obvious.
    """
    out: list[BenchCandidate] = []
    for c in candidates:
        out.append(
            BenchCandidate(
                label=f"{c.label}[pipe]",
                provider=c.provider,
                model=c.model,
                mode="pipeline",
            )
        )
        out.append(
            BenchCandidate(
                label=f"{c.label}[raw]",
                provider=c.provider,
                model=c.model,
                mode="direct",
            )
        )
    return tuple(out)


# Current frontier set — what you'd actually want to use today. Picked to
# span pricing tiers so the bench surfaces $/quality tradeoffs, not just
# raw quality.
_FRONTIER_CANDIDATES: tuple[BenchCandidate, ...] = (
    BenchCandidate(
        label="claude-opus-4.7",
        provider="openrouter",
        # OpenRouter uses dots in the version suffix.
        model="anthropic/claude-opus-4.7",
    ),
    BenchCandidate(
        label="gpt-5",
        provider="openrouter",
        model="openai/gpt-5",
    ),
    BenchCandidate(
        label="gemini-3-pro",
        provider="openrouter",
        model="google/gemini-3.1-pro-preview",
    ),
    BenchCandidate(
        label="gemini-3-flash",
        provider="openrouter",
        model="google/gemini-3-flash-preview",
    ),
)

_DEEPSEEK_UPLIFT_CANDIDATES: tuple[BenchCandidate, ...] = (
    BenchCandidate(
        label="deepseek-session",
        provider="openai_compatible",
        model="deepseek-v4-pro",
        mode="session",
    ),
    BenchCandidate(
        label="deepseek-pipeline",
        provider="openai_compatible",
        model="deepseek-v4-pro",
        mode="pipeline",
    ),
    BenchCandidate(
        label="deepseek-direct",
        provider="openai_compatible",
        model="deepseek-v4-pro",
        mode="direct",
    ),
)


PRESETS: dict[str, BenchPreset] = {
    "paper": BenchPreset(
        name="paper",
        description=(
            "Run a paper-derived preference-ranking comparison "
            "(plus Haiku) using current OpenRouter models. See module docstring "
            "for the substitutions we had to make for retired experimental models."
        ),
        candidates=_PAPER_CANDIDATES,
        suggested_judge="openrouter:google/gemini-3-flash-preview",
    ),
    "paper-aml": BenchPreset(
        name="paper-aml",
        description=(
            "Paper-derived AML recall benchmark. By itself this is a small "
            "generation benchmark, not the paper's 2300-drug screen. For a "
            "method-level reproduction, add a curated approved-drug file via "
            "--candidate-universe and use a custom @session candidate. Recall "
            "is scored against Nanvuranlat, KIRA6, and Leflunomide."
        ),
        candidates=_PAPER_CANDIDATES,
        suggested_judge="openrouter:google/gemini-3-flash-preview",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
    "paper-aml-vs-raw": BenchPreset(
        name="paper-aml-vs-raw",
        description=(
            "AML repurposing benchmark for the paper's baseline models, "
            "under the strict no-prior-evidence methodology — each model "
            "runs TWICE: once through the Generation-only "
            "pipeline (literature tools + tool loop + dedup) and once as a "
            "single raw LM call. Isolates how much of the system's "
            "performance comes from the Generation harness vs the model "
            "itself. Same top-3 gold set as `paper-aml`."
        ),
        candidates=_vs_raw(_PAPER_CANDIDATES),
        suggested_judge="openrouter:google/gemini-3-flash-preview",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
    "frontier-aml-vs-raw": BenchPreset(
        name="frontier-aml-vs-raw",
        description=(
            "Same setup as `paper-aml-vs-raw` but with current frontier "
            "models (Claude Opus 4.7, GPT-5, Gemini 3 Pro, Gemini 3 Flash). "
            "Lets you see whether the multi-agent harness still adds value "
            "with stronger base models."
        ),
        candidates=_vs_raw(_FRONTIER_CANDIDATES),
        suggested_judge="openrouter:google/gemini-3-flash-preview",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
    "deepseek-aml-uplift": BenchPreset(
        name="deepseek-aml-uplift",
        description=(
            "Same-model ablation for DashScope DeepSeek: complete multi-agent "
            "session versus Generation-only pipeline versus one direct call. "
            "Use --candidate-universe for the paper-style approved-drug screen."
        ),
        candidates=_DEEPSEEK_UPLIFT_CANDIDATES,
        suggested_judge="openai_compatible:deepseek-v4-flash",
        default_goal=_PAPER_AML_GOAL,
        goldset=AML_REPURPOSING_PAPER_TOP3,
    ),
}


def get_preset(name: str) -> BenchPreset:
    try:
        return PRESETS[name]
    except KeyError as e:
        names = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown bench preset {name!r}; available: {names}") from e
