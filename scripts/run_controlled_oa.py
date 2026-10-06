"""Simple foreground runner for a bounded local-PDF OA experiment.

Usage:
    python scripts/run_controlled_oa.py
    python scripts/run_controlled_oa.py --output-dir data/runs/my_run --resume

The runner deliberately prints progress in the foreground. Redirect stdout
yourself when desired; no nohup/pid management is required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from co_scientist.orchestrator.oa_pipeline import run_oa_pipeline


PAPERS = [
    "The ganglioside GD3 and its synthase (ST8SIA1) as novel sene.pdf",
    "A ganglioside-based immune checkpoint enables senescent cell.pdf",
    "Senescent repair memory in chronic d SO b Ageing Res Rev b 2.pdf",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Run bounded local-PDF OA discovery.")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--output-dir", default="data/runs/controlled_local/oa_foreground")
    parser.add_argument("--objective", default=(
        "仅基于固定本地PDF，针对骨关节炎提出免疫干预假说；严格区分直接证据、"
        "背景证据、矛盾证据和未经验证的机制桥接，并提出区分性实验。"
    ))
    parser.add_argument("--with-elo", action="store_true", help="Enable optional network Elo judging.")
    parser.add_argument("--resume", action="store_true", help="Resume completed stages from checkpoints.")
    parser.add_argument("--check-resume", action="store_true",
                        help="Validate saved inputs/checkpoints without model calls or running the experiment.")
    parser.add_argument("--graph-queries", type=int, default=1,
                        help="Local evidence-gap query budget; 0 disables graph interventions for ablation.")
    parser.add_argument("--phase-timeout", type=int, default=600)
    parser.add_argument("--overall-timeout", type=int, default=1800)
    args = parser.parse_args()
    print("Starting bounded local-PDF experiment", flush=True)
    print(f"Corpus: {args.paper_dir} ({len(PAPERS)} fixed PDFs)", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    print("Elo: " + ("enabled" if args.with_elo else "skipped"), flush=True)
    print(f"Graph query budget: {args.graph_queries}; remote model/embedding APIs may be used; literature is local only.", flush=True)
    result = asyncio.run(run_oa_pipeline(
        research_objective=args.objective,
        paper_dir=Path(args.paper_dir),
        paper_files=PAPERS,
        output_dir=Path(args.output_dir),
        max_evolution_rounds=1,
        parallelism=2,
        phase_timeout_seconds=args.phase_timeout,
        overall_timeout_seconds=args.overall_timeout,
        run_tournament=args.with_elo,
        resume=args.resume or args.check_resume,
        preflight_only=args.check_resume,
        graph_evidence_queries=args.graph_queries,
    ))
    if args.check_resume:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
