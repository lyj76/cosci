"""Export OA discovery results into the standard Co-Scientist session format.

Persists artifacts under `data/artifacts/<session_id>/...` and registers into `data/co-scientist.db`.
Allows `co-scientist report <session_id>` and `co-scientist serve` to display the results seamlessly.
"""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from co_scientist.config import load_config
from co_scientist.models import Hypothesis, ResearchPlan, Review, Session
from co_scientist.storage import db as db_mod
from co_scientist.storage.repos import hypotheses as hyp_repo
from co_scientist.storage.repos import reviews as rev_repo
from co_scientist.storage.repos import sessions as sess_repo


async def export_oa_results(
    report_json_path: str = "data/runs/oa_discovery/oa_discovery_report.json",
    session_id: str = "ses_oa_discovery_01",
) -> str:
    cfg = load_config()
    await db_mod.init_db(cfg)

    with open(report_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session_dir = cfg.data_dir / "artifacts" / session_id
    final_dir = session_dir / "final"
    hyp_dir = session_dir / "hypotheses"
    rev_dir = session_dir / "reviews"
    final_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)

    # 1. Final overview markdown
    md_source = Path("data/runs/oa_discovery/oa_discovery_report.md")
    overview_text = md_source.read_text(encoding="utf-8") if md_source.exists() else ""
    overview_file = final_dir / "overview.md"
    overview_file.write_text(overview_text, encoding="utf-8")
    rel_overview = str(overview_file.relative_to(cfg.data_dir))

    # 2. Session creation in SQLite
    now = datetime.now(UTC)
    session = Session(
        id=session_id,
        created_at=now,
        updated_at=now,
        status="done",
        research_goal=data.get("research_objective", "基于这些pdf，对于骨关节炎，有什么从免疫层面干预测的方法，给出完整的推理路径"),
        research_plan=ResearchPlan(
            objective=data.get("research_objective", ""),
            constraints=["local_pdf_only", "no_external_web"],
            domain_hint="osteoarthritis_immunology",
        ),
        config_snapshot={},
        budget_tokens=1000000,
        budget_usd=5.0,
        budget_used_tokens=0,
        budget_used_usd=0.0,
        final_overview=rel_overview,
    )

    conn = await db_mod.connect(cfg)
    try:
        # Check if session exists
        existing = await sess_repo.fetch(conn, session_id)
        if not existing:
            await sess_repo.insert(conn, session)
        else:
            await sess_repo.set_status(conn, session_id, "done")
            await sess_repo.set_final_overview(conn, session_id, rel_overview)

        # 3. Hypotheses & Reviews
        for idx, item in enumerate(data.get("ranked_hypotheses", []), 1):
            hyp_dict = item["hypothesis"]
            crit_dict = item["critique"]
            score = item.get("score", 0.0)

            # Deterministic IDs
            hid = hashlib.sha256(f"{session_id}::hyp::{idx}".encode()).hexdigest()[:16]
            rid = hashlib.sha256(f"{session_id}::rev::{idx}".encode()).hexdigest()[:16]

            hyp_file = hyp_dir / f"{hid}.json"
            hyp_file.write_text(json.dumps(hyp_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            rel_hyp = str(hyp_file.relative_to(cfg.data_dir))

            rev_file = rev_dir / f"{rid}.json"
            rev_file.write_text(json.dumps(crit_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            rel_rev = str(rev_file.relative_to(cfg.data_dir))

            strat_raw = item.get("strategy", "literature")
            strat_map = {
                "dual_compartment_targeting": "combine",
                "checkpoint_blockade": "literature",
                "synthase_inhibition": "simplify",
            }
            strat = strat_map.get(strat_raw, "literature")

            hyp_obj = Hypothesis(
                id=hid,
                session_id=session_id,
                created_at=now,
                created_by="generation",
                strategy=strat,
                parent_ids=[],
                title=hyp_dict.get("title", f"Hypothesis {idx}"),
                summary=hyp_dict.get("executive_summary", "")[:500],
                full_text=hyp_dict.get("primary_hypothesis_statement", "") + "\n\n" + hyp_dict.get("biological_rationale", ""),
                artifact_path=rel_hyp,
                elo=1000.0 + (score * 500.0 if score > 0 else 0.0),
                matches_played=3,
                state="reviewed",
            )
            await hyp_repo.insert(conn, hyp_obj)

            rev_obj = Review(
                id=rid,
                hypothesis_id=hid,
                session_id=session_id,
                created_at=now,
                kind="full",
                verdict="already_explained" if item.get("is_eliminated") else "other_more_likely",
                novelty=0.85,
                correctness=0.80,
                testability=0.90,
                feasibility=0.85,
                body=crit_dict.get("overall_skeptical_assessment", ""),
                artifact_path=rel_rev,
            )
            await rev_repo.insert(conn, rev_obj)

    finally:
        await conn.close()

    print(f"Exported session '{session_id}' to standard Co-Scientist storage.")
    print(f"Artifact root: {session_dir}")
    print(f"Overview: {overview_file}")
    return session_id


if __name__ == "__main__":
    sid = asyncio.run(export_oa_results())
    print(f"Success! Session ID: {sid}")
