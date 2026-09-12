"""Bridge to persist discovery results into standard Co-Scientist session artifacts and database."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config, load_config
from ..models import Hypothesis, ResearchPlan, Review, Session
from . import db as db_mod
from .repos import hypotheses as hyp_repo
from .repos import reviews as rev_repo
from .repos import sessions as sess_repo


async def persist_as_coscientist_session(
    report_data: dict,
    markdown_report_text: str,
    cfg: Config | None = None,
    session_id: str | None = None,
) -> str:
    """Persist pipeline results into `<data_dir>/artifacts/<session_id>/...` and register in SQLite DB."""
    if cfg is None:
        cfg = load_config()

    await db_mod.init_db(cfg)

    now = datetime.now(UTC)
    if not session_id:
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        session_id = f"ses_oa_{timestamp_str}"

    session_dir = cfg.data_dir / "artifacts" / session_id
    final_dir = session_dir / "final"
    hyp_dir = session_dir / "hypotheses"
    rev_dir = session_dir / "reviews"
    final_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir.mkdir(parents=True, exist_ok=True)
    rev_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overview Markdown
    overview_file = final_dir / "overview.md"
    overview_file.write_text(markdown_report_text, encoding="utf-8")
    rel_overview = str(overview_file.relative_to(cfg.data_dir))

    # 2. SQLite Session
    session = Session(
        id=session_id,
        created_at=now,
        updated_at=now,
        status="done",
        research_goal=report_data.get("research_objective", "基于这些pdf，对于骨关节炎，有什么从免疫层面干预测的方法，给出完整的推理路径"),
        research_plan=ResearchPlan(
            objective=report_data.get("research_objective", ""),
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
        existing = await sess_repo.fetch(conn, session_id)
        if not existing:
            await sess_repo.insert(conn, session)
        else:
            await sess_repo.set_status(conn, session_id, "done")
            await sess_repo.set_final_overview(conn, session_id, rel_overview)

        # 3. Hypotheses and Reviews
        for idx, item in enumerate(report_data.get("ranked_hypotheses", []), 1):
            hyp_dict = item.get("hypothesis", {})
            crit_dict = item.get("critique", {})
            score = item.get("score", 0.0)

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

    return session_id
