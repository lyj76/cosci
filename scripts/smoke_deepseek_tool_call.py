from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from co_scientist.config import load_config
from co_scientist.llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from co_scientist.llm.budgets import TokenBudget
from co_scientist.llm.provider import get_provider
from co_scientist.llm.routing import ModelRoute
from co_scientist.models import ResearchPlan, Session
from co_scientist.agents.schemas import RECORD_HYPOTHESIS_TOOL
from co_scientist.storage.db import connect, init_db
from co_scientist.storage.repos import sessions as sess_repo


def _tool_names(response) -> list[str]:
    return [
        getattr(block, "name", "")
        for block in response.raw.content
        if getattr(block, "type", None) == "tool_use"
    ]


async def _run_case(llm, *, label: str, tool_choice: dict | None) -> None:
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="tool_smoke", model="deepseek-v4-flash"),
        system_blocks=[CachedBlock("You test function calling. Use the provided tool.")],
        user_blocks=[
            CachedBlock(
                "Create a tiny hypothesis about Nanvuranlat in AML. "
                "Call record_hypothesis exactly once."
            )
        ],
        tools=[RECORD_HYPOTHESIS_TOOL],
        tool_choice=tool_choice,
        max_output_tokens=800,
    )
    ctx = CallContext(
        session_id="smoke_deepseek_tool_call",
        task_id=None,
        agent="generation",
        action=f"tool_smoke_{label}",
        mode=label,
    )
    resp = await llm.call(spec, ctx)
    names = _tool_names(resp)
    print(f"[tool-smoke] {label} stop={getattr(resp.raw, 'stop_reason', None)} tools={names}")
    for block in resp.raw.content:
        if getattr(block, "type", None) == "tool_use":
            inp = getattr(block, "input", {}) or {}
            print(f"[tool-smoke] {label} input_keys={sorted(inp.keys())}")
            print(f"[tool-smoke] {label} statement={(inp.get('statement') or '')[:240]}")


async def main() -> None:
    cfg = load_config()
    await init_db(cfg)
    conn = await connect(cfg)
    try:
        session_id = "smoke_deepseek_tool_call"
        if await sess_repo.fetch(conn, session_id) is None:
            now = datetime.now(UTC)
            await sess_repo.insert(
                conn,
                Session(
                    id=session_id,
                    created_at=now,
                    updated_at=now,
                    status="running",
                    research_goal="Smoke test DeepSeek function calling.",
                    research_plan=ResearchPlan(
                        objective="Smoke test DeepSeek function calling."
                    ),
                    config_snapshot={"smoke": "deepseek_tool_call"},
                    budget_tokens=50_000,
                    budget_usd=0.10,
                ),
            )
        budget = TokenBudget(cfg=cfg, budget_tokens=50_000, budget_usd=0.10)
        llm = get_provider(cfg, db=conn, budget=budget)
        await _run_case(llm, label="auto", tool_choice={"type": "auto"})
        await _run_case(
            llm,
            label="forced",
            tool_choice={"type": "tool", "name": "record_hypothesis"},
        )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
