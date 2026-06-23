"""Event log repository — structured event stream mirror of stdout JSONL."""

from __future__ import annotations

import json
import time
from typing import Any

import aiosqlite


async def emit(
    conn: aiosqlite.Connection,
    *,
    session_id: str | None,
    task_id: str | None,
    agent: str | None,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        """INSERT INTO events(ts, session_id, task_id, agent, event, payload)
           VALUES (?,?,?,?,?,?)""",
        (
            int(time.time() * 1000),
            session_id,
            task_id,
            agent,
            event,
            json.dumps(payload) if payload else None,
        ),
    )
    await conn.commit()


async def recent(
    conn: aiosqlite.Connection, session_id: str, *, limit: int = 100
) -> list[dict[str, Any]]:
    async with conn.execute(
        """SELECT ts, session_id, task_id, agent, event, payload
             FROM events WHERE session_id=?
            ORDER BY ts DESC LIMIT ?""",
        (session_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "ts": r["ts"],
                "session_id": r["session_id"],
                "task_id": r["task_id"],
                "agent": r["agent"],
                "event": r["event"],
                "payload": json.loads(r["payload"]) if r["payload"] else None,
            }
        )
    return out
