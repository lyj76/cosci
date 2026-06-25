"""Typer CLI entrypoint.

M0 surface: init, list, status, tools list, serve, version. Run/resume/report/feedback
land in M3+. All commands resolve config the same way; secrets come from env.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from .config import has_llm_key, load_config, provider_key_env
from .logging import get_logger, setup_logging
from .storage import db as db_mod

if TYPE_CHECKING:
    from .agents.supervisor import Supervisor

VERSION = "0.1.0"

app = typer.Typer(
    help="AI Co-Scientist — multi-agent hypothesis generation, ranking, and synthesis.",
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
)
tools_app = typer.Typer(help="Tool registry inspection and debug invocation.", no_args_is_help=True)
app.add_typer(tools_app, name="tools")

console = Console()
log = get_logger("cli")


def _common_setup(config_file: Path | None = None, verbose: bool = False) -> tuple:
    setup_logging("DEBUG" if verbose else "INFO")
    cfg = load_config(config_file)
    return cfg, log


@app.callback()
def _main(
    ctx: typer.Context,
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to an extra TOML config to overlay."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose (DEBUG) logging."),
) -> None:
    ctx.obj = _common_setup(config_file, verbose)
    if ctx.invoked_subcommand is None:
        _interactive_home(ctx)
        raise typer.Exit()


@app.command()
def version() -> None:
    """Print the co-scientist version."""
    console.print(f"co-scientist {VERSION}")


@app.command()
def init(ctx: typer.Context) -> None:
    """Create the data directory, apply migrations, sanity-check env."""
    cfg, _ = ctx.obj
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "artifacts").mkdir(exist_ok=True)
    (cfg.data_dir / "vectors").mkdir(exist_ok=True)
    (cfg.data_dir / "logs").mkdir(exist_ok=True)

    asyncio.run(db_mod.init_db(cfg))

    # Report
    env_var = provider_key_env(cfg)
    tbl = Table(title="Init complete", show_header=False, box=None)
    tbl.add_row("data dir", str(cfg.data_dir))
    tbl.add_row("database", str(cfg.db_path))
    tbl.add_row("LLM provider", cfg.llm.provider)
    tbl.add_row(
        f"{env_var or '(keyless)'} set",
        "yes" if has_llm_key(cfg) else "[red]no[/red]",
    )
    tbl.add_row("science-skills", cfg.science_skills.path)
    console.print(tbl)

    if not has_llm_key(cfg):
        console.print(
            f"[yellow]{env_var} is not set; set it (or change [llm] provider in your "
            f"config) before running a session. See .env.example.[/yellow]"
        )


@app.command(name="list")
def list_sessions(ctx: typer.Context) -> None:
    """List all sessions in the local database."""
    cfg, _ = ctx.obj

    async def _run() -> list[dict]:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute(
                """SELECT id, status, research_goal, created_at, updated_at,
                          budget_usd, budget_used_usd,
                          (SELECT COUNT(*) FROM hypotheses WHERE session_id = s.id) AS n_hyps,
                          (SELECT MAX(elo) FROM hypotheses WHERE session_id = s.id) AS top_elo
                     FROM sessions s
                     ORDER BY updated_at DESC"""
            ) as cur:
                rows = await cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    rows = asyncio.run(_run())
    if not rows:
        console.print("[dim]No sessions yet. Try:  co-scientist run \"your research goal\"[/dim]")
        return

    tbl = Table(title="Sessions")
    tbl.add_column("id")
    tbl.add_column("status")
    tbl.add_column("goal", overflow="fold", max_width=60)
    tbl.add_column("hyps", justify="right")
    tbl.add_column("top Elo", justify="right")
    tbl.add_column("$ used / $ budget", justify="right")
    tbl.add_column("updated")
    for r in rows:
        tbl.add_row(
            r["id"],
            r["status"],
            (r["research_goal"] or "")[:80],
            str(r["n_hyps"]),
            f"{r['top_elo']:.0f}" if r["top_elo"] is not None else "—",
            f"${r['budget_used_usd']:.2f} / ${r['budget_usd']:.2f}",
            r["updated_at"],
        )
    console.print(tbl)


@app.command()
def status(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Show detailed status for one session."""
    cfg, _ = ctx.obj

    async def _run() -> dict:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return {}
            out = dict(row)
            for col in ("research_plan", "config_snapshot"):
                with contextlib.suppress(Exception):
                    out[col] = json.loads(out[col])
            async with conn.execute(
                """SELECT status, COUNT(*) AS n FROM tasks
                       WHERE session_id = ? GROUP BY status""",
                (session_id,),
            ) as cur:
                out["task_counts"] = {r["status"]: r["n"] for r in await cur.fetchall()}
            async with conn.execute(
                """SELECT state, COUNT(*) AS n FROM hypotheses
                       WHERE session_id = ? GROUP BY state""",
                (session_id,),
            ) as cur:
                out["hypothesis_states"] = {r["state"]: r["n"] for r in await cur.fetchall()}
            return out
        finally:
            await conn.close()

    out = asyncio.run(_run())
    if not out:
        console.print(f"[red]No session {session_id}[/red]")
        raise typer.Exit(1)
    console.print_json(data=out)


async def _run_with_progress(sup: Supervisor, goal: str, **kwargs) -> str:
    from rich.live import Live
    from rich.panel import Panel

    from .obs.metrics import session_metrics_cached
    from .orchestrator.events import GLOBAL_BUS

    session_id: str | None = kwargs.get("resume_session_id")
    events_task: asyncio.Task | None = None

    def on_start(sid: str):
        nonlocal session_id
        session_id = sid

    async def _print_events(sid: str) -> None:
        async for ev in GLOBAL_BUS.subscribe(sid):
            payload = ev.payload or {}
            if ev.name == "task_started":
                console.print(
                    f"[cyan]task started[/cyan] "
                    f"{payload.get('agent')}:{payload.get('action')} "
                    f"task={payload.get('task_id')} target={payload.get('target') or '-'}"
                )
            elif ev.name == "task_completed":
                preview = payload.get("preview") or {}
                console.print(
                    f"[green]task completed[/green] "
                    f"kind={payload.get('kind')} "
                    f"hypotheses={','.join(payload.get('follow_hypothesis_ids') or []) or '-'}"
                )
                for h in preview.get("hypotheses") or []:
                    console.print(
                        f"  [bold]hypothesis[/bold] {h.get('id')} "
                        f"{h.get('created_by')}/{h.get('strategy')} "
                        f"state={h.get('state')} elo={h.get('elo') or '-'}\n"
                        f"  title: {h.get('title') or '(no title)'}\n"
                        f"  summary: {h.get('summary') or '(no summary)'}"
                    )
                for r in preview.get("reviews") or []:
                    console.print(
                        f"  [bold]review[/bold] {r.get('kind')} "
                        f"verdict={r.get('verdict') or '?'} "
                        f"novelty={r.get('novelty')} "
                        f"correctness={r.get('correctness')} "
                        f"testability={r.get('testability')}\n"
                        f"  {r.get('body') or ''}"
                    )
                if preview.get("extra"):
                    console.print(f"  [dim]extra: {preview.get('extra')}[/dim]")
            elif ev.name == "task_failed":
                console.print(
                    f"[red]task failed[/red] task={payload.get('task_id')} "
                    f"err={payload.get('err')}"
                )
            elif ev.name == "tool_call":
                status = "[red]error[/red]" if payload.get("is_error") else "[green]ok[/green]"
                console.print(
                    f"[magenta]tool[/magenta] {payload.get('agent')} "
                    f"{payload.get('tool')} {status} "
                    f"{payload.get('duration_ms')}ms iter={payload.get('iteration')}"
                )
            elif ev.name in {"session_started", "session_done"}:
                console.print(f"[bold]{ev.name}[/bold] {payload}")

    # Start supervisor task
    task = asyncio.create_task(sup.run_session(goal, on_session_start=on_start, **kwargs))

    with Live(refresh_per_second=1, vertical_overflow="visible") as live:
        while not task.done():
            if session_id:
                if events_task is None:
                    events_task = asyncio.create_task(_print_events(session_id))
                conn = await db_mod.connect(sup.cfg)
                try:
                    m = await session_metrics_cached(conn, session_id)
                    tbl = Table(title=f"Session {session_id} Progress", box=None)
                    tbl.add_column("Metric")
                    tbl.add_column("Value", justify="right")
                    tbl.add_row("Hypotheses", str(m.n_hypotheses))
                    tbl.add_row("Reviewed", str(m.n_reviewed))
                    tbl.add_row("Tournament", str(m.n_in_tournament))
                    tbl.add_row("Matches", str(m.n_matches))
                    tbl.add_row("Cost (USD)", f"${m.cost_usd:.2f}")
                    tbl.add_row("Pending Tasks", str(m.pending_tasks))
                    if m.dead_tasks > 0:
                        tbl.add_row("Dead Tasks", f"[red]{m.dead_tasks}[/red]")
                    live.update(Panel(tbl, subtitle="[dim]Press Ctrl+C to pause (aborting workers...)[/dim]"))
                finally:
                    await conn.close()
            await asyncio.sleep(2)

    try:
        return await task
    finally:
        if events_task is not None:
            events_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await events_task


def _start_research_session(
    ctx: typer.Context,
    goal: str,
    *,
    preferences_text: str | None = None,
    n_initial: int = 3,
    wall_clock: int | None = None,
    budget_usd: float | None = None,
    budget_tokens: int | None = None,
    max_ideas: int | None = None,
    concurrency: int | None = None,
    candidate_universe: Path | None = None,
) -> None:
    cfg, _ = ctx.obj
    if not has_llm_key(cfg):
        env_var = provider_key_env(cfg)
        console.print(
            f"[red]{env_var} is not set (LLM provider = {cfg.llm.provider}). "
            "See .env.example.[/red]"
        )
        raise typer.Exit(1)

    if budget_usd is not None:
        cfg.run.budget_usd = budget_usd
    if budget_tokens is not None:
        cfg.run.budget_tokens = budget_tokens
    if max_ideas is not None:
        cfg.run.max_ideas = max_ideas
    if concurrency is not None:
        cfg.run.concurrency = concurrency
    search_space_items = None
    if candidate_universe is not None:
        from .search_space import load_candidate_universe

        search_space_items = load_candidate_universe(candidate_universe)
        cfg.run.max_ideas = len(search_space_items)
        n_initial = min(n_initial, len(search_space_items))
        console.print(
            f"[dim]Constrained candidate universe: {len(search_space_items)} items[/dim]"
        )

    # Pre-flight cost estimate
    from .llm.estimator import estimate as _estimate

    est = _estimate(cfg)
    console.print(
        f"[dim]Pre-flight estimate: ${est.total_usd:.2f} "
        f"(budget ${cfg.run.budget_usd:.2f}, "
        f"token_budget={'unlimited' if cfg.run.budget_tokens <= 0 else cfg.run.budget_tokens}, "
        f"max_ideas={cfg.run.max_ideas}, "
        f"max_matches_per_idea={cfg.run.max_matches_per_idea})[/dim]"
    )
    if est.warning:
        console.print(f"[yellow]{est.warning}[/yellow]")

    from .agents.supervisor import Supervisor

    sup = Supervisor(cfg)
    session_id = asyncio.run(
        _run_with_progress(
            sup,
            goal,
            preferences_text=preferences_text,
            n_initial=n_initial,
            wall_clock_seconds=wall_clock,
            search_space_items=search_space_items,
        )
    )
    console.print(f"[green]Done.[/green] session={session_id}")
    console.print(f"View report:  co-scientist report {session_id}")


def _ask_int(label: str, default: int, *, minimum: int = 1) -> int:
    while True:
        value = IntPrompt.ask(label, default=default)
        if value >= minimum:
            return value
        console.print(f"[red]Enter a value >= {minimum}.[/red]")


def _ask_float(label: str, default: float, *, minimum: float = 0.0) -> float:
    while True:
        value = FloatPrompt.ask(label, default=default)
        if value >= minimum:
            return value
        console.print(f"[red]Enter a value >= {minimum:g}.[/red]")


def _ask_optional_path(label: str) -> Path | None:
    while True:
        raw = Prompt.ask(label, default="").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if path.exists():
            return path
        console.print(f"[red]File not found:[/red] {path}")


def _interactive_home(ctx: typer.Context) -> None:
    if not sys.stdin.isatty():
        console.print(ctx.get_help())
        return

    cfg, _ = ctx.obj
    env_var = provider_key_env(cfg)
    key_status = "[green]ready[/green]" if has_llm_key(cfg) else "[red]missing[/red]"

    console.print()
    console.print(
        Panel(
            "[bold bright_cyan]AI Co-Scientist[/bold bright_cyan]\n"
            "[white]Multi-agent hypothesis generation, review, ranking, and synthesis.[/white]\n\n"
            "[dim]Press Enter to accept defaults. Use Ctrl+C to exit.[/dim]",
            title="[bold bright_magenta]Welcome[/bold bright_magenta]",
            subtitle="[bright_black]Generation -> Reflection -> Ranking -> Meta-review[/bright_black]",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )

    overview = Table(title="Session setup", title_style="bold bright_cyan", box=None)
    overview.add_column("Field", style="bright_cyan")
    overview.add_column("Required", justify="center")
    overview.add_column("Default / current value", style="white")
    overview.add_row("Research goal", "[bold red]yes[/bold red]", "you will type this next")
    overview.add_row("Preferences", "no", "empty")
    overview.add_row("Initial generations", "no", "3")
    overview.add_row("Max ideas", "no", str(cfg.run.max_ideas))
    overview.add_row("Budget", "no", f"${cfg.run.budget_usd:.2f}")
    overview.add_row("Token budget", "no", f"{cfg.run.budget_tokens} (0 = unlimited)")
    overview.add_row("Wall clock", "no", f"{cfg.run.wall_clock_seconds}s")
    overview.add_row("Worker concurrency", "no", str(cfg.run.concurrency))
    overview.add_row("Candidate universe", "no", "none")
    overview.add_row("LLM provider", "config", f"{cfg.llm.provider} / {env_var or 'keyless'} {key_status}")
    console.print(overview)
    console.print()

    goal = ""
    while not goal:
        goal = Prompt.ask("[bold bright_green]Research goal[/bold bright_green]").strip()
        if not goal:
            console.print("[red]Research goal is required.[/red]")

    preferences = Prompt.ask(
        "[bright_green]Preferences[/bright_green] [dim](optional, free text)[/dim]",
        default="",
    ).strip()
    n_initial = _ask_int(
        "[bright_green]Initial generation calls[/bright_green]",
        3,
        minimum=1,
    )
    max_ideas = _ask_int(
        "[bright_green]Max ideas[/bright_green]",
        int(cfg.run.max_ideas),
        minimum=1,
    )
    budget_usd = _ask_float(
        "[bright_green]Budget USD[/bright_green]",
        float(cfg.run.budget_usd),
        minimum=0.01,
    )
    budget_tokens = _ask_int(
        "[bright_green]Token budget[/bright_green] [dim](0 = unlimited)[/dim]",
        int(cfg.run.budget_tokens),
        minimum=0,
    )
    wall_clock = _ask_int(
        "[bright_green]Wall clock seconds[/bright_green]",
        int(cfg.run.wall_clock_seconds),
        minimum=60,
    )
    concurrency = _ask_int(
        "[bright_green]Worker concurrency[/bright_green]",
        int(cfg.run.concurrency),
        minimum=1,
    )
    candidate_universe = _ask_optional_path(
        "[bright_green]Candidate universe file[/bright_green] [dim](optional path)[/dim]"
    )

    summary = Table(title="Ready to launch", title_style="bold bright_magenta")
    summary.add_column("Option", style="bright_cyan")
    summary.add_column("Value", overflow="fold")
    summary.add_row("goal", goal)
    summary.add_row("preferences", preferences or "[dim]none[/dim]")
    summary.add_row("initial generations", str(n_initial))
    summary.add_row("max ideas", str(max_ideas))
    summary.add_row("budget", f"${budget_usd:.2f}")
    summary.add_row("token budget", "unlimited" if budget_tokens <= 0 else str(budget_tokens))
    summary.add_row("wall clock", f"{wall_clock}s")
    summary.add_row("worker concurrency", str(concurrency))
    summary.add_row("candidate universe", str(candidate_universe) if candidate_universe else "[dim]none[/dim]")
    console.print()
    console.print(summary)

    if not Confirm.ask("[bold bright_magenta]Launch session now?[/bold bright_magenta]", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    _start_research_session(
        ctx,
        goal,
        preferences_text=preferences or None,
        n_initial=n_initial,
        wall_clock=wall_clock,
        budget_usd=budget_usd,
        budget_tokens=budget_tokens,
        max_ideas=max_ideas,
        concurrency=concurrency,
        candidate_universe=candidate_universe,
    )


@app.command()
def run(
    ctx: typer.Context,
    goal: str = typer.Argument(..., help="Research goal in natural language."),
    preferences_file: Path | None = typer.Option(
        None, "--preferences-file", help="Path to a text file with extra preferences."
    ),
    n_initial: int = typer.Option(
        3, "--n", help="Number of initial Generation calls (parallel)."
    ),
    wall_clock: int | None = typer.Option(
        None, "--wall-clock", help="Override wall-clock cap in seconds."
    ),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="Override session USD budget."
    ),
    budget_tokens: int | None = typer.Option(
        None, "--budget-tokens", help="Override session token budget. Use 0 for unlimited."
    ),
    max_ideas: int | None = typer.Option(
        None, "--max-ideas", help="Maximum hypotheses to generate before refinement stops."
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Override worker concurrency."
    ),
    candidate_universe: Path | None = typer.Option(
        None,
        "--candidate-universe",
        help="Text/CSV/TSV file defining the exact candidates to evaluate.",
    ),
) -> None:
    """Start a fresh research session. Generation → Reflection → Ranking tournament → Meta-review."""
    prefs = preferences_file.read_text(encoding="utf-8") if preferences_file else None
    _start_research_session(
        ctx,
        goal,
        preferences_text=prefs,
        n_initial=n_initial,
        wall_clock=wall_clock,
        budget_usd=budget_usd,
        budget_tokens=budget_tokens,
        max_ideas=max_ideas,
        concurrency=concurrency,
        candidate_universe=candidate_universe,
    )


@app.command()
def resume(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    budget_usd: float | None = typer.Option(
        None, "--budget-usd", help="Override session USD budget before resuming."
    ),
    budget_tokens: int | None = typer.Option(
        None, "--budget-tokens", help="Override session token budget before resuming. Use 0 for unlimited."
    ),
) -> None:
    """Resume a paused or interrupted session."""
    cfg, _ = ctx.obj
    if not has_llm_key(cfg):
        env_var = provider_key_env(cfg)
        console.print(f"[red]{env_var} is not set (LLM provider = {cfg.llm.provider}). See .env.example.[/red]")
        raise typer.Exit(1)
    if budget_usd is not None or budget_tokens is not None:
        from .storage.db import connect, init_db
        from .storage.repos import sessions as sess_repo

        async def _update_budget() -> None:
            await init_db(cfg)
            conn = await connect(cfg)
            try:
                await sess_repo.set_budget(
                    conn,
                    session_id,
                    budget_tokens=budget_tokens,
                    budget_usd=budget_usd,
                )
            finally:
                await conn.close()

        asyncio.run(_update_budget())
        console.print(
            "[green]Updated budget before resume:[/green] "
            f"token_budget={'unlimited' if budget_tokens == 0 else budget_tokens if budget_tokens is not None else 'unchanged'}, "
            f"budget_usd={budget_usd if budget_usd is not None else 'unchanged'}"
        )
    from .agents.supervisor import Supervisor

    sup = Supervisor(cfg)
    sid = asyncio.run(_run_with_progress(sup, "", resume_session_id=session_id))
    console.print(f"[green]Done.[/green] session={sid}")


@app.command()
def pause(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Pause a running session. Workers drain; the loop sleeps until resume."""
    cfg, _ = ctx.obj

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            from .storage.repos import sessions as sess_repo

            await sess_repo.set_status(conn, session_id, "paused")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[yellow]Paused[/yellow] session={session_id}")


@app.command()
def abort(ctx: typer.Context, session_id: str = typer.Argument(...)) -> None:
    """Abort a running session. The main loop exits at the next check."""
    cfg, _ = ctx.obj

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            from .storage.repos import sessions as sess_repo

            await sess_repo.set_status(conn, session_id, "aborted")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[red]Aborted[/red] session={session_id}")


@app.command()
def feedback(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    text: str = typer.Argument(..., help="Free-text feedback to inject."),
    kind: str = typer.Option(
        "directive", "--kind",
        help="directive | preference | rejection | pin",
    ),
    target: str | None = typer.Option(
        None, "--target", help="Hypothesis ID this feedback is about (optional)."
    ),
) -> None:
    """Inject researcher feedback into a running (or future) session."""
    cfg, _ = ctx.obj
    from . import ids as _ids
    from .models import SystemFeedback
    from .storage.repos import feedback as fb_repo
    from .storage.repos import hypotheses as hyp_repo

    async def _do() -> None:
        conn = await db_mod.connect(cfg)
        try:
            fb = SystemFeedback(
                id=_ids.feedback_id(), session_id=session_id,
                created_at=datetime.now(UTC),
                source="human", kind=kind,
                target_id=target, text=text, active=True,
            )
            await fb_repo.insert(conn, fb)
            # Pinning / rejection also flip the hypothesis state.
            if kind == "pin" and target:
                await hyp_repo.set_state(conn, target, "pinned")
            elif kind == "rejection" and target:
                await hyp_repo.set_state(conn, target, "rejected")
        finally:
            await conn.close()

    asyncio.run(_do())
    console.print(f"[green]Feedback recorded[/green] for session={session_id}")


@app.command()
def report(
    ctx: typer.Context,
    session_id: str = typer.Argument(...),
    format: str = typer.Option("md", "--format", "-f", help="md or json"),
) -> None:
    """Print the final research overview for a session (when available)."""
    cfg, _ = ctx.obj

    async def _path() -> Path | None:
        conn = await db_mod.connect(cfg)
        try:
            async with conn.execute(
                "SELECT final_overview FROM sessions WHERE id = ?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row or not row["final_overview"]:
                return None
            return cfg.data_dir / row["final_overview"]
        finally:
            await conn.close()

    p = asyncio.run(_path())
    if p is None:
        console.print(f"[red]No final overview yet for {session_id}[/red]")
        raise typer.Exit(1)
    text = p.read_text(encoding="utf-8")
    if format == "json":
        console.print_json(data={"session_id": session_id, "path": str(p), "content": text})
    else:
        console.print(text)


@app.command()
def serve(
    ctx: typer.Context,
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Launch the FastAPI + htmx + SSE web UI."""
    cfg, _ = ctx.obj
    host = host or cfg.web_ui.host
    port = port or cfg.web_ui.port
    import uvicorn

    from .web.app import create_app

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


@app.command()
def estimate(ctx: typer.Context) -> None:
    """Print the pre-flight cost estimate without launching a session."""
    cfg, _ = ctx.obj
    from .llm.estimator import estimate as _estimate

    est = _estimate(cfg)
    console.print_json(data=est.to_dict())


@app.command("eval")
def eval_cmd(
    ctx: typer.Context,
    agent: str | None = typer.Argument(None, help="generation|reflection|ranking|overview"),
    offline: bool = typer.Option(False, "--offline", help="Structural checks only; no judge."),
) -> None:
    """Run the eval rubric runner over the bundled fixtures."""
    cfg, _ = ctx.obj
    from .evals.runner import run_agent, run_all

    if agent:
        result = asyncio.run(run_agent(cfg, agent, offline=offline))
    else:
        result = asyncio.run(run_all(cfg, offline=offline))
    console.print_json(data=result)


@app.command("bench")
def bench_cmd(
    ctx: typer.Context,
    goal: str | None = typer.Argument(
        None, help="Research goal. Optional when --preset bundles a default goal."
    ),
    preset: str | None = typer.Option(
        None, "--preset",
        help=(
            "Use a built-in candidate list. Available: "
            "'paper' (Co-Scientist paper baselines + Haiku), "
            "'paper-aml' (same candidates + AML drug-repurposing goal + "
            "gold-set scoring against the paper's answer key), "
            "'deepseek-aml-uplift' (same DeepSeek model in complete session, "
            "Generation-only pipeline, and direct modes). When combined "
            "with --candidate, the preset supplies the goal/gold set while "
            "the custom candidates replace its model list."
        ),
    ),
    candidate: list[str] = typer.Option(
        None, "--candidate", "-c",
        help=(
            "Repeat: label=provider:model[@mode]. Mode is `pipeline` "
            "(default), `session` (complete multi-agent session), or `direct` "
            "(single raw LM call, no tools). e.g. "
            "'gemini-flash=openrouter:google/gemini-3-flash-preview', "
            "'flash-raw=openrouter:google/gemini-3-flash-preview@direct'."
        ),
    ),
    n: int = typer.Option(2, "--n", help="Hypotheses per candidate."),
    matches: int = typer.Option(2, "--matches", help="Tournament matches per pair."),
    judge: str | None = typer.Option(
        None, "--judge",
        help="Judge as provider:model. Defaults to the preset's suggestion, else anthropic:claude-sonnet-4-6.",
    ),
    goldset_label: str | None = typer.Option(
        None, "--goldset",
        help=(
            "Override the preset's gold set, or attach one to a custom-"
            "candidate bench. Built-in labels: 'aml-repurposing-paper-5' "
            "(broader 5-drug list) and 'aml-repurposing-paper-top3' "
            "(paper-derived top-3 entity-recall set). Pass 'none' to "
            "disable gold-set scoring entirely for a preset that defaults "
            "to one."
        ),
    ),
    budget_per_candidate: float = typer.Option(
        3.0, "--budget-per-candidate", help="USD cap per candidate."
    ),
    judge_budget: float = typer.Option(
        5.0, "--judge-budget", help="USD cap for all judge calls combined."
    ),
    candidate_universe: Path | None = typer.Option(
        None,
        "--candidate-universe",
        help="Text/CSV/TSV file defining the exact candidate universe for @session/@pipeline.",
    ),
) -> None:
    """Compare N models on the same goal via a cross-Elo tournament.

    Quick start (paper repro):
      co-scientist bench "Identify hypotheses about X" \\
        --preset paper \\
        --judge openrouter:google/gemini-3-flash-preview

    Custom candidates:
      co-scientist bench "Identify hypotheses about X" \\
        -c gemini-flash=openrouter:google/gemini-3-flash-preview \\
        -c gpt5=openai:gpt-5 \\
        -c opus=anthropic:claude-opus-4-7 \\
        --judge anthropic:claude-sonnet-4-6
    """
    cfg, _ = ctx.obj
    console.print("[dim]bench: parsing candidates and presets...[/dim]")
    from .bench import BenchCandidate, get_preset, run_bench

    def _custom_candidates(entries: list[str]) -> list[BenchCandidate]:
        parsed: list[BenchCandidate] = []
        for entry in entries:
            if "=" not in entry or ":" not in entry.split("=", 1)[1]:
                console.print(
                    f"[red]--candidate must look like label=provider:model[@mode], got {entry!r}[/red]"
                )
                raise typer.Exit(2)
            label, rest = entry.split("=", 1)
            mode = "pipeline"
            if "@" in rest:
                rest, mode = rest.rsplit("@", 1)
                mode = mode.strip().lower()
                if mode not in ("session", "pipeline", "direct"):
                    console.print(
                        f"[red]unknown mode {mode!r} in {entry!r}; "
                        f"use `session`, `pipeline`, or `direct`[/red]"
                    )
                    raise typer.Exit(2)
            provider, model = rest.split(":", 1)
            parsed.append(BenchCandidate(
                label=label, provider=provider, model=model, mode=mode,
            ))
        return parsed

    candidates: list[BenchCandidate]
    goldset = None
    if preset:
        p = get_preset(preset)
        candidates = _custom_candidates(candidate) if candidate else list(p.candidates)
        if judge is None:
            judge = p.suggested_judge
        if goal is None and p.default_goal is not None:
            goal = p.default_goal
        if p.goldset is not None:
            goldset = p.goldset
        console.print(f"[dim]Using preset '{p.name}': {p.description}[/dim]")
        for c in candidates:
            mode_suffix = f" [{c.mode}]" if c.mode != "pipeline" else ""
            console.print(f"[dim]  • {c.label}{mode_suffix}: {c.provider}:{c.model}[/dim]")
    else:
        if not candidate:
            console.print(
                "[red]Must provide either --preset or at least one --candidate[/red]"
            )
            raise typer.Exit(2)
        candidates = _custom_candidates(candidate)

    # --goldset override: take effect after preset defaults so users can
    # swap the gold set without rewriting --candidate lists, or attach a
    # gold set to a custom-candidate bench.
    if goldset_label is not None:
        from .bench import GOLDSETS
        if goldset_label.lower() == "none":
            goldset = None
        elif goldset_label in GOLDSETS:
            goldset = GOLDSETS[goldset_label]
        else:
            names = ", ".join(sorted(GOLDSETS))
            console.print(
                f"[red]unknown gold set {goldset_label!r}. "
                f"Available: {names}, or 'none'.[/red]"
            )
            raise typer.Exit(2)

    if goldset:
        ent_list = ", ".join(e.name for e in goldset.entities)
        console.print(f"[dim]  gold set: {goldset.label} ({ent_list})[/dim]")
    if any(c.mode == "pipeline" for c in candidates):
        console.print(
            "[yellow]Note: bench @pipeline evaluates the Generation agent only "
            "(tool loop + literature retrieval + dedup). It does not run the "
            "Supervisor/Reflection/Ranking/Evolution/Meta-review session loop.[/yellow]"
        )
    if any(c.mode == "session" for c in candidates):
        console.print(
            "[dim]bench @session runs the complete multi-agent session; --n sets "
            "the maximum hypothesis pool size, not just repeated raw samples.[/dim]"
        )

    if goal is None:
        console.print(
            "[red]Must provide a research goal (positional argument) or a "
            "--preset that bundles one.[/red]"
        )
        raise typer.Exit(2)
    if judge is None:
        judge = "anthropic:claude-sonnet-4-6"
    if ":" not in judge:
        console.print(f"[red]--judge must look like provider:model, got {judge!r}[/red]")
        raise typer.Exit(2)
    judge_provider, judge_model = judge.split(":", 1)
    search_space_items = None
    if candidate_universe is not None:
        from .search_space import load_candidate_universe

        console.print(f"[dim]bench: loading candidate universe {candidate_universe}[/dim]")
        search_space_items = load_candidate_universe(candidate_universe)
        if n > len(search_space_items):
            n = len(search_space_items)
        console.print(
            f"[dim]Constrained candidate universe: evaluating {n}/"
            f"{len(search_space_items)} items[/dim]"
        )
        if any(c.mode != "session" for c in candidates):
            console.print(
                "[yellow]Candidate-universe paper reproduction should use "
                "@session. Generation-only/direct modes do not produce an "
                "internal ranked shortlist.[/yellow]"
            )

    console.print("[dim]bench: entering async runner...[/dim]")
    outcome = asyncio.run(
        run_bench(
            cfg, goal=goal, candidates=candidates,
            n_hyps_per_candidate=n,
            matches_per_pair=matches,
            judge_provider=judge_provider, judge_model=judge_model,
            per_candidate_budget_usd=budget_per_candidate,
            judge_budget_usd=judge_budget,
            goldset=goldset,
            search_space_items=search_space_items,
        )
    )

    has_gold = goldset is not None
    title = f"Bench {outcome.bench_id} — {outcome.matches_played} matches"
    if has_gold:
        title += f" • gold-set {goldset.label} (recall / {len(goldset.entities)})"
    tbl = Table(title=title)
    tbl.add_column("rank", justify="right")
    tbl.add_column("label", style="bold")
    tbl.add_column("mode")
    tbl.add_column("model")
    tbl.add_column("n_hyps", justify="right")
    tbl.add_column("W-L", justify="right")
    tbl.add_column("mean_elo", justify="right")
    if has_gold:
        tbl.add_column("gold hits", justify="right")
    tbl.add_column("$ spent", justify="right")
    tbl.add_column("p50_ms", justify="right")
    for i, row in enumerate(outcome.candidates, 1):
        row_cells = [
            str(i), row["label"],
            row.get("mode") or "pipeline",
            row["model"],
            str(row["n_hypotheses"]),
            f"{row['wins']}-{row['losses']}",
            f"{row['mean_elo']:.0f}" if row["mean_elo"] is not None else "—",
        ]
        if has_gold:
            n_hit = row.get("gold_hits") or 0
            row_cells.append(f"{n_hit}/{len(goldset.entities)}")
        row_cells.extend([
            f"{row['cost_usd']:.4f}",
            str(row["mean_latency_ms"] or "—"),
        ])
        tbl.add_row(*row_cells)
    console.print(tbl)
    console.print(f"[dim]Total cost: ${outcome.total_cost_usd:.4f}[/dim]")
    console.print(f"[dim]Artifact: {outcome.artifact_path}[/dim]")

    if has_gold:
        # Per-candidate hit detail so you can see which specific entities surfaced.
        for row in outcome.candidates:
            hits = row.get("gold_hit_names") or []
            if hits:
                console.print(
                    f"[dim]  {row['label']} surfaced: {', '.join(hits)}[/dim]"
                )


@tools_app.command("list")
def tools_list(
    ctx: typer.Context,
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="table | json. JSON shows the Anthropic-style tool schema used internally.",
    ),
    agent: str | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Only show tools available to one agent: generation|reflection|ranking|evolution|proximity|metareview.",
    ),
) -> None:
    """List registered tools (builtins + any discovered science-skills)."""
    cfg, _ = ctx.obj
    from .tools.registry import AGENT_TOOLS, ToolRegistry

    reg = ToolRegistry(cfg).discover()
    if agent is not None and agent not in AGENT_TOOLS:
        console.print(
            f"[red]unknown agent {agent!r}; available: {', '.join(sorted(AGENT_TOOLS))}[/red]"
        )
        raise typer.Exit(2)

    if format == "json":
        if agent is not None:
            console.print_json(data={
                "agent": agent,
                "tools": reg.anthropic_tools_for(agent),
            })
        else:
            console.print_json(data={
                "tools": [
                    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                    for t in sorted(reg.all(), key=lambda x: x.name)
                ],
                "agents": {
                    name: reg.anthropic_tools_for(name)
                    for name in sorted(AGENT_TOOLS)
                },
            })
        return
    if format != "table":
        console.print("[red]--format must be 'table' or 'json'[/red]")
        raise typer.Exit(2)

    tbl = Table(title=f"Tools ({len(reg.all())})")
    tbl.add_column("name", style="bold")
    tbl.add_column("description", overflow="fold")
    tools = reg.tools_for(agent) if agent is not None else sorted(reg.all(), key=lambda x: x.name)
    for item in tools:
        tbl.add_row(item.name, item.description[:200])
    console.print(tbl)

    # Show per-agent allowlist resolution counts
    if agent is None:
        tbl2 = Table(title="Per-agent tool availability")
        tbl2.add_column("agent")
        tbl2.add_column("# tools", justify="right")
        tbl2.add_column("allowlist (patterns)")
        for name, patterns in AGENT_TOOLS.items():
            ts = reg.tools_for(name)
            tbl2.add_row(name, str(len(ts)), ", ".join(sorted(patterns)) or "—")
        console.print(tbl2)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
