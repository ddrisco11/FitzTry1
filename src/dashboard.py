"""Phase 4 — Real-Time Extraction Dashboard.

Run in a separate terminal while phase 4 is extracting:

    python -m src.dashboard
    python -m src.dashboard --progress data/phase4_progress.json
    python -m src.dashboard --refresh 1.5

Displays:
  • Progress bar with chunk count and percentage
  • ETA, elapsed time, rate, avg time per chunk
  • Relation type breakdown table with mini bar charts and projected totals
  • Live relation feed (most recent extractions)
  • Current chunk preview
  • Error log (if any failures)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import click
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ── Colour palette ────────────────────────────────────────────────────────────

RELATION_COLOURS: Dict[str, str] = {
    "within":         "bright_cyan",
    "contains":       "cyan",
    "part_of":        "steel_blue1",
    "near":           "green",
    "far":            "red",
    "north_of":       "yellow",
    "south_of":       "yellow",
    "east_of":        "gold1",
    "west_of":        "gold1",
    "across":         "magenta",
    "on_coast":       "deep_sky_blue1",
    "on_shore_of":    "deep_sky_blue3",
    "borders":        "bright_yellow",
    "connected_via":  "bright_green",
    "distance_approx":"white",
}

STATUS_COLOURS: Dict[str, str] = {
    "running":  "bright_green",
    "complete": "green",
    "error":    "red",
    "waiting":  "dim white",
}

# ── Rendering helpers ─────────────────────────────────────────────────────────

_BLOCK = "█"
_EMPTY = "░"
_BAR_FRACS = "▏▎▍▌▋▊▉█"


def _mini_bar(fraction: float, width: int = 10) -> str:
    filled_f = max(0.0, min(1.0, fraction)) * width
    full     = int(filled_f)
    partial  = filled_f - full
    bar      = _BLOCK * full
    if full < width and partial > 0:
        bar += _BAR_FRACS[int(partial * len(_BAR_FRACS))]
    return bar.ljust(width, " ")


def _big_bar(fraction: float, width: int = 52) -> str:
    filled = int(fraction * width)
    return _BLOCK * filled + _EMPTY * (width - filled)


def _hms(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


# ── Panel builders ────────────────────────────────────────────────────────────

def _header_panel(state: dict) -> Panel:
    doc   = state.get("doc_id", "?").replace("_", " ").title()
    model = state.get("model", "?")
    status = state.get("status", "running")
    sc     = STATUS_COLOURS.get(status, "white")

    title = Text(justify="center")
    title.append("  LITERARY GEOGRAPHY PIPELINE  ", style="bold white on dark_blue")

    sub = Text(justify="center")
    sub.append(f"{doc}", style="bold white")
    sub.append("  ·  ", style="dim white")
    sub.append(f"{model}", style="cyan")
    sub.append("  ·  status: ", style="dim white")
    sub.append(f"{status}", style=f"bold {sc}")

    return Panel(
        Group(Align(title, align="center"), Align(sub, align="center")),
        box=box.DOUBLE_EDGE,
        border_style="bright_blue",
        padding=(0, 1),
    )


def _progress_panel(state: dict) -> Panel:
    total = state.get("total_chunks", 1) or 1
    done  = state.get("processed_chunks", 0)
    errs  = state.get("error_chunks", 0)
    pct   = done / total

    bar = _big_bar(pct, width=52)
    t   = Text()
    t.append(f"  {bar}  ", style="bright_green")
    t.append(f"{pct*100:5.1f}%  ", style="bold white")
    t.append(f"{done}", style="bold white")
    t.append(f"/{total} chunks", style="dim white")
    if errs:
        t.append(f"  ({errs} errors)", style="dim red")

    return Panel(t, box=box.SIMPLE, padding=(0, 0))


def _stats_panel(state: dict, elapsed_s: float) -> Panel:
    total   = state.get("total_chunks", 1) or 1
    done    = state.get("processed_chunks", 0)
    errs    = state.get("error_chunks", 0)
    rels    = state.get("relations_extracted", 0)
    durs    = state.get("chunk_durations_s", [])
    pct     = done / total

    avg_dur      = sum(durs[-20:]) / len(durs[-20:]) if durs else 0.0
    rate_min     = (60.0 / avg_dur) if avg_dur > 0 else 0.0
    remaining    = total - done
    eta_s        = remaining * avg_dur if avg_dur > 0 else 0.0
    proj_total   = int(rels / pct) if pct > 0 else 0

    tbl = Table(box=None, padding=(0, 2), show_header=False)
    tbl.add_column(style="dim white", width=15, no_wrap=True)
    tbl.add_column(style="bold white", no_wrap=True)

    eta_col = "green" if eta_s < 3600 else "yellow" if eta_s < 7200 else "red"

    tbl.add_row("Elapsed",    f"[white]{_hms(elapsed_s)}[/white]")
    tbl.add_row("ETA",        f"[{eta_col}]{_hms(eta_s)}[/{eta_col}]")
    tbl.add_row("Rate",       f"[cyan]{rate_min:.1f}[/cyan] chunks/min")
    tbl.add_row("Avg/chunk",  f"[white]{avg_dur:.1f} s[/white]")
    tbl.add_row("Projected",  f"[bright_white]{proj_total}[/bright_white] relations")
    tbl.add_row("Errors",     f"[{'red' if errs else 'dim green'}]{errs}[/]")

    return Panel(tbl, title="[bold]Stats[/bold]", border_style="blue", box=box.ROUNDED)


def _relations_panel(state: dict) -> Panel:
    total    = state.get("total_chunks", 1) or 1
    done     = state.get("processed_chunks", 0)
    rels     = state.get("relations_extracted", 0)
    counts   = state.get("relation_type_counts", {})
    pct      = done / total

    proj_total = int(rels / pct) if pct > 0 else 0
    max_count  = max(counts.values(), default=1)

    tbl = Table(
        box=box.SIMPLE_HEAD,
        padding=(0, 1),
        show_header=True,
        header_style="bold dim white",
        expand=True,
    )
    tbl.add_column("Relation Type",  style="white",    width=16, no_wrap=True)
    tbl.add_column("Count",          style="bold",     justify="right", width=6)
    tbl.add_column("Projected",      style="dim",      justify="right", width=8)
    tbl.add_column("Distribution",   style="white",    width=12, no_wrap=True)

    for rt, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        proj  = int(cnt / pct) if pct > 0 else cnt
        frac  = cnt / max_count
        col   = RELATION_COLOURS.get(rt, "white")
        bar   = _mini_bar(frac, width=10)
        tbl.add_row(
            f"[{col}]{rt}[/{col}]",
            f"[bold {col}]{cnt}[/]",
            f"[dim]{proj}[/dim]",
            f"[{col}]{bar}[/]",
        )

    title = (
        f"[bold]Relations: [bright_white]{rels}[/bright_white]"
        f"  ·  Projected: [bright_green]{proj_total}[/bright_green][/bold]"
    )
    return Panel(tbl, title=title, border_style="blue", box=box.ROUNDED)


def _feed_panel(state: dict) -> Panel:
    recent = state.get("recent_relations", [])

    if not recent:
        body = Text("  Waiting for first relations...", style="dim white italic")
    else:
        body = Text()
        for r in reversed(recent):
            e1   = str(r.get("entity_1", "?"))
            rt   = str(r.get("relation_type", "?"))
            e2   = str(r.get("entity_2", "") or "∅")
            conf = float(r.get("confidence", 0.0))
            col  = RELATION_COLOURS.get(rt, "white")
            cc   = "bright_green" if conf >= 0.8 else "yellow" if conf >= 0.65 else "dim white"

            body.append(f"  {e1:<22}", style="bold white")
            body.append(f"──[{col}]{rt}[/]──▶ ")
            body.append(f"{e2:<22}", style="bold white")
            body.append(f"  [{cc}]{conf:.2f}[/]\n")

    return Panel(
        body,
        title="[bold]Live Relation Feed[/bold]",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _chunk_panel(state: dict) -> Panel:
    preview = state.get("current_chunk_preview", "").strip()
    if preview:
        body = Text(f'  "{preview[:220]}…"', style="dim italic white")
    else:
        body = Text("  —", style="dim white")
    return Panel(body, title="[bold]Current Chunk[/bold]", border_style="dark_blue", box=box.ROUNDED)


def _error_panel(state: dict) -> Optional[Panel]:
    errors = state.get("error_log", [])
    if not errors:
        return None
    body = Text()
    for e in errors[-4:]:
        body.append(f"  {e}\n", style="dim red")
    return Panel(body, title="[bold red]Errors[/bold red]", border_style="red", box=box.ROUNDED)


def _waiting_panel(progress_path: Path) -> Panel:
    body = Text(justify="center")
    body.append("\n  Waiting for extraction to begin…\n\n", style="dim white italic")
    body.append(f"  Watching: {progress_path}\n\n", style="dim white")
    body.append(
        "  Start extraction:\n"
        "    python -m src.pipeline --config config.yaml --phase 4 --force\n",
        style="dim cyan",
    )
    return Panel(
        Align(body, align="center"),
        title="[bold bright_blue]Phase 4 Dashboard[/bold bright_blue]",
        box=box.DOUBLE_EDGE,
        border_style="bright_blue",
    )


# ── Full display assembly ─────────────────────────────────────────────────────

def _build_display(state: dict, elapsed_s: float) -> Group:
    header   = _header_panel(state)
    progress = _progress_panel(state)

    # Two-column middle: stats | relation breakdown
    middle = Columns(
        [_stats_panel(state, elapsed_s), _relations_panel(state)],
        equal=False,
        expand=True,
    )

    feed  = _feed_panel(state)
    chunk = _chunk_panel(state)
    err   = _error_panel(state)

    items = [header, progress, Rule(style="dim blue"), middle, Rule(style="dim blue"), feed, chunk]
    if err:
        items.append(err)

    return Group(*items)


# ── CLI entry point ───────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--progress",
    default="data/phase4_progress.json",
    show_default=True,
    help="Path to the phase4 progress JSON file written by the extractor.",
)
@click.option(
    "--refresh",
    default=2.0,
    show_default=True,
    type=float,
    help="Dashboard refresh interval in seconds.",
)
def main(progress: str, refresh: float) -> None:
    """Real-time terminal dashboard for Phase 4 relation extraction."""
    progress_path = Path(progress)
    console = Console()

    # Use the extraction's own start time if available; fall back to dashboard start
    wall_start = time.monotonic()

    with Live(
        console=console,
        refresh_per_second=max(0.1, 1.0 / refresh),
        screen=True,
        vertical_overflow="crop",
    ) as live:
        while True:
            state = _load(progress_path)

            if state is None:
                live.update(_waiting_panel(progress_path))
            else:
                # Compute elapsed from the extraction's own start timestamp if present
                started_at = state.get("started_at")
                if started_at:
                    try:
                        t0 = datetime.fromisoformat(started_at)
                        elapsed_s = (datetime.now(timezone.utc) - t0).total_seconds()
                    except Exception:
                        elapsed_s = time.monotonic() - wall_start
                else:
                    elapsed_s = time.monotonic() - wall_start

                live.update(_build_display(state, elapsed_s))

                if state.get("status") in ("complete", "error"):
                    time.sleep(refresh)
                    # One final update so the completed state is clearly shown
                    live.update(_build_display(_load(progress_path) or state, elapsed_s))
                    break

            time.sleep(refresh)

    # Post-exit summary line
    if state and state.get("status") == "complete":
        rels = state.get("relations_extracted", 0)
        console.print(
            f"\n[bold bright_green]Extraction complete —[/] "
            f"[bold white]{rels}[/] relations extracted.\n"
        )
    else:
        console.print("\n[bold red]Extraction ended (check error log above).[/bold red]\n")


if __name__ == "__main__":
    main()
