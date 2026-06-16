"""
CLI entry point for Flint AI CLI.

Usage:
    python -m flintai.cli init
    python -m flintai.cli eval models list
    python -m flintai.cli eval run <model-evaluation-id>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
from rich.panel import Panel
from rich.text import Text

from flintai.cli import eval_cli, scan_cli, init_cli
from flintai.cli.console import CLI_WIDTH, console
from flintai.cli.version import VERSION
from flintai.cli.utils import is_ci
from flintai.eval.common.log import setup_file_logging
from flintai.eval.db.json.repository_json import JsonRepository

logger = logging.getLogger(__name__)

_LOGO_LINES = [
    "███████╗██╗     ██╗███╗   ██╗████████╗     █████╗ ██╗",
    "██╔════╝██║     ██║████╗  ██║╚══██╔══╝    ██╔══██╗██║",
    "█████╗  ██║     ██║██╔██╗ ██║   ██║       ███████║██║",
    "██╔══╝  ██║     ██║██║╚██╗██║   ██║       ██╔══██║██║",
    "██║     ███████╗██║██║ ╚████║   ██║       ██║  ██║██║",
    "╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝╚═╝",
]


_TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")

_BUILTIN_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "builtin_config.json",
)



def _count(repo: JsonRepository) -> dict[str, int]:
    return {
        "models": len(repo.models.list()),
        "evaluations": len(repo.evaluations.list()),
        "detectors": len(repo.detectors.list()),
        "collections": len(repo.message_collections.list()),
        "assignments": len(repo.model_evaluations.list_all()),
    }


def _fmt_count(total: int, builtin: int, user: int) -> str:
    return f"{total} [dim]({builtin} builtin, {user} user)[/dim]"


def _print_logo() -> None:
    console.print()
    for line in _LOGO_LINES:
        pad = max(0, (CLI_WIDTH - len(line)) // 2)
        console.print(" " * pad + f"[bold cyan]{line}[/bold cyan]")
    version_text = f"v{VERSION}"
    pad = max(0, (CLI_WIDTH - len(version_text)) // 2)
    console.print(" " * pad + f"[dim][bold cyan]{version_text}[/bold cyan][/dim]")
    console.print()


def _print_config_info(
    merged: JsonRepository,
    builtin: JsonRepository,
    user: JsonRepository,
    config_path: str,
) -> None:
    m = _count(merged)
    b = _count(builtin)
    u = _count(user)

    console.print(f"[dim]Config:       {config_path}[/dim]")
    console.print(f"[dim]Models:       {_fmt_count(m['models'], b['models'], u['models'])}[/dim]")
    console.print(f"[dim]Evaluations:  {_fmt_count(m['evaluations'], b['evaluations'], u['evaluations'])}[/dim]")
    console.print(f"[dim]Detectors:    {_fmt_count(m['detectors'], b['detectors'], u['detectors'])}[/dim]")
    console.print(f"[dim]Collections:  {_fmt_count(m['collections'], b['collections'], u['collections'])}[/dim]")
    console.print(f"[dim]Assignments:  {_fmt_count(m['assignments'], b['assignments'], u['assignments'])}[/dim]")
    console.print()


def _print_path(
    path: str,
) -> None:
    console.print(f"[dim]Path:         {path}[/dim]")
    console.print()


def _print_shutdown(
    elapsed: float,
    output_path: str | None,
    log_path: str | None,
) -> None:
    console.print()
    parts = [f"[dim]Completed in [bold cyan]{elapsed:.1f}s[/bold cyan][/dim]"]
    if output_path:
        parts.append(f"[dim]Results: [bold cyan]{output_path}[/bold cyan][/dim]")
    if log_path:
        parts.append(f"[dim]Logs: [bold cyan]{log_path}[/bold cyan][/dim]")
    console.print("  ".join(parts))
    console.print()


def _dispatch(args: argparse.Namespace) -> str | None:
    logger.info("Command: %s", " ".join(sys.argv))

    if args.command == "eval":
        return _dispatch_eval(args)
    elif args.command == "scan":
        return _dispatch_scan(args)
    elif args.command == "init":
        return _dispatch_init(args)
    else:
        console.print(f"[red]Unknown command: {args.command}[/red]")
        sys.exit(1)


def _dispatch_init(args: argparse.Namespace) -> str | None:
    init_cli.run_init()
    return None


def _dispatch_scan(args: argparse.Namespace) -> None | None:
    logger.info("flintai v%s | scan mode", VERSION)
    _print_path(args.path)
    return scan_cli.handle_scan(args)


def _dispatch_eval(args: argparse.Namespace) -> str | None:
    """Run the requested command. Returns output file path if any."""
    if args.command != "eval":
        console.print(f"[red]Unknown command: {args.command}[/red]")
        sys.exit(1)

    config_path = getattr(args, "config", None) or str(
        init_cli.get_flintai_config_path(),
    )
    builtin = JsonRepository(_BUILTIN_CONFIG)
    user = JsonRepository(config_path)
    store = user.merge(builtin)

    _print_config_info(store, builtin, user, config_path)

    m = _count(store)
    logger.info(
        "flintai v%s | models=%d, evaluations=%d, assignments=%d",
        VERSION, m["models"], m["evaluations"], m["assignments"],
    )

    cmd = args.eval_cmd
    if cmd == "models":
        eval_cli.handle_models(args, store)
    elif cmd == "evaluations":
        eval_cli.handle_evaluations(args, store)
    elif cmd == "model-evaluations":
        eval_cli.handle_model_evaluations(
            args, store, user,
        )
    elif cmd == "run":
        return asyncio.run(eval_cli.handle_run(args, store))
    else:
        console.print(f"[red]Unknown eval subcommand: {cmd}[/red]")
        sys.exit(1)
    return None


def _print_error(error: BaseException) -> None:
    error_type = type(error).__name__
    error_msg = str(error) or "(no message)"

    body = Text()
    body.append(f"{error_type}: ", style="bold red")
    body.append(error_msg)

    panel = Panel(
        body,
        title="[bold red]Error[/bold red]",
        border_style="red",
        width=min(CLI_WIDTH, max(60, len(error_msg) + 20)),
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="flintai",
        description="Flint AI CLI — AI Agent Evaluation Framework",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    eval_cli.register(subparsers)
    scan_cli.register(subparsers)
    init_cli.register(subparsers)

    args = parser.parse_args(argv)

    ci = is_ci()
    flintai_env = init_cli.get_flintai_env_path()

    if not flintai_env.exists() and args.command != "init":
        if ci:
            console.print(
                "[dim]CI environment detected —"
                " skipping interactive init.[/dim]",
            )
        else:
            console.print(
                "[yellow]First-time setup required."
                " Running flintai init...[/yellow]",
            )
            console.print()
            init_cli.run_init()

    if flintai_env.exists():
        load_dotenv(flintai_env)

    log_path = getattr(args, "log", None) or f"flintai_{_TIMESTAMP}.log"
    setup_file_logging(log_path)
    _print_logo()

    t0 = time.monotonic()
    output_path: str | None = None
    try:
        output_path = _dispatch(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)
    except Exception as e:
        logger.critical(
            "Fatal error: %s: %s\n%s",
            type(e).__name__, e, traceback.format_exc(),
        )
        _print_error(e)
        elapsed = time.monotonic() - t0
        _print_shutdown(elapsed, output_path, log_path)
        sys.exit(1)
    elapsed = time.monotonic() - t0
    _print_shutdown(elapsed, output_path, log_path)