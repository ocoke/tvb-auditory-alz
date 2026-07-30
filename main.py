#!/usr/bin/env python3
"""Run the canonical DTGateFixed TVB379 experiment without Jupyter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from rise_tvb379.notebook_runner import (
    format_validation_summary,
    run_notebook,
    validate_notebook,
)
from rise_tvb379.run_state import (
    RunStateError,
    format_run_status,
    read_run_status,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CACHE = PROJECT_ROOT / "data"


def _worker_count(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    try:
        count = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "workers must be 'auto' or a positive integer"
        ) from error
    if count < 1:
        raise argparse.ArgumentTypeError(
            "workers must be 'auto' or a positive integer"
        )
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical DTGateFixed 379-region semantic-versus-"
            "episodic musical-memory proxy experiment. The default is the "
            "locked 762-call final workload with 40 integration-step work "
            "units."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "final"),
        default=None,
        help="notebook workload (default: final)",
    )
    parser.add_argument(
        "--workers",
        type=_worker_count,
        default=None,
        metavar="N|auto",
        help=(
            "parallel worker processes; 'auto' uses the detected CPU "
            "allocation (default: auto)"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "optional result-directory suffix; the notebook otherwise uses "
            "semantic_episodic_v3_<mode>"
        ),
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=DEFAULT_DATA_CACHE,
        help=(
            "directory containing the five pinned source files "
            f"(default: {DEFAULT_DATA_CACHE})"
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "validate and compile all canonical notebook code cells, print "
            "the locked workloads, and exit without running TVB"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="resume one compatible incomplete DTGateFixed run",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="print the saved status of a DTGateFixed run and exit",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume is not None:
        conflicts = [
            name
            for name, value in (
                ("--mode", args.mode),
                ("--run-id", args.run_id),
                ("--check", args.check),
                ("--status", args.status),
            )
            if value is not None and value is not False
        ]
        if conflicts:
            parser.error(
                "--resume cannot be combined with " + ", ".join(conflicts)
            )
    if args.status is not None:
        conflicts = [
            name
            for name, value in (
                ("--mode", args.mode),
                ("--workers", args.workers),
                ("--run-id", args.run_id),
                ("--check", args.check),
            )
            if value is not None and value is not False
        ]
        if conflicts:
            parser.error(
                "--status cannot be combined with " + ", ".join(conflicts)
            )
    return args


def _require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "This project uses Python 3.12; the active interpreter is "
            f"{sys.version.split()[0]}. Activate a Python 3.12 environment "
            "and try again."
        )


def _configure_environment(
    args: argparse.Namespace,
    *,
    mode: str,
) -> None:
    os.environ["RISE_RUN_MODE"] = mode
    if args.workers is None:
        os.environ.pop("RISE_N_WORKERS", None)
    else:
        os.environ["RISE_N_WORKERS"] = str(args.workers)

    if args.resume is not None:
        os.environ["RISE_RUN_ID"] = (
            f"resume_bootstrap_{os.getpid()}"
        )
    elif args.run_id is None:
        os.environ.pop("RISE_RUN_ID", None)
    else:
        os.environ["RISE_RUN_ID"] = args.run_id

    os.environ["RISE_DATA_CACHE"] = str(args.data_cache.resolve())
    os.environ.setdefault("MPLBACKEND", "Agg")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status is not None:
        run_dir = args.status.expanduser().resolve()
        try:
            status = read_run_status(run_dir)
        except RunStateError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        print(format_run_status(run_dir, status))
        return 0

    _require_python_312()
    validation = validate_notebook()
    print(format_validation_summary(validation), flush=True)
    if args.check:
        print("Static DTGateFixed smoke check passed; no TVB calls ran.")
        return 0

    resume_dir = (
        args.resume.expanduser().resolve()
        if args.resume is not None
        else None
    )
    if resume_dir is None:
        mode = args.mode or "final"
    else:
        try:
            saved_status = read_run_status(resume_dir)
        except RunStateError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        if saved_status.get("state") == "completed":
            print(
                f"Error: run is already completed: {resume_dir}.",
                file=sys.stderr,
            )
            return 1
        mode = str(saved_status.get("mode", ""))
        if mode not in validation.workloads:
            print(
                f"Error: saved run mode is invalid: {mode!r}.",
                file=sys.stderr,
            )
            return 1

    _configure_environment(args, mode=mode)
    os.chdir(PROJECT_ROOT)
    try:
        run_notebook(
            validation=validation,
            resume_dir=resume_dir,
        )
    except KeyboardInterrupt:
        if resume_dir is None:
            print(
                "Run interrupted; the run directory printed above can be "
                "passed to --resume.",
                file=sys.stderr,
            )
        else:
            print(
                f"Run interrupted; resume again with --resume {resume_dir}.",
                file=sys.stderr,
            )
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
