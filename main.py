#!/usr/bin/env python3
"""Command-line entry point for the direct-run RISE TVB379 experiment."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PINNED_RUNTIME = {
    "numpy": "2.0.2",
    "pandas": "2.2.2",
    "scipy": "1.16.3",
    "matplotlib": "3.11.1",
    "tvb-library": "2.10.0",
    "tvb-data": "3.0.0",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 379-region RISE TVB experiment. With no arguments this "
            "starts final mode, configured for 100 spatial shuffles if the "
            "required convergence gate passes."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "final"),
        default=None,
        help="new-run workload (default: final)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"new-run parent directory (default: {PROJECT_ROOT / 'runs'})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"verified input cache (default: {PROJECT_ROOT / 'data'})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="forbid downloads and require all verified inputs locally",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="explicitly resume one compatible incomplete run directory",
    )
    return parser


def parse_args(
    argv: list[str] | None = None,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    parser = parser or build_parser()
    args = parser.parse_args(argv)
    if args.resume is not None:
        conflicting = [
            flag
            for flag, value in (
                ("--mode", args.mode),
                ("--output-root", args.output_root),
                ("--data-dir", args.data_dir),
                ("--offline", args.offline),
            )
            if value is not None
        ]
        if conflicting:
            parser.error(
                "--resume is mutually exclusive with new-run options: "
                + ", ".join(conflicting)
            )
    return args


def _preflight_runtime() -> dict[str, Any]:
    from rise_tvb379.runtime import capture_environment

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "This project requires Python 3.12 for numerical "
            f"reproducibility; current interpreter is {sys.version.split()[0]}. "
            "Create/activate a Python 3.12 environment and try again."
        )
    environment = capture_environment(PINNED_RUNTIME)
    versions = environment["packages"]
    mismatches = [
        f"{name}=={expected} (found {versions.get(name) or 'not installed'})"
        for name, expected in PINNED_RUNTIME.items()
        if versions.get(name) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Pinned runtime dependencies are missing or mismatched:\n  - "
            + "\n  - ".join(mismatches)
            + "\nInstall only the third-party dependencies with:\n"
            "  python -m pip install -r requirements.txt"
        )
    return environment


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _aggregate_tvb_wall_seconds(
    calibration_df: Any,
    manifest_df: Any,
) -> float:
    """Sum every recorded TVB call, including separate calibration rows."""

    return float(
        calibration_df["wall_seconds"].sum()
        + manifest_df["wall_seconds"].sum()
    )


def _configure_logging(run_dir: Path) -> None:
    log_path = run_dir / "run.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(processName)s | "
        "%(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True,
    )


def _prepare_runtime_directories(run_dir: Path) -> None:
    runtime_root = run_dir / ".runtime"
    tvb_home = runtime_root / "tvb"
    matplotlib_home = runtime_root / "matplotlib"
    tvb_home.mkdir(parents=True, exist_ok=True)
    matplotlib_home.mkdir(parents=True, exist_ok=True)
    os.environ["TVB_USER_HOME"] = str(tvb_home)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_home)


def _materialize_new_run_inputs(
    run_dir: Path,
    data_dir: Path,
    *,
    offline: bool,
):
    from rise_tvb379.data import (
        SOURCE_SPECS,
        SourceManifestRecord,
        data_cache_from_environment,
        download_sources,
    )

    cached_records = download_sources(
        data_dir,
        cache_dir=data_cache_from_environment(),
        offline=offline,
    )
    copied_records = download_sources(
        run_dir / "inputs",
        cache_dir=data_dir,
        offline=True,
    )
    cached_by_source = {record.source: record for record in cached_records}
    records = tuple(
        SourceManifestRecord(
            source=copy.source,
            path=copy.path,
            sha256=copy.sha256,
            retrieved_from=cached_by_source[
                copy.source
            ].retrieved_from,
        )
        for copy in copied_records
    )
    manifest_inputs = {
        record.source: {
            "path": record.path,
            "sha256": record.sha256,
            "source": record.source,
            "url": SOURCE_SPECS[record.source].url,
            "retrieved_from": record.retrieved_from,
        }
        for record in records
    }
    return records, manifest_inputs


def _records_from_saved_manifest(manifest: dict[str, Any]):
    from rise_tvb379.data import SourceManifestRecord

    records = []
    for name, value in sorted(manifest["inputs"].items()):
        records.append(
            SourceManifestRecord(
                source=str(value.get("source", name)),
                path=str(value["path"]),
                sha256=str(value["sha256"]),
                retrieved_from=str(
                    value.get("retrieved_from", "saved verified run input")
                ),
            )
        )
    return tuple(records)


def _execute_experiment(
    *,
    run_dir: Path,
    config,
    source_records,
    run_manifest: dict[str, Any],
    command_started: float,
) -> Path:
    from rise_tvb379.data import (
        load_experiment_data,
        source_manifest_dataframe,
    )
    from rise_tvb379.outputs import (
        build_result_archive,
        write_experiment_metadata,
        write_output_tables,
    )
    from rise_tvb379.parallel import execution_details
    from rise_tvb379.pipeline import (
        build_experiment_metadata,
        run_pipeline,
    )
    from rise_tvb379.plots import create_all_figures

    # TVB configures logging during import. Restore the project handlers so
    # every resumable block and final output remains visible and recorded.
    _configure_logging(run_dir)
    logger = logging.getLogger("rise_tvb379")
    execution = execution_details()
    logger.info(
        "Parallel execution: %d worker processes across %d available CPUs; "
        "one native numerical thread per worker; start method=%s",
        execution["worker_count"],
        execution["available_cpu_count"],
        execution["multiprocessing_start_method"],
    )
    data = load_experiment_data(run_dir / "inputs")
    source_manifest_df = source_manifest_dataframe(source_records)
    products = run_pipeline(
        config,
        data,
        source_manifest_df=source_manifest_df,
        run_dir=run_dir,
        worker_count=int(execution["worker_count"]),
    )
    figures = create_all_figures(
        calibration_df=products.calibration_df,
        main_normalized_df=products.main_normalized_df,
        primary_endpoint_df=products.primary_endpoint_df,
        counterfactual_comparison_df=(
            products.counterfactual_comparison_df
        ),
        matched_null_df=products.matched_null_df,
        main_seed=config.seeds[0],
        sensitivity_endpoint_df=products.sensitivity_endpoint_df,
        shuffle_contrast_df=products.shuffle_contrast_df,
        observed_first_seed_df=products.observed_first_seed_df,
        periodic_probes=config.periodic_probes,
        figure_dir=run_dir / "results" / "figures",
    )
    metadata = build_experiment_metadata(
        config,
        data,
        source_manifest_df,
        provenance={
            "run_directory": str(run_dir),
            "fingerprints": run_manifest["fingerprints"],
            "environment_file": "environment.json",
            "resolved_config_file": "resolved_config.json",
            "inputs_file": "inputs.json",
            "checkpoint_directory": "checkpoints",
            "attempt_environment_directory": "attempts",
            "execution": execution,
        },
    )
    results_dir = run_dir / "results"
    table_paths = write_output_tables(results_dir, products.tables)
    metadata_path = write_experiment_metadata(
        results_dir,
        metadata,
    )
    manifest_df = products.tables["run_manifest"]
    logger.info("Wrote %d CSV tables", len(table_paths))
    logger.info("Wrote %d figures", len(figures))
    logger.info("Metadata: %s", metadata_path)
    logger.info("TVB simulations recorded: %d", len(manifest_df))
    logger.info(
        "Aggregate TVB simulation time, including calibration: %.2f minutes",
        _aggregate_tvb_wall_seconds(
            products.calibration_df,
            manifest_df,
        )
        / 60.0,
    )
    logger.info(
        "Command elapsed time before result-archive snapshot: %s",
        _format_elapsed(time.perf_counter() - command_started),
    )
    expected_archive = (
        run_dir / f"RISE_TVB379_results_{config.mode}.zip"
    )
    logger.info("Writing result archive: %s", expected_archive)
    archive_path = build_result_archive(run_dir, config.mode)
    logger.info("Result archive completed: %s", archive_path)
    return archive_path


def _run_new(
    args: argparse.Namespace,
    environment: dict[str, Any],
    *,
    command_started: float,
) -> int:
    from rise_tvb379.config import (
        config_digest,
        config_to_dict,
        get_experiment_config,
        workload_counts,
    )
    from rise_tvb379.runtime import (
        create_run_directory,
        initialize_run_status,
        run_status_lifecycle,
        write_attempt_environment,
        write_run_manifest,
    )

    config = get_experiment_config(args.mode or "final")
    output_root = (
        args.output_root.expanduser()
        if args.output_root is not None
        else PROJECT_ROOT / "runs"
    )
    data_dir = (
        args.data_dir.expanduser()
        if args.data_dir is not None
        else PROJECT_ROOT / "data"
    )
    run_dir = create_run_directory(
        output_root,
        config.mode,
        config_digest(config),
    )
    _configure_logging(run_dir)
    _prepare_runtime_directories(run_dir)
    initialize_run_status(run_dir, mode=config.mode, status="created")
    logger = logging.getLogger("rise_tvb379")
    counts = workload_counts(config)
    logger.info("Run directory: %s", run_dir)
    logger.info(
        "Starting %s mode: planned maximum %d manifested, %d calibration, "
        "%d total TVB calls (conditional on convergence)",
        config.mode,
        counts.manifest,
        counts.calibration,
        counts.total,
    )

    archive_path: Path | None = None
    with run_status_lifecycle(run_dir, mode=config.mode) as status:
        attempt_environment_path = write_attempt_environment(
            run_dir,
            attempt=int(status["attempt"]),
            environment=environment,
        )
        logger.info(
            "Attempt environment: %s",
            attempt_environment_path,
        )
        source_records, manifest_inputs = _materialize_new_run_inputs(
            run_dir,
            data_dir.resolve(),
            offline=bool(args.offline),
        )
        run_manifest = write_run_manifest(
            run_dir,
            mode=config.mode,
            resolved_config=config_to_dict(config),
            project_root=PROJECT_ROOT,
            inputs=manifest_inputs,
            environment=environment,
        )
        archive_path = _execute_experiment(
            run_dir=run_dir,
            config=config,
            source_records=source_records,
            run_manifest=run_manifest,
            command_started=command_started,
        )
    print(f"Completed run: {run_dir}")
    print(f"Result archive: {archive_path}")
    return 0


def _run_resume(
    args: argparse.Namespace,
    environment: dict[str, Any],
    *,
    command_started: float,
) -> int:
    from rise_tvb379.config import config_to_dict, get_experiment_config
    from rise_tvb379.runtime import (
        run_status_lifecycle,
        validate_resume,
        write_attempt_environment,
    )

    run_dir = args.resume.expanduser().resolve()
    _configure_logging(run_dir)
    _prepare_runtime_directories(run_dir)
    run_manifest = validate_resume(
        run_dir,
        project_root=PROJECT_ROOT,
        environment=environment,
    )
    config = get_experiment_config(str(run_manifest["mode"]))
    if config_to_dict(config) != run_manifest["resolved_config"]:
        raise RuntimeError(
            "resume refused: resolved configuration no longer matches "
            "the project configuration"
        )
    source_records = _records_from_saved_manifest(run_manifest)
    logger = logging.getLogger("rise_tvb379")
    logger.info("Resuming compatible run: %s", run_dir)
    archive_path: Path | None = None
    with run_status_lifecycle(run_dir, mode=config.mode) as status:
        attempt_environment_path = write_attempt_environment(
            run_dir,
            attempt=int(status["attempt"]),
            environment=environment,
        )
        logger.info(
            "Attempt environment: %s",
            attempt_environment_path,
        )
        archive_path = _execute_experiment(
            run_dir=run_dir,
            config=config,
            source_records=source_records,
            run_manifest=run_manifest,
            command_started=command_started,
        )
    print(f"Completed resumed run: {run_dir}")
    print(f"Result archive: {archive_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    command_started = time.perf_counter()
    args = parse_args(argv)
    try:
        from rise_tvb379.parallel import (
            configure_native_thread_limits,
            execution_details,
        )

        configure_native_thread_limits(1)
        environment = _preflight_runtime()
        environment["execution"] = execution_details()
        if args.resume is not None:
            return _run_resume(
                args,
                environment,
                command_started=command_started,
            )
        return _run_new(
            args,
            environment,
            command_started=command_started,
        )
    except KeyboardInterrupt:
        print("Run interrupted; use --resume RUN_DIR to continue.", file=sys.stderr)
        return 130
    except Exception as error:
        logging.getLogger("rise_tvb379").exception("Experiment failed")
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
