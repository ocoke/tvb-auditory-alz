"""Final table, metadata, and provenance bundle writers."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_TABLE_FILENAMES: dict[str, str] = {
    "source_manifest": "source_manifest.csv",
    "data_quality": "data_quality_checks.csv",
    "roi_definitions": "roi_definitions.csv",
    "roi_pathology": "roi_pathology_values.csv",
    "pathology_summary": "pathology_summary.csv",
    "calibration": "baseline_coupling_calibration.csv",
    "main_node": "main_node_metrics.csv",
    "main_network": "main_network_metrics.csv",
    "main_normalized": "main_network_metrics_normalized.csv",
    "main_contrasts": "main_music_minus_speech_contrasts.csv",
    "main_stage_summary": "main_stage_summary.csv",
    "local_fixed_node": "local_fixed_node_metrics.csv",
    "local_fixed_network": "local_fixed_network_metrics.csv",
    "local_fixed_contrasts": "local_fixed_contrasts.csv",
    "matched_sets": "matched_control_sets.csv",
    "matched_null": "matched_control_null_metrics.csv",
    "matched_null_summary": "matched_control_null_summary.csv",
    "sensitivity_network": "sensitivity_network_metrics.csv",
    "sensitivity_contrasts": "sensitivity_contrasts.csv",
    "shuffle_network": "spatial_shuffle_network_metrics.csv",
    "shuffle_contrasts": "spatial_shuffle_contrasts.csv",
    "dt_convergence": "integration_step_check.csv",
    "run_manifest": "run_manifest.csv",
}


def _atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_output_tables(
    results_dir: Path,
    tables: Mapping[str, pd.DataFrame],
) -> list[Path]:
    """Write the exact 23 reader-facing CSV outputs."""

    missing = sorted(set(OUTPUT_TABLE_FILENAMES) - set(tables))
    extra = sorted(set(tables) - set(OUTPUT_TABLE_FILENAMES))
    if missing or extra:
        raise ValueError(
            f"Output table mismatch; missing={missing}, extra={extra}"
        )
    written: list[Path] = []
    for key, filename in OUTPUT_TABLE_FILENAMES.items():
        path = results_dir / filename
        _atomic_dataframe(path, tables[key])
        written.append(path)
    return written


def write_experiment_metadata(
    results_dir: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Write the notebook-compatible metadata document atomically."""

    path = results_dir / "experiment_metadata.json"
    _atomic_json(path, metadata)
    return path


def build_result_archive(run_dir: Path, mode: str) -> Path:
    """Archive results and provenance while excluding internal checkpoints."""

    destination = run_dir / f"RISE_TVB379_results_{mode}.zip"
    temporary = destination.with_suffix(".zip.tmp")
    include_roots = [
        run_dir / "results",
        run_dir / "inputs",
        run_dir / "attempts",
    ]
    include_files = [
        run_dir / "resolved_config.json",
        run_dir / "environment.json",
        run_dir / "inputs.json",
        run_dir / "run_manifest.json",
        run_dir / "run.log",
    ]
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for root in include_roots:
                if not root.exists():
                    continue
                for path in sorted(
                    item for item in root.rglob("*") if item.is_file()
                ):
                    archive.write(path, path.relative_to(run_dir))
            for path in include_files:
                if path.is_file():
                    archive.write(path, path.relative_to(run_dir))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_final_bundle(
    run_dir: Path,
    *,
    mode: str,
    tables: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
) -> tuple[list[Path], Path, Path]:
    """Write all reader outputs and produce the final ZIP archive."""

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    table_paths = write_output_tables(results_dir, tables)
    metadata_path = write_experiment_metadata(results_dir, metadata)
    archive_path = build_result_archive(run_dir, mode)
    return table_paths, metadata_path, archive_path
