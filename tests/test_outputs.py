from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from rise_tvb379.outputs import (
    OUTPUT_TABLE_FILENAMES,
    build_result_archive,
    write_experiment_metadata,
    write_output_tables,
)


def test_writes_exact_table_contract_and_provenance_archive(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    results_dir = run_dir / "results"
    inputs_dir = run_dir / "inputs"
    attempts_dir = run_dir / "attempts"
    checkpoints_dir = run_dir / "checkpoints"
    inputs_dir.mkdir(parents=True)
    attempts_dir.mkdir()
    checkpoints_dir.mkdir()
    (inputs_dir / "input.txt").write_text("verified input")
    (checkpoints_dir / "private.txt").write_text("checkpoint")
    execution = {
        "multiprocessing_start_method": "spawn",
        "available_cpu_count": 8,
        "worker_count": 8,
        "native_threads_per_worker": 1,
    }
    (run_dir / "environment.json").write_text(
        json.dumps({"execution": execution}) + "\n",
        encoding="utf-8",
    )
    for filename in (
        "resolved_config.json",
        "inputs.json",
        "run_manifest.json",
    ):
        (run_dir / filename).write_text("{}\n")
    (run_dir / "run.log").write_text(
        "Command elapsed time before result-archive snapshot: 00:01:23\n",
        encoding="utf-8",
    )
    (attempts_dir / "attempt_001_environment.json").write_text(
        json.dumps({"attempt": 1, "environment": {"execution": execution}})
        + "\n",
        encoding="utf-8",
    )

    tables = {
        name: pd.DataFrame([{"table": name, "value": 1}])
        for name in OUTPUT_TABLE_FILENAMES
    }
    paths = write_output_tables(results_dir, tables)
    metadata_path = write_experiment_metadata(
        results_dir, {"run_mode": "smoke"}
    )
    archive_path = build_result_archive(run_dir, "smoke")

    assert len(paths) == 31
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text())["run_mode"] == "smoke"
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        archived_environment = json.loads(
            archive.read("environment.json").decode("utf-8")
        )
        archived_log = archive.read("run.log").decode("utf-8")
    assert "results/experiment_metadata.json" in names
    assert "inputs/input.txt" in names
    assert "attempts/attempt_001_environment.json" in names
    assert "resolved_config.json" in names
    assert "inputs.json" in names
    assert archived_environment["execution"] == execution
    assert "before result-archive snapshot: 00:01:23" in archived_log
    assert not any(name.startswith("checkpoints/") for name in names)
