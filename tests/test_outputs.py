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
    checkpoints_dir = run_dir / "checkpoints"
    inputs_dir.mkdir(parents=True)
    checkpoints_dir.mkdir()
    (inputs_dir / "input.txt").write_text("verified input")
    (checkpoints_dir / "private.txt").write_text("checkpoint")
    for filename in (
        "resolved_config.json",
        "environment.json",
        "inputs.json",
        "run_manifest.json",
        "run.log",
    ):
        (run_dir / filename).write_text("{}\n")

    tables = {
        name: pd.DataFrame([{"table": name, "value": 1}])
        for name in OUTPUT_TABLE_FILENAMES
    }
    paths = write_output_tables(results_dir, tables)
    metadata_path = write_experiment_metadata(
        results_dir, {"run_mode": "smoke"}
    )
    archive_path = build_result_archive(run_dir, "smoke")

    assert len(paths) == 23
    assert metadata_path.is_file()
    assert json.loads(metadata_path.read_text())["run_mode"] == "smoke"
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "results/experiment_metadata.json" in names
    assert "inputs/input.txt" in names
    assert "resolved_config.json" in names
    assert "inputs.json" in names
    assert not any(name.startswith("checkpoints/") for name in names)
