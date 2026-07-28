from __future__ import annotations

from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import pandas as pd
import pytest

from rise_tvb379.analysis import check_integration_step
from rise_tvb379.checkpoints import (
    COMPLETE_MARKER_NAME,
    atomic_write_json,
    atomic_write_text,
    list_completed_blocks,
    read_completed_block,
    write_completed_block,
)
from rise_tvb379.runtime import (
    capture_input_manifest,
    create_run_directory,
    fingerprint_code,
    fingerprint_config,
    fingerprint_environment,
    fingerprint_inputs,
    initialize_run_status,
    load_run_status,
    mark_run_completed,
    run_status_lifecycle,
    update_run_status,
    validate_resume,
    write_attempt_environment,
    write_run_manifest,
)


class TinyFrame:
    """A dependency-free stand-in for the pandas method checkpointing uses."""

    def __init__(self, csv_text: str) -> None:
        self.csv_text = csv_text

    def to_csv(self, path: Path, **options: object) -> None:
        assert options["index"] is False
        assert options["encoding"] == "utf-8"
        Path(path).write_text(self.csv_text, encoding="utf-8")


class FailingFrame:
    def to_csv(self, path: Path, **options: object) -> None:
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("simulated interruption")


def fixed_environment(version: str = "3.12.0") -> dict[str, object]:
    return {
        "python_version": version,
        "python_full_version": f"{version} test build",
        "python_implementation": "CPython",
        "python_executable": "/example/python",
        "platform": "test-platform",
        "machine": "test-machine",
        "processor": "test-processor",
        "packages": {
            "matplotlib": "3.11.1",
            "numpy": "2.0.2",
            "pandas": "2.2.2",
            "scipy": "1.16.3",
            "tvb-data": "3.0.0",
            "tvb-library": "2.10.0",
        },
    }


def make_project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    package = project / "rise_tvb379"
    package.mkdir(parents=True)
    (project / "main.py").write_text("print('experiment')\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    input_path = project / "input.bin"
    input_path.write_bytes(b"verified input")
    return project, input_path


def make_resumable_run(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    project, input_path = make_project(tmp_path)
    config = {"mode": "smoke", "seeds": [11], "nested": {"b": 2, "a": 1}}
    run_dir = create_run_directory(
        tmp_path / "runs",
        "smoke",
        fingerprint_config(config),
        now=datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),
    )
    environment = fixed_environment()
    write_run_manifest(
        run_dir,
        mode="smoke",
        resolved_config=config,
        project_root=project,
        inputs={"connectivity": input_path},
        environment=environment,
        created_at=datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),
    )
    initialize_run_status(
        run_dir,
        mode="smoke",
        now=datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),
    )
    return run_dir, project, input_path, environment


def test_atomic_text_and_json_replace_without_leaving_partial_files(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "state.json"
    atomic_write_text(destination, "old")
    atomic_write_json(destination, {"z": 1, "a": [2, 3]})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "a": [2, 3],
        "z": 1,
    }
    assert list(destination.parent.glob("*.partial")) == []
    assert list(destination.parent.glob(".*.partial")) == []


def test_completed_blocks_are_ordered_verified_and_readable(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    write_completed_block(
        checkpoint_root,
        "main",
        "seed-02",
        {"responses": TinyFrame("seed,value\n2,4.5\n")},
        {"seed": 2, "condition": "high"},
        completed_at="2026-07-28T18:00:00Z",
    )
    write_completed_block(
        checkpoint_root,
        "main",
        "seed-01",
        {
            "responses": TinyFrame("seed,value\n1,3.5\n"),
            "timeseries": TinyFrame("time,value\n0,0.1\n"),
        },
        {"seed": 1},
    )

    assert list_completed_blocks(checkpoint_root, "main") == [
        "seed-01",
        "seed-02",
    ]
    loaded = read_completed_block(
        checkpoint_root,
        "main",
        "seed-01",
        dataframe_reader=lambda path: path.read_text(encoding="utf-8"),
    )
    assert loaded.metadata == {"seed": 1}
    assert loaded.frames["responses"] == "seed,value\n1,3.5\n"
    assert loaded.frames["timeseries"] == "time,value\n0,0.1\n"
    assert loaded.marker["schema_version"] == 1


def test_incomplete_or_corrupt_blocks_are_ignored(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    partial_path = checkpoint_root / "spatial-shuffles" / "shuffle-001"
    partial_path.mkdir(parents=True)
    (partial_path / "responses.csv").write_text("x\n1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete or failed integrity"):
        read_completed_block(
            checkpoint_root,
            "spatial-shuffles",
            "shuffle-001",
            dataframe_reader=lambda path: path.read_text(encoding="utf-8"),
        )
    assert list_completed_blocks(checkpoint_root, "spatial-shuffles") == []

    write_completed_block(
        checkpoint_root,
        "spatial-shuffles",
        "shuffle-002",
        {"responses": TinyFrame("x\n2\n")},
        {"shuffle": 2},
    )
    (checkpoint_root / "spatial-shuffles" / "shuffle-002" / "responses.csv").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    assert list_completed_blocks(checkpoint_root, "spatial-shuffles") == []


def test_marker_is_absent_when_frame_write_is_interrupted(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_completed_block(
            checkpoint_root,
            "sensitivity",
            "gain-01",
            {"results": FailingFrame()},
            {"gain": 0.1},
        )

    block_path = checkpoint_root / "sensitivity" / "gain-01"
    assert not (block_path / COMPLETE_MARKER_NAME).exists()
    assert list_completed_blocks(checkpoint_root, "sensitivity") == []


def test_verified_existing_block_is_immutable_by_default(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    write_completed_block(
        checkpoint_root,
        "calibration",
        "coupling-001",
        {"metrics": TinyFrame("value\n1\n")},
        {"coupling": 0.01},
    )
    write_completed_block(
        checkpoint_root,
        "calibration",
        "coupling-001",
        {"metrics": FailingFrame()},
        {"coupling": 9.99},
    )
    loaded = read_completed_block(
        checkpoint_root,
        "calibration",
        "coupling-001",
        dataframe_reader=lambda path: path.read_text(encoding="utf-8"),
    )
    assert loaded.frames["metrics"] == "value\n1\n"
    assert loaded.metadata == {"coupling": 0.01}


def test_run_directories_are_unique_and_have_expected_layout(
    tmp_path: Path,
) -> None:
    instant = datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc)
    first = create_run_directory(tmp_path, "final", "abcdef012345", now=instant)
    second = create_run_directory(tmp_path, "final", "abcdef012345", now=instant)

    assert first.name == "20260728T180000Z_final_abcdef01"
    assert second.name == "20260728T180000Z_final_abcdef01_01"
    for run_dir in (first, second):
        assert {child.name for child in run_dir.iterdir() if child.is_dir()} == {
            "inputs",
            "checkpoints",
            "results",
        }
        assert (run_dir / "run.log").is_file()


def test_fingerprints_are_deterministic_and_sensitive_to_content(
    tmp_path: Path,
) -> None:
    project, input_path = make_project(tmp_path)
    assert fingerprint_config({"b": 2, "a": [1]}) == fingerprint_config(
        {"a": [1], "b": 2}
    )

    first_code = fingerprint_code(project)
    (project / "main.py").write_text("print('changed')\n", encoding="utf-8")
    assert fingerprint_code(project) != first_code

    manifest = capture_input_manifest({"input": input_path})
    assert fingerprint_inputs(manifest) == fingerprint_inputs(
        {"input": dict(reversed(list(manifest["input"].items())))}
    )
    first_input = fingerprint_inputs(manifest)
    input_path.write_bytes(b"different bytes")
    assert (
        fingerprint_inputs(capture_input_manifest({"input": input_path})) != first_input
    )

    environment = fixed_environment()
    moved_environment = dict(environment)
    moved_environment["python_executable"] = "/another/location/python"
    assert fingerprint_environment(environment) == fingerprint_environment(
        moved_environment
    )
    changed_environment = dict(environment)
    changed_environment["packages"] = dict(environment["packages"])  # type: ignore[arg-type]
    changed_environment["packages"]["numpy"] = "2.0.3"  # type: ignore[index]
    assert fingerprint_environment(environment) != fingerprint_environment(
        changed_environment
    )


def test_status_lifecycle_records_failure_resume_and_completion(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initialize_run_status(
        run_dir,
        mode="smoke",
        now=datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),
    )
    failed = update_run_status(
        run_dir,
        "failed",
        error=ValueError("bad block"),
        now=datetime(2026, 7, 28, 18, 1, tzinfo=timezone.utc),
    )
    assert failed["error"] == {"type": "ValueError", "message": "bad block"}
    assert failed["exit_time"] == "2026-07-28T18:01:00Z"

    resumed = update_run_status(
        run_dir,
        "running",
        now=datetime(2026, 7, 28, 18, 2, tzinfo=timezone.utc),
    )
    assert resumed["attempt"] == 2
    complete = update_run_status(
        run_dir,
        "completed",
        now=datetime(2026, 7, 28, 18, 3, tzinfo=timezone.utc),
    )
    assert complete["completed_at"] == "2026-07-28T18:03:00Z"
    with pytest.raises(ValueError, match="completed -> running"):
        update_run_status(run_dir, "running")


def test_status_context_marks_interruption_then_can_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    initialize_run_status(run_dir, mode="pilot", status="created")

    with pytest.raises(KeyboardInterrupt):
        with run_status_lifecycle(run_dir, mode="pilot"):
            raise KeyboardInterrupt("stop")
    interrupted = load_run_status(run_dir)
    assert interrupted["status"] == "interrupted"
    assert interrupted["error"] == {
        "type": "KeyboardInterrupt",
        "message": "stop",
    }

    with run_status_lifecycle(run_dir, mode="pilot"):
        pass
    completed = load_run_status(run_dir)
    assert completed["status"] == "completed"
    assert completed["attempt"] == 2


def test_manifest_records_exact_paths_and_compatible_resume(tmp_path: Path) -> None:
    run_dir, project, input_path, environment = make_resumable_run(tmp_path)
    manifest = validate_resume(run_dir, environment=environment)

    assert manifest["paths"]["run_dir"] == str(run_dir.resolve())
    assert manifest["paths"]["project_root"] == str(project.resolve())
    assert manifest["paths"]["manifest"] == str(
        (run_dir / "run_manifest.json").resolve()
    )
    assert manifest["paths"]["status"] == str((run_dir / "run_status.json").resolve())
    assert manifest["paths"]["log"] == str((run_dir / "run.log").resolve())
    assert manifest["paths"]["attempt_environment_directory"] == str(
        (run_dir / "attempts").resolve()
    )
    assert manifest["paths"]["input_files"]["connectivity"] == str(input_path.resolve())
    assert set(manifest["fingerprints"]) == {
        "code",
        "config",
        "environment",
        "inputs",
    }


def test_manifest_persists_parallel_execution_provenance(
    tmp_path: Path,
) -> None:
    project, input_path = make_project(tmp_path)
    run_dir = create_run_directory(
        tmp_path / "runs",
        "smoke",
        "abcdef012345",
    )
    environment = fixed_environment()
    environment["execution"] = {
        "multiprocessing_start_method": "spawn",
        "available_cpu_count": 8,
        "worker_count": 8,
        "native_threads_per_worker": 1,
        "native_thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    }

    manifest = write_run_manifest(
        run_dir,
        mode="smoke",
        resolved_config={"mode": "smoke"},
        project_root=project,
        inputs={"connectivity": input_path},
        environment=environment,
    )
    saved_environment = json.loads(
        (run_dir / "environment.json").read_text(encoding="utf-8")
    )

    assert saved_environment["execution"] == environment["execution"]
    assert manifest["environment"]["execution"] == environment["execution"]
    assert (
        manifest["fingerprints"]["environment"]
        == fingerprint_environment(saved_environment)
    )
    initial_attempt = write_attempt_environment(
        run_dir,
        attempt=1,
        environment=environment,
        recorded_at=datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc),
    )
    resumed_environment = dict(environment)
    resumed_environment["platform"] = "different-compatible-host"
    resumed_environment["execution"] = {
        **environment["execution"],  # type: ignore[arg-type]
        "available_cpu_count": 4,
        "worker_count": 4,
    }
    resumed_attempt = write_attempt_environment(
        run_dir,
        attempt=2,
        environment=resumed_environment,
        recorded_at=datetime(2026, 7, 28, 19, 0, tzinfo=timezone.utc),
    )
    assert json.loads(initial_attempt.read_text(encoding="utf-8"))[
        "environment"
    ]["execution"]["worker_count"] == 8
    assert json.loads(resumed_attempt.read_text(encoding="utf-8"))[
        "environment"
    ]["execution"]["worker_count"] == 4
    with pytest.raises(ValueError, match="positive integer"):
        write_attempt_environment(
            run_dir,
            attempt=0,
            environment=environment,
        )


def test_real_convergence_failure_is_recorded_and_resume_compatible(
    tmp_path: Path,
) -> None:
    run_dir, project, _, environment = make_resumable_run(tmp_path)
    checkpoint_root = run_dir / "checkpoints"
    write_completed_block(
        checkpoint_root,
        "dt_reference",
        "severity_1.0_seed_11",
        {"network": TinyFrame("severity,probe,network\n1.0,5Hz,speech\n")},
        {"dt_ms": 0.5},
    )
    main_rows: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    for severity, probe, network in product(
        (0.0, 1.0),
        ("2Hz", "5Hz"),
        ("music", "speech"),
    ):
        reference_transfer = 1.0
        main_transfer = 0.99
        if (severity, probe, network) == (1.0, "5Hz", "speech"):
            main_transfer = 1.05
        main_rows.append(
            {
                "seed": 11,
                "severity": severity,
                "probe": probe,
                "network": network,
                "transfer": main_transfer,
                "median_target_fit_r_squared": 0.8,
            }
        )
        reference_rows.append(
            {
                "severity": severity,
                "probe": probe,
                "network": network,
                "transfer": reference_transfer,
                "median_target_fit_r_squared": 0.79,
            }
        )

    with pytest.raises(RuntimeError, match="5% convergence check"):
        with run_status_lifecycle(run_dir, mode="smoke"):
            check_integration_step(
                main_network_df=pd.DataFrame(main_rows),
                reference_network_df=pd.DataFrame(reference_rows),
                main_seed=11,
            )

    failed = load_run_status(run_dir)
    assert failed["status"] == "failed"
    assert failed["error"]["type"] == "RuntimeError"
    assert failed["exit_time"] is not None
    assert not list(run_dir.glob("RISE_TVB379_results_*.zip"))
    saved = read_completed_block(
        checkpoint_root,
        "dt_reference",
        "severity_1.0_seed_11",
        dataframe_reader=lambda path: path.read_text(encoding="utf-8"),
    )
    assert "speech" in saved.frames["network"]
    compatible_manifest = validate_resume(
        run_dir,
        project_root=project,
        environment=environment,
    )
    assert compatible_manifest["mode"] == "smoke"


@pytest.mark.parametrize(
    ("mutation", "mismatch_name"),
    [
        ("code", "code fingerprint mismatch"),
        ("config", "config fingerprint mismatch"),
        ("environment", "environment fingerprint mismatch"),
        ("inputs", "input fingerprint mismatch"),
    ],
)
def test_resume_refuses_each_incompatible_fingerprint(
    tmp_path: Path,
    mutation: str,
    mismatch_name: str,
) -> None:
    run_dir, project, input_path, environment = make_resumable_run(tmp_path)
    validation_options: dict[str, object] = {"environment": environment}
    if mutation == "code":
        (project / "main.py").write_text("print('modified')\n", encoding="utf-8")
    elif mutation == "config":
        config_path = run_dir / "resolved_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["seeds"] = [12]
        atomic_write_json(config_path, config)
    elif mutation == "environment":
        incompatible_environment = fixed_environment("3.12.1")
        validation_options["environment"] = incompatible_environment
    elif mutation == "inputs":
        input_path.write_bytes(b"modified input")

    with pytest.raises((RuntimeError, ValueError), match=mismatch_name):
        validate_resume(run_dir, **validation_options)


def test_resume_refuses_completed_run(tmp_path: Path) -> None:
    run_dir, _, _, environment = make_resumable_run(tmp_path)
    mark_run_completed(run_dir)
    assert load_run_status(run_dir)["status"] == "completed"

    with pytest.raises(RuntimeError, match="already complete"):
        validate_resume(run_dir, environment=environment)
