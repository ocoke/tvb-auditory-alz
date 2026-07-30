from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import pytest

import main
from rise_tvb379.notebook_runner import (
    CANONICAL_NOTEBOOK,
    CANONICAL_SHA256,
    NotebookValidation,
    Workload,
    run_notebook,
    validate_notebook,
)
from rise_tvb379.run_state import (
    CheckpointDispatcher,
    PROGRESS_LOG_FILENAME,
    RunController,
    RunStateError,
    read_run_status,
)


def test_canonical_notebook_identity_structure_and_scope() -> None:
    validation = validate_notebook()

    assert validation.path == CANONICAL_NOTEBOOK.resolve()
    assert validation.sha256 == CANONICAL_SHA256
    assert len(validation.notebook["cells"]) == 40
    assert len(validation.code_cells) == 18
    assert [index for index, _ in validation.code_cells] == [
        3,
        4,
        *range(6, 37, 2),
    ]

    revision = validation.notebook["metadata"]["rise_revision"]
    assert revision["canonical_notebook"] is True
    assert revision["confirmatory_scope"] == (
        "semantic-versus-episodic proxies only"
    )
    assert revision["speech_confirmatory_analysis_removed"] is True
    assert revision["semantic_expanded_nodes"] == 13
    assert revision["episodic_expanded_nodes"] == 19
    assert validation.csv_output_count == 42
    assert validation.figure_output_count == 6


def test_canonical_code_is_clean_and_outputs_are_cleared() -> None:
    notebook = json.loads(CANONICAL_NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    code = "\n".join(
        "".join(cell["source"]) for cell in code_cells
    ).lower()

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for disallowed in (
        "music_minus_speech",
        "speech_network",
        "checkpoint",
        "resume",
        "todo",
        "placeholder",
        "dummy",
    ):
        assert disallowed not in code


def test_locked_workload_counts() -> None:
    workloads = validate_notebook().workloads

    assert workloads["smoke"] == Workload(
        total_calls=34,
        manifested_calls=32,
        calibration_calls=2,
        main_calls=8,
        integration_step_calls=8,
        local_counterfactual_calls=4,
        parameter_sensitivity_calls=6,
        spatial_shuffle_calls=6,
    )
    assert workloads["pilot"] == Workload(
        total_calls=85,
        manifested_calls=79,
        calibration_calls=6,
        main_calls=24,
        integration_step_calls=8,
        local_counterfactual_calls=8,
        parameter_sensitivity_calls=24,
        spatial_shuffle_calls=15,
    )
    assert workloads["final"] == Workload(
        total_calls=626,
        manifested_calls=614,
        calibration_calls=12,
        main_calls=240,
        integration_step_calls=24,
        local_counterfactual_calls=80,
        parameter_sensitivity_calls=120,
        spatial_shuffle_calls=150,
    )


def test_notebook_sha_is_computed_from_exact_bytes() -> None:
    assert hashlib.sha256(CANONICAL_NOTEBOOK.read_bytes()).hexdigest() == (
        CANONICAL_SHA256
    )


def test_cli_defaults_to_final_and_auto_workers() -> None:
    args = main.parse_args([])

    assert args.mode is None
    assert args.workers is None
    assert args.resume is None


def test_cli_accepts_optional_parallel_worker_count() -> None:
    args = main.parse_args(
        ["--mode", "smoke", "--workers", "2"]
    )

    assert args.mode == "smoke"
    assert args.workers == 2


def test_check_path_never_executes_notebook(monkeypatch, capsys) -> None:
    def fail_if_called(**_kwargs):
        raise AssertionError("static check attempted notebook execution")

    monkeypatch.setattr(main, "run_notebook", fail_if_called)
    assert main.main(["--check"]) == 0
    output = capsys.readouterr().out
    assert "final: 626 total calls (614 manifested)" in output
    assert "no TVB calls ran" in output


def test_direct_runner_shares_namespace_between_cells(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "code", "source": ["values = [2, 3]\n"]},
            {
                "cell_type": "code",
                "source": ["result = sum(value * value for value in values)\n"],
            },
        ]
    }
    validation = NotebookValidation(
        path=tmp_path / "synthetic.ipynb",
        sha256="synthetic",
        notebook=notebook,
        code_cells=(
            (0, "values = [2, 3]\n"),
            (1, "result = sum(value * value for value in values)\n"),
        ),
        workloads={},
        csv_output_count=0,
        figure_output_count=0,
    )

    namespace = run_notebook(validation=validation)

    assert namespace["result"] == 13


class _PickleJoblib:
    @staticmethod
    def hash(value, hash_name="sha1") -> str:
        digest = hashlib.new(hash_name)
        digest.update(pickle.dumps(value))
        return digest.hexdigest()

    @staticmethod
    def dump(value, path, compress=0):
        del compress
        with Path(path).open("wb") as stream:
            pickle.dump(value, stream)

    @staticmethod
    def load(path):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)


def test_interrupted_dispatch_resumes_only_unfinished_work(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results_test"
    environment = {"python": "test", "packages": {"joblib": "test"}}
    controller = RunController.create(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="notebook",
        code_sha256="code",
        environment=environment,
        planned_total_tvb_calls=3,
        worker_processes=1,
    )
    namespace = {
        "joblib": _PickleJoblib,
        "PARALLEL_WORKERS": 1,
    }
    executed: list[int] = []
    should_fail = {"value": True}

    def worker(job, offset):
        ordinal = job["ordinal"]
        executed.append(ordinal)
        if ordinal == 1 and should_fail["value"]:
            should_fail["value"] = False
            raise RuntimeError("synthetic interruption")
        return {"ordinal": ordinal, "value": ordinal + offset}

    jobs = [
        {"ordinal": ordinal, "probes": ()}
        for ordinal in range(3)
    ]
    dispatcher = CheckpointDispatcher(namespace, controller)
    with pytest.raises(RuntimeError, match="synthetic interruption") as error:
        dispatcher(worker, jobs, (10,), "synthetic blocks")
    controller.mark_failed(error.value)
    assert read_run_status(run_dir)["completed_tvb_calls"] == 1

    resumed = RunController.resume(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="notebook",
        code_sha256="code",
        environment=environment,
        planned_total_tvb_calls=3,
        worker_processes=1,
    )
    resumed_dispatcher = CheckpointDispatcher(namespace, resumed)
    outcomes = resumed_dispatcher(
        worker,
        jobs,
        (10,),
        "synthetic blocks",
    )
    resumed.mark_completed()

    assert executed == [0, 1, 1, 2]
    assert outcomes == [
        {"ordinal": 0, "value": 10},
        {"ordinal": 1, "value": 11},
        {"ordinal": 2, "value": 12},
    ]
    status = read_run_status(run_dir)
    assert status["state"] == "completed"
    assert status["completed_tvb_calls"] == 3
    assert status["restored_tvb_calls_this_attempt"] == 1
    assert "restored work unit" in (
        run_dir / PROGRESS_LOG_FILENAME
    ).read_text(encoding="utf-8")


def test_resume_refuses_environment_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "results_test"
    controller = RunController.create(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="notebook",
        code_sha256="code",
        environment={"python": "3.12.8"},
        planned_total_tvb_calls=3,
        worker_processes=1,
    )
    controller.mark_failed(RuntimeError("stop"))

    with pytest.raises(
        RunStateError,
        match="Python or dependency versions",
    ):
        RunController.resume(
            run_dir=run_dir,
            mode="smoke",
            notebook_sha256="notebook",
            code_sha256="code",
            environment={"python": "3.12.9"},
            planned_total_tvb_calls=3,
            worker_processes=1,
        )


def test_damaged_checkpoint_is_recomputed(tmp_path: Path) -> None:
    run_dir = tmp_path / "results_test"
    environment = {"python": "test"}
    controller = RunController.create(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="notebook",
        code_sha256="code",
        environment=environment,
        planned_total_tvb_calls=2,
        worker_processes=1,
    )
    namespace = {
        "joblib": _PickleJoblib,
        "PARALLEL_WORKERS": 1,
    }
    first_executed: list[int] = []
    execution_target = {"items": first_executed}

    def worker(job):
        execution_target["items"].append(job["ordinal"])
        return {"ordinal": job["ordinal"]}

    jobs = [
        {"ordinal": ordinal, "probes": ()}
        for ordinal in range(2)
    ]
    CheckpointDispatcher(namespace, controller)(
        worker,
        jobs,
        (),
        "synthetic blocks",
    )
    controller.mark_failed(RuntimeError("fail after stage"))
    damaged = sorted(controller.checkpoint_root.rglob("*.joblib"))[0]
    damaged.write_bytes(b"damaged")

    resumed = RunController.resume(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="notebook",
        code_sha256="code",
        environment=environment,
        planned_total_tvb_calls=2,
        worker_processes=1,
    )
    second_executed: list[int] = []
    execution_target["items"] = second_executed

    outcomes = CheckpointDispatcher(namespace, resumed)(
        worker,
        jobs,
        (),
        "synthetic blocks",
    )
    resumed.mark_completed()

    assert first_executed == [0, 1]
    assert second_executed == [0]
    assert outcomes == [{"ordinal": 0}, {"ordinal": 1}]
    assert "ignoring incomplete or damaged checkpoint" in (
        run_dir / PROGRESS_LOG_FILENAME
    ).read_text(encoding="utf-8")


def test_status_cli_reads_status_without_starting_run(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = tmp_path / "results_test"
    RunController.create(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="notebook",
        code_sha256="code",
        environment={"python": "test"},
        planned_total_tvb_calls=34,
        worker_processes=1,
    )
    capsys.readouterr()

    assert main.main(["--status", str(run_dir)]) == 0

    output = capsys.readouterr().out
    assert "State: running" in output
    assert "TVB progress: 0/34 calls (0.0%)" in output
