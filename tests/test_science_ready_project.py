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
    COMPATIBLE_PREDECESSOR_NOTEBOOK_SHA256S,
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
    amendment = validation.notebook["metadata"][
        "rise_protocol_amendment"
    ]
    assert amendment["date"] == "2026-07-30"
    assert amendment["name"] == "integration-step interaction gate"
    assert amendment["final_reference_seed_count"] == 20
    assert amendment["planned_final_tvb_calls"] == 762
    assert validation.csv_output_count == 48
    assert validation.figure_output_count == 6
    notebook_code = "\n".join(
        source for _, source in validation.code_cells
    )
    for diagnostic_output in (
        "regional_features.csv",
        "main_parcel_trace_manifest.csv",
        "integration_step_outcome_eligibility.csv",
        "integration_step_interaction_seed_diagnostics.csv",
        "integration_step_a1_snr_seed_diagnostics.csv",
        "integration_step_raw_metric_seed_diagnostics.csv",
    ):
        assert diagnostic_output in notebook_code
    assert "if len(TRACE_REGION_LABELS) != 34" in notebook_code
    assert "stimulated_psp=selected_stimulated" in notebook_code
    assert "control_psp=selected_control" in notebook_code
    assert "evoked_psp=selected_evoked" in notebook_code


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
        integration_step_blocks=2,
        integration_step_calls=8,
        local_counterfactual_calls=4,
        parameter_sensitivity_calls=6,
        spatial_shuffle_calls=6,
        raw_trace_shards=6,
    )
    assert workloads["pilot"] == Workload(
        total_calls=85,
        manifested_calls=79,
        calibration_calls=6,
        main_calls=24,
        integration_step_blocks=2,
        integration_step_calls=8,
        local_counterfactual_calls=8,
        parameter_sensitivity_calls=24,
        spatial_shuffle_calls=15,
        raw_trace_shards=18,
    )
    assert workloads["final"] == Workload(
        total_calls=762,
        manifested_calls=750,
        calibration_calls=12,
        main_calls=240,
        integration_step_blocks=40,
        integration_step_calls=160,
        local_counterfactual_calls=80,
        parameter_sensitivity_calls=120,
        spatial_shuffle_calls=150,
        raw_trace_shards=180,
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
    assert "final: 762 total calls (750 manifested)" in output
    assert "integration step 40 work units / 160 calls" in output
    assert "raw-trace shards 180" in output
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
        planned_integration_step_work_units=0,
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
        planned_integration_step_work_units=0,
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
        planned_integration_step_work_units=0,
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
            planned_integration_step_work_units=0,
            worker_processes=1,
        )


def test_dtgate_migration_recomputes_main_and_restores_other_scopes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results_migration"
    environment = {"python": "test"}
    predecessor_sha = next(
        iter(COMPATIBLE_PREDECESSOR_NOTEBOOK_SHA256S)
    )
    old_controller = RunController.create(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256=predecessor_sha,
        code_sha256="old-code",
        environment=environment,
        planned_total_tvb_calls=2,
        planned_integration_step_work_units=0,
        worker_processes=1,
    )
    namespace = {
        "joblib": _PickleJoblib,
        "PARALLEL_WORKERS": 1,
    }
    executed: list[str] = []

    def worker(job):
        executed.append(str(job["scope"]))
        return {"scope": str(job["scope"])}

    old_dispatcher = CheckpointDispatcher(namespace, old_controller)
    old_dispatcher(
        worker,
        [{"scope": "main_full_field", "probes": ()}],
        (),
        "main_full_field condition-seed blocks",
    )
    old_dispatcher(
        worker,
        [{"scope": "local_fixed", "probes": ()}],
        (),
        "local_fixed condition-seed blocks",
    )
    old_controller.mark_completed()

    resumed = RunController.resume(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="raw-trace-notebook",
        code_sha256="new-code",
        environment=environment,
        planned_total_tvb_calls=2,
        planned_integration_step_work_units=0,
        worker_processes=1,
        planned_raw_trace_shards=1,
        compatible_predecessor_notebook_sha256s={predecessor_sha},
    )
    migrated_status = read_run_status(run_dir)
    assert migrated_status["completed_tvb_calls"] == 1
    assert migrated_status["state"] == "running"
    assert migrated_status["checkpoint_migrations"][0][
        "invalidated_main_tvb_calls"
    ] == 1

    new_dispatcher = CheckpointDispatcher(namespace, resumed)
    new_dispatcher(
        worker,
        [
            {
                "scope": "main_full_field",
                "probes": (),
                "export_parcel_traces": True,
            }
        ],
        (),
        "main_full_field raw-trace-v1 condition-seed blocks",
    )
    new_dispatcher(
        worker,
        [
            {
                "scope": "local_fixed",
                "probes": (),
                "export_parcel_traces": False,
            }
        ],
        (),
        "local_fixed condition-seed blocks",
    )
    resumed.mark_completed()

    assert executed == ["main_full_field", "local_fixed", "main_full_field"]
    final_status = read_run_status(run_dir)
    assert final_status["state"] == "completed"
    assert final_status["restored_tvb_calls_this_attempt"] == 1
    assert final_status["executed_tvb_calls_this_attempt"] == 1
    log = (run_dir / PROGRESS_LOG_FILENAME).read_text(encoding="utf-8")
    assert "will recompute under raw-trace-v1" in log


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
        planned_integration_step_work_units=0,
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
        planned_integration_step_work_units=0,
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


def test_dt_gate_checkpoint_accounting_is_40_units_and_160_calls(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "results_dt_gate"
    controller = RunController.create(
        run_dir=run_dir,
        mode="final",
        notebook_sha256="notebook",
        code_sha256="code",
        environment={"python": "test"},
        planned_total_tvb_calls=762,
        planned_integration_step_work_units=40,
        worker_processes=1,
    )
    namespace = {
        "joblib": _PickleJoblib,
        "PARALLEL_WORKERS": 1,
    }
    jobs = [
        {
            "ordinal": ordinal,
            "probes": ("pulse", "2Hz", "5Hz"),
        }
        for ordinal in range(40)
    ]

    outcomes = CheckpointDispatcher(namespace, controller)(
        lambda job: {"ordinal": job["ordinal"]},
        jobs,
        (),
        "dt_reference_0.25ms condition-seed blocks",
    )

    assert len(outcomes) == 40
    status = read_run_status(run_dir)
    assert status["planned_total_tvb_calls"] == 762
    assert status["planned_integration_step_work_units"] == 40
    assert status["completed_tvb_calls"] == 160


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
        planned_integration_step_work_units=2,
        worker_processes=1,
    )
    capsys.readouterr()

    assert main.main(["--status", str(run_dir)]) == 0

    output = capsys.readouterr().out
    assert "State: running" in output
    assert "TVB progress: 0/34 calls (0.0%)" in output
    assert "Integration-step plan: 2 work units" in output
    assert "Raw-trace plan: 0 shards" in output
