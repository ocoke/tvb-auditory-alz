from __future__ import annotations

import hashlib
from pathlib import Path
import pickle

import main_extended
from rise_tvb379.extended_notebook_runner import (
    EXTENDED_NOTEBOOK,
    EXTENDED_SHA256,
    EXPECTED_CODE_CELL_INDICES,
    ExtendedCheckpointDispatcher,
    ExtendedRunController,
    ExtendedWorkload,
    validate_extended_notebook,
)
from rise_tvb379.run_state import read_run_status


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


def test_extended_notebook_identity_contract_and_primary_cells() -> None:
    validation = validate_extended_notebook()

    assert validation.path == EXTENDED_NOTEBOOK.resolve()
    assert validation.sha256 == EXTENDED_SHA256
    assert hashlib.sha256(EXTENDED_NOTEBOOK.read_bytes()).hexdigest() == (
        EXTENDED_SHA256
    )
    assert len(validation.notebook["cells"]) == 43
    assert len(validation.code_cells) == 20
    assert tuple(index for index, _ in validation.code_cells) == (
        EXPECTED_CODE_CELL_INDICES
    )
    assert validation.csv_output_count == 68
    assert validation.figure_output_count == 8

    artifact = validation.notebook["metadata"]["rise"]
    assert artifact == {
        "artifact_version": (
            "semantic_episodic_v6_late_window_followup_2026-08-02"
        ),
        "fabricated_data_used": False,
        "followup_analysis_version": "late-window-dc-matched-v1",
        "primary_method_changed": False,
    }


def test_extended_locked_workloads() -> None:
    workloads = validate_extended_notebook().workloads

    assert workloads["smoke"] == ExtendedWorkload(
        total_calls=50,
        manifested_calls=48,
        calibration_calls=2,
        main_calls=8,
        integration_step_blocks=4,
        integration_step_calls=16,
        local_counterfactual_calls=4,
        parameter_sensitivity_calls=6,
        spatial_shuffle_calls=6,
        raw_trace_shards=10,
        primary_total_calls=34,
        primary_manifested_calls=32,
        followup_main_blocks=2,
        followup_main_calls=8,
        followup_reference_blocks=2,
        followup_reference_calls=8,
    )
    assert workloads["pilot"] == ExtendedWorkload(
        total_calls=133,
        manifested_calls=127,
        calibration_calls=6,
        main_calls=24,
        integration_step_blocks=8,
        integration_step_calls=32,
        local_counterfactual_calls=8,
        parameter_sensitivity_calls=24,
        spatial_shuffle_calls=15,
        raw_trace_shards=30,
        primary_total_calls=85,
        primary_manifested_calls=79,
        followup_main_blocks=6,
        followup_main_calls=24,
        followup_reference_blocks=6,
        followup_reference_calls=24,
    )
    assert workloads["final"] == ExtendedWorkload(
        total_calls=1242,
        manifested_calls=1230,
        calibration_calls=12,
        main_calls=240,
        integration_step_blocks=100,
        integration_step_calls=400,
        local_counterfactual_calls=80,
        parameter_sensitivity_calls=120,
        spatial_shuffle_calls=150,
        raw_trace_shards=300,
        primary_total_calls=762,
        primary_manifested_calls=750,
        followup_main_blocks=60,
        followup_main_calls=240,
        followup_reference_blocks=60,
        followup_reference_calls=240,
    )


def test_extended_cli_defaults_and_static_check(monkeypatch, capsys) -> None:
    args = main_extended.parse_args([])
    assert args.mode is None
    assert args.workers is None
    assert args.resume is None

    def fail_if_called(**_kwargs):
        raise AssertionError("static check attempted notebook execution")

    monkeypatch.setattr(
        main_extended,
        "run_extended_notebook",
        fail_if_called,
    )
    assert main_extended.main(["--check"]) == 0
    output = capsys.readouterr().out
    assert "final: 1242 total calls (1230 manifested)" in output
    assert "762 unchanged primary + 240 prolonged 0.5 ms" in output
    assert "240 prolonged 0.25 ms" in output
    assert "integration-reference work units 100 / 400 calls" in output
    assert "trace shards 300" in output
    assert "no TVB calls ran" in output


def test_followup_checkpoint_block_records_four_calls(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "extended"
    controller = ExtendedRunController.create(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="extended-notebook",
        code_sha256="extended-code",
        environment={"python": "test", "packages": {}},
        planned_total_tvb_calls=4,
        planned_integration_step_work_units=1,
        planned_raw_trace_shards=1,
        worker_processes=1,
    )
    namespace = {
        "joblib": _PickleJoblib,
        "PARALLEL_WORKERS": 1,
    }
    original_job = {"severity": 1.0, "seed": 11, "dt_ms": 0.25}

    def worker(job):
        return tuple(job["probes"])

    worker.__name__ = "execute_late_followup_block"
    outcomes = ExtendedCheckpointDispatcher(namespace, controller)(
        worker,
        [original_job],
        (),
        "late follow-up synthetic block",
    )

    assert original_job == {"severity": 1.0, "seed": 11, "dt_ms": 0.25}
    assert outcomes == [("dc_matched", "2Hz", "5Hz")]
    status = read_run_status(run_dir)
    assert status["completed_tvb_calls"] == 4
    assert status["executed_tvb_calls_this_attempt"] == 4

    controller.mark_interrupted()
    assert "python main_extended.py --resume" in (
        run_dir / "progress.log"
    ).read_text(encoding="utf-8")

    resumed = ExtendedRunController.resume(
        run_dir=run_dir,
        mode="smoke",
        notebook_sha256="extended-notebook",
        code_sha256="extended-code",
        environment={"python": "test", "packages": {}},
        planned_total_tvb_calls=4,
        planned_integration_step_work_units=1,
        planned_raw_trace_shards=1,
        worker_processes=1,
    )

    def fail_if_recomputed(_job):
        raise AssertionError("completed prolonged block was recomputed")

    fail_if_recomputed.__name__ = "execute_late_followup_block"
    restored_outcomes = ExtendedCheckpointDispatcher(namespace, resumed)(
        fail_if_recomputed,
        [original_job],
        (),
        "late follow-up synthetic block",
    )
    resumed.mark_completed()

    assert restored_outcomes == outcomes
    resumed_status = read_run_status(run_dir)
    assert resumed_status["state"] == "completed"
    assert resumed_status["completed_tvb_calls"] == 4
    assert resumed_status["restored_tvb_calls_this_attempt"] == 4
    assert resumed_status["executed_tvb_calls_this_attempt"] == 0
