from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

from rise_tvb379 import pipeline
from rise_tvb379.checkpoints import COMPLETE_MARKER_NAME, read_completed_block
from rise_tvb379.config import config_to_dict, get_experiment_config
from rise_tvb379.parallel import WorkerJob, WorkerOutcome
from rise_tvb379.simulation import BlockResult


@dataclass(frozen=True)
class TinyContext:
    default_simulation_ms: float = 6500.0


class ReverseFakeRunner:
    """Return lightweight outcomes in reverse without creating processes."""

    def __init__(
        self,
        *,
        checkpoint_root: Path,
        stage: str,
        block_keys: dict[int, str],
        worker_count: int = 4,
    ) -> None:
        self.checkpoint_root = checkpoint_root
        self.stage = stage
        self.block_keys = block_keys
        self.worker_count = worker_count
        self.submissions: list[list[WorkerJob]] = []

    def _assert_checkpointed(self, job: WorkerJob) -> None:
        marker = (
            self.checkpoint_root
            / self.stage
            / self.block_keys[job.ordinal]
            / COMPLETE_MARKER_NAME
        )
        assert marker.is_file(), (
            "The parent must checkpoint each completed outcome before "
            "requesting the next worker result."
        )

    @staticmethod
    def _outcome(job: WorkerJob) -> WorkerOutcome:
        payload = job.payload
        condition = payload["condition"]
        row = {
            "ordinal": job.ordinal,
            "condition": condition["condition"],
            "seed": payload["seed"],
            "scope": payload["scope"],
        }
        result = BlockResult(
            node=pd.DataFrame([row]),
            network=pd.DataFrame([row]),
            manifest=pd.DataFrame([row]),
        )
        return WorkerOutcome(
            ordinal=job.ordinal,
            kind=job.kind,
            result=result,
            worker_pid=10_000 + job.ordinal,
            elapsed_seconds=0.5 + job.ordinal,
        )

    def execute(
        self,
        jobs: Iterable[WorkerJob],
    ) -> Iterable[WorkerOutcome]:
        submitted = list(jobs)
        self.submissions.append(submitted)

        previous: WorkerJob | None = None
        for job in reversed(submitted):
            if previous is not None:
                self._assert_checkpointed(previous)
            yield self._outcome(job)
            previous = job
        if previous is not None:
            self._assert_checkpointed(previous)


def _conditions() -> list[dict[str, object]]:
    return [
        {
            "condition": "Baseline",
            "severity": 0.0,
            "b_values": np.array([0.07, 0.07]),
            "variant": "default_variant",
        },
        {
            "condition": "High",
            "severity": 1.0,
            "b_values": np.array([0.02, 0.03]),
            "variant": "overridden_variant",
            "scope": "condition_specific_scope",
            "global_coupling": 30.0,
            "input_peak_per_ms": 0.04,
            "key_prefix": "condition_specific_",
        },
    ]


def _expected_block_keys() -> dict[int, str]:
    return {
        0: "default_severity_0.0_seed_11",
        1: "default_severity_0.0_seed_23",
        2: "condition_specific_severity_1.0_seed_11",
        3: "condition_specific_severity_1.0_seed_23",
    }


def _run_test_grid(
    checkpoint_root: Path,
    runner: ReverseFakeRunner,
    progress: pipeline._ProgressReporter,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pipeline._run_grid(
        checkpoint_root,
        TinyContext(),  # type: ignore[arg-type]
        runner,  # type: ignore[arg-type]
        progress,
        stage_number=6,
        stage_name="test grid",
        stage="test_grid",
        scope="default_scope",
        conditions=_conditions(),
        seeds=(11, 23),
        probes=("2Hz", "5Hz"),
        global_coupling=60.0,
        input_peak_per_ms=0.02,
        dt_ms=1.0,
        simulation_ms=6500.0,
        key_prefix="default_",
    )


def test_run_grid_parallel_completion_is_checkpointed_and_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    block_keys = _expected_block_keys()
    runner = ReverseFakeRunner(
        checkpoint_root=checkpoint_root,
        stage="test_grid",
        block_keys=block_keys,
    )
    ticks = iter((0.0, 10.0, 20.0, 30.0, 40.0))
    monkeypatch.setattr(
        pipeline.time,
        "perf_counter",
        lambda: next(ticks),
    )
    progress = pipeline._ProgressReporter(
        total_calls=12,
        total_work_units=12.0,
    )
    caplog.set_level("INFO", logger="rise_tvb379.pipeline")

    node, network, manifest = _run_test_grid(
        checkpoint_root,
        runner,
        progress,
    )

    assert len(runner.submissions) == 1
    jobs = runner.submissions[0]
    assert [job.ordinal for job in jobs] == [0, 1, 2, 3]
    assert [job.kind for job in jobs] == ["simulation"] * 4

    # Worker completion was 3, 2, 1, 0, but all reader-facing frames retain
    # condition-then-seed scientific order.
    expected_ordinals = [0, 1, 2, 3]
    assert node["ordinal"].tolist() == expected_ordinals
    assert network["ordinal"].tolist() == expected_ordinals
    assert manifest["ordinal"].tolist() == expected_ordinals
    assert node[["condition", "seed"]].to_records(index=False).tolist() == [
        ("Baseline", 11),
        ("Baseline", 23),
        ("High", 11),
        ("High", 23),
    ]

    default_payload = jobs[0].payload
    assert default_payload["scope"] == "default_scope"
    assert default_payload["global_coupling"] == 60.0
    assert default_payload["input_peak_per_ms"] == 0.02
    assert np.array_equal(
        default_payload["condition"]["b_values"],
        np.array([0.07, 0.07]),
    )

    overridden_payload = jobs[2].payload
    assert overridden_payload["scope"] == "condition_specific_scope"
    assert overridden_payload["global_coupling"] == 30.0
    assert overridden_payload["input_peak_per_ms"] == 0.04
    assert np.array_equal(
        overridden_payload["condition"]["b_values"],
        np.array([0.02, 0.03]),
    )

    for ordinal, block_key in block_keys.items():
        saved = read_completed_block(
            checkpoint_root,
            "test_grid",
            block_key,
        )
        assert saved.frames["node"]["ordinal"].tolist() == [ordinal]

    default_metadata = read_completed_block(
        checkpoint_root,
        "test_grid",
        block_keys[0],
    ).metadata
    assert default_metadata["scope"] == "default_scope"
    assert default_metadata["global_coupling"] == 60.0
    assert default_metadata["input_peak_per_ms"] == 0.02

    overridden_metadata = read_completed_block(
        checkpoint_root,
        "test_grid",
        block_keys[2],
    ).metadata
    assert overridden_metadata["scope"] == "condition_specific_scope"
    assert overridden_metadata["global_coupling"] == 30.0
    assert overridden_metadata["input_peak_per_ms"] == 0.04

    assert progress.completed_calls == 12
    assert progress.completed_work_units == pytest.approx(12.0)
    assert progress.executed_work_units == pytest.approx(12.0)
    assert "Stage 6/8 test grid: 4 blocks, 12 TVB calls" in caplog.text
    assert "Progress 12/12 (100.0%)" in caplog.text
    assert "elapsed 00:00:40" in caplog.text


def test_run_grid_restores_cached_blocks_without_resubmission(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    block_keys = _expected_block_keys()
    first_runner = ReverseFakeRunner(
        checkpoint_root=checkpoint_root,
        stage="test_grid",
        block_keys=block_keys,
    )
    first_progress = pipeline._ProgressReporter(
        total_calls=12,
        total_work_units=12.0,
    )
    expected = _run_test_grid(
        checkpoint_root,
        first_runner,
        first_progress,
    )

    cached_runner = ReverseFakeRunner(
        checkpoint_root=checkpoint_root,
        stage="test_grid",
        block_keys=block_keys,
    )
    cached_progress = pipeline._ProgressReporter(
        total_calls=12,
        total_work_units=12.0,
    )
    restored = _run_test_grid(
        checkpoint_root,
        cached_runner,
        cached_progress,
    )

    assert cached_runner.submissions == [[]]
    for restored_frame, expected_frame in zip(restored, expected, strict=True):
        pd.testing.assert_frame_equal(restored_frame, expected_frame)
    assert cached_progress.completed_calls == 12
    assert cached_progress.completed_work_units == pytest.approx(12.0)
    assert cached_progress.executed_work_units == 0.0


def test_run_grid_mixed_resume_submits_only_missing_blocks(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    block_keys = _expected_block_keys()
    initial_runner = ReverseFakeRunner(
        checkpoint_root=checkpoint_root,
        stage="test_grid",
        block_keys=block_keys,
    )
    initial_progress = pipeline._ProgressReporter(
        total_calls=12,
        total_work_units=12.0,
    )
    expected = _run_test_grid(
        checkpoint_root,
        initial_runner,
        initial_progress,
    )
    for ordinal in (1, 3):
        shutil.rmtree(
            checkpoint_root / "test_grid" / block_keys[ordinal]
        )

    resumed_runner = ReverseFakeRunner(
        checkpoint_root=checkpoint_root,
        stage="test_grid",
        block_keys=block_keys,
    )
    resumed_progress = pipeline._ProgressReporter(
        total_calls=12,
        total_work_units=12.0,
    )
    resumed = _run_test_grid(
        checkpoint_root,
        resumed_runner,
        resumed_progress,
    )

    assert [
        job.ordinal for job in resumed_runner.submissions[0]
    ] == [1, 3]
    for resumed_frame, expected_frame in zip(
        resumed,
        expected,
        strict=True,
    ):
        pd.testing.assert_frame_equal(resumed_frame, expected_frame)
    assert resumed_progress.completed_calls == 12
    assert resumed_progress.completed_work_units == pytest.approx(12.0)
    assert resumed_progress.executed_work_units == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [
        (-5.0, "00:00:00"),
        (0.0, "00:00:00"),
        (61.0, "00:01:01"),
        (3661.0, "01:01:01"),
        (90_061.0, "25:01:01"),
    ],
)
def test_duration_format(seconds: float, formatted: str) -> None:
    assert pipeline._format_duration(seconds) == formatted


def test_expanded_dt_grid_fields_are_serialized() -> None:
    config = get_experiment_config("final")
    payload = config_to_dict(config)

    assert config.dt_check_severities == (0.0, 1.0)
    assert config.dt_check_probes == ("2Hz", "5Hz")
    assert payload["experiment"]["dt_check_severities"] == [0.0, 1.0]
    assert payload["experiment"]["dt_check_probes"] == ["2Hz", "5Hz"]
