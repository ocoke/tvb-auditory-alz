from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import main
from rise_tvb379 import parallel


def test_plain_invocation_has_final_defaults() -> None:
    args = main.parse_args([])
    assert args.mode is None
    assert args.resume is None
    assert args.output_root is None
    assert args.data_dir is None
    assert args.offline is None
    assert args.workers == 1


def test_new_run_options_parse() -> None:
    args = main.parse_args(
        [
            "--mode",
            "smoke",
            "--output-root",
            "/tmp/runs",
            "--data-dir",
            "/tmp/data",
            "--offline",
            "--workers",
            "auto",
        ]
    )
    assert args.mode == "smoke"
    assert args.output_root == Path("/tmp/runs")
    assert args.data_dir == Path("/tmp/data")
    assert args.offline is True
    assert args.workers is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", 1), ("4", 4), ("auto", None), ("AUTO", None)],
)
def test_worker_option_parses(
    value: str,
    expected: int | None,
) -> None:
    assert main.parse_args(["--workers", value]).workers == expected


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_worker_option_rejects_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--workers", value])


def test_resume_may_change_runtime_worker_count() -> None:
    args = main.parse_args(
        ["--resume", "/tmp/run", "--workers", "4"]
    )
    assert args.workers == 4


@pytest.mark.parametrize(
    "option",
    [
        ["--mode", "pilot"],
        ["--output-root", "/tmp/runs"],
        ["--data-dir", "/tmp/data"],
        ["--offline"],
    ],
)
def test_resume_rejects_every_new_run_option(option: list[str]) -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--resume", "/tmp/run", *option])


def test_help_does_not_run_dependency_preflight(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main.parse_args(["--help"])
    assert raised.value.code == 0
    assert "100 spatial shuffles" in capsys.readouterr().out


def test_aggregate_tvb_time_includes_calibration_and_manifest() -> None:
    calibration = pd.DataFrame({"wall_seconds": [2.0, 3.0]})
    manifest = pd.DataFrame({"wall_seconds": [5.0, 7.0]})

    assert main._aggregate_tvb_wall_seconds(calibration, manifest) == 17.0


def test_main_propagates_command_start_and_execution_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    execution = {
        "execution_mode": "single_process",
        "parallel_enabled": False,
        "multiprocessing_start_method": None,
        "available_cpu_count": 8,
        "worker_count": 1,
        "native_threads_per_worker": 1,
    }
    monkeypatch.setattr(main.time, "perf_counter", lambda: 123.5)
    monkeypatch.setattr(
        main,
        "_preflight_runtime",
        lambda: {"python_version": "3.12.0"},
    )
    monkeypatch.setattr(
        parallel,
        "configure_native_thread_limits",
        lambda threads: {},
    )
    monkeypatch.setattr(
        parallel,
        "execution_details",
        lambda workers: dict(execution),
    )
    monkeypatch.setattr(parallel, "available_cpu_count", lambda: 8)

    def fake_run_new(
        args,
        environment,
        *,
        command_started,
    ) -> int:
        captured["args"] = args
        captured["environment"] = environment
        captured["command_started"] = command_started
        return 0

    monkeypatch.setattr(main, "_run_new", fake_run_new)

    assert main.main(["--mode", "smoke"]) == 0
    assert captured["command_started"] == 123.5
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["python_version"] == "3.12.0"
    assert environment["execution"] == {
        **execution,
        "requested_worker_count": 1,
        "worker_count_capped_to_allocation": False,
    }
