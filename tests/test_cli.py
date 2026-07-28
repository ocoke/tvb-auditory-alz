from __future__ import annotations

from pathlib import Path

import pytest

import main


def test_plain_invocation_has_final_defaults() -> None:
    args = main.parse_args([])
    assert args.mode is None
    assert args.resume is None
    assert args.output_root is None
    assert args.data_dir is None
    assert args.offline is None


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
        ]
    )
    assert args.mode == "smoke"
    assert args.output_root == Path("/tmp/runs")
    assert args.data_dir == Path("/tmp/data")
    assert args.offline is True


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
