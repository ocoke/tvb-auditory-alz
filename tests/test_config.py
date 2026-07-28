from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rise_tvb379.config import (
    DEFAULT_CONFIG,
    MODE_CONFIGS,
    config_digest,
    config_digest_input,
    config_to_dict,
    config_to_json,
    get_experiment_config,
    workload_counts,
)


def test_mode_values_match_notebook() -> None:
    smoke = MODE_CONFIGS["smoke"]
    assert smoke.seeds == (11,)
    assert smoke.severities == (0.0, 1.0)
    assert smoke.calibration_couplings == (60.0,)
    assert smoke.matched_null_sets == 40
    assert smoke.spatial_shuffles == 1

    pilot = MODE_CONFIGS["pilot"]
    assert pilot.seeds == (11, 23)
    assert pilot.calibration_couplings == (30.0, 60.0, 100.0)
    assert pilot.matched_null_sets == 200
    assert pilot.spatial_shuffles == 2

    final = MODE_CONFIGS["final"]
    assert final.seeds == (11, 23, 37, 53, 71)
    assert final.severities == (0.0, 0.5, 1.0)
    assert final.calibration_couplings == (
        10.0,
        30.0,
        60.0,
        100.0,
        200.0,
        300.0,
    )
    assert final.matched_null_sets == 500
    assert final.spatial_shuffles == 100
    assert [item.name for item in final.sensitivity_scenarios] == [
        "G30",
        "G100",
        "input_0.01",
        "input_0.04",
    ]


def test_final_workload_counts_include_calibration() -> None:
    counts = workload_counts(DEFAULT_CONFIG)
    assert counts.to_dict() == {
        "calibration": 12,
        "main": 60,
        "local_dynamics_counterfactual": 20,
        "sensitivity": 24,
        "spatial_shuffle": 300,
        "integration_step_check": 2,
        "manifest": 406,
        "total": 418,
    }


@pytest.mark.parametrize(
    ("mode", "manifest", "total"),
    [("smoke", 23, 25), ("pilot", 52, 58), ("final", 406, 418)],
)
def test_all_mode_workload_totals(
    mode: str, manifest: int, total: int
) -> None:
    counts = workload_counts(get_experiment_config(mode))
    assert counts.manifest == manifest
    assert counts.total == total


def test_mode_defaults_to_environment_then_final() -> None:
    assert get_experiment_config(environ={}).mode == "final"
    assert get_experiment_config(environ={"RISE_RUN_MODE": " PILOT "}).mode == "pilot"
    assert (
        get_experiment_config("SMOKE", environ={"RISE_RUN_MODE": "final"}).mode
        == "smoke"
    )
    with pytest.raises(ValueError, match="RUN_MODE"):
        get_experiment_config("quick")


def test_config_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIG.mode = "smoke"  # type: ignore[misc]
    with pytest.raises(TypeError):
        MODE_CONFIGS["final"] = MODE_CONFIGS["smoke"]  # type: ignore[index]


def test_config_serialization_is_canonical_and_digestible() -> None:
    payload = config_to_dict(DEFAULT_CONFIG)
    assert payload["mode"] == "final"
    assert payload["mode_config"]["spatial_shuffles"] == 100
    assert payload["experiment"]["n_regions"] == 379
    assert config_to_json(DEFAULT_CONFIG) == config_to_json(DEFAULT_CONFIG)
    assert config_digest_input(DEFAULT_CONFIG) == config_to_json(
        DEFAULT_CONFIG
    ).encode("utf-8")
    assert len(config_digest(DEFAULT_CONFIG)) == 64
    assert config_digest(DEFAULT_CONFIG) == config_digest(DEFAULT_CONFIG)

