from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest

from rise_tvb379.analysis import (
    build_matched_pair_sets,
    build_matching_features,
    check_integration_step,
)
from rise_tvb379.config import (
    DT_CHECK_NETWORKS,
    DT_CHECK_PROBES,
    DT_CHECK_SEVERITIES,
    SEVERITY_LABELS,
)


def test_matched_sets_are_deterministic_size_preserving_and_disjoint() -> None:
    rng = np.random.default_rng(7)
    weights = rng.random((379, 379))
    labels = np.array([f"region_{index}" for index in range(379)])
    baseline_b = np.full(379, 0.07)
    high_b = np.linspace(0.02, 0.07, 379)
    a1 = np.array([23, 203])
    music = np.array([43, 223, 39, 219, 106, 286, 122, 302])
    speech = np.array(
        [128, 308, 73, 253, 24, 204, 104, 284, 138, 318, 74, 75, 107, 72]
    )
    semantic = np.array([248, 163, 311, 134, 127])
    episodic = np.array([324, 206, 270, 246])
    declared = np.unique(np.r_[a1, music, speech, semantic, episodic])

    _, matching_z = build_matching_features(
        weights=weights,
        baseline_b=baseline_b,
        high_b=high_b,
        labels=labels,
        a1_indices=a1,
        target_groups={
            "music": music,
            "speech": speech,
            "semantic": semantic,
            "episodic": episodic,
        },
    )
    first = build_matched_pair_sets(
        labels=labels,
        matching_z=matching_z,
        all_declared_indices=declared,
        pair_name="primary",
        left_name="music",
        left_indices=music,
        right_name="speech",
        right_indices=speech,
        n_sets=3,
        random_seed=20260727,
    )
    second = build_matched_pair_sets(
        labels=labels,
        matching_z=matching_z,
        all_declared_indices=declared,
        pair_name="primary",
        left_name="music",
        left_indices=music,
        right_name="speech",
        right_indices=speech,
        n_sets=3,
        random_seed=20260727,
    )
    pd.testing.assert_frame_equal(first, second)

    excluded = set(declared.tolist())
    for row in first.itertuples(index=False):
        music_control = np.fromstring(
            row.left_control_indices, sep=";", dtype=int
        )
        speech_control = np.fromstring(
            row.right_control_indices, sep=";", dtype=int
        )
        assert len(music_control) == len(music)
        assert len(speech_control) == len(speech)
        assert not excluded.intersection(music_control)
        assert not excluded.intersection(speech_control)
        assert not set(music_control).intersection(speech_control)
        assert sum(index < 180 for index in music_control) == 4
        assert sum(index < 180 for index in speech_control) == 9


DT_NETWORKS = (*DT_CHECK_NETWORKS, "descriptive_context")
DT_KEYS = list(
    product(
        DT_CHECK_SEVERITIES,
        DT_CHECK_PROBES,
        DT_NETWORKS,
    )
)


def integration_step_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    main_rows = []
    reference_rows = []
    for index, (severity, probe, network) in enumerate(DT_KEYS, start=1):
        reference_transfer = float(index)
        reference_fit = 0.70 + index / 100.0
        direction = 1.0 if index % 2 else -1.0
        main_rows.append(
            {
                "seed": 11,
                "severity": severity,
                "probe": probe,
                "network": network,
                "transfer": reference_transfer * (1.0 + direction * 0.01),
                "median_target_fit_r_squared": (
                    reference_fit + direction * 0.02
                ),
            }
        )
        reference_rows.append(
            {
                "severity": severity,
                "probe": probe,
                "network": network,
                "transfer": reference_transfer,
                "median_target_fit_r_squared": reference_fit,
            }
        )
    return (
        pd.DataFrame(main_rows).sample(frac=1.0, random_state=11),
        pd.DataFrame(reference_rows).sample(frac=1.0, random_state=12),
    )


def test_integration_step_gate_checks_every_endpoint_probe_and_network() -> None:
    main, reference = integration_step_frames()
    other_seed = main.copy()
    other_seed["seed"] = 23
    other_seed["transfer"] = -1.0
    main = pd.concat([main, other_seed], ignore_index=True)

    result = check_integration_step(
        main_network_df=main,
        reference_network_df=reference,
        main_seed=11,
        networks=DT_NETWORKS,
    )

    assert list(result.columns) == [
        "severity",
        "probe",
        "network",
        "transfer_dt_0.5ms",
        "median_target_fit_r_squared_dt_0.5ms",
        "transfer_dt_0.25ms",
        "median_target_fit_r_squared_dt_0.25ms",
        "condition",
        "required_for_inference",
        "relative_difference",
        "convergence_passed",
        "fit_r_squared_difference",
    ]
    actual_keys = list(
        result[["severity", "probe", "network"]].itertuples(
            index=False, name=None
        )
    )
    assert set(actual_keys) == set(DT_KEYS)
    assert result["required_for_inference"].tolist() == (
        [True] * (len(DT_CHECK_SEVERITIES) * len(DT_CHECK_PROBES)
                  * len(DT_CHECK_NETWORKS))
        + [False] * (len(DT_CHECK_SEVERITIES) * len(DT_CHECK_PROBES))
    )
    assert result["condition"].tolist() == [
        SEVERITY_LABELS[severity] for severity, _, _ in actual_keys
    ]
    np.testing.assert_allclose(result["relative_difference"], 0.01)
    np.testing.assert_allclose(result["fit_r_squared_difference"], 0.02)
    assert result["convergence_passed"].all()


def test_integration_step_gate_fails_at_exact_threshold() -> None:
    main, reference = integration_step_frames()
    main_target = (
        (main["severity"] == 1.0)
        & (main["probe"] == "5Hz")
        & (main["network"] == "speech")
    )
    reference_target = (
        (reference["severity"] == 1.0)
        & (reference["probe"] == "5Hz")
        & (reference["network"] == "speech")
    )
    reference.loc[reference_target, "transfer"] = 20.0
    main.loc[main_target, "transfer"] = 21.0

    with pytest.raises(RuntimeError, match="convergence"):
        check_integration_step(
            main_network_df=main,
            reference_network_df=reference,
            main_seed=11,
            networks=DT_NETWORKS,
        )


def test_integration_step_descriptive_failure_warns_but_does_not_gate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    main, reference = integration_step_frames()
    target = (
        (main["severity"] == 1.0)
        & (main["probe"] == "5Hz")
        & (main["network"] == "descriptive_context")
    )
    reference_value = reference.loc[
        (reference["severity"] == 1.0)
        & (reference["probe"] == "5Hz")
        & (reference["network"] == "descriptive_context"),
        "transfer",
    ].iloc[0]
    main.loc[target, "transfer"] = reference_value * 1.10

    result = check_integration_step(
        main_network_df=main,
        reference_network_df=reference,
        main_seed=11,
        networks=DT_NETWORKS,
    )

    assert (~result["convergence_passed"]).sum() == 1
    assert "Descriptive context networks" in caplog.text


@pytest.mark.parametrize("frame_name", ["main", "reference"])
def test_integration_step_gate_rejects_missing_coverage(
    frame_name: str,
) -> None:
    main, reference = integration_step_frames()
    if frame_name == "main":
        main = main[
            ~(
                (main["severity"] == 1.0)
                & (main["probe"] == "5Hz")
                & (main["network"] == "speech")
            )
        ]
    else:
        reference = reference[
            ~(
                (reference["severity"] == 1.0)
                & (reference["probe"] == "5Hz")
                & (reference["network"] == "speech")
            )
        ]

    with pytest.raises(
        RuntimeError,
        match=rf"{frame_name.title()}.*missing required coverage keys",
    ):
        check_integration_step(
            main_network_df=main,
            reference_network_df=reference,
            main_seed=11,
            networks=DT_NETWORKS,
        )


@pytest.mark.parametrize("frame_name", ["main", "reference"])
def test_integration_step_gate_rejects_duplicate_coverage(
    frame_name: str,
) -> None:
    main, reference = integration_step_frames()
    if frame_name == "main":
        main = pd.concat([main, main.iloc[[0]]], ignore_index=True)
    else:
        reference = pd.concat(
            [reference, reference.iloc[[0]]],
            ignore_index=True,
        )

    with pytest.raises(
        RuntimeError,
        match=rf"{frame_name.title()}.*duplicate coverage keys",
    ):
        check_integration_step(
            main_network_df=main,
            reference_network_df=reference,
            main_seed=11,
            networks=DT_NETWORKS,
        )


@pytest.mark.parametrize("frame_name", ["main", "reference"])
def test_integration_step_gate_reports_missing_columns(
    frame_name: str,
) -> None:
    main, reference = integration_step_frames()
    if frame_name == "main":
        main = main.drop(columns="median_target_fit_r_squared")
    else:
        reference = reference.drop(columns="median_target_fit_r_squared")

    with pytest.raises(
        RuntimeError,
        match=rf"{frame_name.title()}.*missing required columns",
    ):
        check_integration_step(
            main_network_df=main,
            reference_network_df=reference,
            main_seed=11,
            networks=DT_NETWORKS,
        )
