from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rise_tvb379.analysis import (
    build_matched_control_sets,
    check_integration_step,
)


def test_matched_sets_are_deterministic_bilateral_and_disjoint() -> None:
    rng = np.random.default_rng(7)
    weights = rng.random((379, 379))
    labels = np.array([f"region_{index}" for index in range(379)])
    baseline_b = np.full(379, 0.07)
    high_b = np.linspace(0.02, 0.07, 379)
    a1 = np.array([23, 203])
    music = np.array([43, 223, 39, 219])
    speech = np.array([128, 308, 73, 253])
    declared = np.r_[a1, music, speech]

    _, first = build_matched_control_sets(
        weights=weights,
        baseline_b=baseline_b,
        high_b=high_b,
        labels=labels,
        a1_indices=a1,
        music_indices=music,
        speech_indices=speech,
        all_declared_indices=declared,
        n_sets=3,
    )
    _, second = build_matched_control_sets(
        weights=weights,
        baseline_b=baseline_b,
        high_b=high_b,
        labels=labels,
        a1_indices=a1,
        music_indices=music,
        speech_indices=speech,
        all_declared_indices=declared,
        n_sets=3,
    )
    pd.testing.assert_frame_equal(first, second)

    excluded = set(declared.tolist())
    for row in first.itertuples(index=False):
        music_control = np.fromstring(
            row.music_control_indices, sep=";", dtype=int
        )
        speech_control = np.fromstring(
            row.speech_control_indices, sep=";", dtype=int
        )
        assert len(music_control) == len(speech_control) == 4
        assert not excluded.intersection(music_control)
        assert not excluded.intersection(speech_control)
        assert not set(music_control).intersection(speech_control)
        assert sum(index < 180 for index in music_control) == 2
        assert sum(index < 180 for index in speech_control) == 2


def test_integration_step_gate_passes_and_fails() -> None:
    main = pd.DataFrame(
        [
            {
                "seed": 11,
                "severity": 0.0,
                "probe": "2Hz",
                "network": "music",
                "transfer": 1.0,
            },
            {
                "seed": 11,
                "severity": 0.0,
                "probe": "2Hz",
                "network": "speech",
                "transfer": 2.0,
            },
        ]
    )
    reference = pd.DataFrame(
        [
            {"network": "music", "transfer": 1.01},
            {"network": "speech", "transfer": 2.01},
        ]
    )
    result = check_integration_step(
        main_network_df=main,
        reference_network_df=reference,
        main_seed=11,
    )
    assert (result["relative_difference"] < 0.05).all()

    reference.loc[reference["network"] == "music", "transfer"] = 2.0
    with pytest.raises(RuntimeError, match="convergence"):
        check_integration_step(
            main_network_df=main,
            reference_network_df=reference,
            main_seed=11,
        )
