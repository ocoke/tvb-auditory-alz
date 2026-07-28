from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rise_tvb379.metrics import (
    harmonic_amplitude,
    make_contrasts,
    normalize_to_baseline,
    pulse_rms,
)


def test_harmonic_amplitude_recovers_exact_signal() -> None:
    time_ms = np.arange(3500.0, 6500.0 + 2.0, 2.0)
    seconds = (time_ms - 3500.0) / 1000.0
    signal = np.column_stack(
        [
            2.5 * np.sin(2 * np.pi * 2.0 * seconds) + 0.2,
            0.75 * np.cos(2 * np.pi * 2.0 * seconds) - 0.1,
        ]
    )

    amplitude, r_squared = harmonic_amplitude(
        time_ms,
        signal,
        "2Hz",
        analysis_start_ms=3500.0,
        simulation_ms=6500.0,
    )

    np.testing.assert_allclose(amplitude, [2.5, 0.75], rtol=1e-10)
    np.testing.assert_allclose(r_squared, [1.0, 1.0], atol=1e-12)


def test_pulse_rms_uses_prespecified_window() -> None:
    time_ms = np.arange(0.0, 4000.0, 2.0)
    evoked = np.zeros((len(time_ms), 2))
    window = (time_ms >= 2500.0) & (time_ms <= 3500.0)
    evoked[window, 0] = 3.0
    evoked[window, 1] = 4.0

    response, fit = pulse_rms(
        time_ms,
        evoked,
        onset_ms=2500.0,
        analysis_end_ms=3500.0,
        n_regions=2,
    )

    np.testing.assert_allclose(response, [3.0, 4.0])
    assert np.isnan(fit).all()


def test_normalization_and_music_minus_speech_contrast() -> None:
    rows = []
    for severity, music, speech in [
        (0.0, 2.0, 4.0),
        (1.0, 4.0, 2.0),
    ]:
        for network, transfer in [("music", music), ("speech", speech)]:
            rows.append(
                {
                    "scope": "main",
                    "variant": "full",
                    "condition": str(severity),
                    "severity": severity,
                    "seed": 11,
                    "probe": "2Hz",
                    "global_coupling": 60.0,
                    "input_peak_per_ms": 0.02,
                    "dt_ms": 1.0,
                    "network": network,
                    "transfer": transfer,
                }
            )
    normalized = normalize_to_baseline(pd.DataFrame(rows))
    contrast = make_contrasts(normalized)
    endpoint = contrast.loc[contrast["severity"] == 1.0].iloc[0]
    assert endpoint["music"] == pytest.approx(1.0)
    assert endpoint["speech"] == pytest.approx(-1.0)
    assert endpoint["music_minus_speech_log2_change"] == pytest.approx(2.0)


def test_normalization_refuses_missing_baseline() -> None:
    frame = pd.DataFrame(
        [
            {
                "seed": 11,
                "probe": "2Hz",
                "network": "music",
                "severity": 1.0,
                "transfer": 1.0,
            }
        ]
    )
    with pytest.raises(RuntimeError, match="baseline"):
        normalize_to_baseline(frame)
