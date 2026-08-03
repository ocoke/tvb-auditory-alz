from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis import recommended_posthoc_analysis as posthoc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_analysis"
OUTPUT_DIR = DATA_DIR / "investigation" / "recommended"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name)


def test_notebook_harmonic_fit_is_reused_exactly() -> None:
    time_ms = np.arange(4500.0, 6500.0, 2.0)
    t_seconds = (time_ms - time_ms[0]) / 1000.0
    amplitudes = np.array([2.5, 0.7])
    phases = np.array([0.4, -0.8])
    values = np.column_stack(
        [
            amplitude * np.sin(2.0 * np.pi * 2.0 * t_seconds + phase)
            for amplitude, phase in zip(amplitudes, phases)
        ]
    )
    fit = posthoc.exact_frequency_fit(
        time_ms, values, "2Hz", np.ones(len(time_ms), dtype=bool)
    )

    assert np.allclose(fit["amplitude"], amplitudes, atol=1e-10)
    assert np.allclose(fit["phase_rad"], phases, atol=1e-10)
    assert np.min(fit["r_squared"]) > 0.999999999


def test_transmission_audit_separates_estimands_and_quality() -> None:
    endpoint = _read("transmission_endpoint.csv")
    quality = _read("frequency_quality.csv")

    assert set(endpoint["metric"]) == {"Broadband RMS", "Frequency locked"}
    assert (endpoint["positive_seed_count"] == 20).all()
    assert np.allclose(
        endpoint[endpoint["metric"] == "Broadband RMS"]["interaction_mean"],
        [2.233173, 2.172442],
        atol=1e-6,
    )
    locked = endpoint[endpoint["metric"] == "Frequency locked"]
    assert locked["analysis_status"].str.contains("Sensitivity only").all()

    high = quality[quality["severity"] == 1.0]
    assert high["median_harmonic_fit_r_squared"].max() < 0.002
    semantic = high[high["network"] == "Semantic proxy"]
    assert semantic["median_frequency_qa_valid_fraction"].max() < 0.20


def test_spectral_audit_distinguishes_drive_from_dominant_power() -> None:
    peaks = _read("spectral_peak_summary.csv")
    evoked = peaks[peaks["signal"] == "Evoked"]
    baseline = evoked[evoked["severity"] == 0.0]
    high = evoked[evoked["severity"] == 1.0]

    assert np.allclose(
        baseline["peak_frequency_hz"], baseline["drive_frequency_hz"]
    )
    assert np.allclose(baseline["drive_to_peak_power_ratio"], 1.0)
    assert high["drive_to_peak_power_ratio"].max() < 0.01
    assert not np.isclose(
        high["peak_frequency_hz"], high["drive_frequency_hz"]
    ).any()


def test_fc_audit_keeps_raw_level_change_and_phase_quality_distinct() -> None:
    fc = _read("fc_trajectory.csv")
    raw = fc[fc["metric"] == "Raw FC"].pivot(
        index=["severity", "probe"], columns="network", values="mean"
    )
    assert (raw["Episodic proxy"] > raw["Semantic proxy"]).all()

    phase_rows = _read("phase_fc_rows.csv")
    phase_summary = _read("phase_fc_summary.csv")
    assert phase_rows["phase_interpretable"].all()
    assert len(phase_rows) == 453
    assert (phase_summary["phase_interpretable_fraction"] < 0.55).all()


def test_fixed_pulse_mask_preserves_original_validity_limitations() -> None:
    masks = _read("pulse_fixed_masks.csv")
    summary = _read("pulse_fixed_summary.csv")
    usable = masks.groupby("network")["usable_seed"].sum().to_dict()

    assert usable == {"Episodic proxy": 16, "Semantic proxy": 18}
    assert (masks["fixed_valid_parcels"] == 0).any()
    assert (summary["median_t20_ms"] < summary["median_t50_ms"]).all()
    assert (summary["median_t50_ms"] < summary["median_t80_ms"]).all()
    assert (summary["median_spread_t80_t20_ms"] > 0).all()


def test_regional_covariate_result_is_explicitly_exploratory() -> None:
    coefficients = _read("regional_covariate_coefficients.csv")
    models = _read("regional_covariate_models.csv")
    semantic = coefficients[
        coefficients["term"] == "semantic_indicator"
    ]

    assert len(semantic) == 2
    assert semantic["estimate"].between(2.30, 2.35).all()
    assert (semantic["ci95_lower"] > 0).all()
    assert (models["parcel_count"] == 32).all()
    assert models["analysis_status"].str.contains("Exploratory").all()
