#!/usr/bin/env python3
"""Run the recommended post-hoc audits on the completed TVB379 export.

This module never imports TVB, changes the ScienceReady notebook, or starts a
simulation.  It reuses the notebook's saved metrics and, where a requested
quantity was not exported as a CSV column, applies the notebook's exact
harmonic-fit, detrending, phase, and pulse-energy definitions to the lossless
``main_parcel_traces/*.npz`` shards.

The outputs are descriptive sensitivity analyses.  Numerical seeds are not
participants, and post-hoc regressions are not confirmatory biological tests.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate, signal, stats


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DATA_DIR / "investigation" / "recommended"
PERIODIC_PROBES = ("2Hz", "5Hz")
SEVERITIES = (0.0, 0.5, 1.0)
PRIMARY_NETWORKS = ("semantic_expanded", "episodic_expanded")
NETWORK_LABELS = {
    "semantic_expanded": "Semantic proxy",
    "episodic_expanded": "Episodic proxy",
}
NETWORK_MEMBERSHIP_KEYS = {
    "semantic_expanded": "semantic_expanded_membership",
    "episodic_expanded": "episodic_expanded_membership",
}
PERIODIC_ANALYSIS_START_MS = 4500.0
PERIODIC_ANALYSIS_END_MS = 14500.0
PERIODIC_SEGMENT_MS = 2000.0
PERIODIC_SEGMENT_COUNT = 5
STIMULUS_ONSET_MS = 2500.0
PULSE_ANALYSIS_END_MS = 6000.0
PULSE_TAIL_WINDOW_MS = 200.0
MULTITAPER_TIME_BANDWIDTH = 3.0
MULTITAPER_TAPERS = 5
SPECTRUM_MIN_HZ = 0.5
SPECTRUM_MAX_HZ = 12.0
EXPECTED_TRACE_SHARDS = 180
EXPECTED_TRACE_REGIONS = 34


class PosthocAnalysisError(RuntimeError):
    """Raised when the completed export cannot support a requested audit."""


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)


def _read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    if not path.is_file():
        raise PosthocAnalysisError(f"Required result table is missing: {path}")
    return pd.read_csv(path)


def _wrap_phase_rad(values: Any) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(values, dtype=float)))


def _label_hemisphere(label: str) -> str:
    if label.startswith("L_"):
        return "L"
    if label.startswith("R_"):
        return "R"
    return "M"


def _ipsilateral_reference(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels).astype(str)
    result = np.empty(len(labels), dtype=float)
    a1_by_hemisphere = {
        "L": int(np.flatnonzero(labels == "L_A1")[0]),
        "R": int(np.flatnonzero(labels == "R_A1")[0]),
    }
    bilateral_reference = float(
        np.nanmean(values[list(a1_by_hemisphere.values())])
    )
    for index, label in enumerate(labels):
        hemisphere = _label_hemisphere(label)
        result[index] = (
            values[a1_by_hemisphere[hemisphere]]
            if hemisphere in a1_by_hemisphere
            else bilateral_reference
        )
    return result


def _phase_lag_to_ipsilateral_a1(
    phase_rad: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    phase_rad = np.asarray(phase_rad, dtype=float)
    labels = np.asarray(labels).astype(str)
    a1_indices = [
        int(np.flatnonzero(labels == label)[0]) for label in ("L_A1", "R_A1")
    ]
    bilateral_reference = float(
        np.angle(np.mean(np.exp(1j * phase_rad[a1_indices])))
    )
    reference = np.empty(len(labels), dtype=float)
    for index, label in enumerate(labels):
        hemisphere = _label_hemisphere(label)
        if hemisphere == "L":
            reference[index] = phase_rad[a1_indices[0]]
        elif hemisphere == "R":
            reference[index] = phase_rad[a1_indices[1]]
        else:
            reference[index] = bilateral_reference
    return _wrap_phase_rad(phase_rad - reference)


def _aggregate_node_metric(
    values: np.ndarray,
    labels: np.ndarray,
    membership: np.ndarray,
    metric_kind: str,
) -> float:
    """Apply the notebook's hemisphere-balanced network aggregation."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels).astype(str)
    membership = np.asarray(membership, dtype=bool)
    hemisphere_values: list[float] = []
    for hemisphere in ("L", "R", "M"):
        selected = membership & np.array(
            [_label_hemisphere(label) == hemisphere for label in labels]
        )
        finite = values[selected & np.isfinite(values)]
        if not len(finite):
            continue
        if metric_kind == "latency":
            hemisphere_values.append(float(np.median(finite)))
        else:
            hemisphere_values.append(float(np.mean(finite)))
    if not hemisphere_values:
        return float("nan")
    if metric_kind == "transfer":
        if any(value <= 0 for value in hemisphere_values):
            raise PosthocAnalysisError(
                "Transfer aggregation received a nonpositive value."
            )
        return float(np.exp(np.mean(np.log(hemisphere_values))))
    return float(np.mean(hemisphere_values))


def detrended_ac_rms(values: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Exact copy of the notebook's broadband periodic-response estimator."""
    y = np.asarray(values[window], dtype=float)
    if y.shape[0] < 100:
        raise PosthocAnalysisError("An evoked-response window is too short.")
    y = signal.detrend(y, axis=0, type="linear")
    return np.sqrt(np.mean(y**2, axis=0))


def exact_frequency_fit(
    time_ms: np.ndarray,
    evoked: np.ndarray,
    probe: str,
    analysis_window: np.ndarray,
) -> dict[str, np.ndarray]:
    """Use the notebook's sine/cosine/intercept/trend least-squares fit."""
    frequency_hz = {"2Hz": 2.0, "5Hz": 5.0}[probe]
    window = np.asarray(analysis_window, dtype=bool)
    t_seconds = (time_ms[window] - float(time_ms[window][0])) / 1000.0
    y = np.asarray(evoked[window], dtype=float)
    if y.shape[0] < 100:
        raise PosthocAnalysisError(
            "Periodic analysis window is unexpectedly short."
        )
    omega_t = 2.0 * np.pi * frequency_hz * t_seconds
    design = np.column_stack(
        [
            np.sin(omega_t),
            np.cos(omega_t),
            np.ones_like(t_seconds),
            t_seconds - np.mean(t_seconds),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    sin_coefficient = coefficients[0]
    cos_coefficient = coefficients[1]
    amplitude = np.sqrt(sin_coefficient**2 + cos_coefficient**2)
    phase_rad = np.arctan2(cos_coefficient, sin_coefficient)
    residual_ss = np.sum((y - fitted) ** 2, axis=0)
    total_ss = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    r_squared = 1.0 - residual_ss / np.maximum(total_ss, 1e-15)
    return {
        "amplitude": amplitude,
        "r_squared": r_squared,
        "sin_coefficient": sin_coefficient,
        "cos_coefficient": cos_coefficient,
        "phase_rad": phase_rad,
    }


def _multitaper_power(
    time_ms: np.ndarray,
    values: np.ndarray,
    window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalize the notebook's exact multitaper SNR power calculation."""
    y = signal.detrend(
        np.asarray(values[window], dtype=float), axis=0, type="linear"
    )
    sampling_hz = 1000.0 / float(np.median(np.diff(time_ms[window])))
    tapers = signal.windows.dpss(
        y.shape[0],
        NW=MULTITAPER_TIME_BANDWIDTH,
        Kmax=MULTITAPER_TAPERS,
        sym=False,
        norm=2,
    )
    frequencies = np.fft.rfftfreq(y.shape[0], d=1.0 / sampling_hz)
    power = np.zeros((len(frequencies), y.shape[1]), dtype=float)
    for taper in tapers:
        spectrum = np.fft.rfft(y * np.asarray(taper)[:, None], axis=0)
        power += np.abs(spectrum) ** 2
    power /= float(len(tapers))
    return frequencies, power


def _pulse_energy_quantiles(
    time_ms: np.ndarray,
    evoked: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extend the notebook's cumulative-energy t50 definition to t20/t80."""
    window = (time_ms >= STIMULUS_ONSET_MS) & (
        time_ms < PULSE_ANALYSIS_END_MS
    )
    times = np.asarray(time_ms[window], dtype=float)
    y = np.asarray(evoked[window], dtype=float)
    energy = y**2
    cumulative = np.cumsum(energy, axis=0)
    total = cumulative[-1]
    valid = total > 1e-24
    fraction = cumulative / np.maximum(total, 1e-30)
    result: dict[str, np.ndarray] = {}
    for percentile in (20, 50, 80):
        first_index = np.argmax(fraction >= percentile / 100.0, axis=0)
        values = times[first_index] - STIMULUS_ONSET_MS
        values[~valid] = np.nan
        result[f"t{percentile}_ms"] = values
    result["spread_t80_t20_ms"] = result["t80_ms"] - result["t20_ms"]
    result["total_energy_psp2_ms"] = integrate.trapezoid(
        y**2, x=times, axis=0
    )
    tail_mask = times >= PULSE_ANALYSIS_END_MS - PULSE_TAIL_WINDOW_MS
    result["tail_energy_fraction"] = np.sum(
        energy[tail_mask], axis=0
    ) / np.maximum(np.sum(energy, axis=0), 1e-30)
    absolute_peak_indices = np.argmax(np.abs(y), axis=0)
    result["absolute_peak_psp"] = np.abs(
        y[absolute_peak_indices, np.arange(y.shape[1])]
    )
    result["absolute_peak_after_onset_ms"] = (
        times[absolute_peak_indices] - STIMULUS_ONSET_MS
    )
    return result


def _read_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "stimulated_psp",
            "control_psp",
            "evoked_psp",
            "time_ms",
            "region_labels",
            "region_indices",
            "semantic_expanded_membership",
            "episodic_expanded_membership",
            "severity",
            "seed",
            "probe",
            "stimulus_onset_ms",
            "integration_step_ms",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise PosthocAnalysisError(
                f"Trace archive {path.name} is missing {missing}."
            )
        return {name: np.asarray(archive[name]) for name in archive.files}


def _mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    numeric = np.asarray(list(values), dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
        }
    mean = float(np.mean(numeric))
    if len(numeric) > 1:
        half_width = float(
            stats.t.ppf(0.975, len(numeric) - 1)
            * np.std(numeric, ddof=1)
            / np.sqrt(len(numeric))
        )
    else:
        half_width = np.nan
    return {
        "n": int(len(numeric)),
        "mean": mean,
        "median": float(np.median(numeric)),
        "minimum": float(np.min(numeric)),
        "maximum": float(np.max(numeric)),
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def build_saved_metric_audits(
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Separate broadband/locked transfer and raw/baseline FC."""
    network = _read_csv(data_dir, "main_network_metrics.csv")
    target = network[
        network["network"].isin(PRIMARY_NETWORKS)
        & network["probe"].isin(PERIODIC_PROBES)
    ].copy()
    expected = len(SEVERITIES) * 20 * len(PERIODIC_PROBES) * len(PRIMARY_NETWORKS)
    if len(target) != expected:
        raise PosthocAnalysisError(
            f"Expected {expected} primary periodic network rows; found {len(target)}."
        )
    baseline = target[target["severity"] == 0.0][
        [
            "seed",
            "probe",
            "network",
            "transfer",
            "locked_transfer_segment_median",
            "evoked_fc_z",
        ]
    ].rename(
        columns={
            "transfer": "baseline_transfer",
            "locked_transfer_segment_median": "baseline_locked_transfer",
            "evoked_fc_z": "baseline_evoked_fc_z",
        }
    )
    target = target.merge(
        baseline, on=["seed", "probe", "network"], validate="many_to_one"
    )
    target["broadband_gain_log2"] = np.log2(
        target["transfer"] / target["baseline_transfer"]
    )
    target["frequency_locked_gain_log2"] = np.log2(
        target["locked_transfer_segment_median"]
        / target["baseline_locked_transfer"]
    )
    target["fc_change_from_baseline_z"] = (
        target["evoked_fc_z"] - target["baseline_evoked_fc_z"]
    )

    trajectory_rows: list[dict[str, Any]] = []
    for metric, column in (
        ("Broadband RMS", "broadband_gain_log2"),
        ("Frequency locked", "frequency_locked_gain_log2"),
    ):
        for (severity, probe, network_name), group in target.groupby(
            ["severity", "probe", "network"], sort=True
        ):
            trajectory_rows.append(
                {
                    "severity": float(severity),
                    "severity_label": f"{float(severity):.1f}",
                    "probe": probe,
                    "network": NETWORK_LABELS[network_name],
                    "metric": metric,
                    "series": f"{probe} · {metric} · {NETWORK_LABELS[network_name]}",
                    **_mean_ci(group[column]),
                }
            )
    transmission_trajectory = pd.DataFrame(trajectory_rows)

    endpoint_rows: list[dict[str, Any]] = []
    high = target[target["severity"] == 1.0]
    for metric, column, status in (
        (
            "Broadband RMS",
            "broadband_gain_log2",
            "Confirmatory primary; direction robust, exact magnitude DT-sensitive",
        ),
        (
            "Frequency locked",
            "frequency_locked_gain_log2",
            "Sensitivity only; target frequency quality is poor",
        ),
    ):
        wide = high.pivot(
            index=["seed", "probe"], columns="network", values=column
        ).reset_index()
        wide["interaction"] = (
            wide["semantic_expanded"] - wide["episodic_expanded"]
        )
        for probe, group in wide.groupby("probe", sort=True):
            summary = _mean_ci(group["interaction"])
            endpoint_rows.append(
                {
                    "probe": probe,
                    "metric": metric,
                    "label": f"{probe} · {metric}",
                    "semantic_mean": float(group["semantic_expanded"].mean()),
                    "episodic_mean": float(group["episodic_expanded"].mean()),
                    "positive_seed_count": int((group["interaction"] > 0).sum()),
                    "analysis_status": status,
                    **{f"interaction_{key}": value for key, value in summary.items()},
                }
            )
    transmission_endpoint = pd.DataFrame(endpoint_rows)

    quality = target.groupby(
        ["severity", "probe", "network"], as_index=False
    ).agg(
        median_harmonic_fit_r_squared=("median_target_fit_r_squared", "median"),
        median_target_snr_db=("median_target_snr_db", "median"),
        median_phase_consistency=("median_target_phase_consistency", "median"),
        median_frequency_qa_valid_fraction=(
            "target_frequency_qa_valid_fraction",
            "median",
        ),
        median_broadband_split_log2=("transfer_split_abs_log2", "median"),
        nonstationary_fraction=("transfer_nonstationary_flag", "mean"),
    )
    quality["severity_label"] = quality["severity"].map(lambda value: f"{value:.1f}")
    quality["network"] = quality["network"].map(NETWORK_LABELS)
    quality["series"] = quality["probe"] + " · " + quality["network"]

    segment_rows: list[dict[str, Any]] = []
    for row in target.itertuples(index=False):
        for segment_index in range(1, PERIODIC_SEGMENT_COUNT + 1):
            segment_rows.append(
                {
                    "severity": float(row.severity),
                    "seed": int(row.seed),
                    "probe": str(row.probe),
                    "network": NETWORK_LABELS[str(row.network)],
                    "segment": segment_index,
                    "segment_start_s": float(2 * (segment_index - 1)),
                    "segment_end_s": float(2 * segment_index),
                    "broadband_transfer": float(
                        getattr(row, f"segment_transfer_{segment_index}")
                    ),
                    "frequency_locked_transfer": float(
                        getattr(row, f"segment_locked_transfer_{segment_index}")
                    ),
                    "broadband_first_half": float(row.transfer_first_half),
                    "broadband_second_half": float(row.transfer_second_half),
                    "broadband_split_abs_log2": float(
                        row.transfer_split_abs_log2
                    ),
                    "fc_first_half_z": float(row.evoked_fc_first_half_z),
                    "fc_second_half_z": float(row.evoked_fc_second_half_z),
                    "fc_split_abs_z": float(row.evoked_fc_split_abs_z),
                }
            )
    segment_saved = pd.DataFrame(segment_rows)

    fc_rows: list[dict[str, Any]] = []
    for (severity, probe, network_name), group in target.groupby(
        ["severity", "probe", "network"], sort=True
    ):
        raw = _mean_ci(group["evoked_fc_z"])
        change = _mean_ci(group["fc_change_from_baseline_z"])
        for metric, summary in (
            ("Raw FC", raw),
            ("Change from baseline", change),
        ):
            fc_rows.append(
                {
                    "severity": float(severity),
                    "severity_label": f"{float(severity):.1f}",
                    "probe": probe,
                    "network": NETWORK_LABELS[network_name],
                    "metric": metric,
                    "series": f"{probe} · {NETWORK_LABELS[network_name]}",
                    **summary,
                }
            )
    fc_trajectory = pd.DataFrame(fc_rows)
    return {
        "transmission_trajectory": transmission_trajectory,
        "transmission_endpoint": transmission_endpoint,
        "frequency_quality": quality,
        "segment_saved": segment_saved,
        "fc_trajectory": fc_trajectory,
    }


def build_phase_fc_audit(data_dir: Path) -> dict[str, pd.DataFrame]:
    node = _read_csv(data_dir, "main_node_metrics.csv")
    regional = _read_csv(data_dir, "regional_features.csv")
    node = node[
        node["probe"].isin(PERIODIC_PROBES)
        & node["severity"].isin((0.0, 1.0))
    ].copy()
    base = node[node["severity"] == 0.0][
        [
            "seed",
            "probe",
            "region_index",
            "phase_lag_to_ipsilateral_a1_rad",
            "evoked_fc_z",
            "frequency_qa_valid",
        ]
    ].rename(
        columns={
            "phase_lag_to_ipsilateral_a1_rad": "baseline_phase_lag_rad",
            "evoked_fc_z": "baseline_fc_z",
            "frequency_qa_valid": "baseline_frequency_qa_valid",
        }
    )
    high = node[node["severity"] == 1.0].merge(
        base, on=["seed", "probe", "region_index"], validate="one_to_one"
    )
    columns = [
        "region_index",
        "region_label",
        "hemisphere",
        "semantic_expanded_membership",
        "episodic_expanded_membership",
        "b_reduction",
        "bilateral_a1_affinity",
        "weighted_structural_strength",
    ]
    high = high.merge(regional[columns], on="region_index", validate="many_to_one")
    high["phase_shift_rad"] = _wrap_phase_rad(
        high["phase_lag_to_ipsilateral_a1_rad"]
        - high["baseline_phase_lag_rad"]
    )
    high["absolute_phase_shift_degrees"] = np.abs(
        np.degrees(high["phase_shift_rad"])
    )
    high["fc_change_z"] = high["evoked_fc_z"] - high["baseline_fc_z"]
    high["fc_loss_z"] = -high["fc_change_z"]
    high["phase_interpretable"] = (
        _bool_series(high["frequency_qa_valid"])
        & _bool_series(high["baseline_frequency_qa_valid"])
        & np.isfinite(high["absolute_phase_shift_degrees"])
        & np.isfinite(high["fc_loss_z"])
    )
    membership_rows: list[pd.DataFrame] = []
    for network, membership_column in (
        ("Semantic proxy", "semantic_expanded_membership"),
        ("Episodic proxy", "episodic_expanded_membership"),
    ):
        selected = high[_bool_series(high[membership_column])].copy()
        selected["network"] = network
        membership_rows.append(selected)
    phase_rows = pd.concat(membership_rows, ignore_index=True)
    keep = [
        "seed",
        "probe",
        "network",
        "region_index",
        "region_label_x",
        "hemisphere_x",
        "baseline_fc_z",
        "evoked_fc_z",
        "fc_change_z",
        "fc_loss_z",
        "baseline_phase_lag_rad",
        "phase_lag_to_ipsilateral_a1_rad",
        "phase_shift_rad",
        "absolute_phase_shift_degrees",
        "phase_interpretable",
        "fit_r_squared",
        "snr_db",
        "phase_consistency",
        "b_reduction",
        "bilateral_a1_affinity",
        "weighted_structural_strength",
    ]
    phase_rows = phase_rows[keep].rename(
        columns={"region_label_x": "region_label", "hemisphere_x": "hemisphere"}
    )
    phase_interpretable = phase_rows[_bool_series(phase_rows["phase_interpretable"])]
    summary_rows: list[dict[str, Any]] = []
    for (probe, network), group in phase_rows.groupby(
        ["probe", "network"], sort=True
    ):
        eligible = group[_bool_series(group["phase_interpretable"])]
        if len(eligible) >= 3:
            correlation, p_value = stats.spearmanr(
                eligible["absolute_phase_shift_degrees"],
                eligible["fc_loss_z"],
            )
        else:
            correlation, p_value = np.nan, np.nan
        summary_rows.append(
            {
                "probe": probe,
                "network": network,
                "all_rows": len(group),
                "phase_interpretable_rows": len(eligible),
                "phase_interpretable_fraction": len(eligible) / len(group),
                "spearman_phase_shift_vs_fc_loss": float(correlation),
                "descriptive_p_value": float(p_value),
                "median_phase_shift_degrees": float(
                    eligible["absolute_phase_shift_degrees"].median()
                )
                if len(eligible)
                else np.nan,
                "median_fc_loss_z": float(eligible["fc_loss_z"].median())
                if len(eligible)
                else np.nan,
            }
        )
    return {
        "phase_fc_rows": phase_interpretable.reset_index(drop=True),
        "phase_fc_summary": pd.DataFrame(summary_rows),
    }


def _network_power(
    power: np.ndarray,
    labels: np.ndarray,
    membership: np.ndarray,
) -> np.ndarray:
    values = []
    for hemisphere in ("L", "R", "M"):
        selected = np.asarray(membership, dtype=bool) & np.array(
            [_label_hemisphere(label) == hemisphere for label in labels]
        )
        if selected.any():
            values.append(np.mean(power[:, selected], axis=1))
    if not values:
        return np.full(power.shape[0], np.nan)
    return np.mean(np.stack(values, axis=1), axis=1)


def build_trace_audits(data_dir: Path) -> dict[str, pd.DataFrame]:
    manifest = _read_csv(data_dir, "main_parcel_trace_manifest.csv")
    if len(manifest) != EXPECTED_TRACE_SHARDS:
        raise PosthocAnalysisError(
            f"Expected {EXPECTED_TRACE_SHARDS} trace shards; found {len(manifest)}."
        )
    node = _read_csv(data_dir, "main_node_metrics.csv")
    pulse_validity = node[node["probe"] == "pulse"][
        ["severity", "seed", "region_index", "latency_valid"]
    ].copy()
    pulse_validity["latency_valid"] = _bool_series(
        pulse_validity["latency_valid"]
    )
    frequency_validity = node[node["probe"].isin(PERIODIC_PROBES)][
        ["severity", "seed", "probe", "region_index", "frequency_qa_valid"]
    ].copy()
    frequency_validity["frequency_qa_valid"] = _bool_series(
        frequency_validity["frequency_qa_valid"]
    )
    frequency_valid_map = frequency_validity.set_index(
        ["severity", "seed", "probe", "region_index"]
    )["frequency_qa_valid"].to_dict()

    segment_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    pulse_region_rows: list[dict[str, Any]] = []
    for ordinal, manifest_row in enumerate(manifest.itertuples(index=False), start=1):
        path = data_dir / str(manifest_row.trace_file)
        if not path.is_file():
            raise PosthocAnalysisError(f"Missing trace shard: {path}")
        archive = _read_trace(path)
        time_ms = np.asarray(archive["time_ms"], dtype=float)
        labels = np.asarray(archive["region_labels"]).astype(str)
        region_indices = np.asarray(archive["region_indices"], dtype=int)
        if len(labels) != EXPECTED_TRACE_REGIONS:
            raise PosthocAnalysisError(
                f"{path.name} contains {len(labels)} regions, expected 34."
            )
        severity = float(np.asarray(archive["severity"]).item())
        seed = int(np.asarray(archive["seed"]).item())
        probe = str(np.asarray(archive["probe"]).item())
        evoked = np.asarray(archive["evoked_psp"], dtype=float)

        if probe in PERIODIC_PROBES:
            full_window = (time_ms >= PERIODIC_ANALYSIS_START_MS) & (
                time_ms < PERIODIC_ANALYSIS_END_MS
            )
            segment_phase_values: dict[str, list[np.ndarray]] = {
                network: [] for network in PRIMARY_NETWORKS
            }
            provisional: list[dict[str, Any]] = []
            for segment_index in range(PERIODIC_SEGMENT_COUNT):
                start_ms = (
                    PERIODIC_ANALYSIS_START_MS
                    + segment_index * PERIODIC_SEGMENT_MS
                )
                window = (time_ms >= start_ms) & (
                    time_ms < start_ms + PERIODIC_SEGMENT_MS
                )
                broadband_response = detrended_ac_rms(evoked, window)
                fit = exact_frequency_fit(time_ms, evoked, probe, window)
                broadband_reference = _ipsilateral_reference(
                    broadband_response, labels
                )
                locked_reference = _ipsilateral_reference(
                    fit["amplitude"], labels
                )
                broadband_transfer = broadband_response / np.maximum(
                    broadband_reference, 1e-30
                )
                locked_transfer = fit["amplitude"] / np.maximum(
                    locked_reference, 1e-30
                )
                phase_lag = _phase_lag_to_ipsilateral_a1(
                    fit["phase_rad"], labels
                )
                valid = np.array(
                    [
                        bool(
                            frequency_valid_map.get(
                                (severity, seed, probe, int(region_index)), False
                            )
                        )
                        for region_index in region_indices
                    ],
                    dtype=bool,
                )
                for network in PRIMARY_NETWORKS:
                    membership = np.asarray(
                        archive[NETWORK_MEMBERSHIP_KEYS[network]], dtype=bool
                    )
                    target_valid = membership & valid
                    segment_phase_values[network].append(phase_lag)
                    provisional.append(
                        {
                            "severity": severity,
                            "seed": seed,
                            "probe": probe,
                            "network": NETWORK_LABELS[network],
                            "segment": segment_index + 1,
                            "segment_start_s": float(2 * segment_index),
                            "segment_end_s": float(2 * (segment_index + 1)),
                            "broadband_transfer": _aggregate_node_metric(
                                broadband_transfer,
                                labels,
                                membership,
                                "transfer",
                            ),
                            "frequency_locked_transfer": _aggregate_node_metric(
                                locked_transfer,
                                labels,
                                membership,
                                "transfer",
                            ),
                            "target_locked_amplitude": _aggregate_node_metric(
                                fit["amplitude"], labels, membership, "amplitude"
                            ),
                            "ipsilateral_a1_locked_amplitude": _aggregate_node_metric(
                                locked_reference, labels, membership, "amplitude"
                            ),
                            "median_segment_fit_r_squared": float(
                                np.nanmedian(fit["r_squared"][membership])
                            ),
                            "phase_eligible_fraction": float(
                                np.mean(valid[membership])
                            ),
                            "phase_lag_degrees": float(
                                np.degrees(
                                    np.angle(np.mean(np.exp(1j * phase_lag[target_valid])))
                                )
                            )
                            if target_valid.any()
                            else np.nan,
                        }
                    )
            for row in provisional:
                network_key = (
                    "semantic_expanded"
                    if row["network"] == "Semantic proxy"
                    else "episodic_expanded"
                )
                membership = np.asarray(
                    archive[NETWORK_MEMBERSHIP_KEYS[network_key]], dtype=bool
                )
                phases = np.stack(segment_phase_values[network_key], axis=0)
                across_segment_consistency = np.abs(
                    np.mean(np.exp(1j * phases), axis=0)
                )
                valid = np.array(
                    [
                        bool(
                            frequency_valid_map.get(
                                (severity, seed, probe, int(region_index)), False
                            )
                        )
                        for region_index in region_indices
                    ],
                    dtype=bool,
                )
                selected = membership & valid
                row["median_phase_consistency_across_segments"] = (
                    float(np.nanmedian(across_segment_consistency[selected]))
                    if selected.any()
                    else np.nan
                )
                segment_rows.append(row)

            frequencies, _ = _multitaper_power(time_ms, evoked, full_window)
            keep_frequency = (frequencies >= SPECTRUM_MIN_HZ) & (
                frequencies <= SPECTRUM_MAX_HZ
            )
            for signal_name, values in (
                ("Stimulated", np.asarray(archive["stimulated_psp"], dtype=float)),
                ("Control", np.asarray(archive["control_psp"], dtype=float)),
                ("Evoked", evoked),
            ):
                _, power = _multitaper_power(time_ms, values, full_window)
                for network in PRIMARY_NETWORKS:
                    membership = np.asarray(
                        archive[NETWORK_MEMBERSHIP_KEYS[network]], dtype=bool
                    )
                    network_power = _network_power(power, labels, membership)
                    for frequency, value in zip(
                        frequencies[keep_frequency],
                        network_power[keep_frequency],
                    ):
                        spectrum_rows.append(
                            {
                                "severity": severity,
                                "seed": seed,
                                "probe": probe,
                                "network": NETWORK_LABELS[network],
                                "signal": signal_name,
                                "frequency_hz": float(frequency),
                                "power": float(value),
                            }
                        )
        elif probe == "pulse":
            quantiles = _pulse_energy_quantiles(time_ms, evoked)
            for index, (region_index, label) in enumerate(
                zip(region_indices, labels)
            ):
                pulse_region_rows.append(
                    {
                        "severity": severity,
                        "seed": seed,
                        "region_index": int(region_index),
                        "region_label": str(label),
                        "hemisphere": _label_hemisphere(str(label)),
                        "semantic_membership": bool(
                            archive["semantic_expanded_membership"][index]
                        ),
                        "episodic_membership": bool(
                            archive["episodic_expanded_membership"][index]
                        ),
                        **{
                            name: float(values[index])
                            for name, values in quantiles.items()
                        },
                    }
                )
        else:
            raise PosthocAnalysisError(f"Unexpected trace probe: {probe}")
        if ordinal % 20 == 0 or ordinal == len(manifest):
            print(
                f"Post-hoc trace audit: {ordinal}/{len(manifest)} shards",
                flush=True,
            )

    segment_trace = pd.DataFrame(segment_rows)
    saved = build_saved_metric_audits(data_dir)["segment_saved"]
    compare = segment_trace.merge(
        saved[
            [
                "severity",
                "seed",
                "probe",
                "network",
                "segment",
                "broadband_transfer",
                "frequency_locked_transfer",
            ]
        ],
        on=["severity", "seed", "probe", "network", "segment"],
        suffixes=("_trace", "_saved"),
        validate="one_to_one",
    )
    broadband_error = np.max(
        np.abs(compare["broadband_transfer_trace"] - compare["broadband_transfer_saved"])
    )
    locked_error = np.max(
        np.abs(
            compare["frequency_locked_transfer_trace"]
            - compare["frequency_locked_transfer_saved"]
        )
    )
    if broadband_error > 1e-9 or locked_error > 1e-9:
        raise PosthocAnalysisError(
            "Trace recomputation did not reproduce saved segment transfer values: "
            f"broadband max error={broadband_error:.3g}, "
            f"locked max error={locked_error:.3g}."
        )

    segment_trace_summary = segment_trace.groupby(
        ["severity", "probe", "network", "segment"], as_index=False
    ).agg(
        median_broadband_transfer=("broadband_transfer", "median"),
        median_frequency_locked_transfer=(
            "frequency_locked_transfer",
            "median",
        ),
        median_target_locked_amplitude=("target_locked_amplitude", "median"),
        median_ipsilateral_a1_locked_amplitude=(
            "ipsilateral_a1_locked_amplitude",
            "median",
        ),
        median_segment_fit_r_squared=(
            "median_segment_fit_r_squared",
            "median",
        ),
        median_phase_eligible_fraction=("phase_eligible_fraction", "median"),
        median_phase_lag_degrees=("phase_lag_degrees", "median"),
        median_phase_consistency_across_segments=(
            "median_phase_consistency_across_segments",
            "median",
        ),
        numerical_initializations=("seed", "nunique"),
    )
    segment_trace_summary["severity_label"] = segment_trace_summary[
        "severity"
    ].map(lambda value: f"{value:.1f}")
    segment_trace_summary["series"] = (
        segment_trace_summary["probe"]
        + " · "
        + segment_trace_summary["network"]
    )

    spectra_per_seed = pd.DataFrame(spectrum_rows)
    spectra_summary = spectra_per_seed.groupby(
        ["severity", "probe", "network", "signal", "frequency_hz"],
        as_index=False,
    ).agg(
        median_power=("power", "median"),
        minimum_power=("power", "min"),
        maximum_power=("power", "max"),
        numerical_initializations=("seed", "nunique"),
    )
    spectra_summary["log10_median_power"] = np.log10(
        np.maximum(spectra_summary["median_power"], 1e-30)
    )
    spectra_summary["severity_label"] = spectra_summary["severity"].map(
        lambda value: f"{value:.1f}"
    )
    spectra_summary["series"] = (
        spectra_summary["signal"] + " · " + spectra_summary["network"]
    )
    spectral_peak_rows: list[dict[str, Any]] = []
    for keys, group in spectra_summary.groupby(
        ["severity", "probe", "network", "signal"], sort=True
    ):
        severity, probe, network, signal_name = keys
        peak = group.loc[group["median_power"].idxmax()]
        drive_frequency = {"2Hz": 2.0, "5Hz": 5.0}[str(probe)]
        drive = group[np.isclose(group["frequency_hz"], drive_frequency)]
        if len(drive) != 1:
            raise PosthocAnalysisError(
                f"Expected one {drive_frequency:g} Hz spectral bin."
            )
        drive_power = float(drive.iloc[0]["median_power"])
        peak_power = float(peak["median_power"])
        spectral_peak_rows.append(
            {
                "severity": float(severity),
                "severity_label": f"{float(severity):.1f}",
                "probe": str(probe),
                "network": str(network),
                "signal": str(signal_name),
                "peak_frequency_hz": float(peak["frequency_hz"]),
                "peak_power": peak_power,
                "drive_frequency_hz": drive_frequency,
                "drive_power": drive_power,
                "drive_to_peak_power_ratio": drive_power
                / max(peak_power, 1e-30),
            }
        )
    spectral_peak_summary = pd.DataFrame(spectral_peak_rows)

    pulse_region = pd.DataFrame(pulse_region_rows)
    fixed_mask_rows: list[dict[str, Any]] = []
    pulse_seed_rows: list[dict[str, Any]] = []
    for network, membership_column in (
        ("Semantic proxy", "semantic_membership"),
        ("Episodic proxy", "episodic_membership"),
    ):
        network_regions = pulse_region[pulse_region[membership_column]][
            ["region_index", "region_label"]
        ].drop_duplicates()
        for seed in sorted(pulse_region["seed"].unique()):
            validity = pulse_validity[
                (pulse_validity["seed"] == seed)
                & pulse_validity["region_index"].isin(network_regions["region_index"])
            ]
            fixed = validity.groupby("region_index")["latency_valid"].all()
            fixed_indices = set(fixed[fixed].index.astype(int))
            fixed_labels = network_regions[
                network_regions["region_index"].isin(fixed_indices)
            ]["region_label"].astype(str)
            fixed_mask_rows.append(
                {
                    "seed": int(seed),
                    "network": network,
                    "declared_parcels": len(network_regions),
                    "fixed_valid_parcels": len(fixed_indices),
                    "fixed_valid_fraction": len(fixed_indices) / len(network_regions),
                    "fixed_region_labels": ", ".join(sorted(fixed_labels)),
                    "usable_seed": bool(fixed_indices),
                }
            )
            for severity in SEVERITIES:
                selected = pulse_region[
                    (pulse_region["seed"] == seed)
                    & (pulse_region["severity"] == severity)
                    & pulse_region["region_index"].isin(fixed_indices)
                ].copy()
                if selected.empty:
                    pulse_seed_rows.append(
                        {
                            "severity": severity,
                            "severity_label": f"{severity:.1f}",
                            "seed": int(seed),
                            "network": network,
                            "fixed_parcel_count": 0,
                        }
                    )
                    continue
                labels = selected["region_label"].to_numpy(dtype=str)
                membership = np.ones(len(selected), dtype=bool)
                row: dict[str, Any] = {
                    "severity": severity,
                    "severity_label": f"{severity:.1f}",
                    "seed": int(seed),
                    "network": network,
                    "fixed_parcel_count": len(selected),
                }
                timing_metrics = (
                    "t20_ms",
                    "t50_ms",
                    "t80_ms",
                    "absolute_peak_after_onset_ms",
                )
                trace_scope = pulse_region[
                    (pulse_region["seed"] == seed)
                    & (pulse_region["severity"] == severity)
                ]
                for metric in timing_metrics:
                    values = selected[metric].to_numpy(dtype=float)
                    a1_values = {
                        hemisphere: float(
                            trace_scope[
                                trace_scope["region_label"]
                                == f"{hemisphere}_A1"
                            ][metric].iloc[0]
                        )
                        for hemisphere in ("L", "R")
                    }
                    bilateral_a1 = float(np.mean(list(a1_values.values())))
                    reference = np.array(
                        [
                            a1_values.get(_label_hemisphere(label), bilateral_a1)
                            for label in labels
                        ],
                        dtype=float,
                    )
                    # Raw times are relative to stimulus onset. Relative times
                    # subtract the ipsilateral A1 value using the notebook rule.
                    row[metric] = _aggregate_node_metric(
                        values, labels, membership, "latency"
                    )
                    row[f"relative_{metric}"] = _aggregate_node_metric(
                        values - reference, labels, membership, "latency"
                    )
                row["spread_t80_t20_ms"] = _aggregate_node_metric(
                    selected["spread_t80_t20_ms"].to_numpy(dtype=float),
                    labels,
                    membership,
                    "latency",
                )
                for metric in (
                    "total_energy_psp2_ms",
                    "tail_energy_fraction",
                    "absolute_peak_psp",
                ):
                    row[metric] = _aggregate_node_metric(
                        selected[metric].to_numpy(dtype=float),
                        labels,
                        membership,
                        "amplitude",
                    )
                pulse_seed_rows.append(row)
    pulse_fixed_masks = pd.DataFrame(fixed_mask_rows)
    pulse_fixed_seed = pd.DataFrame(pulse_seed_rows)
    usable = pulse_fixed_seed[pulse_fixed_seed["fixed_parcel_count"] > 0]
    summary_rows: list[dict[str, Any]] = []
    for (severity, network), group in usable.groupby(
        ["severity", "network"], sort=True
    ):
        row = {
            "severity": float(severity),
            "severity_label": f"{float(severity):.1f}",
            "network": network,
            "usable_seeds": int(group["seed"].nunique()),
            "median_fixed_parcel_count": float(group["fixed_parcel_count"].median()),
            "minimum_fixed_parcel_count": int(group["fixed_parcel_count"].min()),
        }
        for metric in (
            "t20_ms",
            "t50_ms",
            "t80_ms",
            "relative_t20_ms",
            "relative_t50_ms",
            "relative_t80_ms",
            "spread_t80_t20_ms",
            "absolute_peak_after_onset_ms",
            "relative_absolute_peak_after_onset_ms",
            "total_energy_psp2_ms",
            "tail_energy_fraction",
            "absolute_peak_psp",
        ):
            row[f"median_{metric}"] = float(group[metric].median())
            row[f"minimum_{metric}"] = float(group[metric].min())
            row[f"maximum_{metric}"] = float(group[metric].max())
        summary_rows.append(row)
    pulse_fixed_summary = pd.DataFrame(summary_rows)
    return {
        "segment_trace_audit": segment_trace,
        "segment_trace_summary": segment_trace_summary,
        "spectra_summary": spectra_summary,
        "spectral_peak_summary": spectral_peak_summary,
        "pulse_fixed_masks": pulse_fixed_masks,
        "pulse_fixed_seed_metrics": pulse_fixed_seed,
        "pulse_fixed_summary": pulse_fixed_summary,
        "pulse_region_quantiles": pulse_region,
    }


def build_regional_covariate_audit(data_dir: Path) -> dict[str, pd.DataFrame]:
    node = _read_csv(data_dir, "main_node_metrics.csv")
    regional = _read_csv(data_dir, "regional_features.csv")
    periodic = node[
        node["probe"].isin(PERIODIC_PROBES)
        & node["severity"].isin((0.0, 1.0))
    ][
        ["severity", "seed", "probe", "region_index", "node_transfer"]
    ].copy()
    base = periodic[periodic["severity"] == 0.0][
        ["seed", "probe", "region_index", "node_transfer"]
    ].rename(columns={"node_transfer": "baseline_node_transfer"})
    high = periodic[periodic["severity"] == 1.0].merge(
        base, on=["seed", "probe", "region_index"], validate="one_to_one"
    )
    high["log2_transfer_change"] = np.log2(
        high["node_transfer"] / high["baseline_node_transfer"]
    )
    per_region = high.groupby(["probe", "region_index"], as_index=False).agg(
        mean_log2_transfer_change=("log2_transfer_change", "mean"),
        median_log2_transfer_change=("log2_transfer_change", "median"),
        minimum_log2_transfer_change=("log2_transfer_change", "min"),
        maximum_log2_transfer_change=("log2_transfer_change", "max"),
        numerical_initializations=("seed", "nunique"),
    )
    columns = [
        "region_index",
        "region_label",
        "hemisphere",
        "b_reduction",
        "bilateral_a1_affinity",
        "weighted_structural_strength",
        "semantic_expanded_membership",
        "episodic_expanded_membership",
    ]
    per_region = per_region.merge(
        regional[columns], on="region_index", validate="many_to_one"
    )
    semantic = _bool_series(per_region["semantic_expanded_membership"])
    episodic = _bool_series(per_region["episodic_expanded_membership"])
    per_region["proxy_membership"] = np.select(
        [semantic & episodic, semantic, episodic],
        ["Both", "Semantic proxy", "Episodic proxy"],
        default="Neither",
    )

    coefficient_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for probe in PERIODIC_PROBES:
        frame = per_region[
            (per_region["probe"] == probe)
            & per_region["proxy_membership"].isin(
                ["Semantic proxy", "Episodic proxy"]
            )
        ].copy()
        frame["semantic_indicator"] = (
            frame["proxy_membership"] == "Semantic proxy"
        ).astype(float)
        frame["right_hemisphere"] = (frame["hemisphere"] == "R").astype(float)
        continuous = [
            "b_reduction",
            "bilateral_a1_affinity",
            "weighted_structural_strength",
        ]
        standardized = []
        for column in continuous:
            sd = float(frame[column].std(ddof=0))
            if sd <= 0:
                raise PosthocAnalysisError(
                    f"Cannot standardize invariant regional predictor {column}."
                )
            name = f"z_{column}"
            frame[name] = (frame[column] - frame[column].mean()) / sd
            standardized.append(name)
        terms = ["semantic_indicator", *standardized, "right_hemisphere"]
        x = np.column_stack(
            [np.ones(len(frame)), *[frame[column].to_numpy(float) for column in terms]]
        )
        y = frame["mean_log2_transfer_change"].to_numpy(float)
        coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coefficients
        degrees_freedom = len(y) - x.shape[1]
        residual_variance = float(np.sum(residual**2) / degrees_freedom)
        covariance = residual_variance * np.linalg.inv(x.T @ x)
        standard_errors = np.sqrt(np.diag(covariance))
        t_values = coefficients / standard_errors
        p_values = 2.0 * stats.t.sf(np.abs(t_values), degrees_freedom)
        t_critical = float(stats.t.ppf(0.975, degrees_freedom))
        term_labels = [
            "Intercept",
            "Semantic vs episodic proxy",
            "b reduction (1 SD)",
            "Bilateral A1 affinity (1 SD)",
            "Weighted structural strength (1 SD)",
            "Right vs left hemisphere",
        ]
        for term, label, estimate, se, p_value in zip(
            ["intercept", *terms],
            term_labels,
            coefficients,
            standard_errors,
            p_values,
        ):
            coefficient_rows.append(
                {
                    "probe": probe,
                    "term": term,
                    "term_label": label,
                    "estimate": float(estimate),
                    "standard_error": float(se),
                    "ci95_lower": float(estimate - t_critical * se),
                    "ci95_upper": float(estimate + t_critical * se),
                    "descriptive_p_value": float(p_value),
                    "parcel_count": len(frame),
                }
            )
        total_ss = float(np.sum((y - np.mean(y)) ** 2))
        residual_ss = float(np.sum(residual**2))
        r_squared = 1.0 - residual_ss / total_ss
        adjusted = 1.0 - (1.0 - r_squared) * (len(y) - 1) / degrees_freedom
        model_rows.append(
            {
                "probe": probe,
                "parcel_count": len(frame),
                "rank": int(rank),
                "predictor_count_excluding_intercept": len(terms),
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted,
                "residual_degrees_freedom": degrees_freedom,
                "analysis_status": "Exploratory parcel-level OLS; numerical seeds averaged within parcel",
            }
        )
    return {
        "regional_response_covariates": per_region,
        "regional_covariate_coefficients": pd.DataFrame(coefficient_rows),
        "regional_covariate_models": pd.DataFrame(model_rows),
    }


def build_analysis(
    data_dir: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    data_dir = data_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}
    outputs.update(build_saved_metric_audits(data_dir))
    outputs.update(build_phase_fc_audit(data_dir))
    outputs.update(build_trace_audits(data_dir))
    outputs.update(build_regional_covariate_audit(data_dir))
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    metadata = {
        "analysis": "recommended-posthoc-v1",
        "source_data_dir": str(data_dir),
        "notebook_changed": False,
        "tvb_simulations_run": 0,
        "trace_shards_read": EXPECTED_TRACE_SHARDS,
        "formulas": {
            "broadband": "Notebook detrended AC RMS; target / ipsilateral A1",
            "frequency_locked": "Notebook sine/cosine/intercept/trend fit; target amplitude / ipsilateral A1 amplitude",
            "phase": "Notebook wrapped target minus ipsilateral-A1 harmonic phase",
            "spectrum": "Notebook DPSS multitaper parameters NW=3, K=5, generalized to the displayed spectrum",
            "pulse_quantiles": "Notebook cumulative squared-response t50 method extended to t20 and t80",
        },
        "interpretation": [
            "Phase rows are retained only when the notebook frequency-QA flag passes at baseline and high severity.",
            "Fixed pulse masks use the original latency_valid flag across all three severities within each seed.",
            "Regional OLS is exploratory and parcel-level; seeds are averaged within parcel.",
        ],
        "outputs": {name: len(frame) for name, frame in outputs.items()},
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-hoc transmission, FC, pulse, and covariate audits."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = build_analysis(
            args.data_dir.expanduser().resolve(),
            args.output_dir.expanduser().resolve(),
        )
        print(f"Post-hoc outputs: {args.output_dir.expanduser().resolve()}")
        print(
            "Rows: "
            + ", ".join(f"{name}={len(frame)}" for name, frame in outputs.items())
        )
        print("TVB simulations run: 0")
        return 0
    except (OSError, ValueError, PosthocAnalysisError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
