"""Response metrics and deterministic table transformations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def harmonic_amplitude(
    time_ms: np.ndarray,
    evoked: np.ndarray,
    probe: str,
    *,
    analysis_start_ms: float,
    simulation_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the exact probe frequency with an intercept and linear trend."""

    frequency_hz = {"2Hz": 2.0, "5Hz": 5.0}[probe]
    window = (time_ms >= analysis_start_ms) & (time_ms <= simulation_ms)
    t_seconds = (time_ms[window] - analysis_start_ms) / 1000.0
    y = np.asarray(evoked[window], dtype=float)
    if len(t_seconds) < 100:
        raise RuntimeError("Periodic analysis window is unexpectedly short.")

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
    amplitude = np.sqrt(coefficients[0] ** 2 + coefficients[1] ** 2)

    residual_ss = np.sum((y - fitted) ** 2, axis=0)
    total_ss = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    r_squared = 1.0 - residual_ss / np.maximum(total_ss, 1e-15)
    return amplitude, r_squared


def pulse_rms(
    time_ms: np.ndarray,
    evoked: np.ndarray,
    *,
    onset_ms: float,
    analysis_end_ms: float,
    n_regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return node-wise RMS response in the prespecified pulse window."""

    window = (time_ms >= onset_ms) & (time_ms <= analysis_end_ms)
    y = np.asarray(evoked[window], dtype=float)
    if y.shape[0] < 100:
        raise RuntimeError("Pulse analysis window is unexpectedly short.")
    response = np.sqrt(np.mean(y**2, axis=0))
    return response, np.full(n_regions, np.nan)


def extract_node_response(
    time_ms: np.ndarray,
    evoked: np.ndarray,
    probe: str,
    *,
    periodic_analysis_start_ms: float,
    simulation_ms: float,
    stimulus_onset_ms: float,
    pulse_analysis_end_ms: float,
    n_regions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the pulse or exact-frequency response metric."""

    if probe == "pulse":
        return pulse_rms(
            time_ms,
            evoked,
            onset_ms=stimulus_onset_ms,
            analysis_end_ms=pulse_analysis_end_ms,
            n_regions=n_regions,
        )
    return harmonic_amplitude(
        time_ms,
        evoked,
        probe,
        analysis_start_ms=periodic_analysis_start_ms,
        simulation_ms=simulation_ms,
    )


def b_signature(values: np.ndarray) -> str:
    """Stable short signature for a regional inhibitory-rate vector."""

    return hashlib.sha256(
        np.asarray(values, dtype=np.float64).tobytes()
    ).hexdigest()[:16]


def normalize_to_baseline(
    network_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Log2-normalize each target transfer to the matching baseline."""

    target = network_df.copy()
    source = target if baseline_df is None else baseline_df.copy()
    baseline = source[source["severity"] == 0.0][
        ["seed", "probe", "network", "transfer"]
    ].rename(columns={"transfer": "baseline_transfer"})
    target = target.merge(
        baseline,
        on=["seed", "probe", "network"],
        how="left",
        validate="many_to_one",
    )
    if target["baseline_transfer"].isna().any():
        raise RuntimeError("A baseline transfer value is missing.")
    target["log2_transfer_vs_baseline"] = np.log2(
        np.maximum(target["transfer"], 1e-15)
        / np.maximum(target["baseline_transfer"], 1e-15)
    )
    return target


def make_contrasts(normalized_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the music-minus-speech change at each experiment point."""

    pivot = normalized_df.pivot_table(
        index=[
            "scope",
            "variant",
            "condition",
            "severity",
            "seed",
            "probe",
            "global_coupling",
            "input_peak_per_ms",
            "dt_ms",
        ],
        columns="network",
        values="log2_transfer_vs_baseline",
    ).reset_index()
    if not {"music", "speech"}.issubset(pivot.columns):
        raise RuntimeError("Both network results are required for the contrast.")
    pivot.columns.name = None
    pivot["music_minus_speech_log2_change"] = (
        pivot["music"] - pivot["speech"]
    )
    return pivot


def node_response_vector(
    node_df: pd.DataFrame,
    seed: int,
    probe: str,
    severity: float,
    *,
    n_regions: int,
) -> np.ndarray:
    """Extract one complete node-response vector from a long-form table."""

    subset = node_df[
        (node_df["seed"] == seed)
        & (node_df["probe"] == probe)
        & (node_df["severity"] == severity)
    ].sort_values("region_index")
    if len(subset) != n_regions:
        raise RuntimeError("Expected exactly one response per region.")
    return subset["response"].to_numpy()


def a1_response_value(
    network_df: pd.DataFrame,
    seed: int,
    probe: str,
    severity: float,
) -> float:
    """Return the unique bilateral-A1 response for a grid point."""

    subset = network_df[
        (network_df["seed"] == seed)
        & (network_df["probe"] == probe)
        & (network_df["severity"] == severity)
    ]
    values = subset["a1_response"].unique()
    if len(values) != 1:
        raise RuntimeError("Expected one A1 response value.")
    return float(values[0])


def same_hemisphere_candidates(target_index: int) -> np.ndarray:
    """Return cortical candidates in the target parcel's hemisphere."""

    if target_index < 180:
        return np.arange(0, 180)
    if target_index < 360:
        return np.arange(180, 360)
    raise ValueError("Matched targets must be cortical.")


def draw_matched_set(
    target_indices: Sequence[int],
    rng: np.random.Generator,
    *,
    reserved: set[int],
    matching_z: np.ndarray,
    top_k: int = 30,
) -> tuple[np.ndarray, list[float]]:
    """Draw one topology/pathology-matched control set."""

    selected: list[int] = []
    distances_selected: list[float] = []
    for target_index_raw in target_indices:
        target_index = int(target_index_raw)
        candidates = same_hemisphere_candidates(target_index)
        candidates = np.array(
            [
                index
                for index in candidates
                if index not in reserved and index not in selected
            ],
            dtype=int,
        )
        distances = np.linalg.norm(
            matching_z[candidates] - matching_z[target_index], axis=1
        )
        order = np.argsort(distances)
        pool_order = order[: min(top_k, len(order))]
        pool = candidates[pool_order]
        pool_distances = distances[pool_order]
        scale = max(float(np.median(pool_distances)), 1e-6)
        probabilities = np.exp(
            -(pool_distances - pool_distances.min()) / scale
        )
        probabilities /= probabilities.sum()
        chosen = int(rng.choice(pool, p=probabilities))
        selected.append(chosen)
        distances_selected.append(
            float(
                np.linalg.norm(
                    matching_z[chosen] - matching_z[target_index]
                )
            )
        )
    return np.array(selected, dtype=int), distances_selected


def sorted_concat(
    frames: Sequence[pd.DataFrame],
    *,
    sort_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Concatenate checkpoint frames in a deterministic order."""

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    columns = [column for column in (sort_by or ()) if column in combined]
    if columns:
        combined = combined.sort_values(columns, kind="stable").reset_index(
            drop=True
        )
    return combined


def fingerprint_mapping(mapping: Mapping[str, object]) -> str:
    """Return a stable digest for a small JSON-compatible mapping."""

    import json

    payload = json.dumps(
        mapping, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
