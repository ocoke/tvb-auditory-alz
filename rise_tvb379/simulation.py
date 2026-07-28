"""TVB model construction and resumable condition/seed work blocks."""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tvb.simulator.lab import (
    connectivity,
    coupling,
    equations,
    integrators,
    models,
    monitors,
    patterns,
    simulator,
)

from .metrics import b_signature, extract_node_response

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationContext:
    """All immutable arrays and timing constants needed for a simulation."""

    weights: np.ndarray
    labels: np.ndarray
    a1_indices: np.ndarray
    music_indices: np.ndarray
    speech_indices: np.ndarray
    n_regions: int
    monitor_period_ms: float
    stimulus_onset_ms: float
    periodic_analysis_start_ms: float
    pulse_width_ms: float
    pulse_analysis_end_ms: float
    default_simulation_ms: float

    @property
    def network_indices(self) -> dict[str, np.ndarray]:
        return {
            "music": self.music_indices,
            "speech": self.speech_indices,
        }


@dataclass
class BlockResult:
    """Tables produced by one condition and numerical-seed block."""

    node: pd.DataFrame
    network: pd.DataFrame
    manifest: pd.DataFrame


def fresh_connectivity(context: SimulationContext) -> connectivity.Connectivity:
    """Build a new zero-delay TVB connectivity object."""

    return connectivity.Connectivity(
        weights=context.weights.copy(),
        tract_lengths=np.zeros_like(context.weights),
        centres=np.zeros((context.n_regions, 3), dtype=float),
        region_labels=context.labels.copy(),
        speed=np.array([100.0]),
    )


def build_model(b_values: np.ndarray) -> models.JansenRit:
    """Create the Stefanovski-style regional Jansen-Rit model."""

    background = np.array([0.1085])
    model = models.JansenRit(
        v0=np.array([6.0]),
        mu=background,
        p_min=background,
        p_max=background,
        b=np.asarray(b_values, dtype=float),
        variables_of_interest=("y1", "y2"),
    )
    # External p(t) enters the y4 derivative in TVB's Jansen-Rit model.
    model.stvar = np.array([4], dtype=np.int32)
    return model


def make_temporal_equation(
    probe: str,
    model: models.JansenRit,
    input_peak_per_ms: float,
    *,
    onset_ms: float,
    offset_ms: float,
    pulse_width_ms: float,
) -> equations.TemporalApplicableEquation:
    """Build the pulse or sinusoidal A1 drive used by the notebook."""

    derivative_peak = float(
        model.A[0] * model.a[0] * float(input_peak_per_ms)
    )
    if probe == "pulse":
        return equations.TemporalApplicableEquation(
            equation="where((var >= onset) & (var < onset + width), amp, 0.0)",
            parameters={
                "onset": float(onset_ms),
                "width": float(pulse_width_ms),
                "amp": derivative_peak,
            },
        )

    frequency_per_ms = {"2Hz": 0.002, "5Hz": 0.005}[probe]
    return equations.TemporalApplicableEquation(
        equation=(
            "where((var >= onset) & (var <= offset), "
            "0.5 * amp * (1.0 + sin(6.283185307179586 * "
            "frequency * (var - onset))), 0.0)"
        ),
        parameters={
            "onset": float(onset_ms),
            "offset": float(offset_ms),
            "amp": derivative_peak,
            "frequency": frequency_per_ms,
        },
    )


def make_initial_conditions(seed: int, n_regions: int) -> np.ndarray:
    """Generate the deterministic one-sample state history."""

    rng = np.random.default_rng(int(seed))
    return rng.random((1, 6, n_regions, 1))


def run_tvb(
    context: SimulationContext,
    *,
    b_values: np.ndarray,
    probe: str | None,
    global_coupling: float,
    input_peak_per_ms: float,
    seed: int,
    dt_ms: float,
    simulation_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one deterministic TVB simulation and return PSP time series."""

    simulation_ms = float(
        context.default_simulation_ms
        if simulation_ms is None
        else simulation_ms
    )
    b_values = np.asarray(b_values, dtype=float)
    if b_values.shape != (context.n_regions,) or not np.isfinite(
        b_values
    ).all():
        raise ValueError(
            f"b_values must be a finite {context.n_regions}-element vector."
        )
    if probe not in {None, "pulse", "2Hz", "5Hz"}:
        raise ValueError(f"Unknown probe: {probe}")
    ratio = context.monitor_period_ms / float(dt_ms)
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("Monitor period must be an integer multiple of dt.")

    white_matter = fresh_connectivity(context)
    model = build_model(b_values)
    stimulus = None
    if probe is not None:
        regional_weights = np.zeros(context.n_regions, dtype=float)
        regional_weights[context.a1_indices] = 1.0 / np.sqrt(
            len(context.a1_indices)
        )
        stimulus = patterns.StimuliRegion(
            temporal=make_temporal_equation(
                probe,
                model,
                input_peak_per_ms,
                onset_ms=context.stimulus_onset_ms,
                offset_ms=simulation_ms,
                pulse_width_ms=context.pulse_width_ms,
            ),
            connectivity=white_matter,
            weight=regional_weights,
        )

    experiment = simulator.Simulator(
        connectivity=white_matter,
        model=model,
        coupling=coupling.SigmoidalJansenRit(
            a=np.array([float(global_coupling)])
        ),
        integrator=integrators.HeunDeterministic(dt=float(dt_ms)),
        monitors=(monitors.SubSample(period=context.monitor_period_ms),),
        stimulus=stimulus,
        initial_conditions=make_initial_conditions(seed, context.n_regions),
    )
    experiment.configure()

    started = time.perf_counter()
    (time_ms, raw), = experiment.run(simulation_length=simulation_ms)
    wall_seconds = time.perf_counter() - started
    psp = raw[:, 0, :, 0] - raw[:, 1, :, 0]
    time_ms = np.asarray(time_ms, dtype=float)
    psp = np.asarray(psp, dtype=float)

    if psp.shape[1] != context.n_regions:
        raise RuntimeError(f"Unexpected TVB output shape: {psp.shape}")
    if not np.isfinite(psp).all():
        raise RuntimeError("TVB produced NaN or infinite values.")
    if float(np.max(np.abs(psp))) > 100.0:
        raise RuntimeError(
            "TVB activity exceeded the prespecified safety bound of 100."
        )
    return time_ms, psp, wall_seconds


def run_condition_seed_block(
    context: SimulationContext,
    *,
    scope: str,
    condition: dict[str, Any],
    seed: int,
    probes: tuple[str, ...] | list[str],
    global_coupling: float,
    input_peak_per_ms: float,
    dt_ms: float,
    simulation_ms: float | None = None,
) -> BlockResult:
    """Run a matched control and every probe for one condition/seed block."""

    simulation_ms = float(
        context.default_simulation_ms
        if simulation_ms is None
        else simulation_ms
    )
    severity = float(condition["severity"])
    condition_name = str(condition["condition"])
    b_values = np.asarray(condition["b_values"], dtype=float)
    variant = str(condition.get("variant", condition_name))
    node_rows: list[dict[str, Any]] = []
    network_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    LOGGER.info(
        "%s: condition=%s seed=%s control", scope, variant, int(seed)
    )
    control_time, control_psp, control_wall = run_tvb(
        context,
        b_values=b_values,
        probe=None,
        global_coupling=global_coupling,
        input_peak_per_ms=input_peak_per_ms,
        seed=seed,
        dt_ms=dt_ms,
        simulation_ms=simulation_ms,
    )
    manifest_rows.append(
        {
            "scope": scope,
            "variant": variant,
            "condition": condition_name,
            "severity": severity,
            "seed": int(seed),
            "probe": "none",
            "simulation_type": "matched_control",
            "global_coupling": float(global_coupling),
            "input_peak_per_ms": float(input_peak_per_ms),
            "dt_ms": float(dt_ms),
            "simulation_ms": simulation_ms,
            "b_signature": b_signature(b_values),
            "wall_seconds": float(control_wall),
            "max_abs_psp": float(np.max(np.abs(control_psp))),
            "max_abs_evoked": np.nan,
        }
    )

    try:
        for probe in probes:
            LOGGER.info(
                "%s: condition=%s seed=%s probe=%s",
                scope,
                variant,
                int(seed),
                probe,
            )
            stimulated_time, stimulated_psp, stimulated_wall = run_tvb(
                context,
                b_values=b_values,
                probe=probe,
                global_coupling=global_coupling,
                input_peak_per_ms=input_peak_per_ms,
                seed=seed,
                dt_ms=dt_ms,
                simulation_ms=simulation_ms,
            )
            if not np.allclose(control_time, stimulated_time):
                raise RuntimeError("Control and stimulated time axes differ.")

            evoked = stimulated_psp - control_psp
            node_response, node_fit_r2 = extract_node_response(
                stimulated_time,
                evoked,
                probe,
                periodic_analysis_start_ms=(
                    context.periodic_analysis_start_ms
                ),
                simulation_ms=simulation_ms,
                stimulus_onset_ms=context.stimulus_onset_ms,
                pulse_analysis_end_ms=context.pulse_analysis_end_ms,
                n_regions=context.n_regions,
            )
            if not np.isfinite(node_response).all():
                raise RuntimeError("A response metric is nonfinite.")

            a1_response = float(
                np.mean(node_response[context.a1_indices])
            )
            if a1_response <= 1e-8:
                raise RuntimeError(
                    "A1 response is too small for stable normalization: "
                    f"{a1_response}"
                )

            metric_name = (
                "pulse_rms" if probe == "pulse" else "harmonic_amplitude"
            )
            common = {
                "scope": scope,
                "variant": variant,
                "condition": condition_name,
                "severity": severity,
                "seed": int(seed),
                "probe": probe,
                "metric": metric_name,
                "global_coupling": float(global_coupling),
                "input_peak_per_ms": float(input_peak_per_ms),
                "dt_ms": float(dt_ms),
                "b_signature": b_signature(b_values),
            }

            for region_index in range(context.n_regions):
                fit_r2 = node_fit_r2[region_index]
                node_rows.append(
                    {
                        **common,
                        "region_index": region_index,
                        "region_label": context.labels[region_index],
                        "response": float(node_response[region_index]),
                        "fit_r_squared": (
                            float(fit_r2) if np.isfinite(fit_r2) else np.nan
                        ),
                        "b_value": float(b_values[region_index]),
                    }
                )

            for network, indices in context.network_indices.items():
                network_response = float(np.mean(node_response[indices]))
                network_rows.append(
                    {
                        **common,
                        "network": network,
                        "network_response": network_response,
                        "a1_response": a1_response,
                        "transfer": network_response / a1_response,
                        "median_target_fit_r_squared": (
                            float(np.nanmedian(node_fit_r2[indices]))
                            if probe != "pulse"
                            else np.nan
                        ),
                    }
                )

            manifest_rows.append(
                {
                    **common,
                    "simulation_type": "stimulated",
                    "simulation_ms": simulation_ms,
                    "wall_seconds": float(stimulated_wall),
                    "max_abs_psp": float(
                        np.max(np.abs(stimulated_psp))
                    ),
                    "max_abs_evoked": float(np.max(np.abs(evoked))),
                }
            )
            del stimulated_psp, evoked
            gc.collect()
    finally:
        del control_psp
        gc.collect()

    return BlockResult(
        node=pd.DataFrame(node_rows),
        network=pd.DataFrame(network_rows),
        manifest=pd.DataFrame(manifest_rows),
    )
