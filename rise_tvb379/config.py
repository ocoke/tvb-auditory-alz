"""Immutable configuration for the RISE TVB 379-region experiment."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

RunMode = Literal["smoke", "pilot", "final"]

N_REGIONS = 379
MAIN_GLOBAL_COUPLING = 60.0
MAIN_INPUT_PEAK_PER_MS = 0.02
MAIN_DT_MS = 1.0
REFERENCE_DT_MS = 0.5
MONITOR_PERIOD_MS = 2.0

STIMULUS_ONSET_MS = 2500.0
SIMULATION_MS = 6500.0
PERIODIC_ANALYSIS_START_MS = 3500.0
PULSE_WIDTH_MS = 100.0
PULSE_ANALYSIS_END_MS = 3500.0

PROBES = ("pulse", "2Hz", "5Hz")
PERIODIC_PROBES = ("2Hz", "5Hz")
DT_CHECK_SEVERITIES = (0.0, 1.0)
DT_CHECK_PROBES = PERIODIC_PROBES
SEVERITY_LABELS = MappingProxyType(
    {
        0.0: "Baseline",
        0.5: "Intermediate AD-like perturbation",
        1.0: "High AD-like perturbation",
    }
)


@dataclass(frozen=True, slots=True)
class SensitivityScenario:
    """One coupling/input-strength sensitivity run."""

    name: str
    global_coupling: float
    input_peak_per_ms: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Sensitivity scenario name must not be empty.")
        if self.global_coupling <= 0:
            raise ValueError("Sensitivity global coupling must be positive.")
        if self.input_peak_per_ms <= 0:
            raise ValueError("Sensitivity input peak must be positive.")


@dataclass(frozen=True, slots=True)
class ModeConfig:
    """The workload controls that differ between run modes."""

    seeds: tuple[int, ...]
    severities: tuple[float, ...]
    calibration_couplings: tuple[float, ...]
    matched_null_sets: int
    spatial_shuffles: int
    sensitivity_scenarios: tuple[SensitivityScenario, ...]

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Mode seeds must be non-empty and unique.")
        if not self.severities or len(set(self.severities)) != len(self.severities):
            raise ValueError("Mode severities must be non-empty and unique.")
        if not {0.0, 1.0}.issubset(self.severities):
            raise ValueError("Every mode must include baseline and high severity.")
        if any(severity not in SEVERITY_LABELS for severity in self.severities):
            raise ValueError("Mode includes an unknown severity.")
        if (
            not self.calibration_couplings
            or len(set(self.calibration_couplings))
            != len(self.calibration_couplings)
            or any(value <= 0 for value in self.calibration_couplings)
        ):
            raise ValueError(
                "Calibration couplings must be non-empty, positive, and unique."
            )
        if self.matched_null_sets <= 0:
            raise ValueError("matched_null_sets must be positive.")
        if self.spatial_shuffles <= 0:
            raise ValueError("spatial_shuffles must be positive.")
        scenario_names = tuple(item.name for item in self.sensitivity_scenarios)
        if len(set(scenario_names)) != len(scenario_names):
            raise ValueError("Sensitivity scenario names must be unique.")


MODE_CONFIGS: Mapping[RunMode, ModeConfig] = MappingProxyType(
    {
        "smoke": ModeConfig(
            seeds=(11,),
            severities=(0.0, 1.0),
            calibration_couplings=(60.0,),
            matched_null_sets=40,
            spatial_shuffles=1,
            sensitivity_scenarios=(
                SensitivityScenario("G30", 30.0, 0.02),
            ),
        ),
        "pilot": ModeConfig(
            seeds=(11, 23),
            severities=(0.0, 0.5, 1.0),
            calibration_couplings=(30.0, 60.0, 100.0),
            matched_null_sets=200,
            spatial_shuffles=2,
            sensitivity_scenarios=(
                SensitivityScenario("G30", 30.0, 0.02),
                SensitivityScenario("G100", 100.0, 0.02),
            ),
        ),
        "final": ModeConfig(
            seeds=(11, 23, 37, 53, 71),
            severities=(0.0, 0.5, 1.0),
            calibration_couplings=(10.0, 30.0, 60.0, 100.0, 200.0, 300.0),
            matched_null_sets=500,
            # The notebook source specifies 100, despite stale output from 5.
            spatial_shuffles=100,
            sensitivity_scenarios=(
                SensitivityScenario("G30", 30.0, 0.02),
                SensitivityScenario("G100", 100.0, 0.02),
                SensitivityScenario("input_0.01", 60.0, 0.01),
                SensitivityScenario("input_0.04", 60.0, 0.04),
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete, immutable configuration for one experiment invocation."""

    mode: RunMode
    workload: ModeConfig
    n_regions: int = N_REGIONS
    main_global_coupling: float = MAIN_GLOBAL_COUPLING
    main_input_peak_per_ms: float = MAIN_INPUT_PEAK_PER_MS
    main_dt_ms: float = MAIN_DT_MS
    reference_dt_ms: float = REFERENCE_DT_MS
    monitor_period_ms: float = MONITOR_PERIOD_MS
    stimulus_onset_ms: float = STIMULUS_ONSET_MS
    simulation_ms: float = SIMULATION_MS
    periodic_analysis_start_ms: float = PERIODIC_ANALYSIS_START_MS
    pulse_width_ms: float = PULSE_WIDTH_MS
    pulse_analysis_end_ms: float = PULSE_ANALYSIS_END_MS
    probes: tuple[str, ...] = PROBES
    periodic_probes: tuple[str, ...] = PERIODIC_PROBES
    dt_check_severities: tuple[float, ...] = DT_CHECK_SEVERITIES
    dt_check_probes: tuple[str, ...] = DT_CHECK_PROBES
    download_results_at_end: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODE_CONFIGS:
            raise ValueError(f"Unknown run mode: {self.mode!r}")
        if self.workload != MODE_CONFIGS[self.mode]:
            raise ValueError("Mode and workload configuration do not match.")
        if self.n_regions != N_REGIONS:
            raise ValueError(f"This experiment requires exactly {N_REGIONS} regions.")
        if self.main_dt_ms <= 0 or self.reference_dt_ms <= 0:
            raise ValueError("Integration steps must be positive.")
        for step in (self.main_dt_ms, self.reference_dt_ms):
            ratio = self.monitor_period_ms / step
            if abs(ratio - round(ratio)) > 1e-12:
                raise ValueError(
                    "Monitor period must be an integer multiple of each dt."
                )
        if self.periodic_analysis_start_ms >= self.simulation_ms:
            raise ValueError("Periodic analysis must start before simulation end.")
        if not set(self.periodic_probes).issubset(self.probes):
            raise ValueError("Periodic probes must be included in probes.")
        if not set(self.dt_check_severities).issubset(self.severities):
            raise ValueError(
                "Integration-step severities must be included in the mode."
            )
        if not set(self.dt_check_probes).issubset(self.periodic_probes):
            raise ValueError(
                "Integration-step probes must be periodic probes."
            )

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.workload.seeds

    @property
    def severities(self) -> tuple[float, ...]:
        return self.workload.severities

    @property
    def results_directory_name(self) -> str:
        return f"results_{self.mode}"


@dataclass(frozen=True, slots=True)
class WorkloadCounts:
    """TVB simulation counts for one configured run."""

    calibration: int
    main: int
    local_dynamics_counterfactual: int
    sensitivity: int
    spatial_shuffle: int
    integration_step_check: int

    @property
    def manifest(self) -> int:
        """Simulations represented in ``run_manifest.csv``."""

        return (
            self.main
            + self.local_dynamics_counterfactual
            + self.sensitivity
            + self.spatial_shuffle
            + self.integration_step_check
        )

    @property
    def total(self) -> int:
        """All simulations, including the separately recorded calibration scan."""

        return self.calibration + self.manifest

    def to_dict(self) -> dict[str, int]:
        return {
            "calibration": self.calibration,
            "main": self.main,
            "local_dynamics_counterfactual": self.local_dynamics_counterfactual,
            "sensitivity": self.sensitivity,
            "spatial_shuffle": self.spatial_shuffle,
            "integration_step_check": self.integration_step_check,
            "manifest": self.manifest,
            "total": self.total,
        }


def normalize_run_mode(value: str) -> RunMode:
    """Normalize and validate a run-mode string."""

    normalized = value.strip().lower()
    if normalized not in MODE_CONFIGS:
        choices = ", ".join(MODE_CONFIGS)
        raise ValueError(f"RUN_MODE must be one of: {choices}.")
    return normalized  # type: ignore[return-value]


def get_experiment_config(
    mode: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ExperimentConfig:
    """Build a config, defaulting to ``RISE_RUN_MODE`` and then ``final``."""

    environment = os.environ if environ is None else environ
    selected = normalize_run_mode(
        mode if mode is not None else environment.get("RISE_RUN_MODE", "final")
    )
    return ExperimentConfig(mode=selected, workload=MODE_CONFIGS[selected])


def workload_counts(config: ExperimentConfig | ModeConfig) -> WorkloadCounts:
    """Calculate the exact TVB workload implied by a configuration."""

    mode = config.workload if isinstance(config, ExperimentConfig) else config
    probes = config.probes if isinstance(config, ExperimentConfig) else PROBES
    periodic_probes = (
        config.periodic_probes
        if isinstance(config, ExperimentConfig)
        else PERIODIC_PROBES
    )
    dt_check_severities = (
        config.dt_check_severities
        if isinstance(config, ExperimentConfig)
        else DT_CHECK_SEVERITIES
    )
    dt_check_probes = (
        config.dt_check_probes
        if isinstance(config, ExperimentConfig)
        else DT_CHECK_PROBES
    )

    calibration = len(mode.calibration_couplings) * 2
    main = len(mode.severities) * len(mode.seeds) * (1 + len(probes))
    local_counterfactual = len(mode.seeds) * (1 + len(probes))
    sensitivity = (
        len(mode.sensitivity_scenarios)
        * 2
        * (1 + len(periodic_probes))
    )
    spatial_shuffle = mode.spatial_shuffles * (1 + len(periodic_probes))
    integration_step_check = len(dt_check_severities) * (
        1 + len(dt_check_probes)
    )
    return WorkloadCounts(
        calibration=calibration,
        main=main,
        local_dynamics_counterfactual=local_counterfactual,
        sensitivity=sensitivity,
        spatial_shuffle=spatial_shuffle,
        integration_step_check=integration_step_check,
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Return the explicit, version-stable JSON payload for a config."""

    return {
        "schema_version": 1,
        "mode": config.mode,
        "mode_config": {
            "seeds": list(config.workload.seeds),
            "severities": list(config.workload.severities),
            "calibration_couplings": list(
                config.workload.calibration_couplings
            ),
            "matched_null_sets": config.workload.matched_null_sets,
            "spatial_shuffles": config.workload.spatial_shuffles,
            "sensitivity_scenarios": [
                {
                    "scenario": scenario.name,
                    "global_coupling": scenario.global_coupling,
                    "input_peak": scenario.input_peak_per_ms,
                }
                for scenario in config.workload.sensitivity_scenarios
            ],
        },
        "experiment": {
            "n_regions": config.n_regions,
            "main_global_coupling": config.main_global_coupling,
            "main_input_peak_per_ms": config.main_input_peak_per_ms,
            "main_dt_ms": config.main_dt_ms,
            "reference_dt_ms": config.reference_dt_ms,
            "monitor_period_ms": config.monitor_period_ms,
            "stimulus_onset_ms": config.stimulus_onset_ms,
            "simulation_ms": config.simulation_ms,
            "periodic_analysis_start_ms": config.periodic_analysis_start_ms,
            "pulse_width_ms": config.pulse_width_ms,
            "pulse_analysis_end_ms": config.pulse_analysis_end_ms,
            "probes": list(config.probes),
            "periodic_probes": list(config.periodic_probes),
            "dt_check_severities": list(config.dt_check_severities),
            "dt_check_probes": list(config.dt_check_probes),
            "severity_labels": [
                {"severity": severity, "label": label}
                for severity, label in SEVERITY_LABELS.items()
            ],
            "download_results_at_end": config.download_results_at_end,
        },
    }


def config_to_json(config: ExperimentConfig, *, pretty: bool = False) -> str:
    """Serialize a config deterministically."""

    if pretty:
        return json.dumps(
            config_to_dict(config),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    return json.dumps(
        config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_digest_input(config: ExperimentConfig) -> bytes:
    """Return canonical bytes suitable for hashing or cache keys."""

    return config_to_json(config).encode("utf-8")


def config_digest(config: ExperimentConfig) -> str:
    """Return a SHA-256 digest of the canonical configuration."""

    return hashlib.sha256(config_digest_input(config)).hexdigest()


DEFAULT_CONFIG = get_experiment_config("final")
