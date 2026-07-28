"""End-to-end scientific workflow with bounded resumable work blocks."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .analysis import (
    build_matched_control_sets,
    build_sensitivity_tables,
    build_shuffle_tables,
    check_integration_step,
    compare_counterfactual,
    score_matched_control_null,
    summarize_main_stage,
)
from .checkpoints import read_completed_block, write_completed_block
from .config import (
    SEVERITY_LABELS,
    ExperimentConfig,
    config_to_dict,
    workload_counts,
)
from .data import (
    EDUCASE_COMMIT,
    PIPELINE_COMMIT,
    ExperimentData,
)
from .metrics import make_contrasts, normalize_to_baseline, pulse_rms
from .simulation import (
    BlockResult,
    SimulationContext,
    run_condition_seed_block,
    run_tvb,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineProducts:
    """All reader-facing tables and internal plot inputs."""

    tables: dict[str, pd.DataFrame]
    main_normalized_df: pd.DataFrame
    primary_endpoint_df: pd.DataFrame
    counterfactual_comparison_df: pd.DataFrame
    matched_null_df: pd.DataFrame
    sensitivity_endpoint_df: pd.DataFrame
    shuffle_contrast_df: pd.DataFrame
    observed_first_seed_df: pd.DataFrame
    calibration_df: pd.DataFrame
    target_feature_df: pd.DataFrame
    shuffle_summary_df: pd.DataFrame


def make_simulation_context(
    config: ExperimentConfig,
    data: ExperimentData,
) -> SimulationContext:
    """Translate validated data/config into the TVB runner context."""

    return SimulationContext(
        weights=data.weights,
        labels=data.labels,
        a1_indices=data.rois.a1_indices,
        music_indices=data.rois.music_indices,
        speech_indices=data.rois.speech_indices,
        n_regions=config.n_regions,
        monitor_period_ms=config.monitor_period_ms,
        stimulus_onset_ms=config.stimulus_onset_ms,
        periodic_analysis_start_ms=config.periodic_analysis_start_ms,
        pulse_width_ms=config.pulse_width_ms,
        pulse_analysis_end_ms=config.pulse_analysis_end_ms,
        default_simulation_ms=config.simulation_ms,
    )


def _block_key(severity: float, seed: int) -> str:
    return f"severity_{severity:.1f}_seed_{int(seed)}"


def _load_or_run_simulation_block(
    checkpoint_root: Path,
    *,
    stage: str,
    block_key: str,
    runner: Callable[[], BlockResult],
    metadata: dict[str, Any],
) -> BlockResult:
    try:
        saved = read_completed_block(checkpoint_root, stage, block_key)
    except RuntimeError:
        LOGGER.info("Running checkpoint block %s/%s", stage, block_key)
        result = runner()
        write_completed_block(
            checkpoint_root,
            stage,
            block_key,
            {
                "node": result.node,
                "network": result.network,
                "manifest": result.manifest,
            },
            metadata,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        return result
    LOGGER.info("Loaded checkpoint block %s/%s", stage, block_key)
    return BlockResult(
        node=saved.frames["node"],
        network=saved.frames["network"],
        manifest=saved.frames["manifest"],
    )


def _concat_blocks(
    blocks: list[BlockResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not blocks:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (
        pd.concat([block.node for block in blocks], ignore_index=True),
        pd.concat([block.network for block in blocks], ignore_index=True),
        pd.concat([block.manifest for block in blocks], ignore_index=True),
    )


def _run_grid(
    checkpoint_root: Path,
    context: SimulationContext,
    *,
    stage: str,
    scope: str,
    conditions: list[dict[str, Any]],
    seeds: tuple[int, ...],
    probes: tuple[str, ...],
    global_coupling: float,
    input_peak_per_ms: float,
    dt_ms: float,
    simulation_ms: float,
    key_prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks: list[BlockResult] = []
    for condition in conditions:
        for seed in seeds:
            key = (
                f"{key_prefix}{_block_key(float(condition['severity']), seed)}"
            )
            block = _load_or_run_simulation_block(
                checkpoint_root,
                stage=stage,
                block_key=key,
                runner=lambda condition=condition, seed=seed: (
                    run_condition_seed_block(
                        context,
                        scope=scope,
                        condition=condition,
                        seed=seed,
                        probes=probes,
                        global_coupling=global_coupling,
                        input_peak_per_ms=input_peak_per_ms,
                        dt_ms=dt_ms,
                        simulation_ms=simulation_ms,
                    )
                ),
                metadata={
                    "scope": scope,
                    "variant": str(
                        condition.get("variant", condition["condition"])
                    ),
                    "severity": float(condition["severity"]),
                    "seed": int(seed),
                    "probes": list(probes),
                    "global_coupling": float(global_coupling),
                    "input_peak_per_ms": float(input_peak_per_ms),
                    "dt_ms": float(dt_ms),
                    "simulation_ms": float(simulation_ms),
                },
            )
            blocks.append(block)
    return _concat_blocks(blocks)


def _run_calibration(
    config: ExperimentConfig,
    data: ExperimentData,
    context: SimulationContext,
    checkpoint_root: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    calibration_simulation_ms = 4000.0
    calibration_seed = config.seeds[0]

    for candidate_g in config.workload.calibration_couplings:
        block_key = f"g_{candidate_g:g}"
        try:
            saved = read_completed_block(
                checkpoint_root, "calibration", block_key
            )
        except RuntimeError:
            LOGGER.info(
                "Baseline coupling calibration G=%s", candidate_g
            )
            control_time, control_psp, control_wall = run_tvb(
                context,
                b_values=data.baseline_b,
                probe=None,
                global_coupling=candidate_g,
                input_peak_per_ms=config.main_input_peak_per_ms,
                seed=calibration_seed,
                dt_ms=config.main_dt_ms,
                simulation_ms=calibration_simulation_ms,
            )
            pulse_time, pulse_psp, pulse_wall = run_tvb(
                context,
                b_values=data.baseline_b,
                probe="pulse",
                global_coupling=candidate_g,
                input_peak_per_ms=config.main_input_peak_per_ms,
                seed=calibration_seed,
                dt_ms=config.main_dt_ms,
                simulation_ms=calibration_simulation_ms,
            )
            if not np.allclose(control_time, pulse_time):
                raise RuntimeError("Calibration time axes differ.")
            response, _ = pulse_rms(
                pulse_time,
                pulse_psp - control_psp,
                onset_ms=config.stimulus_onset_ms,
                analysis_end_ms=config.pulse_analysis_end_ms,
                n_regions=config.n_regions,
            )
            a1 = float(np.mean(response[data.rois.a1_indices]))
            music = float(np.mean(response[data.rois.music_indices]))
            speech = float(np.mean(response[data.rois.speech_indices]))
            frame = pd.DataFrame(
                [
                    {
                        "global_coupling": candidate_g,
                        "a1_rms": a1,
                        "music_transfer": music / a1,
                        "speech_transfer": speech / a1,
                        "balanced_target_score": math.sqrt(
                            (music / a1) * (speech / a1)
                        ),
                        "max_abs_evoked": float(
                            np.max(np.abs(pulse_psp - control_psp))
                        ),
                        "wall_seconds": control_wall + pulse_wall,
                    }
                ]
            )
            write_completed_block(
                checkpoint_root,
                "calibration",
                block_key,
                {"calibration": frame},
                {
                    "global_coupling": candidate_g,
                    "seed": calibration_seed,
                    "simulation_ms": calibration_simulation_ms,
                },
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        else:
            LOGGER.info(
                "Loaded calibration checkpoint G=%s", candidate_g
            )
            frame = saved.frames["calibration"]
        rows.append(frame)

    calibration_df = pd.concat(rows, ignore_index=True).sort_values(
        "global_coupling"
    )
    if not np.isfinite(
        calibration_df.select_dtypes("number")
    ).all().all():
        raise RuntimeError("Calibration produced a nonfinite value.")
    selected = calibration_df[
        calibration_df["global_coupling"]
        == config.main_global_coupling
    ]
    if selected.empty:
        warnings.warn(
            "The selected coupling was not included in this shortened scan.",
            stacklevel=2,
        )
    elif float(selected["max_abs_evoked"].iloc[0]) >= 50.0:
        raise RuntimeError(
            "The selected coupling produced a saturated response."
        )
    return calibration_df.reset_index(drop=True)


def run_pipeline(
    config: ExperimentConfig,
    data: ExperimentData,
    *,
    source_manifest_df: pd.DataFrame,
    run_dir: Path,
) -> PipelineProducts:
    """Execute or resume every scientific stage and assemble final tables."""

    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    context = make_simulation_context(config, data)
    calibration_df = _run_calibration(
        config, data, context, checkpoint_root
    )

    main_conditions = [
        {
            "condition": SEVERITY_LABELS[severity],
            "severity": severity,
            "b_values": data.b_by_severity[severity],
            "variant": "full_field",
        }
        for severity in config.severities
    ]
    main_node_df, main_network_df, main_manifest_df = _run_grid(
        checkpoint_root,
        context,
        stage="main",
        scope="main_full_field",
        conditions=main_conditions,
        seeds=config.seeds,
        probes=config.probes,
        global_coupling=config.main_global_coupling,
        input_peak_per_ms=config.main_input_peak_per_ms,
        dt_ms=config.main_dt_ms,
        simulation_ms=config.simulation_ms,
    )
    main_normalized_df = normalize_to_baseline(main_network_df)
    main_contrast_df = make_contrasts(main_normalized_df)
    main_stage_summary_df, primary_endpoint_df = summarize_main_stage(
        main_normalized_df,
        main_contrast_df,
        periodic_probes=config.periodic_probes,
    )

    # Fail early, before counterfactuals and robustness stages.
    _, reference_network_df, reference_manifest_df = _run_grid(
        checkpoint_root,
        context,
        stage="dt_reference",
        scope="dt_reference_0.5ms",
        conditions=[
            {
                "condition": SEVERITY_LABELS[0.0],
                "severity": 0.0,
                "b_values": data.baseline_b,
                "variant": "dt_0.5ms",
            }
        ],
        seeds=(config.seeds[0],),
        probes=("2Hz",),
        global_coupling=config.main_global_coupling,
        input_peak_per_ms=config.main_input_peak_per_ms,
        dt_ms=config.reference_dt_ms,
        simulation_ms=config.simulation_ms,
    )
    dt_convergence_df = check_integration_step(
        main_network_df=main_network_df,
        reference_network_df=reference_network_df,
        main_seed=config.seeds[0],
    )

    local_fixed_high_b = data.high_b.copy()
    local_fixed_high_b[data.rois.all_declared_indices] = data.baseline_b[
        data.rois.all_declared_indices
    ]
    (
        local_fixed_node_df,
        local_fixed_network_df,
        local_fixed_manifest_df,
    ) = _run_grid(
        checkpoint_root,
        context,
        stage="local_fixed",
        scope="local_dynamics_counterfactual",
        conditions=[
            {
                "condition": (
                    "High AD-like perturbation, declared local dynamics fixed"
                ),
                "severity": 1.0,
                "b_values": local_fixed_high_b,
                "variant": "local_fixed_endpoint",
            }
        ],
        seeds=config.seeds,
        probes=config.probes,
        global_coupling=config.main_global_coupling,
        input_peak_per_ms=config.main_input_peak_per_ms,
        dt_ms=config.main_dt_ms,
        simulation_ms=config.simulation_ms,
    )
    (
        _local_fixed_normalized_df,
        local_fixed_contrast_df,
        counterfactual_comparison_df,
    ) = compare_counterfactual(
        primary_endpoint_df,
        local_fixed_network_df,
        main_network_df,
        periodic_probes=config.periodic_probes,
    )

    target_feature_df, matched_sets_df = build_matched_control_sets(
        weights=data.weights,
        baseline_b=data.baseline_b,
        high_b=data.high_b,
        labels=data.labels,
        a1_indices=data.rois.a1_indices,
        music_indices=data.rois.music_indices,
        speech_indices=data.rois.speech_indices,
        all_declared_indices=data.rois.all_declared_indices,
        n_sets=config.workload.matched_null_sets,
    )
    matched_null_df, matched_null_summary_df = score_matched_control_null(
        main_node_df=main_node_df,
        main_network_df=main_network_df,
        matched_sets_df=matched_sets_df,
        primary_endpoint_df=primary_endpoint_df,
        seeds=config.seeds,
        periodic_probes=config.periodic_probes,
        n_regions=config.n_regions,
    )

    main_seed = config.seeds[0]
    main_sensitivity_source = main_network_df[
        (main_network_df["seed"] == main_seed)
        & (main_network_df["severity"].isin([0.0, 1.0]))
        & (main_network_df["probe"].isin(config.periodic_probes))
    ].copy()
    main_sensitivity_source["scope"] = "sensitivity_main"
    main_sensitivity_source["variant"] = "G60_input_0.02"
    sensitivity_network_frames = [main_sensitivity_source]
    sensitivity_manifest_frames: list[pd.DataFrame] = []

    for scenario in config.workload.sensitivity_scenarios:
        conditions = [
            {
                "condition": SEVERITY_LABELS[severity],
                "severity": severity,
                "b_values": data.b_by_severity[severity],
                "variant": scenario.name,
            }
            for severity in (0.0, 1.0)
        ]
        _, scenario_network_df, scenario_manifest_df = _run_grid(
            checkpoint_root,
            context,
            stage="sensitivity",
            scope=f"sensitivity_{scenario.name}",
            conditions=conditions,
            seeds=(main_seed,),
            probes=config.periodic_probes,
            global_coupling=scenario.global_coupling,
            input_peak_per_ms=scenario.input_peak_per_ms,
            dt_ms=config.main_dt_ms,
            simulation_ms=config.simulation_ms,
            key_prefix=f"{scenario.name}_",
        )
        sensitivity_network_frames.append(scenario_network_df)
        sensitivity_manifest_frames.append(scenario_manifest_df)
    (
        sensitivity_network_df,
        _sensitivity_normalized_df,
        sensitivity_contrast_df,
        sensitivity_endpoint_df,
    ) = build_sensitivity_tables(sensitivity_network_frames)
    sensitivity_manifest_df = (
        pd.concat(sensitivity_manifest_frames, ignore_index=True)
        if sensitivity_manifest_frames
        else pd.DataFrame()
    )

    shuffle_rng = np.random.default_rng(3792026)
    shuffle_blocks = [
        np.arange(0, 180),
        np.arange(180, 360),
        np.arange(360, 379),
    ]
    shuffle_network_frames: list[pd.DataFrame] = []
    shuffle_manifest_frames: list[pd.DataFrame] = []
    for shuffle_index in range(config.workload.spatial_shuffles):
        shuffle_id = shuffle_index + 1
        shuffled_b = data.high_b.copy()
        for block in shuffle_blocks:
            shuffled_b[block] = shuffle_rng.permutation(data.high_b[block])
        variant = f"shuffle_{shuffle_id:02d}"
        _, network_df, manifest_df = _run_grid(
            checkpoint_root,
            context,
            stage="spatial_shuffle",
            scope=f"spatial_shuffle_{shuffle_id:02d}",
            conditions=[
                {
                    "condition": (
                        "High AD-like perturbation, spatially shuffled"
                    ),
                    "severity": 1.0,
                    "b_values": shuffled_b,
                    "variant": variant,
                }
            ],
            seeds=(main_seed,),
            probes=config.periodic_probes,
            global_coupling=config.main_global_coupling,
            input_peak_per_ms=config.main_input_peak_per_ms,
            dt_ms=config.main_dt_ms,
            simulation_ms=config.simulation_ms,
            key_prefix=f"{variant}_",
        )
        shuffle_network_frames.append(network_df)
        shuffle_manifest_frames.append(manifest_df)
    (
        shuffle_network_df,
        _shuffle_normalized_df,
        shuffle_contrast_df,
        observed_first_seed_df,
        shuffle_summary_df,
    ) = build_shuffle_tables(
        shuffle_network_frames,
        main_network_df=main_network_df,
        primary_endpoint_df=primary_endpoint_df,
        main_seed=main_seed,
    )
    shuffle_manifest_df = pd.concat(
        shuffle_manifest_frames, ignore_index=True
    )

    all_manifest_df = pd.concat(
        [
            main_manifest_df,
            local_fixed_manifest_df,
            sensitivity_manifest_df,
            shuffle_manifest_df,
            reference_manifest_df,
        ],
        ignore_index=True,
    )
    tables = {
        "source_manifest": source_manifest_df,
        "data_quality": data.data_quality_df,
        "roi_definitions": data.roi_definition_df,
        "roi_pathology": data.roi_pathology_df,
        "pathology_summary": data.pathology_summary_df,
        "calibration": calibration_df,
        "main_node": main_node_df,
        "main_network": main_network_df,
        "main_normalized": main_normalized_df,
        "main_contrasts": main_contrast_df,
        "main_stage_summary": main_stage_summary_df,
        "local_fixed_node": local_fixed_node_df,
        "local_fixed_network": local_fixed_network_df,
        "local_fixed_contrasts": local_fixed_contrast_df,
        "matched_sets": matched_sets_df,
        "matched_null": matched_null_df,
        "matched_null_summary": matched_null_summary_df,
        "sensitivity_network": sensitivity_network_df,
        "sensitivity_contrasts": sensitivity_contrast_df,
        "shuffle_network": shuffle_network_df,
        "shuffle_contrasts": shuffle_contrast_df,
        "dt_convergence": dt_convergence_df,
        "run_manifest": all_manifest_df,
    }
    return PipelineProducts(
        tables=tables,
        main_normalized_df=main_normalized_df,
        primary_endpoint_df=primary_endpoint_df,
        counterfactual_comparison_df=counterfactual_comparison_df,
        matched_null_df=matched_null_df,
        sensitivity_endpoint_df=sensitivity_endpoint_df,
        shuffle_contrast_df=shuffle_contrast_df,
        observed_first_seed_df=observed_first_seed_df,
        calibration_df=calibration_df,
        target_feature_df=target_feature_df,
        shuffle_summary_df=shuffle_summary_df,
    )


def build_experiment_metadata(
    config: ExperimentConfig,
    data: ExperimentData,
    source_manifest_df: pd.DataFrame,
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build notebook-compatible metadata plus reproducibility fingerprints."""

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": config.mode,
        "research_question": (
            "How does increasing AD-like amyloid-linked inhibitory "
            "dysfunction affect stimulus-evoked transmission from bilateral "
            "primary auditory cortex to music-associated versus "
            "speech-associated cortical proxy subnetworks?"
        ),
        "source_commits": {
            "educase": EDUCASE_COMMIT,
            "adni_tvb_pipeline": PIPELINE_COMMIT,
        },
        "source_hashes": {
            row.source: row.sha256
            for row in source_manifest_df.itertuples(index=False)
        },
        "resolved_config": config_to_dict(config),
        "workload": workload_counts(config).to_dict(),
        "model": {
            "regions": config.n_regions,
            "global_coupling": config.main_global_coupling,
            "input_peak_per_ms": config.main_input_peak_per_ms,
            "dt_ms": config.main_dt_ms,
            "reference_dt_ms": config.reference_dt_ms,
            "monitor_period_ms": config.monitor_period_ms,
            "simulation_ms": config.simulation_ms,
            "stimulus_onset_ms": config.stimulus_onset_ms,
            "delays": (
                "all zero, following the public educational model"
            ),
            "probes": list(config.probes),
            "numerical_seeds": list(config.seeds),
            "spatial_shuffles": config.workload.spatial_shuffles,
        },
        "parcels": {
            "A1": list(data.rois.a1_labels),
            "music_proxy": list(data.rois.music_labels),
            "speech_proxy": list(data.rois.speech_labels),
        },
        "primary_metric": (
            "network harmonic amplitude divided by bilateral A1 harmonic "
            "amplitude, then log2-normalized to the same network's baseline"
        ),
        "primary_contrast": "music log2 change minus speech log2 change",
        "interpretation_limits": [
            (
                "The public amyloid endpoint is an artificial surrogate, "
                "not patient data."
            ),
            (
                "The model includes amyloid-linked inhibition but not tau, "
                "atrophy, synapse loss, inflammation, vascular disease, or "
                "structural degeneration."
            ),
            (
                "The connectome is one averaged healthy structural matrix "
                "for every condition."
            ),
            "All interregional delays are zero.",
            (
                "The 2 Hz and 5 Hz inputs are temporal probes, not literal "
                "music and speech."
            ),
            "The parcel groups are approximate proxy subnetworks.",
            "The model contains no memory encoding or retrieval mechanism.",
            (
                "Numerical seeds and matched control sets are not human "
                "subjects."
            ),
        ],
        "provenance": provenance,
    }
