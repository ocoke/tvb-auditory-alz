"""End-to-end scientific workflow with bounded resumable work blocks."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from .metrics import make_contrasts, normalize_to_baseline
from .parallel import ParallelRunner, WorkerJob, WorkerOutcome
from .simulation import (
    BlockResult,
    SimulationContext,
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


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(frozen=True)
class _BlockPlan:
    """One ordered worker job plus its parent-owned checkpoint contract."""

    ordinal: int
    stage: str
    block_key: str
    kind: str
    calls: int
    work_units: float
    metadata: dict[str, Any]
    job: WorkerJob


class _ProgressReporter:
    """Log checkpoint-aware stage progress and a weighted ETA."""

    def __init__(self, *, total_calls: int, total_work_units: float) -> None:
        self.total_calls = int(total_calls)
        self.total_work_units = float(total_work_units)
        self.completed_calls = 0
        self.completed_work_units = 0.0
        self.executed_work_units = 0.0
        self.started = time.perf_counter()

    def restore(self, plans: list[_BlockPlan]) -> None:
        self.completed_calls += sum(plan.calls for plan in plans)
        self.completed_work_units += sum(plan.work_units for plan in plans)

    def stage_started(
        self,
        *,
        stage_number: int,
        stage_name: str,
        plans: list[_BlockPlan],
        cached_plans: list[_BlockPlan],
        workers: int,
    ) -> None:
        calls = sum(plan.calls for plan in plans)
        cached_calls = sum(plan.calls for plan in cached_plans)
        pending_blocks = len(plans) - len(cached_plans)
        active_workers = min(workers, pending_blocks)
        LOGGER.info(
            "Stage %d/8 %s: %d blocks, %d TVB calls; "
            "%d blocks/%d calls restored; pool capacity=%d workers, "
            "max active now=%d",
            stage_number,
            stage_name,
            len(plans),
            calls,
            len(cached_plans),
            cached_calls,
            workers,
            active_workers,
        )

    def block_completed(
        self,
        *,
        plan: _BlockPlan,
        stage_number: int,
        stage_name: str,
        stage_completed_blocks: int,
        stage_total_blocks: int,
        elapsed_seconds: float,
    ) -> None:
        self.completed_calls += plan.calls
        self.completed_work_units += plan.work_units
        self.executed_work_units += plan.work_units
        elapsed = time.perf_counter() - self.started
        remaining_work = max(
            0.0, self.total_work_units - self.completed_work_units
        )
        if self.executed_work_units > 0.0 and elapsed > 0.0:
            eta_seconds = remaining_work / (
                self.executed_work_units / elapsed
            )
            eta = f"~{_format_duration(eta_seconds)}"
        else:
            eta = "calculating"
        percent = (
            100.0 * self.completed_calls / self.total_calls
            if self.total_calls
            else 100.0
        )
        LOGGER.info(
            "Progress %d/%d (%.1f%%) | stage %d/8 %s %d/%d blocks | "
            "%s | block %.1fs | elapsed %s | ETA %s",
            self.completed_calls,
            self.total_calls,
            percent,
            stage_number,
            stage_name,
            stage_completed_blocks,
            stage_total_blocks,
            plan.block_key,
            elapsed_seconds,
            _format_duration(elapsed),
            eta,
        )


def _total_work_units(config: ExperimentConfig) -> float:
    """Approximate relative CPU work for ETA calculations."""

    counts = workload_counts(config)
    calibration_scale = 4000.0 / config.simulation_ms
    reference_scale = config.main_dt_ms / config.reference_dt_ms
    return (
        counts.calibration * calibration_scale
        + counts.main
        + counts.local_dynamics_counterfactual
        + counts.sensitivity
        + counts.spatial_shuffle
        + counts.integration_step_check * reference_scale
    )


def _decode_saved_result(plan: _BlockPlan, saved: Any) -> Any:
    if plan.kind == "calibration":
        return saved.frames["calibration"]
    return BlockResult(
        node=saved.frames["node"],
        network=saved.frames["network"],
        manifest=saved.frames["manifest"],
    )


def _checkpoint_frames(plan: _BlockPlan, result: Any) -> dict[str, pd.DataFrame]:
    if plan.kind == "calibration":
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Calibration worker did not return a dataframe.")
        return {"calibration": result}
    if not isinstance(result, BlockResult):
        raise TypeError("Simulation worker did not return a BlockResult.")
    return {
        "node": result.node,
        "network": result.network,
        "manifest": result.manifest,
    }


def _execute_plans(
    checkpoint_root: Path,
    runner: ParallelRunner,
    progress: _ProgressReporter,
    *,
    plans: list[_BlockPlan],
    stage_number: int,
    stage_name: str,
) -> list[Any]:
    """Restore or execute plans, checkpointing completions in the parent."""

    if [plan.ordinal for plan in plans] != list(range(len(plans))):
        raise ValueError("Block-plan ordinals must be contiguous and ordered.")

    ordered_results: list[Any | None] = [None] * len(plans)
    cached_plans: list[_BlockPlan] = []
    pending_plans: list[_BlockPlan] = []
    for plan in plans:
        try:
            saved = read_completed_block(
                checkpoint_root,
                plan.stage,
                plan.block_key,
            )
        except RuntimeError:
            pending_plans.append(plan)
        else:
            ordered_results[plan.ordinal] = _decode_saved_result(plan, saved)
            cached_plans.append(plan)

    progress.restore(cached_plans)
    progress.stage_started(
        stage_number=stage_number,
        stage_name=stage_name,
        plans=plans,
        cached_plans=cached_plans,
        workers=runner.worker_count,
    )
    if cached_plans:
        LOGGER.info(
            "Restored %d completed %s checkpoint blocks; overall %d/%d "
            "TVB calls complete",
            len(cached_plans),
            stage_name,
            progress.completed_calls,
            progress.total_calls,
        )

    pending_by_ordinal = {plan.ordinal: plan for plan in pending_plans}
    completed_blocks = len(cached_plans)
    for outcome in runner.execute(plan.job for plan in pending_plans):
        if not isinstance(outcome, WorkerOutcome):
            raise TypeError("Parallel runner returned an invalid outcome.")
        plan = pending_by_ordinal[outcome.ordinal]
        if outcome.kind != plan.kind:
            raise RuntimeError(
                f"Worker kind mismatch for {plan.stage}/{plan.block_key}."
            )
        result = outcome.result
        write_completed_block(
            checkpoint_root,
            plan.stage,
            plan.block_key,
            _checkpoint_frames(plan, result),
            plan.metadata,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        ordered_results[plan.ordinal] = result
        completed_blocks += 1
        progress.block_completed(
            plan=plan,
            stage_number=stage_number,
            stage_name=stage_name,
            stage_completed_blocks=completed_blocks,
            stage_total_blocks=len(plans),
            elapsed_seconds=outcome.elapsed_seconds,
        )

    if any(result is None for result in ordered_results):
        raise RuntimeError(f"Stage {stage_name} did not produce every block.")
    return list(ordered_results)


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
    runner: ParallelRunner,
    progress: _ProgressReporter,
    *,
    stage_number: int,
    stage_name: str,
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
    plans: list[_BlockPlan] = []
    for condition in conditions:
        for seed in seeds:
            resolved_scope = str(condition.get("scope", scope))
            resolved_global_coupling = float(
                condition.get("global_coupling", global_coupling)
            )
            resolved_input_peak = float(
                condition.get("input_peak_per_ms", input_peak_per_ms)
            )
            resolved_prefix = str(
                condition.get("key_prefix", key_prefix)
            )
            key = (
                f"{resolved_prefix}"
                f"{_block_key(float(condition['severity']), seed)}"
            )
            variant = str(
                condition.get("variant", condition["condition"])
            )
            worker_condition = {
                "condition": str(condition["condition"]),
                "severity": float(condition["severity"]),
                "b_values": np.asarray(condition["b_values"], dtype=float),
                "variant": variant,
            }
            payload = {
                "scope": resolved_scope,
                "condition": worker_condition,
                "seed": int(seed),
                "probes": tuple(probes),
                "global_coupling": resolved_global_coupling,
                "input_peak_per_ms": resolved_input_peak,
                "dt_ms": float(dt_ms),
                "simulation_ms": float(simulation_ms),
            }
            ordinal = len(plans)
            calls = 1 + len(probes)
            plans.append(
                _BlockPlan(
                    ordinal=ordinal,
                    stage=stage,
                    block_key=key,
                    kind="simulation",
                    calls=calls,
                    work_units=(
                        calls
                        * (simulation_ms / context.default_simulation_ms)
                        * (1.0 / dt_ms)
                    ),
                    metadata={
                        "scope": resolved_scope,
                        "variant": variant,
                        "severity": float(condition["severity"]),
                        "seed": int(seed),
                        "probes": list(probes),
                        "global_coupling": resolved_global_coupling,
                        "input_peak_per_ms": resolved_input_peak,
                        "dt_ms": float(dt_ms),
                        "simulation_ms": float(simulation_ms),
                    },
                    job=WorkerJob(
                        ordinal=ordinal,
                        kind="simulation",
                        payload=payload,
                    ),
                )
            )
    blocks = _execute_plans(
        checkpoint_root,
        runner,
        progress,
        plans=plans,
        stage_number=stage_number,
        stage_name=stage_name,
    )
    return _concat_blocks(blocks)


def _run_calibration(
    config: ExperimentConfig,
    data: ExperimentData,
    context: SimulationContext,
    checkpoint_root: Path,
    runner: ParallelRunner,
    progress: _ProgressReporter,
) -> pd.DataFrame:
    calibration_simulation_ms = 4000.0
    calibration_seed = config.seeds[0]
    plans: list[_BlockPlan] = []

    for candidate_g in config.workload.calibration_couplings:
        ordinal = len(plans)
        block_key = f"g_{candidate_g:g}"
        payload = {
            "baseline_b": data.baseline_b,
            "global_coupling": float(candidate_g),
            "input_peak_per_ms": config.main_input_peak_per_ms,
            "seed": calibration_seed,
            "dt_ms": config.main_dt_ms,
            "simulation_ms": calibration_simulation_ms,
        }
        plans.append(
            _BlockPlan(
                ordinal=ordinal,
                stage="calibration",
                block_key=block_key,
                kind="calibration",
                calls=2,
                work_units=(
                    2.0
                    * calibration_simulation_ms
                    / context.default_simulation_ms
                ),
                metadata={
                    "global_coupling": float(candidate_g),
                    "seed": calibration_seed,
                    "simulation_ms": calibration_simulation_ms,
                },
                job=WorkerJob(
                    ordinal=ordinal,
                    kind="calibration",
                    payload=payload,
                ),
            )
        )
    rows = _execute_plans(
        checkpoint_root,
        runner,
        progress,
        plans=plans,
        stage_number=1,
        stage_name="calibration",
    )
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
    worker_count: int | None = None,
) -> PipelineProducts:
    """Execute or resume every scientific stage and assemble final tables."""

    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    context = make_simulation_context(config, data)
    counts = workload_counts(config)
    progress = _ProgressReporter(
        total_calls=counts.total,
        total_work_units=_total_work_units(config),
    )
    with ParallelRunner(
        context,
        run_dir,
        worker_count=worker_count,
    ) as runner:
        products = _run_pipeline_stages(
            config,
            data,
            source_manifest_df=source_manifest_df,
            checkpoint_root=checkpoint_root,
            context=context,
            runner=runner,
            progress=progress,
        )
    LOGGER.info(
        "Simulation pipeline complete: %d/%d TVB calls accounted for in %s",
        progress.completed_calls,
        progress.total_calls,
        _format_duration(time.perf_counter() - progress.started),
    )
    return products


def _run_pipeline_stages(
    config: ExperimentConfig,
    data: ExperimentData,
    *,
    source_manifest_df: pd.DataFrame,
    checkpoint_root: Path,
    context: SimulationContext,
    runner: ParallelRunner,
    progress: _ProgressReporter,
) -> PipelineProducts:
    """Run all stages using one persistent worker pool."""

    calibration_df = _run_calibration(
        config,
        data,
        context,
        checkpoint_root,
        runner,
        progress,
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
        runner,
        progress,
        stage_number=2,
        stage_name="main experiment",
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
        runner,
        progress,
        stage_number=3,
        stage_name="integration-step convergence",
        stage="dt_reference",
        scope="dt_reference_0.5ms",
        conditions=[
            {
                "condition": SEVERITY_LABELS[severity],
                "severity": severity,
                "b_values": data.b_by_severity[severity],
                "variant": f"dt_0.5ms_severity_{severity:.1f}",
            }
            for severity in config.dt_check_severities
        ],
        seeds=(config.seeds[0],),
        probes=config.dt_check_probes,
        global_coupling=config.main_global_coupling,
        input_peak_per_ms=config.main_input_peak_per_ms,
        dt_ms=config.reference_dt_ms,
        simulation_ms=config.simulation_ms,
    )
    dt_convergence_df = check_integration_step(
        main_network_df=main_network_df,
        reference_network_df=reference_network_df,
        main_seed=config.seeds[0],
        severities=config.dt_check_severities,
        probes=config.dt_check_probes,
    )
    LOGGER.info(
        "Integration-step convergence passed for %d endpoint/probe/network "
        "comparisons; maximum relative transfer difference %.4f%%",
        len(dt_convergence_df),
        100.0 * float(dt_convergence_df["relative_difference"].max()),
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
        runner,
        progress,
        stage_number=4,
        stage_name="local-dynamics counterfactual",
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

    matched_started = time.perf_counter()
    LOGGER.info(
        "Stage 5/8 matched controls: constructing and scoring %d "
        "deterministic control sets",
        config.workload.matched_null_sets,
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
    LOGGER.info(
        "Stage 5/8 matched controls complete in %s",
        _format_duration(time.perf_counter() - matched_started),
    )

    main_seed = config.seeds[0]
    main_sensitivity_source = main_network_df[
        (main_network_df["seed"] == main_seed)
        & (main_network_df["severity"].isin([0.0, 1.0]))
        & (main_network_df["probe"].isin(config.periodic_probes))
    ].copy()
    main_sensitivity_source["scope"] = "sensitivity_main"
    main_sensitivity_source["variant"] = "G60_input_0.02"
    sensitivity_conditions = [
        {
            "condition": SEVERITY_LABELS[severity],
            "severity": severity,
            "b_values": data.b_by_severity[severity],
            "variant": scenario.name,
            "scope": f"sensitivity_{scenario.name}",
            "global_coupling": scenario.global_coupling,
            "input_peak_per_ms": scenario.input_peak_per_ms,
            "key_prefix": f"{scenario.name}_",
        }
        for scenario in config.workload.sensitivity_scenarios
        for severity in (0.0, 1.0)
    ]
    (
        _sensitivity_node_df,
        sensitivity_worker_network_df,
        sensitivity_manifest_df,
    ) = _run_grid(
        checkpoint_root,
        context,
        runner,
        progress,
        stage_number=6,
        stage_name="parameter sensitivity",
        stage="sensitivity",
        scope="sensitivity",
        conditions=sensitivity_conditions,
        seeds=(main_seed,),
        probes=config.periodic_probes,
        global_coupling=config.main_global_coupling,
        input_peak_per_ms=config.main_input_peak_per_ms,
        dt_ms=config.main_dt_ms,
        simulation_ms=config.simulation_ms,
    )
    (
        sensitivity_network_df,
        _sensitivity_normalized_df,
        sensitivity_contrast_df,
        sensitivity_endpoint_df,
    ) = build_sensitivity_tables(
        [main_sensitivity_source, sensitivity_worker_network_df]
    )

    shuffle_rng = np.random.default_rng(3792026)
    shuffle_blocks = [
        np.arange(0, 180),
        np.arange(180, 360),
        np.arange(360, 379),
    ]
    shuffle_conditions: list[dict[str, Any]] = []
    for shuffle_index in range(config.workload.spatial_shuffles):
        shuffle_id = shuffle_index + 1
        shuffled_b = data.high_b.copy()
        for block in shuffle_blocks:
            shuffled_b[block] = shuffle_rng.permutation(data.high_b[block])
        variant = f"shuffle_{shuffle_id:02d}"
        shuffle_conditions.append(
            {
                "condition": (
                    "High AD-like perturbation, spatially shuffled"
                ),
                "severity": 1.0,
                "b_values": shuffled_b,
                "variant": variant,
                "scope": f"spatial_shuffle_{shuffle_id:02d}",
                "key_prefix": f"{variant}_",
            }
        )
    (
        _shuffle_node_df,
        shuffle_network_df,
        shuffle_manifest_df,
    ) = _run_grid(
        checkpoint_root,
        context,
        runner,
        progress,
        stage_number=7,
        stage_name="spatial shuffles",
        stage="spatial_shuffle",
        scope="spatial_shuffle",
        conditions=shuffle_conditions,
        seeds=(main_seed,),
        probes=config.periodic_probes,
        global_coupling=config.main_global_coupling,
        input_peak_per_ms=config.main_input_peak_per_ms,
        dt_ms=config.main_dt_ms,
        simulation_ms=config.simulation_ms,
    )
    (
        shuffle_network_df,
        _shuffle_normalized_df,
        shuffle_contrast_df,
        observed_first_seed_df,
        shuffle_summary_df,
    ) = build_shuffle_tables(
        [shuffle_network_df],
        main_network_df=main_network_df,
        primary_endpoint_df=primary_endpoint_df,
        main_seed=main_seed,
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
    LOGGER.info(
        "Stage 8/8 output assembly: preparing 23 tables, figures, metadata, "
        "and archive"
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
