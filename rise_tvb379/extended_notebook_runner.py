"""Validate and run the canonical late-window follow-up notebook.

The scientific implementation remains exclusively in the immutable notebook.
This module adds only direct-process execution, checkpoint recovery, progress
accounting, and static contract validation for ``main_extended.py``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from rise_tvb379.notebook_runner import (
    CANONICAL_NOTEBOOK as PRIMARY_NOTEBOOK,
    NotebookValidation,
    Workload,
    _code_cells,
    _configuration_literals,
    _literal,
    _output_contract,
    _resolved_workloads,
    validate_notebook,
)
from rise_tvb379.run_state import (
    CheckpointDispatcher,
    RunController,
    runtime_environment,
    utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTENDED_NOTEBOOK = (
    PROJECT_ROOT
    / "notebooks"
    / "RISE_TVB379_Semantic_Episodic_Final_LateWindowFollowup_20260802.ipynb"
)
EXTENDED_SHA256 = (
    "6c272953f8813c00af808a86c4c1aef1d8fa4c2acca3baa88947e498eab18c54"
)
EXPECTED_CELL_COUNT = 43
EXPECTED_CODE_CELL_COUNT = 20
EXPECTED_CODE_CELL_INDICES = (
    3,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    35,
    37,
    39,
)
PRIMARY_IMPLEMENTATION_CELL_INDICES = (16, 18, 22, 24, 26, 28, 30, 32)
EXPECTED_REVISION = "semantic_episodic_final_v3_science_ready"
EXPECTED_ARTIFACT_VERSION = (
    "semantic_episodic_v6_late_window_followup_2026-08-02"
)
EXPECTED_OUTPUT_COUNTS = (68, 8)
EXPECTED_WORKLOADS = {
    "smoke": (50, 48, 34, 8, 8, 4, 16, 10),
    "pilot": (133, 127, 85, 24, 24, 8, 32, 30),
    "final": (1242, 1230, 762, 240, 240, 100, 400, 300),
}
FOLLOWUP_OUTPUTS = frozenset(
    {
        "late_followup_dt_check.csv",
        "late_followup_dt_reference_network_metrics.csv",
        "late_followup_dt_reference_node_metrics.csv",
        "late_followup_dt_reference_pair_interactions.csv",
        "late_followup_dt_reference_segment_metrics.csv",
        "late_followup_dt_reference_trace_manifest.csv",
        "late_followup_interaction_statistics.csv",
        "late_followup_metric_reconciliation.csv",
        "late_followup_network_metrics.csv",
        "late_followup_network_metrics_normalized.csv",
        "late_followup_node_metrics.csv",
        "late_followup_original_vs_late.csv",
        "late_followup_outcome_eligibility.csv",
        "late_followup_pair_interactions.csv",
        "late_followup_prefix_reconciliation.csv",
        "late_followup_science_validity.csv",
        "late_followup_segment_metrics.csv",
        "late_followup_stability_equivalence.csv",
        "late_followup_ten_segment_slopes.csv",
        "late_followup_trace_manifest.csv",
    }
)


@dataclass(frozen=True)
class ExtendedWorkload(Workload):
    """Primary and prolonged follow-up TVB-call counts for one mode."""

    primary_total_calls: int
    primary_manifested_calls: int
    followup_main_blocks: int
    followup_main_calls: int
    followup_reference_blocks: int
    followup_reference_calls: int


def _extended_literal(node: ast.AST, known: dict[str, Any]) -> Any:
    if isinstance(node, ast.Subscript):
        container = _extended_literal(node.value, known)
        key = _extended_literal(node.slice, known)
        return container[key]
    return _literal(node, known)


def _extended_configuration_literals(code_source: str) -> dict[str, Any]:
    wanted = {
        "MAIN_DT_MS",
        "REFERENCE_DT_MS",
        "ORIGINAL_WINDOW_MS",
        "LATE_WINDOW_MS",
        "FOLLOWUP_SIMULATION_END_MS",
        "FOLLOWUP_PERIODIC_PROBES",
        "FOLLOWUP_REFERENCE_TYPES",
        "FOLLOWUP_DT_VALUES_MS",
        "FOLLOWUP_SCOPE",
        "FOLLOWUP_REFERENCE_SCOPE",
        "FOLLOWUP_TRACE_FORMAT_VERSION",
        "FOLLOWUP_TRACE_ARCHIVE_SUBDIRECTORY",
        "FOLLOWUP_ANALYSIS_VERSION",
    }
    known: dict[str, Any] = {}
    for statement in ast.parse(code_source).body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            known[target.id] = _extended_literal(statement.value, known)
        except (KeyError, TypeError, ValueError):
            continue
    missing = wanted.difference(known)
    if missing:
        raise ValueError(
            "Extended notebook is missing locked configuration assignments: "
            + ", ".join(sorted(missing))
        )
    return {name: known[name] for name in wanted}


def _resolved_extended_workloads(
    primary_configuration: dict[str, Any],
    extended_configuration: dict[str, Any],
) -> dict[str, ExtendedWorkload]:
    primary_workloads = _resolved_workloads(primary_configuration)
    calls_per_followup_block = len(
        extended_configuration["FOLLOWUP_REFERENCE_TYPES"]
    ) + len(extended_configuration["FOLLOWUP_PERIODIC_PROBES"])
    if calls_per_followup_block != 4:
        raise ValueError(
            "Each prolonged condition/seed block must contain exactly four "
            "TVB calls."
        )

    workloads: dict[str, ExtendedWorkload] = {}
    for mode, mode_configuration in primary_configuration[
        "MODE_CONFIG"
    ].items():
        primary = primary_workloads[mode]
        followup_blocks = (
            len(mode_configuration["severities"])
            * len(mode_configuration["seeds"])
        )
        followup_calls = followup_blocks * calls_per_followup_block
        workloads[mode] = ExtendedWorkload(
            total_calls=primary.total_calls + 2 * followup_calls,
            manifested_calls=(
                primary.manifested_calls + 2 * followup_calls
            ),
            calibration_calls=primary.calibration_calls,
            main_calls=primary.main_calls,
            integration_step_blocks=(
                primary.integration_step_blocks + followup_blocks
            ),
            integration_step_calls=(
                primary.integration_step_calls + followup_calls
            ),
            local_counterfactual_calls=(
                primary.local_counterfactual_calls
            ),
            parameter_sensitivity_calls=(
                primary.parameter_sensitivity_calls
            ),
            spatial_shuffle_calls=primary.spatial_shuffle_calls,
            raw_trace_shards=(
                primary.raw_trace_shards + 2 * followup_blocks
            ),
            primary_total_calls=primary.total_calls,
            primary_manifested_calls=primary.manifested_calls,
            followup_main_blocks=followup_blocks,
            followup_main_calls=followup_calls,
            followup_reference_blocks=followup_blocks,
            followup_reference_calls=followup_calls,
        )
    return workloads


def validate_extended_notebook(
    path: Path = EXTENDED_NOTEBOOK,
) -> NotebookValidation:
    """Validate the immutable extended notebook without running TVB."""

    path = path.resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if path == EXTENDED_NOTEBOOK.resolve() and digest != EXTENDED_SHA256:
        raise ValueError(
            "Extended notebook SHA-256 mismatch: "
            f"expected {EXTENDED_SHA256}, got {digest}."
        )

    notebook = json.loads(payload)
    if notebook.get("nbformat") != 4:
        raise ValueError("Extended notebook must use nbformat 4.")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Extended notebook has no valid cells list.")
    if len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CELL_COUNT} notebook cells, found "
            f"{len(cells)}."
        )

    code_cells = tuple(_code_cells(notebook))
    if len(code_cells) != EXPECTED_CODE_CELL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CODE_CELL_COUNT} code cells, found "
            f"{len(code_cells)}."
        )
    if tuple(index for index, _source in code_cells) != (
        EXPECTED_CODE_CELL_INDICES
    ):
        raise ValueError("Extended notebook code-cell positions changed.")
    for cell_index, source in code_cells:
        compile(source, f"{path.name}:cell-{cell_index + 1}", "exec")
        cell = cells[cell_index]
        if cell.get("execution_count") is not None or cell.get("outputs") != []:
            raise ValueError(
                "Extended notebook code outputs must remain cleared."
            )

    metadata = notebook.get("metadata", {})
    revision = metadata.get("rise_revision", {})
    if revision.get("name") != EXPECTED_REVISION:
        raise ValueError("Extended notebook revision metadata is incorrect.")
    required_revision_values = {
        "canonical_notebook": True,
        "confirmatory_scope": "semantic-versus-episodic proxies only",
        "speech_confirmatory_analysis_removed": True,
        "semantic_expanded_nodes": 13,
        "episodic_expanded_nodes": 19,
        "main_numerical_seeds": 20,
        "final_spatial_shuffles": 50,
        "functional_connectivity_implemented": True,
        "relative_response_latency_implemented": True,
    }
    for key, expected in required_revision_values.items():
        if revision.get(key) != expected:
            raise ValueError(
                f"Extended revision field {key!r} must be {expected!r}."
            )
    artifact = metadata.get("rise", {})
    required_artifact_values = {
        "artifact_version": EXPECTED_ARTIFACT_VERSION,
        "followup_analysis_version": "late-window-dc-matched-v1",
        "primary_method_changed": False,
        "fabricated_data_used": False,
    }
    for key, expected in required_artifact_values.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"Extended artifact field {key!r} must be {expected!r}."
            )

    complete_source = "\n".join(source for _, source in code_cells)
    lowered_source = complete_source.lower()
    for disallowed in (
        "music_minus_speech",
        "speech_network",
        "todo",
        "placeholder",
        "dummy",
        "fabricated_signal",
    ):
        if disallowed in lowered_source:
            raise ValueError(
                f"Extended notebook contains disallowed code: {disallowed}."
            )

    primary_validation = validate_notebook(PRIMARY_NOTEBOOK)
    primary_cells = dict(primary_validation.code_cells)
    extended_cells = dict(code_cells)
    changed_primary_cells = [
        cell_index
        for cell_index in PRIMARY_IMPLEMENTATION_CELL_INDICES
        if primary_cells.get(cell_index) != extended_cells.get(cell_index)
    ]
    if changed_primary_cells:
        raise ValueError(
            "Extended notebook changed locked primary implementation cells: "
            + ", ".join(str(index + 1) for index in changed_primary_cells)
            + "."
        )

    primary_configuration = _configuration_literals(complete_source)
    extended_configuration = _extended_configuration_literals(
        complete_source
    )
    expected_extended_configuration = {
        "MAIN_DT_MS": 0.5,
        "REFERENCE_DT_MS": 0.25,
        "ORIGINAL_WINDOW_MS": (4500.0, 14500.0),
        "LATE_WINDOW_MS": (14500.0, 24500.0),
        "FOLLOWUP_SIMULATION_END_MS": 24500.0,
        "FOLLOWUP_PERIODIC_PROBES": ("2Hz", "5Hz"),
        "FOLLOWUP_REFERENCE_TYPES": ("zero_input", "dc_matched"),
        "FOLLOWUP_DT_VALUES_MS": (0.5, 0.25),
        "FOLLOWUP_SCOPE": "late_window_followup",
        "FOLLOWUP_REFERENCE_SCOPE": (
            "late_window_followup_dt_reference_0.25ms"
        ),
        "FOLLOWUP_TRACE_FORMAT_VERSION": "prolonged_periodic_psp_v1",
        "FOLLOWUP_TRACE_ARCHIVE_SUBDIRECTORY": (
            "late_followup_parcel_traces"
        ),
        "FOLLOWUP_ANALYSIS_VERSION": "late-window-dc-matched-v1",
    }
    if extended_configuration != expected_extended_configuration:
        raise ValueError(
            "Locked extended configuration mismatch: "
            f"expected {expected_extended_configuration}, got "
            f"{extended_configuration}."
        )

    workloads = _resolved_extended_workloads(
        primary_configuration,
        extended_configuration,
    )
    actual_workloads = {
        mode: (
            workload.total_calls,
            workload.manifested_calls,
            workload.primary_total_calls,
            workload.followup_main_calls,
            workload.followup_reference_calls,
            workload.integration_step_blocks,
            workload.integration_step_calls,
            workload.raw_trace_shards,
        )
        for mode, workload in workloads.items()
    }
    if actual_workloads != EXPECTED_WORKLOADS:
        raise ValueError(
            "Locked extended workload mismatch: "
            f"expected {EXPECTED_WORKLOADS}, got {actual_workloads}."
        )

    output_names, figure_output_count = _output_contract(complete_source)
    missing_outputs = FOLLOWUP_OUTPUTS.difference(output_names)
    if missing_outputs:
        raise ValueError(
            "Extended follow-up outputs are missing: "
            + ", ".join(sorted(missing_outputs))
            + "."
        )
    csv_output_count = len(output_names)
    if (csv_output_count, figure_output_count) != EXPECTED_OUTPUT_COUNTS:
        raise ValueError(
            "Locked extended output mismatch: expected 68 CSV tables and "
            f"8 figures, got {csv_output_count} CSV tables and "
            f"{figure_output_count} figures."
        )

    return NotebookValidation(
        path=path,
        sha256=digest,
        notebook=notebook,
        code_cells=code_cells,
        workloads=workloads,
        csv_output_count=csv_output_count,
        figure_output_count=figure_output_count,
    )


def format_extended_validation_summary(
    validation: NotebookValidation,
) -> str:
    lines = [
        "Canonical LateWindowFollowup notebook validated:",
        f"  path: {validation.path}",
        f"  SHA-256: {validation.sha256}",
        (
            f"  cells: {len(validation.notebook['cells'])} total, "
            f"{len(validation.code_cells)} executable"
        ),
        (
            f"  outputs: {validation.csv_output_count} CSV tables, "
            f"{validation.figure_output_count} PNG figures"
        ),
        "  locked TVB workloads:",
    ]
    for mode in ("smoke", "pilot", "final"):
        workload = validation.workloads[mode]
        if not isinstance(workload, ExtendedWorkload):
            raise TypeError("Extended validation contains a primary workload.")
        lines.append(
            f"    {mode}: {workload.total_calls} total calls "
            f"({workload.manifested_calls} manifested) = "
            f"{workload.primary_total_calls} unchanged primary + "
            f"{workload.followup_main_calls} prolonged 0.5 ms + "
            f"{workload.followup_reference_calls} prolonged 0.25 ms; "
            f"integration-reference work units "
            f"{workload.integration_step_blocks} / "
            f"{workload.integration_step_calls} calls; trace shards "
            f"{workload.raw_trace_shards}"
        )
    return "\n".join(lines)


def _extended_execution_code_sha256() -> str:
    files = [
        PROJECT_ROOT / "main_extended.py",
        PROJECT_ROOT / "rise_tvb379" / "__init__.py",
        PROJECT_ROOT / "rise_tvb379" / "notebook_runner.py",
        PROJECT_ROOT / "rise_tvb379" / "extended_notebook_runner.py",
        PROJECT_ROOT / "rise_tvb379" / "run_state.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ExtendedCheckpointDispatcher(CheckpointDispatcher):
    """Apply exact four-call accounting to prolonged follow-up blocks."""

    def __call__(
        self,
        worker_function: Any,
        job_payloads: Sequence[Any],
        shared_args: Sequence[Any],
        description: str,
    ) -> list[Any]:
        worker_name = getattr(
            worker_function,
            "__name__",
            type(worker_function).__name__,
        )
        if worker_name != "execute_late_followup_block":
            return super().__call__(
                worker_function,
                job_payloads,
                shared_args,
                description,
            )

        weighted_jobs: list[dict[str, Any]] = []
        for job in job_payloads:
            if not isinstance(job, Mapping):
                raise TypeError(
                    "A prolonged follow-up checkpoint job must be a mapping."
                )
            weighted_job = dict(job)
            if "probes" in weighted_job:
                raise ValueError(
                    "A prolonged follow-up job unexpectedly defines probes."
                )
            # CheckpointDispatcher counts a control plus every declared probe.
            # Here those four calls are zero input, DC matched, 2 Hz, and 5 Hz.
            weighted_job["probes"] = ("dc_matched", "2Hz", "5Hz")
            weighted_jobs.append(weighted_job)
        return super().__call__(
            worker_function,
            weighted_jobs,
            shared_args,
            description,
        )


class ExtendedRunController(RunController):
    """Keep the saved recovery command specific to the extended launcher."""

    def mark_interrupted(self) -> None:
        self.status["state"] = "interrupted"
        self.status["current_stage"] = "interrupted"
        self.status["exit_utc"] = utc_now()
        self._write_status()
        self.log(
            "Run interrupted. Resume with "
            f"python main_extended.py --resume {self.run_dir}"
        )


def run_extended_notebook(
    *,
    validation: NotebookValidation | None = None,
    resume_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute all extended-notebook code cells in one shared namespace."""

    validation = validation or validate_extended_notebook()
    resume_dir = resume_dir.resolve() if resume_dir is not None else None
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(validation.path),
        "__package__": None,
    }
    total = len(validation.code_cells)
    controller: RunController | None = None
    code_sha256 = _extended_execution_code_sha256()
    try:
        for ordinal, (cell_index, source) in enumerate(
            validation.code_cells,
            start=1,
        ):
            message = (
                f"[notebook {ordinal}/{total}] executing source cell "
                f"{cell_index + 1}/{len(validation.notebook['cells'])}"
            )
            if controller is None:
                print(f"\n{message}", flush=True)
            else:
                controller.begin_source_cell(
                    cell_index + 1,
                    len(validation.notebook["cells"]),
                )
                controller.log(message)
            code = compile(
                source,
                f"{validation.path.name}:cell-{cell_index + 1}",
                "exec",
            )
            exec(code, namespace)

            if cell_index == 6:
                if resume_dir is not None:
                    bootstrap_results_dir = Path(namespace["RESULTS_DIR"])
                    bootstrap_figure_dir = Path(namespace["FIGURE_DIR"])
                    bootstrap_raw_trace_dir = Path(namespace["RAW_TRACE_DIR"])
                    bootstrap_followup_trace_dir = Path(
                        namespace["FOLLOWUP_TRACE_DIR"]
                    )
                    namespace["RESULTS_DIR"] = resume_dir
                    namespace["FIGURE_DIR"] = resume_dir / "figures"
                    namespace["RAW_TRACE_DIR"] = (
                        resume_dir
                        / str(namespace["TRACE_ARCHIVE_SUBDIRECTORY"])
                    )
                    namespace["FOLLOWUP_TRACE_DIR"] = (
                        resume_dir
                        / str(
                            namespace[
                                "FOLLOWUP_TRACE_ARCHIVE_SUBDIRECTORY"
                            ]
                        )
                    )
                    resume_dir.mkdir(parents=True, exist_ok=True)
                    namespace["FIGURE_DIR"].mkdir(parents=True, exist_ok=True)
                    namespace["RAW_TRACE_DIR"].mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    namespace["FOLLOWUP_TRACE_DIR"].mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    if bootstrap_results_dir.resolve() != resume_dir:
                        try:
                            bootstrap_followup_trace_dir.rmdir()
                            bootstrap_raw_trace_dir.rmdir()
                            bootstrap_figure_dir.rmdir()
                            bootstrap_results_dir.rmdir()
                        except OSError:
                            pass

                run_dir = Path(namespace["RESULTS_DIR"]).resolve()
                mode = str(namespace["RUN_MODE"])
                workload = validation.workloads[mode]
                controller_arguments = {
                    "run_dir": run_dir,
                    "mode": mode,
                    "notebook_sha256": validation.sha256,
                    "code_sha256": code_sha256,
                    "environment": runtime_environment(),
                    "planned_total_tvb_calls": workload.total_calls,
                    "planned_integration_step_work_units": (
                        workload.integration_step_blocks
                    ),
                    "planned_raw_trace_shards": workload.raw_trace_shards,
                    "worker_processes": int(namespace["PARALLEL_WORKERS"]),
                }
                if resume_dir is None:
                    controller = ExtendedRunController.create(
                        **controller_arguments
                    )
                else:
                    controller = ExtendedRunController.resume(
                        compatible_predecessor_notebook_sha256s=(),
                        **controller_arguments,
                    )
                namespace["run_parallel_jobs"] = (
                    ExtendedCheckpointDispatcher(namespace, controller)
                )

            if controller is not None and cell_index == 8:
                source_manifest = namespace["source_manifest_df"]
                controller.set_input_hashes(
                    {
                        str(row.source): str(row.sha256)
                        for row in source_manifest.itertuples(index=False)
                    }
                )

            completed_message = (
                f"[notebook {ordinal}/{total}] completed source cell "
                f"{cell_index + 1}"
            )
            if controller is None:
                print(completed_message, flush=True)
            else:
                controller.complete_source_cell(
                    cell_index + 1,
                    len(validation.notebook["cells"]),
                )
                controller.log(completed_message)
        if controller is not None:
            controller.mark_completed()
            archive_path = namespace.get("archive_path")
            if archive_path is not None:
                archive = Path(archive_path)
                refreshed_archive = shutil.make_archive(
                    str(archive.with_suffix("")),
                    "zip",
                    root_dir=controller.run_dir,
                )
                controller.log(
                    f"Refreshed completed result archive: {refreshed_archive}"
                )
    except KeyboardInterrupt:
        if controller is not None:
            controller.mark_interrupted()
        raise
    except BaseException as error:
        if controller is not None:
            controller.mark_failed(error)
        raise
    return namespace
