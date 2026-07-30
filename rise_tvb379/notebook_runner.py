"""Validation and direct execution of the canonical DTGateFixed notebook.

The experiment intentionally has one scientific implementation: the code cells
in the canonical notebook.  This module provides only the non-scientific
command-line bridge needed to validate, compile, and execute those cells.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator

from rise_tvb379.run_state import (
    CheckpointDispatcher,
    RunController,
    execution_code_sha256,
    runtime_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_NOTEBOOK = (
    PROJECT_ROOT
    / "notebooks"
    / "RISE_TVB379_Semantic_Episodic_Final_DTGateFixed_20260730.ipynb"
)
CANONICAL_SHA256 = (
    "c581e82979e158690a7689cca89426818fb2eaa146f9309aa5613d8897e3b92e"
)
EXPECTED_CELL_COUNT = 40
EXPECTED_CODE_CELL_COUNT = 18
EXPECTED_REVISION = "semantic_episodic_final_v3_science_ready"
EXPECTED_WORKLOADS = {
    "smoke": (34, 32),
    "pilot": (85, 79),
    "final": (762, 750),
}


@dataclass(frozen=True)
class Workload:
    """Locked TVB-call counts for one notebook mode."""

    total_calls: int
    manifested_calls: int
    calibration_calls: int
    main_calls: int
    integration_step_blocks: int
    integration_step_calls: int
    local_counterfactual_calls: int
    parameter_sensitivity_calls: int
    spatial_shuffle_calls: int


@dataclass(frozen=True)
class NotebookValidation:
    """Validated canonical notebook and its resolved static workloads."""

    path: Path
    sha256: str
    notebook: dict[str, Any]
    code_cells: tuple[tuple[int, str], ...]
    workloads: dict[str, Workload]
    csv_output_count: int
    figure_output_count: int


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        if not all(isinstance(line, str) for line in source):
            raise ValueError("Notebook cell source contains a non-string line.")
        return "".join(source)
    if isinstance(source, str):
        return source
    raise ValueError("Notebook cell source must be a string or list of strings.")


def _code_cells(notebook: dict[str, Any]) -> Iterator[tuple[int, str]]:
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            yield cell_index, _source_text(cell)


def _literal(node: ast.AST, known: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id]
    if isinstance(node, ast.List):
        return [_literal(item, known) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(item, known) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_literal(item, known) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _literal(key, known): _literal(value, known)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand, known)
    raise ValueError(
        f"Configuration assignment is not a supported literal: "
        f"{ast.dump(node, include_attributes=False)}"
    )


def _configuration_literals(code_source: str) -> dict[str, Any]:
    wanted = {
        "FINAL_NUMERICAL_SEEDS",
        "MODE_CONFIG",
        "PROBES",
        "PERIODIC_PROBES",
        "PULSE_ANALYSIS_END_MS",
    }
    values: dict[str, Any] = {}
    for statement in ast.parse(code_source).body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        values[target.id] = _literal(statement.value, values)
    missing = wanted.difference(values)
    if missing:
        raise ValueError(
            "Canonical notebook is missing locked configuration assignments: "
            + ", ".join(sorted(missing))
        )
    return values


def _resolved_workloads(configuration: dict[str, Any]) -> dict[str, Workload]:
    probes_per_block = 1 + len(configuration["PROBES"])
    periodic_probes_per_block = 1 + len(
        configuration["PERIODIC_PROBES"]
    )
    workloads: dict[str, Workload] = {}
    for mode, mode_config in configuration["MODE_CONFIG"].items():
        calibration = 2 * len(mode_config["calibration_couplings"])
        main = (
            len(mode_config["severities"])
            * len(mode_config["seeds"])
            * probes_per_block
        )
        integration_step_blocks = 2 * len(
            mode_config["dt_check_seeds"]
        )
        integration_step = integration_step_blocks * probes_per_block
        local_counterfactual = (
            len(mode_config["seeds"]) * probes_per_block
        )
        parameter_sensitivity = (
            len(mode_config["sensitivity_scenarios"])
            * 2
            * len(mode_config["sensitivity_seeds"])
            * periodic_probes_per_block
        )
        spatial_shuffle = (
            mode_config["spatial_shuffles"]
            * periodic_probes_per_block
        )
        manifested = (
            main
            + integration_step
            + local_counterfactual
            + parameter_sensitivity
            + spatial_shuffle
        )
        workloads[mode] = Workload(
            total_calls=calibration + manifested,
            manifested_calls=manifested,
            calibration_calls=calibration,
            main_calls=main,
            integration_step_blocks=integration_step_blocks,
            integration_step_calls=integration_step,
            local_counterfactual_calls=local_counterfactual,
            parameter_sensitivity_calls=parameter_sensitivity,
            spatial_shuffle_calls=spatial_shuffle,
        )
    return workloads


def _output_counts(code_source: str) -> tuple[int, int]:
    tree = ast.parse(code_source)
    output_table_count: int | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "output_tables"
            for target in statement.targets
        ):
            continue
        if not isinstance(statement.value, ast.Dict):
            raise ValueError("output_tables must be a dictionary literal.")
        output_table_count = len(statement.value.keys)
        break
    if output_table_count is None:
        raise ValueError("Canonical notebook does not declare output_tables.")

    figure_count = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "savefig"
        for node in ast.walk(tree)
    )
    return output_table_count, figure_count


def validate_notebook(
    path: Path = CANONICAL_NOTEBOOK,
) -> NotebookValidation:
    """Validate identity, structure, code, scope, and locked workloads."""

    path = path.resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if path == CANONICAL_NOTEBOOK.resolve() and digest != CANONICAL_SHA256:
        raise ValueError(
            "Canonical notebook SHA-256 mismatch: "
            f"expected {CANONICAL_SHA256}, got {digest}."
        )

    notebook = json.loads(payload)
    if notebook.get("nbformat") != 4:
        raise ValueError("Canonical notebook must use nbformat 4.")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Canonical notebook has no valid cells list.")
    if len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CELL_COUNT} notebook cells, found "
            f"{len(cells)}."
        )

    revision = notebook.get("metadata", {}).get("rise_revision", {})
    if revision.get("name") != EXPECTED_REVISION:
        raise ValueError("Canonical notebook revision metadata is incorrect.")
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
                f"Canonical notebook revision field {key!r} must be "
                f"{expected!r}."
            )
    amendment = notebook.get("metadata", {}).get(
        "rise_protocol_amendment",
        {},
    )
    required_amendment_values = {
        "date": "2026-07-30",
        "name": "integration-step interaction gate",
        "source_sha256": (
            "d4f6f0b4cf33575c86912368d22a9bd4c6e8567959e23cee1f3058f9ec85e907"
        ),
        "main_dt_ms": 0.5,
        "reference_dt_ms": 0.25,
        "final_reference_seed_count": 20,
        "planned_final_tvb_calls": 762,
        "requires_matching_95pct_interval_conclusion": True,
    }
    for key, expected in required_amendment_values.items():
        if amendment.get(key) != expected:
            raise ValueError(
                f"Canonical protocol-amendment field {key!r} must be "
                f"{expected!r}."
            )

    code_cells = tuple(_code_cells(notebook))
    if len(code_cells) != EXPECTED_CODE_CELL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CODE_CELL_COUNT} code cells, found "
            f"{len(code_cells)}."
        )
    for cell_index, source in code_cells:
        compile(
            source,
            f"{path.name}:cell-{cell_index + 1}",
            "exec",
        )

    complete_source = "\n".join(source for _, source in code_cells)
    configuration = _configuration_literals(complete_source)
    if len(configuration["FINAL_NUMERICAL_SEEDS"]) != 20:
        raise ValueError("Final mode must use 20 numerical seeds.")
    if configuration["PULSE_ANALYSIS_END_MS"] != 6000.0:
        raise ValueError("Pulse analysis must extend through 6000 ms.")

    workloads = _resolved_workloads(configuration)
    actual_counts = {
        mode: (workload.total_calls, workload.manifested_calls)
        for mode, workload in workloads.items()
    }
    if actual_counts != EXPECTED_WORKLOADS:
        raise ValueError(
            "Locked workload mismatch: "
            f"expected {EXPECTED_WORKLOADS}, got {actual_counts}."
        )
    csv_output_count, figure_output_count = _output_counts(complete_source)
    if csv_output_count != 45 or figure_output_count != 6:
        raise ValueError(
            "Locked output mismatch: expected 45 CSV tables and 6 figures, "
            f"got {csv_output_count} CSV tables and {figure_output_count} "
            "figures."
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


def format_validation_summary(validation: NotebookValidation) -> str:
    lines = [
        "Canonical DTGateFixed notebook validated:",
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
        lines.append(
            f"    {mode}: {workload.total_calls} total calls "
            f"({workload.manifested_calls} manifested); integration step "
            f"{workload.integration_step_blocks} work units / "
            f"{workload.integration_step_calls} calls"
        )
    return "\n".join(lines)


def run_notebook(
    *,
    validation: NotebookValidation | None = None,
    resume_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute every canonical code cell in one shared Python namespace."""

    validation = validation or validate_notebook()
    resume_dir = resume_dir.resolve() if resume_dir is not None else None
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(validation.path),
        "__package__": None,
    }
    total = len(validation.code_cells)
    controller: RunController | None = None
    code_sha256 = execution_code_sha256(PROJECT_ROOT)
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
                    namespace["RESULTS_DIR"] = resume_dir
                    namespace["FIGURE_DIR"] = resume_dir / "figures"
                    resume_dir.mkdir(parents=True, exist_ok=True)
                    namespace["FIGURE_DIR"].mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    if bootstrap_results_dir.resolve() != resume_dir:
                        try:
                            bootstrap_figure_dir.rmdir()
                            bootstrap_results_dir.rmdir()
                        except OSError:
                            pass

                run_dir = Path(namespace["RESULTS_DIR"]).resolve()
                mode = str(namespace["RUN_MODE"])
                environment = runtime_environment()
                controller_arguments = {
                    "run_dir": run_dir,
                    "mode": mode,
                    "notebook_sha256": validation.sha256,
                    "code_sha256": code_sha256,
                    "environment": environment,
                    "planned_total_tvb_calls": (
                        validation.workloads[mode].total_calls
                    ),
                    "planned_integration_step_work_units": (
                        validation.workloads[
                            mode
                        ].integration_step_blocks
                    ),
                    "worker_processes": int(
                        namespace["PARALLEL_WORKERS"]
                    ),
                }
                if resume_dir is None:
                    controller = RunController.create(
                        **controller_arguments
                    )
                else:
                    controller = RunController.resume(
                        **controller_arguments
                    )
                namespace["run_parallel_jobs"] = CheckpointDispatcher(
                    namespace,
                    controller,
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
                    f"Refreshed completed result archive: "
                    f"{refreshed_archive}"
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
