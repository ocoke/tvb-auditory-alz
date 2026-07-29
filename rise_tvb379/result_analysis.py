"""Read-only post-processing for completed RISE TVB379 result folders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PERIODIC_PROBES = ("2Hz", "5Hz")
PRIMARY_CONTRAST = "music_minus_speech_log2_change"
SECONDARY_CONTRAST = "semantic_minus_episodic_log2_change"
FIT_R2_WARNING_THRESHOLD = 0.10
EXPECTED_FINAL_SEEDS = (11, 23, 37, 53, 71)
EXPECTED_FINAL_PROBES = ("2Hz", "5Hz", "pulse")
EXPECTED_FINAL_SEVERITIES = (0.0, 0.5, 1.0)
EXPECTED_FINAL_SHUFFLES = 100
EXPECTED_FINAL_MANIFESTED_SIMULATIONS = 430
EXPECTED_FINAL_TVB_CALLS = 442
EXPECTED_DECLARED_NETWORKS = 17

TABLE_FILES: Mapping[str, str] = {
    "calibration": "baseline_coupling_calibration.csv",
    "data_quality": "data_quality_checks.csv",
    "dt_convergence": "integration_step_check.csv",
    "local_contrasts": "local_fixed_contrasts.csv",
    "main_contrasts": "main_music_minus_speech_contrasts.csv",
    "main_normalized": "main_network_metrics_normalized.csv",
    "main_network": "main_network_metrics.csv",
    "main_node": "main_node_metrics.csv",
    "matched_null": "matched_control_null_metrics.csv",
    "matched_summary": "matched_control_null_summary.csv",
    "memory_matched_null": "memory_matched_control_null_metrics.csv",
    "memory_matched_summary": "memory_matched_control_null_summary.csv",
    "roi_definitions": "roi_definitions.csv",
    "run_manifest": "run_manifest.csv",
    "sensitivity": "sensitivity_contrasts.csv",
    "shuffle": "spatial_shuffle_contrasts.csv",
    "source_manifest": "source_manifest.csv",
}

REQUIRED_COLUMNS: Mapping[str, set[str]] = {
    "calibration": {
        "global_coupling",
        "a1_rms",
        "music_transfer",
        "speech_transfer",
        "balanced_target_score",
        "max_abs_evoked",
        "wall_seconds",
        "worker_pid",
    },
    "data_quality": {"check", "result"},
    "dt_convergence": {
        "severity",
        "probe",
        "network",
        "required_for_inference",
        "relative_difference",
        "convergence_passed",
        "transfer_dt_0.5ms",
        "transfer_dt_0.25ms",
        "median_target_fit_r_squared_dt_0.5ms",
        "median_target_fit_r_squared_dt_0.25ms",
        "fit_r_squared_difference",
    },
    "local_contrasts": {
        "variant",
        "severity",
        "seed",
        "probe",
        PRIMARY_CONTRAST,
        SECONDARY_CONTRAST,
    },
    "main_contrasts": {
        "condition",
        "severity",
        "seed",
        "probe",
        "music",
        "speech",
        "music_semantic_task_associated",
        "music_episodic_task_associated",
        PRIMARY_CONTRAST,
        SECONDARY_CONTRAST,
    },
    "main_normalized": {
        "severity",
        "seed",
        "probe",
        "network",
        "log2_transfer_vs_baseline",
    },
    "main_network": {
        "severity",
        "seed",
        "probe",
        "network",
        "transfer",
        "median_target_fit_r_squared",
    },
    "main_node": {
        "severity",
        "seed",
        "probe",
        "region_index",
        "region_label",
        "response",
        "fit_r_squared",
        "b_value",
    },
    "matched_null": {
        "set_id",
        "seed",
        "probe",
        "null_music_minus_speech",
        "mean_standardized_match_distance",
    },
    "matched_summary": {
        "seed",
        "probe",
        "observed_contrast",
        "null_median",
        "null_5th_percentile",
        "null_95th_percentile",
        "observed_percentile_within_simulation_null",
    },
    "memory_matched_null": {
        "set_id",
        "seed",
        "probe",
        "null_semantic_minus_episodic",
        "mean_standardized_match_distance",
    },
    "memory_matched_summary": {
        "seed",
        "probe",
        "observed_contrast",
        "null_median",
        "null_5th_percentile",
        "null_95th_percentile",
        "observed_percentile_within_simulation_null",
    },
    "roi_definitions": {
        "network",
        "analysis_role",
        "label",
        "zero_based_index",
        "interpretation",
    },
    "run_manifest": {
        "scope",
        "variant",
        "condition",
        "severity",
        "seed",
        "probe",
        "simulation_type",
        "wall_seconds",
        "max_abs_psp",
        "max_abs_evoked",
        "worker_pid",
    },
    "sensitivity": {
        "variant",
        "severity",
        "seed",
        "probe",
        PRIMARY_CONTRAST,
        SECONDARY_CONTRAST,
    },
    "shuffle": {
        "variant",
        "severity",
        "seed",
        "probe",
        PRIMARY_CONTRAST,
        SECONDARY_CONTRAST,
    },
    "source_manifest": {"source", "path", "sha256", "retrieved_from"},
}


class ResultValidationError(RuntimeError):
    """Raised before plotting when a completed-result folder is inconsistent."""


@dataclass(frozen=True, slots=True)
class ResultBundle:
    """Loaded metadata and result tables for one completed run."""

    run_dir: Path
    results_dir: Path
    metadata: dict[str, Any]
    tables: Mapping[str, pd.DataFrame]


@dataclass(frozen=True, slots=True)
class AnalysisProducts:
    """Derived reader-facing tables plus deterministic findings."""

    tables: Mapping[str, pd.DataFrame]
    findings: pd.DataFrame


def load_result_bundle(run_dir: str | Path) -> ResultBundle:
    """Load the files required by post-processing without modifying the run."""

    run_path = Path(run_dir).expanduser().resolve()
    results_dir = run_path / "results"
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory has no results subdirectory: {results_dir}"
        )

    metadata_path = results_dir / "experiment_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Required result metadata is missing: {metadata_path}"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResultValidationError(
            f"Result metadata is unreadable: {metadata_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise ResultValidationError("experiment_metadata.json must be an object")

    tables: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    for name, filename in TABLE_FILES.items():
        path = results_dir / filename
        if not path.is_file():
            missing_files.append(filename)
            continue
        try:
            tables[name] = pd.read_csv(path)
        except Exception as error:
            raise ResultValidationError(
                f"Could not read required table {path}: {error}"
            ) from error
    if missing_files:
        raise FileNotFoundError(
            "Required result tables are missing: "
            + ", ".join(sorted(missing_files))
        )
    return ResultBundle(
        run_dir=run_path,
        results_dir=results_dir,
        metadata=metadata,
        tables=tables,
    )


def _canonical_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0"})
    if unknown:
        raise ResultValidationError(
            f"Boolean column contains invalid values: {unknown}"
        )
    return normalized.isin({"true", "1"})


def _finite(frame: pd.DataFrame, columns: Iterable[str]) -> bool:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(values.to_numpy(dtype=float)).all())


def _duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int:
    return int(frame.duplicated(columns, keep=False).sum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_result_bundle(bundle: ResultBundle) -> pd.DataFrame:
    """Validate the scientific and structural assumptions used by analysis."""

    checks: list[dict[str, object]] = []

    def add(
        check: str,
        passed: bool,
        observed: object,
        expected: object,
        *,
        category: str,
    ) -> None:
        checks.append(
            {
                "category": category,
                "check": check,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    for name, required in REQUIRED_COLUMNS.items():
        actual = set(bundle.tables[name].columns)
        missing = sorted(required - actual)
        add(
            f"{TABLE_FILES[name]} required columns",
            not missing,
            "none" if not missing else ";".join(missing),
            "none missing",
            category="schema",
        )

    metadata = bundle.metadata
    workload = metadata.get("workload", {})
    model = metadata.get("model", {})
    resolved = metadata.get("resolved_config", {})
    mode_config = resolved.get("mode_config", {})
    try:
        expected_manifest = int(workload["manifest"])
        expected_total = int(workload["total"])
        expected_shuffles = int(model["spatial_shuffles"])
        expected_seeds = tuple(int(value) for value in model["numerical_seeds"])
        expected_probes = tuple(str(value) for value in model["probes"])
        expected_severities = tuple(
            float(value) for value in mode_config["severities"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ResultValidationError(
            "Metadata lacks resolved workload, model, seed, probe, or severity "
            "information required for validation."
        ) from error

    add(
        "resolved run mode",
        metadata.get("run_mode") == "final"
        and resolved.get("mode") == "final",
        (metadata.get("run_mode"), resolved.get("mode")),
        ("final", "final"),
        category="metadata",
    )
    add(
        "final manifest workload declaration",
        expected_manifest == EXPECTED_FINAL_MANIFESTED_SIMULATIONS,
        expected_manifest,
        EXPECTED_FINAL_MANIFESTED_SIMULATIONS,
        category="metadata",
    )
    add(
        "final total-call workload declaration",
        expected_total == EXPECTED_FINAL_TVB_CALLS,
        expected_total,
        EXPECTED_FINAL_TVB_CALLS,
        category="metadata",
    )
    add(
        "final spatial-shuffle declaration",
        expected_shuffles == EXPECTED_FINAL_SHUFFLES,
        expected_shuffles,
        EXPECTED_FINAL_SHUFFLES,
        category="metadata",
    )
    add(
        "final numerical-seed declaration",
        set(expected_seeds) == set(EXPECTED_FINAL_SEEDS),
        expected_seeds,
        EXPECTED_FINAL_SEEDS,
        category="metadata",
    )
    add(
        "final probe declaration",
        set(expected_probes) == set(EXPECTED_FINAL_PROBES),
        expected_probes,
        EXPECTED_FINAL_PROBES,
        category="metadata",
    )
    add(
        "final severity declaration",
        set(expected_severities) == set(EXPECTED_FINAL_SEVERITIES),
        expected_severities,
        EXPECTED_FINAL_SEVERITIES,
        category="metadata",
    )
    add(
        "model and mode seed consistency",
        set(expected_seeds)
        == {
            int(value)
            for value in mode_config.get("seeds", ())
        },
        tuple(mode_config.get("seeds", ())),
        expected_seeds,
        category="metadata",
    )
    add(
        "model and mode shuffle consistency",
        expected_shuffles == int(mode_config.get("spatial_shuffles", -1)),
        mode_config.get("spatial_shuffles"),
        expected_shuffles,
        category="metadata",
    )

    tables = bundle.tables
    manifest = tables["run_manifest"]
    calibration = tables["calibration"]
    actual_total = len(manifest) + 2 * len(calibration)
    add(
        "manifested simulation count",
        len(manifest) == expected_manifest,
        len(manifest),
        expected_manifest,
        category="completeness",
    )
    add(
        "total TVB call count",
        actual_total == expected_total,
        actual_total,
        expected_total,
        category="completeness",
    )
    actual_seeds = tuple(
        sorted(
            int(value)
            for value in tables["main_contrasts"]["seed"].dropna().unique()
        )
    )
    actual_probes = tuple(
        sorted(str(value) for value in tables["main_contrasts"]["probe"].unique())
    )
    actual_severities = tuple(
        sorted(
            float(value)
            for value in tables["main_contrasts"]["severity"].unique()
        )
    )
    add(
        "main numerical seeds",
        set(actual_seeds) == set(expected_seeds),
        actual_seeds,
        expected_seeds,
        category="completeness",
    )
    add(
        "main probes",
        set(actual_probes) == set(expected_probes),
        actual_probes,
        expected_probes,
        category="completeness",
    )
    add(
        "main severities",
        set(actual_severities) == set(expected_severities),
        actual_severities,
        expected_severities,
        category="completeness",
    )
    actual_shuffles = int(tables["shuffle"]["variant"].nunique())
    add(
        "spatial shuffle count",
        actual_shuffles
        == expected_shuffles
        == EXPECTED_FINAL_SHUFFLES,
        actual_shuffles,
        EXPECTED_FINAL_SHUFFLES,
        category="completeness",
    )
    declared_networks = int(tables["main_network"]["network"].nunique())
    add(
        "declared network count",
        declared_networks == EXPECTED_DECLARED_NETWORKS,
        declared_networks,
        EXPECTED_DECLARED_NETWORKS,
        category="completeness",
    )
    expected_main_rows = (
        len(EXPECTED_FINAL_SEEDS)
        * len(EXPECTED_FINAL_PROBES)
        * len(EXPECTED_FINAL_SEVERITIES)
    )
    add(
        "main contrast grid coverage",
        len(tables["main_contrasts"]) == expected_main_rows,
        len(tables["main_contrasts"]),
        expected_main_rows,
        category="completeness",
    )
    expected_network_rows = expected_main_rows * EXPECTED_DECLARED_NETWORKS
    for name in ("main_network", "main_normalized"):
        add(
            f"{TABLE_FILES[name]} grid coverage",
            len(tables[name]) == expected_network_rows,
            len(tables[name]),
            expected_network_rows,
            category="completeness",
        )
    expected_dt_rows = (
        2 * len(PERIODIC_PROBES) * EXPECTED_DECLARED_NETWORKS
    )
    add(
        "integration-step matrix coverage",
        len(tables["dt_convergence"]) == expected_dt_rows,
        len(tables["dt_convergence"]),
        expected_dt_rows,
        category="completeness",
    )
    expected_shuffle_rows = EXPECTED_FINAL_SHUFFLES * len(PERIODIC_PROBES)
    add(
        "spatial-shuffle contrast coverage",
        len(tables["shuffle"]) == expected_shuffle_rows,
        len(tables["shuffle"]),
        expected_shuffle_rows,
        category="completeness",
    )

    source_manifest = tables["source_manifest"]
    declared_source_hashes = dict(
        zip(
            source_manifest["source"].astype(str),
            source_manifest["sha256"].astype(str),
            strict=True,
        )
    )
    metadata_source_hashes = {
        str(key): str(value)
        for key, value in metadata.get("source_hashes", {}).items()
    }
    add(
        "five source inputs declared",
        len(declared_source_hashes) == 5,
        len(declared_source_hashes),
        5,
        category="inputs",
    )
    add(
        "source manifest matches metadata hashes",
        declared_source_hashes == metadata_source_hashes,
        "matched"
        if declared_source_hashes == metadata_source_hashes
        else "mismatch",
        "matched",
        category="inputs",
    )
    for row in source_manifest.itertuples(index=False):
        input_path = bundle.run_dir / "inputs" / Path(str(row.path)).name
        present = input_path.is_file()
        actual_hash = _sha256(input_path) if present else "missing"
        expected_hash = str(row.sha256)
        add(
            f"verified input {input_path.name}",
            present and actual_hash == expected_hash,
            actual_hash,
            expected_hash,
            category="inputs",
        )

    quality = tables["data_quality"].set_index("check")["result"].astype(str)
    required_quality_checks = (
        "SC is 379 x 379",
        "379 unique labels",
        "379 amyloid values",
        "SC values finite",
        "SC values nonnegative",
        "amyloid values finite",
        "left cortex occupies 0:180",
        "right cortex occupies 180:360",
        "brainstem is final parcel",
    )
    failed_quality = [
        check
        for check in required_quality_checks
        if check not in quality.index
        or quality.loc[check].strip().lower() != "true"
    ]
    add(
        "recorded input data-quality checks",
        not failed_quality,
        "all passed" if not failed_quality else ";".join(failed_quality),
        "all passed",
        category="inputs",
    )

    unique_contracts = {
        "main_contrasts": ["severity", "seed", "probe"],
        "main_normalized": [
            "scope",
            "variant",
            "severity",
            "seed",
            "probe",
            "network",
        ],
        "main_network": [
            "scope",
            "variant",
            "severity",
            "seed",
            "probe",
            "network",
        ],
        "local_contrasts": ["variant", "severity", "seed", "probe"],
        "matched_summary": ["seed", "probe"],
        "memory_matched_summary": ["seed", "probe"],
        "sensitivity": ["variant", "severity", "seed", "probe"],
        "shuffle": ["variant", "severity", "seed", "probe"],
        "dt_convergence": ["severity", "probe", "network"],
        "run_manifest": [
            "scope",
            "variant",
            "severity",
            "seed",
            "probe",
            "simulation_type",
        ],
    }
    for name, columns in unique_contracts.items():
        missing_key_columns = sorted(set(columns) - set(tables[name].columns))
        duplicates = (
            -1
            if missing_key_columns
            else _duplicate_count(tables[name], columns)
        )
        add(
            f"{TABLE_FILES[name]} unique scientific keys",
            duplicates == 0,
            duplicates,
            0,
            category="uniqueness",
        )

    finite_contracts = {
        "main_contrasts": [
            "music",
            "speech",
            "music_semantic_task_associated",
            "music_episodic_task_associated",
            PRIMARY_CONTRAST,
            SECONDARY_CONTRAST,
        ],
        "main_normalized": ["log2_transfer_vs_baseline"],
        "main_network": ["transfer"],
        "main_node": ["response", "b_value"],
        "local_contrasts": [PRIMARY_CONTRAST, SECONDARY_CONTRAST],
        "matched_null": ["null_music_minus_speech"],
        "memory_matched_null": ["null_semantic_minus_episodic"],
        "sensitivity": [PRIMARY_CONTRAST, SECONDARY_CONTRAST],
        "shuffle": [PRIMARY_CONTRAST, SECONDARY_CONTRAST],
        "dt_convergence": [
            "relative_difference",
            "transfer_dt_0.5ms",
            "transfer_dt_0.25ms",
            "fit_r_squared_difference",
        ],
        "run_manifest": ["wall_seconds", "max_abs_psp"],
        "calibration": [
            "a1_rms",
            "music_transfer",
            "speech_transfer",
            "max_abs_evoked",
        ],
    }
    for name, columns in finite_contracts.items():
        missing = sorted(set(columns) - set(tables[name].columns))
        passed = not missing and _finite(tables[name], columns)
        add(
            f"{TABLE_FILES[name]} critical values finite",
            passed,
            "finite" if passed else "nonfinite or missing",
            "finite",
            category="numerical",
        )

    periodic_network = tables["main_network"][
        tables["main_network"]["probe"].isin(PERIODIC_PROBES)
    ]
    fit_finite = _finite(
        periodic_network,
        ["median_target_fit_r_squared"],
    )
    add(
        "periodic target-fit R-squared values finite",
        fit_finite,
        "finite" if fit_finite else "nonfinite",
        "finite",
        category="numerical",
    )

    baseline = tables["main_contrasts"][
        np.isclose(tables["main_contrasts"]["severity"], 0.0)
    ]
    baseline_abs = float(
        baseline[[PRIMARY_CONTRAST, SECONDARY_CONTRAST]]
        .abs()
        .to_numpy()
        .max(initial=0.0)
    )
    add(
        "baseline contrasts normalize to zero",
        baseline_abs <= 1e-12,
        baseline_abs,
        "<= 1e-12",
        category="scientific",
    )

    dt = tables["dt_convergence"].copy()
    dt["required_for_inference"] = _canonical_bool(
        dt["required_for_inference"]
    )
    dt["convergence_passed"] = _canonical_bool(dt["convergence_passed"])
    required_dt = dt[dt["required_for_inference"]]
    failed_required = int((~required_dt["convergence_passed"]).sum())
    add(
        "inferential integration-step convergence",
        failed_required == 0 and len(required_dt) > 0,
        failed_required,
        0,
        category="scientific",
    )

    max_abs_psp = float(manifest["max_abs_psp"].max())
    add(
        "prespecified PSP safety bound",
        max_abs_psp <= 100.0,
        max_abs_psp,
        "<= 100",
        category="scientific",
    )

    summary = pd.DataFrame(checks)
    failures = summary[~summary["passed"]]
    if not failures.empty:
        detail = "\n".join(
            f"- {row.check}: observed={row.observed!r}, "
            f"expected={row.expected!r}"
            for row in failures.itertuples(index=False)
        )
        raise ResultValidationError(
            "Result validation failed before analysis:\n" + detail
        )
    return summary


def _sign_direction(values: pd.Series) -> tuple[str, bool]:
    array = values.to_numpy(dtype=float)
    if np.all(array > 0):
        return "positive", True
    if np.all(array < 0):
        return "negative", True
    if np.allclose(array, 0.0, atol=1e-12):
        return "zero", True
    return "mixed", False


def _summary_row(
    values: pd.Series,
    *,
    extra: Mapping[str, object],
) -> dict[str, object]:
    direction, consistent = _sign_direction(values)
    return {
        **extra,
        "n_seeds": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(values.std(ddof=1))
        if values.size > 1
        else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_count": int((values > 0).sum()),
        "negative_count": int((values < 0).sum()),
        "sign_direction": direction,
        "sign_consistent": consistent,
    }


def build_contrast_summaries(
    main_contrasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize components and contrasts over ordered perturbation levels."""

    series = (
        ("primary", "component", "music", "Music proxy"),
        ("primary", "component", "speech", "Speech proxy"),
        ("primary", "contrast", PRIMARY_CONTRAST, "Music minus speech"),
        (
            "secondary",
            "component",
            "music_semantic_task_associated",
            "Semantic-task-associated proxy",
        ),
        (
            "secondary",
            "component",
            "music_episodic_task_associated",
            "Episodic-task-associated proxy",
        ),
        (
            "secondary",
            "contrast",
            SECONDARY_CONTRAST,
            "Semantic-associated minus episodic-associated",
        ),
    )
    periodic = main_contrasts[
        main_contrasts["probe"].isin(PERIODIC_PROBES)
    ]
    trajectory_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []
    for family, metric_type, column, label in series:
        for (probe, severity, condition), group in periodic.groupby(
            ["probe", "severity", "condition"],
            sort=True,
        ):
            trajectory_rows.append(
                _summary_row(
                    group[column],
                    extra={
                        "contrast_family": family,
                        "metric_type": metric_type,
                        "series": label,
                        "source_column": column,
                        "probe": probe,
                        "severity": float(severity),
                        "condition": condition,
                    },
                )
            )
        endpoint = periodic[np.isclose(periodic["severity"], 1.0)]
        for probe, group in endpoint.groupby("probe", sort=True):
            endpoint_rows.append(
                _summary_row(
                    group[column],
                    extra={
                        "contrast_family": family,
                        "metric_type": metric_type,
                        "series": label,
                        "source_column": column,
                        "probe": probe,
                    },
                )
            )
    return pd.DataFrame(trajectory_rows), pd.DataFrame(endpoint_rows)


def build_counterfactual_attenuation(
    main_contrasts: pd.DataFrame,
    local_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    """Pair full-field and local-fixed contrasts by seed and probe."""

    main = main_contrasts[
        np.isclose(main_contrasts["severity"], 1.0)
        & main_contrasts["probe"].isin(PERIODIC_PROBES)
    ]
    specifications = (
        (
            "primary",
            PRIMARY_CONTRAST,
            "primary_local_fixed_endpoint",
        ),
        (
            "secondary",
            SECONDARY_CONTRAST,
            "memory_local_fixed_endpoint",
        ),
    )
    frames: list[pd.DataFrame] = []
    for family, column, variant in specifications:
        full = main[["seed", "probe", column]].rename(
            columns={column: "full_field_contrast"}
        )
        fixed = local_contrasts[
            (local_contrasts["variant"] == variant)
            & local_contrasts["probe"].isin(PERIODIC_PROBES)
        ][["seed", "probe", column]].rename(
            columns={column: "local_fixed_contrast"}
        )
        paired = full.merge(fixed, on=["seed", "probe"], validate="one_to_one")
        if len(paired) != len(full):
            raise ResultValidationError(
                f"Counterfactual coverage is incomplete for {family}."
            )
        denominator = paired["full_field_contrast"].abs()
        if bool((denominator <= 1e-15).any()):
            raise ResultValidationError(
                f"Cannot calculate {family} attenuation from a zero full-field "
                "contrast."
            )
        paired["contrast_family"] = family
        paired["local_fixed_variant"] = variant
        paired["absolute_full_field"] = denominator
        paired["absolute_local_fixed"] = paired["local_fixed_contrast"].abs()
        paired["absolute_change"] = (
            paired["absolute_full_field"] - paired["absolute_local_fixed"]
        )
        paired["attenuation_percent"] = (
            100.0
            * (
                1.0
                - paired["absolute_local_fixed"]
                / paired["absolute_full_field"]
            )
        )
        paired["direction_preserved"] = (
            np.sign(paired["full_field_contrast"])
            == np.sign(paired["local_fixed_contrast"])
        )
        frames.append(paired)
    result = pd.concat(frames, ignore_index=True)
    return result[
        [
            "contrast_family",
            "seed",
            "probe",
            "local_fixed_variant",
            "full_field_contrast",
            "local_fixed_contrast",
            "absolute_full_field",
            "absolute_local_fixed",
            "absolute_change",
            "attenuation_percent",
            "direction_preserved",
        ]
    ].sort_values(["contrast_family", "probe", "seed"]).reset_index(drop=True)


def _observed_endpoint(
    main_contrasts: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    return main_contrasts[
        np.isclose(main_contrasts["severity"], 1.0)
        & main_contrasts["probe"].isin(PERIODIC_PROBES)
    ][["seed", "probe", column]].rename(columns={column: "observed_contrast"})


def build_matched_null_context(
    main_contrasts: pd.DataFrame,
    primary_null: pd.DataFrame,
    secondary_null: pd.DataFrame,
) -> pd.DataFrame:
    """Compute descriptive matched-null ranks without inferential p-values."""

    specifications = (
        (
            "primary",
            PRIMARY_CONTRAST,
            primary_null,
            "null_music_minus_speech",
        ),
        (
            "secondary",
            SECONDARY_CONTRAST,
            secondary_null,
            "null_semantic_minus_episodic",
        ),
    )
    rows: list[dict[str, object]] = []
    for family, observed_column, null_frame, null_column in specifications:
        observed = _observed_endpoint(main_contrasts, observed_column)
        observed_lookup = {
            (int(row.seed), str(row.probe)): float(row.observed_contrast)
            for row in observed.itertuples(index=False)
        }
        for (seed, probe), group in null_frame.groupby(
            ["seed", "probe"],
            sort=True,
        ):
            values = group[null_column].to_numpy(dtype=float)
            key = (int(seed), str(probe))
            if key not in observed_lookup:
                raise ResultValidationError(
                    f"Matched null has no observed endpoint for {family} {key}."
                )
            observed_value = observed_lookup[key]
            q05, q95 = np.quantile(values, [0.05, 0.95])
            rows.append(
                {
                    "contrast_family": family,
                    "seed": int(seed),
                    "probe": str(probe),
                    "observed_contrast": observed_value,
                    "null_median": float(np.median(values)),
                    "null_5th_percentile": float(q05),
                    "null_95th_percentile": float(q95),
                    "null_minimum": float(values.min()),
                    "null_maximum": float(values.max()),
                    "observed_empirical_percentile": float(
                        100.0 * np.mean(values <= observed_value)
                    ),
                    "inside_central_90_percent": bool(
                        q05 <= observed_value <= q95
                    ),
                    "control_sets": int(values.size),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["contrast_family", "probe", "seed"]
    ).reset_index(drop=True)


def build_spatial_shuffle_context(
    main_contrasts: pd.DataFrame,
    shuffle: pd.DataFrame,
    *,
    main_seed: int,
) -> pd.DataFrame:
    """Summarize spatial-placement shuffles for the main numerical seed."""

    observed = main_contrasts[
        np.isclose(main_contrasts["severity"], 1.0)
        & (main_contrasts["seed"] == main_seed)
        & main_contrasts["probe"].isin(PERIODIC_PROBES)
    ].set_index("probe")
    rows: list[dict[str, object]] = []
    for family, column in (
        ("primary", PRIMARY_CONTRAST),
        ("secondary", SECONDARY_CONTRAST),
    ):
        for probe, group in shuffle.groupby("probe", sort=True):
            values = group[column].to_numpy(dtype=float)
            observed_value = float(observed.loc[probe, column])
            q05, q95 = np.quantile(values, [0.05, 0.95])
            rows.append(
                {
                    "contrast_family": family,
                    "seed": int(main_seed),
                    "probe": str(probe),
                    "observed_contrast": observed_value,
                    "shuffle_median": float(np.median(values)),
                    "shuffle_5th_percentile": float(q05),
                    "shuffle_95th_percentile": float(q95),
                    "shuffle_minimum": float(values.min()),
                    "shuffle_maximum": float(values.max()),
                    "observed_empirical_percentile": float(
                        100.0 * np.mean(values <= observed_value)
                    ),
                    "inside_central_90_percent": bool(
                        q05 <= observed_value <= q95
                    ),
                    "spatial_shuffles": int(values.size),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["contrast_family", "probe"]
    ).reset_index(drop=True)


def build_sensitivity_summary(
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Compare each high-endpoint scenario with the main G60 result."""

    endpoint = sensitivity[
        np.isclose(sensitivity["severity"], 1.0)
        & sensitivity["probe"].isin(PERIODIC_PROBES)
    ].copy()
    reference_variant = "G60_input_0.02"
    rows: list[dict[str, object]] = []
    for family, column in (
        ("primary", PRIMARY_CONTRAST),
        ("secondary", SECONDARY_CONTRAST),
    ):
        for probe, probe_frame in endpoint.groupby("probe", sort=True):
            reference = probe_frame[
                probe_frame["variant"] == reference_variant
            ]
            if len(reference) != 1:
                raise ResultValidationError(
                    f"Sensitivity table must contain one {reference_variant} "
                    f"row for {family} {probe}."
                )
            reference_value = float(reference[column].iloc[0])
            for row in probe_frame.itertuples(index=False):
                value = float(getattr(row, column))
                rows.append(
                    {
                        "contrast_family": family,
                        "variant": str(row.variant),
                        "probe": str(probe),
                        "contrast": value,
                        "reference_variant": reference_variant,
                        "reference_contrast": reference_value,
                        "difference_from_reference": value - reference_value,
                        "same_sign_as_reference": bool(
                            np.sign(value) == np.sign(reference_value)
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["contrast_family", "probe", "variant"]
    ).reset_index(drop=True)


def build_convergence_signal_quality(
    dt_convergence: pd.DataFrame,
    main_network: pd.DataFrame,
) -> pd.DataFrame:
    """Join convergence summaries with high-endpoint fit diagnostics."""

    dt = dt_convergence.copy()
    dt["required_for_inference"] = _canonical_bool(
        dt["required_for_inference"]
    )
    dt["convergence_passed"] = _canonical_bool(dt["convergence_passed"])
    convergence = (
        dt.groupby(["probe", "network"], as_index=False)
        .agg(
            required_for_inference=("required_for_inference", "max"),
            max_relative_difference=("relative_difference", "max"),
            median_relative_difference=("relative_difference", "median"),
            max_fit_r_squared_difference=(
                "fit_r_squared_difference",
                "max",
            ),
            convergence_all_passed=("convergence_passed", "all"),
            convergence_endpoints=("severity", "nunique"),
        )
    )
    endpoint = main_network[
        np.isclose(main_network["severity"], 1.0)
        & main_network["probe"].isin(PERIODIC_PROBES)
    ]
    fit = (
        endpoint.groupby(["probe", "network"], as_index=False)
        .agg(
            fit_r_squared_median=(
                "median_target_fit_r_squared",
                "median",
            ),
            fit_r_squared_minimum=(
                "median_target_fit_r_squared",
                "min",
            ),
            fit_r_squared_maximum=(
                "median_target_fit_r_squared",
                "max",
            ),
            numerical_seeds=("seed", "nunique"),
        )
    )
    result = convergence.merge(
        fit,
        on=["probe", "network"],
        how="outer",
        validate="one_to_one",
    )
    result["fit_r_squared_warning_threshold"] = FIT_R2_WARNING_THRESHOLD
    result["low_fit_r_squared_warning"] = (
        result["fit_r_squared_median"] < FIT_R2_WARNING_THRESHOLD
    )
    return result.sort_values(
        ["required_for_inference", "probe", "network"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_network_high_endpoint_summary(
    main_normalized: pd.DataFrame,
    dt_convergence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize all network changes at the high perturbation endpoint."""

    required = set(
        dt_convergence.loc[
            _canonical_bool(dt_convergence["required_for_inference"]),
            "network",
        ].astype(str)
    )
    endpoint = main_normalized[
        np.isclose(main_normalized["severity"], 1.0)
        & main_normalized["probe"].isin(PERIODIC_PROBES)
    ]
    result = (
        endpoint.groupby(["probe", "network"], as_index=False)
        .agg(
            median_log2_change=("log2_transfer_vs_baseline", "median"),
            minimum_log2_change=("log2_transfer_vs_baseline", "min"),
            maximum_log2_change=("log2_transfer_vs_baseline", "max"),
            mean_log2_change=("log2_transfer_vs_baseline", "mean"),
            numerical_seeds=("seed", "nunique"),
        )
    )
    result["required_for_inference"] = result["network"].isin(required)
    return result.sort_values(
        ["probe", "median_log2_change"],
        ascending=[True, False],
    ).reset_index(drop=True)


def build_calibration_summary(
    calibration: pd.DataFrame,
    *,
    selected_coupling: float,
) -> pd.DataFrame:
    result = calibration.copy()
    result["selected_main_coupling"] = np.isclose(
        result["global_coupling"],
        selected_coupling,
    )
    result["below_saturation_bound"] = result["max_abs_evoked"] < 50.0
    result["target_transfer_ratio_music_to_speech"] = (
        result["music_transfer"] / result["speech_transfer"]
    )
    return result.sort_values("global_coupling").reset_index(drop=True)


def build_runtime_diagnostics(run_manifest: pd.DataFrame) -> pd.DataFrame:
    """Summarize worker utilization and per-call wall-time distribution."""

    rows: list[dict[str, object]] = []
    for worker_pid, group in run_manifest.groupby("worker_pid", sort=True):
        rows.append(
            {
                "record_type": "worker",
                "group": str(int(worker_pid)),
                "simulations": int(len(group)),
                "aggregate_wall_seconds": float(group["wall_seconds"].sum()),
                "median_wall_seconds": float(group["wall_seconds"].median()),
                "minimum_wall_seconds": float(group["wall_seconds"].min()),
                "maximum_wall_seconds": float(group["wall_seconds"].max()),
                "maximum_abs_psp": float(group["max_abs_psp"].max()),
                "maximum_abs_evoked": float(group["max_abs_evoked"].max()),
            }
        )
    for scope, group in run_manifest.groupby("scope", sort=True):
        rows.append(
            {
                "record_type": "scope",
                "group": str(scope),
                "simulations": int(len(group)),
                "aggregate_wall_seconds": float(group["wall_seconds"].sum()),
                "median_wall_seconds": float(group["wall_seconds"].median()),
                "minimum_wall_seconds": float(group["wall_seconds"].min()),
                "maximum_wall_seconds": float(group["wall_seconds"].max()),
                "maximum_abs_psp": float(group["max_abs_psp"].max()),
                "maximum_abs_evoked": float(group["max_abs_evoked"].max()),
            }
        )
    return pd.DataFrame(rows)


def _finding(
    finding_id: str,
    audience: str,
    category: str,
    contrast_family: str,
    probe: str,
    status: str,
    summary: str,
    evidence: str,
    caveat: str,
    source_table: str,
) -> dict[str, str]:
    return {
        "finding_id": finding_id,
        "audience": audience,
        "category": category,
        "contrast_family": contrast_family,
        "probe": probe,
        "status": status,
        "summary": summary,
        "evidence": evidence,
        "caveat": caveat,
        "source_table": source_table,
    }


def build_interpretation_findings(
    bundle: ResultBundle,
    products: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create bounded, deterministic interpretations from reviewed summaries."""

    findings: list[dict[str, str]] = []
    high = products["high_endpoint_summary"]
    attenuation = products["counterfactual_attenuation"]
    matched = products["matched_null_context"]
    shuffle = products["spatial_shuffle_context"]
    sensitivity = products["sensitivity_summary"]
    convergence = products["convergence_signal_quality"]
    runtime = products["runtime_diagnostics"]

    component_columns = {
        "primary": (
            "Music proxy",
            "Speech proxy",
            "Music minus speech",
        ),
        "secondary": (
            "Semantic-task-associated proxy",
            "Episodic-task-associated proxy",
            "Semantic-associated minus episodic-associated",
        ),
    }
    for family, (left_label, right_label, contrast_label) in (
        component_columns.items()
    ):
        for probe in PERIODIC_PROBES:
            subset = high[
                (high["contrast_family"] == family)
                & (high["probe"] == probe)
            ].set_index("series")
            left = float(subset.loc[left_label, "median"])
            right = float(subset.loc[right_label, "median"])
            contrast = float(subset.loc[contrast_label, "median"])
            consistent = bool(subset.loc[contrast_label, "sign_consistent"])
            if family == "primary" and left > 0 and right > 0 and contrast < 0:
                interpretation = (
                    "Both proxy transfers increased relative to their own "
                    "baselines, but the speech proxy increased more, producing "
                    "a negative music-minus-speech contrast."
                )
            elif (
                family == "secondary"
                and left > 0
                and right > 0
                and contrast > 0
            ):
                interpretation = (
                    "Both task-associated proxy transfers increased relative "
                    "to baseline, with the semantic-associated proxy increasing "
                    "more than the episodic-associated proxy."
                )
            else:
                interpretation = (
                    "The contrast direction must be read together with its two "
                    "component changes."
                )
            findings.append(
                _finding(
                    f"{family}_endpoint_{probe}",
                    "publication",
                    "high_endpoint",
                    family,
                    probe,
                    "consistent" if consistent else "mixed",
                    interpretation,
                    (
                        f"Median component changes: {left_label}={left:.3f}, "
                        f"{right_label}={right:.3f}; contrast={contrast:.3f}; "
                        f"sign consistent across seeds={consistent}."
                    ),
                    (
                        "This is a model-internal baseline-normalized transfer "
                        "contrast, not preserved clinical function."
                    ),
                    "high_endpoint_summary.csv",
                )
            )

    for family in ("primary", "secondary"):
        for probe in PERIODIC_PROBES:
            values = attenuation[
                (attenuation["contrast_family"] == family)
                & (attenuation["probe"] == probe)
            ]["attenuation_percent"]
            median = float(values.median())
            findings.append(
                _finding(
                    f"{family}_counterfactual_{probe}",
                    "publication",
                    "counterfactual",
                    family,
                    probe,
                    "strong_attenuation" if median >= 90.0 else "attenuated",
                    (
                        "The contrast was nearly eliminated when the declared "
                        "local dynamics were held at baseline."
                        if median >= 90.0
                        else "The contrast was reduced by local fixing."
                    ),
                    (
                        f"Median absolute-contrast attenuation={median:.1f}% "
                        f"(range {values.min():.1f}% to {values.max():.1f}%)."
                    ),
                    (
                        "This supports dependence on the modeled local-dynamics "
                        "perturbation; it does not establish biological causality."
                    ),
                    "counterfactual_attenuation.csv",
                )
            )

    for context_name, context in (
        ("matched", matched),
        ("shuffle", shuffle),
    ):
        for family in ("primary", "secondary"):
            subset = context[context["contrast_family"] == family]
            inside = bool(subset["inside_central_90_percent"].all())
            percentile_min = float(
                subset["observed_empirical_percentile"].min()
            )
            percentile_max = float(
                subset["observed_empirical_percentile"].max()
            )
            findings.append(
                _finding(
                    f"{family}_{context_name}_context",
                    "publication",
                    context_name,
                    family,
                    "both",
                    "inside_central_90" if inside else "outside_central_90",
                    (
                        f"Observed {family} contrasts remained "
                        f"{'inside' if inside else 'outside at least one'} "
                        f"central 90% {context_name} range."
                    ),
                    (
                        f"Empirical percentile range={percentile_min:.1f} to "
                        f"{percentile_max:.1f}."
                    ),
                    (
                        "These are descriptive simulation ranks, not clinical "
                        "p-values or independent-subject inference."
                    ),
                    f"{context_name}_null_context.csv"
                    if context_name == "matched"
                    else "spatial_shuffle_context.csv",
                )
            )

    for family in ("primary", "secondary"):
        subset = sensitivity[sensitivity["contrast_family"] == family]
        sign_changes = subset[~subset["same_sign_as_reference"]]
        variants = sorted(set(sign_changes["variant"]) - {"G60_input_0.02"})
        findings.append(
            _finding(
                f"{family}_parameter_sensitivity",
                "publication",
                "sensitivity",
                family,
                "both",
                "sign_change" if variants else "sign_stable",
                (
                    "At least one sensitivity scenario reversed the main "
                    f"{family} contrast direction."
                    if variants
                    else f"The {family} contrast direction was stable across "
                    "the tested sensitivity scenarios."
                ),
                (
                    "Sign-changing variants: "
                    + (", ".join(variants) if variants else "none")
                    + "."
                ),
                (
                    "Sensitivity scenarios are robustness diagnostics and do "
                    "not identify a uniquely correct parameter setting."
                ),
                "sensitivity_summary.csv",
            )
        )

    required = convergence[convergence["required_for_inference"]]
    max_dt = float(required["max_relative_difference"].max())
    findings.append(
        _finding(
            "integration_step_convergence",
            "technical",
            "convergence",
            "all",
            "both",
            "passed",
            "Every inferential network passed the integration-step gate.",
            (
                f"Maximum required-network relative difference={max_dt:.4%}; "
                "prespecified threshold=5%."
            ),
            (
                "Passing integration-step convergence does not guarantee a "
                "high-quality harmonic fit."
            ),
            "convergence_signal_quality.csv",
        )
    )
    low_fit = convergence[convergence["low_fit_r_squared_warning"]]
    findings.append(
        _finding(
            "harmonic_fit_quality",
            "technical",
            "signal_quality",
            "all",
            "both",
            "warning" if not low_fit.empty else "acceptable",
            (
                "Several high-endpoint network summaries have low median "
                "harmonic-fit R-squared values."
                if not low_fit.empty
                else "No high-endpoint network crossed the descriptive low-fit "
                "warning threshold."
            ),
            (
                f"{len(low_fit)}/{len(convergence)} network/probe summaries "
                f"below R²={FIT_R2_WARNING_THRESHOLD:.2f}; minimum median "
                f"R²={convergence['fit_r_squared_median'].min():.4f}."
            ),
            (
                "The R² warning threshold is descriptive QA, not a "
                "prespecified experiment gate; amplitude interpretation should "
                "remain cautious."
            ),
            "convergence_signal_quality.csv",
        )
    )

    workload = bundle.metadata["workload"]
    findings.append(
        _finding(
            "workload_completeness",
            "technical",
            "completeness",
            "all",
            "all",
            "complete",
            "The copied final-result folder contains the full planned workload.",
            (
                f"Manifested simulations={workload['manifest']}; total TVB "
                f"calls={workload['total']}."
            ),
            (
                "This confirms result-file completeness, not scientific "
                "validity."
            ),
            "validation_summary.csv",
        )
    )
    worker_rows = runtime[runtime["record_type"] == "worker"]
    findings.append(
        _finding(
            "runtime_execution",
            "technical",
            "runtime",
            "all",
            "all",
            "descriptive",
            "Runtime and worker utilization were summarized for reproducibility.",
            (
                f"Observed worker processes={len(worker_rows)}; aggregate "
                f"simulation wall time="
                f"{worker_rows['aggregate_wall_seconds'].sum()/60.0:.1f} min."
            ),
            (
                "Aggregate simulation time includes concurrently executed "
                "calls and is not command elapsed time."
            ),
            "runtime_diagnostics.csv",
        )
    )
    return pd.DataFrame(findings)


def build_analysis_products(
    bundle: ResultBundle,
    validation_summary: pd.DataFrame,
) -> AnalysisProducts:
    """Build every derived CSV table and interpretation finding."""

    tables = bundle.tables
    trajectory, high_endpoint = build_contrast_summaries(
        tables["main_contrasts"]
    )
    expected_seeds = tuple(
        int(value) for value in bundle.metadata["model"]["numerical_seeds"]
    )
    selected_coupling = float(
        bundle.metadata["model"]["global_coupling"]
    )
    products: dict[str, pd.DataFrame] = {
        "validation_summary": validation_summary,
        "contrast_trajectory_summary": trajectory,
        "high_endpoint_summary": high_endpoint,
        "counterfactual_attenuation": build_counterfactual_attenuation(
            tables["main_contrasts"],
            tables["local_contrasts"],
        ),
        "matched_null_context": build_matched_null_context(
            tables["main_contrasts"],
            tables["matched_null"],
            tables["memory_matched_null"],
        ),
        "spatial_shuffle_context": build_spatial_shuffle_context(
            tables["main_contrasts"],
            tables["shuffle"],
            main_seed=expected_seeds[0],
        ),
        "sensitivity_summary": build_sensitivity_summary(
            tables["sensitivity"]
        ),
        "convergence_signal_quality": build_convergence_signal_quality(
            tables["dt_convergence"],
            tables["main_network"],
        ),
        "network_high_endpoint_summary": (
            build_network_high_endpoint_summary(
                tables["main_normalized"],
                tables["dt_convergence"],
            )
        ),
        "calibration_summary": build_calibration_summary(
            tables["calibration"],
            selected_coupling=selected_coupling,
        ),
        "runtime_diagnostics": build_runtime_diagnostics(
            tables["run_manifest"]
        ),
    }
    findings = build_interpretation_findings(bundle, products)
    return AnalysisProducts(tables=products, findings=findings)


__all__ = [
    "AnalysisProducts",
    "FIT_R2_WARNING_THRESHOLD",
    "PERIODIC_PROBES",
    "PRIMARY_CONTRAST",
    "ResultBundle",
    "ResultValidationError",
    "SECONDARY_CONTRAST",
    "TABLE_FILES",
    "build_analysis_products",
    "build_contrast_summaries",
    "build_counterfactual_attenuation",
    "build_interpretation_findings",
    "build_matched_null_context",
    "build_sensitivity_summary",
    "build_spatial_shuffle_context",
    "load_result_bundle",
    "validate_result_bundle",
]
