#!/usr/bin/env python3
"""Build the self-contained TVB379 visual investigation dashboard.

The builder reads every top-level experiment CSV and JSON file plus the run
log and existing figure assets.  It also runs the deterministic post-hoc audit
over the 180 lossless NPZ trace shards and embeds only bounded derived
summaries—not the raw arrays.  The generated HTML needs no server, CDN, or
sibling data files at viewing time.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    from data_analysis import recommended_posthoc_analysis as posthoc
except ModuleNotFoundError:  # Direct ``python data_analysis/...`` execution.
    import recommended_posthoc_analysis as posthoc


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_HTML = DATA_DIR / "TVB379_visual_investigator.html"
ARTIFACT_JSON = DATA_DIR / "html_investigator" / "artifact.json"
SOURCE_SQL = DATA_DIR / "html_investigator_sources.sql"
EXPECTED_CSV_COUNT = 48
EXPECTED_JSON_COUNT = 3
EXPECTED_TVB_CALLS = 762
EXPECTED_TRACE_SHARDS = 180
PRIMARY_PAIR = "expanded_bilateral"
PRIMARY_NETWORKS = ("semantic_expanded", "episodic_expanded")
SEVERITIES = (0.0, 0.5, 1.0)
PROBES = ("pulse", "2Hz", "5Hz")


class DashboardBuildError(RuntimeError):
    """Raised when the result export cannot support the dashboard."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DashboardBuildError(f"Expected a JSON object: {path}")
    return value


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)


def _finite_summary(values: Iterable[float]) -> dict[str, float | int]:
    numeric = np.asarray(list(values), dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return {
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "minimum": math.nan,
            "maximum": math.nan,
            "ci95_lower": math.nan,
            "ci95_upper": math.nan,
        }
    n = len(numeric)
    mean = float(np.mean(numeric))
    sd = float(np.std(numeric, ddof=1)) if n > 1 else math.nan
    half_width = (
        float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        if n > 1 and np.isfinite(sd)
        else math.nan
    )
    return {
        "n": int(n),
        "mean": mean,
        "median": float(np.median(numeric)),
        "minimum": float(np.min(numeric)),
        "maximum": float(np.max(numeric)),
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def _safe_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value) if not isinstance(value, (str, int)) else value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _safe_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _sample_json(row: pd.Series, limit: int = 280) -> str:
    encoded = json.dumps(
        {str(key): _safe_scalar(value) for key, value in row.items()},
        ensure_ascii=False,
        separators=(", ", ": "),
    )
    return encoded if len(encoded) <= limit else encoded[: limit - 1] + "…"


def _source_group(filename: str) -> str:
    if filename.startswith(("main_", "primary_")):
        return "main experiment"
    if filename.startswith(("integration_step", "dt_reference")):
        return "integration step"
    if filename.startswith("local_fixed") or filename.startswith("counterfactual"):
        return "counterfactual"
    if filename.startswith("matched_control"):
        return "matched controls"
    if filename.startswith("parameter"):
        return "parameter sensitivity"
    if filename.startswith("spatial_shuffle"):
        return "spatial shuffles"
    if filename.startswith(("roi_", "regional_", "pathology_")):
        return "regions and pathology"
    if filename.startswith(("network_evidence", "music_memory", "source_")):
        return "evidence and provenance"
    if filename.startswith(("a1_", "periodic_", "technical_", "data_quality")):
        return "measurement quality"
    if filename.startswith(("definition_", "laterality_")):
        return "definition sensitivity"
    if filename.startswith("baseline_coupling"):
        return "calibration"
    if filename.startswith("run_") or filename == "progress.log":
        return "execution"
    return "other"


def load_export(
    data_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    csv_paths = sorted(data_dir.glob("*.csv"))
    json_paths = sorted(data_dir.glob("*.json"))
    if len(csv_paths) != EXPECTED_CSV_COUNT:
        raise DashboardBuildError(
            f"Expected {EXPECTED_CSV_COUNT} CSV files, found {len(csv_paths)}."
        )
    if len(json_paths) != EXPECTED_JSON_COUNT:
        raise DashboardBuildError(
            f"Expected {EXPECTED_JSON_COUNT} JSON files, found {len(json_paths)}."
        )
    tables = {path.name: pd.read_csv(path) for path in csv_paths}
    objects = {path.name: _json_object(path) for path in json_paths}
    return tables, objects


def source_catalog(
    data_dir: Path,
    tables: dict[str, pd.DataFrame],
    objects: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for filename, frame in tables.items():
        path = data_dir / filename
        numeric = frame.select_dtypes(include=[np.number])
        nonfinite = int(
            (~np.isfinite(numeric.to_numpy(dtype=float))).sum()
        ) if len(numeric.columns) else 0
        rows.append(
            {
                "file": filename,
                "kind": "CSV",
                "group": _source_group(filename),
                "rows": len(frame),
                "columns": len(frame.columns),
                "null_cells": int(frame.isna().sum().sum()),
                "nonfinite_numeric_cells": nonfinite,
                "duplicate_rows": int(frame.duplicated().sum()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "schema": ", ".join(frame.columns),
                "reviewed": True,
            }
        )
        if len(frame):
            positions = sorted({0, len(frame) // 2, len(frame) - 1})
            for position in positions:
                samples.append(
                    {
                        "file": filename,
                        "row_index": int(position),
                        "sample": _sample_json(frame.iloc[position]),
                    }
                )
    for filename, value in objects.items():
        path = data_dir / filename
        rows.append(
            {
                "file": filename,
                "kind": "JSON",
                "group": _source_group(filename),
                "rows": 1,
                "columns": len(value),
                "null_cells": 0,
                "nonfinite_numeric_cells": 0,
                "duplicate_rows": 0,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "schema": ", ".join(value.keys()),
                "reviewed": True,
            }
        )
        sample = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        samples.append(
            {
                "file": filename,
                "row_index": 0,
                "sample": sample[:279] + "…" if len(sample) > 280 else sample,
            }
        )
    extra_paths = [
        data_dir / "progress.log",
        *sorted(data_dir.glob("*overlay.png")),
        *sorted(data_dir.glob("*overlay.svg")),
    ]
    for path in extra_paths:
        if not path.is_file():
            continue
        kind = path.suffix.lstrip(".").upper() or "FILE"
        line_count = (
            len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            if path.suffix == ".log"
            else 0
        )
        rows.append(
            {
                "file": path.name,
                "kind": kind,
                "group": _source_group(path.name),
                "rows": line_count,
                "columns": 0,
                "null_cells": 0,
                "nonfinite_numeric_cells": 0,
                "duplicate_rows": 0,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "schema": "run log" if path.suffix == ".log" else "existing experiment figure",
                "reviewed": True,
            }
        )
    catalog = pd.DataFrame(rows).sort_values(["group", "file"]).reset_index(drop=True)
    sample_frame = pd.DataFrame(samples).sort_values(["file", "row_index"]).reset_index(drop=True)
    return catalog, sample_frame


def build_trajectory_rows(
    interactions: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    current_status = eligibility.set_index(["outcome", "probe"])[
        "analysis_status"
    ].to_dict()
    primary = interactions[interactions["pair"] == PRIMARY_PAIR]
    rows: list[dict[str, Any]] = []
    series_columns = {
        "Semantic proxy": "semantic_value",
        "Episodic proxy": "episodic_value",
        "Semantic − episodic": "semantic_minus_episodic_interaction",
    }
    for (outcome, probe, severity, unit), group in primary.groupby(
        ["outcome", "probe", "severity", "unit"], sort=True
    ):
        for series, column in series_columns.items():
            summary = _finite_summary(group[column])
            rows.append(
                {
                    "outcome": outcome,
                    "probe": probe,
                    "severity": float(severity),
                    "severity_label": f"{float(severity):.1f}",
                    "network": series,
                    "series": f"{probe} · {series}",
                    "unit": unit,
                    "analysis_status": current_status.get(
                        (outcome, probe), "unknown"
                    ),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def build_pulse_summary(
    node_metrics: pd.DataFrame,
    regional: pd.DataFrame,
) -> pd.DataFrame:
    memberships = {
        "Semantic proxy": "semantic_expanded_membership",
        "Episodic proxy": "episodic_expanded_membership",
    }
    joined = node_metrics[node_metrics["probe"] == "pulse"].merge(
        regional[
            [
                "region_index",
                "semantic_expanded_membership",
                "episodic_expanded_membership",
            ]
        ],
        on="region_index",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for network, column in memberships.items():
        selected = joined[_bool_series(joined[column])].copy()
        per_seed = selected.groupby(["severity", "seed"], as_index=False).agg(
            median_peak_psp=("absolute_peak_response", "median"),
            median_energy_psp2_ms=("total_evoked_energy_psp2_ms", "median"),
            median_peak_relative_ms=(
                "peak_time_relative_to_ipsilateral_a1_ms",
                "median",
            ),
            median_latency_ms=("relative_latency_ms", "median"),
            valid_latency_fraction=("latency_valid", "mean"),
        )
        for severity, group in per_seed.groupby("severity", sort=True):
            rows.append(
                {
                    "severity": float(severity),
                    "severity_label": f"{float(severity):.1f}",
                    "probe": "pulse",
                    "network": network,
                    "series": network,
                    "numerical_initializations": len(group),
                    "median_peak_psp": float(group["median_peak_psp"].median()),
                    "minimum_peak_psp": float(group["median_peak_psp"].min()),
                    "maximum_peak_psp": float(group["median_peak_psp"].max()),
                    "median_energy_psp2_ms": float(
                        group["median_energy_psp2_ms"].median()
                    ),
                    "median_peak_relative_ms": float(
                        group["median_peak_relative_ms"].median()
                    ),
                    "median_latency_ms": float(
                        group["median_latency_ms"].median()
                    ),
                    "median_valid_latency_fraction": float(
                        group["valid_latency_fraction"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_quality_views(validity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fc = validity[validity["probe"].isin(("2Hz", "5Hz"))].copy()
    fc["passed"] = _bool_series(fc["functional_connectivity_valid"])
    fc_rows = []
    for (severity, probe, network), group in fc.groupby(
        ["severity", "probe", "network"], sort=True
    ):
        fc_rows.append(
            {
                "severity": float(severity),
                "severity_label": f"{float(severity):.1f}",
                "probe": probe,
                "network": network.replace("_expanded", " proxy"),
                "series": f"{probe} · {network.replace('_expanded', ' proxy')}",
                "valid_rows": int(group["passed"].sum()),
                "required_rows": len(group),
                "pass_rate": float(group["passed"].mean()),
                "median_split_abs_z": float(
                    group["evoked_fc_split_abs_z"].median()
                ),
                "maximum_split_abs_z": float(
                    group["evoked_fc_split_abs_z"].max()
                ),
            }
        )
    pulse = validity[validity["probe"] == "pulse"].copy()
    pulse["passed"] = _bool_series(pulse["response_latency_valid"])
    latency_rows = []
    for (severity, network), group in pulse.groupby(
        ["severity", "network"], sort=True
    ):
        latency_rows.append(
            {
                "severity": float(severity),
                "severity_label": f"{float(severity):.1f}",
                "probe": "pulse",
                "network": network.replace("_expanded", " proxy"),
                "series": network.replace("_expanded", " proxy"),
                "valid_rows": int(group["passed"].sum()),
                "required_rows": len(group),
                "pass_rate": float(group["passed"].mean()),
                "median_tail_fraction": float(
                    group["median_pulse_tail_energy_fraction"].median()
                ),
                "maximum_tail_fraction": float(
                    group["median_pulse_tail_energy_fraction"].max()
                ),
                "minimum_valid_parcel_fraction": float(
                    group["latency_valid_fraction"].min()
                ),
            }
        )
    return pd.DataFrame(fc_rows), pd.DataFrame(latency_rows)


def build_periodic_views(
    temporal: pd.DataFrame,
    network_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = temporal[temporal["network"].isin(PRIMARY_NETWORKS)]
    quality = expanded.groupby(
        ["severity", "probe", "network"], as_index=False
    ).agg(
        mean_transfer_slope_log2_per_s=("transfer_log2_slope_per_s", "mean"),
        median_transfer_slope_log2_per_s=("transfer_log2_slope_per_s", "median"),
        nonstationary_fraction=("transfer_nonstationary_flag", "mean"),
        median_target_snr_db=("median_target_snr_db", "median"),
        median_phase_consistency=("median_target_phase_consistency", "median"),
        median_frequency_valid_fraction=(
            "target_frequency_qa_valid_fraction",
            "median",
        ),
        median_fc_split_abs_z=("evoked_fc_split_abs_z", "median"),
    )
    quality["severity_label"] = quality["severity"].map(lambda x: f"{x:.1f}")
    quality["network"] = quality["network"].str.replace(
        "_expanded", " proxy", regex=False
    )
    quality["series"] = quality["probe"] + " · " + quality["network"]

    high = network_metrics[
        (network_metrics["severity"] == 1.0)
        & network_metrics["probe"].isin(("2Hz", "5Hz"))
        & network_metrics["network"].isin(PRIMARY_NETWORKS)
    ]
    segment_rows: list[dict[str, Any]] = []
    for (probe, network), group in high.groupby(["probe", "network"], sort=True):
        for segment in range(1, 6):
            values = group[f"segment_transfer_{segment}"]
            segment_rows.append(
                {
                    "severity": 1.0,
                    "severity_label": "1.0",
                    "probe": probe,
                    "network": network.replace("_expanded", " proxy"),
                    "series": f"{probe} · {network.replace('_expanded', ' proxy')}",
                    "segment": segment,
                    "segment_window_s": f"{2 * (segment - 1)}–{2 * segment}",
                    "mean_transfer": float(values.mean()),
                    "median_transfer": float(values.median()),
                }
            )
    return quality, pd.DataFrame(segment_rows)


def build_runtime(run_manifest: pd.DataFrame) -> pd.DataFrame:
    frame = run_manifest.copy()
    frame["workload"] = frame["scope"].map(
        lambda value: (
            "Spatial shuffles"
            if str(value).startswith("spatial_shuffle")
            else "Parameter sensitivity"
            if str(value).startswith("parameter_")
            else {
                "main_full_field": "Main full field",
                "dt_reference_0.25ms": "0.25 ms reference",
                "local_dynamics_counterfactual": "Local-fixed counterfactual",
            }.get(str(value), str(value))
        )
    )
    rows = []
    for workload, group in frame.groupby("workload", sort=False):
        rows.append(
            {
                "workload": workload,
                "calls": len(group),
                "recorded_wall_hours": float(group["wall_seconds"].sum() / 3600.0),
                "median_call_seconds": float(group["wall_seconds"].median()),
                "p95_call_seconds": float(group["wall_seconds"].quantile(0.95)),
                "worker_processes": int(group["worker_pid"].nunique()),
                "maximum_abs_psp": float(group["max_abs_psp"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_regional_view(regional: pd.DataFrame) -> pd.DataFrame:
    frame = regional.copy()
    semantic = _bool_series(frame["semantic_expanded_membership"])
    episodic = _bool_series(frame["episodic_expanded_membership"])
    frame["proxy_membership"] = np.select(
        [semantic & episodic, semantic, episodic],
        ["Both proxies", "Semantic proxy", "Episodic proxy"],
        default="Neither proxy",
    )
    return frame[
        [
            "region_index",
            "region_label",
            "hemisphere",
            "proxy_membership",
            "surrogate_amyloid",
            "b_reduction",
            "weighted_structural_strength",
            "bilateral_a1_affinity",
            "semantic_anatomical_core_membership",
            "episodic_anatomical_core_membership",
            "platel_peak_membership",
        ]
    ]


def _source(
    source_id: str,
    label: str,
    description: str,
    *,
    uses_traces: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relative_sql = "data_analysis/html_investigator_sources.sql"
    manifest_source = {
        "id": source_id,
        "label": label,
        "path": relative_sql,
    }
    source = {
        "id": source_id,
        "label": label,
        "path": relative_sql,
        "query": {
            "engine": "DuckDB-compatible source map + pandas builder",
            "language": "sql/python",
            "description": description,
            "tables_used": [
                "data_analysis/*.csv",
                "data_analysis/*.json",
                *(
                    ["data_analysis/main_parcel_traces/*.npz"]
                    if uses_traces
                    else []
                ),
            ],
            "filters": [
                "Raw NPZ arrays are summarized but never embedded"
                if uses_traces
                else "Completed experiment tables only"
            ],
            "metric_definitions": [
                "Numerical initializations are not participants or biological replicates.",
                "Final eligibility comes from outcome_eligibility.csv.",
            ],
        },
    }
    return manifest_source, source


def _chart(
    chart_id: str,
    title: str,
    subtitle: str,
    chart_type: str,
    dataset: str,
    source_id: str,
    x_field: str,
    x_type: str,
    x_label: str,
    y_field: str,
    y_label: str,
    *,
    color_field: str | None = None,
    value_format: str = "number",
    tooltip: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    encodings: dict[str, Any] = {
        "x": {"field": x_field, "type": x_type, "label": x_label},
        "y": {"field": y_field, "type": "quantitative", "label": y_label},
        "tooltip": list(tooltip),
    }
    if color_field:
        encodings["color"] = {
            "field": color_field,
            "type": "nominal",
            "label": "Series",
        }
    return {
        "id": chart_id,
        "title": title,
        "subtitle": subtitle,
        "layout": "full",
        "type": chart_type,
        "dataset": dataset,
        "sourceId": source_id,
        "valueFormat": value_format,
        "encodings": encodings,
    }


def build_artifact(data_dir: Path) -> dict[str, Any]:
    tables, objects = load_export(data_dir)
    catalog, samples = source_catalog(data_dir, tables, objects)
    status = objects["run_status.json"]
    metadata = objects["experiment_metadata.json"]
    if status.get("state") != "completed" or int(
        status.get("completed_tvb_calls", -1)
    ) != EXPECTED_TVB_CALLS:
        raise DashboardBuildError("The experiment run is not complete.")
    posthoc_outputs = posthoc.build_analysis(
        data_dir,
        data_dir / "investigation" / "recommended",
    )

    eligibility = tables["outcome_eligibility.csv"].copy()
    eligibility["valid_fraction"] = (
        eligibility["valid_rows"] / eligibility["required_rows"]
    )
    eligibility["label"] = (
        eligibility["outcome"].str.replace("_", " ")
        + " · "
        + eligibility["probe"]
    )
    eligibility["status_label"] = eligibility["analysis_status"].map(
        {
            "eligible_as_prespecified": "Eligible",
            "direction_robust_exact_magnitude_descriptive_only": "Direction robust; magnitude DT-sensitive",
            "descriptive_only_quality_gate_failed": "Descriptive; quality gate failed",
        }
    ).fillna(eligibility["analysis_status"])
    eligibility["dt_label"] = eligibility["integration_step_status"].map(
        {
            "exact_magnitude_precision_target_passed": "Exact-magnitude target passed",
            "direction_robust_exact_magnitude_dt_sensitive": "Exact magnitude DT-sensitive",
        }
    ).fillna("Not applicable")
    trajectories = build_trajectory_rows(
        tables["main_pair_interactions.csv"], eligibility
    )
    pulse = build_pulse_summary(
        tables["main_node_metrics.csv"], tables["regional_features.csv"]
    )
    fc_quality, latency_quality = build_quality_views(
        tables["main_science_validity.csv"]
    )
    periodic_quality, segment_transfer = build_periodic_views(
        tables["periodic_temporal_qa.csv"], tables["main_network_metrics.csv"]
    )

    dt = tables["integration_step_outcome_eligibility.csv"].copy()
    dt["label"] = dt["outcome"].str.replace("_", " ") + " · " + dt["probe"]
    dt["tolerance_ratio"] = dt["absolute_difference"] / dt["tolerance"]
    counterfactual = tables["counterfactual_summary.csv"].copy()
    counterfactual["label"] = (
        counterfactual["outcome"].str.replace("_", " ")
        + " · "
        + counterfactual["probe"]
    )
    matched = tables["matched_control_null_summary.csv"].groupby(
        ["outcome", "probe"], as_index=False
    ).agg(
        numerical_initializations=("seed", "size"),
        median_empirical_percentile=("observed_empirical_percentile", "median"),
        minimum_empirical_percentile=("observed_empirical_percentile", "min"),
        maximum_empirical_percentile=("observed_empirical_percentile", "max"),
        outside_central_90_count=("outside_central_90_percent", "sum"),
    )
    matched["label"] = matched["outcome"].str.replace("_", " ") + " · " + matched["probe"]
    shuffles = tables["spatial_shuffle_summary.csv"].copy()
    shuffles["label"] = shuffles["outcome"].str.replace("_", " ") + " · " + shuffles["probe"]
    parameter = tables["parameter_interaction_statistics.csv"].copy()
    parameter["series"] = parameter["probe"]
    definitions = tables["definition_sensitivity_statistics.csv"].copy()
    definitions["definition"] = definitions["pair"].str.replace("_", " ")
    definitions["series"] = definitions["probe"]
    laterality = tables["laterality_difference_statistics.csv"].copy()
    laterality["label"] = laterality["outcome"].str.replace("_", " ") + " · " + laterality["probe"]
    runtime = build_runtime(tables["run_manifest.csv"])
    regional = build_regional_view(tables["regional_features.csv"])
    calibration = tables["baseline_coupling_diagnostic.csv"].copy()

    posthoc_endpoint = posthoc_outputs["transmission_endpoint"].copy()
    posthoc_fc = posthoc_outputs["fc_trajectory"].copy()
    posthoc_fc_raw = posthoc_fc[posthoc_fc["metric"] == "Raw FC"].copy()
    posthoc_fc_change = posthoc_fc[
        posthoc_fc["metric"] == "Change from baseline"
    ].copy()
    posthoc_spectra = posthoc_outputs["spectra_summary"].copy()
    high_evoked_spectra = posthoc_spectra[
        (posthoc_spectra["severity"] == 1.0)
        & (posthoc_spectra["signal"] == "Evoked")
    ].copy()
    high_evoked_spectra["series"] = (
        high_evoked_spectra["probe"]
        + " · "
        + high_evoked_spectra["network"]
    )
    fixed_pulse = posthoc_outputs["pulse_fixed_summary"].copy()
    phase_fc_rows = posthoc_outputs["phase_fc_rows"].copy()
    phase_fc_rows["series"] = (
        phase_fc_rows["probe"] + " · " + phase_fc_rows["network"]
    )
    regional_coefficients = posthoc_outputs[
        "regional_covariate_coefficients"
    ].copy()
    regional_coefficients = regional_coefficients[
        regional_coefficients["term"] != "intercept"
    ]
    regional_coefficients["series"] = regional_coefficients["probe"]
    pulse_quantile_rows: list[dict[str, Any]] = []
    for row in fixed_pulse.to_dict(orient="records"):
        for quantile, field in (
            ("t20", "median_relative_t20_ms"),
            ("t50", "median_relative_t50_ms"),
            ("t80", "median_relative_t80_ms"),
        ):
            pulse_quantile_rows.append(
                {
                    "severity": float(row["severity"]),
                    "severity_label": str(row["severity_label"]),
                    "network": str(row["network"]),
                    "quantile": quantile,
                    "series": f"{quantile} · {row['network']}",
                    "relative_time_ms": float(row[field]),
                    "usable_seeds": int(row["usable_seeds"]),
                    "median_fixed_parcel_count": float(
                        row["median_fixed_parcel_count"]
                    ),
                }
            )
    pulse_quantiles = pd.DataFrame(pulse_quantile_rows)

    endpoint = trajectories[
        (trajectories["severity"] == 1.0)
        & (trajectories["network"] == "Semantic − episodic")
    ]
    transfer_2 = endpoint[
        (endpoint["outcome"] == "transfer_gain")
        & (endpoint["probe"] == "2Hz")
    ].iloc[0]
    transfer_5 = endpoint[
        (endpoint["outcome"] == "transfer_gain")
        & (endpoint["probe"] == "5Hz")
    ].iloc[0]
    summary = pd.DataFrame(
        [
            {
                "completed_calls": int(status["completed_tvb_calls"]),
                "csv_tables": len(tables),
                "reviewed_non_npz_files": len(catalog),
                "trace_shards_manifested": len(
                    tables["main_parcel_trace_manifest.csv"]
                ),
                "trace_shards_analyzed": int(
                    posthoc_outputs["pulse_region_quantiles"][
                        ["severity", "seed"]
                    ].drop_duplicates().shape[0]
                    + posthoc_outputs["segment_trace_audit"][
                        ["severity", "seed", "probe"]
                    ].drop_duplicates().shape[0]
                ),
                "transfer_2hz_interaction": float(transfer_2["mean"]),
                "transfer_5hz_interaction": float(transfer_5["mean"]),
                "pulse_fc_defined": 0,
            }
        ]
    )

    datasets = {
        "summary": _records(summary),
        "transfer_trajectory": _records(
            trajectories[trajectories["outcome"] == "transfer_gain"]
        ),
        "fc_trajectory": _records(
            trajectories[trajectories["outcome"] == "functional_connectivity"]
        ),
        "latency_trajectory": _records(
            trajectories[trajectories["outcome"] == "response_latency"]
        ),
        "eligibility": _records(eligibility),
        "pulse_summary": _records(pulse),
        "fc_quality": _records(fc_quality),
        "latency_quality": _records(latency_quality),
        "periodic_quality": _records(periodic_quality),
        "segment_transfer": _records(segment_transfer),
        "dt_eligibility": _records(dt),
        "counterfactual": _records(counterfactual),
        "matched_null": _records(matched),
        "spatial_shuffle": _records(shuffles),
        "parameter_transfer": _records(
            parameter[parameter["outcome"] == "transfer_gain"]
        ),
        "parameter_fc": _records(
            parameter[parameter["outcome"] == "functional_connectivity"]
        ),
        "definition_transfer": _records(
            definitions[definitions["outcome"] == "transfer_gain"]
        ),
        "laterality": _records(laterality),
        "regional_features": _records(regional),
        "pathology": _records(tables["pathology_summary.csv"]),
        "runtime": _records(runtime),
        "calibration": _records(calibration),
        "a1_qa": _records(tables["a1_frequency_qa.csv"]),
        "source_catalog": _records(catalog),
        "source_samples": _records(samples),
        "mapping": _records(tables["music_memory_peak_mapping.csv"]),
        "network_evidence": _records(tables["network_evidence.csv"]),
        "data_checks": _records(tables["data_quality_checks.csv"]),
        "transmission_endpoint_audit": _records(posthoc_endpoint),
        "frequency_quality_audit": _records(
            posthoc_outputs["frequency_quality"]
        ),
        "segment_frequency_audit": _records(
            posthoc_outputs["segment_trace_summary"]
        ),
        "fc_raw_audit": _records(posthoc_fc_raw),
        "fc_change_audit": _records(posthoc_fc_change),
        "phase_fc_rows": _records(phase_fc_rows),
        "phase_fc_summary": _records(posthoc_outputs["phase_fc_summary"]),
        "spectra_stimulated": _records(
            posthoc_spectra[posthoc_spectra["signal"] == "Stimulated"]
        ),
        "spectra_control": _records(
            posthoc_spectra[posthoc_spectra["signal"] == "Control"]
        ),
        "spectra_evoked": _records(
            posthoc_spectra[posthoc_spectra["signal"] == "Evoked"]
        ),
        "high_evoked_spectra": _records(high_evoked_spectra),
        "spectral_peaks": _records(
            posthoc_outputs["spectral_peak_summary"]
        ),
        "pulse_fixed_masks": _records(
            posthoc_outputs["pulse_fixed_masks"]
        ),
        "pulse_fixed_summary": _records(fixed_pulse),
        "pulse_fixed_quantiles": _records(pulse_quantiles),
        "regional_covariate_coefficients": _records(regional_coefficients),
        "regional_covariate_models": _records(
            posthoc_outputs["regional_covariate_models"]
        ),
    }

    source_specs = [
        _source("src_summary", "Completed run and export inventory", "Reads run_status.json, run_manifest.csv, and the top-level export inventory."),
        _source("src_primary", "Primary interaction tables", "Summarizes expanded_bilateral rows from main_pair_interactions.csv and final labels from outcome_eligibility.csv."),
        _source("src_pulse", "Pulse node and regional metrics", "Aggregates pulse rows in main_node_metrics.csv over semantic and episodic expanded memberships from regional_features.csv."),
        _source("src_quality", "Measurement-quality tables", "Uses main_science_validity.csv, periodic_temporal_qa.csv, a1_frequency_qa.csv, and outcome_eligibility.csv."),
        _source("src_dt", "Integration-step diagnostics", "Uses integration_step_outcome_eligibility.csv and the complete integration-step diagnostics family."),
        _source("src_robustness", "Counterfactual and simulation-null tables", "Uses local-fixed, counterfactual, matched-control, and spatial-shuffle tables."),
        _source("src_sensitivity", "Parameter and definition sensitivity", "Uses parameter, definition, and laterality statistics plus their node/network support tables."),
        _source("src_regions", "Regional pathology and evidence", "Uses regional_features.csv, ROI definitions/pathology, evidence, mapping, pathology summary, and source manifest."),
        _source("src_runtime", "Calibration and runtime", "Uses baseline_coupling_diagnostic.csv, run_manifest.csv, run_status.json, progress.log, and trace manifest metadata."),
        _source("src_catalog", "Complete non-NPZ source catalog", "Profiles every experiment CSV/JSON table, the run log, and existing figure assets. Raw trace shards are separately represented by their complete manifest."),
        _source("src_posthoc_transmission", "Post-hoc transmission and FC audit", "Uses saved network/node metrics to separate broadband from frequency-locked transfer and raw from baseline-referenced FC."),
        _source("src_posthoc_traces", "Lossless raw-trace audit", "Applies the notebook's exact harmonic fit, detrending, phase, multitaper, and pulse-energy methods to all 180 manifested NPZ shards; embeds derived summaries only.", uses_traces=True),
        _source("src_posthoc_covariates", "Exploratory regional covariate audit", "Averages high-versus-baseline node-transfer change within each of 32 proxy parcels, then fits descriptive parcel-level OLS models using saved pathology and topology covariates."),
    ]
    manifest_sources = [item[0] for item in source_specs]
    canonical_sources = [item[1] for item in source_specs]

    cards = [
        {"id": "calls", "description": "Completed TVB calls including calibration.", "dataset": "summary", "sourceId": "src_summary", "metrics": [{"label": "Completed TVB calls", "field": "completed_calls", "format": "number"}]},
        {"id": "sources", "description": "All reviewed top-level non-NPZ experiment files.", "dataset": "summary", "sourceId": "src_catalog", "metrics": [{"label": "Reviewed non-NPZ files", "field": "reviewed_non_npz_files", "format": "number"}]},
        {"id": "trace_manifest", "description": "Every manifested lossless shard was read for the post-hoc spectral and pulse audit; raw arrays are not embedded in the HTML.", "dataset": "summary", "sourceId": "src_posthoc_traces", "metrics": [{"label": "Trace shards analyzed", "field": "trace_shards_analyzed", "format": "number"}]},
        {"id": "transfer2", "description": "Mean high-endpoint semantic-minus-episodic interaction across 20 numerical initializations.", "dataset": "summary", "sourceId": "src_primary", "metrics": [{"label": "Transfer interaction · 2 Hz", "field": "transfer_2hz_interaction", "format": "number"}]},
        {"id": "transfer5", "description": "Mean high-endpoint semantic-minus-episodic interaction across 20 numerical initializations.", "dataset": "summary", "sourceId": "src_primary", "metrics": [{"label": "Transfer interaction · 5 Hz", "field": "transfer_5hz_interaction", "format": "number"}]},
    ]

    tooltip_common = [
        {"field": "n", "type": "quantitative", "label": "Numerical initializations", "format": "number"},
        {"field": "analysis_status", "type": "nominal", "label": "Eligibility"},
    ]
    charts = [
        _chart("transfer_chart", "Transfer trajectories", "Baseline-referenced log2 gain; components and their difference are distinct.", "line", "transfer_trajectory", "src_primary", "severity_label", "ordinal", "Perturbation severity", "mean", "Mean log2 transfer change", color_field="series", tooltip=tooltip_common),
        _chart("transmission_audit_chart", "Broadband and frequency-locked endpoint interactions", "High severity versus baseline. Locked-transfer values are sensitivity-only because target frequency quality is poor.", "bar", "transmission_endpoint_audit", "src_posthoc_transmission", "label", "nominal", "Probe · metric", "interaction_mean", "Mean semantic − episodic change (log2)"),
        _chart("locked_segments_chart", "Frequency-locked transfer across two-second segments", "Target/A1 exact-frequency amplitude ratio; numerical magnitude is not interpretable as preserved locking when target QA is poor.", "line", "segment_frequency_audit", "src_posthoc_traces", "segment", "ordinal", "Two-second segment", "median_frequency_locked_transfer", "Median frequency-locked transfer", color_field="series"),
        _chart("spectra_chart", "High-severity evoked spectra", "Notebook DPSS multitaper method (NW=3, five tapers); log10 power over 0.5–12 Hz.", "line", "high_evoked_spectra", "src_posthoc_traces", "frequency_hz", "quantitative", "Frequency (Hz)", "log10_median_power", "log10 median power", color_field="series"),
        _chart("fc_chart", "Periodic functional-connectivity trajectories", "Fisher-z change from baseline; descriptive where the split-window gate failed.", "line", "fc_trajectory", "src_primary", "severity_label", "ordinal", "Perturbation severity", "mean", "Mean Fisher-z change", color_field="series", tooltip=tooltip_common),
        _chart("raw_fc_chart", "Raw periodic functional connectivity", "Absolute Fisher-z evoked A1–target correlation; episodic and semantic levels must be compared separately from change.", "line", "fc_raw_audit", "src_posthoc_transmission", "severity_label", "ordinal", "Perturbation severity", "mean", "Mean raw FC (Fisher-z)", color_field="series"),
        _chart("fc_change_audit_chart", "Baseline-referenced functional-connectivity change", "A positive semantic-minus-episodic interaction can coexist with lower semantic raw FC and declines in both proxies.", "line", "fc_change_audit", "src_posthoc_transmission", "severity_label", "ordinal", "Perturbation severity", "mean", "Mean FC change (Fisher-z)", color_field="series"),
        _chart("phase_fc_chart", "Phase shift and functional-connectivity loss", "Parcel-seed rows are shown only when notebook frequency QA passes at baseline and high severity; association is descriptive.", "scatter", "phase_fc_rows", "src_posthoc_transmission", "absolute_phase_shift_degrees", "quantitative", "Absolute A1-relative phase shift (degrees)", "fc_loss_z", "FC loss (Fisher-z)", color_field="series", tooltip=[{"field": "region_label", "type": "nominal", "label": "Parcel"}, {"field": "fit_r_squared", "type": "quantitative", "label": "High-severity harmonic R²"}, {"field": "snr_db", "type": "quantitative", "label": "High-severity SNR (dB)"}]),
        _chart("latency_chart", "Pulse response-latency trajectories", "Relative model-response timing in ms; not anatomical conduction latency.", "line", "latency_trajectory", "src_primary", "severity_label", "ordinal", "Perturbation severity", "mean", "Mean latency change (ms)", color_field="series", tooltip=tooltip_common),
        _chart("pulse_peak_chart", "Pulse peak magnitude by proxy", "Median parcel-level absolute evoked PSP peak across numerical initializations.", "line", "pulse_summary", "src_pulse", "severity_label", "ordinal", "Perturbation severity", "median_peak_psp", "Median peak PSP", color_field="series"),
        _chart("pulse_quantile_chart", "Fixed-mask pulse energy timing", "t20, t50, and t80 relative to ipsilateral A1; masks are fixed across severity within seed using the original validity rule.", "line", "pulse_fixed_quantiles", "src_posthoc_traces", "severity_label", "ordinal", "Perturbation severity", "relative_time_ms", "Median A1-relative time (ms)", color_field="series"),
        _chart("pulse_energy_chart", "Pulse evoked energy by proxy", "Median parcel-level total evoked energy; raw PSP model units squared × ms.", "line", "pulse_summary", "src_pulse", "severity_label", "ordinal", "Perturbation severity", "median_energy_psp2_ms", "Median evoked energy", color_field="series"),
        _chart("eligibility_chart", "Valid measurement rows by outcome", "Each confirmatory outcome requires all 120 network-condition-seed rows to pass its prespecified gate.", "bar", "eligibility", "src_quality", "label", "nominal", "Outcome · probe", "valid_fraction", "Valid fraction", value_format="percent"),
        _chart("fc_quality_chart", "FC split-window validity", "Failures are concentrated at intermediate severity; threshold is |first-half z − second-half z| ≤ 0.10.", "line", "fc_quality", "src_quality", "severity_label", "ordinal", "Perturbation severity", "pass_rate", "Valid fraction", color_field="series", value_format="percent"),
        _chart("latency_quality_chart", "Pulse-latency validity", "Requires ≥80% valid parcels and ≤10% median tail energy at the network-row level.", "line", "latency_quality", "src_quality", "severity_label", "ordinal", "Perturbation severity", "pass_rate", "Valid fraction", color_field="series", value_format="percent"),
        _chart("periodic_slope_chart", "Periodic transfer escalation", "Mean log2 transfer slope per second; positive values indicate growth across the fixed analysis window.", "bar", "periodic_quality", "src_quality", "severity_label", "ordinal", "Perturbation severity", "mean_transfer_slope_log2_per_s", "Mean log2 slope / s", color_field="series"),
        _chart("segments_chart", "High-severity transfer across five segments", "Five consecutive two-second summaries reveal within-window escalation.", "line", "segment_transfer", "src_quality", "segment", "ordinal", "Two-second segment", "mean_transfer", "Mean transfer", color_field="series"),
        _chart("dt_chart", "Integration-step precision ratio", "Ratio >1 exceeds the unchanged exact-magnitude target; all fatal validity checks passed.", "bar", "dt_eligibility", "src_dt", "label", "nominal", "Outcome · probe", "tolerance_ratio", "Observed difference / tolerance"),
        _chart("counterfactual_chart", "Local-fixed counterfactual attenuation", "Negative values mean amplification; dependence is model-internal, not biological causality.", "bar", "counterfactual", "src_robustness", "label", "nominal", "Outcome · probe", "median_attenuation_percent", "Median attenuation (%)"),
        _chart("shuffle_chart", "Spatial-shuffle empirical percentiles", "Seed 11 compared with 50 spatial shuffles; descriptive simulation ranks, not clinical p-values.", "bar", "spatial_shuffle", "src_robustness", "label", "nominal", "Outcome · probe", "observed_empirical_percentile", "Empirical percentile"),
        _chart("parameter_transfer_chart", "Transfer parameter sensitivity", "High-endpoint interaction under four prespecified parameter variants; five initializations each.", "bar", "parameter_transfer", "src_sensitivity", "variant", "nominal", "Variant", "mean_interaction", "Mean interaction (log2)", color_field="series"),
        _chart("parameter_fc_chart", "FC parameter sensitivity", "G100 reverses the interaction sign at both periodic frequencies.", "bar", "parameter_fc", "src_sensitivity", "variant", "nominal", "Variant", "mean_interaction", "Mean interaction (Fisher-z)", color_field="series"),
        _chart("definition_chart", "Transfer definition sensitivity", "Anatomical-core, Platel-peak, and unilateral operational definitions; five initializations are not participants.", "bar", "definition_transfer", "src_sensitivity", "definition", "nominal", "Proxy definition", "mean_interaction", "Mean interaction (log2)", color_field="series"),
        _chart("regional_chart", "Regional pathology and A1 affinity", "All 379 regions; membership is an operational proxy classification.", "scatter", "regional_features", "src_regions", "b_reduction", "quantitative", "Inhibitory-rate reduction", "bilateral_a1_affinity", "Bilateral A1 structural affinity", color_field="proxy_membership", tooltip=[{"field": "region_label", "type": "nominal", "label": "Region"}, {"field": "weighted_structural_strength", "type": "quantitative", "label": "Weighted strength"}]),
        _chart("regional_covariate_chart", "Exploratory regional transfer coefficients", "Parcel-level OLS after averaging 20 numerical initializations within each of 32 proxy parcels; continuous covariates are standardized.", "bar", "regional_covariate_coefficients", "src_posthoc_covariates", "term_label", "nominal", "Exploratory predictor", "estimate", "Coefficient (log2 transfer change)", color_field="series"),
        _chart("pathology_chart", "Pathology transformation by severity", "Surrogate amyloid-linked b transformation; artificial model endpoint, not patient data.", "line", "pathology", "src_regions", "severity", "quantitative", "Perturbation severity", "b_mean", "Mean inhibitory rate b"),
        _chart("runtime_chart", "Recorded simulation wall time by workload", "Summed worker-call wall time is not elapsed clock time because calls ran in parallel.", "bar", "runtime", "src_runtime", "workload", "nominal", "Workload", "recorded_wall_hours", "Recorded wall hours"),
        _chart("calibration_chart", "Baseline coupling calibration", "G60 maximized the locked balanced target score among six candidate couplings.", "line", "calibration", "src_runtime", "global_coupling", "quantitative", "Global coupling", "balanced_target_score", "Balanced target score"),
    ]
    # Keep the first viewport chart-led while bounding the number of live chart
    # renderers. Removed chart-ready datasets remain available in the snapshot,
    # source catalog, detail tables, and expanded data explorer.
    selected_chart_ids = {
        "transfer_chart",
        "transmission_audit_chart",
        "locked_segments_chart",
        "spectra_chart",
        "fc_chart",
        "raw_fc_chart",
        "fc_change_audit_chart",
        "phase_fc_chart",
        "latency_chart",
        "pulse_peak_chart",
        "pulse_quantile_chart",
        "fc_quality_chart",
        "latency_quality_chart",
        "segments_chart",
        "dt_chart",
        "counterfactual_chart",
        "parameter_transfer_chart",
        "parameter_fc_chart",
        "regional_chart",
        "regional_covariate_chart",
        "calibration_chart",
    }
    charts = [chart for chart in charts if chart["id"] in selected_chart_ids]

    tables_manifest = [
        {"id": "transmission_audit_table", "title": "Transmission endpoint audit", "subtitle": "Broadband and exact-frequency metrics are kept separate; intervals describe numerical initialization variation.", "dataset": "transmission_endpoint_audit", "sourceId": "src_posthoc_transmission", "defaultSort": {"field": "label", "direction": "asc"}, "columns": [{"field": "label", "label": "Probe · metric"}, {"field": "semantic_mean", "label": "Semantic change", "format": "number"}, {"field": "episodic_mean", "label": "Episodic change", "format": "number"}, {"field": "interaction_mean", "label": "Interaction", "format": "number"}, {"field": "interaction_ci95_lower", "label": "CI low", "format": "number"}, {"field": "interaction_ci95_upper", "label": "CI high", "format": "number"}, {"field": "positive_seed_count", "label": "Positive seeds", "format": "number"}, {"field": "analysis_status", "label": "Status"}]},
        {"id": "frequency_quality_table", "title": "Target frequency-quality audit", "subtitle": "Phase is not interpreted where the notebook SNR/phase-consistency flag fails; harmonic R² remains visible as an additional diagnostic.", "dataset": "frequency_quality_audit", "sourceId": "src_posthoc_transmission", "defaultSort": {"field": "severity", "direction": "asc"}, "columns": [{"field": "severity", "label": "Severity", "format": "number"}, {"field": "probe", "label": "Probe"}, {"field": "network", "label": "Proxy"}, {"field": "median_harmonic_fit_r_squared", "label": "Median R²", "format": "number"}, {"field": "median_target_snr_db", "label": "Median SNR dB", "format": "number"}, {"field": "median_phase_consistency", "label": "Phase consistency", "format": "number"}, {"field": "median_frequency_qa_valid_fraction", "label": "QA-valid fraction", "format": "percent"}, {"field": "nonstationary_fraction", "label": "Nonstationary fraction", "format": "percent"}]},
        {"id": "spectral_peak_table", "title": "Spectral peak versus applied frequency", "subtitle": "Stimulated, control, and evoked spectra use the notebook's DPSS settings; drive/peak is descriptive power concentration.", "dataset": "spectral_peaks", "sourceId": "src_posthoc_traces", "defaultSort": {"field": "severity", "direction": "asc"}, "columns": [{"field": "severity", "label": "Severity", "format": "number"}, {"field": "probe", "label": "Probe"}, {"field": "network", "label": "Proxy"}, {"field": "signal", "label": "Signal"}, {"field": "peak_frequency_hz", "label": "Peak Hz", "format": "number"}, {"field": "drive_frequency_hz", "label": "Drive Hz", "format": "number"}, {"field": "drive_to_peak_power_ratio", "label": "Drive/peak power", "format": "percent"}]},
        {"id": "phase_fc_table", "title": "Phase–FC descriptive associations", "subtitle": "Rows failing frequency QA are excluded before phase analysis; p-values are post-hoc descriptive, not confirmatory or clinical.", "dataset": "phase_fc_summary", "sourceId": "src_posthoc_transmission", "defaultSort": {"field": "probe", "direction": "asc"}, "columns": [{"field": "probe", "label": "Probe"}, {"field": "network", "label": "Proxy"}, {"field": "all_rows", "label": "All parcel-seed rows", "format": "number"}, {"field": "phase_interpretable_rows", "label": "Phase-valid rows", "format": "number"}, {"field": "phase_interpretable_fraction", "label": "Valid fraction", "format": "percent"}, {"field": "spearman_phase_shift_vs_fc_loss", "label": "Spearman ρ", "format": "number"}, {"field": "median_phase_shift_degrees", "label": "Median shift °", "format": "number"}]},
        {"id": "pulse_fixed_mask_table", "title": "Fixed pulse-parcel masks by initialization", "subtitle": "The original latency_valid rule is intersected across baseline, intermediate, and high severity within each seed; no threshold is weakened.", "dataset": "pulse_fixed_masks", "sourceId": "src_posthoc_traces", "defaultSort": {"field": "fixed_valid_parcels", "direction": "desc"}, "columns": [{"field": "seed", "label": "Seed", "format": "number"}, {"field": "network", "label": "Proxy"}, {"field": "declared_parcels", "label": "Declared", "format": "number"}, {"field": "fixed_valid_parcels", "label": "Fixed valid", "format": "number"}, {"field": "fixed_valid_fraction", "label": "Retained", "format": "percent"}, {"field": "usable_seed", "label": "Usable"}]},
        {"id": "regional_covariate_table", "title": "Exploratory regional covariate model", "subtitle": "Outcome is mean high-versus-baseline log2 node-transfer change across seeds; 32 operational proxy parcels, classical OLS intervals.", "dataset": "regional_covariate_coefficients", "sourceId": "src_posthoc_covariates", "defaultSort": {"field": "term_label", "direction": "asc"}, "columns": [{"field": "probe", "label": "Probe"}, {"field": "term_label", "label": "Predictor"}, {"field": "estimate", "label": "Coefficient", "format": "number"}, {"field": "standard_error", "label": "SE", "format": "number"}, {"field": "ci95_lower", "label": "CI low", "format": "number"}, {"field": "ci95_upper", "label": "CI high", "format": "number"}, {"field": "descriptive_p_value", "label": "Descriptive p", "format": "number"}]},
        {"id": "eligibility_table", "title": "Final outcome eligibility", "subtitle": "Authoritative final labels; these supersede stale labels in primary_interaction_statistics.csv.", "dataset": "eligibility", "sourceId": "src_quality", "defaultSort": {"field": "hierarchy", "direction": "asc"}, "columns": [{"field": "outcome", "label": "Outcome"}, {"field": "probe", "label": "Probe"}, {"field": "hierarchy", "label": "Hierarchy"}, {"field": "valid_rows", "label": "Valid", "format": "number"}, {"field": "required_rows", "label": "Required", "format": "number"}, {"field": "status_label", "label": "Final status"}, {"field": "dt_label", "label": "DT status"}]},
        {"id": "pulse_table", "title": "Pulse-specific summaries", "subtitle": "FC is not a pulse metric; these peak, energy, timing, and validity fields are the pulse investigation surface.", "dataset": "pulse_summary", "sourceId": "src_pulse", "defaultSort": {"field": "severity", "direction": "asc"}, "columns": [{"field": "severity", "label": "Severity", "format": "number"}, {"field": "network", "label": "Proxy"}, {"field": "median_peak_psp", "label": "Median peak PSP", "format": "number"}, {"field": "median_energy_psp2_ms", "label": "Median energy", "format": "number"}, {"field": "median_peak_relative_ms", "label": "Peak vs A1 (ms)", "format": "number"}, {"field": "median_latency_ms", "label": "Energy latency (ms)", "format": "number"}, {"field": "median_valid_latency_fraction", "label": "Valid parcel fraction", "format": "percent"}]},
        {"id": "matched_table", "title": "Matched-control reference distributions", "subtitle": "Across 20 numerical initializations; model-generated references, not clinical p-values.", "dataset": "matched_null", "sourceId": "src_robustness", "defaultSort": {"field": "median_empirical_percentile", "direction": "desc"}, "columns": [{"field": "outcome", "label": "Outcome"}, {"field": "probe", "label": "Probe"}, {"field": "median_empirical_percentile", "label": "Median percentile", "format": "number"}, {"field": "minimum_empirical_percentile", "label": "Minimum", "format": "number"}, {"field": "maximum_empirical_percentile", "label": "Maximum", "format": "number"}, {"field": "outside_central_90_count", "label": "Outside 90%", "format": "number"}]},
        {"id": "laterality_table", "title": "Right-minus-left interaction sensitivity", "subtitle": "Strong laterality differences show that bilateral conclusions should not be generalized to unilateral proxies.", "dataset": "laterality", "sourceId": "src_sensitivity", "defaultSort": {"field": "mean_right_minus_left_interaction", "direction": "desc"}, "columns": [{"field": "outcome", "label": "Outcome"}, {"field": "probe", "label": "Probe"}, {"field": "unit", "label": "Unit"}, {"field": "mean_right_minus_left_interaction", "label": "Right − left", "format": "number"}, {"field": "ci95_lower_numerical", "label": "CI low", "format": "number"}, {"field": "ci95_upper_numerical", "label": "CI high", "format": "number"}]},
        {"id": "mapping_table", "title": "Platel peak-to-HCP mapping", "subtitle": "Approximate atlas mapping retained with its recorded distances and notes.", "dataset": "mapping", "sourceId": "src_regions", "defaultSort": {"field": "mapping_distance_mm", "direction": "desc"}, "columns": [{"field": "source_contrast", "label": "Source contrast"}, {"field": "reported_region", "label": "Reported region"}, {"field": "hcp_label", "label": "HCP label"}, {"field": "mapping_distance_mm", "label": "Distance (mm)", "format": "number"}, {"field": "mapping_note", "label": "Mapping note"}]},
        {"id": "runtime_table", "title": "Execution diagnostics", "subtitle": "Workload-level call counts, runtime distribution, worker coverage, and maximum PSP.", "dataset": "runtime", "sourceId": "src_runtime", "defaultSort": {"field": "recorded_wall_hours", "direction": "desc"}, "columns": [{"field": "workload", "label": "Workload"}, {"field": "calls", "label": "Calls", "format": "number"}, {"field": "recorded_wall_hours", "label": "Wall hours", "format": "number"}, {"field": "median_call_seconds", "label": "Median call (s)", "format": "number"}, {"field": "p95_call_seconds", "label": "P95 call (s)", "format": "number"}, {"field": "worker_processes", "label": "Workers", "format": "number"}, {"field": "maximum_abs_psp", "label": "Max |PSP|", "format": "number"}]},
        {"id": "catalog_table", "title": "Complete non-NPZ source catalog", "subtitle": "Every top-level experiment CSV/JSON, run log, and existing figure asset is represented; SHA-256 and full schemas remain embedded in the reviewed dataset.", "dataset": "source_catalog", "sourceId": "src_catalog", "defaultSort": {"field": "file", "direction": "asc"}, "columns": [{"field": "file", "label": "File"}, {"field": "kind", "label": "Kind"}, {"field": "group", "label": "Analysis area"}, {"field": "rows", "label": "Rows", "format": "number"}, {"field": "columns", "label": "Columns", "format": "number"}, {"field": "null_cells", "label": "Null cells", "format": "number"}, {"field": "duplicate_rows", "label": "Duplicate rows", "format": "number"}, {"field": "size_bytes", "label": "Bytes", "format": "compact"}]},
    ]

    blocks = [
        {"id": "intro", "type": "markdown", "body": "## Research question\n\nHow does increasing AD-like amyloid-linked inhibitory perturbation differentially affect stimulus-evoked transmission into expanded musical-semantic-associated and musical-episodic-associated proxy parcel sets?\n\nThis is a read-only snapshot of the completed model experiment. Numerical initializations are not participants; the public amyloid endpoint is artificial; proxy parcel sets are operational; and simulation nulls are not clinical p-values."},
        {"id": "metrics", "type": "metric-strip", "cardIds": ["calls", "sources", "trace_manifest", "transfer2", "transfer5"]},
        {"id": "transmission_audit_heading", "type": "markdown", "sourceId": "src_posthoc_transmission", "body": "## Broadband versus frequency-specific transmission\n\nThe audit supports **Outcome A, with an additional warning about ratios**. High-severity semantic-minus-episodic change is positive for broadband RMS transfer at both probes (+2.233 at 2 Hz; +2.172 at 5 Hz). The frequency-locked target/A1 ratio is also numerically positive for all 20 seeds (+3.655 and +3.619 log2), but it is sensitivity-only: perturbed target harmonic fits are extremely weak and most target parcels fail the notebook frequency-QA rule. Therefore the supported claim is greater **broadband evoked-response amplification**, not stronger preservation of the applied frequency."},
        {"id": "transmission_audit", "type": "chart", "chartId": "transmission_audit_chart"},
        {"id": "transmission_audit_detail", "type": "table", "tableId": "transmission_audit_table"},
        {"id": "locked_segments", "type": "chart", "chartId": "locked_segments_chart"},
        {"id": "frequency_quality_detail", "type": "table", "tableId": "frequency_quality_table"},
        {"id": "spectral_heading", "type": "markdown", "sourceId": "src_posthoc_traces", "body": "## Spectral audit of the lossless traces\n\nAt baseline, the evoked spectral peak equals the applied 2 or 5 Hz frequency. At intermediate and high perturbation, the dominant evoked peaks move elsewhere. At high severity, applied-frequency power is only about 0.015%–0.97% of peak power across proxy/probe combinations. This directly explains why a large broadband response or a large target/A1 fitted-amplitude ratio cannot be treated as reliable frequency locking."},
        {"id": "spectra", "type": "chart", "chartId": "spectra_chart"},
        {"id": "spectral_peak_detail", "type": "table", "tableId": "spectral_peak_table"},
        {"id": "fc_audit_heading", "type": "markdown", "sourceId": "src_posthoc_transmission", "body": "## Raw FC, FC change, and phase\n\nRaw episodic-proxy FC is higher than raw semantic-proxy FC at baseline, intermediate, and high severity, while both decline sharply. The positive semantic-minus-episodic interaction means the semantic proxy loses slightly less FC; it does not mean semantic FC is higher or preserved. Phase–FC plots include only parcel-seed rows passing frequency QA at baseline and high severity, so phase is not assigned meaning to weak harmonic fits."},
        {"id": "raw_fc", "type": "chart", "chartId": "raw_fc_chart"},
        {"id": "fc_change_audit", "type": "chart", "chartId": "fc_change_audit_chart"},
        {"id": "phase_fc", "type": "chart", "chartId": "phase_fc_chart"},
        {"id": "phase_fc_detail", "type": "table", "tableId": "phase_fc_table"},
        {"id": "pulse_fc", "type": "markdown", "body": "## What is FC under the pulse probe?\n\n**It is not defined in the locked experiment.** Functional connectivity is a periodic-probe estimand computed from the correlation of target and ipsilateral-A1 evoked PSP traces. For the transient pulse, the notebook deliberately records FC fields as not applicable and instead evaluates response latency, absolute peak magnitude, peak timing relative to A1, total evoked energy, valid-parcel coverage, and tail-energy completeness. This is not missing data and not a failed FC gate."},
        {"id": "pulse_peak", "type": "chart", "chartId": "pulse_peak_chart"},
        {"id": "pulse_fixed_heading", "type": "markdown", "sourceId": "src_posthoc_traces", "body": "### Fixed-parcel pulse timing audit\n\nUsing the original parcel validity rule at all three severities within each seed retains a usable fixed set for 18/20 semantic and 16/20 episodic initializations (median 10 and 15 parcels, respectively). The audit adds t20, t50, t80, and t80−t20 without changing the notebook. Because some seeds retain no parcels and intermediate-severity timing remains highly variable, this is a secondary robustness view rather than a replacement endpoint."},
        {"id": "pulse_quantiles", "type": "chart", "chartId": "pulse_quantile_chart"},
        {"id": "pulse_fixed_mask_detail", "type": "table", "tableId": "pulse_fixed_mask_table"},
        {"id": "pulse_detail", "type": "table", "tableId": "pulse_table"},
        {"id": "primary_heading", "type": "markdown", "body": "## Primary trajectories\n\nTransfer increased in both proxies, with the semantic-associated increase larger at the high endpoint. FC decreased in both proxies, with a smaller decline in the semantic proxy. Pulse latency lengthened in both proxies, and more in the semantic proxy—so the positive latency interaction is not semantic timing preservation."},
        {"id": "transfer", "type": "chart", "chartId": "transfer_chart"},
        {"id": "fc", "type": "chart", "chartId": "fc_chart"},
        {"id": "latency", "type": "chart", "chartId": "latency_chart"},
        {"id": "quality_heading", "type": "markdown", "body": "## Measurement quality and temporal behavior\n\nTransfer passed its within-run measurement gate for all 120 required rows. Periodic FC failures are concentrated at intermediate severity because the first- and second-half FC estimates diverged beyond 0.10 Fisher-z. Pulse-latency failures are also concentrated at intermediate severity because responses were tail-heavy or too few parcels had a complete latency estimate. Raw outputs remain recorded."},
        {"id": "eligibility_detail", "type": "table", "tableId": "eligibility_table"},
        {"id": "fc_quality", "type": "chart", "chartId": "fc_quality_chart"},
        {"id": "latency_quality", "type": "chart", "chartId": "latency_quality_chart"},
        {"id": "segments", "type": "chart", "chartId": "segments_chart"},
        {"id": "dt", "type": "chart", "chartId": "dt_chart"},
        {"id": "robustness_heading", "type": "markdown", "body": "## Robustness and model-null context\n\nLocal fixing attenuated the transfer interaction by about 72% and pulse latency by about 80% at the median, while FC interaction magnitude increased. This is model-internal dependence, not biological causality. Transfer ranks high against the seed-11 spatial shuffles, but observed matched-control contrasts are often inside their central 90% ranges; neither reference is a clinical p-value."},
        {"id": "counterfactual", "type": "chart", "chartId": "counterfactual_chart"},
        {"id": "matched", "type": "table", "tableId": "matched_table"},
        {"id": "sensitivity_heading", "type": "markdown", "body": "## Sensitivity findings\n\nTransfer interaction direction remains positive across the tested G30/G100 and input-amplitude variants. FC reverses sign at G100. Operational parcel definition and laterality materially change magnitude: right-only transfer and latency interactions are much larger than left-only, while right-minus-left FC is negative."},
        {"id": "parameter_transfer", "type": "chart", "chartId": "parameter_transfer_chart"},
        {"id": "parameter_fc", "type": "chart", "chartId": "parameter_fc_chart"},
        {"id": "laterality", "type": "table", "tableId": "laterality_table"},
        {"id": "regional_heading", "type": "markdown", "body": "## Regional model context\n\nThe pathology axis is a surrogate amyloid-linked transformation of the inhibitory rate parameter b. The scatter is structural/model context, not a patient biomarker relationship. Platel-to-HCP mapping is approximate and the proxy networks overlap."},
        {"id": "regional", "type": "chart", "chartId": "regional_chart"},
        {"id": "regional_covariate_heading", "type": "markdown", "sourceId": "src_posthoc_covariates", "body": "### Exploratory regional covariate model\n\nAfter averaging the 20 numerical initializations within each of 32 proxy parcels, the semantic-versus-episodic coefficient remains about +2.32 log2 at both probes after adjustment for b reduction, bilateral A1 affinity, weighted structural strength, and hemisphere. This is a small, post-hoc parcel-level OLS model with correlated anatomical features; its classical interval is descriptive and does not establish biological specificity."},
        {"id": "regional_covariates", "type": "chart", "chartId": "regional_covariate_chart"},
        {"id": "regional_covariate_detail", "type": "table", "tableId": "regional_covariate_table"},
        {"id": "mapping", "type": "table", "tableId": "mapping_table"},
        {"id": "execution_heading", "type": "markdown", "body": "## Calibration, execution, and source coverage\n\nG60 maximized the locked baseline balance score. The 0.25 ms integration-step reference consumed the largest aggregate worker-call time. The source catalog below proves coverage of every non-NPZ experiment table and exposes bounded samples without inflating the HTML with the 128 MB row-level export."},
        {"id": "calibration", "type": "chart", "chartId": "calibration_chart"},
        {"id": "runtime_detail", "type": "table", "tableId": "runtime_table"},
        {"id": "catalog", "type": "table", "tableId": "catalog_table"},
        {"id": "limits", "type": "markdown", "body": "## Interpretation boundaries\n\n- Numerical seeds measure initialization sensitivity, not biological replication.\n- FC is evoked PSP correlation, not BOLD FC or directed effective connectivity.\n- Zero tract delays make pulse latency a relative model-response timing metric.\n- The 2 Hz and 5 Hz probes are temporal inputs, not literal music or speech.\n- The model contains no memory task, encoding, recollection, familiarity, or behavior.\n- Low target harmonic quality and integration-step sensitivity remain visible rather than being silently filtered."},
    ]

    filters = [
        {"id": "severity_filter", "label": "Severity", "dataset": "transfer_trajectory", "field": "severity_label", "includeAll": True, "targets": [{"dataset": name, "field": "severity_label"} for name in ["transfer_trajectory", "fc_trajectory", "latency_trajectory", "pulse_summary", "fc_quality", "latency_quality", "periodic_quality", "frequency_quality_audit", "segment_frequency_audit", "fc_raw_audit", "fc_change_audit", "spectra_stimulated", "spectra_control", "spectra_evoked", "high_evoked_spectra", "pulse_fixed_quantiles"]]},
        {"id": "probe_filter", "label": "Probe", "dataset": "transfer_trajectory", "field": "probe", "includeAll": True, "targets": [{"dataset": name, "field": "probe"} for name in ["transfer_trajectory", "fc_trajectory", "latency_trajectory", "pulse_summary", "fc_quality", "latency_quality", "periodic_quality", "transmission_endpoint_audit", "frequency_quality_audit", "segment_frequency_audit", "fc_raw_audit", "fc_change_audit", "phase_fc_rows", "spectra_stimulated", "spectra_control", "spectra_evoked", "high_evoked_spectra"]]},
        {"id": "network_filter", "label": "Series / proxy", "dataset": "pulse_summary", "field": "network", "includeAll": True, "targets": [{"dataset": name, "field": "network"} for name in ["pulse_summary", "fc_quality", "latency_quality", "periodic_quality", "frequency_quality_audit", "segment_frequency_audit", "fc_raw_audit", "fc_change_audit", "phase_fc_rows", "spectra_stimulated", "spectra_control", "spectra_evoked", "high_evoked_spectra", "pulse_fixed_quantiles"]]},
    ]

    generated_at = str(metadata.get("created_utc", status.get("updated_utc")))
    artifact = {
        "surface": "dashboard",
        "manifest": {
            "version": 1,
            "surface": "dashboard",
            "title": "TVB379 semantic–episodic result investigator",
            "description": "Offline visual investigation of the completed experiment export, including bounded post-hoc summaries from all 180 lossless trace shards.",
            "generatedAt": generated_at,
            "filters": filters,
            "cards": cards,
            "charts": charts,
            "tables": tables_manifest,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": canonical_sources,
        "package_info": {
            "originUrl": "artifact://tvb379-semantic-episodic-investigator",
            "controls": {
                "edit": False,
                "refresh": False,
                "persistence": False,
                "export": False,
                "share": False,
            },
        },
    }
    encoded_size = len(json.dumps(artifact, ensure_ascii=False).encode("utf-8"))
    if encoded_size >= 3_000_000:
        raise DashboardBuildError(
            f"Artifact payload is {encoded_size:,} bytes; expected under 3 MB."
        )
    return artifact


def find_portable_builder() -> Path:
    roots = sorted(
        (Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics").glob(
            "*/skills/build-report/scripts/deliver_portable_artifact.mjs"
        ),
        reverse=True,
    )
    if not roots:
        raise DashboardBuildError(
            "Could not locate the Data Analytics portable HTML builder. "
            "Pass --portable-builder PATH."
        )
    return roots[0]


def build_html(
    artifact_path: Path,
    output_path: Path,
    portable_builder: Path,
) -> None:
    scripts_dir = portable_builder.parent
    packager = Path(__file__).resolve().with_name(
        "package_html_investigator.mjs"
    )
    package_command = [
        "node",
        str(packager),
        "--input",
        str(artifact_path),
        "--output",
        str(output_path),
        "--builder",
        str(scripts_dir / "build_portable_artifact.mjs"),
        "--extractor",
        str(scripts_dir / "extract_portable_chart_svgs.mjs"),
    ]
    subprocess.run(package_command, check=True)
    verify_command = [
        "node",
        str(scripts_dir / "verify_portable_artifact.mjs"),
        "--html",
        str(output_path),
        "--artifact",
        str(artifact_path),
        "--ready-timeout-ms",
        "10000",
        "--action-timeout-ms",
        "5000",
        "--timeout-ms",
        "30000",
    ]
    subprocess.run(verify_command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the self-contained TVB379 HTML result investigator."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_JSON)
    parser.add_argument("--output", type=Path, default=OUTPUT_HTML)
    parser.add_argument("--portable-builder", type=Path, default=None)
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help="write validated-input JSON without packaging the HTML",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    artifact_path = args.artifact.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    try:
        artifact = build_artifact(data_dir)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Artifact: {artifact_path}")
        print(
            "Coverage: "
            f"{len(list(data_dir.glob('*.csv')))} CSV, "
            f"{len(list(data_dir.glob('*.json')))} JSON, log and figure assets; "
            f"{EXPECTED_TRACE_SHARDS} NPZ trace shards analyzed; raw arrays "
            "not embedded"
        )
        if not args.artifact_only:
            builder = (
                args.portable_builder.expanduser().resolve()
                if args.portable_builder is not None
                else find_portable_builder()
            )
            build_html(artifact_path, output_path, builder)
            print(f"HTML: {output_path}")
        return 0
    except (DashboardBuildError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
