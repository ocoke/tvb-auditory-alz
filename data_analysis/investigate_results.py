#!/usr/bin/env python3
"""Validate, summarize, visualize, and inspect the TVB379 result export.

This is a direct-run research tool, not an installable package. It uses the
experiment's saved tables and raw trace shards; it never reruns TVB.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rise_tvb379_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_SUBDIR = "investigation"
PRIMARY_PAIR = "expanded_bilateral"
EXPECTED_SEVERITIES = (0.0, 0.5, 1.0)
EXPECTED_PROBES = ("pulse", "2Hz", "5Hz")
EXPECTED_FINAL_CALLS = 762
EXPECTED_MANIFESTED_CALLS = 750
EXPECTED_TRACE_SHARDS = 180
EXPECTED_REGIONS = 379
EXPECTED_TRACE_REGIONS = 34
EXPECTED_CSV_TABLES = 48

BLUE = "#3568A8"
BLUE_LIGHT = "#AFC6E4"
ORANGE = "#C76E2B"
ORANGE_LIGHT = "#E7BC98"
GOLD = "#B48A24"
OLIVE = "#75854B"
INK = "#242A30"
MID_GREY = "#727A82"
LIGHT_GREY = "#D9DEE3"
PALE_GREY = "#F4F6F7"


OUTCOMES: tuple[tuple[str, str, str, str], ...] = (
    ("transfer_gain", "2Hz", "Transfer gain — 2 Hz", "log2 ratio"),
    ("transfer_gain", "5Hz", "Transfer gain — 5 Hz", "log2 ratio"),
    (
        "functional_connectivity",
        "2Hz",
        "Functional connectivity — 2 Hz",
        "Fisher-z difference",
    ),
    (
        "functional_connectivity",
        "5Hz",
        "Functional connectivity — 5 Hz",
        "Fisher-z difference",
    ),
    ("response_latency", "pulse", "Response latency — pulse", "ms"),
)

REQUIRED_FILES = (
    "analysis_spec.json",
    "experiment_metadata.json",
    "run_status.json",
    "primary_interaction_statistics.csv",
    "main_pair_interactions.csv",
    "outcome_eligibility.csv",
    "integration_step_outcome_eligibility.csv",
    "counterfactual_summary.csv",
    "matched_control_null_summary.csv",
    "spatial_shuffle_summary.csv",
    "parameter_interaction_statistics.csv",
    "main_science_validity.csv",
    "run_manifest.csv",
    "baseline_coupling_diagnostic.csv",
    "main_parcel_trace_manifest.csv",
    "regional_features.csv",
)


class InvestigationError(RuntimeError):
    """Raised when the export cannot support a trustworthy analysis."""


@dataclass(frozen=True)
class FigureRecord:
    figure: str
    title: str
    format: str
    path: str
    sha256: str
    size_bytes: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvestigationError(f"Cannot read valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise InvestigationError(f"Expected a JSON object: {path}")
    return value


def _read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    try:
        return pd.read_csv(path)
    except Exception as error:
        raise InvestigationError(f"Cannot read CSV table: {path}") from error


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    filename: str,
) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise InvestigationError(
            f"{filename} is missing columns: {', '.join(sorted(missing))}"
        )


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_token(value: Any) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    ).strip("_")


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": MID_GREY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _figure_heading(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.07, y=0.985, ha="left", fontsize=15, weight="bold")
    fig.text(0.07, 0.955, subtitle, ha="left", va="top", color=MID_GREY)
    fig.text(0.965, 0.972, "❉", ha="right", va="top", color=BLUE, fontsize=16)


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    title: str,
    formats: Sequence[str],
    dpi: int,
) -> list[FigureRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[FigureRecord] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi if extension == "png" else None)
        records.append(
            FigureRecord(
                figure=stem,
                title=title,
                format=extension,
                path=str(path.resolve()),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
            )
        )
    plt.close(fig)
    return records


def summarize_values(values: Sequence[float]) -> dict[str, float | int | bool]:
    """Return descriptive and numerical-initialization statistics."""

    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if len(numeric) == 0:
        raise InvestigationError("Cannot summarize an empty finite sample.")
    n = len(numeric)
    mean = float(np.mean(numeric))
    median = float(np.median(numeric))
    minimum = float(np.min(numeric))
    maximum = float(np.max(numeric))
    sd = float(np.std(numeric, ddof=1)) if n > 1 else math.nan
    if n > 1 and sd > 1e-15:
        critical = float(stats.t.ppf(0.975, n - 1))
        half_width = critical * sd / math.sqrt(n)
        t_statistic = mean / (sd / math.sqrt(n))
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), n - 1))
        hedges_correction = 1.0 - 3.0 / (4.0 * n - 5.0)
        hedges_gz = hedges_correction * mean / sd
    elif n > 1:
        half_width = 0.0
        t_statistic = math.nan
        p_value = math.nan
        hedges_gz = math.nan
    else:
        half_width = math.nan
        t_statistic = math.nan
        p_value = math.nan
        hedges_gz = math.nan
    return {
        "n_numerical_initializations": n,
        "mean": mean,
        "median": median,
        "sd": sd,
        "minimum": minimum,
        "maximum": maximum,
        "ci95_lower_numerical": mean - half_width,
        "ci95_upper_numerical": mean + half_width,
        "t_statistic_vs_zero": t_statistic,
        "p_value_vs_zero_numerical": p_value,
        "paired_hedges_gz": hedges_gz,
        "positive_fraction": float(np.mean(numeric > 0.0)),
        "sign_consistent": bool(np.all(numeric > 0.0) or np.all(numeric < 0.0)),
    }


def _eligibility_map(data_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame = _read_csv(data_dir, "outcome_eligibility.csv")
    _require_columns(
        frame,
        (
            "outcome",
            "probe",
            "hierarchy",
            "analysis_status",
            "all_quality_gates_passed",
        ),
        "outcome_eligibility.csv",
    )
    return {
        (str(row.outcome), str(row.probe)): row._asdict()
        for row in frame.itertuples(index=False)
    }


def build_hypothesis_statistics(
    data_dir: Path,
    pair: str = PRIMARY_PAIR,
) -> pd.DataFrame:
    interactions = _read_csv(data_dir, "main_pair_interactions.csv")
    _require_columns(
        interactions,
        (
            "pair",
            "outcome",
            "probe",
            "unit",
            "severity",
            "seed",
            "semantic_value",
            "episodic_value",
            "semantic_minus_episodic_interaction",
        ),
        "main_pair_interactions.csv",
    )
    interactions = interactions[interactions["pair"] == pair].copy()
    if interactions.empty:
        raise InvestigationError(f"Pair {pair!r} does not exist.")
    eligibility = _eligibility_map(data_dir)
    series_columns = {
        "semantic_component": "semantic_value",
        "episodic_component": "episodic_value",
        "semantic_minus_episodic_interaction": (
            "semantic_minus_episodic_interaction"
        ),
    }
    rows: list[dict[str, Any]] = []
    for (outcome, probe, unit, severity), group in interactions.groupby(
        ["outcome", "probe", "unit", "severity"],
        sort=True,
    ):
        status = eligibility.get((str(outcome), str(probe)), {})
        for series, column in series_columns.items():
            summary = summarize_values(group[column].to_numpy(dtype=float))
            rows.append(
                {
                    "pair": pair,
                    "outcome": outcome,
                    "probe": probe,
                    "unit": unit,
                    "severity": float(severity),
                    "series": series,
                    **summary,
                    "hierarchy": status.get("hierarchy", "unknown"),
                    "analysis_status": status.get(
                        "analysis_status", "unknown"
                    ),
                    "all_quality_gates_passed": status.get(
                        "all_quality_gates_passed", False
                    ),
                    "integration_step_status": status.get(
                        "integration_step_status", "not_applicable"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["outcome", "probe", "severity", "series"]
    ).reset_index(drop=True)


def validate_export(
    data_dir: Path,
    *,
    verify_trace_hashes: bool = False,
) -> pd.DataFrame:
    """Run high-value completeness, uniqueness, and consistency checks."""

    data_dir = data_dir.resolve()
    rows: list[dict[str, str]] = []

    def add(check: str, status: str, evidence: str, risk: str = "") -> None:
        rows.append(
            {
                "check": check,
                "status": status,
                "evidence": evidence,
                "analytical_risk": risk,
            }
        )

    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    add(
        "required_files",
        "error" if missing else "pass",
        "missing: " + ", ".join(missing) if missing else "all required files present",
        "Core results cannot be interpreted." if missing else "",
    )
    if missing:
        return pd.DataFrame(rows)

    csv_count = len(list(data_dir.glob("*.csv")))
    add(
        "csv_table_count",
        "pass" if csv_count == EXPECTED_CSV_TABLES else "error",
        f"observed {csv_count}; expected {EXPECTED_CSV_TABLES}",
        "Export may be incomplete or mixed with another run."
        if csv_count != EXPECTED_CSV_TABLES
        else "",
    )

    status = _read_json(data_dir / "run_status.json")
    completed = (
        status.get("state") == "completed"
        and int(status.get("completed_tvb_calls", -1)) == EXPECTED_FINAL_CALLS
    )
    add(
        "completed_run",
        "pass" if completed else "error",
        f"state={status.get('state')}; calls="
        f"{status.get('completed_tvb_calls')}/{status.get('planned_total_tvb_calls')}",
        "Partial simulations can bias every downstream comparison."
        if not completed
        else "",
    )

    spec = _read_json(data_dir / "analysis_spec.json")
    metadata = _read_json(data_dir / "experiment_metadata.json")
    spec_question = spec.get("research_question")
    metadata_question = metadata.get("research_question")
    add(
        "research_question_consistency",
        "pass" if spec_question == metadata_question else "error",
        "analysis specification and metadata agree"
        if spec_question == metadata_question
        else "analysis specification and metadata differ",
        "Interpretation may target the wrong estimand."
        if spec_question != metadata_question
        else "",
    )

    manifest = _read_csv(data_dir, "run_manifest.csv")
    manifest_ok = len(manifest) == EXPECTED_MANIFESTED_CALLS
    finite_manifest = np.isfinite(
        manifest[["wall_seconds", "max_abs_psp"]].to_numpy(dtype=float)
    ).all()
    add(
        "run_manifest",
        "pass" if manifest_ok and finite_manifest else "error",
        f"{len(manifest)} manifested calls; finite runtime/PSP="
        f"{bool(finite_manifest)}",
        "Missing or nonfinite simulations invalidate aggregation."
        if not (manifest_ok and finite_manifest)
        else "",
    )

    interactions = _read_csv(data_dir, "main_pair_interactions.csv")
    primary = interactions[interactions["pair"] == PRIMARY_PAIR].copy()
    key = ["pair", "outcome", "probe", "severity", "seed"]
    duplicates = int(primary.duplicated(key).sum())
    expected_combinations = {
        (outcome, probe) for outcome, probe, _title, _unit in OUTCOMES
    }
    observed_combinations = set(
        zip(primary["outcome"], primary["probe"], strict=False)
    )
    observed_severities = set(primary["severity"].astype(float).unique())
    observed_seeds = set(primary["seed"].astype(int).unique())
    expected_seeds = set(int(value) for value in spec.get("main_numerical_seeds", []))
    primary_ok = (
        duplicates == 0
        and observed_combinations == expected_combinations
        and observed_severities == set(EXPECTED_SEVERITIES)
        and observed_seeds == expected_seeds
    )
    add(
        "primary_scientific_grain",
        "pass" if primary_ok else "error",
        f"rows={len(primary)}; duplicate keys={duplicates}; "
        f"seeds={len(observed_seeds)}; combinations={len(observed_combinations)}",
        "The hypothesis-facing paired grid is incomplete or duplicated."
        if not primary_ok
        else "",
    )
    metric_columns = [
        "semantic_value",
        "episodic_value",
        "semantic_minus_episodic_interaction",
    ]
    nonfinite_cells = int(
        (~np.isfinite(primary[metric_columns].to_numpy(dtype=float))).sum()
    )
    endpoint = primary[primary["severity"] == 1.0]
    endpoint_finite = bool(
        np.isfinite(endpoint[metric_columns].to_numpy(dtype=float)).all()
    )
    add(
        "primary_metric_completeness",
        "warning" if endpoint_finite and nonfinite_cells else (
            "pass" if endpoint_finite else "error"
        ),
        f"nonfinite cells across all severities={nonfinite_cells}; "
        f"high-endpoint finite={endpoint_finite}",
        "Some descriptive trajectory points use fewer numerical seeds; the "
        "reported n must be retained."
        if endpoint_finite and nonfinite_cells
        else (
            "High-endpoint values needed for the primary contrast are missing."
            if not endpoint_finite
            else ""
        ),
    )

    regional = _read_csv(data_dir, "regional_features.csv")
    region_ok = (
        len(regional) == EXPECTED_REGIONS
        and not regional["region_index"].duplicated().any()
        and not regional["region_label"].duplicated().any()
    )
    add(
        "regional_features",
        "pass" if region_ok else "error",
        f"rows={len(regional)}; expected={EXPECTED_REGIONS}",
        "Atlas-aligned parcel interpretation may be invalid."
        if not region_ok
        else "",
    )

    trace_manifest = _read_csv(data_dir, "main_parcel_trace_manifest.csv")
    trace_key = ["severity", "seed", "probe"]
    trace_unique = not trace_manifest.duplicated(trace_key).any()
    trace_files = [data_dir / str(value) for value in trace_manifest["trace_file"]]
    missing_traces = [path for path in trace_files if not path.is_file()]
    size_mismatches = []
    for row in trace_manifest.itertuples(index=False):
        path = data_dir / str(row.trace_file)
        if path.is_file() and path.stat().st_size != int(row.trace_byte_size):
            size_mismatches.append(path.name)
    trace_ok = (
        len(trace_manifest) == EXPECTED_TRACE_SHARDS
        and trace_unique
        and not missing_traces
        and not size_mismatches
        and (trace_manifest["trace_region_count"].astype(int) == EXPECTED_TRACE_REGIONS).all()
    )
    add(
        "raw_trace_manifest",
        "pass" if trace_ok else "error",
        f"rows={len(trace_manifest)}; unique={trace_unique}; "
        f"missing={len(missing_traces)}; size mismatches={len(size_mismatches)}",
        "Raw-trace selection or provenance is incomplete."
        if not trace_ok
        else "",
    )

    sample_positions = sorted({0, len(trace_manifest) // 2, len(trace_manifest) - 1})
    trace_schema_ok = True
    trace_schema_evidence: list[str] = []
    required_arrays = {
        "stimulated_psp",
        "control_psp",
        "evoked_psp",
        "time_ms",
        "region_labels",
        "region_indices",
        "stimulus_waveform",
    }
    for position in sample_positions:
        row = trace_manifest.iloc[position]
        path = data_dir / str(row["trace_file"])
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays_ok = required_arrays.issubset(archive.files)
                shape = archive["evoked_psp"].shape
                identity_ok = (
                    int(archive["seed"]) == int(row["seed"])
                    and float(archive["severity"]) == float(row["severity"])
                    and str(archive["probe"]) == str(row["probe"])
                    and shape[1] == EXPECTED_TRACE_REGIONS
                )
                trace_schema_ok &= arrays_ok and identity_ok
                trace_schema_evidence.append(f"{path.name}:{shape}")
        except Exception:
            trace_schema_ok = False
            trace_schema_evidence.append(f"{path.name}:unreadable")
    add(
        "raw_trace_schema_sample",
        "pass" if trace_schema_ok else "error",
        "; ".join(trace_schema_evidence),
        "Trace arrays do not match their manifest identities."
        if not trace_schema_ok
        else "",
    )

    if verify_trace_hashes:
        bad_hashes = []
        for row in trace_manifest.itertuples(index=False):
            path = data_dir / str(row.trace_file)
            if not path.is_file() or _sha256(path) != str(row.trace_sha256):
                bad_hashes.append(str(row.trace_file))
        add(
            "raw_trace_sha256_all",
            "pass" if not bad_hashes else "error",
            f"verified {len(trace_manifest) - len(bad_hashes)}/"
            f"{len(trace_manifest)} shards",
            "One or more raw archives differ from the recorded experiment output."
            if bad_hashes
            else "",
        )

    eligibility = _read_csv(data_dir, "outcome_eligibility.csv")
    fatal = eligibility["fatal_validity_passed"].dropna()
    fatal_ok = bool(_as_bool(fatal).all()) if len(fatal) else False
    statuses = eligibility["analysis_status"].value_counts().to_dict()
    add(
        "scientific_eligibility",
        "warning" if fatal_ok and any(key != "eligible_as_prespecified" for key in statuses) else (
            "pass" if fatal_ok else "error"
        ),
        f"fatal validity passed={fatal_ok}; statuses={statuses}",
        "Some outcomes are direction-only or descriptive because quality or "
        "integration-step precision targets were not satisfied."
        if fatal_ok and any(key != "eligible_as_prespecified" for key in statuses)
        else ("Fatal validity checks failed." if not fatal_ok else ""),
    )

    saved_primary = _read_csv(data_dir, "primary_interaction_statistics.csv")
    current_status = eligibility.set_index(["outcome", "probe"])["analysis_status"]
    stale = []
    for row in saved_primary.itertuples(index=False):
        key_value = (str(row.outcome), str(row.probe))
        if key_value in current_status.index and str(row.analysis_status) != str(
            current_status.loc[key_value]
        ):
            stale.append(f"{row.outcome}/{row.probe}")
    add(
        "primary_status_consistency",
        "warning" if stale else "pass",
        "outcome_eligibility.csv supersedes stale rows: " + ", ".join(stale)
        if stale
        else "primary and final eligibility labels agree",
        "Reading primary_interaction_statistics.csv alone can overstate eligibility."
        if stale
        else "",
    )

    return pd.DataFrame(rows)


def _status_label(status: str) -> str:
    labels = {
        "eligible_as_prespecified": "eligible as prespecified",
        "direction_robust_exact_magnitude_descriptive_only": (
            "direction robust; magnitude DT-sensitive"
        ),
        "descriptive_only_quality_gate_failed": "descriptive only; quality gate failed",
    }
    return labels.get(str(status), str(status).replace("_", " "))


def build_interpretation_findings(
    data_dir: Path,
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for outcome, probe, title, unit in OUTCOMES:
        endpoint = statistics[
            (statistics["outcome"] == outcome)
            & (statistics["probe"] == probe)
            & (statistics["severity"] == 1.0)
        ].set_index("series")
        semantic = endpoint.loc["semantic_component"]
        episodic = endpoint.loc["episodic_component"]
        contrast = endpoint.loc["semantic_minus_episodic_interaction"]
        sem_mean = float(semantic["mean"])
        epi_mean = float(episodic["mean"])
        interaction = float(contrast["mean"])
        status = str(contrast["analysis_status"])
        if outcome == "response_latency" and sem_mean > 0 and epi_mean > 0:
            component_text = (
                "Both proxy response latencies lengthened; the "
                "semantic-associated delay increased more."
            )
        elif sem_mean > 0 and epi_mean > 0:
            component_text = (
                "Both proxy components increased relative to baseline; "
                + (
                    "the semantic-associated increase was larger."
                    if interaction > 0
                    else "the episodic-associated increase was larger."
                )
            )
        elif sem_mean < 0 and epi_mean < 0:
            component_text = (
                "Both proxy components decreased relative to baseline; "
                + (
                    "the semantic-associated decrease was smaller."
                    if interaction > 0
                    else "the semantic-associated decrease was larger."
                )
            )
        else:
            component_text = (
                "The proxy components changed in opposite directions: "
                f"semantic={sem_mean:.4g}, episodic={epi_mean:.4g}."
            )
        if outcome == "response_latency":
            hypothesis_text = (
                "A positive contrast means the semantic proxy became more delayed "
                "than the episodic proxy; it is not evidence of semantic timing preservation."
            )
        elif outcome == "transfer_gain":
            hypothesis_text = (
                "The differential favors the semantic proxy, but the components do "
                "not show literal semantic stability plus episodic decline."
            )
        else:
            hypothesis_text = (
                "The relative direction favors the semantic proxy, but this outcome "
                "is descriptive because its quality gate failed."
            )
        rows.append(
            {
                "category": "primary_hypothesis",
                "severity": "high" if status == "eligible_as_prespecified" else "caution",
                "outcome": outcome,
                "probe": probe,
                "analysis_status": status,
                "finding": f"{title}: {component_text}",
                "evidence": (
                    f"semantic mean={sem_mean:.6g}; episodic mean={epi_mean:.6g}; "
                    f"interaction mean={interaction:.6g} {unit}; numerical 95% CI "
                    f"[{float(contrast['ci95_lower_numerical']):.6g}, "
                    f"{float(contrast['ci95_upper_numerical']):.6g}]; "
                    f"sign-consistent={bool(contrast['sign_consistent'])}"
                ),
                "interpretation": hypothesis_text,
            }
        )

    counterfactual = _read_csv(data_dir, "counterfactual_summary.csv")
    for row in counterfactual.itertuples(index=False):
        attenuation = float(row.median_attenuation_percent)
        if attenuation < 0:
            counterfactual_finding = (
                "Local-dynamics-fixed interaction magnitude was "
                f"{-attenuation:.1f}% larger (signed attenuation "
                f"{attenuation:.1f}%)."
            )
        else:
            counterfactual_finding = (
                "Local-dynamics-fixed median attenuation was "
                f"{attenuation:.1f}%."
            )
        rows.append(
            {
                "category": "counterfactual",
                "severity": "context",
                "outcome": str(row.outcome),
                "probe": str(row.probe),
                "analysis_status": "model_internal_sensitivity",
                "finding": counterfactual_finding,
                "evidence": (
                    f"range [{float(row.minimum_attenuation_percent):.1f}%, "
                    f"{float(row.maximum_attenuation_percent):.1f}%]"
                ),
                "interpretation": (
                    "This is model-internal dependence on the perturbed local dynamics, "
                    "not biological causality."
                ),
            }
        )

    matched = _read_csv(data_dir, "matched_control_null_summary.csv")
    for (outcome, probe), group in matched.groupby(["outcome", "probe"]):
        outside = int(_as_bool(group["outside_central_90_percent"]).sum())
        rows.append(
            {
                "category": "matched_control",
                "severity": "context",
                "outcome": str(outcome),
                "probe": str(probe),
                "analysis_status": "descriptive_simulation_null",
                "finding": (
                    f"Observed interaction was outside its matched-control central "
                    f"90% range for {outside}/{len(group)} numerical seeds."
                ),
                "evidence": (
                    f"median empirical percentile="
                    f"{group['observed_empirical_percentile'].median():.1f}"
                ),
                "interpretation": (
                    "These are model-generated reference distributions, not clinical p-values."
                ),
            }
        )

    shuffles = _read_csv(data_dir, "spatial_shuffle_summary.csv")
    for row in shuffles.itertuples(index=False):
        rows.append(
            {
                "category": "spatial_shuffle",
                "severity": "context",
                "outcome": str(row.outcome),
                "probe": str(row.probe),
                "analysis_status": "descriptive_simulation_null",
                "finding": (
                    f"Observed empirical percentile among 50 spatial shuffles was "
                    f"{float(row.observed_empirical_percentile):.1f}."
                ),
                "evidence": (
                    f"shuffle central 90% [{float(row.shuffle_5th_percentile):.6g}, "
                    f"{float(row.shuffle_95th_percentile):.6g}]"
                ),
                "interpretation": (
                    "The coarse shuffle rank is descriptive and is not a clinical or "
                    "population p-value."
                ),
            }
        )

    parameters = _read_csv(data_dir, "parameter_interaction_statistics.csv")
    reversed_rows = parameters[
        (parameters["variant"] == "G100")
        & (parameters["mean_interaction"] < 0)
    ]
    for row in reversed_rows.itertuples(index=False):
        rows.append(
            {
                "category": "parameter_sensitivity",
                "severity": "caution",
                "outcome": str(row.outcome),
                "probe": str(row.probe),
                "analysis_status": "sensitivity_only",
                "finding": (
                    f"G100 reverses the interaction sign: mean="
                    f"{float(row.mean_interaction):.6g}."
                ),
                "evidence": (
                    f"numerical 95% CI [{float(row.ci95_lower_numerical):.6g}, "
                    f"{float(row.ci95_upper_numerical):.6g}]"
                ),
                "interpretation": (
                    "The direction is parameter-sensitive and should not be generalized "
                    "beyond the locked G60 main configuration."
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_hypothesis_trajectories(
    data_dir: Path,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[FigureRecord]:
    frame = _read_csv(data_dir, "main_pair_interactions.csv")
    frame = frame[frame["pair"] == PRIMARY_PAIR]
    eligibility = _eligibility_map(data_dir)
    _set_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.2))
    _figure_heading(
        fig,
        "Primary proxy-component trajectories",
        "Means and 95% t intervals across up to 20 paired numerical initializations; high endpoints are complete and tables retain n",
    )
    for axis, (outcome, probe, title, unit) in zip(axes.flat, OUTCOMES, strict=False):
        subset = frame[(frame["outcome"] == outcome) & (frame["probe"] == probe)]
        for label, column, color, marker, linestyle in (
            ("Semantic-associated proxy", "semantic_value", BLUE, "o", "-"),
            ("Episodic-associated proxy", "episodic_value", ORANGE, "s", "--"),
        ):
            means = []
            lowers = []
            uppers = []
            for severity in EXPECTED_SEVERITIES:
                summary = summarize_values(
                    subset.loc[subset["severity"] == severity, column]
                )
                means.append(float(summary["mean"]))
                lowers.append(float(summary["ci95_lower_numerical"]))
                uppers.append(float(summary["ci95_upper_numerical"]))
            means_array = np.asarray(means)
            errors = np.vstack(
                [means_array - np.asarray(lowers), np.asarray(uppers) - means_array]
            )
            axis.errorbar(
                EXPECTED_SEVERITIES,
                means_array,
                yerr=errors,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=5,
                capsize=3,
            )
        axis.axhline(0.0, color=INK, linewidth=0.8, linestyle=":")
        axis.grid(axis="y")
        axis.set_xticks(EXPECTED_SEVERITIES, ["Baseline", "Intermediate", "High"])
        axis.set_ylabel(unit)
        status = _status_label(
            str(eligibility.get((outcome, probe), {}).get("analysis_status", "unknown"))
        )
        axis.set_title(f"{title}\n{status}", loc="left")
    axes[2, 1].remove()
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", label="Semantic-associated proxy"),
        Line2D(
            [0],
            [0],
            color=ORANGE,
            marker="s",
            linestyle="--",
            label="Episodic-associated proxy",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.93), ncol=2)
    fig.text(
        0.07,
        0.012,
        "Component trajectories and their semantic-minus-episodic interaction are distinct. Numerical seeds are not participants.",
        color=MID_GREY,
    )
    fig.subplots_adjust(top=0.88, hspace=0.45, wspace=0.25, bottom=0.06)
    return _save_figure(
        fig,
        output_dir,
        "01_hypothesis_trajectories",
        "Primary proxy-component trajectories",
        formats,
        dpi,
    )


def plot_endpoint_interactions(
    data_dir: Path,
    statistics: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[FigureRecord]:
    _set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.6))
    _figure_heading(
        fig,
        "High-perturbation semantic-minus-episodic interactions",
        "Mean and 95% t interval across 20 paired numerical initializations; positive means semantic change minus episodic change",
    )
    outcome_groups = (
        ("transfer_gain", axes[0]),
        ("functional_connectivity", axes[1]),
        ("response_latency", axes[2]),
    )
    eligibility = _eligibility_map(data_dir)
    for outcome, axis in outcome_groups:
        items = [item for item in OUTCOMES if item[0] == outcome]
        y_positions = np.arange(len(items))[::-1]
        annotations = []
        for y, (_outcome, probe, _title, unit) in zip(y_positions, items, strict=True):
            row = statistics[
                (statistics["outcome"] == outcome)
                & (statistics["probe"] == probe)
                & (statistics["severity"] == 1.0)
                & (
                    statistics["series"]
                    == "semantic_minus_episodic_interaction"
                )
            ].iloc[0]
            mean = float(row["mean"])
            lower = float(row["ci95_lower_numerical"])
            upper = float(row["ci95_upper_numerical"])
            status = str(
                eligibility.get((outcome, probe), {}).get("analysis_status", "unknown")
            )
            color = BLUE if status == "eligible_as_prespecified" else GOLD
            axis.errorbar(
                mean,
                y,
                xerr=[[mean - lower], [upper - mean]],
                marker="o",
                color=color,
                capsize=4,
                linewidth=2,
            )
            annotations.append((y, mean, lower, upper))
        minimum = min(0.0, *(item[2] for item in annotations))
        maximum = max(0.0, *(item[3] for item in annotations))
        span = max(maximum - minimum, abs(maximum), 1e-9)
        axis.set_xlim(minimum - 0.04 * span, maximum + 0.34 * span)
        for y, mean, lower, upper in annotations:
            axis.text(
                upper + 0.02 * span,
                y,
                f"{mean:.3g} [{lower:.3g}, {upper:.3g}]",
                va="center",
                fontsize=8.5,
            )
        axis.axvline(0.0, color=INK, linewidth=0.9, linestyle=":")
        axis.set_ylim(-0.5, len(items) - 0.5)
        axis.set_yticks(y_positions, [item[1] for item in items])
        axis.set_xlabel(items[0][3])
        axis.set_title(items[0][2].split(" — ")[0], loc="left")
        axis.grid(axis="x")
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, marker="o", label="eligible as prespecified"),
            Line2D([0], [0], color=GOLD, marker="o", label="caution/descriptive"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.92),
        ncol=2,
    )
    fig.text(
        0.07,
        0.012,
        "Intervals quantify numerical-initialization variation only. They are not participant or population intervals.",
        color=MID_GREY,
    )
    fig.subplots_adjust(top=0.86, hspace=0.6, bottom=0.07, left=0.15, right=0.94)
    return _save_figure(
        fig,
        output_dir,
        "02_endpoint_interactions",
        "High-perturbation interactions",
        formats,
        dpi,
    )


def plot_robustness_context(
    data_dir: Path,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[FigureRecord]:
    counterfactual = _read_csv(data_dir, "counterfactual_summary.csv")
    matched = _read_csv(data_dir, "matched_control_null_summary.csv")
    shuffles = _read_csv(data_dir, "spatial_shuffle_summary.csv")
    labels = [f"{outcome.replace('_', ' ')} · {probe}" for outcome, probe, *_ in OUTCOMES]
    keys = [(outcome, probe) for outcome, probe, *_ in OUTCOMES]
    _set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.7))
    _figure_heading(
        fig,
        "Counterfactual and simulation-null context",
        "Model-internal summaries; matched lines span seeds and spatial shuffles use the prespecified seed-11 contrast",
    )

    cf = counterfactual.set_index(["outcome", "probe"])
    values = [float(cf.loc[key, "median_attenuation_percent"]) for key in keys]
    y = np.arange(len(keys))[::-1]
    colors = [BLUE if value >= 0 else ORANGE_LIGHT for value in values]
    axes[0].barh(y, values, color=colors, edgecolor=INK, linewidth=0.6)
    axes[0].axvline(0.0, color=INK, linewidth=0.9)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("median attenuation (%)")
    axes[0].set_title("Local-dynamics-fixed counterfactual", loc="left")
    axes[0].grid(axis="x")
    axes[0].text(
        0.0,
        -0.16,
        "Negative values indicate amplification, not attenuation.",
        transform=axes[0].transAxes,
        color=MID_GREY,
        fontsize=8.5,
    )

    grouped = matched.groupby(["outcome", "probe"])[
        "observed_empirical_percentile"
    ]
    medians = grouped.median()
    minima = grouped.min()
    maxima = grouped.max()
    for position, key in zip(y, keys, strict=True):
        axes[1].plot(
            [float(minima.loc[key]), float(maxima.loc[key])],
            [position, position],
            color=BLUE_LIGHT,
            linewidth=5,
            solid_capstyle="round",
        )
        axes[1].plot(float(medians.loc[key]), position, "o", color=BLUE)
    axes[1].axvspan(5, 95, color=PALE_GREY, zorder=-1)
    axes[1].axvline(50, color=INK, linestyle=":", linewidth=0.8)
    axes[1].set_xlim(0, 100)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("empirical percentile across matched controls")
    axes[1].set_title(
        "Matched-control percentiles by seed\nline = min–max; dot = median",
        loc="left",
    )
    axes[1].grid(axis="x")

    shuffle_keys = list(zip(shuffles["outcome"], shuffles["probe"], strict=False))
    shuffle_labels = [
        f"{outcome.replace('functional_connectivity', 'FC').replace('transfer_gain', 'transfer')} · {probe}"
        for outcome, probe in shuffle_keys
    ]
    sy = np.arange(len(shuffles))[::-1]
    axes[2].scatter(
        shuffles["observed_empirical_percentile"],
        sy,
        color=OLIVE,
        edgecolor=INK,
        linewidth=0.5,
        zorder=2,
    )
    axes[2].axvspan(5, 95, color=PALE_GREY, zorder=-1)
    axes[2].axvline(50, color=INK, linestyle=":", linewidth=0.8)
    axes[2].set_xlim(-2, 102)
    axes[2].set_yticks(sy, shuffle_labels)
    axes[2].set_xlabel("empirical percentile across 50 shuffles")
    axes[2].set_title("Spatial-shuffle percentiles\nseed 11 vs 50 shuffles", loc="left")
    axes[2].grid(axis="x")
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.22, right=0.97, wspace=0.62)
    return _save_figure(
        fig,
        output_dir,
        "03_robustness_context",
        "Counterfactual and simulation-null context",
        formats,
        dpi,
    )


def plot_parameter_sensitivity(
    data_dir: Path,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[FigureRecord]:
    parameter = _read_csv(data_dir, "parameter_interaction_statistics.csv")
    main = _read_csv(data_dir, "main_pair_interactions.csv")
    sensitivity_seeds = set(
        int(value)
        for value in _read_json(data_dir / "analysis_spec.json").get(
            "parameter_sensitivity_seeds", []
        )
    )
    main = main[
        (main["pair"] == PRIMARY_PAIR)
        & (main["severity"] == 1.0)
        & (main["seed"].isin(sensitivity_seeds))
    ]
    baseline_rows = []
    for (outcome, probe, unit), group in main.groupby(
        ["outcome", "probe", "unit"]
    ):
        summary = summarize_values(group["semantic_minus_episodic_interaction"])
        baseline_rows.append(
            {
                "outcome": outcome,
                "probe": probe,
                "unit": unit,
                "variant": "G60 / input 0.02",
                "mean_interaction": summary["mean"],
                "ci95_lower_numerical": summary["ci95_lower_numerical"],
                "ci95_upper_numerical": summary["ci95_upper_numerical"],
            }
        )
    combined = pd.concat([pd.DataFrame(baseline_rows), parameter], ignore_index=True)
    variant_order = ["G30", "G60 / input 0.02", "G100", "input_0.01", "input_0.04"]
    _set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 9.0))
    _figure_heading(
        fig,
        "Parameter sensitivity of high-endpoint interactions",
        "Five sensitivity seeds; dots and 95% numerical-initialization intervals; G60/input 0.02 is the matched main reference",
    )
    for axis, outcome in zip(axes, ("transfer_gain", "functional_connectivity"), strict=True):
        subset = combined[combined["outcome"] == outcome]
        offsets = {"2Hz": -0.12, "5Hz": 0.12}
        for probe, color, marker in (("2Hz", BLUE, "o"), ("5Hz", ORANGE, "s")):
            probe_rows = subset[subset["probe"] == probe].set_index("variant")
            for index, variant in enumerate(variant_order):
                if variant not in probe_rows.index:
                    continue
                row = probe_rows.loc[variant]
                mean = float(row["mean_interaction"])
                lower = float(row["ci95_lower_numerical"])
                upper = float(row["ci95_upper_numerical"])
                axis.errorbar(
                    index + offsets[probe],
                    mean,
                    yerr=[[mean - lower], [upper - mean]],
                    color=color,
                    marker=marker,
                    capsize=3,
                    linewidth=1.6,
                )
        axis.axhline(0.0, color=INK, linestyle=":", linewidth=0.9)
        axis.set_xticks(range(len(variant_order)), variant_order)
        axis.set_ylabel(
            "log2 ratio" if outcome == "transfer_gain" else "Fisher-z difference"
        )
        axis.set_title(outcome.replace("_", " ").title(), loc="left")
        axis.grid(axis="y")
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, marker="o", label="2 Hz"),
            Line2D([0], [0], color=ORANGE, marker="s", label="5 Hz"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.91),
        ncol=2,
    )
    fig.text(
        0.07,
        0.015,
        "The G100 functional-connectivity sign reversal limits generalization beyond the locked G60 configuration.",
        color=MID_GREY,
    )
    fig.subplots_adjust(top=0.84, hspace=0.48, bottom=0.09, left=0.12, right=0.96)
    return _save_figure(
        fig,
        output_dir,
        "04_parameter_sensitivity",
        "Parameter sensitivity",
        formats,
        dpi,
    )


def plot_quality_and_convergence(
    data_dir: Path,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[FigureRecord]:
    eligibility = _read_csv(data_dir, "outcome_eligibility.csv")
    dt = _read_csv(data_dir, "integration_step_outcome_eligibility.csv")
    keys = [(outcome, probe) for outcome, probe, *_ in OUTCOMES]
    labels = [f"{outcome.replace('_', ' ')} · {probe}" for outcome, probe in keys]
    indexed = eligibility.set_index(["outcome", "probe"])
    fractions = [
        float(indexed.loc[key, "valid_rows"]) / float(indexed.loc[key, "required_rows"])
        for key in keys
    ]
    statuses = [str(indexed.loc[key, "analysis_status"]) for key in keys]
    y = np.arange(len(keys))[::-1]
    colors = [BLUE if status == "eligible_as_prespecified" else GOLD for status in statuses]
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5))
    _figure_heading(
        fig,
        "Outcome eligibility and integration-step precision",
        "Quality-valid row coverage and exact-magnitude difference relative to the prespecified tolerance",
    )
    axes[0].barh(y, fractions, color=colors, edgecolor=INK, linewidth=0.5)
    axes[0].axvline(1.0, color=INK, linestyle=":", linewidth=0.9)
    axes[0].set_xlim(0, 1.04)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("valid rows / required rows")
    axes[0].set_title("Scientific quality eligibility", loc="left")
    axes[0].grid(axis="x")
    for position, fraction in zip(y, fractions, strict=True):
        axes[0].text(min(fraction + 0.015, 1.01), position, f"{fraction:.0%}", va="center")

    dt_indexed = dt.set_index(["outcome", "probe"])
    ratios = [
        float(dt_indexed.loc[key, "absolute_difference"])
        / float(dt_indexed.loc[key, "tolerance"])
        for key in keys
    ]
    precision = [bool(dt_indexed.loc[key, "precision_target_passed"]) for key in keys]
    axes[1].barh(
        y,
        ratios,
        color=[BLUE if passed else GOLD for passed in precision],
        edgecolor=INK,
        linewidth=0.5,
    )
    axes[1].axvline(1.0, color=INK, linestyle=":", linewidth=0.9, label="tolerance")
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("absolute step difference / tolerance")
    axes[1].set_title("Integration-step precision target", loc="left")
    axes[1].grid(axis="x")
    fig.text(
        0.07,
        0.015,
        "All fatal integration-step validity checks passed. Gold bars mark caution/descriptive status, not failed simulation completion.",
        color=MID_GREY,
    )
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.23, right=0.96, wspace=0.5)
    return _save_figure(
        fig,
        output_dir,
        "05_quality_and_convergence",
        "Outcome eligibility and convergence",
        formats,
        dpi,
    )


def _trace_path(data_dir: Path, severity: float, seed: int, probe: str) -> Path:
    manifest = _read_csv(data_dir, "main_parcel_trace_manifest.csv")
    row = manifest[
        np.isclose(manifest["severity"].astype(float), float(severity))
        & (manifest["seed"].astype(int) == int(seed))
        & (manifest["probe"].astype(str) == str(probe))
    ]
    if len(row) != 1:
        raise InvestigationError(
            f"Expected one trace for severity={severity}, seed={seed}, probe={probe}; "
            f"found {len(row)}."
        )
    return data_dir / str(row.iloc[0]["trace_file"])


def plot_raw_trace(
    data_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    probe: str,
    severities: Sequence[float],
    regions: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[FigureRecord], pd.DataFrame]:
    archives = []
    for severity in severities:
        path = _trace_path(data_dir, severity, seed, probe)
        with np.load(path, allow_pickle=False) as archive:
            archives.append({key: np.array(archive[key]) for key in archive.files})
    available_labels = [str(value) for value in archives[0]["region_labels"]]
    if regions:
        missing = set(regions).difference(available_labels)
        if missing:
            raise InvestigationError(
                "Unknown raw-trace region labels: " + ", ".join(sorted(missing))
            )
        if len(regions) > 8:
            raise InvestigationError("Plot at most eight explicit regions at once.")
        definitions = [
            (label, np.asarray([available_labels.index(label)], dtype=int))
            for label in regions
        ]
    else:
        definitions = [
            (
                "Bilateral A1 mean",
                np.flatnonzero(archives[0]["a1_membership"]),
            ),
            (
                "Semantic proxy mean",
                np.flatnonzero(archives[0]["semantic_expanded_membership"]),
            ),
            (
                "Episodic proxy mean",
                np.flatnonzero(archives[0]["episodic_expanded_membership"]),
            ),
        ]
    colors = [INK, BLUE, ORANGE, OLIVE, GOLD, "#945B8C", "#4C8787", "#9B7354"]
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    definition_styles = {
        label: (colors[index], line_styles[index])
        for index, (label, _indices) in enumerate(definitions)
    }
    definition_groups = (
        [[definitions[0]], definitions[1:]] if not regions else [definitions]
    )
    group_titles = (
        ["Bilateral A1", "Semantic and episodic proxy means"]
        if not regions
        else ["Selected raw-trace regions"]
    )
    _set_plot_style()
    fig, axes = plt.subplots(
        len(severities),
        len(definition_groups),
        figsize=(14.2, 3.6 * len(severities) + 1.5),
        squeeze=False,
        sharex=True,
    )
    _figure_heading(
        fig,
        f"Raw evoked PSP traces — seed {seed}, {probe}",
        (
            "Stimulated minus matched control; A1 and proxy panels use separate "
            "amplitude scales to preserve visible raw detail"
            if not regions
            else "Stimulated minus matched control; selected regions share one raw amplitude scale"
        ),
    )
    stats_rows = []
    for row_index, (severity, archive) in enumerate(
        zip(severities, archives, strict=True)
    ):
        time_ms = archive["time_ms"].astype(float)
        evoked = archive["evoked_psp"].astype(float)
        onset = float(archive["stimulus_onset_ms"])
        if probe == "pulse":
            analysis_mask = (time_ms >= onset) & (time_ms <= 6000.0)
            display_mask = (time_ms >= max(0.0, onset - 300.0)) & (time_ms <= 6000.0)
        else:
            analysis_mask = (time_ms >= 4500.0) & (time_ms <= 14500.0)
            display_mask = time_ms >= max(0.0, onset - 300.0)
        display_indices = np.flatnonzero(display_mask)
        stride = max(1, math.ceil(len(display_indices) / 3000))
        display_indices = display_indices[::stride]
        for column_index, group in enumerate(definition_groups):
            axis = axes[row_index, column_index]
            for label, region_indices in group:
                values = np.mean(evoked[:, region_indices], axis=1)
                color, linestyle = definition_styles[label]
                axis.plot(
                    time_ms[display_indices] / 1000.0,
                    values[display_indices],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.2,
                    label=f"{label} (n={len(region_indices)})",
                )
                window_values = values[analysis_mask]
                window_times = time_ms[analysis_mask]
                peak_index = int(np.argmax(np.abs(window_values)))
                stats_rows.append(
                    {
                        "seed": seed,
                        "severity": float(severity),
                        "probe": probe,
                        "trace": label,
                        "region_count": len(region_indices),
                        "analysis_start_ms": float(window_times[0]),
                        "analysis_end_ms": float(window_times[-1]),
                        "mean_psp": float(np.mean(window_values)),
                        "sd_psp": float(np.std(window_values)),
                        "rms_psp": float(np.sqrt(np.mean(window_values**2))),
                        "signed_peak_psp": float(window_values[peak_index]),
                        "absolute_peak_psp": float(abs(window_values[peak_index])),
                        "peak_time_ms": float(window_times[peak_index]),
                    }
                )
            axis.axvline(
                onset / 1000.0,
                color=MID_GREY,
                linestyle=":",
                linewidth=0.9,
            )
            axis.set_ylabel("evoked PSP")
            axis.set_title(
                f"Severity {severity:.1f} · {group_titles[column_index]}",
                loc="left",
            )
            axis.grid(axis="y")
            if row_index == 0:
                axis.legend(loc="upper right", ncol=1)
    for column_index in range(len(definition_groups)):
        axes[-1, column_index].set_xlabel("simulation time (s)")
    fig.text(
        0.07,
        0.012,
        "Raw PSP amplitudes are model variables, not measured voltage. Zero tract delays limit anatomical timing interpretation.",
        color=MID_GREY,
    )
    fig.subplots_adjust(
        top=0.82,
        hspace=0.4,
        wspace=0.25,
        bottom=0.1,
        left=0.08,
        right=0.97,
    )
    stem = f"raw_trace_seed_{seed:04d}_{_safe_token(probe)}"
    records = _save_figure(
        fig,
        output_dir,
        stem,
        f"Raw trace seed {seed} {probe}",
        formats,
        dpi,
    )
    return records, pd.DataFrame(stats_rows)


def create_overview(
    data_dir: Path,
    output_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
    verify_trace_hashes: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_export(
        data_dir,
        verify_trace_hashes=verify_trace_hashes,
    )
    validation.to_csv(output_dir / "data_quality_report.csv", index=False)
    if (validation["status"] == "error").any():
        failed = validation.loc[validation["status"] == "error", "check"].tolist()
        raise InvestigationError(
            "Export validation failed before plotting: " + ", ".join(failed)
        )
    statistics = build_hypothesis_statistics(data_dir)
    statistics.to_csv(output_dir / "hypothesis_statistics.csv", index=False)
    findings = build_interpretation_findings(data_dir, statistics)
    findings.to_csv(output_dir / "interpretation_findings.csv", index=False)
    figure_records: list[FigureRecord] = []
    figure_records.extend(
        plot_hypothesis_trajectories(data_dir, output_dir, formats, dpi)
    )
    figure_records.extend(
        plot_endpoint_interactions(data_dir, statistics, output_dir, formats, dpi)
    )
    figure_records.extend(
        plot_robustness_context(data_dir, output_dir, formats, dpi)
    )
    figure_records.extend(
        plot_parameter_sensitivity(data_dir, output_dir, formats, dpi)
    )
    figure_records.extend(
        plot_quality_and_convergence(data_dir, output_dir, formats, dpi)
    )
    trace_records, trace_statistics = plot_raw_trace(
        data_dir,
        output_dir,
        seed=11,
        probe="2Hz",
        severities=(0.0, 1.0),
        regions=(),
        formats=formats,
        dpi=dpi,
    )
    figure_records.extend(trace_records)
    trace_statistics.to_csv(output_dir / "raw_trace_descriptive_statistics.csv", index=False)
    figure_manifest = pd.DataFrame([record.__dict__ for record in figure_records])
    figure_manifest.to_csv(output_dir / "figure_manifest.csv", index=False)
    return validation, statistics, findings


def _print_overview(
    data_dir: Path,
    output_dir: Path,
    validation: pd.DataFrame,
    statistics: pd.DataFrame,
    findings: pd.DataFrame,
) -> None:
    spec = _read_json(data_dir / "analysis_spec.json")
    print("\nResearch question")
    print(spec.get("research_question", "unknown"))
    print("\nPrimary high-endpoint statistics")
    endpoint = statistics[
        (statistics["severity"] == 1.0)
        & (statistics["series"] == "semantic_minus_episodic_interaction")
    ]
    for row in endpoint.itertuples(index=False):
        print(
            f"- {row.outcome}/{row.probe}: mean {row.mean:.6g} {row.unit}, "
            f"95% numerical CI [{row.ci95_lower_numerical:.6g}, "
            f"{row.ci95_upper_numerical:.6g}], "
            f"{_status_label(row.analysis_status)}"
        )
    print("\nInterpretation anchors")
    for row in findings[findings["category"] == "primary_hypothesis"].itertuples(
        index=False
    ):
        print(f"- {row.finding} {row.interpretation}")
    warnings = validation[validation["status"] == "warning"]
    print("\nTechnical cautions")
    for row in warnings.itertuples(index=False):
        print(f"- {row.check}: {row.evidence} {row.analytical_risk}")
    print(
        "- Numerical initializations are not participants; p-values and intervals "
        "describe numerical-initialization variation only."
    )
    print(f"\nInvestigation outputs: {output_dir.resolve()}")


def _print_stats(frame: pd.DataFrame) -> None:
    columns = [
        "outcome",
        "probe",
        "severity",
        "series",
        "n_numerical_initializations",
        "mean",
        "median",
        "minimum",
        "maximum",
        "ci95_lower_numerical",
        "ci95_upper_numerical",
        "positive_fraction",
        "analysis_status",
    ]
    with pd.option_context("display.max_rows", 200, "display.width", 180):
        print(frame[columns].to_string(index=False))
    print(
        "\nCaution: numerical seeds are not subjects or biological replicates; "
        "the reported t statistics, p-values, and intervals are numerical diagnostics."
    )


def _table_inventory(data_dir: Path, contains: str | None) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("*.csv")):
        if contains and contains.lower() not in path.name.lower():
            continue
        frame = pd.read_csv(path)
        rows.append(
            {
                "file": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "duplicate_rows": int(frame.duplicated().sum()),
                "null_cells": int(frame.isna().sum().sum()),
                "schema": ", ".join(frame.columns),
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Investigate the completed TVB379 semantic-versus-episodic result "
            "export without rerunning TVB."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"experiment export directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="derived-output directory (default: <data-dir>/investigation)",
    )
    subparsers = parser.add_subparsers(dest="command")

    overview = subparsers.add_parser(
        "overview",
        help="generate the complete visual interpretation and statistics pack",
    )
    overview.add_argument("--formats", nargs="+", choices=("png", "svg"), default=("png", "svg"))
    overview.add_argument("--dpi", type=int, default=300)
    overview.add_argument("--verify-trace-hashes", action="store_true")

    validate = subparsers.add_parser(
        "validate",
        help="check export completeness, keys, eligibility, and raw traces",
    )
    validate.add_argument("--verify-trace-hashes", action="store_true")

    stats_parser = subparsers.add_parser(
        "stats",
        help="calculate inspectable component and interaction statistics",
    )
    stats_parser.add_argument("--pair", default=PRIMARY_PAIR)
    stats_parser.add_argument(
        "--outcome",
        choices=("transfer_gain", "functional_connectivity", "response_latency"),
    )
    stats_parser.add_argument("--probe", choices=EXPECTED_PROBES)
    stats_parser.add_argument("--severity", type=float, choices=EXPECTED_SEVERITIES)
    stats_parser.add_argument("--save", action="store_true")

    trace = subparsers.add_parser(
        "trace",
        help="plot selected raw PSP trace shards and save descriptive statistics",
    )
    trace.add_argument("--seed", type=int, default=11)
    trace.add_argument("--probe", choices=EXPECTED_PROBES, default="2Hz")
    trace.add_argument(
        "--severities",
        type=float,
        nargs="+",
        choices=EXPECTED_SEVERITIES,
        default=(0.0, 1.0),
    )
    trace.add_argument(
        "--region",
        action="append",
        default=[],
        help="raw-trace region label; repeat up to eight times",
    )
    trace.add_argument("--formats", nargs="+", choices=("png", "svg"), default=("png", "svg"))
    trace.add_argument("--dpi", type=int, default=300)

    tables = subparsers.add_parser(
        "tables",
        help="list CSV sizes, schemas, missingness, and exact duplicates",
    )
    tables.add_argument("--contains", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_dir / DEFAULT_OUTPUT_SUBDIR
    )
    command = args.command or "overview"
    try:
        if command == "overview":
            formats = tuple(getattr(args, "formats", ("png", "svg")))
            dpi = int(getattr(args, "dpi", 300))
            validation, statistics, findings = create_overview(
                data_dir,
                output_dir,
                formats=formats,
                dpi=dpi,
                verify_trace_hashes=bool(
                    getattr(args, "verify_trace_hashes", False)
                ),
            )
            _print_overview(
                data_dir,
                output_dir,
                validation,
                statistics,
                findings,
            )
        elif command == "validate":
            report = validate_export(
                data_dir,
                verify_trace_hashes=args.verify_trace_hashes,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            report.to_csv(output_dir / "data_quality_report.csv", index=False)
            print(report.to_string(index=False))
            if (report["status"] == "error").any():
                return 1
        elif command == "stats":
            frame = build_hypothesis_statistics(data_dir, pair=args.pair)
            if args.outcome:
                frame = frame[frame["outcome"] == args.outcome]
            if args.probe:
                frame = frame[frame["probe"] == args.probe]
            if args.severity is not None:
                frame = frame[np.isclose(frame["severity"], args.severity)]
            if frame.empty:
                raise InvestigationError("No statistics match the requested filters.")
            _print_stats(frame)
            if args.save:
                output_dir.mkdir(parents=True, exist_ok=True)
                token = "_".join(
                    _safe_token(value)
                    for value in (args.pair, args.outcome or "all", args.probe or "all")
                )
                path = output_dir / f"stats_{token}.csv"
                frame.to_csv(path, index=False)
                print(f"\nSaved: {path}")
        elif command == "trace":
            records, trace_stats = plot_raw_trace(
                data_dir,
                output_dir,
                seed=args.seed,
                probe=args.probe,
                severities=tuple(args.severities),
                regions=tuple(args.region),
                formats=tuple(args.formats),
                dpi=args.dpi,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            token = f"seed_{args.seed:04d}_{_safe_token(args.probe)}"
            path = output_dir / f"raw_trace_stats_{token}.csv"
            trace_stats.to_csv(path, index=False)
            print(trace_stats.to_string(index=False))
            print("\nFigures:")
            for record in records:
                print(f"- {record.path}")
            print(f"Stats: {path}")
        elif command == "tables":
            inventory = _table_inventory(data_dir, args.contains)
            if inventory.empty:
                raise InvestigationError("No CSV tables match the filter.")
            with pd.option_context("display.max_colwidth", 120, "display.width", 220):
                print(inventory.to_string(index=False))
        else:
            raise InvestigationError(f"Unknown command: {command}")
    except InvestigationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
