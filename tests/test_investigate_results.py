from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis import investigate_results as investigation


DATA_DIR = Path(__file__).resolve().parents[1] / "data_analysis"


def test_descriptive_summary_uses_numerical_initializations() -> None:
    summary = investigation.summarize_values([1.0, 2.0, 3.0])

    assert summary["n_numerical_initializations"] == 3
    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    assert summary["sd"] == 1.0
    assert summary["minimum"] == 1.0
    assert summary["maximum"] == 3.0
    assert summary["positive_fraction"] == 1.0
    assert summary["sign_consistent"] is True
    assert math.isfinite(float(summary["p_value_vs_zero_numerical"]))


def test_export_validation_has_no_fatal_errors() -> None:
    report = investigation.validate_export(DATA_DIR)

    assert not (report["status"] == "error").any()
    evidence = report.set_index("check")
    assert evidence.loc["completed_run", "status"] == "pass"
    assert "762/762" in evidence.loc["completed_run", "evidence"]
    assert evidence.loc["raw_trace_manifest", "status"] == "pass"
    assert evidence.loc["primary_metric_completeness", "status"] == "warning"
    assert evidence.loc["primary_status_consistency", "status"] == "warning"


def test_primary_endpoint_statistics_use_final_authoritative_eligibility() -> None:
    frame = investigation.build_hypothesis_statistics(DATA_DIR)
    endpoint = frame[
        (frame["outcome"] == "transfer_gain")
        & (frame["probe"] == "2Hz")
        & np.isclose(frame["severity"], 1.0)
    ].set_index("series")

    semantic = endpoint.loc["semantic_component"]
    episodic = endpoint.loc["episodic_component"]
    interaction = endpoint.loc["semantic_minus_episodic_interaction"]
    assert semantic["n_numerical_initializations"] == 20
    assert semantic["mean"] > episodic["mean"] > 0.0
    assert np.isclose(interaction["mean"], 2.233173, atol=1e-6)
    assert interaction["positive_fraction"] == 1.0
    assert interaction["analysis_status"] == (
        "direction_robust_exact_magnitude_descriptive_only"
    )


def test_interpretation_separates_components_from_the_interaction() -> None:
    statistics = investigation.build_hypothesis_statistics(DATA_DIR)
    findings = investigation.build_interpretation_findings(
        DATA_DIR,
        statistics,
    )
    transfer = findings[
        (findings["category"] == "primary_hypothesis")
        & (findings["outcome"] == "transfer_gain")
        & (findings["probe"] == "2Hz")
    ].iloc[0]
    latency = findings[
        (findings["category"] == "primary_hypothesis")
        & (findings["outcome"] == "response_latency")
    ].iloc[0]

    assert "Both proxy components increased" in transfer["finding"]
    assert "do not show literal semantic stability" in transfer["interpretation"]
    assert "response latencies lengthened" in latency["finding"]
    assert "not evidence of semantic timing preservation" in latency["interpretation"]

    fc_counterfactual = findings[
        (findings["category"] == "counterfactual")
        & (findings["outcome"] == "functional_connectivity")
        & (findings["probe"] == "2Hz")
    ].iloc[0]
    assert "larger" in fc_counterfactual["finding"]
    assert "signed attenuation -10.8%" in fc_counterfactual["finding"]


def test_figure_manifest_covers_all_overview_figures_and_formats() -> None:
    manifest_path = DATA_DIR / "investigation" / "figure_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    assert len(manifest) == 12
    assert set(manifest["format"]) == {"png", "svg"}
    assert manifest["figure"].nunique() == 6
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert all(Path(path).is_file() for path in manifest["path"])
