from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_analysis import build_html_investigator as dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_analysis"
ARTIFACT_PATH = DATA_DIR / "html_investigator" / "artifact.json"
HTML_PATH = DATA_DIR / "TVB379_visual_investigator.html"


def _artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_pulse_fc_is_not_an_experiment_estimand() -> None:
    node = pd.read_csv(
        DATA_DIR / "main_node_metrics.csv",
        usecols=["probe", "evoked_fc_z", "fc_valid"],
    )
    pulse = node[node["probe"] == "pulse"]

    assert len(pulse) > 0
    assert pulse["evoked_fc_z"].isna().all()
    assert not dashboard._bool_series(pulse["fc_valid"]).any()

    artifact = _artifact()
    pulse_text = next(
        block["body"]
        for block in artifact["manifest"]["blocks"]
        if block["id"] == "pulse_fc"
    )
    assert "not defined" in pulse_text
    assert "not missing data and not a failed FC gate" in pulse_text


def test_every_non_npz_experiment_source_is_cataloged() -> None:
    artifact = _artifact()
    catalog = pd.DataFrame(artifact["snapshot"]["datasets"]["source_catalog"])

    assert (catalog["kind"] == "CSV").sum() == 48
    assert (catalog["kind"] == "JSON").sum() == 3
    assert (catalog["kind"] == "LOG").sum() == 1
    assert catalog["kind"].isin(["PNG", "SVG"]).sum() == 4
    assert len(catalog) == 56
    assert catalog["reviewed"].all()
    assert not catalog["file"].str.endswith(".npz").any()
    assert catalog["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_dashboard_snapshot_has_expected_research_surfaces() -> None:
    artifact = _artifact()
    datasets = artifact["snapshot"]["datasets"]
    manifest = artifact["manifest"]

    assert artifact["surface"] == "dashboard"
    assert artifact["snapshot"]["status"] == "ready"
    assert len(manifest["charts"]) == 21
    assert len(manifest["tables"]) == 13
    assert len(manifest["cards"]) == 5
    assert len(datasets["regional_features"]) == 379
    assert len(datasets["pulse_summary"]) == 6
    assert len(datasets["eligibility"]) == 7
    assert len(datasets["source_samples"]) >= 48

    summary = datasets["summary"][0]
    assert summary["completed_calls"] == 762
    assert summary["trace_shards_manifested"] == 180
    assert summary["trace_shards_analyzed"] == 180
    assert summary["pulse_fc_defined"] == 0

    assert len(datasets["transmission_endpoint_audit"]) == 4
    assert len(datasets["phase_fc_rows"]) == 453
    assert sum(
        len(datasets[name])
        for name in ("spectra_stimulated", "spectra_control", "spectra_evoked")
    ) == 4176
    assert len(datasets["pulse_fixed_masks"]) == 40
    assert len(datasets["regional_covariate_coefficients"]) == 10


def test_html_is_self_contained_and_carries_accessible_fallback() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert HTML_PATH.stat().st_size > 1_000_000
    assert 'data-data-analytics-portable-artifact="true"' in html
    assert 'id="data-analytics-portable-artifact-payload-source"' in html
    assert 'data-portable-fallback="true"' in html
    assert 'data-tvb379-portable-overflow-fix="true"' in html
    assert "connect-src &#39;none&#39;" in html
    assert "<script src=" not in html
    assert "<link rel=\"stylesheet\"" not in html
