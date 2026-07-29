from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze_results
from rise_tvb379.result_analysis import (
    AnalysisProducts,
    PERIODIC_PROBES,
    PRIMARY_CONTRAST,
    SECONDARY_CONTRAST,
    ResultBundle,
    build_contrast_summaries,
    build_counterfactual_attenuation,
    build_interpretation_findings,
    build_matched_null_context,
    build_sensitivity_summary,
    build_spatial_shuffle_context,
    build_analysis_products,
    load_result_bundle,
    validate_result_bundle,
)
from rise_tvb379.result_plots import figure_specifications


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_RUN = PROJECT_ROOT / "runs" / "RISE_TVB379_results_final"


def _main_contrasts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed_offset, seed in enumerate((11, 23)):
        for probe in PERIODIC_PROBES:
            for severity, condition in (
                (0.0, "baseline"),
                (0.5, "intermediate"),
                (1.0, "high"),
            ):
                scale = severity * (1.0 + 0.05 * seed_offset)
                music = scale
                speech = 2.0 * scale
                semantic = 3.0 * scale
                episodic = scale
                rows.append(
                    {
                        "condition": condition,
                        "severity": severity,
                        "seed": seed,
                        "probe": probe,
                        "music": music,
                        "speech": speech,
                        "music_semantic_task_associated": semantic,
                        "music_episodic_task_associated": episodic,
                        PRIMARY_CONTRAST: music - speech,
                        SECONDARY_CONTRAST: semantic - episodic,
                    }
                )
    return pd.DataFrame(rows)


def _local_contrasts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (11, 23):
        for probe in PERIODIC_PROBES:
            rows.extend(
                (
                    {
                        "variant": "primary_local_fixed_endpoint",
                        "severity": 1.0,
                        "seed": seed,
                        "probe": probe,
                        PRIMARY_CONTRAST: -0.02,
                        SECONDARY_CONTRAST: 0.0,
                    },
                    {
                        "variant": "memory_local_fixed_endpoint",
                        "severity": 1.0,
                        "seed": seed,
                        "probe": probe,
                        PRIMARY_CONTRAST: 0.0,
                        SECONDARY_CONTRAST: 0.04,
                    },
                )
            )
    return pd.DataFrame(rows)


def _matched_nulls() -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    for seed in (11, 23):
        for probe in PERIODIC_PROBES:
            for set_id, value in enumerate((-3.0, -2.0, -1.0, 0.0, 1.0)):
                primary_rows.append(
                    {
                        "set_id": set_id,
                        "seed": seed,
                        "probe": probe,
                        "null_music_minus_speech": value,
                        "mean_standardized_match_distance": 0.2,
                    }
                )
            for set_id, value in enumerate((0.0, 1.0, 2.0, 3.0, 4.0)):
                secondary_rows.append(
                    {
                        "set_id": set_id,
                        "seed": seed,
                        "probe": probe,
                        "null_semantic_minus_episodic": value,
                        "mean_standardized_match_distance": 0.3,
                    }
                )
    return pd.DataFrame(primary_rows), pd.DataFrame(secondary_rows)


def _shuffle() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for probe in PERIODIC_PROBES:
        for index, (primary, secondary) in enumerate(
            zip(
                (-3.0, -2.0, -1.0, 0.0, 1.0),
                (0.0, 1.0, 2.0, 3.0, 4.0),
                strict=True,
            )
        ):
            rows.append(
                {
                    "variant": f"shuffle_{index:03d}",
                    "severity": 1.0,
                    "seed": 11,
                    "probe": probe,
                    PRIMARY_CONTRAST: primary,
                    SECONDARY_CONTRAST: secondary,
                }
            )
    return pd.DataFrame(rows)


def _sensitivity() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for probe in PERIODIC_PROBES:
        rows.extend(
            (
                {
                    "variant": "G60_input_0.02",
                    "severity": 1.0,
                    "seed": 11,
                    "probe": probe,
                    PRIMARY_CONTRAST: -1.0,
                    SECONDARY_CONTRAST: 2.0,
                },
                {
                    "variant": "G100",
                    "severity": 1.0,
                    "seed": 11,
                    "probe": probe,
                    PRIMARY_CONTRAST: 0.5,
                    SECONDARY_CONTRAST: 1.5,
                },
            )
        )
    return pd.DataFrame(rows)


def test_component_and_contrast_summaries_keep_interpretations_separate() -> None:
    trajectory, endpoint = build_contrast_summaries(_main_contrasts())

    primary = endpoint[
        (endpoint["contrast_family"] == "primary")
        & (endpoint["probe"] == "2Hz")
    ].set_index("series")
    assert primary.loc["Music proxy", "median"] > 0
    assert primary.loc["Speech proxy", "median"] > 0
    assert primary.loc["Music minus speech", "median"] < 0
    assert primary.loc["Music minus speech", "sign_consistent"]
    assert len(trajectory) == 36


def test_attenuation_percent_uses_absolute_contrast_magnitude() -> None:
    result = build_counterfactual_attenuation(
        _main_contrasts(),
        _local_contrasts(),
    )
    primary = result[result["contrast_family"] == "primary"]
    secondary = result[result["contrast_family"] == "secondary"]
    assert primary["attenuation_percent"].min() >= 98.0
    assert secondary["attenuation_percent"].min() >= 98.0


def test_matched_and_shuffle_percentiles_and_interval_flags() -> None:
    primary_null, secondary_null = _matched_nulls()
    matched = build_matched_null_context(
        _main_contrasts(),
        primary_null,
        secondary_null,
    )
    shuffled = build_spatial_shuffle_context(
        _main_contrasts(),
        _shuffle(),
        main_seed=11,
    )

    assert matched["inside_central_90_percent"].all()
    assert shuffled["inside_central_90_percent"].all()
    np.testing.assert_allclose(
        shuffled["observed_empirical_percentile"],
        60.0,
    )


def test_sensitivity_flags_primary_sign_reversal_only() -> None:
    summary = build_sensitivity_summary(_sensitivity())
    g100 = summary[summary["variant"] == "G100"].set_index(
        ["contrast_family", "probe"]
    )
    assert not g100.loc[("primary", "2Hz"), "same_sign_as_reference"]
    assert g100.loc[("secondary", "2Hz"), "same_sign_as_reference"]


def test_interpretation_wording_respects_model_boundaries() -> None:
    trajectory, endpoint = build_contrast_summaries(_main_contrasts())
    del trajectory
    primary_null, secondary_null = _matched_nulls()
    products = {
        "high_endpoint_summary": endpoint,
        "counterfactual_attenuation": build_counterfactual_attenuation(
            _main_contrasts(),
            _local_contrasts(),
        ),
        "matched_null_context": build_matched_null_context(
            _main_contrasts(),
            primary_null,
            secondary_null,
        ),
        "spatial_shuffle_context": build_spatial_shuffle_context(
            _main_contrasts(),
            _shuffle(),
            main_seed=11,
        ),
        "sensitivity_summary": build_sensitivity_summary(_sensitivity()),
        "convergence_signal_quality": pd.DataFrame(
            {
                "required_for_inference": [True],
                "max_relative_difference": [0.03],
                "low_fit_r_squared_warning": [True],
                "fit_r_squared_median": [0.02],
            }
        ),
        "runtime_diagnostics": pd.DataFrame(
            {
                "record_type": ["worker"],
                "aggregate_wall_seconds": [120.0],
            }
        ),
    }
    bundle = ResultBundle(
        run_dir=Path("/run"),
        results_dir=Path("/run/results"),
        metadata={"workload": {"manifest": 430, "total": 442}},
        tables={},
    )

    findings = build_interpretation_findings(bundle, products)
    all_text = " ".join(
        findings[["summary", "evidence", "caveat"]]
        .fillna("")
        .astype(str)
        .to_numpy()
        .ravel()
    ).lower()
    primary = findings[
        findings["finding_id"] == "primary_endpoint_2Hz"
    ].iloc[0]
    assert "both proxy transfers increased" in primary["summary"].lower()
    assert "does not establish biological causality" in all_text
    assert "not clinical p-values" in all_text
    assert "not subjects" not in all_text
    assert "prespecified experiment gate" in all_text


@pytest.mark.parametrize(
    ("audience", "expected"),
    (("publication", 8), ("technical", 6), ("both", 14)),
)
def test_audience_filtering(audience: str, expected: int) -> None:
    specifications = figure_specifications(audience)
    assert len(specifications) == expected
    assert len({item["id"] for item in specifications}) == expected


def test_finding_audience_filtering() -> None:
    findings = pd.DataFrame(
        {
            "audience": ["publication", "technical"],
            "summary": ["scientific", "caution"],
        }
    )
    products = AnalysisProducts(tables={}, findings=findings)
    publication = analyze_results._selected_findings(
        products,
        "publication",
    )
    technical = analyze_results._selected_findings(products, "technical")
    both = analyze_results._selected_findings(products, "both")

    assert publication["summary"].tolist() == ["scientific"]
    assert technical["summary"].tolist() == ["caution"]
    assert both["summary"].tolist() == ["scientific", "caution"]


def test_cli_defaults_and_validation() -> None:
    arguments = analyze_results.build_parser().parse_args([])
    assert arguments.run_dir == analyze_results.DEFAULT_RUN_DIR
    assert arguments.output_dir is None
    assert arguments.audience == "both"
    assert tuple(arguments.formats) == ("png", "svg")
    assert arguments.dpi == 300

    with pytest.raises(SystemExit):
        analyze_results.build_parser().parse_args(["--dpi", "0"])
    with pytest.raises(SystemExit):
        analyze_results.build_parser().parse_args(
            ["--audience", "clinical"]
        )


@pytest.mark.skipif(
    not FINAL_RUN.is_dir(),
    reason="the user-supplied completed final run is not available",
)
def test_supplied_final_run_acceptance(tmp_path: Path) -> None:
    bundle = load_result_bundle(FINAL_RUN)
    validation = validate_result_bundle(bundle)
    products = build_analysis_products(bundle, validation)

    convergence = products.tables["convergence_signal_quality"]
    inferential = convergence[convergence["required_for_inference"]]
    assert inferential["max_relative_difference"].max() == pytest.approx(
        0.0322627,
        abs=1e-6,
    )

    endpoint = products.tables["high_endpoint_summary"]
    primary = endpoint[
        (endpoint["contrast_family"] == "primary")
        & (endpoint["metric_type"] == "contrast")
    ]
    secondary = endpoint[
        (endpoint["contrast_family"] == "secondary")
        & (endpoint["metric_type"] == "contrast")
    ]
    assert (primary["sign_direction"] == "negative").all()
    assert (secondary["sign_direction"] == "positive").all()

    attenuation = products.tables["counterfactual_attenuation"]
    assert (
        attenuation.groupby(["contrast_family", "probe"])[
            "attenuation_percent"
        ].median()
        > 96.0
    ).all()
    assert products.tables["matched_null_context"][
        "inside_central_90_percent"
    ].all()
    assert products.tables["spatial_shuffle_context"][
        "inside_central_90_percent"
    ].all()

    sensitivity = products.tables["sensitivity_summary"]
    primary_g100 = sensitivity[
        (sensitivity["contrast_family"] == "primary")
        & (sensitivity["variant"] == "G100")
    ]
    assert (~primary_g100["same_sign_as_reference"]).all()
    assert convergence["low_fit_r_squared_warning"].any()

    output = tmp_path / "analysis"
    analyze_results.run_analysis(
        run_dir=FINAL_RUN,
        output_dir=output,
        audience="both",
        formats=("png",),
        dpi=90,
    )
    manifest = pd.read_csv(output / "figure_manifest.csv")
    assert len(manifest) == 14
    assert len(list((output / "publication" / "figures").glob("*.png"))) == 8
    assert len(list((output / "technical_qa" / "figures").glob("*.png"))) == 6
    assert len(list((output / "tables").glob("*.csv"))) == 11
    assert all(path.stat().st_size > 0 for path in output.rglob("*") if path.is_file())
