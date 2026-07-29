#!/usr/bin/env python3
"""Create publication and technical-QA analyses from a completed TVB379 run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Mapping

import pandas as pd

from rise_tvb379.result_analysis import (
    AnalysisProducts,
    ResultValidationError,
    build_analysis_products,
    load_result_bundle,
    validate_result_bundle,
)
from rise_tvb379.result_plots import create_analysis_figures


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "RISE_TVB379_results_final"

TABLE_FILENAMES: Mapping[str, str] = {
    "validation_summary": "validation_summary.csv",
    "contrast_trajectory_summary": "contrast_trajectory_summary.csv",
    "high_endpoint_summary": "high_endpoint_summary.csv",
    "counterfactual_attenuation": "counterfactual_attenuation.csv",
    "matched_null_context": "matched_null_context.csv",
    "spatial_shuffle_context": "spatial_shuffle_context.csv",
    "sensitivity_summary": "sensitivity_summary.csv",
    "convergence_signal_quality": "convergence_signal_quality.csv",
    "network_high_endpoint_summary": "network_high_endpoint_summary.csv",
    "calibration_summary": "calibration_summary.csv",
    "runtime_diagnostics": "runtime_diagnostics.csv",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze an existing completed RISE TVB379 run. "
            "No TVB simulations are rerun."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"completed run directory (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="analysis output directory (default: <run-dir>/analysis)",
    )
    parser.add_argument(
        "--audience",
        choices=("publication", "technical", "both"),
        default="both",
        help="figure and finding audience (default: both)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg"),
        default=("png", "svg"),
        help="one or both image formats (default: png svg)",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=300,
        help="PNG resolution (default: 300)",
    )
    return parser


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _selected_findings(
    products: AnalysisProducts,
    audience: str,
) -> pd.DataFrame:
    findings = products.findings.copy()
    if audience == "publication":
        findings = findings[findings["audience"] == "publication"]
    elif audience == "technical":
        findings = findings[findings["audience"] == "technical"]
    return findings.reset_index(drop=True)


def _write_tables(
    products: AnalysisProducts,
    validation: pd.DataFrame,
    output_dir: Path,
    audience: str,
) -> pd.DataFrame:
    tables_dir = output_dir / "tables"
    for key, filename in TABLE_FILENAMES.items():
        frame = validation if key == "validation_summary" else products.tables[key]
        _atomic_csv(frame, tables_dir / filename)
    findings = _selected_findings(products, audience)
    _atomic_csv(findings, output_dir / "interpretation_findings.csv")
    return findings


def _print_findings(findings: pd.DataFrame) -> None:
    sections = (
        ("Scientific findings", "publication"),
        ("Technical cautions", "technical"),
    )
    for heading, audience in sections:
        selected = findings[findings["audience"] == audience]
        if selected.empty:
            continue
        print(f"\n{heading}")
        print("-" * len(heading))
        for row in selected.itertuples(index=False):
            print(f"* {row.summary}")
            print(f"  Evidence: {row.evidence}")
            if row.caveat:
                print(f"  Caution: {row.caveat}")


def run_analysis(
    *,
    run_dir: Path,
    output_dir: Path | None,
    audience: str,
    formats: Iterable[str],
    dpi: int,
) -> Path:
    """Validate first, then atomically write all selected analysis outputs."""

    if audience not in {"publication", "technical", "both"}:
        raise ValueError("audience must be publication, technical, or both")
    image_formats = tuple(dict.fromkeys(formats))
    if not image_formats:
        raise ValueError("at least one image format is required")
    invalid_formats = sorted(set(image_formats) - {"png", "svg"})
    if invalid_formats:
        raise ValueError(
            "unsupported image formats: " + ", ".join(invalid_formats)
        )
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    bundle = load_result_bundle(run_dir)
    validation = validate_result_bundle(bundle)
    products = build_analysis_products(bundle, validation)

    resolved_output = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else bundle.run_dir / "analysis"
    )

    # All scientific validation and derived-table construction happens before
    # this first output write.
    findings = _write_tables(
        products,
        validation,
        resolved_output,
        audience,
    )
    figure_paths, figure_manifest = create_analysis_figures(
        bundle=bundle,
        products=products,
        output_dir=resolved_output,
        audience=audience,
        formats=image_formats,
        dpi=dpi,
    )
    _atomic_csv(figure_manifest, resolved_output / "figure_manifest.csv")

    _print_findings(findings)
    print("\nAnalysis complete")
    print("-----------------")
    print(f"Validated run: {bundle.run_dir}")
    print(f"Output: {resolved_output}")
    print(f"Derived tables: {len(TABLE_FILENAMES)}")
    print(
        f"Figures: {figure_manifest['figure_id'].nunique()} "
        f"({len(figure_paths)} rendered files)"
    )
    return resolved_output


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        run_analysis(
            run_dir=arguments.run_dir,
            output_dir=arguments.output_dir,
            audience=arguments.audience,
            formats=arguments.formats,
            dpi=arguments.dpi,
        )
    except (FileNotFoundError, ResultValidationError, ValueError) as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
