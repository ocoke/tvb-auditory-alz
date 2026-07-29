"""Headless versions of the ten figures from the source notebook."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import MAIN_GLOBAL_COUPLING


NETWORK_COLORS = {"music": "#2B6CB0", "speech": "#D97706"}
NETWORK_NAMES = {"music": "Music proxy", "speech": "Speech proxy"}

FIGURE_FILENAMES = (
    "01_baseline_coupling_calibration.png",
    "02_main_stage_curves.png",
    "03_primary_endpoint_contrast.png",
    "04_local_dynamics_counterfactual.png",
    "05_matched_control_null.png",
    "06_parameter_sensitivity.png",
    "07_spatial_placement_sensitivity.png",
    "08_semantic_episodic_secondary_analysis.png",
    "09_semantic_episodic_matched_null.png",
    "10_semantic_episodic_robustness.png",
)


def _figure_path(figure_dir: Path, filename: str) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir / filename


def plot_baseline_coupling_calibration(
    calibration_df: pd.DataFrame,
    figure_dir: Path,
    *,
    main_global_coupling: float = MAIN_GLOBAL_COUPLING,
) -> Path:
    """Plot the notebook's baseline-only global-coupling scan."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[0])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    try:
        ax.plot(
            calibration_df["global_coupling"],
            calibration_df["music_transfer"],
            marker="o",
            label="Music proxy",
        )
        ax.plot(
            calibration_df["global_coupling"],
            calibration_df["speech_transfer"],
            marker="o",
            label="Speech proxy",
        )
        ax.axvline(
            main_global_coupling,
            color="black",
            linestyle="--",
            label="Main G",
        )
        ax.set(
            xlabel="Global coupling G",
            ylabel="Pulse target RMS / A1 RMS",
            title="Baseline-only coupling calibration",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_main_stage_curves(
    main_normalized_df: pd.DataFrame,
    figure_dir: Path,
    *,
    periodic_probes: Sequence[str],
) -> Path:
    """Plot normalized music/speech response curves for periodic probes."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[1])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    try:
        for ax, probe in zip(axes, periodic_probes):
            plot_data = main_normalized_df[
                main_normalized_df["probe"] == probe
            ]
            for network in ("music", "speech"):
                network_data = plot_data[
                    plot_data["network"] == network
                ]
                for _, seed_data in network_data.groupby("seed"):
                    seed_data = seed_data.sort_values("severity")
                    ax.plot(
                        seed_data["severity"],
                        seed_data["log2_transfer_vs_baseline"],
                        color=NETWORK_COLORS[network],
                        alpha=0.25,
                        linewidth=1.0,
                    )
                median_data = (
                    network_data.groupby("severity", as_index=False)[
                        "log2_transfer_vs_baseline"
                    ].median()
                )
                ax.plot(
                    median_data["severity"],
                    median_data["log2_transfer_vs_baseline"],
                    color=NETWORK_COLORS[network],
                    marker="o",
                    linewidth=3.0,
                    label=NETWORK_NAMES[network],
                )
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
            ax.set(
                title=f"{probe} temporal probe",
                xlabel="AD-like perturbation strength",
                xticks=sorted(main_normalized_df["severity"].unique()),
            )
            ax.legend()
        axes[0].set_ylabel(
            "log2 target/A1 transfer relative to own baseline"
        )
        fig.suptitle("Full-field response across perturbation strengths")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_primary_endpoint_contrast(
    primary_endpoint_df: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    """Plot high-endpoint contrasts across numerical seeds."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[2])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    try:
        positions = {"2Hz": 0, "5Hz": 1}
        for probe, probe_data in primary_endpoint_df.groupby("probe"):
            x0 = positions[probe]
            offsets = np.linspace(-0.09, 0.09, len(probe_data))
            ax.scatter(
                x0 + offsets,
                probe_data["music_minus_speech_log2_change"],
                s=55,
                label=probe,
            )
            ax.hlines(
                probe_data["music_minus_speech_log2_change"].median(),
                x0 - 0.18,
                x0 + 0.18,
                color="black",
                linewidth=3,
            )
        ax.axhline(0.0, color="black", linestyle=":")
        ax.set(
            xticks=[0, 1],
            xticklabels=["2 Hz", "5 Hz"],
            ylabel="Music minus speech log2 change",
            title="High-endpoint contrast across numerical seeds",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_local_dynamics_counterfactual(
    counterfactual_comparison_df: pd.DataFrame,
    figure_dir: Path,
    *,
    periodic_probes: Sequence[str],
) -> Path:
    """Plot full-field and local-dynamics-fixed contrasts."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[3])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    try:
        analysis_order = [
            "Full regional perturbation",
            "A1 and primary targets locally fixed",
        ]
        for ax, probe in zip(axes, periodic_probes):
            subset = counterfactual_comparison_df[
                counterfactual_comparison_df["probe"] == probe
            ]
            for position, analysis in enumerate(analysis_order):
                values = subset[subset["analysis"] == analysis][
                    "music_minus_speech_log2_change"
                ].to_numpy()
                offsets = np.linspace(-0.08, 0.08, len(values))
                ax.scatter(position + offsets, values, s=45)
                ax.hlines(
                    np.median(values),
                    position - 0.17,
                    position + 0.17,
                    color="black",
                    linewidth=3,
                )
            ax.axhline(0.0, color="black", linestyle=":")
            ax.set(
                title=probe,
                xticks=[0, 1],
                xticklabels=["Full field", "Local fixed"],
                xlabel="Analysis",
            )
        axes[0].set_ylabel("Music minus speech log2 change")
        fig.suptitle("Local-dynamics counterfactual")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_matched_control_null(
    matched_null_df: pd.DataFrame,
    primary_endpoint_df: pd.DataFrame,
    figure_dir: Path,
    *,
    main_seed: int,
    periodic_probes: Sequence[str],
) -> Path:
    """Plot matched-control null distributions against observed contrasts."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[4])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    try:
        for ax, probe in zip(axes, periodic_probes):
            null_subset = matched_null_df[
                (matched_null_df["seed"] == main_seed)
                & (matched_null_df["probe"] == probe)
            ]["null_music_minus_speech"]
            observed_value = primary_endpoint_df[
                (primary_endpoint_df["seed"] == main_seed)
                & (primary_endpoint_df["probe"] == probe)
            ]["music_minus_speech_log2_change"].iloc[0]
            ax.hist(
                null_subset,
                bins=24,
                color="#9CA3AF",
                edgecolor="white",
            )
            ax.axvline(
                observed_value,
                color="#B91C1C",
                linewidth=3,
                label="Observed proxy contrast",
            )
            ax.axvline(0.0, color="black", linestyle=":")
            ax.set(title=probe, xlabel="Matched-control contrast")
            ax.legend()
        axes[0].set_ylabel("Matched control-set count")
        fig.suptitle(
            f"Simulation-level matched null, numerical seed {main_seed}"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_parameter_sensitivity(
    sensitivity_endpoint_df: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    """Plot high-endpoint sensitivity-scenario contrasts."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[5])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    try:
        scenario_labels = list(
            dict.fromkeys(sensitivity_endpoint_df["variant"])
        )
        x_positions = np.arange(len(scenario_labels))
        width = 0.28
        for offset, probe in [
            (-width / 2, "2Hz"),
            (width / 2, "5Hz"),
        ]:
            values = [
                sensitivity_endpoint_df[
                    (sensitivity_endpoint_df["variant"] == scenario)
                    & (sensitivity_endpoint_df["probe"] == probe)
                ]["music_minus_speech_log2_change"].iloc[0]
                for scenario in scenario_labels
            ]
            ax.scatter(
                x_positions + offset,
                values,
                s=65,
                label=probe,
            )
        ax.axhline(0.0, color="black", linestyle=":")
        ax.set(
            xticks=x_positions,
            xticklabels=scenario_labels,
            ylabel="Music minus speech log2 change",
            title="Parameter sensitivity at the high endpoint",
        )
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_spatial_placement_sensitivity(
    shuffle_contrast_df: pd.DataFrame,
    observed_first_seed_df: pd.DataFrame,
    figure_dir: Path,
    *,
    periodic_probes: Sequence[str],
) -> Path:
    """Plot spatial-shuffle contrasts and the observed placement."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[6])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    try:
        for probe_index, probe in enumerate(periodic_probes):
            shuffle_values = shuffle_contrast_df[
                shuffle_contrast_df["probe"] == probe
            ]["music_minus_speech_log2_change"].to_numpy()
            offsets = np.linspace(-0.09, 0.09, len(shuffle_values))
            ax.scatter(
                probe_index + offsets,
                shuffle_values,
                color="#6B7280",
                s=48,
                label="Spatial shuffles" if probe_index == 0 else None,
            )
            observed_value = observed_first_seed_df[
                observed_first_seed_df["probe"] == probe
            ]["observed_contrast"].iloc[0]
            ax.scatter(
                [probe_index],
                [observed_value],
                marker="*",
                s=190,
                color="#B91C1C",
                label="Observed placement" if probe_index == 0 else None,
            )
        ax.axhline(0.0, color="black", linestyle=":")
        ax.set(
            xticks=[0, 1],
            xticklabels=["2 Hz", "5 Hz"],
            ylabel="Music minus speech log2 change",
            title="Sensitivity to regional perturbation placement",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_semantic_episodic_secondary_analysis(
    main_normalized_df: pd.DataFrame,
    secondary_endpoint_df: pd.DataFrame,
    memory_counterfactual_comparison_df: pd.DataFrame,
    figure_dir: Path,
    *,
    periodic_probes: Sequence[str],
) -> Path:
    """Plot the separate semantic/episodic secondary analysis."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[7])
    network_colors = {
        "music_semantic_task_associated": "#6B46C1",
        "music_episodic_task_associated": "#0F766E",
    }
    network_names = {
        "music_semantic_task_associated": (
            "Semantic-task-associated proxy"
        ),
        "music_episodic_task_associated": (
            "Episodic-task-associated proxy"
        ),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    try:
        for ax, probe in zip(axes[0], periodic_probes):
            plot_data = main_normalized_df[
                main_normalized_df["probe"] == probe
            ]
            for network, color in network_colors.items():
                network_data = plot_data[
                    plot_data["network"] == network
                ]
                for _, seed_data in network_data.groupby("seed"):
                    seed_data = seed_data.sort_values("severity")
                    ax.plot(
                        seed_data["severity"],
                        seed_data["log2_transfer_vs_baseline"],
                        color=color,
                        alpha=0.25,
                        linewidth=1.0,
                    )
                median_data = (
                    network_data.groupby("severity", as_index=False)[
                        "log2_transfer_vs_baseline"
                    ].median()
                )
                ax.plot(
                    median_data["severity"],
                    median_data["log2_transfer_vs_baseline"],
                    color=color,
                    marker="o",
                    linewidth=3.0,
                    label=network_names[network],
                )
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
            ax.set(
                title=f"{probe} temporal probe",
                xlabel="AD-like perturbation strength",
                xticks=sorted(main_normalized_df["severity"].unique()),
            )
            ax.legend(fontsize=8)
        axes[0, 0].set_ylabel(
            "log2 target/A1 transfer relative to own baseline"
        )

        endpoint_ax = axes[1, 0]
        positions = {"2Hz": 0, "5Hz": 1}
        for probe, probe_data in secondary_endpoint_df.groupby("probe"):
            x0 = positions[probe]
            offsets = np.linspace(-0.09, 0.09, len(probe_data))
            endpoint_ax.scatter(
                x0 + offsets,
                probe_data["semantic_minus_episodic_log2_change"],
                s=55,
            )
            endpoint_ax.hlines(
                probe_data[
                    "semantic_minus_episodic_log2_change"
                ].median(),
                x0 - 0.18,
                x0 + 0.18,
                color="black",
                linewidth=3,
            )
        endpoint_ax.axhline(0.0, color="black", linestyle=":")
        endpoint_ax.set(
            xticks=[0, 1],
            xticklabels=["2 Hz", "5 Hz"],
            ylabel=(
                "Semantic-associated minus episodic-associated log2 change"
            ),
            title="High-endpoint secondary contrast",
        )

        counterfactual_ax = axes[1, 1]
        analysis_order = [
            "Full regional perturbation",
            "A1 and memory-proxy targets locally fixed",
        ]
        probe_colors = {"2Hz": "#7C3AED", "5Hz": "#0F766E"}
        for probe_offset, probe in zip(
            (-0.08, 0.08),
            periodic_probes,
        ):
            subset = memory_counterfactual_comparison_df[
                memory_counterfactual_comparison_df["probe"] == probe
            ]
            for position, analysis in enumerate(analysis_order):
                values = subset[subset["analysis"] == analysis][
                    "semantic_minus_episodic_log2_change"
                ].to_numpy()
                offsets = np.linspace(-0.035, 0.035, len(values))
                counterfactual_ax.scatter(
                    position + probe_offset + offsets,
                    values,
                    s=42,
                    color=probe_colors[probe],
                    label=probe if position == 0 else None,
                )
                counterfactual_ax.hlines(
                    np.median(values),
                    position + probe_offset - 0.07,
                    position + probe_offset + 0.07,
                    color=probe_colors[probe],
                    linewidth=3,
                )
        counterfactual_ax.axhline(
            0.0,
            color="black",
            linestyle=":",
        )
        counterfactual_ax.set(
            xticks=[0, 1],
            xticklabels=["Full field", "Memory targets local-fixed"],
            ylabel=(
                "Semantic-associated minus episodic-associated log2 change"
            ),
            title="Secondary local-dynamics counterfactual",
        )
        counterfactual_ax.legend()
        fig.suptitle(
            "Secondary musical-memory task-associated proxy analysis"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_semantic_episodic_matched_null(
    memory_matched_null_df: pd.DataFrame,
    secondary_endpoint_df: pd.DataFrame,
    figure_dir: Path,
    *,
    main_seed: int,
    periodic_probes: Sequence[str],
) -> Path:
    """Plot the matched-null distributions for the secondary contrast."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[8])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    try:
        for ax, probe in zip(axes, periodic_probes):
            null_subset = memory_matched_null_df[
                (memory_matched_null_df["seed"] == main_seed)
                & (memory_matched_null_df["probe"] == probe)
            ]["null_semantic_minus_episodic"]
            observed_value = secondary_endpoint_df[
                (secondary_endpoint_df["seed"] == main_seed)
                & (secondary_endpoint_df["probe"] == probe)
            ]["semantic_minus_episodic_log2_change"].iloc[0]
            ax.hist(
                null_subset,
                bins=24,
                color="#9CA3AF",
                edgecolor="white",
            )
            ax.axvline(
                observed_value,
                color="#6B21A8",
                linewidth=3,
                label="Observed secondary contrast",
            )
            ax.axvline(0.0, color="black", linestyle=":")
            ax.set(title=probe, xlabel="Matched-control contrast")
            ax.legend()
        axes[0].set_ylabel("Matched control-set count")
        fig.suptitle(
            f"Secondary matched null, numerical seed {main_seed}"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def plot_semantic_episodic_robustness(
    sensitivity_endpoint_df: pd.DataFrame,
    shuffle_contrast_df: pd.DataFrame,
    memory_observed_first_seed_df: pd.DataFrame,
    figure_dir: Path,
    *,
    periodic_probes: Sequence[str],
) -> Path:
    """Plot secondary parameter and spatial-placement sensitivity."""

    output_path = _figure_path(figure_dir, FIGURE_FILENAMES[9])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    try:
        scenario_labels = list(
            dict.fromkeys(sensitivity_endpoint_df["variant"])
        )
        x_positions = np.arange(len(scenario_labels))
        width = 0.28
        for offset, probe in [
            (-width / 2, "2Hz"),
            (width / 2, "5Hz"),
        ]:
            values = [
                sensitivity_endpoint_df[
                    (sensitivity_endpoint_df["variant"] == scenario)
                    & (sensitivity_endpoint_df["probe"] == probe)
                ]["semantic_minus_episodic_log2_change"].iloc[0]
                for scenario in scenario_labels
            ]
            axes[0].scatter(
                x_positions + offset,
                values,
                s=65,
                label=probe,
            )
        axes[0].axhline(0.0, color="black", linestyle=":")
        axes[0].set(
            xticks=x_positions,
            xticklabels=scenario_labels,
            ylabel=(
                "Semantic-associated minus episodic-associated log2 change"
            ),
            title="Secondary parameter sensitivity",
        )
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].legend()

        for probe_index, probe in enumerate(periodic_probes):
            shuffle_values = shuffle_contrast_df[
                shuffle_contrast_df["probe"] == probe
            ]["semantic_minus_episodic_log2_change"].to_numpy()
            offsets = np.linspace(-0.09, 0.09, len(shuffle_values))
            axes[1].scatter(
                probe_index + offsets,
                shuffle_values,
                color="#6B7280",
                s=48,
                label="Spatial shuffles" if probe_index == 0 else None,
            )
            observed_value = memory_observed_first_seed_df[
                memory_observed_first_seed_df["probe"] == probe
            ]["observed_contrast"].iloc[0]
            axes[1].scatter(
                [probe_index],
                [observed_value],
                marker="*",
                s=190,
                color="#6B21A8",
                label="Observed placement" if probe_index == 0 else None,
            )
        axes[1].axhline(0.0, color="black", linestyle=":")
        axes[1].set(
            xticks=[0, 1],
            xticklabels=["2 Hz", "5 Hz"],
            ylabel=(
                "Semantic-associated minus episodic-associated log2 change"
            ),
            title="Secondary spatial-placement sensitivity",
        )
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)
    return output_path


def create_all_figures(
    *,
    calibration_df: pd.DataFrame,
    main_normalized_df: pd.DataFrame,
    primary_endpoint_df: pd.DataFrame,
    secondary_endpoint_df: pd.DataFrame,
    counterfactual_comparison_df: pd.DataFrame,
    memory_counterfactual_comparison_df: pd.DataFrame,
    matched_null_df: pd.DataFrame,
    memory_matched_null_df: pd.DataFrame,
    main_seed: int,
    sensitivity_endpoint_df: pd.DataFrame,
    shuffle_contrast_df: pd.DataFrame,
    observed_first_seed_df: pd.DataFrame,
    memory_observed_first_seed_df: pd.DataFrame,
    periodic_probes: Sequence[str],
    figure_dir: Path,
) -> list[Path]:
    """Create the notebook's ten figures in their original order."""

    destination = Path(figure_dir)
    return [
        plot_baseline_coupling_calibration(calibration_df, destination),
        plot_main_stage_curves(
            main_normalized_df,
            destination,
            periodic_probes=periodic_probes,
        ),
        plot_primary_endpoint_contrast(primary_endpoint_df, destination),
        plot_local_dynamics_counterfactual(
            counterfactual_comparison_df,
            destination,
            periodic_probes=periodic_probes,
        ),
        plot_matched_control_null(
            matched_null_df,
            primary_endpoint_df,
            destination,
            main_seed=main_seed,
            periodic_probes=periodic_probes,
        ),
        plot_parameter_sensitivity(
            sensitivity_endpoint_df,
            destination,
        ),
        plot_spatial_placement_sensitivity(
            shuffle_contrast_df,
            observed_first_seed_df,
            destination,
            periodic_probes=periodic_probes,
        ),
        plot_semantic_episodic_secondary_analysis(
            main_normalized_df,
            secondary_endpoint_df,
            memory_counterfactual_comparison_df,
            destination,
            periodic_probes=periodic_probes,
        ),
        plot_semantic_episodic_matched_null(
            memory_matched_null_df,
            secondary_endpoint_df,
            destination,
            main_seed=main_seed,
            periodic_probes=periodic_probes,
        ),
        plot_semantic_episodic_robustness(
            sensitivity_endpoint_df,
            shuffle_contrast_df,
            memory_observed_first_seed_df,
            destination,
            periodic_probes=periodic_probes,
        ),
    ]
