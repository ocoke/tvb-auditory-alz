"""Publication and technical-QA figures for completed TVB379 results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from .result_analysis import (
    AnalysisProducts,
    FIT_R2_WARNING_THRESHOLD,
    PERIODIC_PROBES,
    PRIMARY_CONTRAST,
    ResultBundle,
    SECONDARY_CONTRAST,
)


INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#D1D5DB"
LIGHT = "#F3F4F6"
BLUE = "#2563EB"
BLUE_LIGHT = "#BFDBFE"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FED7AA"
PURPLE = "#7C3AED"
PURPLE_LIGHT = "#DDD6FE"
TEAL = "#0F766E"
TEAL_LIGHT = "#99F6E4"
GOLD = "#CA8A04"

NETWORK_NAMES = {
    "music": "Music proxy",
    "speech": "Speech proxy",
    "music_semantic_task_associated": "Semantic-task-associated",
    "music_episodic_task_associated": "Episodic-task-associated",
    "shared_auditory_relay": "Shared auditory relay",
    "diagram_music_temporal_proxy": "Music temporal proxy",
    "music_memory_core_proxy": "Music-memory core proxy",
    "shared_early_auditory_relay": "Early auditory relay",
    "shared_parabelt": "Parabelt",
    "shared_auditory_association": "Auditory association",
    "music_anterior_temporal_context": "Music anterior temporal context",
    "music_frontoparietal_context": "Music frontoparietal context",
    "music_medial_temporal_context": "Music medial temporal context",
    "speech_posterior_temporal": "Speech posterior temporal",
    "speech_left_frontal": "Speech left frontal",
    "speech_dorsal_left": "Speech dorsal left",
    "speech_ventral": "Speech ventral",
}

PUBLICATION_FIGURES: tuple[dict[str, str], ...] = (
    {
        "id": "publication_primary_trajectories",
        "filename": "01_primary_component_contrast_trajectories",
        "title": "Primary proxy transfer changes across perturbation strength",
        "question": (
            "How do the music and speech proxy changes combine into the "
            "primary contrast?"
        ),
        "interpretation": (
            "Component changes and their difference are shown together so a "
            "negative contrast is not mistaken for a declining music response."
        ),
        "caveat": "Numerical seeds are model stability runs, not subjects.",
        "sources": "main_music_minus_speech_contrasts.csv",
        "palette": "hard two-root cap plus neutral contrast",
    },
    {
        "id": "publication_secondary_trajectories",
        "filename": "02_secondary_component_contrast_trajectories",
        "title": (
            "Semantic- and episodic-task-associated proxy changes across "
            "perturbation strength"
        ),
        "question": (
            "How do the secondary proxy components combine into the "
            "semantic-minus-episodic contrast?"
        ),
        "interpretation": (
            "The secondary contrast is decomposed into its two operational "
            "task-associated proxy sets."
        ),
        "caveat": "The model contains no memory encoding or retrieval task.",
        "sources": "main_music_minus_speech_contrasts.csv",
        "palette": "hard two-root cap plus neutral contrast",
    },
    {
        "id": "publication_seed_consistency",
        "filename": "03_high_endpoint_seed_consistency",
        "title": "High-endpoint contrasts across numerical seeds",
        "question": "Are contrast directions consistent across seeds?",
        "interpretation": (
            "Every seed is shown directly with a median reference."
        ),
        "caveat": "Seed consistency is numerical robustness, not sample inference.",
        "sources": "high_endpoint_summary.csv",
        "palette": "hard two-root cap",
    },
    {
        "id": "publication_counterfactual",
        "filename": "04_local_dynamics_counterfactual_attenuation",
        "title": "Full-field and local-dynamics-fixed contrasts",
        "question": (
            "How much of each contrast remains when declared local dynamics "
            "are held at baseline?"
        ),
        "interpretation": "Paired seed-level contrast attenuation is shown.",
        "caveat": "Attenuation supports model dependence, not biological causality.",
        "sources": "counterfactual_attenuation.csv",
        "palette": "hard two-root cap",
    },
    {
        "id": "publication_matched_null",
        "filename": "05_matched_control_context",
        "title": "Observed contrasts within matched-control distributions",
        "question": (
            "Where do observed proxy contrasts fall relative to matched "
            "control parcel sets?"
        ),
        "interpretation": (
            "Observed seed-level values are overlaid on 500-set simulation "
            "null distributions."
        ),
        "caveat": "These distributions do not provide clinical p-values.",
        "sources": (
            "matched_control_null_metrics.csv;"
            "memory_matched_control_null_metrics.csv"
        ),
        "palette": "hard two-root cap plus neutrals",
    },
    {
        "id": "publication_spatial_shuffle",
        "filename": "06_spatial_shuffle_context",
        "title": "Observed contrasts within spatial-shuffle distributions",
        "question": (
            "How sensitive are observed contrasts to perturbation placement?"
        ),
        "interpretation": (
            "Observed seed-11 placements are compared with 100 block-preserving "
            "spatial shuffles."
        ),
        "caveat": "Empirical ranks are descriptive simulation diagnostics.",
        "sources": "spatial_shuffle_contrasts.csv",
        "palette": "hard two-root cap plus neutrals",
    },
    {
        "id": "publication_sensitivity",
        "filename": "07_parameter_sensitivity",
        "title": "High-endpoint contrasts under parameter sensitivity scenarios",
        "question": (
            "Do contrast magnitude and direction persist across tested "
            "coupling and input settings?"
        ),
        "interpretation": "Both temporal probes are shown for every scenario.",
        "caveat": "Scenarios are robustness checks, not fitted alternatives.",
        "sources": "sensitivity_summary.csv",
        "palette": "hard two-root cap",
    },
    {
        "id": "publication_network_ranking",
        "filename": "08_all_network_high_endpoint_changes",
        "title": "High-endpoint transfer changes across declared networks",
        "question": (
            "How do the primary, secondary, shared, and context networks compare?"
        ),
        "interpretation": (
            "Median and seed range are ranked separately for each probe."
        ),
        "caveat": "Network groups overlap and differ in parcel count.",
        "sources": "network_high_endpoint_summary.csv",
        "palette": "single-root preferred with inferential highlight",
    },
)

TECHNICAL_FIGURES: tuple[dict[str, str], ...] = (
    {
        "id": "technical_validation",
        "filename": "01_completeness_and_validation",
        "title": "Result completeness and validation checks",
        "question": "Is the copied result set structurally complete?",
        "interpretation": "Expected and observed workload counts are reconciled.",
        "caveat": "Completeness does not establish scientific validity.",
        "sources": "validation_summary.csv",
        "palette": "single-root preferred",
    },
    {
        "id": "technical_calibration",
        "filename": "02_coupling_calibration",
        "title": "Baseline coupling calibration diagnostics",
        "question": "How does the selected coupling compare with the scan?",
        "interpretation": (
            "A1 response, proxy transfers, balance, and evoked maxima are shown."
        ),
        "caveat": "The coupling scan is a stability diagnostic, not model fitting.",
        "sources": "calibration_summary.csv",
        "palette": "hard two-root cap",
    },
    {
        "id": "technical_convergence",
        "filename": "03_full_network_convergence_matrix",
        "title": "Integration-step relative differences across networks",
        "question": "Which inferential and descriptive networks converged?",
        "interpretation": (
            "Every endpoint/probe/network comparison is shown against 5%."
        ),
        "caveat": "Convergence does not guarantee strong periodic signal fit.",
        "sources": "integration_step_check.csv",
        "palette": "single-root preferred",
    },
    {
        "id": "technical_fit_quality",
        "filename": "04_harmonic_fit_r_squared",
        "title": "Periodic harmonic-fit R-squared distributions",
        "question": "How strong is the fitted periodic component by network?",
        "interpretation": (
            "Median and seed range are shown for every severity and probe."
        ),
        "caveat": (
            "R²=0.10 is a descriptive QA warning only, not a prespecified gate."
        ),
        "sources": "main_network_metrics.csv",
        "palette": "single-root preferred",
    },
    {
        "id": "technical_runtime",
        "filename": "05_runtime_activity_and_workers",
        "title": "Simulation runtime, activity bounds, and worker utilization",
        "question": "How was computation distributed and bounded?",
        "interpretation": (
            "Per-scope runtime/activity and per-worker call counts are shown."
        ),
        "caveat": "Aggregate call time includes concurrently executed simulations.",
        "sources": "run_manifest.csv;runtime_diagnostics.csv",
        "palette": "single-root preferred",
    },
    {
        "id": "technical_control_quality",
        "filename": "06_matching_and_shuffle_diagnostics",
        "title": "Matched-control distance and shuffle diagnostics",
        "question": (
            "Are matching quality and shuffle distributions adequately exposed?"
        ),
        "interpretation": (
            "Matching-distance and spatial-shuffle distributions are shown "
            "without inferential claims."
        ),
        "caveat": "Control draws and shuffles are simulation diagnostics.",
        "sources": (
            "matched_control_null_metrics.csv;"
            "memory_matched_control_null_metrics.csv;"
            "spatial_shuffle_contrasts.csv"
        ),
        "palette": "hard two-root cap plus neutrals",
    },
)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _subtitle(fig: plt.Figure, text: str, *, y: float = 0.945) -> None:
    fig.text(0.5, y, text, ha="center", va="top", color=MUTED, fontsize=9)


def _quiet_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _network_name(value: str) -> str:
    return NETWORK_NAMES.get(value, value.replace("_", " ").title())


def _atomic_save(
    fig: plt.Figure,
    directory: Path,
    base_filename: str,
    *,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for image_format in formats:
        destination = directory / f"{base_filename}.{image_format}"
        with tempfile.NamedTemporaryFile(
            dir=directory,
            prefix=f".{base_filename}.",
            suffix=f".{image_format}",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        try:
            fig.savefig(
                temporary,
                format=image_format,
                dpi=dpi,
                bbox_inches="tight",
                metadata={"Creator": "RISE TVB379 result analysis"},
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        written.append(destination)
    return written


def _plot_component_trajectories(
    main: pd.DataFrame,
    *,
    family: str,
) -> plt.Figure:
    if family == "primary":
        left_column, right_column, contrast_column = (
            "music",
            "speech",
            PRIMARY_CONTRAST,
        )
        labels = ("Music proxy", "Speech proxy")
        colors = (BLUE, ORANGE)
        title = "Primary proxy transfer changes across perturbation strength"
        contrast_label = "Music minus speech log2 change"
    else:
        left_column, right_column, contrast_column = (
            "music_semantic_task_associated",
            "music_episodic_task_associated",
            SECONDARY_CONTRAST,
        )
        labels = (
            "Semantic-task-associated proxy",
            "Episodic-task-associated proxy",
        )
        colors = (PURPLE, TEAL)
        title = (
            "Semantic- and episodic-task-associated proxy changes across "
            "perturbation strength"
        )
        contrast_label = "Semantic-associated minus episodic-associated"

    periodic = main[main["probe"].isin(PERIODIC_PROBES)]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.5, 8.2),
        sharex="col",
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    _subtitle(
        fig,
        "Thin paths are numerical seeds; heavy paths are medians. "
        "Values are log2 transfer relative to each network's baseline.",
        y=0.955,
    )
    for column_index, probe in enumerate(PERIODIC_PROBES):
        subset = periodic[periodic["probe"] == probe]
        component_ax = axes[0, column_index]
        contrast_ax = axes[1, column_index]
        for response_column, label, color, marker in zip(
            (left_column, right_column),
            labels,
            colors,
            ("o", "s"),
            strict=True,
        ):
            for _, seed_frame in subset.groupby("seed", sort=True):
                ordered = seed_frame.sort_values("severity")
                component_ax.plot(
                    ordered["severity"],
                    ordered[response_column],
                    color=color,
                    alpha=0.22,
                    linewidth=1.0,
                    marker=marker,
                    markersize=3,
                )
            median = (
                subset.groupby("severity", as_index=False)[response_column]
                .median()
                .sort_values("severity")
            )
            component_ax.plot(
                median["severity"],
                median[response_column],
                color=color,
                marker=marker,
                linewidth=2.7,
                markersize=6,
                label=label,
            )

        for _, seed_frame in subset.groupby("seed", sort=True):
            ordered = seed_frame.sort_values("severity")
            contrast_ax.plot(
                ordered["severity"],
                ordered[contrast_column],
                color=INK,
                alpha=0.22,
                linewidth=1.0,
                marker="D",
                markersize=3,
            )
        median = (
            subset.groupby("severity", as_index=False)[contrast_column]
            .median()
            .sort_values("severity")
        )
        contrast_ax.plot(
            median["severity"],
            median[contrast_column],
            color=INK,
            marker="D",
            linewidth=2.7,
            markersize=6,
        )
        for ax in (component_ax, contrast_ax):
            ax.axhline(0.0, color=INK, linestyle=":", linewidth=1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xticklabels(["Baseline", "Intermediate", "High"])
            _quiet_axis(ax)
        component_ax.set_title(f"{probe} component changes")
        contrast_ax.set_title(f"{probe} contrast")
        contrast_ax.set_xlabel("AD-like perturbation strength")
        component_ax.legend(loc="best")
        if column_index == 0:
            component_ax.set_ylabel("Network log2 transfer change")
            contrast_ax.set_ylabel(contrast_label)
    fig.subplots_adjust(top=0.89, hspace=0.33, wspace=0.20)
    return fig


def _plot_high_endpoint_consistency(
    main: pd.DataFrame,
) -> plt.Figure:
    endpoint = main[
        np.isclose(main["severity"], 1.0)
        & main["probe"].isin(PERIODIC_PROBES)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.suptitle(
        "High-endpoint contrasts across numerical seeds",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Each point is one deterministic numerical seed; black bars are medians.",
        y=0.95,
    )
    for ax, family, column, color, ylabel in (
        (
            axes[0],
            "Primary",
            PRIMARY_CONTRAST,
            BLUE,
            "Music minus speech log2 change",
        ),
        (
            axes[1],
            "Secondary",
            SECONDARY_CONTRAST,
            PURPLE,
            "Semantic-associated minus episodic-associated",
        ),
    ):
        for position, probe in enumerate(PERIODIC_PROBES):
            values = endpoint[endpoint["probe"] == probe][column].to_numpy()
            offsets = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(
                position + offsets,
                values,
                color=color,
                edgecolor=INK,
                linewidth=0.5,
                s=55,
                zorder=3,
            )
            ax.hlines(
                np.median(values),
                position - 0.18,
                position + 0.18,
                color=INK,
                linewidth=3,
            )
        ax.axhline(0.0, color=INK, linestyle=":", linewidth=1.0)
        ax.set(
            title=f"{family} contrast",
            xticks=[0, 1],
            xticklabels=["2 Hz", "5 Hz"],
            ylabel=ylabel,
        )
        _quiet_axis(ax)
    fig.subplots_adjust(top=0.84, wspace=0.25)
    return fig


def _plot_counterfactual(
    attenuation: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
    fig.suptitle(
        "Full-field and local-dynamics-fixed contrasts",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Lines pair the same seed. Local fixing restores A1 and the declared "
        "target groups to baseline local dynamics.",
        y=0.953,
    )
    for row_index, (family, color, ylabel) in enumerate(
        (
            ("primary", BLUE, "Music minus speech log2 change"),
            (
                "secondary",
                PURPLE,
                "Semantic-associated minus episodic-associated",
            ),
        )
    ):
        for column_index, probe in enumerate(PERIODIC_PROBES):
            ax = axes[row_index, column_index]
            subset = attenuation[
                (attenuation["contrast_family"] == family)
                & (attenuation["probe"] == probe)
            ]
            for record in subset.itertuples(index=False):
                ax.plot(
                    [0, 1],
                    [
                        record.full_field_contrast,
                        record.local_fixed_contrast,
                    ],
                    color=color,
                    alpha=0.45,
                    linewidth=1.2,
                    marker="o",
                    markersize=5,
                )
            medians = [
                subset["full_field_contrast"].median(),
                subset["local_fixed_contrast"].median(),
            ]
            ax.plot(
                [0, 1],
                medians,
                color=INK,
                linewidth=2.8,
                marker="D",
                markersize=6,
                label="Median",
            )
            median_attenuation = subset["attenuation_percent"].median()
            ax.text(
                0.98,
                0.94,
                f"Median attenuation: {median_attenuation:.1f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=MUTED,
            )
            ax.axhline(0.0, color=INK, linestyle=":", linewidth=1.0)
            ax.set(
                title=f"{family.title()} · {probe}",
                xticks=[0, 1],
                xticklabels=["Full field", "Local fixed"],
            )
            if column_index == 0:
                ax.set_ylabel(ylabel)
            _quiet_axis(ax)
    fig.subplots_adjust(top=0.89, hspace=0.30, wspace=0.20)
    return fig


def _plot_matched_null(
    bundle: ResultBundle,
    main: pd.DataFrame,
) -> plt.Figure:
    endpoint = main[
        np.isclose(main["severity"], 1.0)
        & main["probe"].isin(PERIODIC_PROBES)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), sharey="row")
    fig.suptitle(
        "Observed contrasts within matched-control distributions",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Boxes summarize 500 matched parcel sets for each numerical seed; "
        "diamonds are observed proxy contrasts.",
        y=0.952,
    )
    specifications = (
        (
            "primary",
            PRIMARY_CONTRAST,
            bundle.tables["matched_null"],
            "null_music_minus_speech",
            BLUE,
            "Music minus speech log2 change",
        ),
        (
            "secondary",
            SECONDARY_CONTRAST,
            bundle.tables["memory_matched_null"],
            "null_semantic_minus_episodic",
            PURPLE,
            "Semantic-associated minus episodic-associated",
        ),
    )
    for row_index, (
        family,
        observed_column,
        null_frame,
        null_column,
        color,
        ylabel,
    ) in enumerate(specifications):
        for column_index, probe in enumerate(PERIODIC_PROBES):
            ax = axes[row_index, column_index]
            probe_null = null_frame[null_frame["probe"] == probe]
            seeds = sorted(int(value) for value in probe_null["seed"].unique())
            distributions = [
                probe_null[probe_null["seed"] == seed][null_column].to_numpy()
                for seed in seeds
            ]
            box = ax.boxplot(
                distributions,
                tick_labels=[str(seed) for seed in seeds],
                widths=0.58,
                whis=(5, 95),
                showfliers=False,
                patch_artist=True,
            )
            for patch in box["boxes"]:
                patch.set(facecolor=LIGHT, edgecolor=MUTED, linewidth=1.0)
            for median in box["medians"]:
                median.set(color=INK, linewidth=1.8)
            observed = endpoint[endpoint["probe"] == probe].set_index("seed")
            ax.scatter(
                np.arange(1, len(seeds) + 1),
                [observed.loc[seed, observed_column] for seed in seeds],
                marker="D",
                color=color,
                edgecolor=INK,
                linewidth=0.6,
                s=48,
                zorder=4,
                label="Observed",
            )
            ax.axhline(0.0, color=INK, linestyle=":", linewidth=1.0)
            ax.set(
                title=f"{family.title()} · {probe}",
                xlabel="Numerical seed",
            )
            if column_index == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0 and column_index == 1:
                ax.legend(loc="upper right")
            _quiet_axis(ax)
    fig.subplots_adjust(top=0.89, hspace=0.30, wspace=0.16)
    return fig


def _plot_spatial_shuffle(
    bundle: ResultBundle,
    context: pd.DataFrame,
) -> plt.Figure:
    shuffle = bundle.tables["shuffle"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharey="row")
    fig.suptitle(
        "Observed contrasts within spatial-shuffle distributions",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Histograms contain 100 block-preserving perturbation shuffles for "
        "the main numerical seed; dashed lines bound the central 90%.",
        y=0.952,
    )
    for row_index, (family, column, color, xlabel) in enumerate(
        (
            (
                "primary",
                PRIMARY_CONTRAST,
                BLUE,
                "Music minus speech log2 change",
            ),
            (
                "secondary",
                SECONDARY_CONTRAST,
                PURPLE,
                "Semantic-associated minus episodic-associated",
            ),
        )
    ):
        for column_index, probe in enumerate(PERIODIC_PROBES):
            ax = axes[row_index, column_index]
            values = shuffle[shuffle["probe"] == probe][column].to_numpy()
            record = context[
                (context["contrast_family"] == family)
                & (context["probe"] == probe)
            ].iloc[0]
            ax.hist(
                values,
                bins=18,
                color=LIGHT,
                edgecolor=MUTED,
                linewidth=0.7,
            )
            ax.axvline(
                record["shuffle_5th_percentile"],
                color=MUTED,
                linestyle="--",
                linewidth=1.2,
            )
            ax.axvline(
                record["shuffle_95th_percentile"],
                color=MUTED,
                linestyle="--",
                linewidth=1.2,
            )
            ax.axvline(
                record["observed_contrast"],
                color=color,
                linewidth=2.8,
                label="Observed",
            )
            ax.axvline(0.0, color=INK, linestyle=":", linewidth=1.0)
            ax.text(
                0.98,
                0.94,
                f"Observed percentile: "
                f"{record['observed_empirical_percentile']:.1f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=MUTED,
            )
            ax.set(
                title=f"{family.title()} · {probe}",
                xlabel=xlabel,
                ylabel="Shuffle count" if column_index == 0 else None,
            )
            if row_index == 0 and column_index == 1:
                ax.legend(loc="upper left")
            _quiet_axis(ax)
    fig.subplots_adjust(top=0.89, hspace=0.34, wspace=0.18)
    return fig


def _scenario_order(values: Sequence[str]) -> list[str]:
    preferred = [
        "G30",
        "G60_input_0.02",
        "G100",
        "input_0.01",
        "input_0.04",
    ]
    present = set(values)
    return [value for value in preferred if value in present] + sorted(
        present - set(preferred)
    )


def _plot_parameter_sensitivity(
    sensitivity: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    fig.suptitle(
        "High-endpoint contrasts under parameter sensitivity scenarios",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "G60/input-0.02 is the main result; scenario order is categorical.",
        y=0.947,
    )
    for ax, family, color, ylabel in (
        (
            axes[0],
            "primary",
            BLUE,
            "Music minus speech log2 change",
        ),
        (
            axes[1],
            "secondary",
            PURPLE,
            "Semantic-associated minus episodic-associated",
        ),
    ):
        subset = sensitivity[sensitivity["contrast_family"] == family]
        variants = _scenario_order(subset["variant"].tolist())
        positions = np.arange(len(variants))
        for probe, marker, offset, fill in (
            ("2Hz", "o", -0.09, color),
            ("5Hz", "s", 0.09, "white"),
        ):
            values = [
                float(
                    subset[
                        (subset["variant"] == variant)
                        & (subset["probe"] == probe)
                    ]["contrast"].iloc[0]
                )
                for variant in variants
            ]
            ax.scatter(
                positions + offset,
                values,
                marker=marker,
                facecolor=fill,
                edgecolor=color,
                linewidth=1.5,
                s=70,
                label=probe,
                zorder=3,
            )
        main_position = (
            variants.index("G60_input_0.02")
            if "G60_input_0.02" in variants
            else None
        )
        if main_position is not None:
            ax.axvspan(
                main_position - 0.35,
                main_position + 0.35,
                color=LIGHT,
                zorder=0,
            )
        ax.axhline(0.0, color=INK, linestyle=":", linewidth=1.0)
        ax.set(
            title=f"{family.title()} contrast",
            xticks=positions,
            xticklabels=variants,
            ylabel=ylabel,
        )
        ax.tick_params(axis="x", rotation=25)
        ax.legend(loc="best")
        _quiet_axis(ax)
    fig.subplots_adjust(top=0.82, bottom=0.20, wspace=0.23)
    return fig


def _plot_network_ranking(
    network_summary: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 9.0))
    fig.suptitle(
        "High-endpoint transfer changes across declared networks",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Points are seed medians; lines span the five-seed range. "
        "Filled points mark prespecified inferential networks.",
        y=0.953,
    )
    for ax, probe in zip(axes, PERIODIC_PROBES, strict=True):
        subset = network_summary[
            network_summary["probe"] == probe
        ].sort_values("median_log2_change")
        y = np.arange(len(subset))
        for position, record in enumerate(subset.itertuples(index=False)):
            color = BLUE if record.required_for_inference else MUTED
            marker_face = color if record.required_for_inference else "white"
            ax.hlines(
                position,
                record.minimum_log2_change,
                record.maximum_log2_change,
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )
            ax.scatter(
                record.median_log2_change,
                position,
                facecolor=marker_face,
                edgecolor=color,
                linewidth=1.2,
                s=48,
                zorder=3,
            )
        ax.axvline(0.0, color=INK, linestyle=":", linewidth=1.0)
        ax.set(
            title=probe,
            yticks=y,
            yticklabels=[_network_name(value) for value in subset["network"]],
            xlabel="log2 transfer change at high perturbation",
        )
        _quiet_axis(ax, grid_axis="x")
    fig.subplots_adjust(top=0.89, left=0.20, right=0.98, wspace=0.55)
    return fig


def _plot_validation(
    validation: pd.DataFrame,
    bundle: ResultBundle,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 6.0),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    fig.suptitle(
        "Result completeness and validation checks",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Checks run before any derived table or figure is written.",
        y=0.95,
    )
    by_category = (
        validation.groupby("category", as_index=False)
        .agg(total=("passed", "size"), passed=("passed", "sum"))
        .sort_values("category")
    )
    y = np.arange(len(by_category))
    axes[0].barh(y, by_category["total"], color=LIGHT, edgecolor=MUTED)
    axes[0].barh(y, by_category["passed"], color=BLUE, alpha=0.85)
    for position, record in enumerate(by_category.itertuples(index=False)):
        axes[0].text(
            record.total + 0.15,
            position,
            f"{record.passed}/{record.total}",
            va="center",
            color=INK,
        )
    axes[0].set(
        yticks=y,
        yticklabels=[value.title() for value in by_category["category"]],
        xlabel="Validation checks",
        title="Passed checks by category",
    )
    _quiet_axis(axes[0], grid_axis="x")

    workload = bundle.metadata["workload"]
    manifest = len(bundle.tables["run_manifest"])
    calibration_calls = 2 * len(bundle.tables["calibration"])
    count_rows = [
        ("Manifested simulations", manifest, int(workload["manifest"])),
        (
            "Calibration calls",
            calibration_calls,
            int(workload["calibration"]),
        ),
        (
            "Total TVB calls",
            manifest + calibration_calls,
            int(workload["total"]),
        ),
        (
            "Spatial shuffles",
            bundle.tables["shuffle"]["variant"].nunique(),
            int(bundle.metadata["model"]["spatial_shuffles"]),
        ),
    ]
    axes[1].axis("off")
    cell_text = [
        [label, f"{observed}", f"{expected}", "Pass"]
        for label, observed, expected in count_rows
    ]
    table = axes[1].table(
        cellText=cell_text,
        colLabels=["Measure", "Observed", "Expected", "Status"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.48, 0.18, 0.18, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(LIGHT if row == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
    axes[1].set_title("Expected and observed workload", pad=14)
    fig.subplots_adjust(top=0.84, wspace=0.28)
    return fig


def _plot_calibration(calibration: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    fig.suptitle(
        "Baseline coupling calibration diagnostics",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "The shaded vertical band marks the selected main coupling.",
        y=0.95,
    )
    selected = calibration.loc[
        calibration["selected_main_coupling"],
        "global_coupling",
    ]
    selected_value = float(selected.iloc[0]) if len(selected) else np.nan
    x = calibration["global_coupling"]
    panels = (
        (
            axes[0, 0],
            [("A1 RMS", calibration["a1_rms"], BLUE, "o")],
            "A1 pulse response",
            "RMS response",
        ),
        (
            axes[0, 1],
            [
                ("Music proxy", calibration["music_transfer"], BLUE, "o"),
                ("Speech proxy", calibration["speech_transfer"], ORANGE, "s"),
            ],
            "Target-to-A1 pulse transfer",
            "Transfer",
        ),
        (
            axes[1, 0],
            [
                (
                    "Balanced target score",
                    calibration["balanced_target_score"],
                    PURPLE,
                    "D",
                )
            ],
            "Balanced target score",
            "Geometric-mean transfer",
        ),
        (
            axes[1, 1],
            [
                (
                    "Maximum evoked activity",
                    calibration["max_abs_evoked"],
                    TEAL,
                    "^",
                )
            ],
            "Maximum absolute evoked activity",
            "Absolute PSP",
        ),
    )
    for ax, series, title, ylabel in panels:
        for label, values, color, marker in series:
            ax.plot(
                x,
                values,
                color=color,
                marker=marker,
                linewidth=2.0,
                label=label,
            )
        if np.isfinite(selected_value):
            ax.axvspan(
                selected_value - 3.0,
                selected_value + 3.0,
                color=LIGHT,
                zorder=0,
            )
            ax.axvline(selected_value, color=INK, linestyle="--", linewidth=1.0)
        if ax is axes[1, 1]:
            ax.axhline(
                50.0,
                color=MUTED,
                linestyle=":",
                linewidth=1.2,
                label="Saturation check (50)",
            )
        ax.set(title=title, xlabel="Global coupling G", ylabel=ylabel)
        ax.legend(loc="best")
        _quiet_axis(ax)
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.24)
    return fig


def _plot_convergence(dt: pd.DataFrame) -> plt.Figure:
    frame = dt.copy()
    required = frame["required_for_inference"]
    if not pd.api.types.is_bool_dtype(required):
        required = required.astype(str).str.lower().isin({"true", "1"})
    frame["required_for_inference"] = required
    columns = [
        (0.0, "2Hz"),
        (0.0, "5Hz"),
        (1.0, "2Hz"),
        (1.0, "5Hz"),
    ]
    networks = (
        frame.groupby("network")["required_for_inference"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    matrix = np.full((len(networks), len(columns)), np.nan)
    for row_index, network in enumerate(networks):
        for column_index, (severity, probe) in enumerate(columns):
            selected = frame[
                (frame["network"] == network)
                & np.isclose(frame["severity"], severity)
                & (frame["probe"] == probe)
            ]["relative_difference"]
            if len(selected) == 1:
                matrix[row_index, column_index] = 100.0 * float(selected.iloc[0])

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    fig.suptitle(
        "Integration-step relative differences across networks",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "0.5 ms versus 0.25 ms transfer difference; starred labels are "
        "prespecified inferential networks; threshold is 5%.",
        y=0.95,
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="Blues",
        norm=Normalize(vmin=0.0, vmax=5.0),
    )
    labels = []
    for network in networks:
        is_required = bool(
            frame.loc[
                frame["network"] == network,
                "required_for_inference",
            ].max()
        )
        labels.append(
            ("★ " if is_required else "  ") + _network_name(network)
        )
    ax.set(
        xticks=np.arange(len(columns)),
        xticklabels=[
            "Baseline · 2 Hz",
            "Baseline · 5 Hz",
            "High · 2 Hz",
            "High · 5 Hz",
        ],
        yticks=np.arange(len(networks)),
        yticklabels=labels,
    )
    ax.tick_params(axis="x", rotation=20)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value > 2.8 else INK,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Relative transfer difference (%)")
    fig.subplots_adjust(top=0.87, left=0.28, bottom=0.12)
    return fig


def _plot_fit_quality(main_network: pd.DataFrame) -> plt.Figure:
    periodic = main_network[
        main_network["probe"].isin(PERIODIC_PROBES)
    ]
    severities = sorted(float(value) for value in periodic["severity"].unique())
    networks = sorted(str(value) for value in periodic["network"].unique())
    fig, axes = plt.subplots(
        len(severities),
        2,
        figsize=(14.0, 5.0 * len(severities)),
        sharex=True,
    )
    fig.suptitle(
        "Periodic harmonic-fit R-squared distributions",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    _subtitle(
        fig,
        "Points are seed medians within each network; lines span the seed range. "
        "The 0.10 line is descriptive QA only.",
        y=0.977,
    )
    for row_index, severity in enumerate(severities):
        for column_index, probe in enumerate(PERIODIC_PROBES):
            ax = axes[row_index, column_index]
            subset = periodic[
                np.isclose(periodic["severity"], severity)
                & (periodic["probe"] == probe)
            ]
            summary = (
                subset.groupby("network", as_index=False)
                .agg(
                    median=("median_target_fit_r_squared", "median"),
                    minimum=("median_target_fit_r_squared", "min"),
                    maximum=("median_target_fit_r_squared", "max"),
                )
                .set_index("network")
                .reindex(networks)
            )
            y = np.arange(len(networks))
            ax.hlines(
                y,
                summary["minimum"],
                summary["maximum"],
                color=BLUE_LIGHT,
                linewidth=2.0,
            )
            ax.scatter(
                summary["median"],
                y,
                color=BLUE,
                edgecolor=INK,
                linewidth=0.4,
                s=32,
                zorder=3,
            )
            ax.axvline(
                FIT_R2_WARNING_THRESHOLD,
                color=INK,
                linestyle=":",
                linewidth=1.1,
            )
            ax.set(
                title=f"Severity {severity:g} · {probe}",
                yticks=y,
                yticklabels=(
                    [_network_name(value) for value in networks]
                    if column_index == 0
                    else []
                ),
                xlabel="Median target harmonic-fit R²"
                if row_index == len(severities) - 1
                else None,
            )
            _quiet_axis(ax, grid_axis="x")
    fig.subplots_adjust(
        top=0.94,
        left=0.22,
        right=0.98,
        hspace=0.27,
        wspace=0.10,
    )
    return fig


def _plot_runtime(
    manifest: pd.DataFrame,
    runtime: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    fig.suptitle(
        "Simulation runtime, activity bounds, and worker utilization",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Per-call durations overlap under multiprocessing; activity bounds are "
        "shown against the prespecified PSP safety limit.",
        y=0.95,
    )
    scope_groups = [
        (scope, group["wall_seconds"].to_numpy())
        for scope, group in manifest.groupby("scope", sort=True)
    ]
    axes[0, 0].boxplot(
        [values for _, values in scope_groups],
        tick_labels=[name for name, _ in scope_groups],
        showfliers=False,
        patch_artist=True,
    )
    for patch in axes[0, 0].artists:
        patch.set_facecolor(LIGHT)
    axes[0, 0].set(
        title="Per-call runtime by scope",
        ylabel="Wall seconds",
    )
    axes[0, 0].tick_params(axis="x", rotation=30)
    _quiet_axis(axes[0, 0])

    workers = runtime[runtime["record_type"] == "worker"].sort_values("group")
    axes[0, 1].bar(
        workers["group"],
        workers["simulations"],
        color=BLUE,
        edgecolor=INK,
        linewidth=0.5,
    )
    axes[0, 1].set(
        title="Recorded calls per worker",
        xlabel="Worker PID",
        ylabel="Simulation calls",
    )
    axes[0, 1].tick_params(axis="x", rotation=30)
    _quiet_axis(axes[0, 1])

    activity_groups = [
        (scope, group["max_abs_psp"].to_numpy())
        for scope, group in manifest.groupby("scope", sort=True)
    ]
    axes[1, 0].boxplot(
        [values for _, values in activity_groups],
        tick_labels=[name for name, _ in activity_groups],
        showfliers=False,
    )
    axes[1, 0].axhline(100.0, color=INK, linestyle=":", linewidth=1.2)
    axes[1, 0].set(
        title="Maximum absolute PSP by scope",
        ylabel="Absolute PSP",
    )
    axes[1, 0].tick_params(axis="x", rotation=30)
    _quiet_axis(axes[1, 0])

    stimulated = manifest[manifest["simulation_type"] == "stimulated"]
    evoked_groups = [
        (scope, group["max_abs_evoked"].dropna().to_numpy())
        for scope, group in stimulated.groupby("scope", sort=True)
    ]
    axes[1, 1].boxplot(
        [values for _, values in evoked_groups],
        tick_labels=[name for name, _ in evoked_groups],
        showfliers=False,
    )
    axes[1, 1].set(
        title="Maximum absolute evoked response by scope",
        ylabel="Absolute evoked PSP",
    )
    axes[1, 1].tick_params(axis="x", rotation=30)
    _quiet_axis(axes[1, 1])
    fig.subplots_adjust(top=0.86, bottom=0.17, hspace=0.48, wspace=0.24)
    return fig


def _plot_control_quality(bundle: ResultBundle) -> plt.Figure:
    primary = bundle.tables["matched_null"]
    secondary = bundle.tables["memory_matched_null"]
    shuffle = bundle.tables["shuffle"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    fig.suptitle(
        "Matched-control distance and shuffle diagnostics",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    _subtitle(
        fig,
        "Distributions are simulation diagnostics; no independent-subject "
        "inference is implied.",
        y=0.95,
    )
    axes[0, 0].hist(
        primary["mean_standardized_match_distance"],
        bins=24,
        color=BLUE_LIGHT,
        edgecolor=BLUE,
        label="Primary controls",
        alpha=0.75,
    )
    axes[0, 0].hist(
        secondary["mean_standardized_match_distance"],
        bins=24,
        histtype="step",
        color=PURPLE,
        linewidth=1.8,
        label="Secondary controls",
    )
    axes[0, 0].set(
        title="Standardized matching distance",
        xlabel="Mean standardized distance",
        ylabel="Rows",
    )
    axes[0, 0].legend()
    _quiet_axis(axes[0, 0])

    match_summary = pd.DataFrame(
        {
            "family": ["Primary", "Secondary"],
            "median": [
                primary["mean_standardized_match_distance"].median(),
                secondary["mean_standardized_match_distance"].median(),
            ],
            "q05": [
                primary["mean_standardized_match_distance"].quantile(0.05),
                secondary["mean_standardized_match_distance"].quantile(0.05),
            ],
            "q95": [
                primary["mean_standardized_match_distance"].quantile(0.95),
                secondary["mean_standardized_match_distance"].quantile(0.95),
            ],
        }
    )
    x = np.arange(2)
    axes[0, 1].errorbar(
        x,
        match_summary["median"],
        yerr=np.vstack(
            [
                match_summary["median"] - match_summary["q05"],
                match_summary["q95"] - match_summary["median"],
            ]
        ),
        fmt="o",
        color=INK,
        ecolor=BLUE,
        capsize=5,
    )
    axes[0, 1].set(
        title="Matching-distance central 90% range",
        xticks=x,
        xticklabels=match_summary["family"],
        ylabel="Mean standardized distance",
    )
    _quiet_axis(axes[0, 1])

    for ax, column, color, title in (
        (
            axes[1, 0],
            PRIMARY_CONTRAST,
            BLUE,
            "Primary spatial-shuffle contrast",
        ),
        (
            axes[1, 1],
            SECONDARY_CONTRAST,
            PURPLE,
            "Secondary spatial-shuffle contrast",
        ),
    ):
        for probe, linestyle in (("2Hz", "-"), ("5Hz", "--")):
            values = np.sort(
                shuffle[shuffle["probe"] == probe][column].to_numpy()
            )
            percentile = 100.0 * (
                np.arange(1, len(values) + 1) - 0.5
            ) / len(values)
            ax.plot(
                values,
                percentile,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=probe,
            )
        ax.axvline(0.0, color=INK, linestyle=":", linewidth=1.0)
        ax.set(
            title=title,
            xlabel="Shuffle contrast",
            ylabel="Empirical percentile",
        )
        ax.legend()
        _quiet_axis(ax)
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.24)
    return fig


def figure_specifications(audience: str) -> list[dict[str, str]]:
    if audience not in {"publication", "technical", "both"}:
        raise ValueError("audience must be publication, technical, or both")
    specifications: list[dict[str, str]] = []
    if audience in {"publication", "both"}:
        specifications.extend(dict(value, audience="publication") for value in PUBLICATION_FIGURES)
    if audience in {"technical", "both"}:
        specifications.extend(dict(value, audience="technical") for value in TECHNICAL_FIGURES)
    return specifications


def create_analysis_figures(
    bundle: ResultBundle,
    products: AnalysisProducts,
    *,
    output_dir: Path,
    audience: str,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    """Create the selected static figure sets and their chart manifest."""

    invalid_formats = sorted(set(formats) - {"png", "svg"})
    if invalid_formats:
        raise ValueError(
            "Unsupported figure formats: " + ", ".join(invalid_formats)
        )
    if not formats or len(set(formats)) != len(formats):
        raise ValueError("formats must be non-empty and unique")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    _apply_style()
    specifications = figure_specifications(audience)
    specification_by_id = {item["id"]: item for item in specifications}
    output_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    main = bundle.tables["main_contrasts"]

    publication_builders: Mapping[str, Any] = {
        "publication_primary_trajectories": lambda: (
            _plot_component_trajectories(main, family="primary")
        ),
        "publication_secondary_trajectories": lambda: (
            _plot_component_trajectories(main, family="secondary")
        ),
        "publication_seed_consistency": lambda: (
            _plot_high_endpoint_consistency(main)
        ),
        "publication_counterfactual": lambda: _plot_counterfactual(
            products.tables["counterfactual_attenuation"]
        ),
        "publication_matched_null": lambda: _plot_matched_null(
            bundle,
            main,
        ),
        "publication_spatial_shuffle": lambda: _plot_spatial_shuffle(
            bundle,
            products.tables["spatial_shuffle_context"],
        ),
        "publication_sensitivity": lambda: _plot_parameter_sensitivity(
            products.tables["sensitivity_summary"]
        ),
        "publication_network_ranking": lambda: _plot_network_ranking(
            products.tables["network_high_endpoint_summary"]
        ),
    }
    technical_builders: Mapping[str, Any] = {
        "technical_validation": lambda: _plot_validation(
            products.tables["validation_summary"],
            bundle,
        ),
        "technical_calibration": lambda: _plot_calibration(
            products.tables["calibration_summary"]
        ),
        "technical_convergence": lambda: _plot_convergence(
            bundle.tables["dt_convergence"]
        ),
        "technical_fit_quality": lambda: _plot_fit_quality(
            bundle.tables["main_network"]
        ),
        "technical_runtime": lambda: _plot_runtime(
            bundle.tables["run_manifest"],
            products.tables["runtime_diagnostics"],
        ),
        "technical_control_quality": lambda: _plot_control_quality(bundle),
    }
    builders = {**publication_builders, **technical_builders}

    for figure_id, specification in specification_by_id.items():
        figure = builders[figure_id]()
        try:
            destination = (
                output_dir
                / (
                    "publication"
                    if specification["audience"] == "publication"
                    else "technical_qa"
                )
                / "figures"
            )
            paths = _atomic_save(
                figure,
                destination,
                specification["filename"],
                formats=formats,
                dpi=dpi,
            )
        finally:
            plt.close(figure)
        output_paths.extend(paths)
        manifest_rows.append(
            {
                "figure_id": figure_id,
                "audience": specification["audience"],
                "base_filename": specification["filename"],
                "title": specification["title"],
                "analytical_question": specification["question"],
                "supported_interpretation": specification["interpretation"],
                "caveat": specification["caveat"],
                "source_tables": specification["sources"],
                "palette_policy": specification["palette"],
                "formats": ";".join(formats),
                "dpi": int(dpi),
            }
        )
    return output_paths, pd.DataFrame(manifest_rows)


__all__ = [
    "PUBLICATION_FIGURES",
    "TECHNICAL_FIGURES",
    "create_analysis_figures",
    "figure_specifications",
]
