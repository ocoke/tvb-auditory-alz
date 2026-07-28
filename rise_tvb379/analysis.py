"""Pure dataframe analyses used after TVB simulation blocks complete."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    DT_CHECK_PROBES,
    DT_CHECK_SEVERITIES,
    SEVERITY_LABELS,
)
from .metrics import (
    a1_response_value,
    draw_matched_set,
    make_contrasts,
    node_response_vector,
    normalize_to_baseline,
)

LOGGER = logging.getLogger(__name__)


def summarize_main_stage(
    main_normalized_df: pd.DataFrame,
    main_contrast_df: pd.DataFrame,
    *,
    periodic_probes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build stage summaries and the prespecified high endpoint."""

    main_stage_summary_df = (
        main_normalized_df.groupby(
            ["probe", "severity", "condition", "network"], as_index=False
        )
        .agg(
            median_log2_change=("log2_transfer_vs_baseline", "median"),
            minimum_log2_change=("log2_transfer_vs_baseline", "min"),
            maximum_log2_change=("log2_transfer_vs_baseline", "max"),
            numerical_seeds=("seed", "nunique"),
        )
        .sort_values(["probe", "severity", "network"])
        .reset_index(drop=True)
    )
    primary_endpoint_df = main_contrast_df[
        (main_contrast_df["severity"] == 1.0)
        & (main_contrast_df["probe"].isin(periodic_probes))
    ].copy()
    return main_stage_summary_df, primary_endpoint_df


def compare_counterfactual(
    primary_endpoint_df: pd.DataFrame,
    local_fixed_network_df: pd.DataFrame,
    main_network_df: pd.DataFrame,
    *,
    periodic_probes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize and contrast the local-dynamics-held-baseline endpoint."""

    local_fixed_normalized_df = normalize_to_baseline(
        local_fixed_network_df,
        baseline_df=main_network_df,
    )
    local_fixed_contrast_df = make_contrasts(local_fixed_normalized_df)
    counterfactual_comparison_df = pd.concat(
        [
            primary_endpoint_df.assign(
                analysis="Full regional perturbation"
            ),
            local_fixed_contrast_df[
                local_fixed_contrast_df["probe"].isin(periodic_probes)
            ].assign(analysis="A1 and target local dynamics fixed"),
        ],
        ignore_index=True,
    )
    return (
        local_fixed_normalized_df,
        local_fixed_contrast_df,
        counterfactual_comparison_df,
    )


def build_matched_control_sets(
    *,
    weights: np.ndarray,
    baseline_b: np.ndarray,
    high_b: np.ndarray,
    labels: np.ndarray,
    a1_indices: np.ndarray,
    music_indices: np.ndarray,
    speech_indices: np.ndarray,
    all_declared_indices: np.ndarray,
    n_sets: int,
    random_seed: int = 20260727,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create equal-sized topology/pathology-matched control groups."""

    match_weights = 0.5 * (weights + weights.T)
    cortical_indices = np.arange(360)
    weighted_strength = match_weights.sum(axis=1)
    direct_a1_affinity = match_weights[:, a1_indices].sum(axis=1)
    local_b_reduction = baseline_b - high_b

    raw_features = np.column_stack(
        [
            np.log10(weighted_strength + 1e-15),
            np.log10(direct_a1_affinity + 1e-15),
            local_b_reduction,
        ]
    )
    cortical_mean = raw_features[cortical_indices].mean(axis=0)
    cortical_std = raw_features[cortical_indices].std(axis=0)
    if np.any(cortical_std <= 0):
        raise RuntimeError("A matching feature has zero variance.")
    matching_z = (raw_features - cortical_mean) / cortical_std

    target_indices = np.r_[music_indices, speech_indices]
    target_feature_df = pd.DataFrame(
        {
            "label": [*labels[music_indices], *labels[speech_indices]],
            "network": ["music"] * len(music_indices)
            + ["speech"] * len(speech_indices),
            "weighted_strength": weighted_strength[target_indices],
            "direct_A1_affinity": direct_a1_affinity[target_indices],
            "local_b_reduction": local_b_reduction[target_indices],
        }
    )

    matched_rng = np.random.default_rng(random_seed)
    excluded = set(all_declared_indices.tolist())
    matched_set_rows: list[dict[str, object]] = []
    for set_id in range(n_sets):
        music_control, music_distances = draw_matched_set(
            music_indices,
            matched_rng,
            reserved=excluded,
            matching_z=matching_z,
        )
        speech_control, speech_distances = draw_matched_set(
            speech_indices,
            matched_rng,
            reserved=excluded.union(music_control.tolist()),
            matching_z=matching_z,
        )
        matched_set_rows.append(
            {
                "set_id": set_id,
                "music_control_indices": ";".join(
                    map(str, music_control)
                ),
                "speech_control_indices": ";".join(
                    map(str, speech_control)
                ),
                "music_control_labels": ";".join(labels[music_control]),
                "speech_control_labels": ";".join(labels[speech_control]),
                "mean_standardized_match_distance": float(
                    np.mean(music_distances + speech_distances)
                ),
            }
        )
    return target_feature_df, pd.DataFrame(matched_set_rows)


def score_matched_control_null(
    *,
    main_node_df: pd.DataFrame,
    main_network_df: pd.DataFrame,
    matched_sets_df: pd.DataFrame,
    primary_endpoint_df: pd.DataFrame,
    seeds: tuple[int, ...],
    periodic_probes: tuple[str, ...],
    n_regions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score declared contrasts against the simulation-level matched null."""

    matched_null_rows: list[dict[str, object]] = []
    for seed in seeds:
        for probe in periodic_probes:
            baseline_response = node_response_vector(
                main_node_df,
                seed,
                probe,
                0.0,
                n_regions=n_regions,
            )
            high_response = node_response_vector(
                main_node_df,
                seed,
                probe,
                1.0,
                n_regions=n_regions,
            )
            baseline_a1 = a1_response_value(
                main_network_df, seed, probe, 0.0
            )
            high_a1 = a1_response_value(
                main_network_df, seed, probe, 1.0
            )

            for row in matched_sets_df.itertuples(index=False):
                music_control = np.fromstring(
                    row.music_control_indices, sep=";", dtype=int
                )
                speech_control = np.fromstring(
                    row.speech_control_indices, sep=";", dtype=int
                )
                music_baseline_transfer = (
                    np.mean(baseline_response[music_control]) / baseline_a1
                )
                music_high_transfer = (
                    np.mean(high_response[music_control]) / high_a1
                )
                speech_baseline_transfer = (
                    np.mean(baseline_response[speech_control]) / baseline_a1
                )
                speech_high_transfer = (
                    np.mean(high_response[speech_control]) / high_a1
                )
                music_change = np.log2(
                    max(music_high_transfer, 1e-15)
                    / max(music_baseline_transfer, 1e-15)
                )
                speech_change = np.log2(
                    max(speech_high_transfer, 1e-15)
                    / max(speech_baseline_transfer, 1e-15)
                )
                matched_null_rows.append(
                    {
                        "set_id": row.set_id,
                        "seed": seed,
                        "probe": probe,
                        "music_control_log2_change": music_change,
                        "speech_control_log2_change": speech_change,
                        "null_music_minus_speech": (
                            music_change - speech_change
                        ),
                        "mean_standardized_match_distance": (
                            row.mean_standardized_match_distance
                        ),
                    }
                )

    matched_null_df = pd.DataFrame(matched_null_rows)
    summary_rows: list[dict[str, object]] = []
    for observed in primary_endpoint_df.itertuples(index=False):
        null_values = matched_null_df[
            (matched_null_df["seed"] == observed.seed)
            & (matched_null_df["probe"] == observed.probe)
        ]["null_music_minus_speech"].to_numpy()
        observed_value = observed.music_minus_speech_log2_change
        summary_rows.append(
            {
                "seed": observed.seed,
                "probe": observed.probe,
                "observed_contrast": observed_value,
                "null_median": float(np.median(null_values)),
                "null_5th_percentile": float(
                    np.quantile(null_values, 0.05)
                ),
                "null_95th_percentile": float(
                    np.quantile(null_values, 0.95)
                ),
                "observed_percentile_within_simulation_null": float(
                    100.0 * np.mean(null_values <= observed_value)
                ),
            }
        )
    return matched_null_df, pd.DataFrame(summary_rows)


def build_sensitivity_tables(
    frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize each sensitivity scope and calculate high endpoints."""

    sensitivity_network_df = pd.concat(frames, ignore_index=True)
    normalized_frames = [
        normalize_to_baseline(group)
        for _, group in sensitivity_network_df.groupby("scope", sort=False)
    ]
    sensitivity_normalized_df = pd.concat(
        normalized_frames, ignore_index=True
    )
    sensitivity_contrast_df = make_contrasts(
        sensitivity_normalized_df
    )
    sensitivity_endpoint_df = sensitivity_contrast_df[
        sensitivity_contrast_df["severity"] == 1.0
    ].copy()
    return (
        sensitivity_network_df,
        sensitivity_normalized_df,
        sensitivity_contrast_df,
        sensitivity_endpoint_df,
    )


def build_shuffle_tables(
    frames: list[pd.DataFrame],
    *,
    main_network_df: pd.DataFrame,
    primary_endpoint_df: pd.DataFrame,
    main_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize spatial shuffles to the main baseline and summarize them."""

    shuffle_network_df = pd.concat(frames, ignore_index=True)
    normalized_frames = [
        normalize_to_baseline(group, baseline_df=main_network_df)
        for _, group in shuffle_network_df.groupby("scope", sort=False)
    ]
    shuffle_normalized_df = pd.concat(
        normalized_frames, ignore_index=True
    )
    shuffle_contrast_df = make_contrasts(shuffle_normalized_df)
    observed_first_seed_df = primary_endpoint_df[
        primary_endpoint_df["seed"] == main_seed
    ][["probe", "music_minus_speech_log2_change"]].rename(
        columns={"music_minus_speech_log2_change": "observed_contrast"}
    )
    shuffle_summary_df = (
        shuffle_contrast_df.groupby("probe", as_index=False)
        .agg(
            shuffle_median=(
                "music_minus_speech_log2_change",
                "median",
            ),
            shuffle_minimum=(
                "music_minus_speech_log2_change",
                "min",
            ),
            shuffle_maximum=(
                "music_minus_speech_log2_change",
                "max",
            ),
            spatial_shuffles=("variant", "nunique"),
        )
        .merge(observed_first_seed_df, on="probe", how="left")
    )
    return (
        shuffle_network_df,
        shuffle_normalized_df,
        shuffle_contrast_df,
        observed_first_seed_df,
        shuffle_summary_df,
    )


def check_integration_step(
    *,
    main_network_df: pd.DataFrame,
    reference_network_df: pd.DataFrame,
    main_seed: int,
    severities: tuple[float, ...] = DT_CHECK_SEVERITIES,
    probes: tuple[str, ...] = DT_CHECK_PROBES,
    networks: tuple[str, ...] = ("music", "speech"),
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Enforce the 1.0 ms versus 0.5 ms transfer gate at every endpoint."""

    if not severities or not probes or not networks:
        raise ValueError(
            "Convergence severities, probes, and networks must be non-empty."
        )
    if threshold <= 0.0:
        raise ValueError("The convergence threshold must be positive.")

    key_columns = ["severity", "probe", "network"]
    metric_columns = ["transfer", "median_target_fit_r_squared"]
    main_required = {"seed", *key_columns, *metric_columns}
    reference_required = {*key_columns, *metric_columns}
    missing_main_columns = sorted(main_required - set(main_network_df.columns))
    missing_reference_columns = sorted(
        reference_required - set(reference_network_df.columns)
    )
    if missing_main_columns:
        raise RuntimeError(
            "Main integration-step results are missing required columns: "
            + ", ".join(missing_main_columns)
        )
    if missing_reference_columns:
        raise RuntimeError(
            "Reference integration-step results are missing required columns: "
            + ", ".join(missing_reference_columns)
        )

    expected_keys = {
        (float(severity), str(probe), str(network))
        for severity in severities
        for probe in probes
        for network in networks
    }

    def select_and_validate(
        frame: pd.DataFrame,
        *,
        label: str,
        seed: int | None,
    ) -> pd.DataFrame:
        selected = frame
        if seed is not None:
            selected = selected[selected["seed"] == seed]
        selected = selected[
            selected["severity"].isin(severities)
            & selected["probe"].isin(probes)
            & selected["network"].isin(networks)
        ][key_columns + metric_columns].copy()

        duplicate_rows = selected[
            selected.duplicated(key_columns, keep=False)
        ][key_columns].drop_duplicates()
        if not duplicate_rows.empty:
            duplicate_keys = sorted(
                (
                    float(row.severity),
                    str(row.probe),
                    str(row.network),
                )
                for row in duplicate_rows.itertuples(index=False)
            )
            raise RuntimeError(
                f"{label} integration-step results contain duplicate "
                f"coverage keys: {duplicate_keys}"
            )

        actual_keys = {
            (float(row.severity), str(row.probe), str(row.network))
            for row in selected[key_columns].itertuples(index=False)
        }
        missing_keys = sorted(expected_keys - actual_keys)
        if missing_keys:
            raise RuntimeError(
                f"{label} integration-step results are missing required "
                f"coverage keys: {missing_keys}"
            )
        return selected

    dt_main = select_and_validate(
        main_network_df,
        label="Main",
        seed=main_seed,
    ).rename(
        columns={
            "transfer": "transfer_dt_1.0ms",
            "median_target_fit_r_squared": (
                "median_target_fit_r_squared_dt_1.0ms"
            ),
        }
    )
    dt_reference = select_and_validate(
        reference_network_df,
        label="Reference",
        seed=None,
    ).rename(
        columns={
            "transfer": "transfer_dt_0.5ms",
            "median_target_fit_r_squared": (
                "median_target_fit_r_squared_dt_0.5ms"
            ),
        }
    )
    result = dt_main.merge(
        dt_reference,
        on=key_columns,
        validate="one_to_one",
    )
    result["condition"] = result["severity"].map(SEVERITY_LABELS)
    if result["condition"].isna().any():
        unknown = sorted(
            result.loc[result["condition"].isna(), "severity"].unique()
        )
        raise RuntimeError(
            "Integration-step results contain severities without labels: "
            f"{unknown}"
        )
    result["relative_difference"] = (
        (
            result["transfer_dt_1.0ms"]
            - result["transfer_dt_0.5ms"]
        ).abs()
        / result["transfer_dt_0.5ms"].abs().clip(lower=1e-15)
    )
    result["fit_r_squared_difference"] = (
        result["median_target_fit_r_squared_dt_1.0ms"]
        - result["median_target_fit_r_squared_dt_0.5ms"]
    ).abs()
    result = result.sort_values(key_columns).reset_index(drop=True)
    LOGGER.info(
        "Integration-step convergence comparisons:\n%s",
        result[
            [
                "condition",
                "probe",
                "network",
                "transfer_dt_1.0ms",
                "transfer_dt_0.5ms",
                "relative_difference",
                "median_target_fit_r_squared_dt_1.0ms",
                "median_target_fit_r_squared_dt_0.5ms",
                "fit_r_squared_difference",
            ]
        ].to_string(index=False),
    )

    failed_rows = result[result["relative_difference"] >= threshold]
    if not failed_rows.empty:
        LOGGER.error(
            "Integration-step convergence failures:\n%s",
            failed_rows.to_string(index=False),
        )
        raise RuntimeError(
            "The 1.0 ms integration step failed the prespecified "
            f"{threshold:.0%} convergence check."
        )
    return result
