"""Pure dataframe analyses used after TVB simulation blocks complete."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (
    DT_CHECK_NETWORKS,
    DT_CHECK_PROBES,
    DT_CHECK_SEVERITIES,
    DT_RELATIVE_TOLERANCE,
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
    endpoint_df: pd.DataFrame,
    local_fixed_network_df: pd.DataFrame,
    main_network_df: pd.DataFrame,
    *,
    periodic_probes: tuple[str, ...],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the separate primary and memory local-dynamics comparisons."""

    local_fixed_normalized_df = normalize_to_baseline(
        local_fixed_network_df,
        baseline_df=main_network_df,
    )
    local_fixed_contrast_df = make_contrasts(local_fixed_normalized_df)
    periodic_local = local_fixed_contrast_df[
        local_fixed_contrast_df["probe"].isin(periodic_probes)
    ]
    primary_counterfactual_comparison_df = pd.concat(
        [
            endpoint_df.assign(
                analysis="Full regional perturbation"
            ),
            periodic_local[
                periodic_local["variant"]
                == "primary_local_fixed_endpoint"
            ].assign(analysis="A1 and primary targets locally fixed"),
        ],
        ignore_index=True,
    )
    memory_counterfactual_comparison_df = pd.concat(
        [
            endpoint_df.assign(
                analysis="Full regional perturbation"
            ),
            periodic_local[
                periodic_local["variant"]
                == "memory_local_fixed_endpoint"
            ].assign(
                analysis="A1 and memory-proxy targets locally fixed"
            ),
        ],
        ignore_index=True,
    )
    return (
        local_fixed_normalized_df,
        local_fixed_contrast_df,
        primary_counterfactual_comparison_df,
        memory_counterfactual_comparison_df,
    )


def build_matching_features(
    *,
    weights: np.ndarray,
    baseline_b: np.ndarray,
    high_b: np.ndarray,
    labels: np.ndarray,
    a1_indices: np.ndarray,
    target_groups: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build standardized topology/pathology matching features."""

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

    rows: list[dict[str, object]] = []
    for group_name, indices in target_groups.items():
        for index in np.asarray(indices, dtype=int):
            rows.append(
                {
                    "label": labels[index],
                    "network": group_name,
                    "weighted_strength": weighted_strength[index],
                    "direct_A1_affinity": direct_a1_affinity[index],
                    "local_b_reduction": local_b_reduction[index],
                }
            )
    return pd.DataFrame(rows), matching_z


def build_matched_pair_sets(
    *,
    labels: np.ndarray,
    matching_z: np.ndarray,
    all_declared_indices: np.ndarray,
    pair_name: str,
    left_name: str,
    left_indices: np.ndarray,
    right_name: str,
    right_indices: np.ndarray,
    n_sets: int,
    random_seed: int,
) -> pd.DataFrame:
    """Create deterministic, size-preserving matched controls for one pair."""

    matched_rng = np.random.default_rng(random_seed)
    excluded = set(all_declared_indices.tolist())
    matched_set_rows: list[dict[str, object]] = []
    for set_id in range(n_sets):
        left_control, left_distances = draw_matched_set(
            left_indices,
            matched_rng,
            reserved=excluded,
            matching_z=matching_z,
        )
        right_control, right_distances = draw_matched_set(
            right_indices,
            matched_rng,
            reserved=excluded.union(left_control.tolist()),
            matching_z=matching_z,
        )
        matched_set_rows.append(
            {
                "pair_name": pair_name,
                "set_id": set_id,
                "left_name": left_name,
                "right_name": right_name,
                "left_control_indices": ";".join(map(str, left_control)),
                "right_control_indices": ";".join(map(str, right_control)),
                "left_control_labels": ";".join(labels[left_control]),
                "right_control_labels": ";".join(labels[right_control]),
                "mean_standardized_match_distance": float(
                    np.mean(left_distances + right_distances)
                ),
            }
        )
    return pd.DataFrame(matched_set_rows)


def score_matched_control_null(
    *,
    main_node_df: pd.DataFrame,
    main_network_df: pd.DataFrame,
    pair_sets_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    observed_column: str,
    left_output_column: str,
    right_output_column: str,
    null_output_column: str,
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

            for row in pair_sets_df.itertuples(index=False):
                left_control = np.fromstring(
                    row.left_control_indices, sep=";", dtype=int
                )
                right_control = np.fromstring(
                    row.right_control_indices, sep=";", dtype=int
                )
                left_baseline_transfer = (
                    np.mean(baseline_response[left_control]) / baseline_a1
                )
                left_high_transfer = (
                    np.mean(high_response[left_control]) / high_a1
                )
                right_baseline_transfer = (
                    np.mean(baseline_response[right_control]) / baseline_a1
                )
                right_high_transfer = (
                    np.mean(high_response[right_control]) / high_a1
                )
                left_change = np.log2(
                    max(left_high_transfer, 1e-15)
                    / max(left_baseline_transfer, 1e-15)
                )
                right_change = np.log2(
                    max(right_high_transfer, 1e-15)
                    / max(right_baseline_transfer, 1e-15)
                )
                matched_null_rows.append(
                    {
                        "pair_name": row.pair_name,
                        "set_id": row.set_id,
                        "seed": seed,
                        "probe": probe,
                        "left_name": row.left_name,
                        "right_name": row.right_name,
                        left_output_column: left_change,
                        right_output_column: right_change,
                        null_output_column: left_change - right_change,
                        "mean_standardized_match_distance": (
                            row.mean_standardized_match_distance
                        ),
                    }
                )

    matched_null_df = pd.DataFrame(matched_null_rows)
    summary_rows: list[dict[str, object]] = []
    for observed in observed_df.itertuples(index=False):
        null_values = matched_null_df[
            (matched_null_df["seed"] == observed.seed)
            & (matched_null_df["probe"] == observed.probe)
        ][null_output_column].to_numpy()
        observed_value = getattr(observed, observed_column)
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
    endpoint_df: pd.DataFrame,
    main_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
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
    observed_first_seed_df = endpoint_df[
        endpoint_df["seed"] == main_seed
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
    memory_observed_first_seed_df = endpoint_df[
        endpoint_df["seed"] == main_seed
    ][["probe", "semantic_minus_episodic_log2_change"]].rename(
        columns={
            "semantic_minus_episodic_log2_change": "observed_contrast"
        }
    )
    memory_shuffle_summary_df = (
        shuffle_contrast_df.groupby("probe", as_index=False)
        .agg(
            shuffle_median=(
                "semantic_minus_episodic_log2_change",
                "median",
            ),
            shuffle_minimum=(
                "semantic_minus_episodic_log2_change",
                "min",
            ),
            shuffle_maximum=(
                "semantic_minus_episodic_log2_change",
                "max",
            ),
            spatial_shuffles=("variant", "nunique"),
        )
        .merge(memory_observed_first_seed_df, on="probe", how="left")
    )
    return (
        shuffle_network_df,
        shuffle_normalized_df,
        shuffle_contrast_df,
        observed_first_seed_df,
        shuffle_summary_df,
        memory_observed_first_seed_df,
        memory_shuffle_summary_df,
    )


def check_integration_step(
    *,
    main_network_df: pd.DataFrame,
    reference_network_df: pd.DataFrame,
    main_seed: int,
    severities: tuple[float, ...] = DT_CHECK_SEVERITIES,
    probes: tuple[str, ...] = DT_CHECK_PROBES,
    networks: tuple[str, ...] = DT_CHECK_NETWORKS,
    inferential_networks: tuple[str, ...] = DT_CHECK_NETWORKS,
    threshold: float = DT_RELATIVE_TOLERANCE,
) -> pd.DataFrame:
    """Compare 0.5 ms with 0.25 ms and gate inferential networks."""

    if not severities or not probes or not networks:
        raise ValueError(
            "Convergence severities, probes, and networks must be non-empty."
        )
    if threshold <= 0.0:
        raise ValueError("The convergence threshold must be positive.")
    if not set(inferential_networks).issubset(networks):
        raise ValueError(
            "Inferential convergence networks must be included in networks."
        )

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
            "transfer": "transfer_dt_0.5ms",
            "median_target_fit_r_squared": (
                "median_target_fit_r_squared_dt_0.5ms"
            ),
        }
    )
    dt_reference = select_and_validate(
        reference_network_df,
        label="Reference",
        seed=None,
    ).rename(
        columns={
            "transfer": "transfer_dt_0.25ms",
            "median_target_fit_r_squared": (
                "median_target_fit_r_squared_dt_0.25ms"
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
    result["required_for_inference"] = result["network"].isin(
        inferential_networks
    )
    result["relative_difference"] = (
        (
            result["transfer_dt_0.5ms"]
            - result["transfer_dt_0.25ms"]
        ).abs()
        / result["transfer_dt_0.25ms"].abs().clip(lower=1e-15)
    )
    result["convergence_passed"] = (
        result["relative_difference"] < threshold
    )
    result["fit_r_squared_difference"] = (
        result["median_target_fit_r_squared_dt_0.5ms"]
        - result["median_target_fit_r_squared_dt_0.25ms"]
    ).abs()
    result = result.sort_values(
        ["required_for_inference", *key_columns],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    LOGGER.info(
        "Integration-step convergence comparisons:\n%s",
        result[
            [
                "condition",
                "probe",
                "network",
                "required_for_inference",
                "transfer_dt_0.5ms",
                "transfer_dt_0.25ms",
                "relative_difference",
                "convergence_passed",
                "median_target_fit_r_squared_dt_0.5ms",
                "median_target_fit_r_squared_dt_0.25ms",
                "fit_r_squared_difference",
            ]
        ].to_string(index=False),
    )

    failed_rows = result[
        result["required_for_inference"]
        & ~result["convergence_passed"]
    ]
    if not failed_rows.empty:
        LOGGER.error(
            "Integration-step convergence failures:\n%s",
            failed_rows.to_string(index=False),
        )
        raise RuntimeError(
            "The 0.5 ms integration step failed the prespecified "
            f"{threshold:.0%} convergence check for at least one "
            "inferential network."
        )
    descriptive_failures = result[
        ~result["required_for_inference"]
        & ~result["convergence_passed"]
    ]
    if not descriptive_failures.empty:
        LOGGER.warning(
            "Descriptive context networks outside the convergence tolerance; "
            "they remain saved but must not be interpreted inferentially:\n%s",
            descriptive_failures.to_string(index=False),
        )
    return result
