from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from rise_tvb379.plots import FIGURE_FILENAMES, create_all_figures


def test_create_all_figures_writes_ten_nonempty_pngs(tmp_path) -> None:
    periodic_probes = ("2Hz", "5Hz")
    seeds = (11, 23)

    calibration_df = pd.DataFrame(
        {
            "global_coupling": [30.0, 60.0, 100.0],
            "music_transfer": [0.7, 1.0, 1.2],
            "speech_transfer": [0.8, 0.9, 1.0],
        }
    )

    main_rows = []
    for probe_index, probe in enumerate(periodic_probes):
        for network_index, network in enumerate(
            (
                "music",
                "speech",
                "music_semantic_task_associated",
                "music_episodic_task_associated",
            )
        ):
            for seed_index, seed in enumerate(seeds):
                for severity in (0.0, 0.5, 1.0):
                    main_rows.append(
                        {
                            "probe": probe,
                            "network": network,
                            "seed": seed,
                            "severity": severity,
                            "log2_transfer_vs_baseline": (
                                severity
                                * (probe_index + 1)
                                * (1.0 if network_index == 0 else 0.4)
                                + seed_index * 0.02
                            ),
                        }
                    )
    main_normalized_df = pd.DataFrame(main_rows)

    primary_endpoint_df = pd.DataFrame(
        [
            {
                "seed": seed,
                "probe": probe,
                "music_minus_speech_log2_change": (
                    0.3 + 0.1 * probe_index + 0.01 * seed_index
                ),
            }
            for probe_index, probe in enumerate(periodic_probes)
            for seed_index, seed in enumerate(seeds)
        ]
    )
    secondary_endpoint_df = primary_endpoint_df.assign(
        semantic_minus_episodic_log2_change=[
            0.18 + 0.04 * index
            for index in range(len(primary_endpoint_df))
        ]
    )

    analyses = (
        "Full regional perturbation",
        "A1 and primary targets locally fixed",
    )
    memory_counterfactual_comparison_df = pd.DataFrame(
        [
            {
                "probe": probe,
                "analysis": analysis,
                "semantic_minus_episodic_log2_change": (
                    0.15
                    + 0.06 * probe_index
                    - 0.04 * analysis_index
                    + 0.01 * seed_index
                ),
            }
            for probe_index, probe in enumerate(periodic_probes)
            for analysis_index, analysis in enumerate(
                (
                    "Full regional perturbation",
                    "A1 and memory-proxy targets locally fixed",
                )
            )
            for seed_index, _ in enumerate(seeds)
        ]
    )
    counterfactual_comparison_df = pd.DataFrame(
        [
            {
                "probe": probe,
                "analysis": analysis,
                "music_minus_speech_log2_change": (
                    0.2
                    + 0.1 * probe_index
                    - 0.05 * analysis_index
                    + 0.01 * seed_index
                ),
            }
            for probe_index, probe in enumerate(periodic_probes)
            for analysis_index, analysis in enumerate(analyses)
            for seed_index, _ in enumerate(seeds)
        ]
    )

    matched_null_df = pd.DataFrame(
        [
            {
                "seed": 11,
                "probe": probe,
                "null_music_minus_speech": (
                    -0.2 + 0.04 * sample + 0.02 * probe_index
                ),
            }
            for probe_index, probe in enumerate(periodic_probes)
            for sample in range(12)
        ]
    )
    memory_matched_null_df = pd.DataFrame(
        [
            {
                "seed": 11,
                "probe": probe,
                "null_semantic_minus_episodic": (
                    -0.15 + 0.03 * sample + 0.01 * probe_index
                ),
            }
            for probe_index, probe in enumerate(periodic_probes)
            for sample in range(12)
        ]
    )

    sensitivity_endpoint_df = pd.DataFrame(
        [
            {
                "variant": variant,
                "probe": probe,
                "music_minus_speech_log2_change": (
                    0.1 + 0.05 * variant_index + 0.1 * probe_index
                ),
                "semantic_minus_episodic_log2_change": (
                    0.08 + 0.03 * variant_index + 0.06 * probe_index
                ),
            }
            for variant_index, variant in enumerate(("G30", "G100"))
            for probe_index, probe in enumerate(periodic_probes)
        ]
    )

    shuffle_contrast_df = pd.DataFrame(
        [
            {
                "probe": probe,
                "music_minus_speech_log2_change": (
                    -0.1 + 0.05 * shuffle + 0.04 * probe_index
                ),
                "semantic_minus_episodic_log2_change": (
                    -0.08 + 0.03 * shuffle + 0.02 * probe_index
                ),
            }
            for probe_index, probe in enumerate(periodic_probes)
            for shuffle in range(5)
        ]
    )
    observed_first_seed_df = pd.DataFrame(
        {
            "probe": periodic_probes,
            "observed_contrast": [0.35, 0.55],
        }
    )
    memory_observed_first_seed_df = pd.DataFrame(
        {
            "probe": periodic_probes,
            "observed_contrast": [0.22, 0.31],
        }
    )

    figure_paths = create_all_figures(
        calibration_df=calibration_df,
        main_normalized_df=main_normalized_df,
        primary_endpoint_df=primary_endpoint_df,
        secondary_endpoint_df=secondary_endpoint_df,
        counterfactual_comparison_df=counterfactual_comparison_df,
        memory_counterfactual_comparison_df=(
            memory_counterfactual_comparison_df
        ),
        matched_null_df=matched_null_df,
        memory_matched_null_df=memory_matched_null_df,
        main_seed=11,
        sensitivity_endpoint_df=sensitivity_endpoint_df,
        shuffle_contrast_df=shuffle_contrast_df,
        observed_first_seed_df=observed_first_seed_df,
        memory_observed_first_seed_df=memory_observed_first_seed_df,
        periodic_probes=periodic_probes,
        figure_dir=tmp_path / "figures",
    )

    assert [path.name for path in figure_paths] == list(FIGURE_FILENAMES)
    assert len(figure_paths) == 10
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in figure_paths
    )
    assert plt.get_fignums() == []
