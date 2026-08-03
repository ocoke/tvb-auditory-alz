-- Reproducible DuckDB source map for TVB379_visual_investigator.html.
--
-- The executable dashboard builder is build_html_investigator.py. These views
-- document the equivalent file-level source layer exposed by the HTML reader.
-- Paths are relative to the project root. All 180 raw NPZ trace arrays are
-- audited by recommended_posthoc_analysis.py using their manifested paths and
-- hashes. DuckDB views below expose the resulting bounded derived tables; raw
-- arrays themselves are never embedded in the portable HTML.

CREATE OR REPLACE VIEW a1_frequency_qa AS SELECT * FROM read_csv_auto('data_analysis/a1_frequency_qa.csv', header = true);
CREATE OR REPLACE VIEW baseline_coupling_diagnostic AS SELECT * FROM read_csv_auto('data_analysis/baseline_coupling_diagnostic.csv', header = true);
CREATE OR REPLACE VIEW counterfactual_comparison AS SELECT * FROM read_csv_auto('data_analysis/counterfactual_comparison.csv', header = true);
CREATE OR REPLACE VIEW counterfactual_summary AS SELECT * FROM read_csv_auto('data_analysis/counterfactual_summary.csv', header = true);
CREATE OR REPLACE VIEW data_quality_checks AS SELECT * FROM read_csv_auto('data_analysis/data_quality_checks.csv', header = true);
CREATE OR REPLACE VIEW definition_sensitivity_statistics AS SELECT * FROM read_csv_auto('data_analysis/definition_sensitivity_statistics.csv', header = true);
CREATE OR REPLACE VIEW dt_reference_network_metrics AS SELECT * FROM read_csv_auto('data_analysis/dt_reference_network_metrics.csv', header = true);
CREATE OR REPLACE VIEW dt_reference_node_metrics AS SELECT * FROM read_csv_auto('data_analysis/dt_reference_node_metrics.csv', header = true);
CREATE OR REPLACE VIEW integration_step_a1_snr_seed_diagnostics AS SELECT * FROM read_csv_auto('data_analysis/integration_step_a1_snr_seed_diagnostics.csv', header = true);
CREATE OR REPLACE VIEW integration_step_check AS SELECT * FROM read_csv_auto('data_analysis/integration_step_check.csv', header = true);
CREATE OR REPLACE VIEW integration_step_interaction_seed_diagnostics AS SELECT * FROM read_csv_auto('data_analysis/integration_step_interaction_seed_diagnostics.csv', header = true);
CREATE OR REPLACE VIEW integration_step_outcome_eligibility AS SELECT * FROM read_csv_auto('data_analysis/integration_step_outcome_eligibility.csv', header = true);
CREATE OR REPLACE VIEW integration_step_raw_metric_seed_diagnostics AS SELECT * FROM read_csv_auto('data_analysis/integration_step_raw_metric_seed_diagnostics.csv', header = true);
CREATE OR REPLACE VIEW laterality_difference_statistics AS SELECT * FROM read_csv_auto('data_analysis/laterality_difference_statistics.csv', header = true);
CREATE OR REPLACE VIEW local_fixed_interaction_statistics AS SELECT * FROM read_csv_auto('data_analysis/local_fixed_interaction_statistics.csv', header = true);
CREATE OR REPLACE VIEW local_fixed_network_metrics AS SELECT * FROM read_csv_auto('data_analysis/local_fixed_network_metrics.csv', header = true);
CREATE OR REPLACE VIEW local_fixed_node_metrics AS SELECT * FROM read_csv_auto('data_analysis/local_fixed_node_metrics.csv', header = true);
CREATE OR REPLACE VIEW local_fixed_pair_interactions AS SELECT * FROM read_csv_auto('data_analysis/local_fixed_pair_interactions.csv', header = true);
CREATE OR REPLACE VIEW main_interaction_statistics AS SELECT * FROM read_csv_auto('data_analysis/main_interaction_statistics.csv', header = true);
CREATE OR REPLACE VIEW main_network_metrics AS SELECT * FROM read_csv_auto('data_analysis/main_network_metrics.csv', header = true);
CREATE OR REPLACE VIEW main_network_metrics_normalized AS SELECT * FROM read_csv_auto('data_analysis/main_network_metrics_normalized.csv', header = true);
CREATE OR REPLACE VIEW main_node_metrics AS SELECT * FROM read_csv_auto('data_analysis/main_node_metrics.csv', header = true);
CREATE OR REPLACE VIEW main_pair_interactions AS SELECT * FROM read_csv_auto('data_analysis/main_pair_interactions.csv', header = true);
CREATE OR REPLACE VIEW main_parcel_trace_manifest AS SELECT * FROM read_csv_auto('data_analysis/main_parcel_trace_manifest.csv', header = true);
CREATE OR REPLACE VIEW main_science_validity AS SELECT * FROM read_csv_auto('data_analysis/main_science_validity.csv', header = true);
CREATE OR REPLACE VIEW matched_control_null_metrics AS SELECT * FROM read_csv_auto('data_analysis/matched_control_null_metrics.csv', header = true);
CREATE OR REPLACE VIEW matched_control_null_summary AS SELECT * FROM read_csv_auto('data_analysis/matched_control_null_summary.csv', header = true);
CREATE OR REPLACE VIEW matched_control_sets AS SELECT * FROM read_csv_auto('data_analysis/matched_control_sets.csv', header = true);
CREATE OR REPLACE VIEW music_memory_peak_mapping AS SELECT * FROM read_csv_auto('data_analysis/music_memory_peak_mapping.csv', header = true);
CREATE OR REPLACE VIEW network_evidence AS SELECT * FROM read_csv_auto('data_analysis/network_evidence.csv', header = true);
CREATE OR REPLACE VIEW outcome_eligibility AS SELECT * FROM read_csv_auto('data_analysis/outcome_eligibility.csv', header = true);
CREATE OR REPLACE VIEW parameter_interaction_statistics AS SELECT * FROM read_csv_auto('data_analysis/parameter_interaction_statistics.csv', header = true);
CREATE OR REPLACE VIEW parameter_network_metrics AS SELECT * FROM read_csv_auto('data_analysis/parameter_network_metrics.csv', header = true);
CREATE OR REPLACE VIEW parameter_node_metrics AS SELECT * FROM read_csv_auto('data_analysis/parameter_node_metrics.csv', header = true);
CREATE OR REPLACE VIEW parameter_pair_interactions AS SELECT * FROM read_csv_auto('data_analysis/parameter_pair_interactions.csv', header = true);
CREATE OR REPLACE VIEW pathology_summary AS SELECT * FROM read_csv_auto('data_analysis/pathology_summary.csv', header = true);
CREATE OR REPLACE VIEW periodic_temporal_qa AS SELECT * FROM read_csv_auto('data_analysis/periodic_temporal_qa.csv', header = true);
CREATE OR REPLACE VIEW primary_interaction_statistics AS SELECT * FROM read_csv_auto('data_analysis/primary_interaction_statistics.csv', header = true);
CREATE OR REPLACE VIEW regional_features AS SELECT * FROM read_csv_auto('data_analysis/regional_features.csv', header = true);
CREATE OR REPLACE VIEW roi_definitions AS SELECT * FROM read_csv_auto('data_analysis/roi_definitions.csv', header = true);
CREATE OR REPLACE VIEW roi_pathology_values AS SELECT * FROM read_csv_auto('data_analysis/roi_pathology_values.csv', header = true);
CREATE OR REPLACE VIEW run_manifest AS SELECT * FROM read_csv_auto('data_analysis/run_manifest.csv', header = true);
CREATE OR REPLACE VIEW source_manifest AS SELECT * FROM read_csv_auto('data_analysis/source_manifest.csv', header = true);
CREATE OR REPLACE VIEW spatial_shuffle_network_metrics AS SELECT * FROM read_csv_auto('data_analysis/spatial_shuffle_network_metrics.csv', header = true);
CREATE OR REPLACE VIEW spatial_shuffle_node_metrics AS SELECT * FROM read_csv_auto('data_analysis/spatial_shuffle_node_metrics.csv', header = true);
CREATE OR REPLACE VIEW spatial_shuffle_pair_interactions AS SELECT * FROM read_csv_auto('data_analysis/spatial_shuffle_pair_interactions.csv', header = true);
CREATE OR REPLACE VIEW spatial_shuffle_summary AS SELECT * FROM read_csv_auto('data_analysis/spatial_shuffle_summary.csv', header = true);
CREATE OR REPLACE VIEW technical_preflight_validity AS SELECT * FROM read_csv_auto('data_analysis/technical_preflight_validity.csv', header = true);

-- Core hypothesis-facing endpoint contrast. Severity is coded 0, 0.5, 1;
-- therefore the semantic-minus-episodic severity-slope interaction equals the
-- high-minus-baseline endpoint difference in baseline-referenced values.
CREATE OR REPLACE VIEW expanded_bilateral_interactions AS
SELECT *
FROM main_pair_interactions
WHERE pair = 'expanded_bilateral';

-- Pulse has no FC estimand in the locked notebook. FC is periodic-only; pulse
-- uses relative latency, peak timing/magnitude, energy, and tail completeness.
CREATE OR REPLACE VIEW pulse_node_metrics AS
SELECT *
FROM main_node_metrics
WHERE probe = 'pulse';

-- Prespecified within-run measurement eligibility.
CREATE OR REPLACE VIEW final_outcome_eligibility AS
SELECT *, valid_rows::DOUBLE / NULLIF(required_rows, 0) AS valid_fraction
FROM outcome_eligibility;

-- Final integration-step eligibility supersedes the earlier status embedded in
-- primary_interaction_statistics.csv.
CREATE OR REPLACE VIEW dt_precision_ratio AS
SELECT *, absolute_difference / NULLIF(tolerance, 0) AS tolerance_ratio
FROM integration_step_outcome_eligibility;

-- Post-hoc sensitivity outputs. These are regenerated only from saved result
-- tables and lossless trace shards; the analysis performs zero TVB calls.
CREATE OR REPLACE VIEW posthoc_transmission_endpoint AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/transmission_endpoint.csv', header = true);
CREATE OR REPLACE VIEW posthoc_frequency_quality AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/frequency_quality.csv', header = true);
CREATE OR REPLACE VIEW posthoc_segment_frequency_audit AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/segment_trace_summary.csv', header = true);
CREATE OR REPLACE VIEW posthoc_fc_trajectory AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/fc_trajectory.csv', header = true);
CREATE OR REPLACE VIEW posthoc_phase_fc AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/phase_fc_rows.csv', header = true);
CREATE OR REPLACE VIEW posthoc_phase_fc_summary AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/phase_fc_summary.csv', header = true);
CREATE OR REPLACE VIEW posthoc_spectra AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/spectra_summary.csv', header = true);
CREATE OR REPLACE VIEW posthoc_spectral_peaks AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/spectral_peak_summary.csv', header = true);
CREATE OR REPLACE VIEW posthoc_pulse_fixed_masks AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/pulse_fixed_masks.csv', header = true);
CREATE OR REPLACE VIEW posthoc_pulse_fixed_summary AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/pulse_fixed_summary.csv', header = true);
CREATE OR REPLACE VIEW posthoc_regional_coefficients AS SELECT * FROM read_csv_auto('data_analysis/investigation/recommended/regional_covariate_coefficients.csv', header = true);
