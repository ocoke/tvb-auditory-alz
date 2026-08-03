# TVB379 result investigation tool

`investigate_results.py` validates and explores this completed result export.
It reads the saved CSV/JSON/NPZ files and never reruns TVB or modifies the
experiment outputs. Derived material is written only under `investigation/` by
default.

## Self-contained HTML investigator

Open `TVB379_visual_investigator.html` directly in a browser. It is a portable,
read-only dashboard with embedded data, charts, detail tables, filters, source
metadata, and a no-script semantic fallback. It does not need a local server,
CDN, Python process, or sibling data files after it has been built.

The dashboard reviews every top-level experiment CSV and JSON file, the run
log, and the existing experiment figure assets. It also reads all 180 lossless
NPZ trace shards for the post-hoc transmission, spectral, phase, and fixed-mask
pulse audits. Only bounded derived summaries are embedded; raw arrays are not.

Regenerate it after replacing or updating the export:

```bash
python data_analysis/build_html_investigator.py
```

The reproducible source files are:

- `build_html_investigator.py`: loads, validates, summarizes, and constructs
  the canonical dashboard artifact;
- `recommended_posthoc_analysis.py`: reproduces saved segment metrics and adds
  broadband-versus-locked, spectral, phase–FC, fixed-mask pulse, and regional
  covariate sensitivity analyses without importing TVB;
- `html_investigator_sources.sql`: equivalent DuckDB file-source map;
- `package_html_investigator.mjs`: packages the canonical offline reader and
  applies scoped desktop-scrollbar and mobile-legend containment fixes;
- `html_investigator/artifact.json`: reviewed embedded snapshot used to build
  the final HTML.

Pulse functional connectivity is shown as **not applicable**, not failed or
missing. In the locked experiment FC is defined only for the periodic probes;
the pulse investigation uses latency, peak magnitude, peak timing, evoked
energy, parcel coverage, and tail-energy completeness.

The post-hoc audit can also be run independently:

```bash
python data_analysis/recommended_posthoc_analysis.py
```

Its derived CSVs are written under `investigation/recommended/`. The audit
uses the notebook's exact detrended-RMS, harmonic-fit, wrapped-phase,
multitaper, and cumulative-energy definitions. It makes zero TVB calls and
does not modify the notebook or original result files.

The principal post-hoc interpretation is that the semantic proxy has greater
broadband evoked-response amplification. Although the fitted-frequency ratio
also increases, perturbed target harmonic R² and frequency-QA coverage are too
poor to claim stronger preservation of the applied 2 or 5 Hz frequency.

Use the project Python 3.12 environment:

```bash
python data_analysis/investigate_results.py overview
```

That produces six research figures in PNG and SVG, hypothesis statistics,
deterministic interpretation findings, a data-quality report, raw-trace
descriptive statistics, and a figure manifest.

Useful investigation commands:

```bash
# Validate files, scientific keys, eligibility, and trace headers.
python data_analysis/investigate_results.py validate

# Also re-hash all 180 lossless trace shards (slower).
python data_analysis/investigate_results.py validate --verify-trace-hashes

# Inspect one hypothesis-facing outcome.
python data_analysis/investigate_results.py stats \
  --outcome transfer_gain --probe 2Hz --severity 1.0 --save

# Plot default bilateral-A1, semantic-proxy, and episodic-proxy trace means.
python data_analysis/investigate_results.py trace \
  --seed 11 --probe 2Hz --severities 0 1

# Plot particular saved parcels instead.
python data_analysis/investigate_results.py trace \
  --seed 23 --probe pulse --severities 0 0.5 1 \
  --region L_A1 --region L_9m --region L_31pd

# Inventory result table schemas.
python data_analysis/investigate_results.py tables --contains interaction
```

Global source/output overrides go before the command:

```bash
python data_analysis/investigate_results.py \
  --data-dir /path/to/results --output-dir /path/to/derived overview
```

## Interpretation rules built into the tool

- A component change is not the same thing as the
  semantic-minus-episodic interaction.
- A positive interaction can occur when both components increase or when both
  decline; the component trajectories are always reported alongside it.
- Final eligibility comes from `outcome_eligibility.csv`. This supersedes the
  earlier status labels embedded in `primary_interaction_statistics.csv`.
- Numerical seeds are not participants or biological replicates. Reported
  intervals and p-values describe numerical-initialization variation only.
- Local-dynamics-fixed counterfactuals indicate model-internal dependence, not
  biological causality.
- Matched-control and spatial-shuffle distributions are simulation references,
  not clinical p-values.
- Raw PSP values are model variables, not measured voltage. Zero tract delays
  limit anatomical interpretation of response timing.
