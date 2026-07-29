# Modeling Speech- and Music-associated Auditory Network Connectivity Across Increasing Alzheimer’s-Like Amyloid Perturbation Using The Virtual Brain

Boston University Research in Science and Engineering (RISE), Computational Neurobiology.

> Grace O'Leary¹, Ellian Darlow², Shirley Ni³, Junxin Yu⁴ ¹Archbishop Mitty High School, 5000 Mitty Avenue, San Jose, CA 95129; ²Arlington High School, Arlington, MA 02476; ³North Shore Country Day School, Winnetka, IL 60093; ⁴Portola High School, 1001 Cadence, Irvine, CA 92618

Code repository and notebook created by Junxin Yu.




This is the runnable Python-project form of
`RISE_TVB379_Complete_Experiment_Semantic_Episodic_Parallel.ipynb`. It studies
how an AD-like, amyloid-linked change in inhibitory dynamics affects simulated
stimulus transmission from bilateral primary auditory cortex (A1) through
explicit shared-auditory, music-associated, and speech-associated proxy
pathways. A prespecified secondary analysis compares semantic-task-associated
and episodic-task-associated musical-memory proxy parcels.

The project runs directly from this repository:

```bash
python main.py
```

There is no project package to install. In particular, do **not** run
`pip install .` or use an editable install. The only installation step is
installing the pinned third-party dependencies.

> `python main.py` starts the `final`-mode experiment immediately, without an
> interactive confirmation. It is configured for 100 spatial shuffles if the
> required convergence gate passes. It uses one process by default. Opt in to
> multiprocessing with `--workers auto` or `--workers N`. Wall-clock time
> depends strongly on the machine; the runner prints elapsed time and an
> estimated time remaining as measurements become available.

## Requirements and one-time setup

- A user-managed CPython 3.12 installation
- Internet access on the first online run, unless the five verified inputs are
  already available locally
- Enough disk space for downloaded inputs, checkpoints, tables, figures, and
  the result archive

From the project root, create your own virtual environment and install only the
dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The runtime versions are pinned in `requirements.txt`:

- NumPy 2.0.2
- pandas 2.2.2
- SciPy 1.16.3
- Matplotlib 3.11.1
- `tvb-library` 2.10.0
- `tvb-data` 3.0.0

## Running the experiment

The default command starts a new final run:

```bash
python main.py
```

Select a diagnostic workload with `--mode`:

```bash
python main.py --mode smoke
python main.py --mode pilot
python main.py --mode final
```

| Mode | Purpose and workload |
| --- | --- |
| `smoke` | Gate-reaching technical diagnostic using seed 11, baseline and high endpoints, 40 matched controls per comparison, and one spatial shuffle. Its planned maximum is 33 TVB calls, 31 of which appear in the run manifest. It is not adequate for scientific interpretation. |
| `pilot` | Intermediate diagnostic using seeds 11 and 23, all three perturbation levels, 200 matched controls per comparison, and two spatial shuffles. Its planned maximum is 70 TVB calls, with 64 manifested. |
| `final` | Full experiment using five numerical seeds, all three perturbation levels, 500 matched controls per comparison, four parameter-sensitivity scenarios, and 100 spatial shuffles. This is the default; its planned maximum is 442 TVB calls, with 430 manifested and 12 separately recorded calibration calls. |

Those are resolved workloads, not a promise that every call will execute. All
three modes stop after stage 3 if the required convergence gate fails.

Numerical seeds probe sensitivity to initial conditions. They are repeated model
runs, not participants or independent biological samples.

Use custom run and input locations as needed:

```bash
python main.py --mode smoke \
  --output-root /path/to/experiment-runs \
  --data-dir /path/to/tvb379-inputs
```

The defaults are the project-local `runs/` and `data/` directories. Paths are
resolved independently of the shell's current directory.

Show the complete command-line help with:

```bash
python main.py --help
```

### CPU use, progress, and timing

The default is deliberately single-process:

```bash
python main.py --mode smoke --workers 1
```

Parallel execution is optional:

```bash
python main.py --mode smoke --workers auto  # all CPUs allocated to this job
python main.py --mode smoke --workers 4     # at most four worker processes
```

TVB work here is CPU-bound, so `--workers` uses independent **processes**, not
Python threads. `auto` respects process affinity and common scheduler limits;
an explicit count above the detected allocation is capped. Each worker uses
one native numerical thread, preventing nested BLAS/OpenMP oversubscription.
Active CPU use is also bounded by the independent blocks ready in a stage.

Checkpoint loading and writing remain in the parent process. Completed worker
results are aggregated in their declared scientific order, rather than worker
completion order, so parallel scheduling does not change table ordering or
checkpoint identity.

Progress messages are written both to the terminal and `run.log`. They identify
the current stage and work unit and report completed/total TVB calls, percentage
complete, elapsed time, and an ETA once enough current-run measurements exist.
On resume, already verified checkpoints are reported as restored work.
`--workers` may be changed for a resume because it affects execution only, not
the scientific configuration. The ETA is an estimate and can change as the
workflow moves between 0.5 ms, 0.25 ms, and shorter calibration simulations.

The completion summary distinguishes:

- **command elapsed time before the result-archive snapshot**, measured from
  command entry and including setup, input verification, computation, and
  output generation up to that snapshot; and
- **aggregate simulation time**, the sum of all individual TVB simulation
  durations, including calibration and simulations that ran concurrently.

Because simulations overlap across workers, aggregate simulation time can be
greater than command elapsed time. The archived `run.log` contains entries
through the archive snapshot; the on-disk log also receives the final archive
completion message.

## Offline use

An online run downloads missing inputs to a temporary file, verifies the
pinned SHA-256 digest, and only then places the file in `--data-dir`. Transient
downloads are retried. Existing files are also verified before use; corrupt or
unexpected content is never silently accepted.

For a network-free run, put all five files below in `--data-dir` and use
`--offline`:

```bash
python main.py --mode smoke --offline
```

| File | Expected SHA-256 |
| --- | --- |
| `avg_healthy_normSC_mod.txt` | `141fc993c84bde0b2f0ee0280ce1ccc47e1731ddcbf37845a4eef38dad9fa562` |
| `AD_LH.txt` | `566e770e93f50d3378a0cf7d2dc8b1fa5af9475ca432463acf7bc2b5507907af` |
| `AD_RH.txt` | `4f42c31c08e6d191d415953f763889f816d9c2443d115612f0c076cfd5b2f129` |
| `AD_subcortical.txt` | `f28926c7955db2c2762b5ba032f0bbde770a0da3d3705dd037a746382506d824` |
| `region_labels.txt` | `f9688592130a034210b482a0556fdc14383eaaf578bef39d2ddba072537e3484` |

Offline mode fails clearly if any file is absent or does not match its digest.
The loader also verifies the 379-by-379 structural matrix, 379 labels, the
180/180/19 amyloid-vector split, parcel anchors, and shared region ordering.
Verified inputs are copied into each run directory for provenance.

## Run directories and recovery

Every new run receives a unique directory:

```text
<output-root>/<UTC timestamp>_<mode>_<digest>/
├── inputs/                 verified source-file copies
├── checkpoints/            restart state for bounded work units
├── attempts/               environment details for each execution attempt
├── results/                CSV tables, metadata, and figures
├── run.log                 timestamped execution log
├── run_manifest.json       resolved configuration and provenance
├── run_status.json         running/interrupted/failed/completed state
└── RISE_TVB379_results_<mode>.zip
```

The manifest records the resolved configuration, initial exact environment,
source hashes, and a fingerprint of the experiment code. Each new or resumed
attempt also gets a separate environment record under `attempts/`, including
its host, CPU availability, and worker settings. Status and completion files
are written atomically, with completion recorded last.

The workflow checkpoints after bounded units of work, including calibration
couplings, condition/seed grid blocks, integration-step convergence blocks,
sensitivity blocks, and individual spatial shuffles. The parent process writes
each completed block atomically after receiving it from a worker. After an
interruption, explicitly resume the incomplete run:

```bash
python main.py --resume /path/to/runs/20260728T180000Z_final_ab12cd34
python main.py --resume /path/to/runs/20260728T180000Z_final_ab12cd34 \
  --workers auto
```

`--resume` cannot be combined with `--mode`, `--output-root`, `--data-dir`, or
`--offline`. Resume refuses:

- an already completed run;
- changed experiment code or resolved configuration;
- changed verified input hashes;
- a different Python version; or
- different dependency versions.

Only a fully written checkpoint is accepted. An in-flight or partially written
work unit is rerun, while every earlier completed unit remains reusable.

## Workflow

The stages run in this order:

1. Baseline-only coupling calibration
2. Main full-field experiment
3. Required 0.5 ms versus 0.25 ms integration-step convergence gate
4. Separate primary-pathway and memory-proxy local-dynamics counterfactuals
5. Separate topology/pathology-matched primary and memory control subnetworks
6. Coupling and stimulus-strength sensitivity
7. Within-anatomical-block spatial shuffles
8. Figures, tables, metadata, and final ZIP export

The convergence gate is deliberately early. It recomputes the baseline and
high AD-like endpoints at both 2 Hz and 5 Hz using 0.25 ms, then compares them
with the main 0.5 ms results. The five inferential networks are music, speech,
semantic-task-associated, episodic-task-associated, and shared auditory relay.
Every declared context network is also reported. Each row includes transfer
and median target-fit R² at both steps, relative transfer difference, and the
absolute R² difference. A relative difference of 5% or more stops the run for
an inferential network. A descriptive context-network failure is preserved and
warned about but does not change the prespecified inferential gate.

Matplotlib uses a noninteractive backend. Figures are written to disk rather
than displayed, so the project can run in a terminal or headless environment.

## Results

A completed run preserves the notebook's 31 CSV outputs:

```text
source_manifest.csv
data_quality_checks.csv
roi_definitions.csv
music_memory_peak_mapping.csv
roi_pathology_values.csv
pathology_summary.csv
baseline_coupling_calibration.csv
main_node_metrics.csv
main_network_metrics.csv
main_network_metrics_normalized.csv
main_music_minus_speech_contrasts.csv
main_semantic_minus_episodic_contrasts.csv
main_stage_summary.csv
local_fixed_node_metrics.csv
local_fixed_network_metrics.csv
local_fixed_contrasts.csv
memory_counterfactual_comparison.csv
matched_control_sets.csv
matched_control_null_metrics.csv
matched_control_null_summary.csv
memory_matched_control_sets.csv
memory_matched_control_null_metrics.csv
memory_matched_control_null_summary.csv
sensitivity_network_metrics.csv
sensitivity_contrasts.csv
spatial_shuffle_network_metrics.csv
spatial_shuffle_contrasts.csv
spatial_shuffle_summary.csv
memory_spatial_shuffle_summary.csv
integration_step_check.csv
run_manifest.csv
```

It also writes `experiment_metadata.json` and ten PNG figures:

```text
01_baseline_coupling_calibration.png
02_main_stage_curves.png
03_primary_endpoint_contrast.png
04_local_dynamics_counterfactual.png
05_matched_control_null.png
06_parameter_sensitivity.png
07_spatial_placement_sensitivity.png
08_semantic_episodic_secondary_analysis.png
09_semantic_episodic_matched_null.png
10_semantic_episodic_robustness.png
```

The final ZIP contains the scientific outputs, metadata, run log, resolved
configuration/environment provenance, and verified input copies. Internal
restart checkpoints are intentionally excluded.

## Analyzing a completed final run

`analyze_results.py` is a read-only post-processing command for an existing
completed run. It does not import TVB or execute any simulations, and it never
changes the original result tables or experiment figures. By default it reads
the supplied final run and writes supplemental analysis beneath that run:

```bash
python analyze_results.py
python analyze_results.py \
  --run-dir runs/RISE_TVB379_results_final
```

The default produces both a concise publication/poster set and a denser
technical-QA set in PNG and SVG at 300 DPI. Select an audience, image format,
resolution, or separate destination when needed:

```bash
python analyze_results.py --audience publication --formats svg
python analyze_results.py --audience technical --formats png --dpi 200
python analyze_results.py \
  --run-dir /path/to/completed-run \
  --output-dir /path/to/analysis
```

The output layout is:

```text
analysis/
├── publication/figures/       eight paper/poster figures
├── technical_qa/figures/      six detailed diagnostic figures
├── tables/                    eleven derived CSV summaries
├── interpretation_findings.csv
└── figure_manifest.csv
```

The derived tables cover completeness and scientific-key validation, component
and contrast trajectories, high endpoints, local-fixing attenuation, matched
controls, spatial shuffles, parameter sensitivity, full-network integration
step convergence, harmonic-fit quality, calibration, and runtime/worker
diagnostics. `interpretation_findings.csv` keeps each finding, supporting
evidence, caveat, and source table in separate columns. The command also prints
separate `Scientific findings` and `Technical cautions` sections; it does not
create a Markdown or HTML report.

Before creating the output directory, the analyzer verifies required files and
columns, finite analysis values, unique scientific keys, configured
seeds/probes/severities, 100 final shuffles, 430 manifested simulations, 442
total TVB calls, the PSP safety bound, and the prespecified inferential
integration-step convergence gate. An incomplete or failed run is rejected
before plotting.

The supplemental analysis preserves the experiment's interpretation limits.
In particular, component changes are reported separately from contrasts;
numerical seeds are not treated as participants; matched and shuffled ranks are
descriptive simulation diagnostics rather than clinical p-values; and collapse
under local fixing is described as model-internal dependence rather than
biological causality. A low harmonic-fit R² is prominently retained as a signal
quality warning, even when the independent integration-step convergence gate
passed.

## Scientific interpretation

The public high endpoint is an artificial amyloid surrogate derived from noise
and selected properties of ADNI-derived data. Baseline uses `b = 0.07`
throughout; the intermediate endpoint is exactly halfway to the transformed
high endpoint in model-parameter space. The intermediate condition is **not**
an MCI brain.

The primary 2 Hz and 5 Hz response is a fitted harmonic amplitude from the last
three seconds of the control-subtracted response. For each proxy network,
transfer is its mean target amplitude divided by the mean bilateral-A1
amplitude, then normalized to that network's baseline with a base-2 log ratio.
The primary contrast is:

```text
music baseline-normalized log2 transfer
    - speech baseline-normalized log2 transfer
```

A positive value means that the music proxy changed more favorably than the
speech proxy **inside this model**. It does not demonstrate preserved music
pathways, musical memory, or clinical function.

The prespecified secondary contrast is:

```text
semantic-task-associated baseline-normalized log2 transfer
    - episodic-task-associated baseline-normalized log2 transfer
```

Those parcels are approximate HCP-MMP mappings of nine SPM99 peaks reported by
Platel et al. (2003). The mapping is exported in
`music_memory_peak_mapping.csv`. This is a separate mechanistic proxy analysis,
not a behavioral memory experiment.

Interpret results in context:

1. Check main-stage changes and consistency across 2 Hz, 5 Hz, and seeds.
2. Check whether the direction persists when A1 and target local inhibitory
   dynamics are held at baseline.
3. Compare the predeclared targets with topology/pathology-matched parcel sets.
4. Check coupling and stimulus-strength sensitivity.
5. Check dependence on the exact spatial placement of the artificial
   perturbation.

Important limitations include:

- This is a mechanistic simulation of one proposed AD-related process, not a
  clinical HC/MCI/AD comparison or a disease-progression model.
- The amyloid-to-inhibitory-rate mapping is a modeling hypothesis, not an
  established biological law.
- Tau, atrophy, synaptic loss, neuroinflammation, vascular disease, and
  white-matter degeneration are absent.
- Every condition uses the same averaged healthy structural connectome.
- Interregional delays are zero, excluding realistic latency and phase claims.
- Tractography can omit or falsely identify connections and does not establish
  biological direction.
- The parcel groups are unequal-sized operational proxy subnetworks, not
  complete or exclusive biological pathways. Matched controls preserve each
  declared group size and hemispheric composition.
- The 2 Hz and 5 Hz probes are temporal envelopes, not recordings of music or
  speech; they omit melody, pitch, timbre, language, meaning, familiarity, and
  emotion.
- The model contains no memory mechanism and cannot test preservation of
  musical memory.
- Numerical seeds, control sets, and shuffle percentiles are simulation
  diagnostics, not human subjects or clinical p-values.

The source notebooks are retained unchanged for auditability:

- `notebooks/RISE_TVB379_Complete_Experiment_Semantic_Episodic_Parallel.ipynb`
  is the current scientific specification.
- `notebooks/RISE_TVB379_Complete_Experiment.ipynb` is the earlier experiment.

The Python project is the supported command-line runner.

## Development checks

Install the separately pinned test dependency and run the tests from the
project root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The test suite uses lightweight deterministic substitutes where possible.
Even `--mode smoke` is a substantial 33-call gate-reaching integration
diagnostic. The planned 442-call `final` workload is not intended to run as
part of the normal test suite; its actual elapsed time depends on allocated
CPUs, memory bandwidth, host load, and whether the convergence gate passes.

## Primary references

- Stefanovski, L., et al. (2019). “Linking molecular pathways and large-scale
  computational modeling to assess candidate disease mechanisms and
  pharmacodynamics in Alzheimer's disease.”
  [Frontiers in Computational Neuroscience, 13, 54](https://doi.org/10.3389/fncom.2019.00054).
- BrainModes.
  [TVB Educase AD molecular pathways](https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB),
  public educational model and surrogate data.
- BrainModes.
  [ADNI-TVB pipeline](https://github.com/BrainModes/ADNI-TVB-pipeline),
  source of the 379-region label order.
- Glasser, M. F., et al. (2016). “A multi-modal parcellation of human cerebral
  cortex.” [Nature, 536, 171–178](https://doi.org/10.1038/nature18933).
- Ding, N., et al. (2017). “Temporal modulations in speech and music.”
  [Neuroscience & Biobehavioral Reviews, 81, 181–187](https://doi.org/10.1016/j.neubiorev.2017.02.011).
- Jacobsen, J.-H., et al. (2015). “Why musical memory can be preserved in
  advanced Alzheimer's disease.”
  [Brain, 138, 2438–2450](https://doi.org/10.1093/brain/awv135).
- Hickok, G., and Poeppel, D. (2007). “The cortical organization of speech
  processing.”
  [Nature Reviews Neuroscience, 8, 393–402](https://doi.org/10.1038/nrn2113).
- Norman-Haignere, S., Kanwisher, N. G., and McDermott, J. H. (2015).
  “Distinct cortical pathways for music and speech revealed by hypothesis-free
  voxel decomposition.”
  [Neuron, 88, 1281–1296](https://doi.org/10.1016/j.neuron.2015.11.035).
- Platel, H., et al. (2003). “Semantic and episodic memory of music are
  subserved by distinct neural networks.”
  [NeuroImage, 20, 244–256](https://doi.org/10.1016/S1053-8119(03)00287-8).
- Slattery, C. F., et al. (2019). “The functional neuroanatomy of musical
  memory in Alzheimer’s disease.”
  [Cortex, 115, 357–370](https://doi.org/10.1016/j.cortex.2019.02.003).
- Maier-Hein, K. H., et al. (2017). “The challenge of mapping the human
  connectome based on diffusion tractography.”
  [Nature Communications, 8, 1349](https://doi.org/10.1038/s41467-017-01285-x).
