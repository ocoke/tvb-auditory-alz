# Modeling Speech- and Music-associated Auditory Network Connectivity Across Increasing Alzheimer’s-Like Amyloid Perturbation Using The Virtual Brain

Boston University Research in Science and Engineering (RISE), Computational Neurobiology.

> Grace O'Leary¹, Ellian Darlow², Shirley Ni³, Junxin Yu⁴ ¹Archbishop Mitty High School, 5000 Mitty Avenue, San Jose, CA 95129; ²Arlington High School, Arlington, MA 02476; ³North Shore Country Day School, Winnetka, IL 60093; ⁴Portola High School, 1001 Cadence, Irvine, CA 92618

Code repository and notebook created by Junxin Yu.




This is a runnable Python project for the 379-region TVB experiment originally
implemented in `RISE_TVB379_Complete_Experiment.ipynb`. It studies how an
AD-like, amyloid-linked change in inhibitory dynamics affects simulated
stimulus transmission from bilateral primary auditory cortex (A1) to
predeclared music-associated and speech-associated proxy subnetworks.

The project runs directly from this repository:

```bash
python main.py
```

There is no project package to install. In particular, do **not** run
`pip install .` or use an editable install. The only installation step is
installing the pinned third-party dependencies.

> `python main.py` starts the full `final` experiment immediately, without an
> interactive confirmation. It performs 100 spatial shuffles and is expected
> to take roughly 8.5 hours on comparable CPU hardware. Use `smoke` for an
> initial technical check.

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
| `smoke` | Technical end-to-end check using seed 11, baseline and high endpoints, 40 matched controls, and one spatial shuffle. Expect roughly 30 minutes on comparable hardware. It is not adequate for scientific interpretation. |
| `pilot` | Intermediate check using seeds 11 and 23, all three perturbation levels, 200 matched controls, and two spatial shuffles. |
| `final` | Full experiment using five numerical seeds, all three perturbation levels, 500 matched controls, four parameter-sensitivity scenarios, and 100 spatial shuffles. This is the default and makes 418 TVB calls including calibration. |

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
├── results/                CSV tables, metadata, and figures
├── run.log                 timestamped execution log
├── run_manifest.json       resolved configuration and provenance
├── run_status.json         running/interrupted/failed/completed state
└── RISE_TVB379_results_<mode>.zip
```

The manifest records the resolved configuration, exact Python and dependency
versions, source hashes, and a fingerprint of the experiment code. Status and
completion files are written atomically, with completion recorded last.

The workflow checkpoints after bounded units of work, including calibration
couplings, condition/seed grid blocks, the integration-step check, sensitivity
blocks, and individual spatial shuffles. After an interruption, explicitly
resume the incomplete run:

```bash
python main.py --resume /path/to/runs/20260728T180000Z_final_ab12cd34
```

`--resume` cannot be combined with `--mode`, `--output-root`, `--data-dir`, or
`--offline`. Resume refuses:

- an already completed run;
- changed experiment code or resolved configuration;
- changed verified input hashes;
- a different Python version; or
- different dependency versions.

Only a fully written checkpoint is accepted, so a partially written work unit
is rerun while earlier completed units remain reusable.

## Workflow

The stages run in this order:

1. Baseline-only coupling calibration
2. Main full-field experiment
3. Required 1.0 ms versus 0.5 ms integration-step convergence gate
4. Local-dynamics-held-baseline counterfactual
5. Topology/pathology-matched control subnetworks
6. Coupling and stimulus-strength sensitivity
7. Within-anatomical-block spatial shuffles
8. Figures, tables, metadata, and final ZIP export

The convergence gate is deliberately early. It stops the run if either target
transfer differs by 5% or more, avoiding hours of downstream computation from a
numerically unacceptable main integration step.

Matplotlib uses a noninteractive backend. Figures are written to disk rather
than displayed, so the project can run in a terminal or headless environment.

## Results

A completed run preserves the notebook's 23 CSV outputs:

```text
source_manifest.csv
data_quality_checks.csv
roi_definitions.csv
roi_pathology_values.csv
pathology_summary.csv
baseline_coupling_calibration.csv
main_node_metrics.csv
main_network_metrics.csv
main_network_metrics_normalized.csv
main_music_minus_speech_contrasts.csv
main_stage_summary.csv
local_fixed_node_metrics.csv
local_fixed_network_metrics.csv
local_fixed_contrasts.csv
matched_control_sets.csv
matched_control_null_metrics.csv
matched_control_null_summary.csv
sensitivity_network_metrics.csv
sensitivity_contrasts.csv
spatial_shuffle_network_metrics.csv
spatial_shuffle_contrasts.csv
integration_step_check.csv
run_manifest.csv
```

It also writes `experiment_metadata.json` and seven PNG figures:

```text
01_baseline_coupling_calibration.png
02_main_stage_curves.png
03_primary_endpoint_contrast.png
04_local_dynamics_counterfactual.png
05_matched_control_null.png
06_parameter_sensitivity.png
07_spatial_placement_sensitivity.png
```

The final ZIP contains the scientific outputs, metadata, run log, resolved
configuration/environment provenance, and verified input copies. Internal
restart checkpoints are intentionally excluded.

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
- The small, equal-sized parcel groups are proxy subnetworks, not complete or
  exclusive music and speech pathways.
- The 2 Hz and 5 Hz probes are temporal envelopes, not recordings of music or
  speech; they omit melody, pitch, timbre, language, meaning, familiarity, and
  emotion.
- The model contains no memory mechanism and cannot test preservation of
  musical memory.
- Numerical seeds, control sets, and shuffle percentiles are simulation
  diagnostics, not human subjects or clinical p-values.

The unchanged source notebook is preserved at
`notebooks/RISE_TVB379_Complete_Experiment.ipynb` for auditability. The Python
project, not the notebook, is the supported command-line runner.

## Development checks

Install the separately pinned test dependency and run the tests from the
project root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The test suite uses lightweight deterministic substitutes where possible.
A real `smoke` experiment remains a substantial integration check; the
approximately 8.5-hour `final` workload is not intended to run as part of the
normal test suite.

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
- Maier-Hein, K. H., et al. (2017). “The challenge of mapping the human
  connectome based on diffusion tractography.”
  [Nature Communications, 8, 1349](https://doi.org/10.1038/s41467-017-01285-x).
