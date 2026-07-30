# RISE TVB379 ScienceReady semantic-versus-episodic experiment

This repository runs the final 379-region TVB semantic-versus-episodic
musical-memory proxy experiment from
[`RISE_TVB379_Semantic_Episodic_Final_ScienceReady_20260729.ipynb`](notebooks/RISE_TVB379_Semantic_Episodic_Final_ScienceReady_20260729.ipynb).

The notebook is the single scientific source. `main.py` validates its exact
identity, compiles all 18 code cells, and executes those cells in their
original order in one shared namespace. The Python wrapper does not duplicate
or modify the perturbation, simulation, transfer, functional-connectivity,
latency, matching, shuffle, sensitivity, or statistical formulas. Recovery is
implemented around the notebook's existing parallel work-unit dispatcher, not
inside those scientific calculations.

This is a directly runnable project, not an installable library. Do not run
`pip install .` or use an editable install.

## Setup

Use CPython 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The TVB versions are locked to `tvb-library==2.10.0` and
`tvb-data==3.0.0`. Numerical dependencies are listed in
`requirements.txt`, together with the notebook's joblib and IPython runtime
requirements.

## Run

With no arguments, the experiment starts the locked final workload:

```bash
python main.py
```

The command does not ask for confirmation. It uses all CPUs visible to the
process by default, with one native numerical thread per worker process:

```bash
python main.py --workers auto
python main.py --workers 8
python main.py --workers 1
```

`--workers 1` is the sequential path. Explicit counts are capped at the CPU
allocation detected by the notebook. Independent blocks run through joblib's
loky process backend; returned results retain their declared scientific order.

Diagnostic modes use the same notebook code path:

```bash
python main.py --mode smoke --workers 2
python main.py --mode pilot --workers auto
python main.py --mode final --workers auto
```

| Mode | Calibration | Main | Integration step | Local fixed | Parameters | Shuffles | Total TVB calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke` | 2 | 8 | 8 | 4 | 6 | 6 | 34 |
| `pilot` | 6 | 24 | 8 | 8 | 24 | 15 | 85 |
| `final` | 12 | 240 | 24 | 80 | 120 | 150 | **626** |

Final mode uses 20 paired numerical initializations, 50 spatial shuffles,
500 matched parcel-set controls, three severity values, and pulse/2 Hz/5 Hz
probes. Of its 626 calls, 614 are in `run_manifest.csv`; the 12 calibration
calls are recorded separately because each calibration row contains a control
and pulse simulation.

The five pinned input files in `data/` are used as a verified local cache by
default. Select another cache or result name with:

```bash
python main.py --data-cache /path/to/verified/files
python main.py --run-id my_final_run
```

Missing cache files are downloaded by the notebook and checked against its
pinned SHA-256 values. Results are written under:

```text
rise_tvb379_work/
├── source_data/
├── results_semantic_episodic_v3_<mode>/
└── RISE_TVB379_results_semantic_episodic_v3_<mode>.zip
```

If a result directory already contains files, the notebook creates a
timestamped directory instead of overwriting it.

## Checkpoints, resume, and run status

Every completed calibration candidate and condition/seed block is serialized
immediately through a same-filesystem temporary file and atomic replacement.
This covers the expensive main, integration-step, local-counterfactual,
parameter, and individual spatial-shuffle work units. A partially written or
incompatible checkpoint is ignored and recomputed. Each payload is verified by
size and SHA-256, and its completion marker is written last.

Checkpoints are stored outside the scientific result directory so they are not
included in the result ZIP:

```text
rise_tvb379_work/
├── results_<run-id>/
│   ├── run_status.json
│   └── progress.log
└── .science_ready_checkpoints/
    └── results_<run-id>/
```

If a run is interrupted or fails, use the exact result directory printed at
startup:

```bash
python main.py --resume \
  rise_tvb379_work/results_semantic_episodic_v3_final
```

Worker count may be changed on resume:

```bash
python main.py --resume /path/to/results_dir --workers 8
```

Resume reruns the notebook's setup and deterministic table/figure aggregation,
but restores completed TVB work units and executes only missing or invalid
ones. It refuses a completed run or a mismatch in the canonical notebook,
Python execution code, resolved workload, Python version, dependency versions,
or verified input hashes.

Inspect progress without starting the experiment:

```bash
python main.py --status /path/to/results_dir
```

`run_status.json` is atomically updated after every durable work unit and
records the state, attempt, current stage, completed/planned TVB calls,
restored/executed calls for the current attempt, environment, input hashes,
completed source cells, timestamps, and any error. `progress.log` contains the
same stage-level progress messages, percentages, elapsed time, and ETA shown in
the terminal.

## Progress and static smoke check

The terminal and `progress.log` print every notebook-cell boundary. Within long
stages they print each saved or restored work unit, stage counts, overall TVB
calls, percentage, elapsed time, and an ETA after newly executed work is
available. Worker processes use private TVB and Matplotlib runtime directories.

Validate the notebook identity, 40-cell/18-code-cell structure, compilation,
ScienceReady metadata, 6000 ms pulse window, and all locked workload counts
without importing TVB or starting a simulation:

```bash
python main.py --check
```

Run the focused project tests with:

```bash
python -m pytest -q
```

Neither command performs a genuine TVB experiment. Even smoke mode runs 34 TVB
calls and is therefore not part of routine conversion validation.

## Scientific scope

The confirmatory comparison asks how increasing AD-like amyloid-linked
inhibitory perturbation differentially affects modeled transmission into:

- a 13-node expanded semantic-associated proxy; and
- a 19-node expanded episodic-associated proxy.

Each expanded definition is the union of an anatomically supported core and
the mapped musical-task peaks from Platel et al. The notebook separately
retains anatomical-core-only, Platel-only, left-only, and right-only
sensitivity analyses. The main outcomes are A1-normalized transfer gain,
stimulus-induced functional connectivity, and relative pulse-response latency.

The required preflight covers seeds 11 and 23 at baseline and high
perturbation. It applies the locked transfer, functional-connectivity, and
latency gates before the remaining main blocks. The integration-step check
compares 0.5 ms with 0.25 ms for seeds 11, 71, and 503 at both endpoints and
all three probes. Pulse analysis extends through 6000 ms.

## Outputs

A completed run writes 42 CSV tables, including:

- source, data-quality, parcel-definition, evidence, mapping, and pathology
  tables;
- calibration, node, network, normalized, interaction, and statistics tables;
- preflight, eligibility, A1 frequency, temporal, and integration-step QA;
- local-dynamics-fixed counterfactual outputs;
- matched-control sets, null metrics, and summaries;
- parameter and laterality sensitivity outputs;
- spatial-shuffle metrics and summaries; and
- the complete simulation manifest.

It also writes:

- `analysis_spec.json`
- `experiment_metadata.json`
- `run_status.json`
- `progress.log`
- six PNG figures under `figures/`
- a ZIP archive of the result directory

The six figures cover calibration, primary metric trajectories, primary
interactions, the local-dynamics counterfactual, definition/laterality
sensitivity, and matched/spatial robustness.

## Interpretation boundaries

- Numerical initializations are not participants or biological replicates.
- The downloadable amyloid endpoint is an artificial surrogate, not patient
  data.
- The model has no behavioral memory task, encoding, recollection,
  familiarity, or recognition outcome.
- Proxy parcel sets are operational definitions, not proven isolated
  pathways or complete memory systems.
- Structural weights are fixed and tract delays are zero. Pulse timing is a
  relative model-response measure, not anatomical conduction latency.
- Evoked PSP correlation is model functional connectivity, not BOLD
  functional connectivity or directed effective connectivity.
- Bilateral common input remains a possible contributor after unstimulated
  matched-control subtraction.
- Technical gates are prespecified numerical-quality rules, not biological
  diagnostic thresholds.
- The 2 Hz and 5 Hz inputs are temporal probes without melody, pitch, timbre,
  meaning, familiarity, emotion, learning, delay, or recognition.
- Spatial-shuffle ranks are descriptive; 50 shuffles do not support a precise
  formal permutation p-value.
- High-perturbation periodic responses may evolve across the analysis window.
  Interpret the fixed-window transfer together with the five saved segments
  and their log2 slope.

## Primary references

- Stefanovski, L., et al. (2019), *Frontiers in Computational Neuroscience*,
  13, 54. <https://doi.org/10.3389/fncom.2019.00054>
- Glasser, M. F., et al. (2016), *Nature*, 536, 171–178.
  <https://doi.org/10.1038/nature18933>
- Platel, H., et al. (2003), *NeuroImage*, 20, 244–256.
  <https://doi.org/10.1016/S1053-8119(03)00287-8>
- Slattery, C. F., et al. (2019), *Cortex*, 115, 357–370.
  <https://doi.org/10.1016/j.cortex.2019.02.003>
- Rolls, E. T., Wirth, S., et al. (2023), *Human Brain Mapping*, 44,
  629–655. <https://doi.org/10.1002/hbm.26089>
- BrainModes TVB Educase AD molecular pathways:
  <https://github.com/BrainModes/TVB_EducaseAD_molecular_pathways_TVB>
- BrainModes ADNI-TVB pipeline:
  <https://github.com/BrainModes/ADNI-TVB-pipeline>
