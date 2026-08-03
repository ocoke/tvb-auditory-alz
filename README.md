<h1 align="center">Modeling Music-associated Auditory Network Connectivity Across Increasing Alzheimer’s-Like Amyloid Perturbation Using The Virtual Brain</h1>

<p align="center">Grace O'Leary<sup>1,5</sup>, Ellian Darlow<sup>2,5</sup>, Shirley Ni<sup>3,5</sup>, Junxin Yu<sup>4,5</sup></p>

<p align="center">Archbishop Mitty High School, San Jose, CA<sup>1</sup>; Arlington High School, Arlington, MA<sup>2</sup>; North Shore Country Day School, Winnetka, IL<sup>3</sup>; Portola High School, Irvine, CA<sup>4</sup>; Boston University, Boston, MA<sup>5</sup></p>

## Abstract
Alzheimer’s disease (AD), characterized by gradual loss of memory and cognitive function, affects over 30 million people worldwide. Many patients are unable to recognize familiar concepts, people, and places, but recognize familiar music. Specifically, experiments have shown that despite deficits in episodic musical memory (e.g., the context in which music was heard), semantic musical memory (e.g., the melody) remains unimpaired; however, the neurological mechanisms underlying this phenomenon remain poorly understood. We hypothesize that changes in functional connectivity (FC) of music processing pathways induce differences in semantic and episodic musical memory capacity, with the episodic musical network regressing across healthy control (HC), mild cognitive impairment (MCI)-like, and AD-like groups, and the semantic musical network remaining comparatively stable across these same groups.. Plaque formation due to AD alters FC, particularly in brain regions associated with episodic memory, like the posterior cingulate cortex, postero-medial cortex, and precuneus. Regions associated with semantic memory–the anterior temporal, inferior, and supero-medial prefrontal cortices–develop AD later. To assess the changes in these networks as AD-like perturbation increases, we use the Virtual Brain (TVB) to simulate a 379-region brain network. Following a previous study, we transform a surrogate amyloid map to produce AD-like perturbation conditions in TVB. We stimulated primary auditory cortices bilaterally with a 100-ms constant-amplitude pulse, a 2 Hz signal, and a 5 Hz signal, maintaining total input power. We subtracted the unstimulated output, then normalized FC, transfer gain, and response latency per network. We reported the differences in FC, gain, and latency between the networks using interaction analysis. Across 20 paired initializations, we tested the semantic-versus-episodic interaction contrast against the null hypothesis using a two-sided one-sample t-test. This work has the potential to identify circuits that remain viable targets for music-based interventions in AD patients, providing potential avenues for novel treatment strategies.
 
## Setup

Use Python 3.12 and dependencies listed in [`requirements.txt`](https://github.com/ocoke/tvb-auditory-alz/blob/main/requirements.txt), venv is recommended.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run experiments

The direct entry point executes the unmodified canonical notebook
[`RISE_TVB379_Semantic_Episodic_Final_RawTraceExport_20260731.ipynb`](notebooks/RISE_TVB379_Semantic_Episodic_Final_RawTraceExport_20260731.ipynb).

The script uses all CPUs visible to the process by default, with one native numerical thread per worker process:

```bash
python main.py --workers auto
python main.py --workers 8 
```

Diagnostic modes, which are useful for debug, use the same notebook code path:

```bash
python main.py --mode smoke --workers 2
python main.py --mode pilot --workers auto
python main.py --mode final --workers auto
```


| Mode | Calibration | Main | Integration step | Local fixed | Parameters | Shuffles | Raw trace shards | Total TVB calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke` | 2 | 8 | 8 | 4 | 6 | 6 | 6 | 34 |
| `pilot` | 6 | 24 | 8 | 8 | 24 | 15 | 18 | 85 |
| `final` | 12 | 240 | 160 | 80 | 120 | 150 | **180** | **762** |


Final mode uses 20 paired numerical initializations, 50 spatial shuffles,
500 matched parcel-set controls, three severity values, and pulse/2 Hz/5 Hz
probes. Of its 762 calls, 750 are in `run_manifest.csv`; the 12 calibration
calls are recorded separately because each calibration row contains a control
and pulse simulation. The integration-step stage contains 40 endpoint
condition/seed work units and 160 TVB calls.

Final mode writes 180 losslessly compressed `.npz` shards under
`main_parcel_traces/`, one for every severity/seed/probe combination. Each
contains the untouched stimulated, control, and evoked PSP samples for the 34
declared parcels, the exact time and stimulus arrays, labels, indices, and run
parameters. The complete export also contains 48 CSV tables, including
`regional_features.csv` for all 379 regions, six PNG figures, metadata, and a
larger result ZIP. Plan disk space accordingly.

The integration-step gate preserves the notebook's fatal validity rules:
nonfinite output, failed A1 quality, direction reversal, or a changed 95%
interval conclusion still terminates the run. An exceeded exact-magnitude
precision target is recorded and warned about but does not terminate an
otherwise valid run.

### Extended late-window follow-up

The separate `main_extended.py` entry point executes the unmodified canonical
notebook
[`RISE_TVB379_Semantic_Episodic_Final_LateWindowFollowup_20260802.ipynb`](notebooks/RISE_TVB379_Semantic_Episodic_Final_LateWindowFollowup_20260802.ipynb),
whose locked SHA-256 is
`6c272953f8813c00af808a86c4c1aef1d8fa4c2acca3baa88947e498eab18c54`.
It preserves all 762 primary calls and adds the complete prolonged experiment
at 0.5 ms and 0.25 ms:

```bash
python main_extended.py --check
python main_extended.py --mode smoke --workers 2
python main_extended.py --mode final --workers auto \
  --run-id semantic_episodic_late_followup_final
```

| Mode | Unchanged primary | Prolonged 0.5 ms | Prolonged 0.25 ms | Manifested calls | Trace shards | Total TVB calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke` | 34 | 8 | 8 | 48 | 10 | **50** |
| `pilot` | 85 | 24 | 24 | 127 | 30 | **133** |
| `final` | 762 | 240 | 240 | 1,230 | 300 | **1,242** |

Final mode adds 60 condition/seed blocks at each prolonged integration step.
Each block contains four genuine TVB calls: zero input, DC matched, 2 Hz, and
5 Hz. The block is one atomic checkpoint, and progress advances by all four
calls only after its saved result is complete. The combined status plan has
100 integration-reference work units and 400 associated calls: the original
40-work-unit gate plus 60 prolonged 0.25 ms blocks.

The extended result contains 68 CSV tables, eight PNG figures, and 300 lossless
trace shards: 180 unchanged primary shards plus 60 prolonged shards at each
integration step. Its `run_manifest.csv` has 1,230 calls; the 12 calibration
calls remain recorded separately in the coupling diagnostic, exactly as in the
primary notebook.

## Progress and recovery

Every bounded worker result is checkpointed atomically, and progress is written
to both the terminal and `<run-dir>/progress.log`. Inspect or resume a run with:

```bash
python main.py --status /path/to/run-directory
python main.py --resume /path/to/run-directory --workers auto
python main_extended.py --status /path/to/extended-run-directory
python main_extended.py --resume /path/to/extended-run-directory --workers auto
```

Resume requires the same mode, Python/dependency environment, input hashes,
and resolved workload. A run produced by the immediately preceding validated
DTGateFixed notebook is migrated one way: unchanged non-main checkpoints stay
eligible, while every old `main_full_field` checkpoint is invalidated and all
main blocks recompute under the `raw-trace-v1` tag. This is required because
the predecessor checkpoints did not contain the raw PSP arrays. Subsequent
resumes reuse completed `raw-trace-v1` main checkpoints normally.

Extended runs use the same atomic status, log, and checkpoint machinery, but
`main_extended.py --resume` accepts only an incomplete run created from the
exact extended notebook and matching execution environment. It does not alter
or migrate a completed primary result directory.

Missing cache files are downloaded by the notebook and checked against its
pinned SHA-256 values. Results are written under:

```text
rise_tvb379_work/
├── source_data/
├── results_semantic_episodic_v3_<mode>/
└── RISE_TVB379_results_semantic_episodic_v3_<mode>.zip
```

## Run experiments on BU SCC

### 1. Clone the repository
```bash
git clone https://github.com/ocoke/tvb-auditory-alz.git
cd tvb-auditory-alz
```

### 2. Set up Python 3.12 venv
```bash
mkdir -p logs/

module purge
module load python3/3.12.4

python -m venv venvs/tvb
source venvs/tvb/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Submit to queue

```bash
qsub run_tvb.sh
```
