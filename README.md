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

## Run Experiments

You can either run the experiments in Jupyter Notebook or directly. 

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


| Mode | Calibration | Main | Integration step | Local fixed | Parameters | Shuffles | Total TVB calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke` | 2 | 8 | 8 | 4 | 6 | 6 | 34 |
| `pilot` | 6 | 24 | 8 | 8 | 24 | 15 | 85 |
| `final` | 12 | 240 | 160 | 80 | 120 | 150 | **762** |


Final mode uses 20 paired numerical initializations, 50 spatial shuffles,
500 matched parcel-set controls, three severity values, and pulse/2 Hz/5 Hz
probes. Of its 762 calls, 750 are in `run_manifest.csv`; the 12 calibration
calls are recorded separately because each calibration row contains a control
and pulse simulation. The integration-step stage contains 40 endpoint
condition/seed work units and 160 TVB calls.

Missing cache files are downloaded by the notebook and checked against its
pinned SHA-256 values. Results are written under:

```text
rise_tvb379_work/
├── source_data/
├── results_semantic_episodic_v3_<mode>/
└── RISE_TVB379_results_semantic_episodic_v3_<mode>.zip
```