# Coalescing Graph Prediction Requests Can Change Query Results

This is the research artifact for the PVLDB manuscript by Zebin Xu and Zhouxu Wei. The central comparison uses two explicit DuckDB statements over the same paper IDs. Separate execution uses one model call per request. Pooled execution concatenates two requests and uses one call. The artifact includes both checkpoints, their source snapshots, all model-ready input matrices, saved predictions, resource measurements, SQL, protocols and analysis programs. Use the commands from the extracted archive root.

## Inspect the evidence

| Question | Files |
| --- | --- |
| Which IDs move to another topic when requests are pooled? | `vldb2027/experiment/records/{cora,pubmed}-{ephris,nodepfn}.json`, field `cases[].transition` |
| How did the full probability vector change? | Matching `.npz` files. Each panel, preprocessing setting and companion condition has an ID array and two probability matrices in the same row order. The first 24 rows belong to the focal request. |
| Which query and model adapter ran? | `vldb2027/experiment/plans.sql`, `query.py`, and the decision rules in `PROTOCOL.md` in that directory |
| Where are the four-workload data and interventions? | `companion_dose/data`, `runs`, `mechanism`, `curves.csv`, `summary.json`, and the hash-locked inputs and predictions in `design-lock.json` and `prediction-lock.json` |
| Where are the larger-graph results? | `study_runs/gpu-scale` for 120 configurations, `study_runs/gpu-scale-controls` for 24 later numerical controls, and `study_runs/gpu-scale-summary.json` for the checked aggregation |
| How are datasets and features rebuilt? | `prepare_sources.py`, `prepare_data.py`, `prepare_study_materials.py`, `prepare_scale.py` and `companion_dose/build_cohorts.py`. Original protocol files accompany these programs. |

The SQL study reuses the existing Cora and PubMed panels. It has 80 cases (two datasets, two models, two preprocessing settings, five panels, two companion conditions), each with one warm-up and three timed executions per plan. A case is a fixed input, not a timing repeat or an independent dataset. The Cora witness uses paper 189577 in panel zero with joint NodePFN and the low companion request.

## Environment

Use Linux, Bash, sha256sum and Python 3.11. The commands below install the CPU environment into a new directory. A Python with an appropriate PyTorch wheel already installed can also be selected with the `PYTHON` environment variable.

```bash
python3.11 -m venv .cpu
source .cpu/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'
python -m pip install -r requirements.txt 'duckdb==1.4.3'
./reproduce.sh verify
./reproduce.sh smoke
```

The verify target checks the SHA-256 of bundled files and reconstructs the two model-study summaries and the SQL table from archived records. The smoke target executes a one-panel live Cora/NodePFN comparison and checks focal outputs against the archive. Both print the name of a fresh output directory. Exact results can also be inspected offline directly from the bundled predictions. Eleven original graph-distribution files used in provenance checks are fetched from immutable upstream revisions on the first run (about 17 MB in total). Prepared inputs and weights are bundled. The complete URL, byte count and SHA-256 for every fetched file are in `vldb2027/reproduction/raw-sources.json`. Running `python vldb2027/reproduction/fetch_raw.py` once populates those files for subsequent offline reruns. The projected PersRefEx annotations are bundled with their data card. Source and license details are in `vldb2027/reproduction/notices/ORIGINS.md`. Exact installed CPU package versions from the graph experiment are also in `results/packages.json`.

## Live comparisons

```bash
./reproduce.sh sql
./reproduce.sh cpu
```

The exact primary SQL runner is preserved in `vldb2027/experiment/registered/query.py`. The replay runner additionally logs numeric differences between timing repeats, requires identical class labels on every row, and stops if a repeated probability differs by more than 0.000001. See `vldb2027/experiment/REPLAY.md` for the reason and controls.

The full independent SQL replay and the PubMed CPU repeat diagnostics are in `vldb2027/experiment/replay/`. Recompute their per-row comparison with `python vldb2027/experiment/audit_replay.py`.

The SQL target scores every paper in both plans, checks all focal IDs and probabilities against the prior standalone model executions, saves fresh row-level outputs and recomputes the results table. The CPU target replays each checkpoint on all 100 registered workload/panel/count combinations and runs the four supplemental NodePFN path interventions. CPU probability comparisons allow a maximum absolute error of one part in a million and check all focal top choices for the main experiment. Timings depend on the host. The reported SQL timings came from a single i9-9940X with one DuckDB and two PyTorch execution threads.

The figures can be regenerated in a separate plotting environment based on the CPU environment. Matplotlib 3.11.2 was used for the supplied vector PDFs. Keep the inference environment's NumPy pin when installing plotting dependencies, or use a separate environment for plotting.

```bash
python -m pip install 'matplotlib==3.11.2' 'numpy==1.26.4'
./reproduce.sh figures
```

## GPU grid

Install the CUDA build in a **separate** Python 3.11 environment. The observed GPU was one NVIDIA RTX 4090. Across the registered commands, the largest peak of the PyTorch reserved-memory counter was 10.438 GB. Host memory and CUDA/driver overhead are additional. Execute commands serially using the target below. The archival grid summed to 1,229.07 seconds of process wall time, excluding data preparation and the 24 numerical controls.

```bash
python3.11 -m venv .gpu
source .gpu/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0'
python -m pip install -r requirements.txt
./reproduce.sh gpu
```

This target creates fresh result directories, uses the prepared PubMed splits and reruns 120 formal configurations plus 24 numerical diagnostic configurations. It checks data identities, class transitions, resource records and the deterministic contrasts while constructing its own summary. Default mixed-precision kernels can give different labels on a repeat. The strict FP32 results and per-draw records are given separately. The archived inputs and result files stay available after every run. All node counts include labeled supports, focal papers and companions. Frozen checkpoint size stays fixed across graph sizes.
