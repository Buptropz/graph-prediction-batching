# Artifact of the revised paper (v3)

This folder adds the experiments of the revised paper to the v2 archive. The v2 material under
`vldb2027/upgrade` (panels, SQL statements, the main grid and its per-record outputs) is unchanged
and is still verified by `reproduce-upgrade.sh`. Everything below is relative to the archive root.

## Start here

Python 3.11 on Linux. Install the PyTorch 2.11.0 wheel for the intended device, then

~~~bash
python -m pip install -r requirements.txt 'duckdb==1.4.3' 'matplotlib==3.11.2'
bash reproduce-upgrade.sh verify                       # main grid, from stored outputs
bash vldb2027/revision/reproduce-revision.sh verify    # every table, figure and number of the paper
bash vldb2027/revision/reproduce-revision.sh smoke     # live GraphAny run on one Cora panel, ~10 s
~~~

`verify` runs offline. It regenerates `source/figures/*` and `results/headlines-rev.json` from the
stored results and fails if any number of the paper changes. It rewrites the figure files, whose
bytes depend on the installed Matplotlib and fonts, so check `vldb2027/revision/MANIFEST.sha256`
(`sha256sum -c`, from the archive root) before running it. The other targets first restore the
upstream corpus files from pinned URLs with `vldb2027/reproduction/fetch_raw.py` (about 17 MB).
`paper.pdf` is the revised paper and `source/` compiles it with `latexmk -pdf main.tex`.

## Live reruns

~~~bash
bash vldb2027/revision/reproduce-revision.sh cpu RUN        # answer-only experiments, on the order of hours
CUDA_VISIBLE_DEVICES=0 bash vldb2027/revision/reproduce-revision.sh gpu RUN   # timing, scale, arrivals
~~~

Both write to `RUN` and leave the stored results untouched. Latencies were measured on an idle
RTX 4090 with an Intel Core i9-9940X; the timing experiments must not share the card. To rebuild
the paper from a fresh run, copy `RUN/*` over `vldb2027/revision/results/` and run
`python vldb2027/revision/experiment/report.py`, then `latexmk -pdf main.tex` in
`vldb2027/revision/source`.

## What answers which question

| Section of the paper | Script (`vldb2027/revision/experiment`) | Stored result (`vldb2027/revision/results`) |
| --- | --- | --- |
| Release default configuration, fit-by-edge factorial | `default.py` | `default/` |
| Accuracy and confidence of changed records | `accuracy.py` | `accuracy.json` |
| Adjacency sweep and uniformly sampled queries | `dose.py` | `dose.json` |
| Scalar vectorized function, no explicit batching | `chunks.py` | `chunks.json` |
| Trained GCN and GraphSAGE networks | `gnn.py` | `gnn.json` |
| Ephris lane plan | `ephris_axis.py` | `ephris_axis.json`, `ephris_axis.npz` |
| GraphAny, node union and lane plan | `anygraph.py` | `graphany.json`, `graphany.npz` |
| Operator interfaces, phase breakdown, unequal lengths, guard | `timing.py` | `timing/` |
| Larger supports, longer requests, more requests | `scale.py` | `scale.json` |
| Poisson arrivals at one operator | `arrivals.py` | `arrivals.json` |
| Tables, figures, numbers, plain-text abstract | `report.py` | `headlines-rev.json`, `../source/figures`, `../ABSTRACT.txt` |

Absolute path prefixes of the measured machine were removed from the files in `logs/`; nothing
else in them was edited. `env.sh` records the interpreter and dependency paths of that machine.

`lib.py` holds the shared request-axis engine (padding, ensemble lanes), the Arrow and row
operators, and the NumPy endpoints. `PROTOCOL-REVISION.md` states when each experiment was
specified, how the engines were validated, and every failure and relaxed check. `logs/` keeps the
console output of the stored runs; `logs/queue.log` records whether another process was on the GPU
when a timing job ended.

## Upstream material

The NodePFN, Ephris and GraphAny sources and checkpoints are the pinned snapshots listed in
`vendor/sources.json` and are used unmodified. The Ephris lane plan re-expresses one released
reduction with the released weights (see the protocol file); the GraphAny script imports the
released fusion module and re-implements propagation and the least-squares solve.
