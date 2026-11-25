# Additional experiments of the revised paper

Recorded 2026-11-25. Every experiment below was specified after the results of the main grid
(`vldb2027/upgrade`, protocol of 2026-11-24) were known. All of them are exploratory. They reuse
the 15 archived supports; repeated use of a record adds no independent sample. Nothing in
`vldb2027/upgrade` was modified.

| Script | Question | Device | Specified |
| --- | --- | --- | --- |
| `accuracy.py` | How accurate is each plan, and how confident was the separate plan about changed records? Joins held-out labels to the archived probability vectors. | none (stored results) | After the main grid. Labels were never part of the fixtures or plans. |
| `default.py` | Does the effect hold in the released default configuration (three ensemble members, support fit)? Full fit-by-edge factorial, both assignments. | CPU | After the main grid. The engine validation on two panels had already shown three-member outputs. |
| `dose.py` | How does the change rate depend on the number of query records linked to the support? Levels 0/16/32/48/64 of 64 and one uniformly sampled set per panel; sets are nested within a panel. | CPU | After the accuracy analysis. |
| `chunks.py` | Does a scalar vectorised function produce the effect without explicit bucketing? 8,192 PubMed records, default configuration. | CPU | After the main grid. The list of executions (repeat, row order, filter placement, prefix rows, threads, per-analyst reference) was fixed before the first run. |
| `gnn.py` | Do trained GCN and GraphSAGE networks without support labels change under node union? One and two layers, three seeds, panel nodes removed from training. | CPU | After the main grid. |
| `timing.py` | Latency through two operator interfaces, per-phase breakdown, unequal lengths with per-length grouping, padding and a boundary guard. | GPU | After profiling showed that the row interface's result conversion dominated the archived latencies. |
| `scale.py` | Latency, memory and agreement for supports of 32-512, requests of 16-256 and 2-32 requests per call on PubMed. One draw per configuration. | GPU | After the timing experiment. |
| `ephris_axis.py` | Can Ephris be isolated inside one call? One schema fit, one set-attention set, one copy of the global nodes and one edge block per lane; all 120 archived Ephris conditions. | CPU | After all experiments above; the paper had listed the plan as not built. |
| `anygraph.py` | Does a third released model (GraphAny, Wisconsin checkpoint) change under node union, through which path, and does a lane plan with one least-squares solve per lane restore the answers? Panel and raw features. | CPU | After all experiments above. |
| `arrivals.py` | Response time and answers of the three plans under Poisson arrivals at a work-conserving operator (at most 32 requests per call). Virtual-time replay with every model call executed and timed. | GPU | After the scale experiment was specified and before either had run. Rates, cap, backlog rule (stop at 256 waiting requests) and arrival seeds were fixed in the script before the first GPU run; a CPU smoke test with 60 arrivals preceded it. |

## Engine

`lib.py` contains a request-axis engine that generalises the archived adapter
(`upgrade/experiment/request_axis.py`) in two ways: requests of unequal length are padded with rows
that have no edge except a self loop, and one lane is opened per (request, ensemble member) pair.
Before any experiment was run, the engine was checked against the archived single-member results
(separate and union identical; request axis within 2e-5 of the archived adapter and within 1e-5 of
its own separate plan) and against the released classifier with one and three members (largest
difference 2.3e-7, no changed class).

## Decisions and failures recorded as they occurred

- The numerical tolerance stays at the prespecified 1e-5 per probability component. The padded plan
  exceeded it in some cases (reported in the paper). No result was dropped for that reason.
- Latency in the archived study was measured with a Python row interface. Profiling on 2026-11-25
  showed that DuckDB's conversion of the returned nested list costs roughly 0.16 ms per returned
  value, independent of the plan. The revised timing reports both that interface and an Arrow
  interface, measured in the same run.
- The first run of `timing.py` stopped at Chameleon panel 3 on an assertion that compared the new
  engine's request-axis output with the archived adapter's output under the 1e-5 tolerance. The two
  are different floating-point executions; the assertion was relaxed to 2e-5 and an assertion on the
  distance to the separate plan of the same run (1e-5) was added. The two remaining panels were
  then timed while another process occupied the GPU; those timings were discarded and both panels
  were run again on 2026-11-25 at 17:05 after the card had been idle for three minutes. `logs/queue.log`
  records for every GPU job how many other processes were on the card when it ended (8 for the discarded
  run, 0 for the rerun, the scale sweep and the arrival replay).
- The boundary guard's re-runs were all triggered by the ranking condition, not by the class or
  threshold condition. A first draft of the paper attributed them to classes with negligible
  probability; the stored outputs did not support that explanation and it was removed.
- Ephris: the released graph attention could not be given lanes, so `ephris_axis.py` re-expresses it as a
  dense masked attention with the released weights. Run as a single lane this engine is a different
  floating-point execution from the released path: it stays within 1e-5 of the archived outputs (largest
  difference 9.4e-6, no changed class or threshold pair) and changes 85 of 2,400 ranked lists at Chameleon
  ties. The paper reports this next to the lane plan's own 83 changed lists. No assertion was relaxed.
- GraphAny: the fusion module and distance features are the released code (imported with stub modules for
  packages that are not installed); propagation and the least-squares solve are re-implemented so that lanes
  are possible. Against the released preprocessing, 149 of 150 calls agree within 1.2e-7 and one differs by
  9.0e-3 without a class change, because the released perplexity search is discontinuous. The checkpoint
  stores unresolved configuration strings; values were taken from the release README and `configs/model.yaml`.
  Probabilities are the softmax of the fused output (the release only takes the argmax).
- Arrival replay: the stop rule (256 waiting requests) did not fire for the request-axis plan at 320 requests per
  second, although that rate exceeds the plan's capacity at 32 requests per call (255 requests per second, from
  the median duration of the 35 full calls of the same experiment); 1,000 arrivals at that rate last 3.1 s, too
  short for the backlog to reach the limit. After the run, `report.py` was changed to compute each plan's capacity
  from the measured call durations, to mark such a cell in the table, and to count as sustained only rates below
  the capacity. No run was repeated and no rate was removed. Each rate has one arrival sequence, so the
  percentiles carry no interval; the 99th percentile rests on ten responses.
- Per-panel resampling intervals for the two headline rates were added to `report.py` after the counts were
  known (10,000 resamples of the five panels within each corpus, seed 20261125).
- Answer-only experiments ran on CPU because the GPU was shared on 2026-11-25. All comparisons are
  between plans on one device.

## Reproduction

From the repository root, with the interpreter of the archived GPU runs:

~~~bash
source vldb2027/revision/env.sh
$PY vldb2027/revision/experiment/accuracy.py
$PY vldb2027/revision/experiment/default.py --device cpu
$PY vldb2027/revision/experiment/dose.py --device cpu
$PY vldb2027/revision/experiment/chunks.py --device cpu
$PY vldb2027/revision/experiment/gnn.py --device cpu
$PY vldb2027/revision/experiment/ephris_axis.py --device cpu
$PY vldb2027/revision/experiment/anygraph.py
CUDA_VISIBLE_DEVICES=0 $PY vldb2027/revision/experiment/timing.py
CUDA_VISIBLE_DEVICES=0 $PY vldb2027/revision/experiment/scale.py
CUDA_VISIBLE_DEVICES=0 $PY vldb2027/revision/experiment/arrivals.py
$PY vldb2027/revision/experiment/report.py     # tables, figures, numbers
~~~

`env.sh` points at the local interpreter and dependency paths of the measured machine; adjust it
for another installation (Python 3.11, NumPy 1.26.4, DuckDB 1.4.3, PyArrow 17.0.0, PyTorch 2.11.0,
PyTorch Geometric 2.8.0, scikit-learn 1.9.1, Matplotlib 3.11.2).
