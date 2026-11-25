# When Batching Changes Query Answers: Request Isolation for Graph Inference

Research artifact by Zebin Xu and Zhouxu Wei, Beijing University of Posts and Telecommunications.

A database that batches model calls must still return to each caller the rows it would have returned alone. For graph classifiers that predict from a labeled support, appending the nodes of several requests to one graph (node union) shares context between them, while giving each request its own lane on the model's batch dimension (request axis) does not. The study measures both plans in DuckDB with three released checkpoints (NodePFN, Ephris, GraphAny) and with trained GCN and GraphSAGE networks on Cora, PubMed and Chameleon. On 960 query records, node union changed the predicted class of 208 records under NodePFN's default configuration and of 250 with query-inclusive normalization. The request-axis operator preserved every predicted class and threshold predicate, ran 3.26 times as fast as separate calls and cost 16% more than node union.

## Run the artifact (version 3.0.0)

Download **reproduction-v3.tar.gz** and **ARTIFACT-SHA256SUMS-v3.txt** from the [version 3.0.0 release](https://github.com/Buptropz/graph-prediction-batching/releases/tag/v3.0.0) into the same directory.

```bash
sha256sum --check ARTIFACT-SHA256SUMS-v3.txt
tar -xzf reproduction-v3.tar.gz
cd graph-prediction-batching
```

Use Python 3.11 on Linux. Install the PyTorch 2.11.0 wheel for the intended device, then

```bash
python -m pip install -r requirements.txt 'duckdb==1.4.3' 'matplotlib==3.11.2'
bash reproduce-upgrade.sh verify                       # main grid, from stored outputs
bash vldb2027/revision/reproduce-revision.sh verify    # every table, figure and number of the paper
bash vldb2027/revision/reproduce-revision.sh smoke     # live GraphAny run on one Cora panel
```

The archive holds the paper and its LaTeX source, the 15 panels, the SQL statements, the three checkpoints with their upstream sources, unrounded probability vectors for every plan and record, per-query timings, console logs and the scripts that produce each table and figure. It contains the complete earlier archive unchanged. [What answers which question, and the live rerun targets](revision/ARTIFACT.md) · [when each experiment was specified, engine validation, failures and relaxed checks](revision/experiment/PROTOCOL-REVISION.md)

## Browse without downloading

- [Experiment scripts](revision/experiment/) and the [numbers printed in the paper](revision/results/headlines-rev.json).
- Stored results small enough to read here: [accuracy](revision/results/accuracy.json), [scalar vectorized function](revision/results/chunks.json), [trained networks](revision/results/gnn.json), [scale sweep](revision/results/scale.json) and [Poisson arrivals](revision/results/arrivals.json). Per-record probability vectors are in the release archive.

Latencies were measured on an idle RTX 4090 with an Intel Core i9-9940X. Every experiment after the primary comparison was specified with earlier results in view and is exploratory; the protocol file records this for each one. Original third-party licenses, model cards and attribution accompany their files in the release.

## Earlier study (version 1.0.0)

The [version 1.0.0 release](https://github.com/Buptropz/graph-prediction-batching/releases/tag/v1.0.0) is the preliminary two-request study reported in the appendix of the paper, under its earlier title "Coalescing Graph Prediction Requests Can Change Query Results". The material below describes that release.

The two DuckDB plans evaluate the same paper IDs against the same graph classifier. One scores each request separately. The other concatenates both requests into one graph. On Cora, the pooled plan changed the predicted topic of 41 of 120 distinct focal papers for public Ephris and 22 of 120 for jointly normalized NodePFN in at least one of two companion conditions. For public Ephris on PubMed the corresponding count was 6 of 120. Both plans return a full probability vector for every requested ID.

## Run the artifact

Download **reproduction.tar.gz** and **ARTIFACT-SHA256SUMS.txt** from the [version 1.0.0 release](https://github.com/Buptropz/graph-prediction-batching/releases/tag/v1.0.0). Verify and extract these files in the same directory.

```bash
sha256sum --check ARTIFACT-SHA256SUMS.txt
tar -xzf reproduction.tar.gz
cd graph-prediction-batching
```

Follow the included README to create the Python 3.11 CPU environment. Run `./reproduce.sh verify` to reconstruct the published summaries or `./reproduce.sh smoke` to execute a live Cora query. The release includes checkpoint bytes, upstream inference implementations, all model-ready inputs, preserved predictions, numerical controls, original protocols, exact dependency versions and the full experimental directory structure. An upstream file manifest records exact public revisions and hashes.

[Full environment and rerun commands](REPRODUCTION.md)

## Inspect individual results

- [Executable SQL](experiment/plans.sql) and [model adapter with ID and archive checks](experiment/query.py).
- [Paired query measurements by model and dataset](experiment/sql-summary.json) and [per-paper IDs, transitions, probabilities and timings](experiment/records/).
- [Seven probabilities for Cora paper 189577](experiment/witness.json) and [the plotted result](evidence/witness.pdf).
- [Four-workload companion-count curves](evidence/companion-curves.csv) and [all three GPU scales with numerical controls](evidence/gpu-scale-summary.json).

The repository view contains selected records for browsing. Use the versioned release archive for the complete executable layout. Each case contains the same IDs under both plans. The timings describe a warm CPU query on the measured host. The GPU scale study uses one citation graph and one frozen Ephris checkpoint.

Original third-party licenses, model cards and attribution accompany their files in the release.

A full independent replay matched all 160 original SQL probability matrices for the two requests. The [repeat-check record](experiment/REPLAY.md) documents an earlier unquantified failure of a bitwise assertion, the subsequent checks and the limits of that evidence.
