# Coalescing Graph Prediction Requests Can Change Query Results

Research artifact by Zebin Xu and Zhouxu Wei, Beijing University of Posts and Telecommunications.

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
