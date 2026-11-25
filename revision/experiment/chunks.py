"""A scalar vectorised operator makes answers depend on the engine's vectors.

No explicit bucketing appears in these queries.  ``topic(node)`` is an ordinary Arrow user-defined
function: DuckDB hands it one vector of rows at a time, and it scores the rows it receives against
the labelled support with the released classifier in its default configuration.  The same row is
then scored under several executions that a relational engine may regard as equivalent.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

import lib
from build_cohorts import project

ROWS, ANALYST = 8192, 16


class Scorer:
    def __init__(self, ref, task, support, y, nodes):
        c = lib.corpus(task)
        self.ref, self.y = ref, y
        z = project(task, c["x"][list(support) + list(nodes)], np.arange(len(support)))
        self.tx = z[:len(support)]
        self.z = dict(zip(map(int, nodes), z[len(support):]))
        self.support = list(support)
        self.adj = lib.neighbours(task)
        self.vectors = []
        self.classes = lib.CLASSES[task]

    def edges(self, nodes):
        order = self.support + list(nodes)
        pos = {v: i for i, v in enumerate(order)}
        out = [(pos[a], pos[b]) for a in order for b in self.adj[a] if b in pos]
        return np.asarray(out, dtype=np.int64).reshape(-1, 2).T

    def score(self, nodes):
        nodes = [int(v) for v in nodes]
        q = np.stack([self.z[v] for v in nodes])
        return lib.public(self.ref, self.tx, self.y, q, self.edges(nodes), "support")

    def udf(self, column):
        nodes = column.to_numpy(zero_copy_only=False)
        self.vectors.append(len(nodes))
        p = self.score(nodes)
        return pa.ListArray.from_arrays(pa.array(np.arange(0, p.size + 1, self.classes, dtype=np.int32)),
                                        pa.array(p.ravel()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/chunks.json")
    ap.add_argument("--task", default="pubmed")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    lib.setup_torch()
    task = args.task
    ref = lib.reference(args.device, 3)
    c = lib.corpus(task)
    fixture = lib.panel(task, 0)
    support = [c["pos"][str(v)] for v in fixture["support_ids"]]
    taken = {c["pos"][str(v)] for p in range(5) for v in lib.panel(task, p)["support_ids"]}
    rng = np.random.default_rng(20261126)
    pool = np.array([i for i in range(len(c["names"])) if i not in taken])
    nodes = rng.choice(pool, ROWS, replace=False)
    analyst = np.arange(ROWS) // ANALYST
    scorer = Scorer(ref, task, support, fixture["y"], nodes)
    shuffled = rng.permutation(ROWS)
    extra = rng.choice(np.setdiff1d(pool, nodes), 100, replace=False)
    scorer.z.update(Scorer(ref, task, support, fixture["y"], extra).z)

    db = duckdb.connect(":memory:")
    db.execute("SET threads=1")
    db.create_function("topic", scorer.udf, ["INTEGER"], "DOUBLE[]", type="arrow", side_effects=True)
    for name, order in (("papers", np.arange(ROWS)), ("papers_shuffled", shuffled)):
        db.execute(f"CREATE TABLE {name}(node INTEGER, analyst INTEGER)")
        db.executemany(f"INSERT INTO {name} VALUES (?,?)", [(int(nodes[i]), int(analyst[i])) for i in order])
    db.execute("CREATE TABLE papers_prefixed(node INTEGER, analyst INTEGER)")
    db.executemany("INSERT INTO papers_prefixed VALUES (?,?)",
                   [(int(v), -1) for v in extra] + [(int(nodes[i]), int(analyst[i])) for i in range(ROWS)])

    def run(sql, threads=1):
        db.execute(f"SET threads={threads}")
        scorer.vectors = []
        rows = db.execute(sql).fetchall()
        return {int(r[0]): np.asarray(r[1]) for r in rows}, list(scorer.vectors)

    queries = {
        "base": ("SELECT node, topic(node) FROM papers", 1),
        "repeat": ("SELECT node, topic(node) FROM papers", 1),
        "row order": ("SELECT node, topic(node) FROM papers_shuffled", 1),
        "filter first": ("SELECT node, topic(node) FROM papers WHERE analyst % 2 = 0", 1),
        "filter last": ("SELECT node, p FROM (SELECT node, analyst, topic(node) AS p FROM papers) WHERE analyst % 2 = 0", 1),
        "100 rows ahead": ("SELECT node, topic(node) FROM papers_prefixed WHERE analyst >= 0", 1),
        "four threads": ("SELECT node, topic(node) FROM papers", 4),
    }
    results, vectors = {}, {}
    for name, (sql, threads) in queries.items():
        results[name], vectors[name] = run(sql, threads)
        print(name, len(results[name]), vectors[name], flush=True)

    # Per-analyst reference: each analyst's 16 rows evaluated as their own request.
    separate = {}
    for a in range(ROWS // ANALYST):
        group = nodes[analyst == a]
        separate.update(zip(map(int, group), scorer.score(group)))
    results["per analyst"] = separate

    def versus(a, b):
        keys = sorted(set(results[a]) & set(results[b]))
        pa_, pb = np.stack([results[a][k] for k in keys]), np.stack([results[b][k] for k in keys])
        return dict(rows=len(keys), flips=int((pa_.argmax(1) != pb.argmax(1)).sum()),
                    threshold=int(np.count_nonzero((pa_ >= .5) != (pb >= .5))),
                    max_abs=float(np.abs(pa_ - pb).max()), tv=float(np.abs(pa_ - pb).sum(1).mean() / 2))

    truth = dict(zip(map(int, nodes), c["y"][nodes]))
    report = dict(device=args.device, task=task, rows=ROWS, analyst_rows=ANALYST, support=len(support), members=3, fit="support",
                  vectors=vectors, sql={k: v[0] for k, v in queries.items()},
                  versus_base={k: versus("base", k) for k in results if k != "base"},
                  filter_pair=versus("filter first", "filter last"),
                  accuracy={k: float(np.mean([v.argmax() == truth[n] for n, v in r.items()])) for k, r in results.items()})
    print({k: (v["flips"], v["rows"]) for k, v in report["versus_base"].items()}, report["filter_pair"])
    lib.dump(args.out, report)


if __name__ == "__main__":
    main()
