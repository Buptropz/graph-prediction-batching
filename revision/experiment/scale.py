"""Latency and memory of the three plans as the support, the requests and the fan-in grow.

PubMed supplies enough records for supports of up to 512 nodes and 32 requests of 256 records.
Queries are sampled uniformly (no enrichment of support neighbours) and assigned to requests at
random.  Each configuration is timed through the complete Arrow SQL query.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import torch

import lib
from build_cohorts import project

SUPPORTS, REQUESTS, FANS = (32, 128, 512), (16, 64, 256), (2, 4, 8, 16, 32)


def build(task, s, q, k, rng):
    c = lib.corpus(task)
    order = rng.permutation(len(c["names"]))
    per = [iter([i for i in order if c["y"][i] == label]) for label in range(lib.CLASSES[task])]
    support = [int(next(per[j % len(per)])) for j in range(s)]
    chosen = set(support)
    rest = [int(i) for i in order if int(i) not in chosen]
    query = rest[:q * k]
    z = project(task, c["x"][support + query], np.arange(s))
    edge = lib.induced(task, support + query)
    return lib.Workload(z[:s], c["y"][support], z[s:], c["names"][query], np.repeat(np.arange(k), q), edge), query


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/scale.json")
    ap.add_argument("--task", default="pubmed")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    lib.setup_torch()
    ref = lib.reference("cuda")
    rows = []
    for a, s in enumerate(SUPPORTS):
        for b, q in enumerate(REQUESTS):
            for k in FANS:
                rng = np.random.default_rng([20261127, a, b, k])
                work, query = build(args.task, s, q, k, rng)
                engine = lib.Engine(ref, work.y)
                y = lib.corpus(args.task)["y"][query]
                op = lib.Operator(engine, work, "joint")
                times, memory, out = {p: [] for p in lib.PLANS}, {}, {}
                with duckdb.connect(":memory:") as db:
                    db.execute("SET threads=1")
                    op.load(db)
                    op.register(db, "arrow")
                    for rep in range(args.repeats + 1):
                        shift = rep % 3
                        for plan in lib.PLANS[shift:] + lib.PLANS[:shift]:
                            torch.cuda.reset_peak_memory_stats()
                            p, ms, calls, _ = op.run(db, plan)
                            if rep:
                                times[plan].append(ms)
                            else:
                                out[plan], memory[plan] = p, torch.cuda.max_memory_allocated() / 2**20
                row = dict(support=s, request=q, fan=k, nodes=s + q * k, cross_edges=work.cross,
                           times=times, memory_mib=memory,
                           joint={plan: lib.endpoints(work.ids, work.rid, out["separate"], out[plan])
                                  for plan in ("union", "axis")},
                           accuracy={plan: float((out[plan].argmax(1) == y).mean()) for plan in lib.PLANS})
                fit = {plan: lib.execute(engine, work, plan, "support")[0] for plan in lib.PLANS}
                row["support_fit"] = {plan: lib.endpoints(work.ids, work.rid, fit["separate"], fit[plan])
                                      for plan in ("union", "axis")}
                rows.append(row)
                med = {p: float(np.median(t)) for p, t in times.items()}
                print(s, q, k, {p: round(v, 1) for p, v in med.items()}, {p: round(v) for p, v in memory.items()},
                      row["joint"]["union"]["flips"], row["joint"]["axis"]["flips"], flush=True)
                lib.dump(args.out, rows)


if __name__ == "__main__":
    main()
