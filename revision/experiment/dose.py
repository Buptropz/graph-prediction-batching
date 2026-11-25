"""Change rate of node union as a function of how many query records touch the support.

The main panels deliberately draw up to half of their queries from neighbours of the support.
Here each panel keeps its support and receives new query sets with a controlled number of
support-adjacent records (0, 16, 32, 48, 64 of 64) and one set sampled uniformly from the corpus.
The sets are nested within a panel, so the levels are not independent samples.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

import lib
from build_cohorts import pc, project

LEVELS = (0, 16, 32, 48, 64)
Q = 64


def pools(task):
    c = lib.corpus(task)
    supports = [[c["pos"][str(v)] for v in lib.panel(task, p)["support_ids"]] for p in range(5)]
    taken = {i for s in supports for i in s}
    seen = {hashlib.sha256(c["x"][i].tobytes()).digest() for i in taken}
    pool = []
    for i in range(len(c["names"])):
        if i in taken:
            continue
        key = hashlib.sha256(c["x"][i].tobytes()).digest()
        if key not in seen:
            seen.add(key)
            pool.append(i)
    return supports, pool


def workloads(task, index, support, pool, adj, rng):
    c = lib.corpus(task)
    fixture = lib.panel(task, index)
    near = {b for s in support for b in adj[s]}
    a = rng.permutation(sorted(set(pool) & near)).tolist()
    n = rng.permutation(sorted(set(pool) - near)).tolist()
    sets = {str(m): a[:m] + n[:Q - min(m, len(a))] for m in LEVELS}
    sets["uniform"] = rng.choice(pool, Q, replace=False).tolist()
    for level, query in sets.items():
        assert len(query) == Q == len(set(query))
        nodes = support + query
        z = project(task, c["x"][nodes], np.arange(len(support)))
        assert np.allclose(z[:len(support)], fixture["tx"], atol=1e-5)
        coord = pc(z, np.arange(len(support)))[len(support):]
        orders = {"sorted": np.argsort(coord, kind="stable"), "random": rng.permutation(Q)}
        for layout, order in orders.items():
            q = [query[i] for i in order]
            edge = lib.induced(task, support + q)
            work = lib.Workload(z[:len(support)], fixture["y"], z[len(support):][order], c["names"][q],
                                np.repeat(np.arange(4), 16), edge, attached=True)
            touching = len({int(v) for v in work.edge[:, (work.edge[0] < work.ns) & (work.edge[1] >= work.ns)][1]})
            yield level, layout, work, touching


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/dose.json")
    ap.add_argument("--tasks", nargs="+", default=list(lib.TASKS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    lib.setup_torch()
    ref = lib.reference(args.device)
    rows = []
    for t, task in enumerate(args.tasks):
        supports, pool = pools(task)
        adj = lib.neighbours(task)
        for index, support in enumerate(supports):
            rng = np.random.default_rng([20261125, t, index])
            for level, layout, work, touching in workloads(task, index, support, pool, adj, rng):
                engine = lib.Engine(ref, work.y)
                y = lib.truth(task, work.ids)
                for recipe in ("joint", "support"):
                    out = {plan: lib.execute(engine, work, plan, recipe)[0] for plan in lib.PLANS}
                    row = dict(device=args.device, task=task, panel=index, level=level, layout=layout, recipe=recipe,
                               touching=touching, cross_edges=work.cross)
                    for plan in ("union", "axis"):
                        row[plan] = lib.endpoints(work.ids, work.rid, out["separate"], out[plan])
                    row["accuracy"] = {plan: int((out[plan].argmax(1) == y).sum()) for plan in lib.PLANS}
                    rows.append(row)
            print(task, index, flush=True)
    lib.dump(args.out, rows)


if __name__ == "__main__":
    main()
