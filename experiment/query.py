"""Run two explicit DuckDB grouping plans on the same requests and checkpoints."""
import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "companion_dose"))

import duckdb
import numpy as np
import torch
from inference import focal_prob, model
from run import assemble

PLANS = (HERE / "plans.sql").read_text()
MEMBERSHIP = """
WITH separate AS (SELECT paper_id, topic FROM labels WHERE plan = 'separate' AND rid = 0),
     pooled AS (SELECT paper_id, topic FROM labels WHERE plan = 'pooled' AND rid = 0)
SELECT * FROM (SELECT * FROM separate EXCEPT SELECT * FROM pooled)
ORDER BY paper_id, topic
"""


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_function(net, d, q, edge, recipe, calls):
    ns = len(d["y"])

    def scores(positions):
        pos = np.asarray(positions, dtype=np.int64)
        index = np.full(ns + len(q), -1, dtype=np.int64)
        index[:ns] = np.arange(ns)
        index[ns + pos] = ns + np.arange(len(pos))
        remapped = index[edge]
        selected = remapped[:, (remapped >= 0).all(axis=0)]
        calls.append(tuple(positions))
        raw = net(d["tx"], d["y"], q[pos], selected, recipe)
        prob = focal_prob(raw, len(pos), False)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1), 1)
        return prob.tolist()
    return scores


def execute_case(net, d, reference, recipe, mode, panel, repeats):
    q, edge = assemble(d, mode, 32)
    nf = len(d["qx"])
    ids = list(map(str, d["units"])) + list(map(str, d["donor_units"][d[mode]]))
    assert len(ids) == len(q) == len(set(ids))
    calls = []
    with duckdb.connect(":memory:") as db:
        db.execute("SET threads=1")
        db.execute("CREATE TABLE requests(pos INTEGER PRIMARY KEY, request_id INTEGER, paper_id VARCHAR)")
        db.executemany("INSERT INTO requests VALUES (?, ?, ?)",
                       [(i, int(i >= nf), uid) for i, uid in enumerate(ids)])
        db.create_function("graph_scores", model_function(net, d, q, edge, recipe, calls),
                           ["INTEGER[]"], "DOUBLE[][]", side_effects=True)
        out, times = {}, {"separate": [], "pooled": []}
        for rep in range(-1, repeats):
            order = ["separate", "pooled"]
            if (rep + panel + int(mode == "high")) % 2: order.reverse()
            for plan in order:
                calls.clear()
                start = time.perf_counter_ns()
                rows = db.execute(PLANS, [plan == "pooled"]).fetchall()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                groups = [tuple(range(len(ids)))] if plan == "pooled" else [
                    tuple(range(nf)), tuple(range(nf, len(ids)))]
                assert sorted(calls) == groups
                assert [(p, r, i) for p, r, i, _, _ in rows] == [
                    (p, int(p >= nf), i) for p, i in enumerate(ids)]
                prob = np.asarray([row[3] for row in rows])
                label = np.asarray([row[4] for row in rows])
                assert np.array_equal(prob.argmax(1), label)
                if rep == -1:
                    out[plan] = prob
                else:
                    assert np.array_equal(out[plan], prob), "repetition changed a probability"
                    times[plan].append(elapsed)
        db.execute("CREATE TABLE labels(plan VARCHAR, paper_id VARCHAR, rid INTEGER, topic INTEGER)")
        db.executemany("INSERT INTO labels VALUES (?,?,?,?)",
                       [(plan, uid, int(p >= nf), int(a[p].argmax())) for plan, a in out.items()
                        for p, uid in enumerate(ids)])
        moved = db.execute(MEMBERSHIP).fetchall()
    source, target = [out[p][:nf] for p in ("separate", "pooled")]
    mask = source.argmax(1) != target.argmax(1)
    assert {u for u, _ in moved} == set(np.asarray(ids[:nf])[mask])
    parity = {}
    for plan, tag in (("separate", "base"), ("pooled", mode)):
        a, b = out[plan][:nf], reference[recipe + "_" + tag]
        err = float(np.abs(a - b).max())
        assert err <= 1e-6 and np.array_equal(a.argmax(1), b.argmax(1)), (plan, err)
        parity[plan] = err
    diff = (target - source)
    return {"panel": panel, "mode": mode, "moved": sorted(u for u, _ in moved),
            "transition": [[ids[p], int(source[p].argmax()), int(target[p].argmax())]
                           for p in np.flatnonzero(mask)],
            "n": nf, "rows": len(ids), "timings_ms": times, "archive_max_abs": parity,
            "max_abs": float(abs(diff).max()), "mean_tv": float(0.5 * abs(diff).sum(1).mean())}, {
                "ids": np.asarray(ids), **out}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["cora", "pubmed"])
    parser.add_argument("--model", required=True, choices=["ephris", "nodepfn"])
    parser.add_argument("--panels", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1: parser.error("repeats must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    net = model(args.model, "cpu")
    archive = ROOT / "companion_dose"
    cases, vectors = [], {}
    for panel in range(args.panels):
        src = archive / "data" / args.task / f"panel-{panel}.npz"
        ref = archive / "runs" / args.task / f"{args.task}_{args.model}_p{panel}_k32.npz"
        with np.load(src, allow_pickle=False) as d, np.load(ref, allow_pickle=False) as reference:
            for recipe in ["joint", "support"]:
                for mode in ["low", "high"]:
                    case, prob = execute_case(net, d, reference, recipe, mode, panel, args.repeats)
                    case.update(recipe=recipe, source_sha256=file_hash(src), reference_sha256=file_hash(ref))
                    cases.append(case)
                    for key, arr in prob.items(): vectors[f"p{panel}_{recipe}_{mode}_{key}"] = arr
        print(args.model, args.task, panel, flush=True)
    cpu = next((l.partition(":")[2].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.processor())
    result = dict(task=args.task, model=args.model, repeats=args.repeats, duckdb=duckdb.__version__,
                  torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
                  cpu=cpu, model_threads=torch.get_num_threads(), db_threads=1,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  code={p.name:file_hash(p) for p in [Path(__file__), HERE/"plans.sql", HERE/"PROTOCOL.md"]},
                  cases=cases)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.task}-{args.model}"
    np.savez_compressed(stem.with_suffix(".npz"), **vectors)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__": main()
