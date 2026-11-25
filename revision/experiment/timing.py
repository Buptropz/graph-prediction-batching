"""End-to-end latency of every plan, with a per-phase breakdown and two operator interfaces.

Part A replays the main condition (feature-sorted requests, attached edges, joint fit) at fan-in two
and four.  Each plan is timed through the complete SQL query twice: once with the row-at-a-time
Python operator of the archived study and once with an Arrow operator that returns one columnar
value.  Instrumented repetitions then split the Arrow query into feature fitting, graph packing,
the neural forward, the remaining operator work and the time spent inside DuckDB.

Part B varies request lengths and compares one call per distinct length, one padded call, and the
padded call followed by a guard that re-evaluates requests whose answers lie within tolerance of a
decision boundary.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import torch

import lib

PROFILES = {"equal": (16, 16, 16, 16), "paired": (8, 24, 8, 24), "distinct": (7, 13, 19, 25),
            "skewed": (4, 8, 20, 32)}
MAIN = ("separate", "union-2", "axis-2", "union", "axis")
RAGGED = ("separate", "union", "axis-length", "axis", "axis-guard")
lib.BUCKET.update({"union-2": "rid // 2", "axis-2": "rid // 2"})


def timed(op, db, plans, repeats, ordinal):
    times = {p: [] for p in plans}
    udf = {p: [] for p in plans}
    out, calls, fallback = {}, {}, {}
    for rep in range(repeats + 1):
        shift = (rep + ordinal) % len(plans)
        for plan in plans[shift:] + plans[:shift]:
            p, ms, n, inside = op.run(db, plan)
            if rep:
                assert np.abs(p - out[plan]).max() <= lib.TOL
                times[plan].append(ms)
                udf[plan].append(inside)
            else:
                out[plan], calls[plan], fallback[plan] = p, n, list(op.fallback)
    return out, times, udf, calls, fallback


def phases(op, db, plans, repeats):
    clock = op.engine.clock
    clock.on = True
    rows = {}
    for plan in plans:
        op.run(db, plan)
        acc = []
        for _ in range(repeats):
            clock.reset()
            _, ms, _, inside = op.run(db, plan)
            t = dict(clock.t)
            t["operator"] = inside - sum(t.values())
            t["engine"] = ms - inside
            t["total"] = ms
            acc.append(t)
        rows[plan] = {k: float(np.median([a[k] for a in acc])) for k in acc[0]}
    clock.on = False
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/timing")
    ap.add_argument("--tasks", nargs="+", default=list(lib.TASKS))
    ap.add_argument("--panels", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    lib.setup_torch()
    ref = lib.reference("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for index in args.panels:
            fixture = lib.panel(task, index)
            record, arrays = {"main": {}, "lengths": {}}, {}
            work = lib.Workload.from_panel(fixture)
            engine = lib.Engine(ref, work.y)
            old = np.load(lib.OLD / "results" / f"{task}-nodepfn-cuda-p{index}.npz", allow_pickle=False)
            for kind in ("native", "arrow"):
                op = lib.Operator(engine, work, "joint")
                with duckdb.connect(":memory:") as db:
                    db.execute("SET threads=1")
                    op.load(db)
                    op.register(db, kind)
                    out, times, udf, calls, _ = timed(op, db, MAIN, args.repeats, index)
                    entry = dict(times=times, udf=udf, calls=calls)
                    if kind == "arrow":
                        entry["phases"] = phases(op, db, ("separate", "union", "axis"), 3)
                    # Both interfaces must return the archived answers.
                    for plan in ("separate", "union", "axis"):
                        ref_p = old["sorted_attached_joint_4__" + plan]
                        assert np.array_equal(out[plan].argmax(1), ref_p.argmax(1))
                        # Separate and union repeat the archived arithmetic.  The padded engine packs
                        # lanes differently from the archived adapter, so only its distance to the
                        # separate plan is held to the tolerance.
                        bound = lib.TOL if plan != "axis" else 2 * lib.TOL
                        assert np.abs(out[plan] - ref_p).max() <= bound
                    assert np.abs(out["axis"] - out["separate"]).max() <= lib.TOL
                    entry["archive_max"] = {plan: float(np.abs(out[plan] - old["sorted_attached_joint_4__" + plan]).max())
                                            for plan in ("separate", "union", "axis")}
                    record["main"][kind] = entry
            for j, (name, sizes) in enumerate(PROFILES.items()):
                work = lib.Workload.from_panel(fixture, sizes=sizes)
                op = lib.Operator(engine, work, "joint")
                with duckdb.connect(":memory:") as db:
                    db.execute("SET threads=1")
                    op.load(db)
                    op.register(db, "arrow")
                    out, times, udf, calls, fallback = timed(op, db, RAGGED, args.repeats, index + j)
                checks = {plan: lib.endpoints(work.ids, work.rid, out["separate"], out[plan]) for plan in RAGGED[1:]}
                record["lengths"][name] = dict(sizes=sizes, times=times, calls=calls, checks=checks,
                                               fallback=fallback["axis-guard"])
                arrays.update({f"{name}__{plan}": out[plan] for plan in RAGGED})
                arrays[f"{name}__ids"], arrays[f"{name}__rid"] = work.ids, work.rid
            record["device"] = torch.cuda.get_device_name()
            record["duckdb"] = duckdb.__version__
            lib.dump(args.out / f"{task}-p{index}.json", record)
            np.savez_compressed(args.out / f"{task}-p{index}.npz", **arrays)
            m = record["main"]
            print(task, index, {k: round(float(np.median(m[k]["times"]["separate"]) / np.median(m[k]["times"]["axis"])), 2)
                                for k in m}, flush=True)


if __name__ == "__main__":
    main()
