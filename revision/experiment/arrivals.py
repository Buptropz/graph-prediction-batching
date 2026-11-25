"""Open-loop arrivals at one model operator: response time and answers of the three plans under load.

Requests of 16 uniformly sampled PubMed records arrive as a Poisson process.  The operator is
work-conserving: whenever it is free it takes the requests that are waiting, at most ``CAP`` of
them, and evaluates them in one model call (one request per call under the separate plan).  No call
is delayed to collect a larger batch.

The server is emulated in virtual time.  Arrival instants are drawn in advance, every model call is
executed and timed on the device, and the clock advances by the measured duration; idle periods are
skipped.  The measured call covers edge remapping, feature fitting, graph packing and the forward
pass, not the SQL stages around the operator.  A run stops as saturated once ``BACKLOG`` requests
wait.  Every answer is compared with the separate evaluation of the same request.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import lib
from scale import build

RATES = (5, 10, 20, 40, 80, 160, 320, 640)
CAP, BACKLOG, POOL, LENGTH = 32, 256, 256, 16


def call(engine, work, lanes, plan, recipe):
    """One model call for the given requests (arrays of row positions); ms and probabilities."""
    lib.sync(engine.device)
    tic = time.perf_counter_ns()
    groups = [np.concatenate(lanes)] if plan == "union" else lanes
    q, e = zip(*[work.part(g) for g in groups])
    p = engine.predict(work.tx, q, e, recipe)
    lib.sync(engine.device)
    return (time.perf_counter_ns() - tic) / 1e6, p


def compare(work, lanes, reference, p, acc, seen):
    rows = np.concatenate(lanes)
    d = lib.endpoints(work.ids[rows], work.rid[rows], reference[rows], p)
    for key in ("n", "flips", "threshold", "lists", "changed_lists"):
        acc[key] = acc.get(key, 0) + d[key]
    acc["max_abs"] = max(acc.get("max_abs", 0.0), d["max_abs"])
    acc["calls_over_tolerance"] = acc.get("calls_over_tolerance", 0) + int(d["max_abs"] > lib.TOL)
    start = 0
    for g in lanes:
        seen.setdefault(int(work.rid[g[0]]), set()).add(tuple(p[start:start + len(g)].argmax(1).tolist()))
        start += len(g)


def replay(engine, work, reqs, plan, recipe, rate, arrivals, reference):
    rng = np.random.default_rng([20261128, rate])           # the same arrival instants for every plan
    arrive = np.cumsum(rng.exponential(1.0 / rate, arrivals))
    cap = 1 if plan == "separate" else CAP
    clock, head, busy = 0.0, 0, 0.0
    done = np.full(arrivals, np.nan)
    batches, service, acc, seen, saturated = [], [], {}, {}, False
    while head < arrivals:
        clock = max(clock, arrive[head])
        waiting = int(np.searchsorted(arrive, clock, side="right")) - head
        if waiting > BACKLOG:
            saturated = True
            break
        batch = np.arange(head, head + min(waiting, cap))
        lanes = [reqs[i % len(reqs)] for i in batch]
        ms, p = call(engine, work, lanes, plan, recipe)
        compare(work, lanes, reference, p, acc, seen)
        clock += ms / 1e3
        busy += ms / 1e3
        done[batch] = clock
        batches.append(batch.tolist())
        service.append((len(batch), ms))
        head += len(batch)
    served = ~np.isnan(done)
    resp = (done[served] - arrive[served]) * 1e3
    size = np.asarray([s for s, _ in service])
    row = dict(plan=plan, rate=rate, arrivals=arrivals, served=int(served.sum()), saturated=saturated,
               response_ms={k: float(np.percentile(resp, v)) for k, v in (("p50", 50), ("p95", 95), ("p99", 99))},
               mean_response_ms=float(resp.mean()), mean_batch=float(size.mean()), max_batch=int(size.max()),
               calls=len(service), utilisation=float(busy / (clock - arrive[0])),
               varying=int(sum(len(v) > 1 for v in seen.values())), requests_seen=len(seen), **acc)
    return row, batches, service


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/arrivals.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arrivals", type=int, default=1000)
    ap.add_argument("--rates", type=int, nargs="+", default=list(RATES))
    ap.add_argument("--pool", type=int, default=POOL)
    args = ap.parse_args()
    lib.setup_torch()
    ref = lib.reference(args.device)
    work, _ = build("pubmed", lib.SUPPORT, LENGTH, args.pool, np.random.default_rng(20261128))
    engine = lib.Engine(ref, work.y)
    reqs = work.requests()
    reference = {}
    for recipe in ("joint", "support"):
        reference[recipe] = np.empty((len(work.ids), engine.classes))
        for g in reqs:
            reference[recipe][g] = call(engine, work, [g], "separate", recipe)[1]
    for k in (1, 2, 4, 8, 16, CAP):                         # warm every call shape before timing
        for plan in ("union", "axis"):
            call(engine, work, reqs[:k], plan, "joint")
    rows, table = [], {p: {} for p in lib.PLANS}
    for rate in args.rates:
        for plan in lib.PLANS:
            row, batches, service = replay(engine, work, reqs, plan, "joint", rate, args.arrivals, reference["joint"])
            for size, ms in service:
                table[plan].setdefault(size, []).append(ms)
            if plan == "union":                             # the same batches under the default fit, untimed
                acc, seen = {}, {}
                for batch in batches:
                    lanes = [reqs[i % len(reqs)] for i in batch]
                    compare(work, lanes, reference["support"], call(engine, work, lanes, plan, "support")[1], acc, seen)
                row["support_fit"] = dict(varying=int(sum(len(v) > 1 for v in seen.values())), **acc)
            rows.append(row)
            print(rate, plan, "saturated" if row["saturated"] else "", round(row["response_ms"]["p50"], 1),
                  round(row["response_ms"]["p99"], 1), round(row["mean_batch"], 2), row["flips"], row["n"],
                  row["varying"], flush=True)
            service_ms = {p: {str(k): dict(median=float(np.median(v)), calls=len(v)) for k, v in sorted(t.items())}
                          for p, t in table.items()}
            lib.dump(args.out, dict(device=args.device, cap=CAP, backlog=BACKLOG, pool=args.pool, length=LENGTH,
                                    support=lib.SUPPORT, cross_edges=work.cross, runs=rows, service_ms=service_ms))


if __name__ == "__main__":
    main()
