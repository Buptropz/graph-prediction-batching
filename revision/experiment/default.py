"""The released default configuration: three ensemble members.

The archived grid used one ensemble member.  This run repeats the fit-by-edge factorial on all
15 panels with the classifier's default of three members, two of which share a class and feature
shift and differ in a power transform.  Separate and node-union outputs are checked against the
released interface; the request-axis plan stacks 3 members x 4 requests in one forward.
"""
import argparse
from pathlib import Path

import numpy as np

import lib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/default")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    lib.setup_torch()
    ref = lib.reference(args.device, args.members)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {}
    for task in lib.TASKS:
        for index in range(5):
            fixture = lib.panel(task, index)
            for layout in ("sorted", "random"):
                for attached in (True, False):
                    work = lib.Workload.from_panel(fixture, layout, attached)
                    engine = lib.Engine(ref, work.y)
                    y = lib.truth(task, work.ids)
                    for recipe in ("joint", "support"):
                        out = {plan: lib.execute(engine, work, plan, recipe)[0] for plan in lib.PLANS}
                        row = dict(device=args.device, task=task, panel=index, layout=layout, recipe=recipe,
                                   topology="attached" if attached else "cut", members=args.members)
                        for plan in ("union", "axis"):
                            row[plan] = lib.endpoints(work.ids, work.rid, out["separate"], out[plan])
                        row["accuracy"] = {plan: int((out[plan].argmax(1) == y).sum()) for plan in lib.PLANS}
                        if layout == "sorted":
                            pub_sep = np.vstack([lib.public(ref, work.tx, work.y, *work.part(g), recipe)
                                                 for g in work.requests()])
                            pub_union = lib.public(ref, work.tx, work.y, *work.part(np.arange(len(work.ids))), recipe)
                            row["public"] = dict(separate=float(np.abs(pub_sep - out["separate"]).max()),
                                                 union=float(np.abs(pub_union - out["union"]).max()),
                                                 flips=int((pub_sep.argmax(1) != out["separate"].argmax(1)).sum()
                                                           + (pub_union.argmax(1) != out["union"].argmax(1)).sum()))
                        rows.append(row)
                        key = f"{task}-p{index}__{layout}_{row['topology']}_{recipe}__"
                        arrays.update({key + plan: out[plan] for plan in lib.PLANS})
                        arrays[key + "ids"] = work.ids
            print(task, index, flush=True)
    lib.dump(args.out / "summary.json", rows)
    np.savez_compressed(args.out / "probabilities.npz", **arrays)


if __name__ == "__main__":
    main()
