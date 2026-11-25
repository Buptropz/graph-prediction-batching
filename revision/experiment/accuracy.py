"""Accuracy and confidence of the archived predictions.

Held-out query labels were kept out of the fixtures and the plans.  This script joins them to the
stored probability vectors afterwards.  It reports the accuracy of every plan, how the records that
node union changed moved between right and wrong, and how confident the separate plan was about
the records that changed.
"""
import argparse
from pathlib import Path

import numpy as np

import lib

GRID = [(layout, topology, recipe) for layout in ("sorted", "random") for topology in ("attached", "cut")
        for recipe in ("joint", "support")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/accuracy.json")
    args = ap.parse_args()
    rows, margins = [], {}
    for model, device in (("nodepfn", "cuda"), ("ephris", "cpu")):
        plans = lib.PLANS if model == "nodepfn" else lib.PLANS[:2]
        for task in lib.TASKS:
            for layout, topology, recipe in GRID:
                row = dict(model=model, task=task, layout=layout, topology=topology, recipe=recipe, n=0,
                           correct={p: 0 for p in plans},
                           moves=dict(right_to_wrong=0, wrong_to_right=0, wrong_to_wrong=0))
                for index in range(5):
                    z = np.load(lib.OLD / "results" / f"{task}-{model}-{device}-p{index}.npz", allow_pickle=False)
                    key = f"{layout}_{topology}_{recipe}_4__"
                    ids = z[key + "ids"]
                    y = lib.truth(task, ids)
                    pred = {p: z[key + p].argmax(1) for p in plans}
                    row["n"] += len(ids)
                    for p in plans:
                        row["correct"][p] += int((pred[p] == y).sum())
                    moved = pred["separate"] != pred["union"]
                    a, b = pred["separate"][moved] == y[moved], pred["union"][moved] == y[moved]
                    row["moves"]["right_to_wrong"] += int((a & ~b).sum())
                    row["moves"]["wrong_to_right"] += int((~a & b).sum())
                    row["moves"]["wrong_to_wrong"] += int((~a & ~b).sum())
                    if (model, layout, topology, recipe) == ("nodepfn", "sorted", "attached", "joint"):
                        top = np.sort(z[key + "separate"], 1)
                        m = margins.setdefault(task, dict(changed=[], unchanged=[]))
                        m["changed"] += (top[moved, -1] - top[moved, -2]).tolist()
                        m["unchanged"] += (top[~moved, -1] - top[~moved, -2]).tolist()
                rows.append(row)
    lib.dump(args.out, dict(rows=rows, margins=margins, chance={t: 1 / c for t, c in lib.CLASSES.items()}))
    for r in rows:
        if (r["layout"], r["topology"]) == ("sorted", "attached"):
            print(r["model"], r["task"], r["recipe"], {p: round(v / r["n"], 3) for p, v in r["correct"].items()}, r["moves"])


if __name__ == "__main__":
    main()
