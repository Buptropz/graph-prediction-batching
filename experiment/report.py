"""Recompute the manuscript's SQL tables and figures from per-row records."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = {"nodepfn": "NodePFN", "ephris": "Ephris", "pubmed": "PubMed", "cora": "Cora"}


def summary(records):
    out = []
    for task in ("cora", "pubmed"):
        for model in ("nodepfn", "ephris"):
            key = task + "-" + model
            data = json.loads((records / (key + ".json")).read_text())
            for k, v in data["code"].items():
                candidates = [HERE / k, HERE / "registered" / k]
                assert any(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == v
                           for p in candidates), f"Missing measured source revision for {k}"
            assert data["model"] == model and data["task"] == task and data["repeats"] == 3
            with np.load(records / (key + ".npz"), allow_pickle=False) as vectors:
                for recipe in ("joint", "support"):
                    cases = [c for c in data["cases"] if c["recipe"] == recipe]
                    assert {(c["panel"], c["mode"]) for c in cases} == {
                        (p, m) for p in range(5) for m in ["low", "high"]} and len(cases) == 10
                    moved, seen = set(), set()
                    counts = {"low": 0, "high": 0}
                    timings = {"separate": [], "pooled": []}
                    paired_ratio, all_tv = [], []
                    for c in cases:
                        tag = f"p{c['panel']}_{recipe}_{c['mode']}_"
                        n = c["n"]
                        ids = vectors[tag + "ids"][:n]
                        a, b = (vectors[tag + k][:n] for k in ("separate", "pooled"))
                        assert len(a) == len(b) == n == 24
                        changes = (a.argmax(1) != b.argmax(1))
                        changed_ids = ids[changes].tolist()
                        assert sorted(changed_ids) == c["moved"]
                        moved.update(changed_ids)
                        seen.update(ids)
                        counts[c["mode"]] += len(changed_ids)
                        all_tv.extend(.5 * np.abs(a-b).sum(1))
                        for plan in timings:
                            ts = c["timings_ms"][plan]
                            assert len(ts) == 3 and min(ts) > 0
                            timings[plan].extend(ts)
                        paired_ratio.extend(a/b for a, b in zip(c["timings_ms"]["separate"],
                                                               c["timings_ms"]["pooled"]))
                    assert len(seen) == 120
                    out.append(dict(task=task, model=model, recipe=recipe, low=counts["low"],
                                    high=counts["high"], unique=len(moved), n_unique=len(seen),
                                    comparisons=len(all_tv), tv=float(np.mean(all_tv)),
                                    separate_ms=statistics.median(timings["separate"]),
                                    pooled_ms=statistics.median(timings["pooled"]),
                                    paired_speedup=statistics.median(paired_ratio),
                                    timings_per_plan=len(paired_ratio), peak_rss=data["peak_rss_bytes"],
                                    max_parity=max(max(c["archive_max_abs"].values()) for c in cases)))
    return out


def figures(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    with np.load(records / "cora-nodepfn.npz") as z:
        tag = "p0_joint_low_"
        ids, a, b = (z[tag+k] for k in ("ids","separate","pooled"))
        valid = np.where(a[:24].argmax(1) != b[:24].argmax(1))[0]
        p = valid[0]
        names = sorted({row.split()[-1] for row in (ROOT/"data/raw/cora.content").open()})
        names = [s.replace("_"," ") for s in names]
        pos = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(3.31,2.02), layout="constrained")
        ax.barh(pos+.17, a[p], height=.32, color="#24445b", label="Separate")
        ax.barh(pos-.17, b[p], height=.32, color="#ba5320", label="Pooled")
        ax.set_yticks(pos,names,fontsize=7); ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, max(a[p].max(),b[p].max())*1.12)
        ax.invert_yaxis();ax.legend(loc="upper right",fontsize=7,frameon=False)
        fig.savefig(out/"witness.pdf");plt.close(fig)
        witness = dict(id=str(ids[p]), separate_class=names[a[p].argmax()], pooled_class=names[b[p].argmax()],
                       separate=a[p].tolist(), pooled=b[p].tolist())
    with (ROOT/"companion_dose/curves.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    fig,ax=plt.subplots(figsize=(3.33,2.12),layout="constrained")
    series=[("cora","Cora","#a44922","o"),("persrefex","PersRefEx","#2d5277","s"),
            ("pubmed","PubMed","#00857a","^"),("chameleon","Chameleon","#575966","D")]
    for task, label, color, marker in series:
        chosen=[r for r in rows if r["task"]==task and r["method"]=="ephris" and r["recipe"]=="support"]
        ax.plot(range(len(chosen)),[float(r["excess_TV"]) for r in chosen],marker=marker,
                color=color,linewidth=1.35,markersize=4,label=label)
    ax.set_xticks(range(5),["1","4","8","16","32"]);ax.set_xlabel("Distinct companions per batch")
    ax.set_ylabel("Mean excess TV");ax.axhline(0,color="#999999",linewidth=.65)
    ax.legend(frameon=False,fontsize=7,ncol=2,loc="upper left")
    fig.savefig(out/"dose.pdf");plt.close(fig)
    return witness


def table(rows):
    lines = [r"\begin{table*}[t]\centering",
             r"\caption{DuckDB separate versus pooled execution. Low and High each count changed focal papers out of 120. Either counts distinct papers affected in at least one condition, also out of 120. TV averages 240 paired focal exposures. Times are median milliseconds for 30 measured SQL executions per plan. Speedup is the median paired separate-to-pooled time ratio.}\label{tab:sql}",
             r"\begin{tabular}{lllrrrrrrr}\toprule",
             r"Dataset & Model & Preprocessing & Low & High & Either & Mean TV & Separate (ms) & Pooled (ms) & Speedup\\\midrule"]
    for r in rows:
        if r["task"] == "pubmed" and r["model"] == "nodepfn" and r["recipe"] == "joint":
            lines.append(r"\midrule")
        lines.append(" & ".join([NAMES[r["task"]], NAMES[r["model"]], r["recipe"].capitalize(),
                     str(r["low"]), str(r["high"]), str(r["unique"]), f"{r['tv']:.3f}",
                     f"{r['separate_ms']:.1f}", f"{r['pooled_ms']:.1f}",
                     f"{r['paired_speedup']:.2f}" + r"\(\times\)"]) + r"\\")
    return "\n".join([*lines, r"\bottomrule\end{tabular}\end{table*}", ""])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, default=HERE/"records")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figures", action="store_true")
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    rows=summary(a.records)
    (a.out/"sql-summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    (a.out/"sql-table.tex").write_text(table(rows))
    if a.figures:
        witness=figures(a.records,a.out)
        (a.out/"witness.json").write_text(json.dumps(witness,indent=2)+"\n")
    for r in rows:
        print(r["task"],r["model"],r["recipe"],r["low"],r["high"],r["unique"],
              f"{r['separate_ms']:.1f}",f"{r['pooled_ms']:.1f}",f"{r['paired_speedup']:.2f}",
              f"{r['max_parity']:.2g}")


if __name__=="__main__": main()
