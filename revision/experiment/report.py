"""Tables, figures and numbers of the revised paper, generated from stored results only."""
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1795564800")   # fixed creation date: figures are byte-stable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parent
OLD = REV.parent / "upgrade"
RES = REV / "results"
FIG = REV / "source/figures"
TASKS = ["cora", "pubmed", "chameleon"]
NAMES = dict(cora="Cora", pubmed="PubMed", chameleon="Chameleon")
CLASSES = dict(cora=7, pubmed=3, chameleon=5)
BLUE, ORANGE, TEAL, INK, MUTED = "#355e80", "#b4582c", "#117c79", "#233744", "#6d7b82"
PLAN = dict(separate=(BLUE, "o", "Separate"), union=(ORANGE, "^", "Node union"), axis=(TEAL, "D", "Request axis"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK,
                     "axes.edgecolor": "#8b999e", "xtick.color": INK, "ytick.color": INK,
                     "text.color": INK})
MACROS = {}


def put(key, value):
    MACROS[key] = str(value)


def load(name):
    return json.loads((RES / name).read_text())


def med(values):
    return float(np.median(values))


def select(rows, **criteria):
    return [r for r in rows if all(r[k] == v for k, v in criteria.items())]


def total(rows, key):
    return sum(r[key] for r in rows)


def table(name, spec, head, lines):
    body = "\n".join(line if isinstance(line, str) else " & ".join(map(str, line)) + "\\\\" for line in lines)
    (FIG / (name + ".tex")).write_text("\\begin{tabular}{" + spec + "}\\toprule\n" + head + "\\\\ \\midrule\n"
                                       + body + "\n\\bottomrule\\end{tabular}\n")


def sci(v):
    mantissa, exponent = f"{v:.2e}".split("e")
    return mantissa + r"\times10^{" + str(int(exponent)) + "}"


def pct(a, b, digits=1):
    return f"{100 * a / b:.{digits}f}"


# ------------------------------------------------------------------------------------------------
def answers(audit):
    """Main table: archived single-member grid next to the release default."""
    gpu = select(audit["rows"], model="nodepfn", device="cuda", fan=4, layout="sorted", topology="attached")
    default = load("default/summary.json")
    lines, sums = [], collections.defaultdict(lambda: collections.Counter())
    configs = [("1 member, joint fit", lambda t, plan: select(gpu, task=t, recipe="joint", plan=plan), "threshold_symmetric"),
               ("default (3, support fit)", None, "threshold")]
    for t in TASKS:
        for label, pick, thr in configs:
            if pick:
                u, a = pick(t, "union"), pick(t, "axis")
                cell = dict(uf=total(u, "flips"), af=total(a, "flips"), ut=total(u, thr), at=total(a, thr),
                            ul=total(u, "changed_lists"), al=total(a, "changed_lists"), L=total(u, "lists"))
            else:
                d = select(default, task=t, layout="sorted", topology="attached", recipe="support")
                cell = dict(uf=sum(r["union"]["flips"] for r in d), af=sum(r["axis"]["flips"] for r in d),
                            ut=sum(r["union"]["threshold"] for r in d), at=sum(r["axis"]["threshold"] for r in d),
                            ul=sum(r["union"]["changed_lists"] for r in d), al=sum(r["axis"]["changed_lists"] for r in d),
                            L=sum(r["union"]["lists"] for r in d))
            sums[label].update(cell)
            lines.append([NAMES[t] if label == configs[0][0] else "", label, cell["uf"], cell["af"], cell["ut"], cell["at"],
                          f"{cell['ul']}/{cell['L']}", f"{cell['al']}/{cell['L']}"])
    lines.append("\\midrule")
    for label, _, _ in configs:
        c = sums[label]
        lines.append(["All" if label == configs[0][0] else "", label, c["uf"], c["af"], c["ut"], c["at"],
                      f"{c['ul']}/{c['L']}", f"{c['al']}/{c['L']}"])
    table("membership", "llrrrrrr",
          "Corpus & NodePFN setting & \\multicolumn{2}{c}{Changed class} & \\multicolumn{2}{c}{Threshold pairs} & "
          "\\multicolumn{2}{c}{Changed top-three lists} \\\\ \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8} "
          "& & Union & Axis & Union & Axis & Union & Axis", lines)
    panels = {"Main": [], "Def": []}
    for t in TASKS:
        per = [r["flips"] for r in select(gpu, task=t, recipe="joint", plan="union")]
        per_d = [r["union"]["flips"] for r in select(default, task=t, layout="sorted", topology="attached", recipe="support")]
        assert len(per) == len(per_d) == 5
        put("Panel" + NAMES[t], f"{min(per)}--{max(per)}"); put("PanelDef" + NAMES[t], f"{min(per_d)}--{max(per_d)}")
        panels["Main"].append(per); panels["Def"].append(per_d)
    # Percentile interval of the pooled rate when the five panels of each corpus are resampled.
    rng = np.random.default_rng(20261125)
    for key, counts in panels.items():
        counts = np.asarray(counts)
        draws = sum(c[rng.integers(0, 5, (10000, 5))].sum(1) for c in counts) / 960
        low, high = np.percentile(draws, [2.5, 97.5])
        put(key + "CILow", f"{100 * low:.0f}"); put(key + "CIHigh", f"{100 * high:.0f}")
    c = sums[configs[1][0]]
    put("DefFlips", c["uf"]); put("DefAxis", c["af"]); put("DefThreshold", c["ut"]); put("DefLists", c["ul"])
    put("DefAxisLists", c["al"]); put("DefAxisThreshold", c["at"])
    put("DefFlipPct", pct(c["uf"], 960, 0)); put("MainFlipPct", pct(int(MACROS_OLD["MainFlips"]), 960, 0))
    def agg(**kw):
        d = select(default, layout="sorted", **kw)
        return sum(r["union"]["flips"] for r in d)
    put("DefJointAttached", agg(topology="attached", recipe="joint"))
    put("DefJointCut", agg(topology="cut", recipe="joint"))
    put("DefSupportCut", agg(topology="cut", recipe="support"))
    put("DefRandom", sum(r["union"]["flips"] for r in select(default, layout="random", topology="attached", recipe="support")))
    put("DefConditions", len(default))
    put("DefAxisAll", sum(r["axis"]["flips"] + r["axis"]["threshold"] for r in default))
    put("DefAxisListsAll", sum(r["axis"]["changed_lists"] for r in default))
    put("DefAxisMax", sci(max(r["axis"]["max_abs"] for r in default)) if max(r["axis"]["max_abs"] for r in default) > 0 else "0")
    put("DefPublic", sci(max(max(r["public"].values()) for r in default if "public" in r)))
    put("DefPublicChecks", 2 * sum("public" in r for r in default))
    return default


def accuracy(default):
    acc = load("accuracy.json")
    rows = acc["rows"]
    lines = []
    tot = collections.Counter()
    for t in TASKS:
        main = select(rows, model="nodepfn", task=t, layout="sorted", topology="attached", recipe="joint")[0]
        eph = select(rows, model="ephris", task=t, layout="sorted", topology="attached", recipe="joint")[0]
        d = select(default, task=t, layout="sorted", topology="attached", recipe="support")
        ds, du = sum(r["accuracy"]["separate"] for r in d), sum(r["accuracy"]["union"] for r in d)
        m = main["moves"]
        lines.append([NAMES[t], f"{100 / CLASSES[t]:.1f}", pct(main["correct"]["separate"], 320), pct(main["correct"]["union"], 320),
                      f"{m['right_to_wrong']}/{m['wrong_to_right']}/{m['wrong_to_wrong']}", pct(ds, 320), pct(du, 320),
                      pct(eph["correct"]["separate"], 320), pct(eph["correct"]["union"], 320)])
        tot.update(rw=m["right_to_wrong"], wr=m["wrong_to_right"], ww=m["wrong_to_wrong"])
        put("Acc" + NAMES[t] + "Sep", pct(main["correct"]["separate"], 320))
        put("Acc" + NAMES[t] + "Union", pct(main["correct"]["union"], 320))
    table("accuracy", "lrrrcrrrr",
          " & & \\multicolumn{3}{c}{NodePFN, 1 member, joint} & \\multicolumn{2}{c}{NodePFN default} & \\multicolumn{2}{c}{Ephris} \\\\ "
          "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9} Corpus & Chance & Sep. & Union & Moves & Sep. & Union & Sep. & Union", lines)
    put("MovesRW", tot["rw"]); put("MovesWR", tot["wr"]); put("MovesWW", tot["ww"])
    changed = np.concatenate([acc["margins"][t]["changed"] for t in TASKS])
    same = np.concatenate([acc["margins"][t]["unchanged"] for t in TASKS])
    put("MarginChanged", f"{np.median(changed):.2f}"); put("MarginSame", f"{np.median(same):.2f}")
    put("MarginHigh", int((changed > .5).sum())); put("MarginHighPct", pct((changed > .5).sum(), len(changed), 0))
    fig, ax = plt.subplots(figsize=(3.37, 1.95))
    fig.subplots_adjust(left=.15, right=.97, bottom=.21, top=.95)
    for data, color, label, style in ((same, BLUE, f"unchanged ({len(same)})", "-"), (changed, ORANGE, f"changed by node union ({len(changed)})", "-")):
        x = np.sort(data)
        ax.step(np.r_[0, x, 1], np.r_[0, np.arange(1, len(x) + 1) / len(x), 1], where="post", color=color, lw=1.6)
    ax.text(.07, .90, f"changed ({len(changed)})", color=INK, fontsize=7, ha="left")
    ax.plot([.03, .06], [.925, .925], color=ORANGE, lw=1.6)
    ax.text(.56, .10, f"unchanged ({len(same)})", color=INK, fontsize=7, ha="left")
    ax.plot([.52, .55], [.125, .125], color=BLUE, lw=1.6)
    ax.axvline(.5, color="#9aa7ac", lw=.6, ls=":")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Separate plan: top-1 minus top-2 probability", ylabel="Share of records")
    ax.grid(axis="y", lw=.4, alpha=.4)
    fig.savefig(FIG / "margins.pdf"); plt.close(fig)


def adjacency():
    rows = [r for r in load("dose.json") if r["layout"] == "sorted"]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.05), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.21, top=.86, wspace=.08)
    styles = {"joint": (ORANGE, "o", "joint fit"), "support": (TEAL, "D", "support fit")}
    for ax, t in zip(axes, TASKS):
        for recipe, (color, marker, label) in styles.items():
            xs, ys = [], []
            for level in ("0", "16", "32", "48", "64"):
                d = select(rows, task=t, recipe=recipe, level=level)
                xs.append(np.mean([r["touching"] for r in d])); ys.append(100 * sum(r["union"]["flips"] for r in d) / 320)
            ax.plot(xs, ys, color=color, marker=marker, ms=4.2, lw=1.5, label=label, markeredgecolor="white", markeredgewidth=.6)
            d = select(rows, task=t, recipe=recipe, level="uniform")
            ax.scatter([np.mean([r["touching"] for r in d])], [100 * sum(r["union"]["flips"] for r in d) / 320], s=46,
                       facecolors="white", edgecolors=color, marker=marker, lw=1.3, zorder=5)
        ax.set_title(NAMES[t], loc="left", weight="bold", fontsize=9)
        ax.set(xticks=[0, 16, 32, 48, 64], xlim=(-3, 67), ylim=(-2, 45))
        ax.grid(axis="y", lw=.4, alpha=.4)
    axes[1].set_xlabel("Query records with an edge to the support (of 64 per panel)")
    axes[0].set_ylabel("Changed class (%)")
    axes[0].legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=1.6)
    axes[2].text(.98, .04, "hollow: uniformly sampled queries", transform=axes[2].transAxes, ha="right", fontsize=6.5, color=MUTED)
    fig.savefig(FIG / "adjacency.pdf"); plt.close(fig)
    every = load("dose.json")
    def count(**kw):
        d = select(rows, **kw)
        return sum(r["union"]["flips"] for r in d)
    put("DoseZeroSupport", count(recipe="support", level="0")); put("DoseZeroJoint", count(recipe="joint", level="0"))
    put("DoseUniformSupport", count(recipe="support", level="uniform")); put("DoseUniformJoint", count(recipe="joint", level="uniform"))
    for t in TASKS:
        d = select(rows, task=t, recipe="support", level="uniform")
        put("Uniform" + NAMES[t], sum(r["union"]["flips"] for r in d)); put("Touch" + NAMES[t], sum(r["touching"] for r in d))
    put("DoseSixteenSupport", count(recipe="support", level="16")); put("DoseFullSupport", count(recipe="support", level="64"))
    full = select(rows, recipe="support", level="64")
    put("DoseFullAccSep", pct(sum(r["accuracy"]["separate"] for r in full), 960)); put("DoseFullAccUnion", pct(sum(r["accuracy"]["union"] for r in full), 960))
    zero = select(rows, recipe="support", level="0")
    put("DoseZeroAcc", pct(sum(r["accuracy"]["separate"] for r in zero), 960))
    put("DoseConditions", len(every)); put("DoseAxisChanges", sum(r["axis"]["flips"] + r["axis"]["threshold"] for r in every))
    put("DoseAxisLists", sum(r["axis"]["changed_lists"] for r in every)); put("DoseLists", sum(r["axis"]["lists"] for r in every))
    put("DoseAxisMax", sci(max(r["axis"]["max_abs"] for r in every))); put("DoseAxisOver", sum(r["axis"]["max_abs"] > 1e-5 for r in every))


def vectors():
    c = load("chunks.json")
    v, b = c["versus_base"], c["vectors"]
    def sizes(key):
        s = b[key]
        return f"{len(s)}$\\times${s[0]:,}" if len(set(s)) == 1 else "shifted"
    lines = [["Same query repeated", f"{v['repeat']['rows']:,}", v["repeat"]["flips"], v["repeat"]["threshold"]],
             ["Four threads", f"{v['four threads']['rows']:,}", v["four threads"]["flips"], v["four threads"]["threshold"]],
             ["Rows stored in another order", f"{v['row order']['rows']:,}", v["row order"]["flips"], f'{v["row order"]["threshold"]:,}'],
             ["100 unrelated rows stored first", f"{v['100 rows ahead']['rows']:,}", v["100 rows ahead"]["flips"], v["100 rows ahead"]["threshold"]],
             ["Filter before vs.\\ after scoring", f"{c['filter_pair']['rows']:,}", c["filter_pair"]["flips"], c["filter_pair"]["threshold"]],
             ["One request per analyst", f"{v['per analyst']['rows']:,}", v["per analyst"]["flips"], f'{v["per analyst"]["threshold"]:,}']]
    table("vectors", "lrrr", "Compared with the base query & Rows & Class & Pairs", lines)
    put("VecRows", f"{c['rows']:,}"); put("VecOrder", v["row order"]["flips"]); put("VecOrderPct", pct(v["row order"]["flips"], c["rows"]))
    put("VecFilter", c["filter_pair"]["flips"]); put("VecFilterRows", f"{c['filter_pair']['rows']:,}")
    put("VecPrefix", v["100 rows ahead"]["flips"]); put("VecAnalyst", v["per analyst"]["flips"])
    put("VecAnalystPct", pct(v["per analyst"]["flips"], c["rows"]))
    put("VecAccBase", f"{100 * c['accuracy']['base']:.1f}"); put("VecAccAnalyst", f"{100 * c['accuracy']['per analyst']:.1f}")


def networks():
    rows = load("gnn.json")
    lines = []
    for kind, label in (("gcn", "GCN"), ("sage", "GraphSAGE")):
        for layers in (1, 2):
            cells = []
            for t in TASKS:
                d = select(rows, model=kind, layers=layers, task=t, topology="attached")
                f = [r["union"]["flips"] for r in d]
                cells.append(str(f[0]) if len(set(f)) == 1 else f"{min(f)}--{max(f)}")
            cut = sum(r["union"]["flips"] for r in select(rows, model=kind, layers=layers, topology="cut"))
            lines.append([label, layers] + cells + [cut])
    table("networks", "lrrrrr", "Network & Layers & Cora & PubMed & Chameleon & Cut", lines)
    put("GnnAxisFlips", sum(r["axis"]["flips"] for r in rows)); put("GnnAxisMax", sci(max(r["axis"]["max_abs"] for r in rows)) if max(r["axis"]["max_abs"] for r in rows) > 0 else "0")
    put("GnnCut", sum(r["union"]["flips"] for r in select(rows, topology="cut")))
    two = select(rows, layers=2, topology="attached")
    put("GnnTwoMax", max(r["union"]["flips"] for r in two)); put("GnnTwoMin", min(r["union"]["flips"] for r in two))
    put("GnnSageOne", sum(r["union"]["flips"] for r in select(rows, model="sage", layers=1)))
    put("GnnGcnOneMax", max(r["union"]["flips"] for r in select(rows, model="gcn", layers=1, topology="attached")))
    acc = [r["accuracy"]["separate"] / r["n"] for r in select(rows, layers=2, topology="attached", task="cora")]
    put("GnnCoraAcc", f"{100 * np.mean(acc):.0f}")


def cost():
    runs = {t: [json.loads(p.read_text()) for p in sorted((RES / "timing").glob(f"{t}-p*.json"))] for t in TASKS}
    assert all(len(v) == 5 for v in runs.values())
    lines, speed, speed_native = [], {}, {}
    for t in TASKS:
        row = [NAMES[t]]
        for kind in ("native", "arrow"):
            m = {p: med([med(r["main"][kind]["times"][p]) for r in runs[t]]) for p in ("separate", "union", "axis")}
            ratio = med([med(np.array(r["main"][kind]["times"]["separate"]) / np.array(r["main"][kind]["times"]["axis"])) for r in runs[t]])
            row += [f"{m['separate']:.0f}", f"{m['union']:.0f}", f"{m['axis']:.0f}", f"{ratio:.2f}"]
            (speed if kind == "arrow" else speed_native)[t] = ratio
        lines.append(row)
    table("times", "lrrrrrrrr", " & \\multicolumn{4}{c}{Row interface} & \\multicolumn{4}{c}{Arrow interface} \\\\ "
          "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9} Corpus & Sep. & Union & Axis & Ratio & Sep. & Union & Axis & Ratio", lines)
    put("ArrowLow", f"{min(speed.values()):.2f}"); put("ArrowHigh", f"{max(speed.values()):.2f}")
    put("RowLow", f"{min(speed_native.values()):.2f}"); put("RowHigh", f"{max(speed_native.values()):.2f}")
    every = [r for v in runs.values() for r in v]
    def pooled(kind, plan):
        return med([med(r["main"][kind]["times"][plan]) for r in every])
    put("ArrowTwo", f"{med([med(np.array(r['main']['arrow']['times']['separate']) / np.array(r['main']['arrow']['times']['axis-2'])) for r in every]):.2f}")
    phase = {p: {k: med([r["main"]["arrow"]["phases"][p][k] for r in every]) for k in ("fit", "graph", "forward", "operator", "engine", "total")}
             for p in ("separate", "union", "axis")}
    put("ForwardCall", f"{phase['union']['forward']:.0f}"); put("FitCall", f"{phase['union']['fit']:.1f}")
    put("AxisFit", f"{phase['axis']['fit']:.1f}"); put("AxisGraph", f"{phase['axis']['graph']:.1f}")
    put("AxisForward", f"{phase['axis']['forward']:.0f}"); put("UnionForward", f"{phase['union']['forward']:.0f}")
    put("EngineMs", f"{phase['axis']['engine']:.1f}")
    over = pooled("arrow", "axis") - pooled("arrow", "union")
    put("IsoMs", f"{over:.0f}"); put("IsoPct", f"{100 * over / pooled('arrow', 'union'):.0f}")
    put("ArrowAxis", f"{pooled('arrow', 'axis'):.0f}"); put("ArrowUnion", f"{pooled('arrow', 'union'):.0f}"); put("ArrowSep", f"{pooled('arrow', 'separate'):.0f}")
    convert = {t: med([med(r["main"]["native"]["times"]["axis"]) - med(r["main"]["arrow"]["times"]["axis"]) for r in runs[t]]) for t in TASKS}
    put("ConvertLow", f"{min(convert.values()):.0f}"); put("ConvertHigh", f"{max(convert.values()):.0f}")
    put("ConvertShare", f"{100 * med([1 - med(r['main']['arrow']['times']['axis']) / med(r['main']['native']['times']['axis']) for r in every]):.0f}")
    put("ArchiveAxisMax", sci(max(r["main"]["arrow"]["archive_max"]["axis"] for r in every)))

    fig, ax = plt.subplots(figsize=(3.37, 1.95))
    fig.subplots_adjust(left=.34, right=.97, bottom=.19, top=.80)
    parts = [("forward", "Neural forward", "#355e80"), ("fit", "Feature fit", "#7fa3bf"), ("graph", "Graph packing", "#c9d6de"),
             ("rest", "Engine and interface", "#b4582c")]
    labels, y = [], 0
    for kind, name in (("native", "Row"), ("arrow", "Arrow")):
        for plan in ("separate", "union", "axis"):
            tot = pooled(kind, plan)
            seg = dict(forward=phase[plan]["forward"], fit=phase[plan]["fit"], graph=phase[plan]["graph"])
            seg["rest"] = max(tot - sum(seg.values()), 0)
            left = 0
            for key, _, color in parts:
                ax.barh(y, seg[key], left=left, color=color, height=.62, edgecolor="white", linewidth=.8)
                left += seg[key]
            ax.text(left + 4, y, f"{tot:.0f}", va="center", fontsize=6.8)
            labels.append(f"{name} · {PLAN[plan][2]}")
            y += 1
        y += .45
    ax.set_yticks([0, 1, 2, 3.45, 4.45, 5.45], labels, fontsize=6.8)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set(xlabel="Latency of the SQL query (ms)", xlim=(0, 1.12 * max(pooled("native", "separate"), 1)))
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    fig.legend(handles, [n for _, n, _ in parts], loc="upper center", ncol=2, fontsize=6.5, frameon=False, handlelength=1,
               labelspacing=.25, columnspacing=1.2)
    fig.savefig(FIG / "cost.pdf"); plt.close(fig)

    lines = []
    names = dict(equal="16/16/16/16", paired="8/24/8/24", distinct="7/13/19/25", skewed="4/8/20/32")
    worst, over_tol, lists_pad, lists_guard, fall = 0., 0, 0, 0, 0
    guard_speeds = []
    for key, label in names.items():
        def ms(plan):
            return med([med(r["lengths"][key]["times"][plan]) for r in every])
        def ratio(plan):
            return med([med(np.array(r["lengths"][key]["times"]["separate"]) / np.array(r["lengths"][key]["times"][plan])) for r in every])
        calls = every[0]["lengths"][key]["calls"]["axis-length"]
        chk = lambda plan, k: sum(r["lengths"][key]["checks"][plan][k] for r in every)
        fb = sum(len(r["lengths"][key]["fallback"]) for r in every)
        lines.append([label, f"{ms('separate'):.0f}", calls, f"{ratio('axis-length'):.2f}", f"{ms('axis'):.0f}", f"{ratio('axis'):.2f}",
                      chk("axis", "changed_lists"), f"{fb}/60", f"{ratio('axis-guard'):.2f}", chk("axis-guard", "changed_lists")])
        worst = max(worst, max(r["lengths"][key]["checks"]["axis"]["max_abs"] for r in every))
        over_tol += sum(r["lengths"][key]["checks"]["axis"]["max_abs"] > 1e-5 for r in every)
        lists_pad += chk("axis", "changed_lists"); lists_guard += chk("axis-guard", "changed_lists"); fall += fb
        assert chk("axis", "flips") == chk("axis", "threshold") == chk("axis-length", "flips") == 0
        guard_speeds.append(ratio('axis-guard'))
        if key == "distinct":
            put("PadSpeed", f"{ratio('axis'):.2f}"); put("LenSpeed", f"{ratio('axis-length'):.2f}")
    table("lengths", "lrrrrrrrrr", " & & \\multicolumn{2}{c}{Per length} & \\multicolumn{3}{c}{Padded} & \\multicolumn{3}{c}{Padded + guard} \\\\ "
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10} Lengths & Sep. & Calls & Speed & ms & Speed & Lists & Re-run & Speed & Lists", lines)
    put("GuardSpeedLow", f"{min(guard_speeds):.2f}"); put("GuardSpeedHigh", f"{max(guard_speeds):.2f}")
    put("PadMax", sci(worst)); put("PadOver", over_tol); put("PadCases", 4 * len(every)); put("PadLists", lists_pad)
    put("GuardLists", lists_guard); put("GuardFallback", fall); put("GuardRequests", 4 * 4 * len(every))
    put("GuardPct", pct(fall, 4 * 4 * len(every), 0)); put("PadListTotal", 4 * 300)


def scale():
    path = RES / "scale.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7, 1.9), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.21, top=.86, wspace=.08)
    shades = {16: "#9fc9c7", 64: "#4da3a0", 256: "#0f6e6b"}
    for ax, s in zip(axes, (32, 128, 512)):
        ax.axhline(1, color="#78858c", lw=.8, ls="--")
        for q, color in shades.items():
            d = sorted(select(rows, support=s, request=q), key=lambda r: r["fan"])
            xs = [r["fan"] for r in d]
            ax.plot(xs, [med(r["times"]["separate"]) / med(r["times"]["axis"]) for r in d], color=color, marker="D", ms=3.8, lw=1.5,
                    markeredgecolor="white", markeredgewidth=.5, label=f"{q} records per request")
            ax.plot(xs, [med(r["times"]["separate"]) / med(r["times"]["union"]) for r in d], color=color, lw=.9, ls=":")
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
        ax.set_title(f"Support of {s}", loc="left", weight="bold", fontsize=9)
        ax.grid(axis="y", lw=.4, alpha=.4)
    axes[1].set_xlabel("Requests per call")
    axes[0].set_ylabel("Separate / plan latency")
    axes[0].legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.6, title="Request axis (solid)", title_fontsize=6.5)
    axes[2].text(.98, .12, "dotted: node union", transform=axes[2].transAxes, ha="right", fontsize=6.5, color=MUTED)
    fig.savefig(FIG / "scale.pdf"); plt.close(fig)
    best = max(rows, key=lambda r: med(r["times"]["separate"]) / med(r["times"]["axis"]))
    put("ScaleBest", f"{med(best['times']['separate']) / med(best['times']['axis']):.1f}")
    put("ScaleBestCfg", f"{best['fan']} requests of {best['request']} records, support {best['support']}")
    big = select(rows, support=512, request=256, fan=32)[0]
    put("ScaleBigAxis", f"{med(big['times']['separate']) / med(big['times']['axis']):.1f}")
    put("ScaleBigUnion", f"{med(big['times']['separate']) / med(big['times']['union']):.1f}")
    put("ScaleBigAxisMem", f"{big['memory_mib']['axis'] / 1024:.1f}"); put("ScaleBigUnionMem", f"{big['memory_mib']['union'] / 1024:.1f}")
    put("ScaleBigSepMem", f"{big['memory_mib']['separate'] / 1024:.2f}")
    put("ScaleBigAxisMs", f"{med(big['times']['axis']):.0f}"); put("ScaleBigUnionMs", f"{med(big['times']['union']):.0f}")
    put("ScaleBigSepMs", f"{med(big['times']['separate']):.0f}")
    ratio = [med(r["times"]["axis"]) / med(r["times"]["union"]) for r in rows]
    put("ScaleAxisUnionLow", f"{min(ratio):.2f}"); put("ScaleAxisUnionHigh", f"{max(ratio):.2f}")
    put("ScaleConfigs", len(rows)); put("ScaleExposures", f"{sum(r['joint']['axis']['n'] for r in rows):,}")
    put("ScaleAxisFlips", sum(r[k]["axis"]["flips"] for r in rows for k in ("joint", "support_fit")))
    put("ScaleAxisThreshold", sum(r[k]["axis"]["threshold"] for r in rows for k in ("joint", "support_fit")))
    put("ScaleAxisMax", sci(max(r[k]["axis"]["max_abs"] for r in rows for k in ("joint", "support_fit"))))
    put("ScaleAxisOver", sum(r[k]["axis"]["max_abs"] > 1e-5 for r in rows for k in ("joint", "support_fit")))
    n = sum(r["joint"]["union"]["n"] for r in rows)
    put("ScaleUnionJoint", pct(sum(r["joint"]["union"]["flips"] for r in rows), n)); put("ScaleUnionSupport", pct(sum(r["support_fit"]["union"]["flips"] for r in rows), n))
    large = select(rows, support=512)
    put("ScaleAccLarge", f"{100 * np.mean([r['accuracy']['separate'] for r in large]):.0f}")
    put("ScaleAccSmall", f"{100 * np.mean([r['accuracy']['separate'] for r in select(rows, support=32)]):.0f}")
    put("ScaleUnionLarge", pct(sum(r["support_fit"]["union"]["flips"] for r in large), sum(r["support_fit"]["union"]["n"] for r in large)))
    small = select(rows, support=32)
    def rate(d, fit):
        return pct(sum(r[fit]["union"]["flips"] for r in d), sum(r[fit]["union"]["n"] for r in d))
    put("ScaleUnionSmall", rate(small, "support_fit")); put("ScaleUnionSmallJoint", rate(small, "joint")); put("ScaleUnionLargeJoint", rate(large, "joint"))
    put("ScaleAccLargeUnion", f"{100 * np.mean([r['accuracy']['union'] for r in large]):.0f}")
    put("ScaleCrossSmall", f"{sum(r['cross_edges'] for r in small):,}"); put("ScaleCrossLarge", f"{sum(r['cross_edges'] for r in large):,}")
    def speed(d, plan):
        v = [med(r["times"]["separate"]) / med(r["times"][plan]) for r in d]
        return f"{min(v):.1f}", f"{max(v):.1f}"
    two, top = select(rows, fan=2), select(rows, fan=32)
    for name, d in (("Two", two), ("Top", top)):
        for plan, tag in (("axis", ""), ("union", "Union")):
            lo, hi = speed(d, plan)
            put(f"Scale{name}{tag}Low", lo); put(f"Scale{name}{tag}High", hi)
    lane = [(med(select(rows, support=r["support"], request=r["request"], fan=32)[0]["times"]["axis"]) - med(r["times"]["axis"])) / 16
            for r in select(rows, fan=16)]
    put("ScaleLaneLow", f"{min(lane):.1f}"); put("ScaleLaneHigh", f"{max(lane):.1f}")
    put("ScaleSepCall", f"{np.median([med(r['times']['separate']) / r['fan'] for r in rows]):.0f}")
    put("ScaleUnionFlat", f"{np.median([med(r['times']['union']) for r in rows]):.0f}")
    put("ScaleAxisLists", sum(r[k]["axis"]["changed_lists"] for r in rows for k in ("joint", "support_fit")))
    put("ScaleLists", f"{sum(r[k]['axis']['lists'] for r in rows for k in ('joint', 'support_fit')):,}")
    wide = select(rows, support=512, fan=32)
    put("ScaleWideAxisMem", f"{min(r['memory_mib']['axis'] for r in wide):.0f}--{max(r['memory_mib']['axis'] for r in wide):.0f}")
    put("ScaleWideUnionMem", f"{min(r['memory_mib']['union'] for r in wide):.0f}--{max(r['memory_mib']['union'] for r in wide):.0f}")
    put("ScaleRepeats", len(rows[0]["times"]["axis"]))


def others():
    """Ephris and GraphAny: node union and the lane plan on the 15 panels (CPU)."""
    eph = load("ephris_axis.json")["summary"]
    any_ = load("graphany.json")
    pooled = any_["summary"]["pooled"]
    lines = []
    def cells(u, a):
        return [u["flips"], a["flips"], u["threshold"], a["threshold"], f"{u['changed_lists']}/{u['lists']}", f"{a['changed_lists']}/{a['lists']}"]
    for recipe, label in (("joint", "call"), ("support", "support")):
        for top in ("attached", "cut"):
            d = eph[f"sorted_{top}_{recipe}"]
            lines.append(["Ephris" if (recipe, top) == ("joint", "attached") else "", f"{label} fit", top] + cells(d["union"], d["axis"]))
    lines.append("\\midrule")
    for feats, label in (("panel", "15 features"), ("raw", "raw features")):
        for top in ("attached", "cut"):
            d = select(pooled, features=feats, layout="sorted", topology=top, recipe="joint", fan=4)[0]
            lines.append(["GraphAny" if (feats, top) == ("panel", "attached") else "", label, top] + cells(d["union"], d["axis"]))
    table("others", "lllrrrrrr",
          " & & & \\multicolumn{2}{c}{Class} & \\multicolumn{2}{c}{Pairs} & \\multicolumn{2}{c}{Top-three lists} \\\\ "
          "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9} Model & Input & Edges & Union & Axis & Union & Axis & Union & Axis", lines)
    a = eph["all"]
    put("EphConditions", a["axis"]["conditions"]); put("EphExposures", f"{a['axis']['n']:,}")
    put("EphAxisFlips", a["axis"]["flips"] + a["axis"]["threshold"]); put("EphAxisLists", a["axis"]["changed_lists"])
    put("EphLists", f"{a['axis']['lists']:,}"); put("EphAxisMax", sci(a["axis"]["max_abs"])); put("EphAxisOver", a["axis"]["over_tolerance"])
    put("EphUnionFlips", f"{a['union']['flips']:,}"); put("EphCutSupport", eph["sorted_cut_support"]["union"]["flips"])
    put("EphOwnArchive", sci(max(a["own_separate_vs_archive"]["max_abs"], a["own_union_vs_archive"]["max_abs"])))
    put("EphOwnArchiveLists", a["own_separate_vs_archive"]["changed_lists"])
    assert a["own_separate_vs_archive"]["flips"] == a["own_union_vs_archive"]["flips"] == 0
    put("EphChamLists", eph["chameleon"]["axis"]["changed_lists"]); put("EphChamListTotal", eph["chameleon"]["axis"]["lists"])
    assert eph["cora"]["axis"]["changed_lists"] == eph["pubmed"]["axis"]["changed_lists"] == 0
    put("EphSepMs", f"{a['latency_ms']['separate']:.0f}"); put("EphUnionMs", f"{a['latency_ms']['union']:.0f}"); put("EphAxisMs", f"{a['latency_ms']['axis']:.0f}")
    put("EphAxisSpeed", f"{a['speedup_axis_over_separate']:.2f}"); put("EphUnionSpeed", f"{a['speedup_union_over_separate']:.2f}")
    put("EphAxisOverUnion", f"{a['latency_ms']['axis'] / a['latency_ms']['union']:.1f}")
    put("EphAccSep", f"{100 * a['accuracy']['separate']:.1f}"); put("EphAccUnion", f"{100 * a['accuracy']['union']:.1f}")
    # Boundary gaps of the Ephris lists: g between third and fourth score, delta the largest difference in the condition.
    z = np.load(RES / "ephris_axis.npz")
    certified = changed = certified_changed = ties = 0
    widest = 0.0
    for key in [k[:-len("__separate")] for k in z.files if k.endswith("__separate")]:
        sep, axis, ids = z[key + "__separate"], z[key + "__axis"], z[key + "__ids"]
        delta = np.abs(sep - axis).max()
        for r in range(4):
            rows = slice(16 * r, 16 * r + 16)
            for c in range(sep.shape[1]):
                def top(p):
                    order = np.lexsort((ids[rows], -p[rows, c]))
                    return frozenset(ids[rows][order[:3]].tolist())
                ranked = -np.sort(-sep[rows, c])
                gap = ranked[2] - ranked[3]
                moved = top(sep) != top(axis)
                certified += gap > 2 * delta; changed += moved; certified_changed += moved and gap > 2 * delta
                if moved:
                    ties += gap == 0; widest = max(widest, gap)
    assert changed == a["axis"]["changed_lists"] and certified_changed == 0
    put("EphCertified", f"{int(certified):,}"); put("EphTies", int(ties)); put("EphWidest", sci(widest))

    def pick(**kw):
        return select(pooled, layout="sorted", fan=4, **kw)[0]
    main, raw = pick(features="panel", topology="attached", recipe="joint"), pick(features="raw", topology="attached", recipe="joint")
    put("AnyFlips", main["union"]["flips"]); put("AnyLists", main["union"]["changed_lists"]); put("AnyRawFlips", raw["union"]["flips"])
    put("AnyRawLists", raw["union"]["changed_lists"])
    put("AnyWeights", main["union_w"]["flips"]); put("AnyFeatures", main["union_f"]["flips"])
    put("AnyRawWeights", raw["union_w"]["flips"]); put("AnyRawFeatures", raw["union_f"]["flips"])
    put("AnyCut", sum(r["union"]["flips"] + r["union"]["threshold"] + r["union"]["changed_lists"] for r in pooled if r["topology"] == "cut"))
    put("AnyCutMax", sci(max(r["union"]["max_abs"] for r in pooled if r["topology"] == "cut")))
    rows = any_["rows"]
    put("AnyConditions", len(rows))
    put("AnyAxisChanges", sum(r["axis"]["flips"] + r["axis"]["threshold"] + r["axis"]["changed_lists"] for r in rows))
    worst = max(r["axis"]["max_abs"] for r in rows)
    put("AnyAxisMax", "0" if worst == 0 else sci(worst))
    put("AnyAccSep", pct(main["correct"]["separate"], 960)); put("AnyAccUnion", pct(main["correct"]["union"], 960))
    put("AnyRawAccSep", pct(raw["correct"]["separate"], 960)); put("AnyRawAccUnion", pct(raw["correct"]["union"], 960))
    put("AnySupportFlips", pick(features="panel", topology="attached", recipe="support")["union"]["flips"])
    put("AnySupportAcc", pct(pick(features="panel", topology="attached", recipe="support")["correct"]["separate"], 960))
    v = any_["validation"]
    put("AnyChecks", v["calls"]); put("AnyCheckOver", v["over_tolerance"]); put("AnyCheckMax", sci(max(v["max_abs_separate"], v["max_abs_union"])))
    assert v["flips"] == 0
    for feats, name in (("panel", ""), ("raw", "Raw")):
        t = [r for r in any_["timing"] if r["features"] == feats]
        m = {p: med([r["median_ms"][p] for r in t]) for p in ("separate", "union", "axis")}
        put(f"Any{name}SepMs", f"{m['separate']:.0f}"); put(f"Any{name}UnionMs", f"{m['union']:.0f}"); put(f"Any{name}AxisMs", f"{m['axis']:.0f}")
        put(f"Any{name}Speed", f"{med([r['median_ms']['separate'] / r['median_ms']['axis'] for r in t]):.2f}")
        put(f"Any{name}Solve", f"{med([r['phases_ms']['axis']['solve'] for r in t]):.0f}")


def arrivals():
    path = RES / "arrivals.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    runs = d["runs"]
    # Capacity of a plan: requests per second at the most productive call size that was observed at
    # least five times.  A run whose arrival rate exceeds it is overloaded even if the 1,000 arrivals
    # ended before the backlog rule stopped it; such cells are marked and do not count as sustained.
    capacity = {plan: max(1000 * int(k) / v["median"] for k, v in svc.items() if v["calls"] >= 5)
                for plan, svc in d["service_ms"].items()}
    over = lambda r: not r["saturated"] and r["rate"] > capacity[r["plan"]]
    lines = []
    def cell(r):
        if r["saturated"]:
            return "sat."
        return f"{r['response_ms']['p50']:.0f}/{r['response_ms']['p99']:.0f}" + ("$^\\dagger$" if over(r) else "")
    limit = {}
    for rate in sorted({r["rate"] for r in runs}):
        row = {r["plan"]: r for r in select(runs, rate=rate)}
        u = row["union"]
        lines.append([rate, cell(row["separate"]), cell(u), cell(row["axis"]),
                      "--" if row["axis"]["saturated"] else f"{row['axis']['mean_batch']:.1f}",
                      "--" if u["saturated"] else pct(u["flips"], u["n"]),
                      "--" if u["saturated"] else pct(u["support_fit"]["flips"], u["support_fit"]["n"])])
        for plan, r in row.items():
            if not r["saturated"] and not over(r):
                limit[plan] = max(limit.get(plan, 0), rate)
    table("arrivals", "rrrrrrr", " & \\multicolumn{3}{c}{Response p50/p99 (ms)} & Axis & \\multicolumn{2}{c}{Union changed (\\%)} \\\\ "
          "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7} Req./s & Sep. & Union & Axis & batch & joint & support", lines)
    for plan, name in (("separate", "Sep"), ("union", "Union"), ("axis", "Axis")):
        put("Arr" + name + "Limit", limit[plan])
        put("Arr" + name + "Capacity", f"{capacity[plan]:.0f}")
    put("ArrAxisSepRatio", f"{capacity['axis'] / capacity['separate']:.1f}")
    put("ArrAxisShare", f"{100 * capacity['axis'] / capacity['union']:.0f}")
    put("ArrSepSat", min(r["rate"] for r in runs if r["plan"] == "separate" and r["saturated"]))
    marked = [r for r in runs if over(r)]
    assert [(r["plan"], r["rate"]) for r in marked] == [("axis", 320)], marked
    put("ArrAxisOverRate", marked[0]["rate"])
    svc = d["service_ms"]
    cap = str(d["cap"])
    for plan, name in (("union", "Union"), ("axis", "Axis")):
        put("Arr" + name + "FullMs", f"{svc[plan][cap]['median']:.0f}")
        put("Arr" + name + "LaneMs", f"{(svc[plan][cap]['median'] - svc[plan]['1']['median']) / (d['cap'] - 1):.2f}")
    put("ArrOneMs", f"{svc['separate']['1']['median']:.0f}")
    ok = [r for r in runs if r["plan"] == "axis"]
    put("ArrAxisFlips", sum(r["flips"] + r["threshold"] for r in ok)); put("ArrAxisLists", sum(r["changed_lists"] for r in ok))
    put("ArrAxisListTotal", f"{sum(r['lists'] for r in ok):,}"); put("ArrAxisMax", sci(max(r["max_abs"] for r in ok)))
    put("ArrAxisOver", sum(r["calls_over_tolerance"] for r in ok)); put("ArrAxisCalls", f"{sum(r['calls'] for r in ok):,}")
    put("ArrAxisExposures", f"{sum(r['n'] for r in ok):,}")
    put("ArrSepChanges", sum(r["flips"] + r["threshold"] + r["changed_lists"] for r in runs if r["plan"] == "separate"))
    un = [r for r in runs if r["plan"] == "union" and not r["saturated"]]
    low, high = min(un, key=lambda r: r["rate"]), max(un, key=lambda r: r["rate"])
    put("ArrLowRate", low["rate"]); put("ArrHighRate", high["rate"])
    put("ArrUnionLow", pct(low["flips"], low["n"])); put("ArrUnionHigh", pct(high["flips"], high["n"]))
    put("ArrUnionLowSupport", pct(low["support_fit"]["flips"], low["support_fit"]["n"]))
    put("ArrUnionHighSupport", pct(high["support_fit"]["flips"], high["support_fit"]["n"]))
    put("ArrVarying", high["varying"]); put("ArrVaryingOf", high["requests_seen"]); put("ArrHighBatch", f"{high['mean_batch']:.1f}")
    put("ArrVaryingSupport", high["support_fit"]["varying"]); put("ArrLowBatch", f"{low['mean_batch']:.2f}")
    put("ArrArrivals", f"{runs[0]['arrivals']:,}"); put("ArrCap", d["cap"]); put("ArrPool", d["pool"])
    sep = {r["rate"]: r for r in runs if r["plan"] == "separate"}
    axis = {r["rate"]: r for r in runs if r["plan"] == "axis"}
    union = {r["rate"]: r for r in runs if r["plan"] == "union"}
    top = limit["separate"]
    put("ArrLowMedian", f"{sep[low['rate']]['response_ms']['p50']:.0f}")
    put("ArrSepTop", top); put("ArrSepTopP", f"{sep[top]['response_ms']['p99']:.0f}"); put("ArrAxisAtSepTopP", f"{axis[top]['response_ms']['p99']:.0f}")
    put("ArrUnionAtSepTopP", f"{union[top]['response_ms']['p99']:.0f}")
    put("ArrAxisTopP", f"{axis[limit['axis']]['response_ms']['p99']:.0f}")
    put("ArrAxisTopBatch", f"{axis[limit['axis']]['mean_batch']:.1f}")
    put("ArrUnionTopP", f"{union[limit['union']]['response_ms']['p99']:.0f}")


def carried(audit):
    """Redraw the two archived figures whose labels collided."""
    gpu = select(audit["rows"], model="nodepfn", device="cuda")
    eph = select(audit["rows"], model="ephris", device="cpu", plan="union")
    cmap = LinearSegmentedColormap.from_list("counts", ["#f7f4eb", "#efc18e", "#ca783f", "#874023"])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55))
    fig.subplots_adjust(left=.105, right=.985, bottom=.17, top=.76, wspace=.37)
    panels = []
    for source in [select(gpu, plan="union", fan=4, layout="sorted"), select(eph, fan=4, layout="sorted")]:
        panels.append(np.asarray([[total(select(source, task=t, recipe=rec, topology=top), "flips")
                                   for rec in ("joint", "support") for top in ("attached", "cut")] for t in TASKS]))
    vmax = max(x.max() for x in panels)
    for ax, data, title in zip(axes, panels, ["(a)  NodePFN, one member, GPU", "(b)  Ephris, CPU"]):
        ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set(yticks=range(3), yticklabels=[NAMES[t] for t in TASKS], xticks=range(4), xticklabels=["Attached", "Cut", "Attached", "Cut"])
        ax.tick_params(length=0, pad=3, labelsize=7)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9, pad=22)
        for x, l in ((.5, "Joint fit"), (2.5, "Support fit")):
            ax.text(x, -.62, l, ha="center", va="bottom", fontsize=7.5, color=INK)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, str(data[i, j]), ha="center", va="center", color="white" if data[i, j] > .63 * vmax else "#302821", weight="bold")
        ax.axvline(1.5, color="white", lw=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.savefig(FIG / "edges.pdf"); plt.close(fig)
    put("JointCut", int(panels[0][:, 1].sum()))

    w = audit["witness"]
    labels = ["Case-based", "Genetic algorithms", "Neural networks", "Probabilistic methods", "Reinforcement learning", "Rule learning", "Theory"]
    fig, ax = plt.subplots(figsize=(3.37, 2.35))
    fig.subplots_adjust(left=.37, right=.95, bottom=.20, top=.86)
    ys = np.arange(7)[::-1]; x, y = np.array(w["separate"]), np.array(w["union"])
    ax.hlines(ys, np.minimum(x, y), np.maximum(x, y), color="#b9c1c4", lw=1.2)
    ax.scatter(x, ys, s=30, facecolors="white", edgecolors=BLUE, lw=1.2, label="Separate")
    ax.scatter(w["axis"], ys, s=8, c=TEAL, marker="D", label="Request axis", zorder=3)
    ax.scatter(y, ys, s=21, c=ORANGE, marker="^", label="Node union", zorder=3)
    ax.set(yticks=ys, yticklabels=labels, xlabel="Predicted probability", xlim=(-.025, 1.025), ylim=(-.55, 6.55))
    ax.tick_params(axis="y", length=0, labelsize=6.9)
    ax.legend(loc="lower center", bbox_to_anchor=(.42, 1.0), ncol=3, fontsize=6.5, frameon=False, handletextpad=.2, columnspacing=1.0)
    ax.spines["left"].set_visible(False); fig.savefig(FIG / "witness.pdf"); plt.close(fig)


def preliminary():
    """Redraw the preliminary dose figure with the terminology of the revised text."""
    import csv
    with (REV.parents[1] / "companion_dose/curves.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(3.33, 1.85), layout="constrained")
    series = [("cora", "Cora", "#a44922", "o"), ("persrefex", "PersRefEx", "#2d5277", "s"),
              ("pubmed", "PubMed", "#00857a", "^"), ("chameleon", "Chameleon", "#575966", "D")]
    for task, label, color, marker in series:
        chosen = [r for r in rows if r["task"] == task and r["method"] == "ephris" and r["recipe"] == "support"]
        ax.plot(range(len(chosen)), [float(r["excess_TV"]) for r in chosen], marker=marker, color=color,
                linewidth=1.35, markersize=4, label=label)
    ax.set_xticks(range(5), ["1", "4", "8", "16", "32"])
    ax.set_xlabel("Distinct co-batched records per call")
    ax.set_ylabel("Mean excess total variation")
    ax.axhline(0, color="#999999", linewidth=.65)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.savefig(FIG / "dose.pdf"); plt.close(fig)


if __name__ == "__main__":
    audit = json.loads((OLD / "results/audit.json").read_text())
    MACROS_OLD = json.loads((OLD / "results/headlines.json").read_text())
    default = answers(audit)
    accuracy(default)
    adjacency()
    vectors()
    networks()
    cost()
    scale()
    others()
    arrivals()
    carried(audit)
    preliminary()
    (FIG / "numbers-rev.tex").write_text("".join("\\newcommand{\\" + k + "}{" + v + "}\n" for k, v in sorted(MACROS.items())))
    (RES / "headlines-rev.json").write_text(json.dumps(MACROS, indent=1, sort_keys=True) + "\n")
    # Plain-text abstract for the submission form, with every number substituted.
    import re
    text = (REV / "source/abstract.tex").read_text().strip()
    values = {**MACROS_OLD, **MACROS}
    text = re.sub(r"\\([A-Za-z]+)\{\}", lambda m: values[m.group(1)], text)
    text = re.sub(r"\\([A-Za-z]+)(?=\\%)", lambda m: values[m.group(1)], text).replace("\\%", "%")
    assert "\\" not in text, text
    (REV / "ABSTRACT.txt").write_text(text + "\n")
    print(json.dumps(MACROS, indent=0, sort_keys=True))
