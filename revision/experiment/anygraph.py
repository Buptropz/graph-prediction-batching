"""GraphAny as a third pretrained model: separate calls, node union and a lane-local plan.

GraphAny (vendor/GraphAny, checkpoint trained on Wisconsin) predicts in three steps.  It propagates
the input features over the graph of the call (mean over neighbours; channels X, L1, L2, H1, H2),
solves one least-squares problem per channel on the labelled rows, and fuses the channel predictions
of every row with an attention MLP whose input is computed from that row alone.

The fusion (``graphany.model.GraphAny``) and the distance features are the released code, imported
unchanged with stubs for the training-only dependencies that are not installed.  Propagation and
the solve are re-implemented in plain torch so that several requests can occupy lanes of one call;
``--validate`` compares that path with the released preprocessing (DGL) on single-graph calls.

The file is not called graphany.py because that name would shadow the vendored package.
"""
import argparse
import contextlib
import logging
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch

import lib

VENDOR = lib.ROOT / "vendor/GraphAny"
CHECKPOINT = VENDOR / "checkpoints/graph_any_wisconsin.pt"
CHANNELS = ["X", "L1", "L2", "H1", "H2"]
# The checkpoint stores unresolved config strings.  Values below are the README command for the
# Wisconsin checkpoint (n_hidden=32, n_mlp_layer=2, entropy=1) and configs/model.yaml (channels,
# attn_temp=5); the weight shapes (32 x 20, 3 x 32) agree with them.
CONFIG = dict(n_hidden=32, feat_channels=CHANNELS, pred_channels=["X", "L1", "L2"], att_temperature=5,
              entropy=1, n_mlp_layer=2)
HOPS = 2
PLANS = ("separate", "union", "axis")
HYBRIDS = ("union_w", "union_f")   # union weights with separate features; union features with separate weights
GRID = [(layout, topology, recipe, fan) for layout in ("sorted", "random") for topology in ("attached", "cut")
        for recipe in ("joint", "support") for fan in (4, 2)]
PRIMARY = ("sorted", "attached", "joint", 4)


def released():
    """Import the vendored package unchanged and load the checkpoint into its model class."""
    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules.setdefault(name, module)

    @contextlib.contextmanager
    def timer(*args, **kwargs):
        yield

    stub("pytorch_lightning", LightningDataModule=type("LightningDataModule", (), {}))
    stub("hydra")
    stub("hydra.utils", instantiate=None)
    stub("graphany.utils", logger=logging.getLogger("graphany"), timer=timer)
    os.environ.setdefault("DGLBACKEND", "pytorch")
    sys.path.insert(0, str(VENDOR))
    import graphany.data as data
    import graphany.model as model
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net = model.GraphAny(**CONFIG)
    state = {k.removeprefix("gnn_model."): v for k, v in ckpt["state_dict"].items()}
    assert all(k.startswith("gnn_model.") for k in ckpt["state_dict"])
    result = net.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    net.eval()
    return net, data, dict(keys=len(state), parameters=int(sum(p.numel() for p in net.parameters())))


def simple(edge):
    """Released graph preprocessing: no self loop, both directions, no duplicate (sorted by source)."""
    a, b = np.asarray(edge, dtype=np.int64)
    keep = a != b
    pairs = np.unique(np.r_[np.c_[a[keep], b[keep]], np.c_[b[keep], a[keep]]], axis=0).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def propagate(x, src, dst):
    """X, mean-over-neighbours low-pass (L) and high-pass (H) features; a node without neighbour reads zero."""
    src, dst = torch.as_tensor(src), torch.as_tensor(dst)
    degree = torch.zeros(len(x)).index_add_(0, dst, torch.ones(len(dst))).clamp_(min=1).unsqueeze(1)
    mean = lambda h: torch.zeros_like(h).index_add_(0, dst, h[src]) / degree
    out, low, high = {"X": x}, x, x
    for hop in range(HOPS):
        low, high = mean(low), high - mean(high)
        out[f"L{hop + 1}"], out[f"H{hop + 1}"] = low, high
    return out


class Adapter:
    """One support, any number of lanes per call."""
    def __init__(self, net, work, classes, recipe):
        self.net, self.work, self.recipe, self.classes = net, work, recipe, classes
        self.ns = work.ns
        self.tx = torch.from_numpy(np.asarray(work.tx, dtype=np.float32))
        self.target = torch.nn.functional.one_hot(torch.from_numpy(work.y), classes).float()
        self.clock = {}
        if recipe == "support":   # weights from the support alone, shared by every plan
            a, b = work.edge
            inner = (a < self.ns) & (b < self.ns)
            feats = propagate(self.tx, *simple(np.stack([a[inner], b[inner]])))
            self.fixed = {c: torch.linalg.lstsq(feats[c], self.target, driver="gelss").solution for c in CHANNELS}

    def lap(self, name, start):
        self.clock[name] = self.clock.get(name, 0.) + (time.perf_counter() - start) * 1e3

    def parts(self, lanes):
        """Query features and solved weights per lane.  ``lanes`` is a list of row-position arrays.

        All lanes share one propagation over a block-diagonal graph in which every lane owns a copy of
        the support, and one batched solve with one right-hand side block per lane.
        """
        tic = time.perf_counter()
        xs, src, dst, spans, offset = [], [], [], [], 0
        for rows in lanes:
            q, e = self.work.part(rows)
            s, d = simple(e)
            xs += [self.tx, torch.from_numpy(np.asarray(q, dtype=np.float32))]
            src.append(s + offset)
            dst.append(d + offset)
            spans.append((offset, len(rows)))
            offset += self.ns + len(rows)
        feats = propagate(torch.cat(xs), np.concatenate(src), np.concatenate(dst))
        self.lap("propagate", tic)
        tic = time.perf_counter()
        query = [{c: feats[c][o + self.ns:o + self.ns + m] for c in CHANNELS} for o, m in spans]
        if self.recipe == "support":
            weight = [self.fixed for _ in spans]
        else:
            solved = {}
            for c in CHANNELS:
                a = torch.stack([feats[c][o:o + self.ns] for o, _ in spans])
                b = self.target.expand(len(spans), -1, -1)
                solved[c] = torch.linalg.lstsq(a, b, driver="gelss").solution
            weight = [{c: solved[c][i] for c in CHANNELS} for i in range(len(spans))]
        self.lap("solve", tic)
        return query, weight

    def fuse(self, query, weight):
        """Channel predictions and released attention fusion for the rows of one call."""
        tic = time.perf_counter()
        sizes = [len(q["X"]) for q in query]
        assert min(sizes) > 1   # the released MLP skips batch normalisation for a batch of one row
        logits = {}
        for c in CHANNELS:
            f = torch.zeros(len(query), max(sizes), query[0][c].shape[1])
            for i, q in enumerate(query):
                f[i, :sizes[i]] = q[c]
            out = torch.bmm(f, torch.stack([w[c] for w in weight]))
            logits[c] = torch.cat([out[i, :m] for i, m in enumerate(sizes)])
        self.lap("predict", tic)
        tic = time.perf_counter()
        with torch.no_grad():
            fused, _ = self.net(logits)
            prob = torch.softmax(fused, 1).double().numpy()
        self.lap("fuse", tic)
        return prob

    def groups(self, fan):
        reqs = self.work.requests()
        return [reqs[i:i + fan] for i in range(0, len(reqs), fan)]

    def run(self, plan, fan):
        out = np.empty((len(self.work.ids), self.classes))
        if plan == "separate":
            for rows in self.work.requests():
                out[rows] = self.fuse(*self.parts([rows]))
        elif plan == "union":
            for group in self.groups(fan):
                rows = np.concatenate(group)
                out[rows] = self.fuse(*self.parts([rows]))
        elif plan == "axis":
            for group in self.groups(fan):
                out[np.concatenate(group)] = self.fuse(*self.parts(group))
        else:   # hybrids of the separate and the union computation
            for group in self.groups(fan):
                rows = np.concatenate(group)
                uq, uw = self.parts([rows])
                start = 0
                for r in group:
                    sq, sw = self.parts([r])
                    cut = {c: uq[0][c][start:start + len(r)] for c in CHANNELS}
                    out[r] = self.fuse(sq, uw) if plan == "union_w" else self.fuse([cut], sw)
                    start += len(r)
        assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-5)
        return out


def reference(net, data, tx, y, q, e, classes, tmp, tag):
    """One single-graph call through the released preprocessing (DGL propagation, solve, distances)."""
    import dgl
    ns, n = len(y), len(y) + len(q)
    g = dgl.graph((torch.as_tensor(e[0]), torch.as_tensor(e[1])), num_nodes=n, idtype=torch.int32)
    g = dgl.to_simple(dgl.to_bidirected(dgl.remove_self_loop(g)))     # add_self_loop=false, to_bidirected=true
    ds = object.__new__(data.GraphDataset)
    ds.cfg = types.SimpleNamespace(feat_channels=CHANNELS, pred_channels=CONFIG["pred_channels"],
                                   n_per_label_examples=5, entropy=CONFIG["entropy"], feat_chn="+".join(CHANNELS))
    ds.name, ds.num_class, ds.preprocess_device = tag, classes, torch.device("cpu")
    ds.label = torch.cat([torch.as_tensor(y), torch.zeros(len(q), dtype=torch.long)])
    ds.train_indices = torch.arange(ns)
    ds.cache_f_name, ds.dist_f_name = str(Path(tmp) / f"{tag}.pt"), str(Path(tmp) / f"{tag}-dist.pt")
    x = torch.from_numpy(np.r_[tx, q].astype(np.float32))
    _, pred, dist = ds.prepare_prop_features_logits_and_dist_features(g, x, n_hops=HOPS)
    nodes = torch.arange(ns, n)
    with torch.no_grad():
        fused, _ = net({c: p[nodes] for c, p in pred.items()}, dist=dist[nodes])
    for name in (ds.cache_f_name, ds.dist_f_name):
        os.remove(name)
    return torch.softmax(fused, 1).double().numpy()


def workload(task, fixture, features, layout, topology):
    work = lib.Workload.from_panel(fixture, layout, topology == "attached")
    if features == "raw":   # the corpus features GraphAny is released for, instead of the 15 panel features
        c = lib.corpus(task)
        work.tx = c["x"][[c["pos"][str(i)] for i in fixture["support_ids"]]]
        work.q = c["x"][[c["pos"][str(i)] for i in work.ids]]
    return work


def moves(y, a, b):
    moved = a != b
    ra, rb = a[moved] == y[moved], b[moved] == y[moved]
    return dict(right_to_wrong=int((ra & ~rb).sum()), wrong_to_right=int((~ra & rb).sum()),
                wrong_to_wrong=int((~ra & ~rb).sum()))


def pooled(rows, keys):
    out = []
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        part = [r for r in rows if tuple(r[k] for k in keys) == key]
        item = dict(zip(keys, key), conditions=len(part), n=sum(r["n"] for r in part))
        for plan in ("union", "axis") + HYBRIDS:
            have = [r[plan] for r in part if plan in r]
            if have:
                item[plan] = dict(flips=sum(h["flips"] for h in have), threshold=sum(h["threshold"] for h in have),
                                  lists=sum(h["lists"] for h in have),
                                  changed_lists=sum(h["changed_lists"] for h in have),
                                  max_abs=max(h["max_abs"] for h in have),
                                  over_tolerance=sum(h["max_abs"] > lib.TOL for h in have), n=sum(h["n"] for h in have))
        item["correct"] = {p: sum(r["correct"][p] for r in part) for p in PLANS}
        item["untouched"] = dict(n=sum(r["untouched"]["n"] for r in part),
                                 flips=sum(r["untouched"]["flips"] for r in part))
        item["moves"] = {k: sum(r["moves"][k] for r in part) for k in part[0]["moves"]}
        item["panels_affected"] = sum(r["union"]["flips"] > 0 for r in part)
        out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/graphany.json")
    ap.add_argument("--vectors", type=Path, default=lib.REV / "results/graphany.npz")
    ap.add_argument("--tasks", nargs="+", default=list(lib.TASKS))
    ap.add_argument("--panels", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    lib.setup_torch()
    net, data, loaded = released()
    rows, timing, checks, vectors = [], [], [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in args.tasks:
            classes = lib.CLASSES[task]
            for index in args.panels:
                fixture = lib.panel(task, index)
                for features in ("panel", "raw"):
                    for layout, topology, recipe, fan in GRID:
                        work = workload(task, fixture, features, layout, topology)
                        adapter = Adapter(net, work, classes, recipe)
                        primary = (layout, topology, recipe, fan) == PRIMARY
                        plans = PLANS + (HYBRIDS if recipe == "joint" and fan == 4 else ())
                        p = {plan: adapter.run(plan, fan) for plan in plans}
                        y = lib.truth(task, work.ids)
                        pred = {plan: v.argmax(1) for plan, v in p.items()}
                        a, b = work.edge
                        touched = np.zeros(len(work.ids), bool)
                        cross = (a < work.ns) != (b < work.ns)
                        touched[np.maximum(a, b)[cross] - work.ns] = True
                        row = dict(task=task, panel=index, features=features, layout=layout, topology=topology,
                                   recipe=recipe, fan=fan, n=len(work.ids), cross_edges=work.cross,
                                   touching=int(touched.sum()),
                                   support_edges=int(((a < work.ns) & (b < work.ns) & (a < b)).sum()),
                                   correct={plan: int((pred[plan] == y).sum()) for plan in PLANS},
                                   moves=moves(y, pred["separate"], pred["union"]),
                                   untouched=dict(n=int((~touched).sum()),
                                                  flips=int((pred["separate"] != pred["union"])[~touched].sum())))
                        for plan in plans[1:]:
                            row[plan] = lib.endpoints(work.ids, work.rid, p["separate"], p[plan])
                        rows.append(row)
                        if primary:
                            key = f"{task}_p{index}_{features}__"
                            vectors[key + "ids"] = work.ids
                            vectors.update({key + plan: p[plan] for plan in PLANS})
                            if not args.no_validate:   # released preprocessing on the same single-graph calls
                                ref_sep = np.empty_like(p["separate"])
                                for j, r in enumerate(work.requests()):
                                    q, e = work.part(r)
                                    ref_sep[r] = reference(net, data, work.tx, work.y, q, e, classes, tmp, f"s{j}")
                                everything = np.concatenate(work.requests())
                                q, e = work.part(everything)
                                ref_union = np.empty_like(ref_sep)
                                ref_union[everything] = reference(net, data, work.tx, work.y, q, e, classes, tmp, "u")
                                checks.append(dict(task=task, panel=index, features=features,
                                                   separate=float(np.abs(ref_sep - p["separate"]).max()),
                                                   union=float(np.abs(ref_union - p["union"]).max()),
                                                   separate_flips=int((ref_sep.argmax(1) != pred["separate"]).sum()),
                                                   union_flips=int((ref_union.argmax(1) != pred["union"]).sum())))
                            times, phases, drift = {plan: [] for plan in PLANS}, {}, {plan: 0. for plan in PLANS}
                            for rep in range(args.repeats + 1):
                                shift = rep % 3
                                for plan in PLANS[shift:] + PLANS[:shift]:
                                    adapter.clock = {}
                                    tic = time.perf_counter()
                                    again = adapter.run(plan, fan)
                                    drift[plan] = max(drift[plan], float(np.abs(again - p[plan]).max()))
                                    if rep:
                                        times[plan].append((time.perf_counter() - tic) * 1e3)
                                        for name, ms in adapter.clock.items():
                                            phases.setdefault(plan, {}).setdefault(name, []).append(ms)
                            timing.append(dict(task=task, panel=index, features=features, repeat_max_abs=drift,
                                               median_ms={plan: float(np.median(v)) for plan, v in times.items()},
                                               phases_ms={plan: {k: float(np.median(v)) for k, v in ph.items()}
                                                          for plan, ph in phases.items()}))
                    done = [r for r in rows if (r["task"], r["panel"], r["features"]) == (task, index, features)
                            and (r["layout"], r["topology"], r["recipe"], r["fan"]) == PRIMARY][0]
                    print(task, index, features, "union", done["union"]["flips"], done["union"]["threshold"],
                          done["union"]["changed_lists"], "axis", done["axis"]["flips"], done["axis"]["max_abs"],
                          "acc", done["correct"], timing[-1]["median_ms"] if timing else "", flush=True)
    keys = ("features", "layout", "topology", "recipe", "fan")
    summary = dict(pooled=pooled(rows, keys), by_task=pooled(rows, keys + ("task",)))
    validation = dict(state_dict_keys=loaded["keys"], parameters=loaded["parameters"], calls=len(checks) * 5,
                      max_abs_separate=max((c["separate"] for c in checks), default=None),
                      max_abs_union=max((c["union"] for c in checks), default=None),
                      flips=sum(c["separate_flips"] + c["union_flips"] for c in checks),
                      over_tolerance=sum((c["separate"] > lib.TOL) + (c["union"] > lib.TOL) for c in checks),
                      checks=checks)
    lib.dump(args.out, dict(config=dict(CONFIG, hops=HOPS, checkpoint=str(CHECKPOINT.relative_to(lib.ROOT)),
                                        dtype="float32", probability="softmax of the fused channel predictions",
                                        threads=torch.get_num_threads(), torch=str(torch.__version__)),
                            validation=validation, rows=rows, summary=summary, timing=timing,
                            chance={t: 1 / c for t, c in lib.CLASSES.items()}))
    np.savez_compressed(args.vectors, **vectors)
    for item in summary["pooled"]:
        if item["fan"] == 4:
            print({k: item[k] for k in keys}, "union", item["union"]["flips"], "/", item["n"], item["union"]["threshold"],
                  item["union"]["changed_lists"], "/", item["union"]["lists"], "axis", item["axis"]["flips"],
                  item["axis"]["changed_lists"], item["axis"]["max_abs"], item["correct"], flush=True)
    print("validation", {k: v for k, v in validation.items() if k != "checks"})


if __name__ == "__main__":
    main()
