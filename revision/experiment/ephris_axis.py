"""Request-axis execution for the Ephris checkpoint.

The released Ephris model has no batch dimension: one call holds one graph, and three of its parts
read every row of the call.  (1) The feature schema and the standardisation statistics are fitted
on the rows of the call.  (2) Every feature refiner runs an induced set attention over all nodes.
(3) Every graph block appends learned global nodes that attend to, and are read by, all nodes, and
scales queries by a degree that counts them.  This script evaluates k requests that share a
support in one pass through the network while keeping all three lane-local:

* preprocessing is fitted once per lane (on the lane's rows for the joint recipe, on the support
  for the support recipe); the row-local tokenizer runs once per distinct feature schema;
* the set attention of the feature refiners receives (feature, lane) pairs on its batch dimension,
  so the released module is applied unchanged to sets of one lane;
* the graph blocks are evaluated with one copy of the global nodes per lane.  The released graph
  attention is an edge-list reduction that hard-wires the global nodes to all rows, so this one
  reduction is re-expressed as a dense masked attention with a lane dimension.  Projections,
  query scaling, normalisation and feed-forward layers are the released modules and weights.

Lanes must have equal length (set attention over a lane would read padded rows).  The same engine
with one lane evaluates the separate and the node-union plan, which validates the dense reduction
against the released interface and the archived outputs.
"""
import argparse
import math
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import lib
from inference import Ephris, focal_prob

GRID = [(layout, topology, recipe) for layout in ("sorted", "random") for topology in ("attached", "cut")
        for recipe in ("joint", "support")]
PLANS = {"separate": 1, "union": 4, "axis": 4, "union-2": 2, "axis-2": 2}   # requests per model call
LANES = SimpleNamespace(k=0, n=0, mask=None)


def lane_refine(self, feature_state):
    """FeatureRefiner.forward with the set attention confined to one lane: [k*n, F, d] -> same."""
    k, n = LANES.k, LANES.n
    rows, feats, width = feature_state.shape
    sets = feature_state.view(k, n, feats, width).permute(2, 0, 1, 3).reshape(feats * k, n, width)
    step = 64 * k                                   # the released module refines 64 features at a time
    out = torch.cat([self.pool(sets[i:i + step]) for i in range(0, feats * k, step)])
    return out.view(feats, k, n, width).permute(1, 2, 0, 3).reshape(rows, feats, width)


def dense_attention(att, state, mask):
    """GraphAttention as a masked dense attention per lane.  mask[lane, destination, source]."""
    k, m, width = state.shape
    heads, hw = att.heads, att.head_width
    from ephris.nn.layers.fused_linear import fused_linear
    projected = fused_linear(state.reshape(k * m, width), (att.q_proj, att.k_proj, att.v_proj))
    query, key, value = (t.view(k * m, heads, hw) for t in projected.split(heads * hw, dim=-1))
    query = att._scale_query_by_lengths(query, mask.sum(-1).reshape(-1).clamp_min(1))
    query, key, value = (t.view(k, m, heads, hw).transpose(1, 2) for t in (query, key, value))
    scores = torch.matmul(query, key.transpose(-1, -2)).float() / math.sqrt(hw)
    scores = scores.masked_fill(~mask.unsqueeze(1), -torch.inf)
    out = torch.matmul(torch.softmax(scores, dim=-1), value.float())
    return out.transpose(1, 2).reshape(k, m, heads * hw).to(state.dtype)


def lane_block(self, node_state, edge_index):
    """GraphBlock.forward with one copy of the learned global nodes per lane: [k*n, w] -> same."""
    k, n, mask = LANES.k, LANES.n, LANES.mask
    globals_ = self.global_nodes.to(device=node_state.device, dtype=node_state.dtype)
    state = torch.cat((node_state.view(k, n, self.width), globals_.unsqueeze(0).expand(k, -1, -1)), dim=1)
    for layer in self.layers:
        state = layer.attention_norm(state + dense_attention(layer.attention, state, mask))
        state = layer.feedforward_norm(state + layer.feedforward(state))
    return state[:, :n].reshape(k * n, self.width)


@contextmanager
def lanes(k, n, mask):
    from ephris.nn.graph import GraphBlock
    from ephris.nn.refinement import FeatureRefiner
    LANES.k, LANES.n, LANES.mask = k, n, mask
    with patch.object(FeatureRefiner, "forward", lane_refine), patch.object(GraphBlock, "forward", lane_block):
        yield
    LANES.k, LANES.n, LANES.mask = 0, 0, None


class Engine:
    """Lane-batched Ephris for one support.  ``predict`` has the interface of ``lib.Engine``."""
    def __init__(self, ref, y):
        from ephris.checkpoints import build_model_from_checkpoint
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.classes = int(self.y.max()) + 1
        assert torch.unique(self.y).tolist() == list(range(self.classes))
        self.device = torch.device(ref.device)
        self.model = build_model_from_checkpoint(ref.payload, SimpleNamespace(num_outputs=self.classes), self.device)
        self.passes, self.schemas = 0, 0
        self.model.icl.layers[0].attention_norm.register_forward_hook(self._count)

    def _count(self, *_):
        self.passes += 1

    @torch.inference_mode()
    def predict(self, support, requests, edges, recipe):
        from ephris.data.permutation import make_inference_permutation
        from ephris.data.preprocessing import FeaturePreprocessor
        from ephris.execution.runner import _align_logits, _permute_labels
        from ephris.nn.cache import undirected_edges
        model, ns, k = self.model, len(self.y), len(requests)
        n = ns + len(requests[0])
        assert all(len(q) == n - ns for q in requests), "lanes must have equal length"
        g = model.num_globals
        features, types = [], []
        mask = torch.zeros(k, n + g, n + g, dtype=torch.bool)
        for lane, (q, e) in enumerate(zip(requests, edges)):
            x = torch.as_tensor(np.vstack((support, q)), dtype=torch.float32)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            fit = FeaturePreprocessor()               # (i) statistics from this lane only
            fit.prepare(x if recipe == "joint" else x[:ns])
            features.append(fit.transform(x))
            types.append(fit.feature_metadata(feature_dim=x.size(1)))
            ei = undirected_edges(torch.as_tensor(np.asarray(e), dtype=torch.long).reshape(2, -1), num_nodes=n)
            mask[lane, ei[1], ei[0]] = True           # (ii) edges of this lane's copy of S and Q_i
        mask[:, :n, n:] = True                         # (iii) each lane reads its own global nodes
        mask[:, n:, :n] = True
        mask[:, n:, n:] = torch.eye(g, dtype=torch.bool)
        permutation = make_inference_permutation(x=features[0], num_outputs=self.classes)
        observed = torch.zeros(n, dtype=torch.bool)
        observed[:ns] = True
        labels = torch.full((n,), -1, dtype=torch.long)
        labels[:ns] = self.y
        labels = _permute_labels(labels, permutation.class_permutation, num_outputs=self.classes)
        groups = {}
        for lane, t in enumerate(types):
            groups.setdefault(tuple(t.tolist()), []).append(lane)
        self.schemas = len(groups)
        state = None
        for members in groups.values():                # the tokenizer is row-local given a schema
            tokens = model.tokenize_features(
                x=torch.cat([features[i] for i in members]), y=labels.repeat(len(members)),
                context_mask=observed.repeat(len(members)), x_is_preprocessed=True,
                feature_type_ids=types[members[0]], chunk_budget=None, feature_order=permutation.feature_order)
            if state is None:
                state = tokens.new_empty((k * n,) + tuple(tokens.shape[1:]))
            for j, lane in enumerate(members):
                state[lane * n:(lane + 1) * n] = tokens[j * n:(j + 1) * n]
        unused = torch.empty((2, 0), dtype=torch.long)
        with lanes(k, n, mask):
            node = model.flatten_node_state(model.encode_features(state, unused))
            node = model.inject_labels(node, labels.repeat(k), observed.repeat(k))
            logits = model.project_logits(model.propagate(node, unused)).float()
        logits = _align_logits(logits[:, :self.classes], permutation.class_permutation)
        logp = F.softmax(logits, dim=-1).clamp_min(1e-12).log().view(k, n, self.classes)[:, ns:]
        return np.vstack([focal_prob(v.numpy(), n - ns, False) for v in logp])


def groups_of(work, plan):
    """Model calls of a plan: each call is a list of lanes, each lane a list of row positions."""
    reqs, width = work.requests(), PLANS[plan]
    calls = [reqs[i:i + width] for i in range(0, len(reqs), width)]
    return [[np.concatenate(c)] for c in calls] if plan.startswith("union") else calls


def run(engine, work, plan, recipe):
    out = np.empty((len(work.ids), engine.classes))
    before = engine.passes
    for call in groups_of(work, plan):
        q, e = zip(*[work.part(g) for g in call])
        out[np.concatenate(call)] = engine.predict(work.tx, q, e, recipe)
    return out, engine.passes - before


def released(ref, work, plan, recipe):
    out = np.empty((len(work.ids), int(work.y.max()) + 1))
    calls = groups_of(work, plan)
    for (g,) in calls:
        q, e = work.part(g)
        out[g] = focal_prob(ref(work.tx, work.y, q, e, recipe), len(q), False)
    return out, len(calls)


def pool(rows, key, fields=("n", "flips", "threshold", "lists", "changed_lists")):
    total = {f: int(sum(r[key][f] for r in rows)) for f in fields}
    total["max_abs"] = float(max(r[key]["max_abs"] for r in rows))
    total["over_tolerance"] = int(sum(r[key]["max_abs"] > lib.TOL for r in rows))
    total["conditions"] = len(rows)
    return total


def summarise(rows):
    keys = ("union", "axis", "axis_vs_released", "union-2", "axis-2", "own_separate_vs_archive",
            "own_union_vs_archive", "released_separate_vs_archive", "released_union_vs_archive",
            "archive_union")
    scopes = {"all": rows}
    for layout, topology, recipe in GRID:
        cell = [r for r in rows if (r["layout"], r["topology"], r["recipe"]) == (layout, topology, recipe)]
        scopes[f"{layout}_{topology}_{recipe}"] = cell
        for task in lib.TASKS:
            scopes[f"{task}_{layout}_{topology}_{recipe}"] = [r for r in cell if r["task"] == task]
    for task in lib.TASKS:
        scopes[task] = [r for r in rows if r["task"] == task]
    out = {}
    for name, cell in scopes.items():
        if not cell:
            continue
        s = {k: pool(cell, k) for k in keys if k in cell[0]}
        s["accuracy"] = {p: sum(r["correct"][p] for r in cell) / sum(r["n"] for r in cell) for p in cell[0]["correct"]}
        s["latency_ms"] = {p: float(np.median([np.median(r["latency_ms"][p]) for r in cell]))
                           for p in cell[0]["latency_ms"]}
        s["released_latency_ms"] = {p: float(np.median([r["released_latency_ms"][p] for r in cell]))
                                    for p in cell[0]["released_latency_ms"]}
        s["speedup_axis_over_separate"] = float(np.median(
            [np.median(r["latency_ms"]["separate"]) / np.median(r["latency_ms"]["axis"]) for r in cell]))
        s["speedup_union_over_separate"] = float(np.median(
            [np.median(r["latency_ms"]["separate"]) / np.median(r["latency_ms"]["union"]) for r in cell]))
        s["passes"] = {p: sorted({r["passes"][p] for r in cell}) for p in cell[0]["passes"]}
        s["max_schemas_per_call"] = max(r["schemas"] for r in cell)
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu"], default="cpu")
    ap.add_argument("--tasks", nargs="+", default=list(lib.TASKS))
    ap.add_argument("--panels", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/ephris_axis.json")
    args = ap.parse_args()
    lib.setup_torch()
    ref = Ephris(args.device)
    rows, vectors = [], {}
    for task in args.tasks:
        for index in args.panels:
            fixture = lib.panel(task, index)
            archive = np.load(lib.OLD / "results" / f"{task}-ephris-cpu-p{index}.npz", allow_pickle=False)
            engine = Engine(ref, fixture["y"])
            for layout, topology, recipe in GRID:
                work = lib.Workload.from_panel(fixture, layout, topology == "attached")
                key = f"{layout}_{topology}_{recipe}_4__"
                assert np.array_equal(work.ids, archive[key + "ids"])
                old = {p: archive[key + p] for p in ("separate", "union")}
                out, passes, ms, public, public_ms = {}, {}, {}, {}, {}
                for plan in PLANS:
                    out[plan], passes[plan] = run(engine, work, plan, recipe)
                schemas = engine.schemas
                for plan in ("separate", "union", "axis"):          # one warm-up above, then timed runs
                    ms[plan] = []
                    for _ in range(args.repeats):
                        tic = time.perf_counter_ns()
                        again, _ = run(engine, work, plan, recipe)
                        ms[plan].append((time.perf_counter_ns() - tic) / 1e6)
                        assert np.array_equal(again, out[plan])
                for plan in ("separate", "union", "union-2"):
                    tic = time.perf_counter_ns()
                    public[plan], calls = released(ref, work, plan, recipe)
                    public_ms[plan] = (time.perf_counter_ns() - tic) / 1e6
                    assert calls == 4 // PLANS[plan]
                assert passes == {"separate": 4, "union": 1, "axis": 1, "union-2": 2, "axis-2": 2}
                y = lib.truth(task, work.ids)
                cmp = lambda a, b: lib.endpoints(work.ids, work.rid, a, b)
                row = dict(task=task, panel=index, layout=layout, topology=topology, recipe=recipe,
                           n=len(work.ids), cross_edges=work.cross, schemas=schemas, passes=passes,
                           latency_ms=ms, released_latency_ms=public_ms,
                           union=cmp(out["separate"], out["union"]), axis=cmp(out["separate"], out["axis"]),
                           axis_vs_released=cmp(public["separate"], out["axis"]),
                           archive_union=cmp(old["separate"], old["union"]),
                           own_separate_vs_archive=cmp(old["separate"], out["separate"]),
                           own_union_vs_archive=cmp(old["union"], out["union"]),
                           released_separate_vs_archive=cmp(old["separate"], public["separate"]),
                           released_union_vs_archive=cmp(old["union"], public["union"]),
                           own_union2_vs_released=cmp(public["union-2"], out["union-2"]),
                           correct={p: int((v.argmax(1) == y).sum()) for p, v in out.items()})
                row["union-2"], row["axis-2"] = cmp(out["separate"], out["union-2"]), cmp(out["separate"], out["axis-2"])
                rows.append(row)
                stem = f"{task}_p{index}_{layout}_{topology}_{recipe}__"
                vectors.update({stem + p.replace("-", ""): v for p, v in out.items()})
                vectors[stem + "ids"] = work.ids
                print(task, index, layout, topology, recipe, "union", row["union"]["flips"],
                      "axis", row["axis"]["flips"], f"{row['axis']['max_abs']:.2e}",
                      "own-vs-archive", f"{row['own_separate_vs_archive']['max_abs']:.2e}",
                      f"{row['own_union_vs_archive']['max_abs']:.2e}",
                      "ms", {p: round(float(np.median(v)), 1) for p, v in ms.items()}, flush=True)
            lib.dump(args.out, dict(rows=rows, summary=summarise(rows), threads=torch.get_num_threads(),
                                    tolerance=lib.TOL, plans=PLANS))
            np.savez_compressed(args.out.with_suffix(".npz"), **vectors)


if __name__ == "__main__":
    main()
