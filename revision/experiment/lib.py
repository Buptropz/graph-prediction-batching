"""Shared pieces of the revision experiments: engine, endpoints, corpora and the SQL operator.

The archived study lives in vldb2027/upgrade and is imported unchanged.  Everything here adds to it:
a request-axis engine that pads unequal requests and stacks ensemble members, an Arrow-typed DuckDB
operator, per-phase timers, and NumPy implementations of the three relational endpoints.
"""
import itertools
import json
import os
import random
import sys
import time
import warnings
from contextlib import ExitStack
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pyarrow as pa
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT / "vldb2027/upgrade"
sys.path.insert(0, str(OLD / "experiment"))
import run as archived  # noqa: E402  (also puts companion_dose on sys.path)
from inference import NodePFN, focal_prob  # noqa: E402

TASKS = ("cora", "pubmed", "chameleon")
CLASSES = {"cora": 7, "pubmed": 3, "chameleon": 5}
TOL = 1e-5
SUPPORT = 32


def setup_torch():
    torch.set_num_threads(int(os.environ.get("REV_THREADS", 2)))
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Clock:
    """Accumulates wall time per phase; synchronises the GPU at phase boundaries when enabled."""
    def __init__(self, device):
        self.device, self.on, self.t = device, False, {}

    def reset(self):
        self.t = {}

    def lap(self, name, start):
        if not self.on:
            return start
        sync(self.device)
        now = time.perf_counter_ns()
        self.t[name] = self.t.get(name, 0.0) + (now - start) / 1e6
        return now

    def start(self):
        if self.on:
            sync(self.device)
        return time.perf_counter_ns()


class Engine:
    """NodePFN request-axis execution for any number of ensemble members and unequal request lengths.

    One lane is one (request, ensemble member) pair.  Lanes occupy the network's batch dimension.
    Each lane has its own feature fit, its own copy of the support and its own block of graph edges.
    Requests shorter than the longest one are padded with rows that have no edge except a self loop
    and that no attention query reads, because query rows attend to support rows only.
    """
    def __init__(self, reference, y):
        clf = reference.clf
        self.ref, self.api, self.clf = reference, reference.api, clf
        self.net = clf.model[2].to(reference.device).eval()
        self.device = reference.device
        self.y = np.asarray(y)
        self.classes = len(np.unique(y))
        assert np.array_equal(np.unique(y), np.arange(self.classes))
        assert (clf.multiclass_decoder == "permutation" and clf.feature_shift_decoder
                and not clf.differentiable_hps_as_style and not clf.c["normalize_to_ranking"]
                and not clf.c.get("normalize_with_sqrt", False))
        assert self.net.use_gps_style and self.net.pos_encoder is None and self.net.input_ln is None
        assert self.net.efficient_eval_masking and not self.net.full_attention
        self.layers = [layer.local_model for layer in self.net.transformer_encoder.layers]
        assert all(type(layer).__name__ == "GCNConv" and not layer.cached for layer in self.layers)
        self.width = clf.c["num_features"]
        self.members = clf.N_ensemble_configurations
        self.clock = Clock(self.device)
        self.forwards = 0
        self._configs = {}

    def configs(self, features):
        """The ensemble members the release would draw for this number of input features."""
        if features not in self._configs:
            generator = torch.Generator().manual_seed(self.clf.seed)
            feats = torch.randperm(features, generator=generator).tolist()
            classes = torch.randperm(self.classes, generator=generator).tolist()
            shifts = list(itertools.product(classes, feats))
            random.Random(self.clf.seed).shuffle(shifts)
            transforms = ["none"] if self.clf.no_preprocess_mode else ["none", "power_all"]
            self._configs[features] = list(itertools.product(shifts, transforms))[:self.members]
        return self._configs[features]

    def prepare(self, support, queries, recipe):
        """Feature preparation for one request; returns one tensor per ensemble member."""
        from sklearn.preprocessing import PowerTransformer
        ns = len(support)
        raw = torch.tensor(np.vstack((support, queries)), device=self.device).float()[:, None, :]
        end = ns if recipe == "support" else -1
        base = self.api.normalize_data(raw, normalize_positions=end)[:, 0, :]
        keep = [base[:ns, j].unique().numel() > 1 for j in range(base.shape[-1])]
        base = base[:, keep]
        cache, out = {}, []
        for (class_shift, feature_shift), transform in self.configs(raw.shape[-1]):
            if transform not in cache:
                x = base
                if transform == "power_all":
                    values = x.cpu().numpy()
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        for col in range(values.shape[1]):
                            try:
                                pt = PowerTransformer(standardize=True)
                                pt.fit(values[:ns, col:col + 1])
                                values[:, col:col + 1] = pt.transform(values[:, col:col + 1])
                            except Exception:
                                pass
                    x = torch.tensor(values).float()  # the release finishes this member on the host
                x = self.api.remove_outliers(x[:, None, :], normalize_positions=end)
                cache[transform] = self.api.normalize_by_used_features_f(x, x.shape[-1], self.width).to(self.device)
            x = cache[transform]
            x = torch.cat((x[..., feature_shift:], x[..., :feature_shift]), dim=-1)
            out.append(torch.nn.functional.pad(x, (0, self.width - x.shape[-1])))
        return out

    @torch.inference_mode()
    def predict(self, support, requests, edges, recipe):
        """Probabilities for every query row of every request, in request order."""
        clock = self.clock
        tic = clock.start()
        ns, members = len(support), self.members
        prepared = [self.prepare(support, q, recipe) for q in requests]
        tic = clock.lap("fit", tic)
        sizes = [len(q) for q in requests]
        n = ns + max(sizes)
        lanes = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, n - x.shape[0])) for xs in prepared for x in xs]
        x = torch.cat(lanes, dim=1)
        batch = x.shape[1]
        shifts = [cfg[0][0] for cfg in self.configs(np.shape(support)[1])]
        labels = torch.tensor(self.y, device=self.device)
        y = torch.stack([((labels + s) % self.classes).float() for _ in requests for s in shifts], dim=1)
        ee = [torch.as_tensor(np.ascontiguousarray(e), device=self.device, dtype=torch.long) for e in edges]
        with ExitStack() as stack:
            if batch > 1:
                blocks = []
                for j, e in enumerate(ee):
                    canonical, _ = remove_self_loops(to_undirected(e))
                    canonical, _ = add_self_loops(canonical, num_nodes=n)
                    blocks += [canonical + (j * members + m) * n for m in range(members)]
                block = torch.cat(blocks, dim=1)

                def pack(module, args):
                    values, _ = args
                    assert values.shape[:2] == (batch, n)
                    return values.reshape(batch * n, -1), block

                def unpack(module, args, out):
                    return out.reshape(batch, n, -1)

                for layer in self.layers:
                    stack.callback(layer.register_forward_pre_hook(pack).remove)
                    stack.callback(layer.register_forward_hook(unpack).remove)
            tic = clock.lap("graph", tic)
            self.forwards += 1
            logits = self.net((None, x, y, ee[0]), single_eval_pos=ns)[..., :self.classes] / 0.8
        logits = logits.reshape(n - ns, len(requests), members, self.classes)
        mean = sum(torch.roll(logits[:, :, m], -s, dims=-1) for m, s in enumerate(shifts)) / members
        prob = mean.softmax(-1)
        out = torch.cat([prob[:size, j] for j, size in enumerate(sizes)]).double().cpu().numpy()
        clock.lap("forward", tic)
        return out


def reference(device, ensemble=1):
    return NodePFN(device, ensemble=ensemble)


def public(ref, tx, y, q, e, recipe):
    """The released single-request classifier interface."""
    return focal_prob(ref(tx, y, q, e, recipe), len(q), False)


# ---------------------------------------------------------------------------------------------
# Relational endpoints
# ---------------------------------------------------------------------------------------------
def top_lists(ids, rid, p, k=3):
    """Top-k ID sets per (request, class), ordered by descending score then stable ID."""
    out = {}
    for r in np.unique(rid):
        ix = np.flatnonzero(rid == r)
        for c in range(p.shape[1]):
            out[(int(r), c)] = frozenset(ids[ix[np.lexsort((ids[ix], -p[ix, c]))[:k]]].tolist())
    return out


def endpoints(ids, rid, a, b, k=3):
    """Differences between a reference result ``a`` and a plan result ``b`` on the same rows."""
    ids, rid = np.asarray(ids), np.asarray(rid)
    la, lb = top_lists(ids, rid, a, k), top_lists(ids, rid, b, k)
    moved = a.argmax(1) != b.argmax(1)
    diff = np.abs(a - b)
    return dict(n=int(len(ids)), flips=int(moved.sum()),
                threshold=int(np.count_nonzero((a >= .5) != (b >= .5))),
                lists=len(la), changed_lists=int(sum(la[key] != lb[key] for key in la)),
                max_abs=float(diff.max()), tv=float(diff.sum(1).mean() / 2))


def uncertified(rid, p, k=3, tol=TOL):
    """Requests whose answer to a class, threshold or top-k query could change under error <= tol."""
    risky = []
    for r in np.unique(rid):
        v = p[rid == r]
        top = np.sort(v, axis=1)
        margin = (top[:, -1] - top[:, -2]).min()
        near = np.abs(v - .5).min()
        gap = np.inf
        if len(v) > k:
            ranked = -np.sort(-v, axis=0)
            gap = (ranked[k - 1] - ranked[k]).min()
        if margin <= 2 * tol or near <= tol or gap <= 2 * tol:
            risky.append(int(r))
    return risky


# ---------------------------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------------------------
_CORPUS = {}


def corpus(task):
    """Raw features, labels, node names and undirected node pairs of one source graph."""
    if task not in _CORPUS:
        from build_cohorts import node_data
        x, y, names, edges, _ = node_data(task)
        _, y = np.unique(y, return_inverse=True)
        pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        _CORPUS[task] = dict(x=x, y=y, names=np.asarray([str(v) for v in names]), pairs=pairs,
                             pos={str(v): i for i, v in enumerate(names)})
    return _CORPUS[task]


def truth(task, ids):
    c = corpus(task)
    return c["y"][[c["pos"][str(i)] for i in ids]]


def neighbours(task):
    c = corpus(task)
    adj = [set() for _ in c["names"]]
    for a, b in c["pairs"]:
        if a != b:
            adj[a].add(int(b))
            adj[b].add(int(a))
    return adj


def panel(task, index):
    return np.load(OLD / "experiment/inputs" / task / f"p{index}.npz", allow_pickle=False)


def induced(task, nodes):
    """Symmetric edge index among ``nodes`` (corpus indices), positions in the given order."""
    pos = {int(v): i for i, v in enumerate(nodes)}
    out = set()
    for a, b in corpus(task)["pairs"]:
        if a != b and int(a) in pos and int(b) in pos:
            out.add((pos[int(a)], pos[int(b)]))
            out.add((pos[int(b)], pos[int(a)]))
    return np.asarray(sorted(out), dtype=np.int64).reshape(-1, 2).T


class Workload:
    """A support, query rows in a fixed order, a request id per row and the logical edges.

    Logical edges are support-support, query-query inside one request and, when ``attached``,
    support-query.  No logical edge joins two requests.
    """
    def __init__(self, tx, y, qx, ids, rid, edge, attached=True):
        self.tx, self.y, self.q = np.asarray(tx), np.asarray(y), np.asarray(qx)
        self.ids, self.rid = np.asarray(ids), np.asarray(rid)
        self.ns = ns = len(self.y)
        a, b = np.asarray(edge)
        group = np.r_[np.full(ns, -1), self.rid]
        both_support = (a < ns) & (b < ns)
        both_query = (a >= ns) & (b >= ns) & (group[a] == group[b])
        cross = (a < ns) != (b < ns)
        keep = both_support | both_query | (cross if attached else False)
        self.edge = np.stack([a[keep], b[keep]])
        self.cross = int((cross & keep).sum())

    def part(self, positions):
        positions = np.asarray(positions)
        remap = np.full(self.ns + len(self.q), -1)
        remap[:self.ns] = np.arange(self.ns)
        remap[self.ns + positions] = self.ns + np.arange(len(positions))
        edges = remap[self.edge]
        return self.q[positions], edges[:, (edges >= 0).all(0)]

    def requests(self):
        return [np.flatnonzero(self.rid == r) for r in np.unique(self.rid)]

    @classmethod
    def from_panel(cls, fixture, layout="sorted", attached=True, sizes=(16, 16, 16, 16)):
        order = fixture["ranked"] if layout == "sorted" else np.arange(len(fixture["ids"]))
        ns = len(fixture["y"])
        remap = np.argsort(np.r_[np.arange(ns), ns + order])
        return cls(fixture["tx"], fixture["y"], fixture["qx"][order], fixture["ids"][order],
                   np.repeat(np.arange(len(sizes)), sizes), remap[fixture["edge"]], attached)


# ---------------------------------------------------------------------------------------------
# Physical plans
# ---------------------------------------------------------------------------------------------
PLANS = ("separate", "union", "axis")


def buckets(work, plan):
    """Which requests share a model call under each plan (lists of row positions per lane)."""
    reqs = work.requests()
    if plan == "separate":
        return [[r] for r in reqs]
    if plan == "union":
        return [[np.concatenate(reqs)]]
    if plan == "axis":          # one padded call for all requests
        return [reqs]
    if plan == "axis-length":   # one call per distinct request length, no padding
        sizes = sorted({len(r) for r in reqs})
        return [[r for r in reqs if len(r) == s] for s in sizes]
    raise ValueError(plan)


def execute(engine, work, plan, recipe, guard=False):
    """Run one plan directly (no SQL).  Returns probabilities in row order and the call count."""
    out = np.empty((len(work.ids), engine.classes))
    calls = 0
    for lanes in buckets(work, "axis" if plan == "axis-guard" else plan):
        parts = [work.part(g) for g in lanes]
        q, e = zip(*parts)
        out[np.concatenate(lanes)] = engine.predict(work.tx, q, e, recipe)
        calls += 1
    fallback = []
    if plan == "axis-guard":
        fallback = uncertified(work.rid, out)
        for r in fallback:
            g = np.flatnonzero(work.rid == r)
            q, e = work.part(g)
            out[g] = engine.predict(work.tx, [q], [e], recipe)
            calls += 1
    return out, calls, fallback


SQL = """
WITH buckets AS MATERIALIZED (
  SELECT {bucket} AS bucket, list(pos ORDER BY pos) AS positions
  FROM requests GROUP BY 1
), evaluated AS MATERIALIZED (
  SELECT positions, graph_scores(positions) AS scores FROM buckets
), rows AS (
  SELECT unnest(positions) AS pos, unnest(scores) AS p FROM evaluated
)
SELECT r.pos, r.id, e.p, list_position(e.p, list_max(e.p)) - 1 AS topic
FROM rows e JOIN requests r USING (pos) ORDER BY r.pos
"""
BUCKET = {"separate": "rid", "union": "0", "axis": "0", "axis-guard": "0", "axis-length": "request_size"}


class Operator:
    """The DuckDB table operator.  ``arrow`` returns a columnar list; ``native`` returns Python lists."""
    def __init__(self, engine, work, recipe):
        self.engine, self.work, self.recipe = engine, work, recipe
        self.plan, self.calls, self.fallback, self.udf_ms = None, 0, [], 0.0

    def score(self, positions):
        positions = np.asarray(positions)
        if self.plan == "separate" or self.plan.startswith("union"):
            lanes = [positions]
        else:
            rid = self.work.rid[positions]
            lanes = [positions[rid == r] for r in np.unique(rid)]
        parts = [self.work.part(g) for g in lanes]
        q, e = zip(*parts)
        prob = self.engine.predict(self.work.tx, q, e, self.recipe)
        self.calls += 1
        if self.plan == "axis-guard":
            rid = self.work.rid[np.concatenate(lanes)]
            risky = uncertified(rid, prob)
            for r in risky:
                rows = np.flatnonzero(rid == r)
                q1, e1 = self.work.part(np.concatenate(lanes)[rows])
                prob[rows] = self.engine.predict(self.work.tx, [q1], [e1], self.recipe)
                self.calls += 1
            self.fallback += risky
        return prob

    def native(self, positions):
        tic = time.perf_counter_ns()
        out = self.score(positions).tolist()
        self.udf_ms += (time.perf_counter_ns() - tic) / 1e6
        return out

    def arrow(self, column):
        tic = time.perf_counter_ns()
        probs = [self.score(np.asarray(v)) for v in column.to_pylist()]
        flat = pa.array(np.concatenate([p.ravel() for p in probs]))
        c = self.engine.classes
        inner = pa.ListArray.from_arrays(pa.array(np.arange(0, len(flat) + 1, c, dtype=np.int32)), flat)
        outer = pa.array(np.r_[0, np.cumsum([len(p) for p in probs])].astype(np.int32))
        out = pa.ListArray.from_arrays(outer, inner)
        self.udf_ms += (time.perf_counter_ns() - tic) / 1e6
        return out

    def register(self, db, kind):
        if kind == "arrow":
            db.create_function("graph_scores", self.arrow, ["INTEGER[]"], "DOUBLE[][]", type="arrow",
                               side_effects=True)
        else:
            db.create_function("graph_scores", self.native, ["INTEGER[]"], "DOUBLE[][]", side_effects=True)

    def load(self, db):
        sizes = np.bincount(self.work.rid)
        db.execute("CREATE TABLE requests(pos INTEGER, rid INTEGER, id VARCHAR, request_size INTEGER)")
        db.executemany("INSERT INTO requests VALUES (?,?,?,?)",
                       [(i, int(r), str(u), int(sizes[r])) for i, (r, u) in enumerate(zip(self.work.rid, self.work.ids))])

    def run(self, db, plan):
        """Execute the complete SQL query for one plan; returns probabilities, ms, calls, UDF ms."""
        self.plan, self.calls, self.fallback, self.udf_ms = plan, 0, [], 0.0
        sync(self.engine.device)
        tic = time.perf_counter_ns()
        rows = db.execute(SQL.format(bucket=BUCKET[plan])).fetchall()
        sync(self.engine.device)
        ms = (time.perf_counter_ns() - tic) / 1e6
        assert [r[0] for r in rows] == list(range(len(self.work.ids)))
        assert [r[1] for r in rows] == self.work.ids.tolist()
        p = np.asarray([r[2] for r in rows])
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-6)
        assert np.array_equal(p.argmax(1), [r[3] for r in rows])
        return p, ms, self.calls, self.udf_ms


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")
