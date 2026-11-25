"""Does node union also change the answers of an ordinary trained message-passing network?

GCN and GraphSAGE classifiers with one and two layers are trained on each corpus with every panel
node removed from the training graph.  They take no support labels at inference: the support is
only a set of context nodes that several requests share.  Each model then scores the archived
panels under separate execution, node union and block-diagonal batching with one support copy
per request.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

import lib

SEEDS = (0, 1, 2)


class Net(torch.nn.Module):
    def __init__(self, kind, layers, width, classes, hidden=64):
        super().__init__()
        conv = {"gcn": GCNConv, "sage": SAGEConv}[kind]
        dims = [width] + [hidden] * (layers - 1) + [classes]
        self.convs = torch.nn.ModuleList(conv(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x, edge):
        for i, conv in enumerate(self.convs):
            x = conv(F.dropout(x, .5, self.training), edge)
            if i + 1 < len(self.convs):
                x = F.relu(x)
        return x


def features(task):
    x = lib.corpus(task)["x"]
    return x / np.maximum(np.abs(x).sum(1, keepdims=True), 1e-12)


def train(task, kind, layers, seed, device):
    c = lib.corpus(task)
    held = set()
    for p in range(5):
        fixture = lib.panel(task, p)
        held |= {c["pos"][str(v)] for v in fixture["support_ids"]} | {c["pos"][str(v)] for v in fixture["ids"]}
    nodes = np.array([i for i in range(len(c["names"])) if i not in held])
    x = torch.tensor(features(task)[nodes], device=device)
    y = torch.tensor(c["y"][nodes], device=device)
    edge = torch.tensor(lib.induced(task, nodes), device=device)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(nodes))
    val, fit = order[:len(order) // 5], order[len(order) // 5:]
    torch.manual_seed(seed)
    net = Net(kind, layers, x.shape[1], lib.CLASSES[task]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=.01, weight_decay=5e-4)
    best, state = -1, None
    for _ in range(200):
        net.train()
        opt.zero_grad()
        F.cross_entropy(net(x, edge)[fit], y[fit]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(x, edge)[val].argmax(1) == y[val]).float().mean())
        if acc > best:
            best, state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net.eval(), best


@torch.no_grad()
def score(net, x, edge, device):
    e = torch.as_tensor(np.ascontiguousarray(edge), dtype=torch.long, device=device)
    return net(torch.as_tensor(x, device=device), e).softmax(-1).double().cpu().numpy()


def plans(net, task, work, raw, device):
    """Probabilities for the 64 query rows under the three plans."""
    c = lib.corpus(task)
    support = raw["support"]
    query = raw["x"][[c["pos"][str(i)] for i in work.ids]]
    ns = work.ns
    sep = np.empty((len(work.ids), lib.CLASSES[task]))
    blocks_x, blocks_e, offset = [], [], 0
    for g in work.requests():
        _, e = work.part(g)
        x = np.vstack((support, query[g]))
        sep[g] = score(net, x, e, device)[ns:]
        blocks_x.append(x)
        blocks_e.append(e + offset)
        offset += len(x)
    union = score(net, np.vstack((support, query)), work.edge, device)[ns:]
    stacked = score(net, np.vstack(blocks_x), np.concatenate(blocks_e, 1), device)
    axis, offset = np.empty_like(sep), 0
    for g in work.requests():
        axis[g] = stacked[offset + ns:offset + ns + len(g)]
        offset += ns + len(g)
    return dict(separate=sep, union=union, axis=axis)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=lib.REV / "results/gnn.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    lib.setup_torch()
    torch.use_deterministic_algorithms(False)  # scatter-based training; inference comparisons are same-device
    rows = []
    for task in lib.TASKS:
        c = lib.corpus(task)
        x = features(task)
        for kind in ("gcn", "sage"):
            for layers in (1, 2):
                for seed in SEEDS:
                    net, val = train(task, kind, layers, seed, args.device)
                    for attached in (True, False):
                        total = dict(task=task, model=kind, layers=layers, seed=seed, val_accuracy=val,
                                     topology="attached" if attached else "cut",
                                     union=dict(flips=0, threshold=0, changed_lists=0, lists=0, max_abs=0.),
                                     axis=dict(flips=0, max_abs=0.), n=0,
                                     accuracy=dict(separate=0, union=0))
                        for index in range(5):
                            fixture = lib.panel(task, index)
                            work = lib.Workload.from_panel(fixture, "sorted", attached)
                            raw = dict(x=x, support=x[[c["pos"][str(v)] for v in fixture["support_ids"]]])
                            out = plans(net, task, work, raw, args.device)
                            y = lib.truth(task, work.ids)
                            u = lib.endpoints(work.ids, work.rid, out["separate"], out["union"])
                            a = lib.endpoints(work.ids, work.rid, out["separate"], out["axis"])
                            for k in ("flips", "threshold", "changed_lists", "lists"):
                                total["union"][k] += u[k]
                            total["union"]["max_abs"] = max(total["union"]["max_abs"], u["max_abs"])
                            total["axis"]["flips"] += a["flips"]
                            total["axis"]["max_abs"] = max(total["axis"]["max_abs"], a["max_abs"])
                            total["n"] += u["n"]
                            for plan in ("separate", "union"):
                                total["accuracy"][plan] += int((out[plan].argmax(1) == y).sum())
                        rows.append(total)
                print(task, kind, layers, [(r["topology"], r["union"]["flips"]) for r in rows[-6:]], flush=True)
    lib.dump(args.out, rows)


if __name__ == "__main__":
    main()
