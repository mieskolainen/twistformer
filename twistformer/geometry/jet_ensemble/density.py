# Persistence and prong kinematics of angular energy-density peaks
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import torch
from .kinematics import boost, displacement, packed, pt, unit


def _mass(p):
    return (p[..., 0].square() - p[..., 1:].square().sum(-1)).clamp_min(0.0).sqrt()


def _edges(d2):
    """Connect five nearest neighbours and the minimum spanning tree"""
    n = len(d2)
    edge = {}

    def add(i, j):
        if i != j:
            edge[tuple(sorted((i, j)))] = True

    for i, neighbors in enumerate(d2.argsort(dim=1, stable=True)[:, :5].tolist()):
        for j in neighbors:
            add(i, j)
    best = d2.new_full((n,), 1.0e30)
    best[0] = 0
    nearest = [0] * n
    used = torch.zeros(n, dtype=torch.bool, device=d2.device)
    for _ in range(n):
        j = int(best.masked_fill(used, torch.inf).argmin())
        used[j] = True
        add(j, nearest[j])
        mask = (~used) & (d2[j] < best)
        for i in torch.where(mask)[0].tolist():
            nearest[i] = j
        best = torch.where(mask, d2[j], best)
    return list(edge)


def _peaks(p, points, d2, edges, z, sigma):
    """Merge adjacent density peaks in decreasing saddle density order"""
    n = len(p)
    half = (-d2 / (4 * sigma * sigma)).exp()
    birth = half.square() @ z
    parent = list(range(n))
    up = list(range(n))
    fraction = z.clone()
    dying = torch.zeros_like(z)
    death = torch.zeros_like(z)

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if edges:
        a, b = torch.tensor(edges, device=p.device).T
        displacement2 = 0.5 * (d2[a] + d2[b]) - 0.25 * d2[a, b, None]
        mid = (-displacement2.clamp_min(0) / (2 * sigma * sigma)).exp() @ z
        level = torch.minimum(mid, torch.minimum(birth[a], birth[b]))
        for idx in level.argsort(descending=True, stable=True).tolist():
            i, j = map(find, edges[idx])
            if i == j:
                continue
            if birth[i] < birth[j] or (birth[i] == birth[j] and i > j):
                i, j = j, i
            parent[j], up[j] = i, i
            death[j], dying[j] = level[idx], fraction[j]
            fraction[i] += fraction[j]
    root = find(0)
    dying[root] = 1.0
    norm = birth[root].clamp_min(1.0e-10)
    life = (birth - death).clamp_min(0)
    life[root] = 0.0
    peaks = torch.where(life >= 1.0e-12)[0]
    peaks = peaks[life[peaks].argsort(descending=True, stable=True)].tolist()
    values = life[peaks]
    total = values.sum()
    out = [
        norm,
        total / norm,
        values.square().sum() / norm.square(),
        values.pow(3).sum() / norm.pow(3),
        -(values / total * (values / total).log()).sum() if len(peaks) else 0.0,
        (values > 0.05 * birth[root]).sum(),
        (values > 0.2 * birth[root]).sum(),
    ]
    for j in range(4):
        if j >= len(peaks):
            out.append(p.new_zeros(4))
        else:
            i = peaks[j]
            out.extend((life[i] / norm, death[i] / birth[i], dying[i], d2[i, up[i]].sqrt()))
    selected = [root] + peaks[:3]
    ids = []
    for i in range(n):
        while i not in selected:
            i = up[i]
        ids.append(selected.index(i))
    group = torch.tensor(ids, device=p.device)
    seed = torch.tensor(selected, device=p.device)[group]
    membership = (group == torch.arange(4, device=p.device)[:, None]).to(p.dtype)
    weights = membership @ z
    axes = membership @ p
    width = (membership @ (z * d2[torch.arange(n, device=p.device), seed]) / weights.clamp_min(1.0e-12)).sqrt()
    order = weights.argsort(descending=True, stable=True)
    out.append(torch.stack((weights, _mass(axes), width), 1)[order])
    a, b, c = axes[order[:3]]
    out.extend((_mass(a + b), _mass(a + c), _mass(b + c), _mass(a + b + c)))
    comb = [(i, j, k) for i, j in combinations(range(4), 2) for k in range(4) if i != k and j != k]
    i, j, k = torch.tensor(comb, device=p.device).T
    w = axes[i] + axes[j]
    mw, mt = _mass(w), _mass(w + axes[k])
    chi = (((mw - 80.4) / 14.0) ** 2 + ((mt - 172.5) / 23.0) ** 2).masked_fill(
        weights[i] * weights[j] * weights[k] <= 0.0, 1.0e5
    )
    best = chi.argmin()
    out.extend(
        (
            torch.where(chi[best] < 1.0e5, mw[best], 0.0),
            torch.where(chi[best] < 1.0e5, mt[best], 0.0),
            chi[best].clamp_max(1.0e5).log1p(),
        )
    )
    return packed(p, *out)


def density_tree(event):
    """Measure six density scales in canonical lab and rest coordinates"""
    p = event.p
    order = torch.arange(len(p), device=p.device)
    for axis in (3, 2, 1, 0):
        order = order[p[order, axis].argsort(descending=True, stable=True)]
    p = p[order]
    root = p.sum(0)
    out = []
    for rest in (False, True):
        if rest:
            q = boost(p, root) if _mass(root) > 1.0e-7 else p
            points, w = unit(q[:, 1:]), q[:, 0].clamp_min(0)
        else:
            points = torch.stack(((p[:, 3] / pt(p)).asinh(), displacement(root, p)[:, 1]), 1)
            w = pt(p)
        d2 = (points[:, None] - points).square().sum(2)
        edges = _edges(d2)
        for scale in (0.13, 0.27, 0.52) if rest else (0.035, 0.08, 0.17):
            out.append(_peaks(p, points, d2, edges, w / w.sum().clamp_min(1.0e-15), scale))
    return torch.cat(out)
