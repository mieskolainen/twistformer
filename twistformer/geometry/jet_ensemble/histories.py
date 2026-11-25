# QCD and top decay histories on microjet subsets
#
# mikael.mieskolainen@cern.ch (2026)

from functools import lru_cache
from itertools import combinations

import torch

from .kinematics import Sequence, angles, boost, cosine, dot, dr, mass, mass2, packed, pt, pt2


@lru_cache
def partitions(n, k):
    """Enumerate unordered binary splittings at fixed subset multiplicity"""
    masks = [m for m in range(1, 1 << n) if m.bit_count() == k]
    left = [[a for a in range(1, m) if a & m == a and a > (m ^ a)] for m in masks]
    right = [[m ^ a for a in row] for m, row in zip(masks, left)]
    return masks, left, right


def subsets(axes):
    """Sum all nonempty microjet subsets"""
    n = len(axes)
    masks = torch.arange(1 << n, device=axes.device)
    bits = torch.arange(n, device=axes.device)
    members = (masks[:, None] >> bits) & 1
    return members.double() @ axes, members.double()


def radiation(axes):
    """Sum QCD, W and top likelihoods over binary recombination histories"""
    n = len(axes)
    vec, membership = subsets(axes)
    weights, m = pt(vec), mass(vec)
    fields = axes.new_zeros((6, len(vec)))
    fields[:2, membership.sum(1) == 1] = 1.0
    for k in range(2, n + 1):
        ix, a, b = [torch.tensor(v, device=axes.device) for v in partitions(n, k)]
        zl = (weights[a] / (weights[a] + weights[b]).clamp_min(1.0e-10)).clamp(0.0001, 0.9999)
        zr = 1.0 - zl
        mp = m[ix, None]
        scale = 1000 / (100 + (mp.square() - m[a].square() - m[b].square()).clamp_min(1.0))
        scale = scale / (1 + 4 * ((m[a] + m[b]) / mp.clamp_min(1.0)).pow(4))
        lq, lg, lw, _, lb, _ = fields[:, a]
        rq, rg, rw, _, rb, _ = fields[:, b]
        q = scale * ((1 + zl.square()) / zr * lq * rg + (1 + zr.square()) / zl * rq * lg)
        g = scale * (2.25 * (zl / zr + zr / zl + zl * zr) * lg * rg + 0.35 * (zl.square() + zr.square()) * lq * rq)
        w = lq * rq / (1 + ((mp - 80.4) / 10).square())
        tp = 1 / (1 + ((mp - 172.5) / 17).square())
        t = tp * (lw * rq + rw * lq)
        best = tp * torch.maximum(lb * rq, rb * lq)
        fields[:, ix] = torch.stack((q.sum(1), g.sum(1), w.sum(1), t.sum(1), w.amax(1), best.amax(1)))
    q, g, w, t, bw, bt = fields
    z = (weights / weights[-1].clamp_min(1.0e-10)).clamp_max(1.0)
    outside = n - membership.sum(1)
    keep = (z >= 0.45) & (outside <= 3) & (membership.sum(1) > 0)
    weight = z.pow(8) * 0.45**outside * keep
    rt, rw = t / q.clamp_min(1.0e-30), w / q.clamp_min(1.0e-30)
    joint = rt * weight
    best = joint.argmax()
    ratio = lambda a, b: (a / b.clamp_min(1.0e-30)).log1p()
    return packed(
        axes,
        m[-1],
        weights[-1],
        *[ratio(fields[i, -1], fields[j, -1]) for i, j in ((0, 1), (2, 0), (3, 0), (4, 0), (5, 0), (3, 1))],
        ratio(joint.sum(), weight.sum()),
        ratio((rw * weight).sum(), weight.sum()),
        joint.max().log1p(),
        joint @ m / joint.sum().clamp_min(1.0e-30),
        joint @ z / joint.sum().clamp_min(1.0e-30),
        torch.where(joint[best] > 0, m[best], 0),
        torch.where(joint[best] > 0, z[best], 0),
        *[
            (rt * z.pow(k) * coeff**outside * keep).sum().log1p()
            for coeff, ks in ((0.7, (2, 5, 8)), (0.3, (12, 18, 24)))
            for k in ks
        ],
    )


def history(event):
    """Measure inclusive anti-kt and exclusive kt subset likelihoods"""
    out = []
    for radius in (0.12, 0.22):
        seq = Sequence(event.p, "antikt", radius)
        axes = seq.vectors[seq.inclusive]
        axes = axes[pt(axes) >= 1.0]
        axes = axes[pt2(axes).argsort(descending=True)[:7]] if len(axes) else event.p[:1]
        out.append(radiation(axes))
    return torch.cat(out + [radiation(event.kt.axes(n)) for n in (5, 7)])


def microjets(event):
    """Integrate five-microjet partitions with four stochastic clustering measures"""
    p = event.p
    out = p.new_zeros((75, 40))
    valid = torch.zeros(75, dtype=torch.bool, device=p.device)
    probability = p.new_zeros((75, 4))
    if len(p) < 3:
        return out, valid, probability
    seq = event.kt
    ids = seq.indices(5)
    micro = seq.vectors[ids]
    n = len(ids)
    full = (1 << n) - 1
    natural = []
    for node in seq.indices(3):
        members = set(seq.members[node])
        natural.append(sum(1 << j for j, k in enumerate(ids) if seq.members[k][0] in members))
    vec, which = subsets(micro)
    z, m = pt(vec), mass(vec)
    distance = (torch.minimum(z[:, None].square(), z.square()) * dr(vec[:, None], vec).square()).clamp_min(1.0e-15)
    paths = {}

    def collapse(groups, path):
        if len(groups) == 3:
            key = tuple(sorted(groups))
            paths[key] = paths.get(key, 0.0) + path
            return
        pairs = list(combinations(range(len(groups)), 2))
        d = distance[[groups[i] for i, j in pairs], [groups[j] for i, j in pairs]]
        q = d.min() / d
        trans = torch.stack((q.sqrt(), q, q.square(), q.pow(4)), 1)
        trans = trans / trans.sum(0)
        for k, (i, j) in enumerate(pairs):
            merged = list(groups)
            merged[i] |= merged[j]
            merged[j] = merged[-1]
            collapse(merged[:-1], path * trans[k])

    collapse([1 << j for j in range(n)], p.new_ones(4))
    choices = []
    for b in range(1, full):
        rest = full ^ b
        a = (rest - 1) & rest
        while a:
            c = rest ^ a
            if a <= c:
                choices.append((b, a, c))
            a = (a - 1) & rest
    b, a, c = torch.tensor(choices, device=p.device).T
    conventional = torch.tensor([bb in natural and aa in natural for bb, aa, _ in choices], device=p.device)
    hard = z[a] >= z[c]
    a, c = torch.where(hard, a, c), torch.where(hard, c, a)
    w = vec[a] + vec[c]
    mw, mt = mass(w), m[-1].expand(len(a))
    den = mt.clamp_min(1.0e-9)
    aw, bw = boost(vec[a], w), boost(vec[b], w)
    hel = torch.where(mass2(w) > 1.0e-8, cosine(aw, bw).abs(), 0)
    m1, m2 = mass(vec[a] + vec[b]), mass(vec[c] + vec[b])
    da, dc = dr(vec[b], vec[a]), dr(vec[b], vec[c])
    rpt = z[-1].clamp_min(1.0e-9)
    cols = [
        conventional,
        mt,
        mw,
        mw / den,
        torch.minimum(m1, m2),
        torch.maximum(m1, m2),
        (mw - 80.4).abs(),
        (mt - 172.5).abs(),
        hel,
        pt(w) / rpt,
        torch.minimum(z[a], z[c]) / (z[a] + z[c]).clamp_min(1.0e-10),
        dr(vec[a], vec[c]),
        torch.minimum(da, dc),
        torch.maximum(da, dc),
        z[-1].expand_as(mt),
        angles(vec[-1])[0].abs().expand_as(mt),
    ]
    for g in (b, a, c):
        membership = which[g]
        scalar = membership @ pt(micro)
        cols.extend(
            (
                membership.sum(1),
                membership @ p.new_tensor([len(seq.members[k]) for k in ids]),
                m[g],
                z[g] / rpt,
                dot(vec[g], vec[-1]) / den.square(),
                (membership * pt(micro)).amax(1) / scalar.clamp_min(1.0e-12),
                membership @ mass(micro),
                dr(vec[g], vec[-1]),
            )
        )
    out[: len(a)] = torch.stack(cols, 1)
    valid[: len(a)] = True
    probability[: len(a)] = torch.stack([paths[tuple(sorted(row))] for row in choices])
    return out, valid, probability
