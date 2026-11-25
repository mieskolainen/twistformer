# Rest-frame top decays and competing colour dipoles
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import torch

from .kinematics import (
    Sequence,
    angles,
    boost,
    cosine,
    displacement,
    dot,
    dr,
    mass,
    mass2,
    packed,
    pt,
    signed_mass,
    unit,
)


def _decay(p, root):
    """Measure a Durham decay tree in its rest frame"""
    if len(p) < 2 or mass2(root) < 1.0e-8:
        return p.new_zeros(188)
    m = mass(root)
    rest = boost(p, root)
    seq = Sequence(rest, "ee")
    out = []
    for count in (2, 3, 4, 5, 6, 8):
        ids = seq.indices(count, True)
        axes = seq.vectors[ids]
        cos = cosine(rest[:, None], axes)
        c = cos.amax(1)
        w = rest[:, 0] / m
        out.extend(
            (
                (w * (2 * (1 - c)).clamp_min(0.0).sqrt()).sum(),
                (w * (1 - c)).sum(),
                w[c < 0.8].sum(),
                (w[:, None] * cos.abs()).sum(0).amax(),
                seq.merges.get(count, 0.0) / m.square(),
            )
        )
        if count in (3, 4):
            for i in range(count):
                if i >= len(ids):
                    out.append(p.new_zeros(10))
                    continue
                a = axes[i]
                members = seq.members[ids[i]]
                fractions = rest[members, 0] / a[0].clamp_min(1.0e-12)
                d = 1.0 - cos[members, i]
                out.extend(
                    (
                        a[0],
                        a[0] / m,
                        mass(a),
                        cosine(a, root),
                        len(members),
                        (fractions * (2 * d).sqrt()).sum(),
                        fractions.square().sum(),
                        (fractions * d).sum(),
                        fractions.max(),
                        pt(boost(a, root, True)) / pt(root),
                    )
                )
            for i, j in combinations(range(count), 2):
                if j >= len(ids):
                    out.append(p.new_zeros(4))
                else:
                    a, b = axes[i], axes[j]
                    out.extend((mass(a + b), cosine(a, b), (a[0] - b[0]).abs() / (a[0] + b[0]), (a[0] + b[0]) / m))
        if count >= 3:
            comb = [
                (i, j, k) for i, j in combinations(range(len(ids)), 2) for k in range(len(ids)) if k != i and k != j
            ]
            if not comb:
                out.append(p.new_zeros(8))
                continue
            i, j, k = torch.tensor(comb, device=p.device).T
            a, b, q = axes[i], axes[j], axes[k]
            pair = a + b
            top = pair + q
            tm, wm = mass(top), mass(pair)
            chi = (((tm - 172.5) / 23.0) ** 2 + ((wm - 80.4) / 13.0) ** 2).masked_fill(
                mass2(pair) <= 1.0e-8, torch.inf
            )
            best = int(chi.argmin())
            if not chi[best].isfinite():
                out.append(p.new_zeros(8))
                continue
            h = cosine(boost(a[best], pair[best]), boost(q[best], pair[best])).abs()
            out.extend(
                (
                    tm[best],
                    wm[best],
                    q[best, 0],
                    h,
                    top[best, 1:].norm() / m,
                    torch.minimum(signed_mass(a[best] + q[best]), signed_mass(b[best] + q[best])),
                    (-0.5 * chi).exp().sum(),
                    len(seq.members[ids[k[best]]]) / len(p),
                )
            )
    for j in range(3):
        if j >= min(len(p), 3):
            out.append(p.new_zeros(4))
            continue
        c = cosine(rest, seq.axes(3, True)[j])
        for cut in (0.9, 0.97):
            out.extend((w[c > cut].sum(), (c > cut).sum()))
    return packed(p, *out)


def decay(event):
    """Measure ungroomed and groomed rest-frame decays"""
    ca = event.ca
    root = ca.indices(1)[0]
    groom = root
    while groom in ca.children:
        a, b = ca.hard_soft(groom)
        if pt(ca.vectors[b]) > 0.075 * (pt(ca.vectors[a]) + pt(ca.vectors[b])):
            break
        groom = a
    return torch.cat((_decay(event.p, ca.vectors[root]), _decay(event.p[ca.members[groom]], ca.vectors[groom])))


def candidate(event):
    """Measure twelve resolved and template W assignments"""
    p, root = event.p, event.root
    out = p.new_zeros((12, 116))
    for mode, groups in enumerate(event.groups):
        if not all(groups):
            continue
        axes = torch.stack([p[g].sum(0) for g in groups])
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            if pt(axes[j]) > pt(axes[i]):
                i, j = j, i
            b, a, c = axes[[k, i, j]]
            w = a + c
            t = w + b
            mt, mw = mass(t).clamp_min(1.0e-5), mass(w).clamp_min(1.0e-5)
            aw, bw = (boost(a, w), boost(b, w)) if mass2(w) > 1.0e-8 else (a, b)
            rpt = pt(root)
            fields = [
                mode + 1,
                k,
                mt,
                mw,
                mw / mt,
                mass(a + b),
                mass(c + b),
                pt(b) / rpt,
                pt(w) / rpt,
                pt(t) / rpt,
                dot(b, t) / mt,
                dot(b, t) / mt**2,
                cosine(aw, bw).abs(),
                dr(a, c),
                dr(w, b),
                dr(b, a),
                dr(b, c),
                pt(a) / pt(w).clamp_min(1.0e-9),
                mass(root),
                angles(root)[0].abs(),
                len(p),
                len(groups[k]),
                (dot(a, b) - dot(c, b)).abs() / dot(w, b).clamp_min(1.0e-9),
                (mw - 80.4).abs(),
                (mt - 172.5).abs(),
                (mw / mt - 80.4 / 172.5).abs(),
            ]
            for g in (k, i, j):
                axis = axes[g]
                parts = p[groups[g]]
                z = pt(parts) / pt(parts).sum().clamp_min(1.0e-9)
                dy = displacement(axis, parts)
                d = dy.norm(dim=1)
                fields.extend((z.square().sum(), z.sqrt().sum(), (z * d).sum(), (z * d.square()).sum(), z.max()))
                fields.extend(z[d < r].sum() for r in (0.025, 0.05, 0.1, 0.2))
                fields.extend((d < r).sum() for r in (0.025, 0.05, 0.1, 0.2))
                fields.extend(
                    ((z * d) @ dy, pt(axis) / rpt, mass(axis), len(parts), dot(axis, t) / mt, dr(axis, root))
                )
            for left, right, third in ((a, c, b), (c, b, a), (b, a, c)):
                line = displacement(left, right)
                denom = line.square().sum().clamp_min(1.0e-8)
                delta = displacement(left, p)
                along = (delta @ line) / denom
                height = (delta[:, 0] * line[1] - delta[:, 1] * line[0]).abs() / denom
                dl, dd = delta.square().sum(1), dr(p, right).square()
                inside = (
                    (dl > 0.04 * denom)
                    & (dd > 0.04 * denom)
                    & (dr(p, third).square() > torch.minimum(dl, dd))
                    & (along > 0)
                    & (along < 1)
                )
                z = pt(p) / rpt
                for r in (0.5, 0.25, 0.12):
                    keep = inside & (height < r)
                    fields.extend((z[keep].sum(), keep.sum()))
                    if r == 0.5:
                        fields.append((z[keep] * height[keep].square()).sum())
                fields.extend(
                    (signed_mass(left + right), denom.sqrt(), torch.minimum(pt(left), pt(right)) * denom.sqrt())
                )
            out[3 * mode + k] = packed(p, *fields)
    return out


def antenna(event):
    """Measure primary and secondary radiation around colour dipoles"""
    p = event.p
    output = p.new_zeros((4, 2, 62))
    for mode, groups in enumerate(event.groups):
        axes = torch.stack([p[g].sum(0) for g in groups])
        if pt(axes).min() < 1.0e-8:
            continue
        hard, directions = unit(axes[:, 1:]), unit(p[:, 1:])
        sep = (1 - hard @ hard.T).clamp_min(0.0)
        minsep = sep[[0, 1, 2], [1, 2, 0]].min().clamp_min(0.0003)
        cut = (0.008 * minsep).clamp_min(0.00003)
        ds = (1.0 - directions @ hard.T).clamp_min(0.0)
        near, idx = ds.min(1)
        z = pt(p) / pt(event.root).clamp_min(0.01)
        soft = pt(p) / pt(axes[idx])
        keep = (z <= 0.15) & (soft <= 0.5)
        z, near, d, soft, v = z[keep], near[keep], ds[keep], soft[keep], directions[keep]

        def ratio(x, y):
            return (x.clamp_min(1.0e-12) / y.clamp_min(1.0e-12)).log().clamp(-6.0, 6.0)

        f = p.new_zeros((3, 62))
        for b in range(3):
            a, c = (b + 1) % 3, (b + 2) % 3

            def dip(i, j):
                return sep[i, j] / ((d[:, i] + cut) * (d[:, j] + cut))

            def beam(i, sign):
                return (1 - sign * hard[i, 2]) / ((d[:, i] + cut) * (1 - sign * v[:, 2] + cut))

            def avg(i):
                return 0.5 * (beam(i, 1) + beam(i, -1))

            ac, ab, bc = dip(a, c), dip(a, b), dip(b, c)
            sing, c1, c2 = ac + avg(b), ab + avg(c), bc + avg(a)
            lr = ratio(sing, 0.5 * (c1 + c2))
            ring = ((d[:, a] + d[:, c]) / sep[a, c].clamp_min(0.0001)).log1p()
            close = torch.minimum(d[:, a], d[:, c])
            active = torch.stack(
                (
                    torch.ones_like(near, dtype=torch.bool),
                    near > 0.025 * minsep,
                    (near > 0.025 * minsep) & (d[:, b] > close),
                    (soft < 0.12) & (close / sep[a, c].clamp_min(0.0001) > 0.03),
                )
            )
            plus, minus = [ratio(ac + beam(b, s), 0.5 * (ab + bc + beam(c, s) + beam(a, s))) for s in (1, -1)]
            measures = torch.stack(
                (
                    lr,
                    z * lr,
                    z.sqrt() * lr,
                    lr.square(),
                    lr > 0,
                    z * (lr > 0),
                    ratio(sing, torch.maximum(c1, c2)),
                    z * ring,
                    ring,
                    ratio(ac + 0.2 * avg(b), ab + bc + 0.2 * avg(b)),
                    (plus - minus).abs(),
                    z * torch.minimum(plus, minus),
                ),
                1,
            )
            f[b, :48] = (active.double() @ measures).flatten()
            for g, off in ((0, 48), (2, 50)):
                f[b, off] = f[b, 12 * g] / active[g].sum().clamp_min(1)
                f[b, off + 1] = f[b, 12 * g + 1] / z[active[g]].sum().clamp_min(0.001)
            seed = torch.where(near >= 0.02 * minsep)[0][:5]
            uv = (1 - v[seed] @ v.T).clamp_min(0.0)
            select = (
                (torch.arange(len(z), device=p.device) > seed[:, None])
                & (near >= 0.01 * minsep)
                & (z <= 0.7 * z[seed, None])
                & (uv >= 0.015 * minsep)
            )

            def change(h, k):
                return (d[seed, h, None] * (d[:, k] + cut) + d[seed, k, None] * (d[:, h] + cut)) / (
                    sep[h, k].clamp_min(0.00001) * (uv + cut)
                )

            x = ratio(change(a, c), 0.5 * (change(a, b) + change(b, c)))
            f[b, 52:58] = packed(
                p,
                *[
                    (field * select).sum()
                    for field in (x, z * x, (z[seed, None] * z).sqrt() * x, x.square(), x > 0, torch.ones_like(x))
                ],
            )
            f[b, 58:] = packed(
                p,
                f[b, 52] / f[b, 57].clamp_min(1),
                mass(axes[a] + axes[c]),
                mass(axes[a] + axes[b] + axes[c]),
                active[2].sum(),
            )
        weights = (-0.5 * ((f[:, 59] - 80.4) / 15.0) ** 2).softmax(0)
        output[mode, 0], output[mode, 1] = f[weights.argmax()], weights @ f
    return output.flatten()
