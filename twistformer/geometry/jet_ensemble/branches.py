# Local particle and clustering-tree observables for the frozen tagger
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from .kinematics import angles, boost, pt2, displacement, dot, dr, mass, mass2, packed, pt


def branch_ids(seq):
    """Enumerate splittings in harder-daughter-first order"""
    stack = [(seq.indices(1)[0], -1, -1, -1, 0, 0)]
    rows = []
    while stack:
        node, parent, sibling, uncle, depth, soft = stack.pop()
        if node not in seq.children:
            continue
        h, s = seq.hard_soft(node)
        rows.append((node, h, s, parent, sibling, uncle, depth, soft))
        stack.append((s, node, h, sibling, depth + 1, soft + 1))
        stack.append((h, node, s, sibling, depth + 1, soft))
    return rows


def _split(seq, rows):
    """Resolve branch and ancestor kinematics for every splitting"""
    v = seq.vectors
    ids = torch.tensor(rows, device=v.device).T
    p, h, s = (v[i] for i in ids[:3])
    angle = dr(h, s)
    rel = pt(s) * angle
    z = pt(s) / (pt(h) + pt(s)).clamp_min(1.0e-10)
    count = v.new_tensor([len(seq.members.get(i, [])) for i in range(len(v))])
    where = torch.zeros(len(v), dtype=torch.long, device=v.device)
    where[ids[0]] = torch.arange(len(p), device=v.device)
    return ids, p, h, s, angle, rel, z, count, where


def shower(event):
    """Measure 32 ranked splittings for each clustering metric"""
    root = event.root
    out = root.new_zeros((2, 32, 52))
    rpt, mt = pt(root).clamp_min(1.0e-8), mass(root).clamp_min(0.001)
    for mode, seq in enumerate((event.ca, event.kt)):
        rows = branch_ids(seq)
        if not rows:
            continue
        ids, p, h, s, d, rel, z, count, where = _split(seq, rows)
        node, hi, si, par, sib, unc, depth, soft = ids
        valid = par >= 0
        pi = where[par]
        v = seq.vectors
        f = root.new_zeros((len(rows), 52))
        for j, value in enumerate(
            (
                pt(p),
                mass(p),
                pt(p) / rpt,
                dot(p, root) / mt**2,
                dr(p, root),
                angles(p)[:, 0].abs(),
                angles(root)[0].abs(),
                mt,
                rpt,
                len(event.p),
                count[node],
                depth,
                soft,
                torch.where(valid, rel[pi], 0),
                torch.where(valid, z[pi], 0),
                torch.where(valid, d[pi], 0),
                torch.where(valid, pt(p) / pt(v[par]).clamp_min(0.001), 1),
            )
        ):
            f[:, j] = value
        for start, relative in ((17, sib), (22, unc)):
            q = v[relative]
            data = torch.stack((pt(q) / rpt, mass(q), dr(p, q), mass(p + q), count[relative]), 1)
            f[:, start : start + 5] = torch.where((relative >= 0)[:, None], data, 0)
        f[:, 27] = mode
        delta = displacement(h, s)
        f[:, 28:43] = torch.stack(
            (
                z,
                d,
                rel,
                mass(h),
                mass(s),
                pt(h) / rpt,
                pt(s) / rpt,
                count[hi],
                count[si],
                delta[:, 0],
                delta[:, 1],
                mass(h) / mass(p).clamp_min(0.001),
                mass(s) / mass(p).clamp_min(0.001),
                dot(h, root) / mt**2,
                dot(s, root) / mt**2,
            ),
            1,
        )
        for start, relative in ((43, sib), (47, unc)):
            q = v[relative]
            vals = [mass(h + q), mass(s + q), dr(s, q) - dr(h, q)]
            if start == 43:
                line = displacement(p, q)
                vals.append((delta * line).sum(1) / (d * dr(p, q)).clamp_min(1.0e-10))
            f[:, start : start + len(vals)] = torch.where((relative >= 0)[:, None], torch.stack(vals, 1), 0)
        f[:, 50] = rel / f[:, 13].clamp_min(0.01)
        f[:, 51] = d / f[:, 15].clamp_min(0.005)
        take = torch.arange(min(32, len(rows)), device=p.device)
        if len(rows) > 32:
            take = torch.where(take < 12, take, 12 + (take - 12) * (len(rows) - 12) // 20)
        out[mode, : len(take)] = f[rel.argsort(descending=True, stable=True)[take]]
    return out


def objects(event, family):
    """Measure local coordinates on constituents or Cambridge–Aachen branches"""
    seq = event.ca if family else event.kt
    particles = event.p
    root = event.kt.axes(1)[0]
    rpt, mt = pt(root).clamp_min(1.0e-12), mass(root).clamp_min(0.001)
    if family:
        rows = branch_ids(seq)
        if not rows:
            return (
                particles.new_empty((0, 72)).float(),
                particles.new_empty(0),
                torch.empty(0, dtype=torch.long, device=root.device),
            )
        ids, p, h, s, d, rel, z, count, where = _split(seq, rows)
    else:
        p = particles
    f = p.new_zeros((len(p), 72), dtype=torch.float32)
    frac = pt(p) / rpt
    disp = displacement(root, p)
    for j, value in enumerate(
        (
            pt(p),
            frac,
            dr(root, p),
            disp[:, 0],
            disp[:, 1],
            (dot(root, p) / mt**2).clamp_min(0),
            angles(root)[0].abs(),
            mt,
            pt(p).sqrt() * dr(root, p),
        )
    ):
        f[:, j] = value
    axes = event.kt.axes(3)
    ds = dr(p[:, None], axes)
    nearest = ds.argmin(1)
    a = axes[nearest]
    f[:, 9 : 9 + len(axes)] = ds
    f[:, 12 : 12 + len(axes)] = pt(axes) / rpt
    f[:, 15:21] = torch.stack(
        (nearest, pt(p) / pt(a).clamp_min(1.0e-12), mass(a), displacement(root, a)[:, 0], dr(a, p), mass(p + a)), 1
    )
    if len(axes) == 3:
        b, c = axes[(nearest + 1) % 3], axes[(nearest + 2) % 3]
        d1, d2 = dr(a, b), dr(a, c)
        offset = displacement(a, p)
        f[:, 21:28] = torch.stack(
            (
                mass(b + c),
                mass(b + p),
                mass(c + p),
                (offset * displacement(a, b)).sum(1) / d1.clamp_min(1.0e-10),
                (offset * displacement(a, c)).sum(1) / d2.clamp_min(1.0e-10),
                d1,
                d2,
            ),
            1,
        )
    core = event.kt.axes(8)
    q = core[dr(p[:, None], core).argmin(1)]
    f[:, 28:31] = torch.stack((pt(q) / rpt, dr(p, q), mass(q)), 1)
    f[:, 31] = len(particles)
    if family:
        node, hi, si, par, sibling, _, depth, soft = ids
        pi = where[par]
        valid = par >= 0
        f[:, 32:50] = torch.stack(
            (
                rel,
                d,
                z,
                mass(p),
                mass(h),
                mass(s),
                pt(h) / rpt,
                pt(s) / rpt,
                -displacement(h, s)[:, 0],
                displacement(h, s)[:, 1],
                count[hi],
                count[si],
                torch.where(valid, rel[pi], 0),
                torch.where(valid, d[pi], 0),
                torch.where(valid, z[pi], 0),
                depth,
                soft,
                torch.where(valid, pt(p) / pt(seq.vectors[par]).clamp_min(1.0e-15), 1),
            ),
            1,
        )
        for start, child, other in ((50, hi, s), (58, si, h)):
            leaf = count[child] == 1
            ix = where[child]
            f[:, start : start + 8] = torch.where(
                leaf[:, None],
                0.0,
                torch.stack(
                    (
                        rel[ix],
                        d[ix],
                        z[ix],
                        mass(h[ix]),
                        mass(s[ix]),
                        mass(other + s[ix]),
                        mass(other + h[ix]),
                        count[si[ix]],
                    ),
                    1,
                ),
            )
        f[:, 66:] = torch.stack((mass(h) / mt, mass(s) / mt, mass(p) / mt, dr(h, root), dr(s, root), rel / rpt), 1)
    else:
        ds = dr(p[:, None], p)
        ds.fill_diagonal_(torch.inf)
        lower = 0.0
        for j, upper in enumerate((0.01, 0.03, 0.06, 0.1, 0.2, 0.4)):
            mask = (ds >= lower) & (ds < upper)
            weights = mask * frac
            f[:, 32 + j], f[:, 38 + j], f[:, 44 + j], f[:, 50 + j] = (
                weights.sum(1),
                mask.sum(1),
                weights.amax(1),
                weights.square().sum(1),
            )
            lower = upper
        for start, distance in ((56, ds), (58, ds.masked_fill(pt(p)[None, :] <= pt(p)[:, None], torch.inf))):
            val, idx = distance.min(1)
            f[:, start] = val.clamp_max(2.0)
            f[:, start + 1] = torch.where(val.isfinite(), frac[idx], 0)
        pm = mass(p[:, None] + p)
        pm.fill_diagonal_(torch.inf)
        f[:, 60] = pm.amin(1).clamp_max(200.0)
        k = min(4, len(p))
        f[:, 61 : 61 + k], f[:, 65 : 65 + k] = dr(p[:, None], p[:k]), mass(p[:, None] + p[:k])
        f[:, 69] = torch.arange(len(p), device=p.device)
        f[:, 70] = pt(p) / (pt(p) + rpt * f[:, 32:35].sum(1))
        f[:, 71] = pt(p) / (pt(p) + rpt * f[:, 32:37].sum(1))
    return f, frac, nearest


def particle_cascade(event):
    """Contract rooted energy correlations in lab and rest coordinates"""
    p = event.p
    # The fitted quantiles resolve roundoff in nearly collinear energy moments
    root = p.cumsum(0)[-1]
    output = p.new_zeros((len(p), 120), dtype=torch.float32)
    output[:, :72] = objects(event, 0)[0]
    for frame in (0, 1):
        if frame and mass2(root) < 1.0e-8:
            continue
        if frame:
            q = boost(p, root)
            points = q[:, 1:] / (pt2(q) + q[:, 3].square()).sqrt().clamp_min(1.0e-15)[:, None]
            w = q[:, 0].clamp_min(0.0)
        else:
            transverse = pt2(p)

            # Evaluate the rapidity logarithm after the squared energy ratio
            def rapidity(v):
                ratio = (pt2(v) + mass2(v).clamp_min(0.0)) / (v[..., 0] + v[..., 3].abs()).square().clamp_min(
                    1.0e-300
                )
                return torch.where(v[..., 3] > 0.0, -0.5, 0.5) * ratio.clamp_min(1.0e-300).log()

            offset = torch.stack((rapidity(p) - rapidity(root), displacement(root, p)[:, 1]), 1)
            points = offset / (mass(root) / pt(root).clamp_min(1.0e-12)).clamp_min(0.075)
            w = transverse.sqrt()
        w = w / w.cumsum(0)[-1].clamp_min(1.0e-15)
        d2 = (points[:, None] - points).square().sum(2)
        for k, r in enumerate((0.18, 0.5, 1.0, 2.0)):
            kernel = w * (1 - d2 / r**2).clamp_min(0)
            first = kernel.cumsum(1)[:, -1]
            norm = first.clamp_min(1.0e-16)
            q = kernel / norm[:, None]
            second = (kernel * first).cumsum(1)[:, -1]
            mean = second / norm
            projected = q[:, :, None] * points
            center = projected.cumsum(1)[:, -1]
            moment = (projected[:, :, :, None] * points[:, None, :]).cumsum(1)[:, -1]
            cov = moment - center[:, :, None] * center[:, None, :]
            diagonal = cov.diagonal(dim1=1, dim2=2)
            trace = diagonal.sum(1)
            trace2 = diagonal.square().sum(1) + 2 * cov.tril(-1).square().sum((1, 2))
            fork = (q * first * first).cumsum(1)[:, -1]
            path = (q * second).cumsum(1)[:, -1]
            start = 72 + frame * 24 + 6 * k
            output[:, start : start + 6] = torch.stack(
                (
                    first,
                    mean,
                    path,
                    (fork - mean.square()).clamp_min(0),
                    2 * trace.clamp_min(0),
                    torch.where(trace > 1.0e-12, (trace2 / trace.square()).clamp(0, 1), 0),
                ),
                1,
            )
    return output


def pool(event, dictionaries):
    """Pool the learned jet partitions and tree scores by subjet"""
    result = event.p.new_zeros((2, 248))
    for family, table in enumerate(dictionaries):
        x, w, g = objects(event, family)
        if not len(x):
            continue
        tree = torch.arange(table.size(0), device=x.device)
        node = torch.zeros((len(x), len(tree)), dtype=torch.long, device=x.device)
        for _ in range(5):
            row = table[tree, node]
            feature = row[:, :, 0].long()
            take = x.gather(1, feature.clamp_min(0)) < row[:, :, 3]
            node = torch.where(feature >= 0, torch.where(take, row[:, :, 1], row[:, :, 2]).long(), node)
        row = table[tree, node]
        score = row[:, :, 3].double().sum(1)
        rank = row[:, :, 4].long()
        keep = rank >= 0
        result[family, :160].scatter_add_(0, rank[keep], torch.ones_like(rank[keep], dtype=w.dtype))
        keep &= rank < 64
        result[family, 160:224].scatter_add_(0, rank[keep], w[:, None].expand_as(rank)[keep])
        for j in range(4):
            take = torch.ones_like(g, dtype=torch.bool) if j == 0 else g == j - 1
            if take.any():
                s = score[take]
                result[family, 224 + 6 * j : 230 + 6 * j] = packed(
                    x.double(),
                    s.sum(),
                    s.mean(),
                    s.square().mean(),
                    s.max(),
                    s.min(),
                    (s * w[take]).sum() / w[take].sum().clamp_min(1.0e-10),
                )
    return result.flatten()
