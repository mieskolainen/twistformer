# Harmonic, projected and spectral jet observables
#
# mikael.mieskolainen@cern.ch (2026)

from math import cos, exp, lgamma, log, pi, sin, sqrt

import torch
import torch.nn.functional as F

from .kinematics import Sequence, boost, cosine, displacement, dr, mass, mass2, packed, pt, pt2, signed_mass, unit


def _harmonics(v, share):
    """Integrate real spherical harmonics with four log-energy weights"""
    x = v[:, 2].clamp(-1.0, 1.0)
    radius = (1 - x.square()).clamp_min(0.0).sqrt()
    t = 2 * (1000 * share).log1p() / log(1001.0) - 1
    weights = share.sqrt() / share.sqrt().sum()
    z = torch.stack((weights, weights * t, weights * (2 * t * t - 1), weights * t * (4 * t * t - 3)))
    phi = v[:, 1].atan2(v[:, 0])
    diag = torch.ones_like(x)
    result = v.new_zeros((4, 169))
    for m in range(13):
        cs, sn = (m * phi).cos(), (m * phi).sin()
        prev, curr = torch.zeros_like(x), diag
        for l in range(m, 13):
            j = l * l + (2 * m - 1 if m else 0)
            value = sqrt((2 * l + 1) * (2 if m else 1) * exp(lgamma(l - m + 1) - lgamma(l + m + 1))) * curr
            result[:, j] = z @ (value * cs)
            if m:
                result[:, j + 1] = z @ (value * sn)
            prev, curr = curr, ((2 * l + 1) * x * curr - (l + m) * prev) / (l - m + 1)
        diag = diag * (-(2 * m + 1)) * radius
    power = [
        (result[:, l * l : (l + 1) ** 2] * result[0, l * l : (l + 1) ** 2]).sum(1) / (2 * l + 1) for l in range(10)
    ]
    return torch.cat([result[k, : (13 - k) ** 2] for k in range(4)] + power)


def multipole(event):
    """Measure lab and rest-frame angular harmonics through degree twelve"""
    p, root = event.p, event.root
    weights = pt(p)
    z = weights / weights.sum()
    eta = (p[:, 3] / weights).asinh()
    phi = displacement(root, p)[:, 1]
    u, v = eta - z @ eta, phi - z @ phi
    alpha = 0.5 * torch.atan2(2 * (weights * u * v).sum(), (weights * (u * u - v * v)).sum())
    scale = (mass(root) / pt(root).clamp_min(1.0e-10)).clamp_min(0.08)
    x, y = (alpha.cos() * u + alpha.sin() * v) / scale, (-alpha.sin() * u + alpha.cos() * v) / scale
    x = torch.where((weights * x.pow(3)).sum() >= 0, x, -x)
    y = torch.where((weights * y.pow(3)).sum() >= 0, y, -y)
    r = x * x + y * y
    lab = _harmonics(torch.stack((2 * x, 2 * y, 1 - r), 1) / (1 + r[:, None]), z)
    if mass(root) < 1.0e-4:
        return torch.cat((lab, torch.zeros_like(lab)))
    q = event.rest
    w = q[:, 0].clamp_min(0.0)
    v = unit(q[:, 1:])
    e1 = v[w.argmax()]
    cross = torch.linalg.cross(e1.expand_as(v), v)
    e3 = unit(cross[(cross.square().sum(1) * w.square()).argmax()])
    e2 = unit(torch.linalg.cross(e3, e1))
    e3 = torch.where(e3 @ root[1:] < 0, -e3, e3)
    return torch.cat((lab, _harmonics(v @ torch.stack((e2, e3, e1)).T, w / w.sum())))


def _quantiles(points, weight, dirs):
    """Integrate three empirical energy measures along each direction"""
    x, ids = (dirs @ points.T).sort(dim=1, stable=True)
    w = torch.stack((weight, weight.sqrt(), torch.ones_like(weight)))
    cw = w[:, ids].cumsum(2)
    levels = points.new_tensor([0.02, 0.07, 0.15, 0.25, 0.37, 0.5, 0.63, 0.75, 0.85, 0.93, 0.98])
    where = torch.searchsorted(
        cw.contiguous(), (w.sum(1)[:, None, None] * levels - 1.0e-12).expand(-1, len(dirs), -1).contiguous()
    ).clamp_max(len(points) - 1)
    return x.expand(3, -1, -1).gather(2, where).transpose(0, 1).flatten()


def radon(event):
    """Measure projected quantiles in canonical jet coordinates"""
    p, root = event.p, event.kt.axes(1)[0]
    axes = event.kt.axes(3)
    dy = displacement(axes[0], axes[1]) if len(axes) > 1 else displacement(root, axes[0])
    cs, sn = dy / dy.norm().clamp_min(1.0e-12)
    sign = 1.0
    if len(axes) > 2:
        u, v = displacement(axes[0], axes[2])
        sign = 1.0 if -sn * u + cs * v >= 0 else -1.0
    u, v = displacement(root, p).T
    points = torch.stack((cs * u + sn * v, sign * (-sn * u + cs * v)), 1) / (
        mass(root) / pt(root).clamp_min(1.0e-12)
    ).clamp_min(0.075)
    lab = _quantiles(points, pt(p), p.new_tensor([(cos(j * pi / 8), sin(j * pi / 8)) for j in range(8)]))
    if mass2(root) < 1.0e-8:
        return torch.cat((lab, p.new_zeros(330)))
    dirs = unit(event.ee.axes(3, True)[:, 1:])
    e1 = dirs[0]
    ref = dirs[1] if len(dirs) > 1 else unit(root[1:])
    e3 = torch.linalg.cross(e1, ref)
    if e3.square().sum() < 1.0e-8:
        e3 = torch.linalg.cross(e1, p.new_tensor([0, 0, 1]))
    e3 = unit(e3)
    e2 = unit(torch.linalg.cross(e3, e1))
    e3 = torch.where(e3 @ root[1:] < 0, -e3, e3)
    points = unit(event.rest[:, 1:]) @ torch.stack((e1, e2, e3)).T
    dirs = unit(
        p.new_tensor(
            [
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 1, 0),
                (1, -1, 0),
                (1, 0, 1),
                (1, 0, -1),
                (0, 1, 1),
                (1, 1, 1),
                (1, -1, 1),
            ]
        )
    )
    return torch.cat((lab, _quantiles(points, event.rest[:, 0].clamp_min(1.0e-12), dirs)))


def _resolution(p, maximum, seq=None):
    """Measure angular widths around one through maximum exclusive axes"""
    out = p.new_zeros((maximum, 7))
    if len(p) < 2:
        return out.flatten()
    seq = seq or Sequence(p)
    w = pt(p) / pt(p).sum()
    for n in range(1, min(len(p), maximum + 1)):
        d = dr(p[:, None], seq.axes(n)).amin(1)
        out[n - 1, :3] = torch.stack((w @ d.sqrt(), w @ d, w @ d.square()))
        out[n - 1, 6] = d.mean()
    out[1:, 3:6] = torch.where(out[:-1, :3] >= 1.0e-12, out[1:, :3] / out[:-1, :3].clamp_min(1.0e-12), 0.0)
    return out.flatten()


def resolution(event):
    """Measure subjet resolution in the jet and reconstructed decay prongs"""
    p = event.p
    g = event.groups[0]
    ax = torch.stack([p[j].sum(0) for j in g])
    b = int((mass(ax.roll(1, 0) + ax.roll(-1, 0)) - 80.4).abs().argmin())
    i, j = (b + 1) % 3, (b + 2) % 3
    if pt2(ax[j]) > pt2(ax[i]):
        i, j = j, i
    return torch.cat(
        (
            _resolution(p, 24, event.kt),
            _resolution(p[g[i] + g[j]], 10),
            _resolution(p[g[b]], 8),
            _resolution(p[g[i]], 5),
            _resolution(p[g[j]], 5),
        )
    )


def morphology(event):
    """Measure topology of the radiation image and angular spectra of subjets"""
    p, root = event.p, event.root
    weight = pt(p)
    z = weight / weight.sum()
    delta = displacement(root, p)
    output = []
    for cell in (0.045, 0.085):
        ix = (delta / cell).floor().long() + 32
        keep = ((ix >= 0) & (ix < 64)).all(1)
        energy = p.new_zeros(4096).scatter_add_(0, ix[keep, 0] * 64 + ix[keep, 1], weight[keep]).view(64, 64)
        grid = energy > p.new_tensor([0.25, 1.0, 2.0, 4.0, 8.0])[:, None, None]
        rows = []
        for r in range(8):
            if r not in (4, 6):
                area = grid.sum((1, 2))
                overlap = grid[:, 1:] & grid[:, :-1]
                edges = overlap.sum((1, 2)) + (grid[:, :, 1:] & grid[:, :, :-1]).sum((1, 2))
                faces = (overlap[:, :, 1:] & overlap[:, :, :-1]).sum((1, 2))
                rows.append(torch.stack((area, 4 * area - 2 * edges, area - edges + faces), 1))
            padded = F.pad(grid, (1, 1, 1, 1))
            grid = grid | padded[:, :-2, 1:-1] | padded[:, 2:, 1:-1] | padded[:, 1:-1, :-2] | padded[:, 1:-1, 2:]
        output.append(torch.stack(rows, 1).flatten())
    i, j = torch.tril_indices(len(p), len(p), device=p.device)
    bin_id = torch.where(i == j, 0, (1 + (dr(p[i], p[j]) / 0.05).long()).clamp_max(21))
    for r in (0.12, 0.24):
        seq = Sequence(p, "kt", r)
        order = sorted(seq.inclusive, key=lambda k: float(pt2(seq.vectors[k])), reverse=True)
        category = torch.empty(len(p), dtype=torch.long, device=p.device)
        fields = p.new_zeros((5, 9))
        for rank, node in enumerate(order):
            axis = seq.vectors[node]
            members = seq.members[node]
            parts = p[members]
            category[members] = 0 if rank == 0 else (1 if pt(axis) > 0.05 * weight.sum() else 2)
            if rank < 5:
                q = pt(parts) / pt(axis)
                fields[rank] = packed(
                    p,
                    pt(axis) / weight.sum(),
                    signed_mass(axis),
                    len(parts),
                    q @ dr(parts, axis),
                    q.square().sum(),
                    *((pt(parts) > cut).sum() for cut in (1, 2, 4, 8)),
                )
        high, low = torch.maximum(category[i], category[j]), torch.minimum(category[i], category[j])
        code = (high * (high + 1) // 2 + low) * 22 + bin_id
        value = z[i] * z[j]
        f = torch.where(i == j, 1.0, 2.0).to(p.dtype)
        hist = p.new_zeros((132, 3)).index_add_(0, code, torch.stack((f * value, f * value.sqrt(), f), 1))
        output.extend((hist, fields))
    return packed(p, *output)


def diffusion(event):
    """Measure normalized diffusion eigenvalues and spectral prong assignments"""
    p = event.p
    if len(p) < 3 or mass2(event.root) < 1.0e-8:
        return p.new_zeros(640)
    out = []
    for rest in (False, True):
        axes = (event.ee if rest else event.kt).axes(14, rest)
        lab = boost(axes, event.root, True) if rest else axes
        energy = axes[:, 0] if rest else pt(axes)
        d2 = 2 * (1 - cosine(axes[:, None], axes)) if rest else dr(axes[:, None], axes).square()
        for r in (0.18, 0.4, 0.78, 1.3) if rest else (0.06, 0.14, 0.30, 0.60):
            kernel = (-d2 / (2 * r * r)).exp() + 0.001
            for weight in (energy, energy.sqrt()):
                w = weight / weight.sum().clamp_min(0.001)
                affinity = w[:, None] * kernel * w
                degree = affinity.sum(1).clamp_min(1.0e-18)
                inv = degree.rsqrt()
                vals, vec = torch.linalg.eigh(inv[:, None] * affinity * inv)
                ev = vals.flip(0).clamp(0.0, 1.0)
                padded = F.pad(ev, (0, max(0, 8 - len(ev))))
                trace = ev.sum()
                rel = ev[ev > 1.0e-12] / trace
                out += [
                    padded[1:8],
                    padded[1:5] - padded[2:6],
                    *[ev.pow(k).sum() - 1 for k in (2, 3, 5, 8, 12, 20)],
                    trace,
                    -(rel * rel.log()).sum(),
                    degree.max() / degree.sum(),
                    affinity.diag().sum() / affinity.sum(),
                    degree.square().sum() / degree.sum().square(),
                ]
                coords = inv[:, None] * vec.flip(1)[:, 1:4] * ev[1:4]
                center = [coords[w.argmax()]]
                near = w.new_full(w.shape, torch.inf)
                for _ in range(2):
                    near = torch.minimum(near, (coords - center[-1]).square().sum(1))
                    center.append(coords[(near * w.sqrt()).argmax()])
                center = torch.stack(center)
                for _ in range(8):
                    ids = (coords[:, None] - center).square().sum(2).argmin(1)
                    labels = F.one_hot(ids, 3).T.double()
                    weighted = labels * w
                    ws = weighted.sum(1, keepdim=True)
                    center = torch.where(ws > 0, (weighted @ coords) / ws.clamp_min(1.0e-15), center)
                groups = labels @ lab
                rank = pt2(groups).argsort(descending=True, stable=True)
                a, b, c = groups[rank]
                norm = pt(a + b + c).clamp_min(0.01)
                out += [torch.stack((pt(groups) / norm, mass(groups), labels.sum(1)), 1)[rank]]
                pairs = signed_mass(torch.stack((a + b, a + c, b + c)))
                out += [
                    pairs,
                    (pairs - 80.4).abs().min(),
                    pairs.min(),
                    pairs.max(),
                    torch.where(pt(b) > 0, dr(a, b), 0.0),
                    torch.where(pt(c) > 0, dr(a, c), 0.0),
                    torch.where(pt(c) > 0, dr(b, c), 0.0),
                ]
    return packed(p, *out)
