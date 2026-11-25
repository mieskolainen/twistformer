# Jet partitions in the laboratory and Durham rest frames
#
# mikael.mieskolainen@cern.ch (2026)

from functools import cached_property
from itertools import combinations

import torch

from ..jet_clustering import angles, angular_difference, distance2, mass, mass2, pt2


def pt(p):
    """Return transverse momentum"""
    return pt2(p).sqrt()


def dot(p, q):
    """Contract two four-momenta in the Minkowski metric"""
    return p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(-1)


def signed_mass(p):
    """Return the signed square root of invariant mass squared"""
    m2 = mass2(p)
    return m2.sign() * m2.abs().sqrt()


def unit(v):
    """Normalize spatial directions"""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1.0e-15)


def cosine(p, q):
    """Measure the spatial opening angle cosine"""
    v, w = p[..., 1:], q[..., 1:]
    return ((v * w).sum(-1) / (v.norm(dim=-1) * w.norm(dim=-1)).clamp_min(1.0e-15)).clamp(-1.0, 1.0)


def boost(p, reference, inverse=False):
    """Boost measured momenta into or out of the reference rest frame"""
    m = mass(reference).clamp_min(1.0e-15)
    e = reference[..., 0]
    vec = reference[..., 1:]
    projection = p[..., 1] * vec[..., 0] + p[..., 2] * vec[..., 1] + p[..., 3] * vec[..., 2]
    sign = 1.0 if inverse else -1.0
    energy = (sign * projection + e * p[..., 0]) / m
    factor = (p[..., 0] + energy) / (e + m).clamp_min(1.0e-15)
    return torch.cat((energy.unsqueeze(-1), p[..., 1:] + sign * factor.unsqueeze(-1) * vec), -1)


def displacement(p, q):
    """Measure the signed angular displacement from p to q"""
    return angular_difference(angles(p), angles(q))


def dr(p, q):
    """Measure the rapidity and azimuth distance"""
    return displacement(p, q).norm(dim=-1)


def packed(reference, *values):
    """Pack scalar and vector observables in feature order"""
    return torch.cat(
        [torch.as_tensor(v, device=reference.device, dtype=reference.dtype).reshape(-1) for v in values]
    )


class Sequence:
    """Exclusive and inclusive recombination history for one measured jet"""

    def __init__(self, p, algorithm="kt", radius=1000.0):
        self.p = p
        n = len(p)
        self.children = {}
        self.members = {i: [i] for i in range(n)}
        self.states = {}
        self.merges = {}
        self.inclusive = []
        self.vectors = p.new_zeros((max(1, 2 * n), 4))
        self.vectors[:n] = p
        live = list(range(n))
        coords = angles(p)
        scale = self.scale(p, algorithm)
        costs = self.pair_cost(p[:, None], p, coords[:, None], coords, scale[:, None], scale, algorithm, radius)
        for step in range(n):
            size = len(live)
            self.states[size] = live.copy()
            costs.diagonal().copy_(torch.full_like(scale, torch.inf) if algorithm == "ee" else scale)
            i, j = divmod(int(costs.argmin()), size)
            a, b = live[i], live[j]
            self.merges[size - 1] = costs[i, j].clone()
            if i == j:
                self.inclusive.append(a)
                live[i] = live[-1]
            else:
                parent = n + step
                self.children[parent] = tuple(sorted((a, b)))
                self.members[parent] = self.members[min(a, b)] + self.members[max(a, b)]
                self.vectors[parent] = self.vectors[a] + self.vectors[b]
                i, j = sorted((i, j))
                live[i] = parent
                live[j] = live[-1]
            live.pop()
            # Move the last active slot into the removed daughter slot
            removed = i if a == b else j
            costs[removed] = costs[-1].clone()
            costs[:, removed] = costs[:, -1].clone()
            costs = costs[:-1, :-1]
            scale[removed] = scale[-1].clone()
            coords[removed] = coords[-1].clone()
            scale, coords = scale[:-1], coords[:-1]
            if a != b:
                vec = self.vectors[parent]
                scale[i] = self.scale(vec, algorithm)
                coords[i] = angles(vec)
                distance = self.pair_cost(
                    vec,
                    self.vectors[live] if algorithm == "ee" else vec,
                    coords[i],
                    coords,
                    scale[i],
                    scale,
                    algorithm,
                    radius,
                )
                costs[i] = distance
                costs[:, i] = distance

    @staticmethod
    def scale(momentum, algorithm):
        """Return the energy or transverse scale of a clustering metric"""
        if algorithm == "ee":
            return momentum[..., 0].square()
        scale = pt2(momentum)
        if algorithm == "ca":
            return torch.ones_like(scale)
        return scale.clamp_min(1.0e-300).reciprocal() if algorithm == "antikt" else scale

    @staticmethod
    def pair_cost(a, b, xy_a, xy_b, scale_a, scale_b, algorithm, radius):
        """Measure the pair recombination distance"""
        scale = torch.minimum(scale_a, scale_b)
        if algorithm == "ee":
            return 2 * (1.0 - cosine(a, b)) * scale
        return distance2(xy_a, xy_b) * scale / radius**2

    def indices(self, n, energy=False):
        """Rank exclusive subjets by transverse momentum or rest energy"""
        ids = self.states[min(n, len(self.p))]
        v = self.vectors[ids]
        rank = (v[:, 0] if energy else pt2(v)).argsort(descending=True, stable=True)
        return [ids[i] for i in rank.tolist()]

    def axes(self, n, energy=False):
        """Return sorted exclusive subjet four-momenta"""
        return self.vectors[self.indices(n, energy)]

    def hard_soft(self, node):
        """Return children with the harder transverse momentum first"""
        a, b = self.children[node]
        return (a, b) if pt2(self.vectors[a]) >= pt2(self.vectors[b]) else (b, a)


class Event:
    """Measured constituents with shared lab and rest-frame partitions"""

    def __init__(self, p):
        self.p = p
        self.root = p.sum(0)

    @cached_property
    def kt(self):
        return Sequence(self.p)

    @cached_property
    def ca(self):
        return Sequence(self.p, "ca")

    @cached_property
    def rest(self):
        return boost(self.p, self.root) if mass2(self.root) > 1.0e-8 else self.p

    @cached_property
    def ee(self):
        return Sequence(self.rest, "ee")

    @cached_property
    def groups(self):
        """Build the four measured three-prong partitions"""
        result = []
        for seq in (self.kt, self.ee):
            groups = [seq.members[j] for j in seq.indices(3)]
            groups += [[] for _ in range(3 - len(groups))]
            result.append(sorted(groups, key=lambda g: float(pt2(self.p[g].sum(0))), reverse=True))
        seeds = self.kt.axes(6)
        choices = [c for c in combinations(range(len(seeds)), 3) if c[0] < 3 and c[1] < 5]
        for radius in (0.18, 0.34):
            if not choices:
                result.append([[], [], []])
                continue
            ids = torch.tensor(choices, device=self.p.device)
            d = dr(self.p[:, None, :], seeds[None, :, :])[:, ids].square()
            nearest, which = d.min(-1)
            keep = (nearest < radius**2) | (pt(self.p)[:, None] > 0.095 * pt(self.root))
            membership = (which[:, :, None] == torch.arange(3, device=self.p.device)) & keep[:, :, None]
            sub = torch.einsum("ncg,nd->cgd", membership.double(), self.p)
            missing = ((~keep) * pt(self.p)[:, None]).sum(0) / pt(self.root)
            dm = (mass(sub + sub.roll(1, 1)) - 80.4).abs().amin(1)
            cost = (
                ((mass(sub.sum(1)) - 172.5) / 24.0) ** 2
                + (dm / 14.0) ** 2
                + 6 * missing
                + mass(sub).square().sum(1) / 1200.0
            )
            cost = cost.masked_fill(pt(sub).amin(1) < 2.0, torch.inf)
            best = int(cost.argmin())
            groups = []
            for j in pt2(sub[best]).argsort(descending=True, stable=True).tolist():
                groups.append(torch.where(membership[:, best, j])[0].tolist() if cost[best].isfinite() else [])
            result.append(groups)
        return result
