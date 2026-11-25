# Exclusive longitudinally invariant jet clustering in rapidity and azimuth
#
# mikael.mieskolainen@cern.ch (2026)

from typing import NamedTuple

import torch
from torch import Tensor


def pt2(p: Tensor) -> Tensor:
    """Return squared transverse momentum"""
    return p[..., 1].square() + p[..., 2].square()


def mass2(p: Tensor) -> Tensor:
    """Return the invariant mass squared in energy-first coordinates"""
    return (p[..., 0] + p[..., 3]) * (p[..., 0] - p[..., 3]) - pt2(p)


def mass(p: Tensor) -> Tensor:
    """Return the timelike mass"""
    return mass2(p).clamp_min(0.).sqrt()


def angles(p: Tensor) -> Tensor:
    """Return rapidity and azimuth including the lightlike beam limit"""
    z = p[..., 3]
    scale = p[..., 0] + z.abs()
    transverse = pt2(p)
    y = .5 * ((transverse + mass2(p).clamp_min(0.)).clamp_min(1.e-300).log()
              - 2. * scale.clamp_min(1.e-300).log())
    y = torch.where(z > 0., -y, y)
    beam = (transverse == 0.) & (p[..., 0] == z.abs())
    y = torch.where(beam, torch.where(z >= 0., 1.e5 + z.abs(), -1.e5 - z.abs()), y)
    phi = torch.atan2(p[..., 2], p[..., 1])
    return torch.stack((y, torch.where(phi < 0., phi + 2. * torch.pi, phi)), -1)


def angular_difference(a: Tensor, b: Tensor) -> Tensor:
    """Measure the signed displacement from a to b"""
    dy = b[..., 0] - a[..., 0]
    dphi = b[..., 1] - a[..., 1]
    dphi = torch.where(dphi > torch.pi, dphi - 2. * torch.pi, dphi)
    dphi = torch.where(dphi < -torch.pi, dphi + 2. * torch.pi, dphi)
    return torch.stack((dy, dphi), -1)


def distance2(a: Tensor, b: Tensor) -> Tensor:
    """Return the squared rapidity and azimuth distance"""
    return angular_difference(a, b).square().sum(-1)


class JetTree(NamedTuple):
    axes: dict[int, Tensor]
    owners: dict[int, Tensor]
    splits: Tensor
    groomed: Tensor


def splitting_history(vectors: Tensor, children: Tensor, sizes: Tensor, real: Tensor) -> tuple[Tensor, Tensor]:
    """
    Measure all splittings and groom the hardest branch for four momentum cuts

    Args:
        vectors:   Particle and recombined four-momenta in birth order
        children:  Two daughter node indices per recombination
        sizes:     Constituent count of each node
        real:      Physical recombination mask
    Returns:
        rows:  Splitting kinematics and groomed jet observables
    """
    event = torch.arange(vectors.size(0), device=vectors.device)[:, None]
    left, right = children.unbind(2)
    transverse = pt2(vectors)
    left_hard = transverse[event, left] >= transverse[event, right]
    hard = torch.where(left_hard, left, right)
    soft = torch.where(left_hard, right, left)
    coords = angles(vectors)
    delta = angular_difference(coords[event, hard], coords[event, soft])
    angle = delta.norm(p=2, dim=2)
    soft_pt = transverse[event, soft].sqrt()
    total = transverse[event, hard].sqrt() + soft_pt
    width = children.size(1) + 1
    parent_mass = mass(vectors[:, width:])
    parent_pt = transverse[:, width:].sqrt()
    fields = torch.stack((soft_pt * angle, angle, soft_pt / total.clamp_min(1.e-300),
                          parent_mass, mass(vectors[event, hard]), mass(vectors[event, soft]), parent_pt,
                          delta[:, :, 1], delta[:, :, 0], sizes[event, soft].to(vectors.dtype)), 2)
    splits = torch.where(real.unsqueeze(2), fields, 0.)

    # Double the ancestor distance until each hard daughter reaches the end of its hard branch
    nodes = torch.arange(vectors.size(1), device=vectors.device).expand_as(sizes)
    parents = nodes[:, width:]
    ancestor = nodes.clone()
    ancestor.scatter_reduce_(1, hard, torch.where(real, parents, 0), reduce="amax")
    reach = 1
    while reach < width:
        ancestor = ancestor.gather(1, ancestor)
        reach *= 2
    path = real & (ancestor[:, width:] == vectors.size(1) - 1)
    cuts = torch.tensor([.03, .08, .15, .25], dtype=vectors.dtype, device=vectors.device)
    passes = path.unsqueeze(2) & (soft_pt.unsqueeze(2) > total.unsqueeze(2) * cuts)
    last = torch.where(passes, parents.unsqueeze(2) - width, -1).amax(1)
    selected = torch.stack((parent_mass, angle, parent_pt), 2)[event, last.clamp_min(0)]
    groomed = torch.cat((torch.where(last.unsqueeze(2) >= 0, selected, 0.),
                         passes.sum(1).to(vectors.dtype).unsqueeze(2)), 2)
    return splits, groomed


def cluster(p: Tensor, valid: Tensor, kt: bool, history: bool) -> JetTree:
    """
    Merge the closest pair and record exclusive subjet partitions

    Args:
        p:        Batched double four-momenta with physical constituents first
        valid:    Physical particle mask with at least 16 available slots
        kt:       Include the softer squared transverse momentum in the metric
        history:  Record splitting and grooming observables
    Returns:
        tree:  Subjet momenta, constituent assignments and splitting history
    """
    batch, width = valid.shape
    event = torch.arange(batch, device=p.device)
    slot = torch.arange(width, device=p.device)
    count = valid.sum(1)
    coords = angles(p)
    transverse = pt2(p)
    costs = distance2(coords.unsqueeze(2), coords.unsqueeze(1))
    if kt:
        costs *= torch.minimum(transverse.unsqueeze(2), transverse.unsqueeze(1))
    costs.masked_fill_(~(valid.unsqueeze(2) & valid.unsqueeze(1)), float('inf'))
    costs.diagonal(dim1=1, dim2=2).fill_(float('inf'))
    # Each node stores momentum, angular coordinates, transverse scale and multiplicity
    nodes = p.new_empty((batch, 2 * width - 1, 8))
    nodes[:, :width] = torch.cat((p, coords, transverse.unsqueeze(2), valid.unsqueeze(2).to(p.dtype)), 2)
    live = slot.expand(batch, -1).clone()
    children = torch.empty((batch, width - 1, 2), device=p.device, dtype=torch.long)
    real_steps = count[:, None] >= width - slot[:-1]
    snapshots: dict[int, Tensor] = {}
    for step in range(width):
        size = width - step
        if size in [1, 2, 3, 4, 5, 6, 8, 12, 16]:
            snapshots[size] = live[:, :size].clone()
        if size == 1:
            break
        last = size - 1
        nearest, partner = costs[:, :size, :size].min(2)
        row = nearest.argmin(1)
        col = partner[event, row]
        real = real_steps[:, step]
        i = torch.where(real, torch.minimum(row, col), 0)
        j = torch.where(real, torch.maximum(row, col), last)
        a, b = live[event, i], live[event, j]
        children[:, step] = torch.stack((a, b), 1)
        left, right = nodes[event, a], nodes[event, b]
        parent = left[:, :4] + right[:, :4]
        parent_pt2 = pt2(parent)
        parent_angles = angles(parent)
        nodes[:, width + step] = torch.cat((parent, parent_angles, parent_pt2.unsqueeze(1),
                                            left[:, 7:8] + right[:, 7:8]), 1)
        live[event, j] = live[:, last].clone()
        live[event, i] = torch.where(real, width + step, a)
        costs[event, j, :size] = costs[:, last, :size].clone()
        costs[event, :size, j] = costs[:, :size, last].clone()
        active = nodes[event[:, None], live[:, :last], 4:7]
        updated = distance2(parent_angles.unsqueeze(1), active[:, :, :2])
        if kt:
            updated *= torch.minimum(parent_pt2.unsqueeze(1), active[:, :, 2])
        updated.masked_fill_((slot[:last] >= count.unsqueeze(1)) | (slot[:last] == i.unsqueeze(1)), float('inf'))
        costs[event, i, :last] = updated
        costs[event, :last, i] = updated
        costs[event, j, j] = float('inf')
    if history:
        splits, groomed = splitting_history(nodes[:, :, :4], children, nodes[:, :, 7], real_steps)
    else:
        splits = p.new_zeros((batch, width - 1, 10))
        groomed = p.new_zeros((batch, 4, 4))
    removal = torch.full((batch, 2*width), 2*width, device=p.device, dtype=torch.long)
    first, second = children.amin(2), children.amax(2)
    removal.scatter_reduce_(1, first, torch.where(real_steps, 2*slot[:-1], 2*width), reduce="amin")
    removal.scatter_reduce_(1, second, torch.where(real_steps, 2*slot[:-1]+1, 2*width), reduce="amin")
    # Recover all exclusive partitions from binary ancestors outside the merge loop
    parent_ids = torch.arange(2*width-1, device=p.device).expand(batch, -1)
    direct = parent_ids.clone()
    assigned = torch.where(real_steps, parent_ids[:, width:], 0)
    direct.scatter_reduce_(1, first, assigned, reduce="amax")
    direct.scatter_reduce_(1, second, assigned, reduce="amax")
    jumps = [direct]
    reach = 1
    while 2*reach < width:
        jumps.append(jumps[-1].gather(1, jumps[-1]))
        reach *= 2
    counts = [1, 2, 3, 4, 5, 6, 8, 12, 16]
    limit = 2*width - torch.tensor(counts, device=p.device)[:, None]
    ancestors = slot.expand(batch, len(counts), -1)
    for k in range(len(jumps)-1, -1, -1):
        candidate = jumps[k].gather(1, ancestors.flatten(1)).reshape_as(ancestors)
        ancestors = torch.where(candidate < limit, candidate, ancestors)
    axes: dict[int, Tensor] = {}
    members: dict[int, Tensor] = {}
    for k, n in enumerate(counts):
        ids = snapshots[n]
        # Resolve equal transverse momenta in recombination history order
        order = removal.gather(1, ids).argsort(1)
        ids = ids.gather(1, order)
        order = torch.argsort(nodes[:, :, 6].gather(1, ids), dim=1, descending=True, stable=True)
        ids = ids.gather(1, order)
        axes[n] = nodes[event[:, None], ids, :4].contiguous()
        rank = torch.zeros_like(direct)
        rank.scatter_(1, ids, slot[:n].expand(batch, -1))
        members[n] = torch.where(valid, rank.gather(1, ancestors[:, k]), -1)
    return JetTree(axes, members, splits, groomed)
