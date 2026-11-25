# Angular flow and energy correlations in the lab and jet rest frames
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import torch

from .jet_clustering import angles, angular_difference, distance2, mass, mass2, pt2
from .jet_history import axis_fields


SUBJET_COMBINATIONS = {n: torch.tensor(list(combinations(range(12), n))) for n in range(2, 6)}


def energy_correlations(axes, scalar_pt):
    """
    Contract two- through five-subjet energy correlators

    Args:
        axes:       Up to twelve exclusive kT axes with zero padding
        scalar_pt:  Scalar transverse momentum of the constituents
    Returns:
        rows:  Energy correlations and their ratios for three angular exponents
    """
    weights = pt2(axes).sqrt() / scalar_pt[:, None]
    coords = angles(axes)
    angle = distance2(coords[:, :, None], coords[:, None, :]).sqrt()
    by_order = []
    for n in range(2, 6):
        indices = SUBJET_COMBINATIONS[n].to(axes.device)
        i, j = torch.triu_indices(n, n, 1, device=axes.device)
        edges = angle[:, indices[:, i], indices[:, j]]
        by_order.append((weights[:, indices].prod(-1), edges.prod(-1), edges.amin(-1)))
    result = []
    for beta in (.5, 1., 2.):
        e2, e3, e4, e5 = [(w * product.pow(beta)).sum(1) for w, product, _ in by_order]
        m3, m4, m5 = [(w * shortest.pow(beta)).sum(1) for w, _, shortest in by_order[1:]]
        path = (by_order[2][0] * by_order[2][2].pow(2 * beta)).sum(1)
        result.extend((e2, e3, e4, e5, m3, m4, m5, path, e3 / e2.pow(3).clamp_min(1.e-15),
                       e4 * e2 / e3.square().clamp_min(1.e-15), e5 * e3 / e4.square().clamp_min(1.e-15),
                       m4 / m3.clamp_min(1.e-14), m5 / m4.clamp_min(1.e-14)))
    return torch.stack(result, 1)


def rest_moments(axes, root):
    """
    Measure Fox Wolfram moments and the four leading rest energies

    Args:
        axes:  Exclusive kT subjet momenta in the lab
        root:  Jet four-momentum
    Returns:
        rows:  Rest-frame angular and energy observables
    """
    m = mass(root).clamp_min(1.e-12)
    direction = root[:, None, 1:]
    projection = (axes[:, :, 1:] * direction).sum(2)
    energy = (root[:, 0:1] * axes[:, :, 0] - projection) / m[:, None]
    factor = (projection / (root[:, 0:1] + m[:, None]).clamp_min(1.e-12) - axes[:, :, 0]) / m[:, None]
    vector = axes[:, :, 1:] + factor[:, :, None] * direction
    norm = vector.norm(dim=2)
    cosine = ((vector @ vector.transpose(1, 2)) / (norm[:, :, None] * norm[:, None, :]).clamp_min(1.e-12)).clamp(-1., 1.)
    share = energy / m[:, None].clamp_min(.001)
    weights = share[:, :, None] * share[:, None, :]
    prev, curr = torch.ones_like(cosine), cosine
    moments = []
    for h in range(2, 11):
        prev, curr = curr, ((2 * h - 1) * cosine * curr - (h - 1) * prev) / h
        moments.append((weights * curr).sum((1, 2)))
    rows = torch.cat((torch.stack(moments, 1), share.topk(4, dim=1).values), 1)
    return torch.where(mass2(root)[:, None] >= 1.e-8, rows, 0.)


def flow_rows(p, valid, tree):
    """
    Measure constituent radiation, exclusive-subjet correlations and prong profiles

    Args:
        p:      Energy-first particle four-momenta
        valid:  Physical constituent mask
        tree:   Exclusive kT partitions
    Returns:
        rows:  Angular flow observables in flow_names order
    """
    root = tree.axes[1][:, 0]
    pt = pt2(p).sqrt()
    scalar_pt = pt.sum(1).clamp_min(1.e-300)
    z = pt / scalar_pt[:, None]
    result = [torch.stack((angles(root)[:, 0].abs(), pt2(root).sqrt(), mass(root).clamp_min(.001), valid.sum(1)), 1)]
    for cut in (.5, 1., 2., 4., 8., 16.):
        result.append(torch.stack(((pt > cut).sum(1), (z * (pt > cut)).sum(1)), 1))
    coord = angles(p)
    d = distance2(coord[:, :, None], coord[:, None, :]).sqrt()
    bins = (d / .05).long().clamp(0, 13)
    pair = (valid[:, :, None] & valid[:, None, :]).triu(1)
    spectra = []
    for weight in (z[:, :, None] * z[:, None, :], z[:, :, None] + z[:, None, :], torch.ones_like(d)):
        spectrum = p.new_zeros((p.size(0), 14))
        spectrum.scatter_add_(1, bins.flatten(1), (pair * weight).flatten(1))
        spectra.append(spectrum)
    result.append(torch.stack(spectra, 2).flatten(1))
    for radius in (.01, .025, .05, .075, .1, .15, .25, .4, .65):
        inside = (d < radius) & valid[:, None, :]
        count = inside.sum(2) - 1
        share = (inside * z[:, None, :]).sum(2)
        isolated = (count == 0) & valid
        result.append(torch.stack((isolated.sum(1), (isolated * z).sum(1),
                                   (valid / (1. + count.to(p.dtype)).clamp_min(1.)).sum(1),
                                   share.masked_fill(~valid, 0.).amax(1), (z * share.square()).sum(1),
                                   (z * share.pow(3)).sum(1), (z * count).sum(1), (z * count.square()).sum(1)), 1))
    result.append(energy_correlations(tree.axes[12], scalar_pt))
    for n in (4, 8, 16):
        result.append(rest_moments(tree.axes[n], root))
    axes, member, active, delta, radius = axis_fields(p, tree, 3, valid)
    share = pt[:, None] * member
    share = share / share.sum(2, keepdim=True).clamp_min(1.e-300)
    fields = []
    for weight in (share, member):
        for cut in (.025, .05, .075, .1, .15, .25):
            fields.append((weight * (radius < cut)).sum(2))
    for weight in (share, member):
        for cut in (1., 2., 4., 8., 16.):
            fields.append((weight * (pt[:, None] > cut)).sum(2))
    pull = ((share * radius)[:, :, :, None] * delta).sum(2)
    fields.extend((share.sqrt().sum(2), share.square().sum(2), share.pow(3).sum(2),
                   (share * radius.sqrt()).sum(2), (share * radius.square()).sum(2),
                   pull[:, :, 0], pull[:, :, 1], share.amax(2)))
    coords = angles(axes)
    for offset in (1, 2):
        direction = angular_difference(coords, coords.roll(-offset, 1))
        projection = (pull * direction).sum(2) / direction.norm(dim=2).clamp_min(.00001)
        fields.append(torch.where(active.roll(-offset, 1), projection, 0.))
    result.append(torch.stack(fields, 2).flatten(1))
    return torch.where(valid.any(1)[:, None], torch.cat(result, 1), 0.)
