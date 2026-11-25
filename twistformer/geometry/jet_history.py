# Exclusive subjet masses, grooming and N-subjettiness
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import torch

from .jet_clustering import angles, angular_difference, distance2, mass, pt2


DECAY_ASSIGNMENTS = {n: torch.tensor([(i, j, k) for i, j in combinations(range(n), 2)
                                    for k in range(n) if k != i and k != j]).T
                     for n in (3, 4, 5, 6, 8)}


def axis_fields(p, tree, n, valid):
    """
    Measure constituent offsets and membership at a fixed subjet count

    Args:
        p:      Measured particle four-momenta
        tree:   Exclusive jet partitions
        n:      Requested subjet count
        valid:  Physical particle mask
    Returns:
        result:  Subjets, membership, physical axis mask and angular offsets
    """
    axes = tree.axes[n]
    index = torch.arange(n, device=p.device)
    members = tree.owners[n][:, None, :] == index[None, :, None]
    active = members.any(2)
    delta = angular_difference(angles(axes)[:, :, None], angles(p)[:, None, :])
    radius = delta.norm(dim=-1)
    return axes, members, active, delta, radius


def decay_fields(axes, active, fit, root_pt):
    """
    Select the pair and parent closest to the supplied resonance masses

    Args:
        axes:     Exclusive subjet momenta
        active:   Physical axes
        fit:      Parent and pair mass hypotheses and widths
        root_pt:  Jet transverse momentum
    Returns:
        fields:  Five kinematic observables of the best assignment
    """
    n = axes.size(1)
    i, j, k = DECAY_ASSIGNMENTS[n].to(axes.device)
    pair = axes[:, i] + axes[:, j]
    parent = pair + axes[:, k]
    pair_mass, parent_mass = mass(pair), mass(parent)
    chi = ((parent_mass - fit[0]) / fit[2]).square() + ((pair_mass - fit[1]) / fit[3]).square()
    physical = active[:, i] & active[:, j] & active[:, k]
    best = chi.masked_fill(~physical, torch.inf).argmin(1)
    event = torch.arange(len(axes), device=axes.device)
    parent_pt = pt2(parent[event, best]).sqrt()
    cross = torch.minimum(mass(axes[event, i[best]] + axes[event, k[best]]),
                          mass(axes[event, j[best]] + axes[event, k[best]]))
    rows = torch.stack((parent_mass[event, best], pair_mass[event, best],
                        pt2(pair[event, best]).sqrt() / parent_pt.clamp_min(1.e-10), parent_pt / root_pt, cross), 1)
    return torch.where(physical.any(1)[:, None], rows, 0.)


def history_rows(p, valid, tree, fit):
    """
    Project one clustering tree into its ordered jet feature columns

    Args:
        p:      Particle momenta
        valid:  Physical particle mask
        tree:   Exclusive clusters and splitting history
        fit:    Optional pair and parent mass fit
    Returns:
        rows:  Clustering observables in history_names order
    """
    root = tree.axes[1][:, 0]
    root_pt = pt2(root).sqrt().clamp_min(1.e-8)
    scale = mass(root).clamp_min(1.e-8)
    scale_kt = tree.splits[:, :, 0].masked_fill(tree.splits[:, :, 9] == 0, -torch.inf)
    order = scale_kt.argsort(dim=1, descending=True, stable=True)[:, :7]
    splits = tree.splits.gather(1, order[:, :, None].expand(-1, -1, 10))
    splits = torch.cat((splits[:, :, :6], splits[:, :, 6:7] / root_pt[:, None, None], splits[:, :, 7:]), 2)
    groomed = tree.groomed
    groomed = torch.cat((groomed[:, :, :2], groomed[:, :, 2:3] / root_pt[:, None, None], groomed[:, :, 3:]), 2)
    output = [splits.flatten(1), groomed.flatten(1)]
    previous = torch.ones_like(root_pt)
    pt = pt2(p).sqrt()
    for n in (2, 3, 4, 5, 6, 8):
        axes, members, active, _, radius = axis_fields(p, tree, n, valid)
        nearest = radius.masked_fill(~active[:, :, None], 1.e5).amin(1)
        tau = (pt * nearest).sum(1) / root_pt
        output.append(torch.stack((tau, (pt * nearest.square()).sum(1) / root_pt,
                                   (pt * nearest.sqrt()).sum(1) / root_pt, tau / previous.clamp_min(1.e-10)), 1))
        previous = tau
        if n in (2, 3, 5):
            axis_pt = pt2(axes).sqrt()
            fractions = members * pt[:, None] / axis_pt.clamp_min(1.e-12)[:, :, None]
            delta = angular_difference(angles(root)[:, None], angles(axes))
            fields = torch.stack((axis_pt / root_pt[:, None], mass(axes), members.sum(2), fractions.amax(2),
                                  fractions.square().sum(2), (fractions * radius).sum(2),
                                  delta[:, :, 0], delta[:, :, 1]), 2)
            output.append(torch.where(active[:, :, None], fields, 0.).flatten(1))
            i, j = torch.triu_indices(n, n, 1, device=p.device)
            pair_mass = mass(axes[:, i] + axes[:, j])
            angle = distance2(angles(axes[:, i]), angles(axes[:, j])).sqrt()
            pairs = torch.stack((pair_mass, pair_mass / scale[:, None], angle, axis_pt[:, j] * angle), 2)
            output.append(torch.where((active[:, i] & active[:, j])[:, :, None], pairs, 0.).flatten(1))
        if n >= 4 and fit is not None:
            output.append(decay_fields(axes, active, fit, root_pt))
    return torch.cat(output, 1)
