# Angular flow multigraphs and resolved subjet mass edges
#
# mikael.mieskolainen@cern.ch (2026)

import torch

from twistformer.geometry.angular import angular_distance_squared, wrap_azimuth
from twistformer.geometry.graph_polynomials import angular_flow_graphs


RADII = (.025, .05, .1, .2, .4)
LOCAL_GRAPHS = ("edge", "double_edge", "wedge", "star3", "path3", "count_edge")
ANGULAR_GRAPHS = ("edge", "wedge", "triangle", "triangle_over_edge3")
PRONG_FIELDS = ("pt_share", "mass", "multiplicity", "leading_share", "width")
PRONG_NAMES = (
    tuple(f"flow_r{radius:g}_{name}" for radius in RADII for name in LOCAL_GRAPHS)
    + tuple(f"angular_beta{beta:g}_{name}" for beta in (.5, 1.) for name in ANGULAR_GRAPHS)
    + tuple(f"tau{n}_beta{beta}" for n in (1, 2, 3, 4) for beta in (1, 2))
    + ("tau21_beta1", "tau32_beta1", "tau43_beta1", "tau21_beta2", "tau32_beta2", "tau43_beta2")
    + tuple(f"prong{rank}_{name}" for rank in (1, 2, 3) for name in PRONG_FIELDS)
    + tuple(f"prong{i}{j}_{name}" for i, j in ((1, 2), (1, 3), (2, 3))
            for name in ("mass", "mass_fraction", "angle", "kt"))
)


def seed_axis(score, points, order):
    """
    Choose the hardest unresolved angular direction with geometric tie breaks

    Args:
        score:   Transverse momentum fraction times distance to existing axes
        points:  Jet-centred eta and phi
        order:   Stable descending eta and phi priority
    Returns:
        axis:  Selected direction with a singleton axis dimension
    """
    best = score.gather(1, order).argmax(1, keepdim=True)
    index = order.gather(1, best).squeeze(1)
    return points.gather(1, index[:, None, None].expand(-1, 1, 2))


def mass(p):
    """Return the timelike invariant mass in GeV"""
    return (p[..., 0].square() - p[..., 1:].square().sum(-1)).clamp_min(0).sqrt()


def subjet_rows(p, w, closest, axes, distance2):
    """
    Form three subjets and contract their pair masses in transverse momentum fraction order

    Args:
        p:          Constituent four-momenta in GeV
        w:          Constituent fractions of scalar transverse momentum
        closest:    Index of the closest subjet axis
        axes:       Three directions in jet-centred eta and phi
        distance2:  Squared distance to the assigned axis
    Returns:
        prongs:  Energy shares, masses, multiplicities and widths
        pairs:   Invariant masses, opening angles and relative transverse momenta
    """
    membership = (closest[..., None] == torch.arange(3, device=p.device)) & (w > 0)[..., None]
    assigned = membership * w[..., None]
    charge = assigned.sum(1)
    rank = charge.argsort(dim=1, descending=True, stable=True)
    subjet = membership.to(p.dtype).transpose(1, 2) @ p
    directions = axes.gather(1, rank[..., None].expand(-1, -1, 2))
    width = (assigned * distance2.sqrt()[..., None]).sum(1)
    stats = (charge, mass(subjet), membership.sum(1),
             assigned.amax(1) / charge.clamp_min(1e-30), width / charge.clamp_min(1e-30))
    prongs = torch.stack(tuple(value.gather(1, rank) for value in stats), -1)
    subjet = subjet.gather(1, rank[..., None].expand(-1, -1, 4))
    hard, soft = (0, 0, 1), (1, 2, 2)
    pair_mass = mass(subjet[:, hard] + subjet[:, soft])
    angle = angular_distance_squared(directions[:, hard], directions[:, soft], periodic_azimuth=True).sqrt()
    subjet_pt = torch.linalg.vector_norm(subjet[..., 1:3], dim=-1)
    kt = torch.minimum(subjet_pt[:, hard], subjet_pt[:, soft]) * angle
    pairs = torch.stack((pair_mass, pair_mass / mass(p.sum(1))[:, None].clamp_min(1e-6), angle, kt), -1)
    pairs = pairs * (charge.gather(1, rank)[:, soft, None] > 0)
    return prongs, pairs


def prong_flow(points, w, radii=RADII, angular=True):
    """
    Contract angular flow between constituents

    Args:
        points:  Jet-centred eta and phi
        w:       Fractions of the scalar transverse momentum
        radii:   Gaussian angular scales required by the requested graph columns
        angular: Include the angular edge, wedge and triangle observables
    Returns:
        rows:  Angular correlation columns
    """
    return angular_flow_graphs(points, w, radii, angular, periodic_azimuth=True)


def resolved_prongs(p, points, w):
    """Measure N-subjettiness and the three resolved prong momenta"""
    tau1, tau2 = [], []
    eta, phi = points.unbind(-1)
    by_phi = phi.argsort(dim=1, descending=True, stable=True)
    by_eta = eta.gather(1, by_phi).argsort(dim=1, descending=True, stable=True)
    seed_order = by_phi.gather(1, by_eta)
    axes = seed_axis(w, points, seed_order)
    for n in (1, 2, 3, 4):
        groups = torch.arange(n, device=p.device)
        for _ in range(2):
            delta_eta, delta_phi = (points[:, :, None] - axes[:, None]).unbind(-1)
            delta_phi = wrap_azimuth(delta_phi)
            closest = (delta_eta.square() + delta_phi.square()).argmin(2)
            assigned = (closest[..., None] == groups) * w[..., None]
            charge = assigned.sum(1)
            # Update phi in the local chart of each subjet
            step = (assigned[..., None] * torch.stack((delta_eta, delta_phi), -1)).sum(1)
            axes = axes + step / charge.clamp_min(1e-30)[..., None]
        distance2, closest = angular_distance_squared(points[:, :, None], axes[:, None], periodic_azimuth=True).min(2)
        tau1.append((w * distance2.sqrt()).sum(1))
        tau2.append((w * distance2).sum(1))
        if n == 3:
            prongs, pairs = subjet_rows(p, w, closest, axes, distance2)
        if n < 4:
            axes = torch.cat((axes, seed_axis(w * distance2, points, seed_order)), 1)
    fields = [value for pair in zip(tau1, tau2) for value in pair]
    fields.extend(tau[i] / tau[i-1].clamp_min(1e-12) for tau in (tau1, tau2) for i in (1, 2, 3))
    return torch.cat((torch.stack(fields, 1), prongs.flatten(1), pairs.flatten(1)), 1)


def prong_rows(p, points, w):
    """Return the full angular-flow and resolved-prong columns"""
    return torch.cat((prong_flow(points, w), resolved_prongs(p, points, w)), 1)
