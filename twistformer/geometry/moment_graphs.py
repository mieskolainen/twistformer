# Higher-order angular flow graphs with momentum and multiplicity weights
#
# mikael.mieskolainen@cern.ch (2026)

import torch


RADII = (.10, .25, .50)
CHARGES = ("pt", "sqrt_pt", "count")
STAR_ORDERS = (1, 2, 3, 5, 8, 12)
HUB_ORDERS = ((2, 2), (2, 4), (3, 6), (6, 6))
FIELDS = (
    tuple(f"star{order}" for order in STAR_ORDERS)
    + tuple(f"bridge{a}_{b}" for a, b in HUB_ORDERS)
    + ("triangle", "butterfly", "three_triangles", "four_triangles", "triangle_star4")
)
MOMENT_NAMES = tuple(
    f"moment_r{radius:g}_{charge}_{field}"
    for radius in RADII for charge in CHARGES for field in FIELDS
)


def jet_angular_flow(p, px=None, py=None):
    """
    Centre angular coordinates on the transverse momentum flow

    Args:
        p:   Batched constituent four-momenta
        px:  Optional total jet momentum along x
        py:  Optional total jet momentum along y
    Returns:
        points:     Centred pseudorapidity and wrapped azimuth
        w:          Fractions of scalar transverse momentum
        scalar_pt:  Scalar transverse momentum sum
    """
    if px is None:
        px = p[:, :, 1].sum(1)
    if py is None:
        py = p[:, :, 2].sum(1)
    pt = torch.hypot(p[:, :, 1], p[:, :, 2])
    scalar_pt = pt.sum(1, keepdim=True)
    w = pt / torch.where(scalar_pt > 0, scalar_pt, 1.)
    eta = torch.asinh(p[:, :, 3] / torch.where(pt > 0, pt, 1.))
    phi = torch.atan2(p[:, :, 2], p[:, :, 1]) - torch.atan2(py, px)[:, None]
    points = torch.stack((eta, (phi + torch.pi).remainder(2 * torch.pi) - torch.pi), -1)
    points = torch.where((pt > 0)[..., None], points, 0.)
    points = points - (w[..., None] * points).sum(1, keepdim=True)
    return points, w, scalar_pt


def moment_rows(points, fraction):
    """
    Contract rooted stars, joined stars and triangles with a common root

    Args:
        points:    Jet-centred pseudorapidity and wrapped azimuth
        fraction:  Normalized constituent transverse momenta with zero padding
    Returns:
        rows:  Multigraph coefficients through thirteen edges
    """
    positive = fraction > 0
    charges = torch.stack((fraction, fraction.sqrt(), positive.to(fraction.dtype)), dim=1)
    charges = charges / charges.sum(-1, keepdim=True).clamp_min(torch.finfo(fraction.dtype).tiny)
    rows = []
    for radius in RADII:
        planar = points / radius
        radius2 = planar.square().sum(-1, keepdim=True)
        sphere = torch.cat((2 * planar, 1 - radius2), dim=-1) / (1 + radius2)
        unit = torch.ones_like(radius2)
        left = torch.cat((unit, sphere), dim=-1)
        right = .5 * torch.cat((unit, -sphere), dim=-1)

        # Each leaf supplies a charge-weighted integral of the spherical edge
        charge_moment = charges @ sphere
        neighbor = (.5 * (1 - charge_moment @ sphere.transpose(1, 2))).clamp(0, 1)
        root_weight = fraction[:, None]
        powers = {order: neighbor.pow(order) for order in (*STAR_ORDERS, 4, 6)}
        fields = [(root_weight * powers[order]).sum(-1) for order in STAR_ORDERS]

        # Join two stars by contracting the scalar and vector moments of each hub
        hub = {order: (root_weight * powers[order]) @ left for order in (2, 3, 4, 6)}
        for a, b in HUB_ORDERS:
            scalar = hub[a][..., 0] * hub[b][..., 0]
            vector = (hub[a][..., 1:] * hub[b][..., 1:]).sum(-1)
            fields.append((.5 * (scalar - vector)).clamp_min(0))

        # K = L R^T has rank four, so a rooted triangle needs two small matrices
        transfer = right.transpose(1, 2)[:, None] @ (charges[..., None] * left[:, None])
        triangle = ((left[:, None] @ (transfer @ transfer)) * right[:, None]).sum(-1).clamp(0, 1)
        fields.extend((root_weight * triangle.pow(order)).sum(-1) for order in (1, 2, 3, 4))
        fields.append((root_weight * triangle * powers[4]).sum(-1))
        rows.append(torch.stack(fields, dim=-1).flatten(1))
    return torch.cat(rows, dim=1)


def moment_features(momenta):
    """
    Measure the full angular moment basis on the input tensor device

    Args:
        momenta:  Floating-point (jets, particles, 4) tensor in E, px, py, pz order
    Returns:
        rows:  Tensor of shape (jets, 135) in MOMENT_NAMES order
    """
    points, fraction, _ = jet_angular_flow(momenta)
    return moment_rows(points, fraction)
