# Torch cell buffers for ECAL shower graphs
#
# mikael.mieskolainen@cern.ch (2026)

import torch

from twistformer.geometry.graph_polynomials import point_tensor


# Radial midpoints between the eleven ALLEGRO barrel samplings in mm
SAMPLING_EDGES_MM = (18, 46, 79, 113, 149, 187, 226, 268, 312, 358)
N_LAYERS = len(SAMPLING_EDGES_MM) + 1

# Analysis scales in the shower frame
ANGLE_UNIT_RAD = .025
ANGULAR_GRID_STEP = .2  # 5 mrad in transverse coordinate units
RADIAL_GRID_STEP = .1  # 2.5 mrad in transverse coordinate units
RADIAL_DEPTH_UNIT_MM = 40.
SHOWER_DEPTH_EDGES_MM = (-225, -175, -125, -75, -25, 25, 75, 125, 175, 225)
SHOWER_DEPTH_REGIONS = ((0, 11), (0, 4), (3, 8), (7, 11), (4, 7))


def shower_frame_hits(position, energy, feature_mode, *, include_radial=False):
    """
    Express measured deposits in a frame set by the shower barycentre

    Args:
        position:  Cell positions in mm from the interaction point
        energy:  Positive cell energies in GeV
        feature_mode:  Angular offsets or normalized tangent-plane coordinates
        include_radial:  Append beam angles and radial depth from the first hit
    Returns:
        result:  Projective shower coordinates and log energy, with a radial view if requested
    """
    position = point_tensor(position)
    energy = point_tensor(energy, position.device)
    if not torch.isfinite(position).all():
        raise ValueError("Positive-energy cell positions must be finite")
    weight = energy / energy.sum()
    center = weight @ position
    radius = center.norm()
    rho = center[:2].norm()
    if rho <= 1e-12 * (weight @ position.norm(dim=1)):
        raise ValueError("The shower azimuth axis is undefined")
    normal = center / radius
    delta = position - center
    depth = (delta @ normal) / 50
    phi_axis = torch.stack((-center[1], center[0], center.new_zeros(()))) / rho
    theta_axis = torch.linalg.cross(phi_axis, normal)
    x, y = delta @ theta_axis, delta @ phi_axis
    if feature_mode == "cartesian":
        x, y = x / radius, y / radius
    else:
        distance = position @ normal
        x, y = torch.atan2(x, distance), torch.atan2(y, distance)
    log_energy = energy.log()
    projective = (x / ANGLE_UNIT_RAD, y / ANGLE_UNIT_RAD, depth, log_energy)
    if not include_radial:
        return torch.column_stack(projective).float()
    cell_radius = torch.hypot(position[:, 0], position[:, 1])
    theta = torch.atan2(cell_radius, position[:, 2])
    phi = torch.atan2(position[:, 1], position[:, 0])
    phi = phi - torch.atan2(weight @ phi.sin(), weight @ phi.cos())
    # Use the first measured radial deposit as the depth reference
    radial_depth = (cell_radius - cell_radius.amin()) / RADIAL_DEPTH_UNIT_MM - len(SHOWER_DEPTH_EDGES_MM) / 2
    radial = ((theta - weight @ theta) / ANGLE_UNIT_RAD,
              torch.atan2(phi.sin(), phi.cos()) / ANGLE_UNIT_RAD, radial_depth, log_energy)
    return torch.column_stack((*projective, *radial)).float()


def depth_indices(depth, shower_frame):
    """
    Assign cells to fixed analysis depth bins or the original ECAL samplings

    Args:
        depth:  Cell depth in 50 mm units
        shower_frame:  Use depth relative to the shower barycentre
    Returns:
        indices: Depth bin of each cell
        count:  Number of depth bins
    """
    edges = SHOWER_DEPTH_EDGES_MM if shower_frame else SAMPLING_EDGES_MM
    return torch.bucketize((50 * depth).contiguous(), depth.new_tensor(edges), right=True), len(edges) + 1


def packed_hits(showers, device):
    """
    Pack variable-length showers with one event index per cell

    Args:
        showers:  Local cell coordinates and log energies in source order
        device:  Device for the packed graph inputs
    Returns:
        hits:   Float64 cell inputs
        event:  Shower index of each cell
    """
    showers = tuple(showers)
    first = showers[0] if showers else None
    if torch.is_tensor(first) and all(torch.is_tensor(shower) and shower.dtype == first.dtype and shower.device == first.device
                                      for shower in showers):
        hits = torch.cat(showers).to(device=device, dtype=torch.float64)
    else:
        hits = torch.cat([point_tensor(shower) for shower in showers]).to(device)
    counts = torch.tensor([len(shower) for shower in showers], device=device)
    event = torch.repeat_interleave(torch.arange(len(showers), device=device), counts, output_size=len(hits))
    return hits, event


def oriented_core(xy, depth, weight, scales):
    """
    Centre the shower core and fix its beam reflection

    Args:
        xy:      Transverse coordinates in 50 mm units
        depth:   Cell depths in 50 mm units
        weight:  Cell energy fractions
        scales:  Successive search radii in 50 mm units
    Returns:
        result:  Oriented transverse coordinates
    """
    center = weight @ xy
    for scale in scales:
        delta = xy - center
        core = weight / (1 + delta.square().sum(1) / scale**2).square()
        center = core @ xy / core.sum()
    x, y = (xy - center).unbind(-1)
    dipole = weight @ x
    if dipole.abs() <= 1e-12:
        # Resolve a vanishing dipole with groups invariant under beam reflection
        keys = torch.stack((weight, depth, y, x.abs()), dim=1)
        groups, indices = torch.unique(keys, dim=0, return_inverse=True)
        signed = x.new_zeros(len(groups)).index_add_(0, indices, x)
        order = torch.arange(len(groups), device=x.device)
        dipole = signed[order.masked_fill(signed.abs() <= 1e-12, 0).amax()]
    return torch.stack((torch.where(dipole < -1e-12, -x, x), y), dim=1)
