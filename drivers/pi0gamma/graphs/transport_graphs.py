# Connected transverse moments along the longitudinal shower axis
#
# mikael.mieskolainen@cern.ch (2026)

from math import isfinite

import torch

from .graph_inputs import ANGLE_UNIT_RAD, SHOWER_DEPTH_REGIONS, depth_indices, packed_hits

BANDS = ((1, 2), (0, 3), (2, 7), (6, 11), (0, 11))
WINDOWS = ("12mm", "32mm", "80mm", "mass")
SHOWER_WINDOWS = ("6mrad", "16mrad", "40mrad", "mass")
DIRECTIONS = ("theta", "phi", "diagonal_mean", "diagonal_difference")
CONTRACTIONS = ("variance", "skew_squared", "excess", "bimodality")
BAND_TERMS = ("mass_fraction", "ellipticity", "cross_covariance", "tilt_alignment")
BAND_PAIRS = ((0, 2), (1, 2), (2, 3))
def transport_names(bands, windows):
    """
    Name the transverse cumulants and depth-band comparisons

    Args:
        bands:  Half-open depth regions
        windows:  Labels of the transverse apertures
    Returns:
        names:  Longitudinal transport feature columns
    """
    return tuple(f"transport_{window}_{tag}" for window in windows for tag in (
        *(f"L{lo}to{hi}_{term}" for lo, hi in bands for term in
          (*BAND_TERMS, *(f"{c}_{d}" for c in CONTRACTIONS for d in DIRECTIONS))),
        *(f"bands{i}_{j}_{term}" for i, j in BAND_PAIRS for term in ("centroid_edge", "tensor_alignment", "width_ratio")),
    ))


TRANSPORT_NAMES = transport_names(BANDS, WINDOWS)
SHOWER_TRANSPORT_NAMES = transport_names(SHOWER_DEPTH_REGIONS, SHOWER_WINDOWS)


def transport_rows(showers, inner_radius_mm=None, device="cpu", *, shower_frame=False):
    """
    Contract transverse energy cumulants after removing the common shower slope

    Args:
        showers:  Local transverse coordinates and depth in units of 50 mm, log cell energy
        inner_radius_mm:  Inner ECAL radius for the energy-scaled aperture
        device:  Device for the graph contractions
        shower_frame:  Use angular apertures and relative depth
    Returns:
        rows:  Reflection-even connected moments and correlations between depth bands
    """
    if not shower_frame and (inner_radius_mm is None or not isfinite(inner_radius_mm) or inner_radius_mm <= 0):
        raise ValueError("Longitudinal shower graphs require a positive ECAL inner radius in mm")
    count = len(showers)
    hits, event = packed_hits(showers, device)
    x, y, z, log_energy = hits.unbind(-1)
    energy = log_energy.exp()
    layer, n_layers = depth_indices(z, shower_frame)
    bands = SHOWER_DEPTH_REGIONS if shower_frame else BANDS
    group = event * n_layers + layer

    def total(value):
        """Sum a cell mark in each shower"""
        return value.new_zeros(count).index_add_(0, event, value)

    deposited = total(energy)
    # Fit a common line to the energy core in all samplings
    slope_x = slope_y = deposited.new_zeros(count)
    intercept_x, intercept_y = total(energy*x) / deposited, total(energy*y) / deposited
    for radius in (1.2, .5):
        dx = x - intercept_x[event] - slope_x[event]*z
        dy = y - intercept_y[event] - slope_y[event]*z
        mark = energy / (1 + (dx*dx + dy*dy) / radius**2)**2
        q = mark / total(mark)[event]
        depth = total(q*z)
        dz = z - depth[event]
        longitudinal = total(q*dz*dz)
        slope_x = total(q*dz*x) / longitudinal.clamp_min(1e-12)
        slope_y = total(q*dz*y) / longitudinal.clamp_min(1e-12)
        intercept_x = total(q*x) - slope_x*depth
        intercept_y = total(q*y) - slope_y*depth
    dx = x - intercept_x[event] - slope_x[event]*z
    dy = y - intercept_y[event] - slope_y[event]*z
    r2 = dx*dx + dy*dy
    direction = torch.column_stack((dx, dy, (dx + dy) * 2**-.5, (dx - dy) * 2**-.5))
    direction2 = direction.square()
    tilt2 = slope_x**2 + slope_y**2
    slope_difference = (slope_x**2 - slope_y**2)[:, None]
    slope_product = (slope_x*slope_y)[:, None]

    def pool(value):
        """Sum a cell mark over each depth band"""
        shape = value.shape[1:]
        charge = value.new_zeros((count * n_layers, *shape)).index_add_(0, group, value)
        charge = charge.reshape(count, n_layers, *shape)
        return torch.stack([charge[:, lo:hi].sum(1) for lo, hi in bands], dim=1)

    fields = []
    scale, unit = (1., ANGLE_UNIT_RAD) if shower_frame else (inner_radius_mm, 50)
    mass_radius = torch.hypot(hits.new_tensor(.1), scale * .24 / (unit * deposited[event]))
    for radius in (.24, .64, 1.6, mass_radius):
        mark = energy / (1 + r2 / radius**2)**3
        mass = pool(mark)
        norm = mass.clamp_min(1e-30)
        first = mark[:, None] * direction
        second = mark[:, None] * direction2
        moments = torch.stack((first, second, second*direction, second*direction2), dim=-1)
        mean, raw2, raw3, raw4 = (pool(moments) / norm[..., None, None]).unbind(-1)
        var = (raw2 - mean**2).clamp_min(0)
        central3 = raw3 - 3*mean*raw2 + 2*mean**3
        central4 = raw4 - 4*mean*raw3 + 6*mean**2*raw2 - 3*mean**4
        # A width floor bounds the normalized moments in sparse depth bins
        scale = var.clamp_min(.0004)
        skew2 = central3**2 / scale**3
        kurt = (central4 / scale**2).clamp_min(0)
        vx, vy, vp, vm = var.unbind(-1)
        trace = vx + vy
        norm2 = trace.clamp_min(.0004)
        # The diagonal variances recover twice the transverse covariance
        anis_x, anis_y = (vx - vy) / norm2, (vp - vm) / norm2
        align = anis_x * slope_difference + 2 * anis_y * slope_product
        align = align / tilt2[:, None].clamp_min(1e-8)
        family = [mass / deposited[:, None], torch.hypot(anis_x, anis_y), anis_y.abs(), align]
        for term in (var, skew2, kurt - 3, (skew2 + 1) / kurt.clamp_min(1)):
            theta, phi, plus, minus = term.unbind(-1)
            family.extend((theta, phi, (plus + minus) * .5, (plus - minus).abs()))
        fields.append((torch.stack(family, -1) * (mass > 0)[..., None]).reshape(count, -1))
        for i, j in BAND_PAIRS:
            valid = (mass[:, i] > 0) & (mass[:, j] > 0)
            distance = (mean[:, i, 0] - mean[:, j, 0])**2 + (mean[:, i, 1] - mean[:, j, 1])**2
            alignment = anis_x[:, i]*anis_x[:, j] + anis_y[:, i]*anis_y[:, j]
            ratio = trace[:, i] / trace[:, j].clamp_min(.0004)
            fields.append(torch.column_stack((distance, alignment, ratio)) * valid[:, None])
    return torch.column_stack(fields).float()
