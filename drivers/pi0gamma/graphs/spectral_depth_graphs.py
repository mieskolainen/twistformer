# Energy-scaled angular graphs in measured longitudinal depth windows
#
# mikael.mieskolainen@cern.ch (2026)

from math import cos, pi, sin

import torch
from torch.nn.utils.rnn import pad_sequence

from twistformer.geometry.graph_polynomials import point_tensor, unit_phase
from .graph_inputs import ANGLE_UNIT_RAD, RADIAL_DEPTH_UNIT_MM, packed_hits


DEPTH_WINDOWS = ((0., .3), (.1, .45), (.15, .65), (.3, .75), (.45, .95))
MASS_SCALES = (.12, .24, .48)
DIRECTIONS = tuple((cos(index*pi/8), sin(index*pi/8)) for index in range(8))
AZIMUTH_MOMENTS = tuple(tuple(frequency(index * order*pi/8) for index in range(8))
                        for order in (2, 4) for frequency in (cos, sin))
CONTRACTIONS = ("width", "width2", "closure112", "closure123")
SHAPE_TERMS = ("mean", "range", "quadrupole_cos", "quadrupole_sin_abs", "quadrupole", "harmonic4")


SPECTRAL_DEPTH_NAMES = tuple(
    f"spectral_depth_{lo:g}_{hi:g}_{name}"
    for lo, hi in DEPTH_WINDOWS
    for name in ("energy_fraction", *(f"{contraction}_{term}_{scale:g}GeV"
                                     for contraction in CONTRACTIONS for term in SHAPE_TERMS for scale in MASS_SCALES))
)


def normalized_depth(shower):
    """
    Measure each hit depth from the front to the 99.5 percent energy depth

    Args:
        shower:  Radial shower coordinates and log energies
    Returns:
        depth:  Dimensionless hit depth in source order
    """
    hits = point_tensor(shower)
    depth, order = hits[:, 2].sort(stable=True)
    energy = hits[order, 3].exp()

    # Use the same energy depth for deposits at nearly equal radii
    gap = (depth.diff() > .5/RADIAL_DEPTH_UNIT_MM).long()
    shell = torch.cat((gap.new_zeros(1), gap)).cumsum(0)
    charge = energy.new_zeros(len(hits)).index_add_(0, shell, energy)
    cumulative = charge.cumsum(0)[shell] / energy.sum()
    bound = torch.searchsorted(cumulative, depth.new_tensor(.995)).clamp_max(len(depth)-1)
    span = (depth.gather(0, bound.reshape(1)) - depth[0]).clamp_min(.25)
    return ((hits[:, 2] - depth[0]) / span).clamp_max(1)


def spectral_depth_rows(showers, device="cpu"):
    """
    Contract phase widths and triplets across smooth measured depth windows

    Args:
        showers:  Beam angles, radial depth and log energy per hit
        device:  Device for the graph contractions
    Returns:
        rows:  Energy fractions and angular harmonics per depth window
    """
    count = len(showers)
    hits, event = packed_hits(showers, device)
    depth = torch.cat([normalized_depth(shower) for shower in showers]).to(device)
    energy = hits[:, 3].exp()
    total = energy.new_zeros(count).index_add_(0, event, energy)
    scale = ((hits.new_tensor(MASS_SCALES) / total[:, None]).square() + .001**2).sqrt()
    phase = ANGLE_UNIT_RAD * (hits[:, :2] @ hits.new_tensor(DIRECTIONS).T)[..., None] / scale[event, None, :]
    wave = unit_phase(phase)
    orders = torch.stack((wave, wave.square(), wave.pow(3)), dim=-1)
    lo, hi = hits.new_tensor(DEPTH_WINDOWS).unbind(-1)
    depth = depth[:, None]
    weight = energy[:, None] * ((depth-lo+.04)/.08).clamp(0, 1) * ((hi-depth+.04)/.08).clamp(0, 1)
    lengths = [len(shower) for shower in showers]
    weights = weight.split(lengths)
    phases = torch.view_as_real(orders).flatten(1).split(lengths)
    windows = len(DEPTH_WINDOWS)
    # Bound padding for batches with widely different shower sizes
    if count * max(lengths) > 2 * len(hits):
        summed = torch.stack([mark.T @ values for mark, values in zip(weights, phases, strict=True)])
    else:
        summed = torch.bmm(pad_sequence(weights, batch_first=True).transpose(1, 2),
                           pad_sequence(phases, batch_first=True))
    moments = torch.view_as_complex(summed.reshape(count, windows, *orders.shape[1:], 2))
    charge = energy.new_zeros((count, windows)).index_add_(0, event, weight)
    first, second, third = (moments / charge.clamp_min(1e-20)[..., None, None, None]).unbind(-1)

    width = (1-first.abs().square()).clamp_min(0)
    width2 = 1-second.abs().square()
    closure112 = (first.square() * second.conj()).real
    closure123 = (first * second * third.conj()).real
    terms = torch.stack((width, width2, closure112, closure123), dim=2)
    azimuth = hits.new_tensor(AZIMUTH_MOMENTS).T
    cos2, sin2, cos4, sin4 = ((terms.transpose(-2, -1) @ azimuth) / len(DIRECTIONS)).unbind(-1)
    angular = torch.stack((terms.mean(-2), terms.amax(-2)-terms.amin(-2), cos2, sin2.abs(),
                           torch.hypot(cos2, sin2), torch.hypot(cos4, sin4)), dim=-2)
    fraction = charge / total[:, None]
    return torch.cat((fraction[..., None], angular.flatten(2)), dim=-1).flatten(1).float()
