# Energy-scaled Fourier multigraphs and mixed-depth triplets
#
# mikael.mieskolainen@cern.ch (2026)

from math import isfinite

import torch

from twistformer.geometry.graph_polynomials import unit_phase
from .graph_inputs import ANGLE_UNIT_RAD, SHOWER_DEPTH_REGIONS, depth_indices, packed_hits


MASS_SCALES = (.06, .12, .24, .48)
REGIONS = ((0, 11), (1, 2), (0, 3), (2, 7), (6, 11))
REGION_PAIRS = ((1, 2), (1, 3), (2, 3), (3, 4))
SHOWER_REGION_PAIRS = ((1, 2), (1, 3), (2, 3), (2, 4))
DIRECTIONS = ((1., 0.), (0., 1.), (2**-.5, 2**-.5), (2**-.5, -2**-.5))
REGION_TERMS = ("width", "width2", "gram3", "gram3_ratio", "closure112", "closure123")
PAIR_TERMS = ("difference", "alignment", "quadrupole")
def energy_scaled_names(regions, pairs):
    """
    Name angular widths and phase closures in contraction order

    Args:
        regions:  Half-open depth regions
        pairs:  Region pairs for mixed-depth terms
    Returns:
        names:  Energy-scaled feature columns
    """
    return tuple(
        f"mass_{mark}_{tag}_{axis}_{scale}GeV"
        for mark in ("energy", "sqrt_energy") for tag in (
            *(f"L{lo}to{hi}_{name}" for lo, hi in regions for name in REGION_TERMS),
            *(f"region{i}_region{j}_{name}" for i, j in pairs for name in PAIR_TERMS),
        ) for axis in ("theta", "phi", "diagonal_mean", "diagonal_difference") for scale in MASS_SCALES
    )


ENERGY_SCALED_GRAPH_NAMES = energy_scaled_names(REGIONS, REGION_PAIRS)
SHOWER_ENERGY_SCALED_NAMES = energy_scaled_names(SHOWER_DEPTH_REGIONS, SHOWER_REGION_PAIRS)


def energy_scaled_graph_rows(showers, device, inner_radius_mm=None, *, shower_frame=False, depth_regions=None):
    """
    Resolve two-prong structure at angular scales proportional to inverse shower energy

    Args:
        showers:  Local cell coordinates in 50 mm and log energy in GeV
        device:  Device for the graph contractions
        inner_radius_mm:  Inner radius of the ECAL barrel in mm
        shower_frame:  Use measured angular offsets and relative depth
        depth_regions:  Analysis bands for a custom depth coordinate
    Returns:
        rows:  Energy-marked edges, triplets and cross-layer contractions
    """
    if not shower_frame and (inner_radius_mm is None or not isfinite(inner_radius_mm) or inner_radius_mm <= 0):
        raise ValueError("Energy-scaled graphs require a positive ECAL inner radius in mm")
    count = len(showers)
    hits, event = packed_hits(showers, device)
    regions = depth_regions if depth_regions is not None else (SHOWER_DEPTH_REGIONS if shower_frame else REGIONS)
    pairs = SHOWER_REGION_PAIRS if shower_frame else REGION_PAIRS
    energy, orders = energy_scaled_phases(hits, event, count, inner_radius_mm=None if shower_frame else inner_radius_mm)
    return energy_scaled_depth_rows(hits[:, 2], event, energy, orders, count,
                                    shower_frame=shower_frame, regions=regions, pairs=pairs)


def energy_scaled_phases(hits, event, count, *, inner_radius_mm=None):
    """
    Form angular phases independent of the longitudinal depth coordinate

    Args:
        hits:  Packed transverse positions, depth and log energy
        event:  Shower index of each hit
        count:  Number of showers
        inner_radius_mm:  Detector radius for detector-frame distances or None for angles
    Returns:
        energy:  Cell energies in GeV
        orders:  Complex angular phases for the first three Fourier orders
    """
    energy = hits[:, 3].exp()
    total = energy.new_zeros(count).index_add_(0, event, energy)
    coordinate_scale = ANGLE_UNIT_RAD if inner_radius_mm is None else 50
    scale_radius = 1. if inner_radius_mm is None else inner_radius_mm
    scale_floor = .001 if inner_radius_mm is None else 2.5
    projected = coordinate_scale * (hits[:, :2] @ hits.new_tensor(DIRECTIONS).T)
    scales = ((scale_radius * hits.new_tensor(MASS_SCALES) / total[:, None]).square() + scale_floor**2).sqrt()
    phase = projected[:, :, None] / scales[event, None, :]
    wave = unit_phase(phase)
    return energy, torch.stack((wave, wave.square(), wave.pow(3)), dim=-1)


def energy_scaled_depth_rows(depth, event, energy, orders, count, *, shower_frame, regions, pairs):
    """
    Contract the angular phases in one depth coordinate

    Args:
        depth:  Longitudinal coordinate per cell
        event:  Shower index of each hit
        energy:  Cell energies in GeV
        orders:  First three angular Fourier orders
        count:  Number of showers
        shower_frame:  Use the relative shower depth grid
        regions:  Half-open depth bands
        pairs:  Depth band pairs for mixed terms
    Returns:
        rows:  Angular edges, triplets and mixed-depth contractions
    """
    layer, n_layers = depth_indices(depth, shower_frame)
    index = event * n_layers + layer
    first, second = torch.tensor(pairs, device=energy.device).unbind(-1)
    fields = []

    def symmetrize(terms):
        """
        Symmetrize the oblique projections and flatten in column order

        Args:
            terms:  Region terms with direction and scale as the final axes
        Returns:
            rows:  Flattened features in the original contraction order
        """
        theta, phi, plus, minus = terms.unbind(-2)
        return torch.stack((theta, phi, (plus + minus) / 2, (plus - minus).abs()), dim=-2).flatten(1)

    for mark in (energy, energy.sqrt()):
        mass = energy.new_zeros(count * n_layers).index_add_(0, index, mark).reshape(count, n_layers)
        moments = orders.new_zeros((count * n_layers, *orders.shape[1:]))
        moments.index_add_(0, index, mark[:, None, None, None] * orders)
        moments = moments.reshape(count, n_layers, *orders.shape[1:])
        charge = torch.stack([mass[:, lo:hi].sum(1) for lo, hi in regions], dim=1)
        phases = torch.stack([moments[:, lo:hi].sum(1) for lo, hi in regions], dim=1)
        f1, f2, f3 = (phases / charge.clamp_min(1e-30)[..., None, None, None]).unbind(-1)
        present = (charge > 0)[..., None, None]
        width = (1 - f1.abs().square()).clamp_min(0)
        width2 = width.square()
        f1_squared = f1.square()
        quad = f2 - f1_squared
        # The Gram determinant sums squared Vandermonde factors over hit triplets
        triple = (width2 - quad.abs().square()).clamp_min(0)
        region_terms = torch.stack((width, 1-f2.abs().square(), triple, triple/(width2+1e-5),
                                    (f1_squared*f2.conj()).real, (f1*f2*f3.conj()).real), dim=2)
        fields.append(symmetrize(region_terms * present[:, :, None]))

        a, b = f1[:, first], f1[:, second]
        pair_terms = torch.stack(((a-b).abs().square(), (a*b.conj()).real,
                                  (quad[:, first]*quad[:, second].conj()).real /
                                  (width[:, first]*width[:, second] + 1e-4)), dim=2)
        fields.append(symmetrize(pair_terms * (present[:, first] & present[:, second])[:, :, None]))
    return torch.cat(fields, dim=1).float()
