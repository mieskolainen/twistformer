# Projected charge transport and peak contrasts between ECAL samplings
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import torch
from torch.nn.functional import pad

from .graph_inputs import ANGULAR_GRID_STEP, N_LAYERS, SHOWER_DEPTH_REGIONS, depth_indices, packed_hits


REGIONS = tuple((k, k+1) for k in range(N_LAYERS)) + ((0, 3), (1, 5), (2, 7), (4, 9), (0, 11))
BINS = 65
TRANSPORT_TERMS = ("wasserstein", "cramer", "cubic", "maximum", "shape")
PEAK_TERMS = ("leading", "second", "prominence", "ratio", "prominence_distance", "distance",
              "central1", "central3", "central6", "edge_energy", "curvature")
PAIR_TERMS = ("overlap", "edge", "squared_difference", "marked_difference", "fourth_difference")
def profile_graph_names(regions, n_layers):
    """
    Name projected peak and profile features in contraction order

    Args:
        regions:  Half-open depth regions
        n_layers:  Number of depth bins
    Returns:
        names:  Projected profile feature columns
    """
    pairs = tuple(combinations(range(n_layers), 2))
    return tuple(
        f"profile_{axis}_{tag}" for axis in ("theta", "phi") for tag in (
            *(f"{term}_L{i}_L{j}" for term in TRANSPORT_TERMS for i, j in pairs),
            *(f"s{scale}_{name}" for scale in (1, 3) for name in (
                *(f"{term}_L{lo}to{hi}" for term in PEAK_TERMS for lo, hi in regions),
                *(f"{term}_L{i}_L{j}" for term in PAIR_TERMS for i, j in pairs),
            )),
        )
    )


PROFILE_GRAPH_NAMES = profile_graph_names(REGIONS, N_LAYERS)


def peak_contrasts(profile, peaks, pitch):
    """
    Measure the two highest separated peaks and their intervening valley

    Args:
        profile:  Projected energy fractions in transverse bins
        peaks:  Local maxima of the projected profile
        pitch:  Bin size in mm
    Returns:
        terms:  Peak energies, prominence, distance and energy containment
    """
    bins = torch.arange(BINS, device=profile.device)
    first = peaks.argmax(-1)[..., None]
    offset = (bins-first).abs()
    other = peaks.masked_fill(offset <= 1, 0.)
    second = other.argmax(-1)[..., None]
    high = peaks.gather(-1, first)[..., 0]
    low = other.gather(-1, second)[..., 0]
    bridge = (bins >= torch.minimum(first, second)) & (bins <= torch.maximum(first, second))
    prominence = (low - profile.masked_fill(~bridge, 1.).amin(-1)).clamp_min(0.)
    distance = (first-second)[..., 0].abs().to(profile.dtype) * pitch
    return torch.stack((high, low, prominence, low / high.clamp_min(1e-20), prominence * distance,
                        distance.masked_fill(low <= 0, 0.), *((profile * (offset <= r)).sum(-1) for r in (1, 3, 6))))


def profile_graph_rows(showers, device="cpu", *, shower_frame=False, angular_grid_step=ANGULAR_GRID_STEP):
    """
    Contract projected hit-energy profiles over individual layers and depth bands

    Args:
        showers:  Local cell coordinates in 50 mm and log energy in GeV
        device:   Device for the graph contractions
        shower_frame:  Project to a common angular grid and use relative depth
        angular_grid_step:  Profile pitch in transverse units for shower-frame rows
    Returns:
        rows:  Transport distances, peak contrasts and marked profile edges
    """
    count = len(showers)
    hits, event = packed_hits(showers, device)
    layer, n_layers = depth_indices(hits[:, 2], shower_frame)
    regions = (*((k, k+1) for k in range(n_layers)), *SHOWER_DEPTH_REGIONS) if shower_frame else REGIONS
    group = event * n_layers + layer
    energy = hits[:, 3].exp()
    first, second = torch.triu_indices(n_layers, n_layers, offset=1, device=device)
    fields = []
    pitches = (angular_grid_step,) * 2 if shower_frame else (5.333, 8.89)
    coordinate_scale = 1 if shower_frame else 50
    for axis, pitch in enumerate(pitches):
        position = (coordinate_scale * hits[:, axis] / pitch + BINS//2).clamp(0., BINS-1.)
        left = position.long()
        share = position-left
        size = count*n_layers*BINS
        charge = energy.new_zeros(size).index_add_(0, BINS*group + left, energy*(1-share))
        charge = charge + energy.new_zeros(size).index_add_(0, BINS*group + (left+1).clamp_max(BINS-1), energy*share)
        charge = charge.reshape(count, n_layers, BINS)
        groups = torch.stack([charge[:, lo:hi].sum(1) for lo, hi in regions], dim=1)
        groups = groups / groups.sum(-1, keepdim=True).clamp_min(1e-30)
        cdf = groups[:, :n_layers].cumsum(-1)
        diff = cdf[:, first] - cdf[:, second]
        valid = (cdf[:, first, -1] > .5) & (cdf[:, second, -1] > .5)
        distance = diff.abs().sum(-1)
        fields.extend(value * valid for value in (distance, diff.square().sum(-1), (diff**3).sum(-1).abs(),
                                                  diff.abs().amax(-1), distance - diff.sum(-1).abs()))
        profile = groups
        padded = pad(profile, (1, 1))
        for step in range(1, 4):
            profile = .25 * (padded[..., :-2] + 2*profile + padded[..., 2:])
            padded = pad(profile, (1, 1))
            if step == 2:
                continue
            lower, upper = padded[..., :-2], padded[..., 2:]
            peaks = profile * ((profile >= lower) & (profile >= upper))
            # Average both directions to resolve equal-height peaks under reflection
            forward = peak_contrasts(profile, peaks, pitch)
            reverse = peak_contrasts(profile.flip(-1), peaks.flip(-1), pitch)
            fields.extend((forward + reverse) / 2)
            fields.extend((padded.diff(dim=-1).square().sum(-1), (profile - .5*(lower+upper)).abs().sum(-1)))
            a, b = profile[:, first], profile[:, second]
            product = a*b
            difference = a-b
            squared = difference.square()
            fields.extend(value * valid for value in (product.sqrt().sum(-1), product.sum(-1),
                                                      squared.sum(-1), ((a+b)*squared).sum(-1),
                                                      (difference**4).sum(-1)))
    return torch.column_stack(fields).float()
