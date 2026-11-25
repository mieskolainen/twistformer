# Energy-scaled graphs in the relative order of measured ECAL samplings
#
# mikael.mieskolainen@cern.ch (2026)

import torch

from twistformer.geometry.graph_polynomials import point_tensor
from .energy_scaled_graphs import (
    DIRECTIONS, MASS_SCALES, REGION_TERMS, SHOWER_REGION_PAIRS, energy_scaled_depth_rows, energy_scaled_graph_rows,
    energy_scaled_names, energy_scaled_phases,
)
from .graph_inputs import RADIAL_DEPTH_UNIT_MM, SHOWER_DEPTH_EDGES_MM, SHOWER_DEPTH_REGIONS, packed_hits


# Rank bands share the eleven analysis bins of the shower depth coordinate
RANK_REGIONS = ((0, 11), (1, 2), (0, 3), (2, 7), (6, 11))
FULL_SHOWER_WIDTH = len(REGION_TERMS) * len(DIRECTIONS) * len(MASS_SCALES)
NAMES = energy_scaled_names(RANK_REGIONS, SHOWER_REGION_PAIRS)
HALF_WIDTH = len(NAMES) // 2
SAMPLING_RANK_NAMES = tuple(
    f"sampling_rank_{name.replace('_L', '_Q')}"
    for index in (0, HALF_WIDTH) for name in NAMES[index + FULL_SHOWER_WIDTH:index + HALF_WIDTH]
)


def ranked_depth_hits(shower):
    """
    Scale measured sampling ranks to the analysis depth coordinate

    Args:
        shower:  Beam angles, radial depth and log energy in source order
    Returns:
        hits:  Beam angles, relative sampling rank and log energy
    """
    hits = point_tensor(shower).clone()
    depth, order = hits[:, 2].sort(stable=True)
    gap = (depth.diff() > .5 / RADIAL_DEPTH_UNIT_MM).long()
    rank = torch.cat((gap.new_zeros(1), gap)).cumsum(0).to(hits.dtype)
    span = len(SHOWER_DEPTH_EDGES_MM)
    hits[order, 2] = span * rank / rank[-1].clamp_min(1) - span / 2
    return hits


def sampling_rank_rows(showers, device="cpu"):
    """
    Contract angular graphs across relative sampling bands

    Args:
        showers:  Radial hits grouped by shower
        device:  Device for the graph contractions
    Returns:
        rows:  Energy-scaled angular edges and triplets across sampling bands
    """
    hits = [ranked_depth_hits(shower) for shower in showers]
    rows = energy_scaled_graph_rows(hits, device, shower_frame=True, depth_regions=RANK_REGIONS)
    return rank_columns(rows)


def rank_columns(rows):
    """
    Keep the sampling bands and omit the repeated full shower terms

    Args:
        rows:  Fourier graph columns in contraction order
    Returns:
        result:  Columns specific to the sampling-rank coordinate
    """
    return torch.cat([mark[:, FULL_SHOWER_WIDTH:] for mark in rows.chunk(2, dim=1)], dim=1)


def radial_and_sampling_rank_rows(showers, device="cpu"):
    """
    Reuse the measured radial angles for both depth coordinates

    Args:
        showers:  Radial hits grouped by shower
        device:  Device for the graph contractions
    Returns:
        radial:  Energy-scaled graphs in measured depth
        ranked:  Energy-scaled graphs in sampling-rank depth
    """
    count = len(showers)
    hits, event = packed_hits(showers, device)
    energy, orders = energy_scaled_phases(hits, event, count)
    depth = torch.cat([ranked_depth_hits(shower)[:, 2] for shower in showers]).to(device)
    radial = energy_scaled_depth_rows(hits[:, 2], event, energy, orders, count,
                                      shower_frame=True, regions=SHOWER_DEPTH_REGIONS, pairs=SHOWER_REGION_PAIRS)
    ranked = energy_scaled_depth_rows(depth, event, energy, orders, count,
                                      shower_frame=True, regions=RANK_REGIONS, pairs=SHOWER_REGION_PAIRS)
    return radial, rank_columns(ranked)
