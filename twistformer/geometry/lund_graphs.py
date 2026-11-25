# Lund splittings, angular flow and exclusive subjet energy correlations
#
# mikael.mieskolainen@cern.ch (2026)

from functools import lru_cache
from math import isfinite
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from .jet_clustering import JetTree, cluster
from .jet_flow import flow_rows
from .jet_history import history_rows


class MassHypothesis(NamedTuple):
    """Parent mass, pair mass and their mass resolutions in GeV"""

    parent_mass: float
    pair_mass: float
    parent_width: float
    pair_width: float


DECAY_FIELDS = ("parent_mass", "pair_mass", "pair_pt_fraction", "parent_pt_fraction", "min_cross_pair_mass")
BLOCKS = {"ca": 1, "kt": 2, "flow": 4, "all": 7}


def history_names(algorithm, decay_fields=()):
    """
    Name the observables from one exclusive clustering tree

    Args:
        algorithm:     Cambridge Aachen or longitudinally invariant kT
        decay_fields:  Column names for the optional pair and parent mass fit
    Returns:
        names:  Features in output order
    """
    names = []
    for rank in range(1, 8):
        names.extend(f"split{rank}_{field}" for field in (
            "kt", "angle", "z", "mass", "hard_mass", "soft_mass", "pt_fraction",
            "dphi", "dy", "soft_count",
        ))
    for cut in (.03, .08, .15, .25):
        names.extend(f"zcut{cut:g}_{field}" for field in ("mass", "angle", "pt_fraction", "count"))
    for count in (2, 3, 4, 5, 6, 8):
        prefix = f"n{count}_"
        names.extend(prefix + field for field in ("tau1", "tau2", "tau_half", "tau1_ratio"))
        if count in (2, 3, 5):
            for rank in range(1, count + 1):
                names.extend(f"{prefix}axis{rank}_{field}" for field in (
                    "pt_fraction", "mass", "count", "leading_fraction", "pt_fraction_squared",
                    "width", "dy", "dphi",
                ))
            for i in range(1, count + 1):
                for j in range(i + 1, count + 1):
                    names.extend(f"{prefix}pair{i}{j}_{field}" for field in ("mass", "mass_fraction", "angle", "kt"))
        if count >= 4:
            names.extend(prefix + field for field in decay_fields)
    return tuple(f"lund_{algorithm}_{name}" for name in names)


def flow_names():
    """
    Name the constituent and exclusive microjet flow moments

    Returns:
        names:  Features in output order
    """
    names = ["abs_rapidity", "pt", "mass", "count"]
    for cut in (.5, 1, 2, 4, 8, 16):
        names.extend(f"pt{cut:g}_{field}" for field in ("count", "share"))
    for index in range(14):
        names.extend(f"angular_bin{index}_{field}" for field in ("pt_pt", "pt_count", "count_count"))
    for radius in (.01, .025, .05, .075, .1, .15, .25, .4, .65):
        names.extend(f"r{radius:g}_{field}" for field in (
            "isolated_count", "isolated_share", "inverse_occupancy", "max_share", "star2",
            "star3", "pt_count", "pt_count_squared",
        ))
    for beta in (.5, 1, 2):
        names.extend(f"beta{beta:g}_{field}" for field in (
            "ecf2", "ecf3", "ecf4", "ecf5", "min3", "min4", "min5", "min4_squared",
            "d2", "c3", "c4", "min4_over_min3", "min5_over_min4",
        ))
    for count in (4, 8, 16):
        names.extend(f"rest_n{count}_fox_wolfram{order}" for order in range(2, 11))
        names.extend(f"rest_n{count}_energy_fraction{rank}" for rank in range(1, 5))
    for rank in range(1, 4):
        prefix = f"prong{rank}_"
        for field in ("share", "count"):
            names.extend(f"{prefix}r{r:g}_{field}" for r in (.025, .05, .075, .1, .15, .25))
        for field in ("share", "count"):
            names.extend(f"{prefix}pt{cut}_{field}" for cut in (1, 2, 4, 8, 16))
        names.extend(prefix + field for field in (
            "sqrt_share", "share_squared", "share_cubed", "width_half", "width_squared",
            "pull_y", "pull_phi", "leading_share", "pull_to_next", "pull_to_previous",
        ))
    return tuple(f"lund_flow_{name}" for name in names)


@lru_cache
def jet_feature_names(component="all", mass_hypothesis=None):
    """
    Name the columns returned by jet_rows in output order

    Args:
        component:        ca, kt, flow or all
        mass_hypothesis:  Optional pair and parent mass fit
    Returns:
        names:  Feature names in output order
    """
    if component not in BLOCKS:
        raise ValueError("Expected ca, kt, flow or all")
    fields = DECAY_FIELDS if mass_hypothesis is not None else ()
    if component == "all":
        return history_names("ca", fields) + history_names("kt", fields) + flow_names()
    if component == "flow":
        return flow_names()
    return history_names(component, fields)


def _projection_tree(tree, rows, padding):
    """Restore the dense projection layout after clustering a compact jet batch"""
    if rows.start == 0 and rows.stop >= tree.splits.size(0) and not padding:
        return tree
    return JetTree(
        {n: axes[rows].contiguous() for n, axes in tree.axes.items()},
        {n: (F.pad(owners[rows], (0, padding), value=-1) if padding else owners[rows]) for n, owners in tree.owners.items()},
        F.pad(tree.splits[rows], (0, 0, 0, padding)) if padding else tree.splits[rows],
        tree.groomed[rows].contiguous(),
    )


@lru_cache
def jet_kernel(component="all", mass_hypothesis=None, compile_graphs=False):
    """
    Bind PyTorch jet observables to a fixed column schema

    Args:
        component:        ca, kt, flow or all
        mass_hypothesis:  Optional MassHypothesis in GeV
        compile_graphs:   Compile the dense feature projections
    Returns:
        kernel:  Batch function for energy-first momentum tensors
    """
    jet_feature_names(component, mass_hypothesis)
    if mass_hypothesis is not None:
        if len(mass_hypothesis) != 4 or not all(isfinite(value) for value in mass_hypothesis):
            raise ValueError("Expected two finite target masses and two finite widths in GeV")
        if min(mass_hypothesis[:2]) < 0 or min(mass_hypothesis[2:]) <= 0:
            raise ValueError("Target masses must be nonnegative and mass widths must be positive")
    history = torch.compile(history_rows, fullgraph=True) if compile_graphs else history_rows
    flow = torch.compile(flow_rows, fullgraph=True) if compile_graphs else flow_rows

    def kernel(momenta, cluster_width=None, projection_batch=None):
        """
        Compute the selected rows on the input device

        Args:
            momenta:           Tensor with shape (jets, particles, 4)
            cluster_width:     Occupied prefix length from the packed dataset
            projection_batch:  Maximum number of jets per dense projection
        Returns:
            rows:  Float32 matrix in jet_feature_names order
        """
        if not len(momenta):
            return momenta.new_empty((0, len(jet_feature_names(component, mass_hypothesis))), dtype=torch.float32)
        p = momenta[:, :cluster_width].float().double()
        valid = (p[:, :, 0] > 0) & (p[:, :, 1:3] != 0).any(2)
        order = valid.long().argsort(dim=1, descending=True, stable=True)
        if p.device.type == "cpu" and not compile_graphs:
            order = order[:, :int(valid.sum(1).max())]
        valid = valid.gather(1, order)
        p = p.gather(1, order[:, :, None].expand(-1, -1, 4))
        p = torch.where(valid[:, :, None], p, 0.)
        padding = max(0, 16 - p.size(1))
        p = F.pad(p, (0, 0, 0, padding))
        valid = F.pad(valid, (0, padding))
        ca = cluster(p, valid, False, True) if component in ("ca", "all") else None
        kt = cluster(p, valid, True, component != "flow") if component != "ca" else None
        # Two fixed projection widths cover the 200-particle GPU dataset
        dense_width = momenta.size(1)
        if p.is_cuda and cluster_width is not None and cluster_width <= 80:
            dense_width = min(dense_width, 80)
        extra = max(0, dense_width - p.size(1)) if compile_graphs and cluster_width is not None else 0
        size = projection_batch or len(p)
        result = []
        for first in range(0, len(p), size):
            rows = slice(first, first + size)
            q = F.pad(p[rows], (0, 0, 0, extra)) if extra else p[rows]
            mask = F.pad(valid[rows], (0, extra)) if extra else valid[rows]
            blocks = []
            if ca is not None:
                blocks.append(history(q, mask, _projection_tree(ca, rows, extra), mass_hypothesis))
            if kt is not None:
                tree = _projection_tree(kt, rows, extra)
                if component != "flow":
                    blocks.append(history(q, mask, tree, mass_hypothesis))
                if component in ("flow", "all"):
                    blocks.append(flow(q, mask, tree))
            result.append(torch.cat(blocks, 1).float())
        return torch.cat(result)

    return kernel


def jet_rows(momenta, component="all", mass_hypothesis=None):
    """
    Measure exclusive clustering histories and angular flow

    Args:
        momenta:          NumPy array or Torch tensor in E, px, py, pz order in GeV
        component:        ca, kt, flow or all
        mass_hypothesis:  Optional MassHypothesis in GeV
    Returns:
        rows:  Float32 tensor in jet_feature_names order
    """
    p = momenta if torch.is_tensor(momenta) else torch.from_numpy(np.ascontiguousarray(momenta, dtype=np.float32))
    if p.ndim != 3 or p.size(-1) != 4 or not torch.isfinite(p).all():
        raise ValueError("Expected finite jet momenta with shape (events, particles, 4)")
    kernel = jet_kernel(component, mass_hypothesis)
    rows = torch.cat([kernel(batch) for batch in p.split(64)])
    if not torch.isfinite(rows).all():
        raise ValueError("Nonfinite jet observable")
    return rows
