# Transverse momentum flow in jet angular coordinates
#
# mikael.mieskolainen@cern.ch (2026)

from functools import lru_cache

import torch
from torch import nn

from .lund_graphs import DECAY_NAMES, LORENTZ_NAMES, LUND_NAMES, TOP_DECAY, native_top_rows
from twistformer.geometry.lund_graphs import jet_kernel
from twistformer.geometry.native_lund import cpu_workers
from .prong_graphs import PRONG_NAMES, RADII, prong_flow, resolved_prongs

from twistformer.geometry.graph_polynomials import (
    GRAPHS, GRAPH_NAMES, SphericalGraphs, point_tensor, projected_graph_rows, squared_distance_graphs,
)
from twistformer.geometry.moment_graphs import MOMENT_NAMES, jet_angular_flow, moment_rows


SCALES = (0.05, 0.1, 0.2, 0.4)
# Angular wave numbers in the eta and phi projections
FREQUENCIES = (2.5, 5., 10.)
HARMONICS = tuple(f"{graph}_{axis}_k{k:g}" for graph, frequencies in
                  (("power", FREQUENCIES), ("closure_re", FREQUENCIES[:-1]))
                  for k in frequencies for axis in ("eta", "phi"))
SUMMARY = ("constituent_count", "jet_pt", "jet_mass", "leading_pt_fraction", "sum_pt_fraction_squared")
PROJECTED = tuple(f"{axis}_{scale}_{graph}" for axis in ("eta", "phi") for scale in SCALES for graph in ("edge", "wedge"))
BASES = {
    "harmonic_graphs": SUMMARY + GRAPHS + HARMONICS,
    "local_graphs": SUMMARY + GRAPHS + HARMONICS + PROJECTED,
    "complete_graphs": SUMMARY + GRAPHS + HARMONICS + PROJECTED + GRAPH_NAMES,
}
BASES["prong_graphs"] = BASES["complete_graphs"] + PRONG_NAMES
BASES["moment_graphs"] = BASES["prong_graphs"] + MOMENT_NAMES
BASES["lund_graphs"] = BASES["prong_graphs"] + LUND_NAMES
BASES["rest_hypergraphs"] = DECAY_NAMES
BASES["lorentz_hypergraphs"] = LORENTZ_NAMES


def jet_descriptors(p4, basis):
    """
    Contract jet graphs with transverse momentum fractions at the vertices

    Args:
        p4:  Selected constituent four-momenta (E, px, py, pz) in GeV
        basis:  Jet graph basis in BASES
    Returns:
        result:  Float32 tensor in the declared feature order
    """
    momentum = point_tensor(p4)
    counts = torch.tensor([len(momentum)], device=momentum.device)
    if not len(momentum):
        momentum = momentum.new_zeros((1, 4))
    return jet_graph_kernel(basis, str(momentum.device), False)(momentum.unsqueeze(0), counts)[0]


class JetGraphKernel(nn.Module):
    def __init__(self, basis, graph_features=()):
        """
        Select the contractions for a fixed jet feature schema

        Args:
            basis:           Top-tag graph basis from the steering card
            graph_features:  Requested columns in model order
        """
        super().__init__()
        if basis not in BASES:
            raise ValueError(f"Unknown jet graph basis: {basis}")
        requested = graph_features or BASES[basis]
        self.lund = basis == "lund_graphs"
        self.native = basis in ("rest_hypergraphs", "lorentz_hypergraphs")
        if self.native:
            self.features = tuple(requested)
            return
        if self.lund:
            prefix = tuple(name for name in requested if name in BASES["prong_graphs"])
            self.lund_prongs = JetGraphKernel("prong_graphs", prefix) if prefix else None
            self.lund_features = jet_kernel(mass_hypothesis=TOP_DECAY) if len(prefix) < len(requested) else None
            layout = prefix + (LUND_NAMES if self.lund_features else ())
            self._select_columns(layout, requested)
            return
        self.moments = basis == "moment_graphs" and any(name in MOMENT_NAMES for name in requested)
        self.prongs = any(name in PRONG_NAMES for name in requested)
        self.local = any(name in PROJECTED for name in requested)
        self.radii = tuple(r for r in RADII if any(name.startswith(f"flow_r{r:g}_") for name in requested))
        self.angular = any(name.startswith("angular_beta") for name in requested)
        spherical = tuple(i for i, name in enumerate(GRAPH_NAMES) if name in requested)
        self.spherical = SphericalGraphs(spherical if len(spherical) < len(GRAPH_NAMES) else None) if spherical else None
        layout = SUMMARY + GRAPHS + HARMONICS
        if self.local:
            layout += PROJECTED
        layout += tuple(GRAPH_NAMES[i] for i in spherical)
        if self.prongs:
            prefixes = tuple(f"flow_r{r:g}_" for r in self.radii)
            layout += tuple(name for name in PRONG_NAMES
                            if (not name.startswith("flow_r") or name.startswith(prefixes))
                            and (self.angular or not name.startswith("angular_beta")))
        if self.moments:
            layout += MOMENT_NAMES
        self._select_columns(layout, requested)
        self.register_buffer("frequencies", torch.tensor(FREQUENCIES, dtype=torch.float64))
        self.register_buffer("scales", torch.tensor(SCALES, dtype=torch.float64))

    def _select_columns(self, layout, requested):
        """Map the computed columns to the model input order"""
        columns = None if layout == requested else torch.tensor([layout.index(name) for name in requested])
        self.register_buffer("columns", columns)

    def forward(self, p4, counts, cluster_width=None, projection_batch=None):
        """
        Contract momentum flow for a batch of jets

        Args:
            p4:                Selected constituent four-momenta with zero padding
            counts:            Selected constituent counts including zero transverse momentum
            cluster_width:     Occupied prefix of a batch ordered by transverse momentum
            projection_batch:  Jet count per dense feature projection
        Returns:
            rows:  Float32 jet feature rows in steering order
        """
        if self.native:
            with native_top_rows(self.features, cpu_workers()) as rows:
                return rows(p4.cpu()).to(p4.device)
        if self.lund:
            blocks = []
            if self.lund_prongs is not None:
                # The dataset supplies constituents in descending transverse momentum
                leading = p4
                if cluster_width is not None:
                    leading = p4[:, :80]
                elif p4.size(1) > 80:
                    order = p4[..., 1:3].square().sum(-1).argsort(dim=1, descending=True, stable=True)[:, :80]
                    leading = p4.gather(1, order[..., None].expand(-1, -1, 4))
                size = projection_batch or max(1, len(p4))
                blocks.append(torch.cat([self.lund_prongs(p.contiguous(), n.clamp_max(80))
                                         for p, n in zip(leading.split(size), counts.split(size))]))
            if self.lund_features is not None:
                blocks.append(self.lund_features(p4, cluster_width, projection_batch))
            rows = torch.cat(blocks, dim=1)
            return rows if self.columns is None else rows.index_select(1, self.columns)
        fields, points, w = self.dense_rows(p4, counts)
        if self.prongs:
            fields = torch.cat((fields, resolved_prongs(p4.double(), points, w)), 1)
        if self.moments:
            fields = torch.cat((fields, moment_rows(points, w)), 1)
        rows = fields.float()
        return rows if self.columns is None else rows.index_select(1, self.columns)

    def dense_rows(self, p4, counts):
        """Contract the continuous jet observables before discrete subjet assignment"""
        p = p4.double()
        total = p.sum(1)
        energy, px, py, _ = total.unbind(1)
        mass = (energy.square() - total[:, 1:].square().sum(1)).clamp_min(0).sqrt()
        points, w, scalar_pt = jet_angular_flow(p, px, py)
        phase = points[:, :, None, :] * self.frequencies[None, None, :, None]
        real = (w[:, :, None, None] * phase.cos()).sum(1)
        imag = (w[:, :, None, None] * phase.sin()).sum(1)
        # Real part of F(k)^2 times the conjugate of F(2k)
        closure = ((real[:, :-1].square() - imag[:, :-1].square()) * real[:, 1:]
                   + 2 * real[:, :-1] * imag[:, :-1] * imag[:, 1:])
        summary = torch.stack((counts, torch.hypot(px, py), mass, w.amax(1), w.square().sum(1)), 1)
        fields = [summary, squared_distance_graphs(points, w), (real.square() + imag.square()).flatten(1), closure.flatten(1)]
        if self.local:
            # Sort both angular projections in one batch
            coordinates = points.transpose(1, 2).reshape(-1, p.shape[1])
            weights = w[:, None].expand(-1, 2, -1).reshape_as(coordinates)
            projections = projected_graph_rows(coordinates, weights, self.scales)
            fields.append(projections.reshape(p.shape[0], -1))
        if self.spherical is not None:
            graphs = self.spherical(points / .4, w)
            fields.append(torch.where(scalar_pt > 0, graphs, 0.))
        if self.prongs:
            fields.append(prong_flow(points, w, self.radii, self.angular))
        return torch.cat(fields, 1), points, w


@lru_cache(maxsize=8)
def jet_graph_kernel(basis, device, compile_graphs, graph_features=()):
    """
    Reuse the graph kernel and its device-resident catalogue across data splits

    Args:
        basis:           Top-tag graph basis
        device:          CPU or resolved CUDA device
        compile_graphs:  Compile the tensor contractions with TorchInductor
        graph_features:  Optional tuple of requested feature names in model order
    Returns:
        kernel:  Callable on a padded jet batch
    """
    kernel = JetGraphKernel(basis, graph_features).to(device)
    if kernel.native or not compile_graphs:
        return kernel
    if kernel.lund:
        if kernel.lund_features is not None:
            kernel.lund_features = jet_kernel(mass_hypothesis=TOP_DECAY, compile_graphs=True)
        if kernel.lund_prongs is not None:
            kernel.lund_prongs.dense_rows = torch.compile(kernel.lund_prongs.dense_rows, fullgraph=True, dynamic=False)
        return kernel
    return torch.compile(kernel, fullgraph=True, dynamic=False)
