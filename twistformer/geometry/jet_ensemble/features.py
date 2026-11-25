# Jet observables for the frozen constituent ensemble
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import torch
from torch.nn.functional import pad

from ..lund_graphs import jet_rows, MassHypothesis
from . import branches, decays, density, histories, spectra
from .kinematics import Event, pt


def overlap(event):
    """Measure shared transverse momentum across four three-prong assignments"""
    weight = pt(event.p)
    labels = []
    for groups in event.groups:
        label = torch.full_like(weight, 3, dtype=torch.long)
        for j, members in enumerate(groups):
            label[members] = j
        labels.append(label)
    return torch.stack(
        [weight.new_zeros(16).scatter_add_(0, 4 * a + b, weight) for a, b in combinations(labels, 2)]
    ).reshape(6, 4, 4) / weight.sum().clamp_min(1.0e-12)


MEASUREMENTS = dict(
    history=histories.history,
    candidate=decays.candidate,
    decay=decays.decay,
    microjets=histories.microjets,
    antenna=decays.antenna,
    shower=branches.shower,
    particle_cascade=branches.particle_cascade,
    density_tree=density.density_tree,
    overlap=overlap,
    morphology=spectra.morphology,
    multipole=spectra.multipole,
    radon=spectra.radon,
    diffusion=spectra.diffusion,
    resolution=spectra.resolution,
)


WIDTHS = dict(
    history=84,
    candidate=(12, 116),
    decay=376,
    antenna=496,
    shower=(2, 32, 52),
    density_tree=252,
    overlap=(6, 4, 4),
    morphology=1062,
    multipole=1148,
    radon=594,
    diffusion=640,
    resolution=364,
    pool=496,
)


class JetMeasurements:
    """Share clustering histories between measurements of the same batch"""

    def __init__(self, momentum, partitions=None):
        """
        Prepare one batch of measured jets

        Args:
            momentum:    Padded (jets, constituents, 4) tensor in descending transverse momentum order
            partitions:  Optional pair of fitted tree dictionaries for the pooled coordinates
        """
        self.momentum = momentum
        self.partitions = partitions
        physical = (momentum[..., 0] > 0) & (momentum[..., 1:3] != 0).any(-1)
        self.events = [Event(row[mask].double()) for row, mask in zip(momentum, physical)]
        self.lund = None

    def measure(self, family):
        """
        Measure the requested family on the constituent device

        Args:
            family:  Frozen measurement name
        Returns:
            rows:  Tensor or tuple of tensors in checkpoint order
        """
        if family in ("tree", "flow"):
            if self.lund is None:
                self.lund = jet_rows(self.momentum, mass_hypothesis=MassHypothesis(172.5, 80.4, 25.0, 15.0))
            return self.lund[:, :532] if family == "tree" else self.lund[:, 532:]
        if family not in MEASUREMENTS and family != "pool":
            raise ValueError(f"Unknown jet measurement {family}")
        if not self.events:
            p = self.momentum
            if family == "microjets":
                return (
                    p.new_empty((0, 75, 40), dtype=torch.float32),
                    p.new_empty((0, 75), dtype=torch.bool),
                    p.new_empty((0, 75, 4), dtype=torch.float32),
                )
            shape = (p.size(1), 120) if family == "particle_cascade" else WIDTHS[family]
            shape = (shape,) if isinstance(shape, int) else shape
            return p.new_empty((0, *shape), dtype=torch.float32)
        result = []
        for event in self.events:
            if not len(event.p) and family not in ("microjets", "particle_cascade"):
                shape = WIDTHS[family]
                row = event.p.new_zeros((shape,) if isinstance(shape, int) else shape)
            elif not len(event.p) and family == "particle_cascade":
                row = event.p.new_empty((0, 120))
            elif family == "pool":
                if self.partitions is None:
                    raise ValueError("Supply fitted partition dictionaries for pooled observables")
                row = branches.pool(event, self.partitions)
            else:
                row = MEASUREMENTS[family](event)
            if family == "particle_cascade":
                row = pad(row, (0, 0, 0, self.momentum.size(1) - len(row)))
            result.append(row)
        if family == "microjets":
            rows, valid, probability = zip(*result)
            return torch.stack(rows).float(), torch.stack(valid), torch.stack(probability).float()
        return torch.stack(result).float()
