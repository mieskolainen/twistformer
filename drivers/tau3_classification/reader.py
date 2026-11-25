# ALLEGRO ROOT file selection and calorimeter hit packing
#
# mikael.mieskolainen@cern.ch (2026)

from twistformer.logging import print
from dataclasses import dataclass
from typing import Dict, Tuple
import os
import time
import awkward as ak
import numpy as np
import uproot
from tqdm import tqdm

ECAL_BRANCHES_BY_MODE = {
    "sim": {
        "x": "ECalBarrelModuleThetaMerged.position.x",
        "y": "ECalBarrelModuleThetaMerged.position.y",
        "z": "ECalBarrelModuleThetaMerged.position.z",
        "e": "ECalBarrelModuleThetaMerged.energy",
    },
    "reco-cluster": {
        "x": "EMBCaloClusterCells/EMBCaloClusterCells.position.x",
        "y": "EMBCaloClusterCells/EMBCaloClusterCells.position.y",
        "z": "EMBCaloClusterCells/EMBCaloClusterCells.position.z",
        "e": "EMBCaloClusterCells/EMBCaloClusterCells.energy",
    },
    "reco-topocluster": {
        "x": "EMBCaloTopoClusterCells/EMBCaloTopoClusterCells.position.x",
        "y": "EMBCaloTopoClusterCells/EMBCaloTopoClusterCells.position.y",
        "z": "EMBCaloTopoClusterCells/EMBCaloTopoClusterCells.position.z",
        "e": "EMBCaloTopoClusterCells/EMBCaloTopoClusterCells.energy",
    },
}

def _ecal_vars_for_mode(input_mode: str) -> Tuple[str, str, str, str]:
    """
    Resolve calorimeter position and energy branches

    Args:
        input_mode:  sim, reco-cluster or reco-topocluster
    Returns:
        branches: Ordered x,y,z,E branch names
    """
    if input_mode not in ECAL_BRANCHES_BY_MODE:
        raise ValueError(f"input_mode must be one of {list(ECAL_BRANCHES_BY_MODE.keys())}")
    m = ECAL_BRANCHES_BY_MODE[input_mode]
    return m["x"], m["y"], m["z"], m["e"]

MC_PARTICLE_VARS = [
    "MCParticles/MCParticles.PDG",
    "MCParticles/MCParticles.generatorStatus",
    "MCParticles/MCParticles.simulatorStatus",
    "MCParticles/MCParticles.momentum.x",
    "MCParticles/MCParticles.momentum.y",
    "MCParticles/MCParticles.momentum.z",
]

MC_RELATION_VARS = [
    "MCParticles/MCParticles.parents_begin",
    "MCParticles/MCParticles.parents_end",
    "_MCParticles_parents/_MCParticles_parents.index",
]

def _eta_from_momentum(px: ak.Array, py: ak.Array, pz: ak.Array) -> ak.Array:
    """
    Compute particle pseudorapidity from jagged momenta

    Args:
        px:  Transverse x components
        py:  Transverse y components
        pz:  Longitudinal components
    Returns:
        eta: Jagged pseudorapidity array
    """
    px2 = px * px
    py2 = py * py
    pz2 = pz * pz
    p2 = px2 + py2 + pz2
    p = np.sqrt(p2)
    eps = 1e-9

    return 0.5 * np.log(((p + pz) + eps) / ((p - pz) + eps))

def _fiducial_event_mask_tau_decay_products(arrays: Dict[str, ak.Array], fiducial_eta_max: float) -> ak.Array:
    """
    Require visible generator tau descendants to lie inside eta acceptance

    Only particles outside acceptance need a parent scan. Flat NumPy buffers
    avoid per-event Awkward conversions and a shared visited set avoids
    repeated parent traversal within each event

    Args:
        arrays:  MC particle and parent-relation arrays
        fiducial_eta_max:  Maximum absolute pseudorapidity
    Returns:
        mask: Boolean event acceptance array
    """
    pdg_jag = arrays[MC_PARTICLE_VARS[0]]
    counts = ak.to_numpy(ak.num(pdg_jag, axis=1))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    pdg, gsta, ssta = (ak.to_numpy(ak.flatten(arrays[key])) for key in MC_PARTICLE_VARS[:3])
    eta = ak.to_numpy(ak.flatten(_eta_from_momentum(*(arrays[key] for key in MC_PARTICLE_VARS[3:]))))
    from_gen = ((ssta >> 30) & 1) == 0
    abs_pdg = np.abs(pdg)
    tau = from_gen & (abs_pdg == 15)
    bad = np.flatnonzero(
        from_gen
        & (gsta == 1)
        & (abs_pdg != 12)
        & (abs_pdg != 14)
        & (abs_pdg != 16)
        & (~np.isfinite(eta) | (np.abs(eta) >= fiducial_eta_max))
    )
    keep = np.ones(len(counts), dtype=bool)
    if not len(bad):
        return ak.Array(keep)

    par_b, par_e, parents = (ak.to_numpy(ak.flatten(arrays[key])) for key in MC_RELATION_VARS)
    relation_counts = ak.to_numpy(ak.num(arrays[MC_RELATION_VARS[2]], axis=1))
    relation_offsets = np.concatenate(([0], np.cumsum(relation_counts)))
    events = np.searchsorted(offsets[1:], bad, side="right")
    boundaries = np.flatnonzero(np.r_[True, events[1:] != events[:-1], True])
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        ev = int(events[left])
        start, stop = offsets[ev : ev + 2]
        if not tau[start:stop].any():
            continue
        relation_start, relation_stop = relation_offsets[ev : ev + 2]
        pending = bad[left:right].tolist()
        visited = set()
        while pending:
            i = pending.pop()
            if i in visited:
                continue
            visited.add(i)
            if tau[i]:
                keep[ev] = False
                break
            begin, end = int(par_b[i]), int(par_e[i])
            if not 0 <= begin <= end <= relation_stop - relation_start:
                raise ValueError("Invalid MC parent relation range")
            for parent in parents[relation_start + begin : relation_start + end]:
                if not 0 <= parent < stop - start:
                    raise ValueError("Invalid MC parent particle index")
                parent = int(start + parent)
                if from_gen[parent]:
                    pending.append(parent)
    return ak.Array(keep)

@dataclass
class Tau3Chunk:
    """Retained calorimeter hits and original ROOT entry numbers"""

    features: np.ndarray
    offsets: np.ndarray
    weights: np.ndarray
    entries: np.ndarray
    targets: np.ndarray | None = None

    def __len__(self):
        """
        Count selected events

        Returns:
            size: Number of selected events
        """
        return len(self.weights)

def _pack_hits(x, y, z, energy, max_hits, e_min):
    """
    Pack energy-ordered hits without event padding

    Args:
        x:  Jagged hit x coordinates
        y:  Jagged hit y coordinates
        z:  Jagged hit z coordinates
        energy:  Jagged hit energies
        max_hits:  Maximum retained hits per event
        e_min:  Minimum hit energy
    Returns:
        packed: Flat features, event offsets and counts before truncation
    """
    indices = ak.local_index(energy, axis=1)
    if e_min > 0:
        selected = energy >= e_min
        indices = indices[selected]
        energy = energy[selected]
    counts = ak.to_numpy(ak.num(energy, axis=1))
    order = ak.argsort(energy, axis=1, ascending=False, stable=True)[:, :max_hits]
    indices = indices[order]
    offsets = np.concatenate(([0], np.cumsum(np.minimum(counts, max_hits))))
    features = np.empty((int(offsets[-1]), 4), dtype=np.float32)
    for column, values in enumerate((x, y, z)):
        features[:, column] = ak.to_numpy(ak.flatten(values[indices]))
    features[:, 3] = ak.to_numpy(ak.flatten(energy[order]))
    return features, offsets, counts

@dataclass
class _HitStats:
    """Calorimeter multiplicity moments before hit truncation"""

    events: int = 0
    total: float = 0.0
    square: float = 0.0
    minimum: float = np.inf
    maximum: int = 0

    def update(self, counts):
        """
        Accumulate event multiplicity moments

        Args:
            counts: Hit counts per event
        """
        if not len(counts):
            return
        values = np.asarray(counts, dtype=np.float64)
        self.events += len(values)
        self.total += float(values.sum())
        self.square += float(np.dot(values, values))
        self.minimum = min(self.minimum, int(values.min()))
        self.maximum = max(self.maximum, int(values.max()))

    def report(self, path, selection):
        """
        Print calorimeter multiplicity statistics

        Args:
            path:  ROOT input path
            selection:  Event and hit selection label
        """
        if not self.events:
            return
        mean = self.total / self.events
        std = np.sqrt(max(0.0, self.square / self.events - mean * mean))
        print(
            f"Hit-count stats {selection} for '{os.path.basename(path)}' "
            f"(events={self.events}): mean={mean:.2f}, std={std:.2f}, "
            f"min={self.minimum}, max={self.maximum}"
        )

def iter_tau3(
    path: str,
    tree_name: str,
    *,
    max_hits: int,
    max_events: int = 1_000_000_000,
    chunk_size: int = 40_000,
    e_min: float = 0.0,
    fiducial_eta_max: float = 1.6,
    input_mode: str = "sim",
    weight_branch: str | None = None,
    target_branches: tuple[str, ...] = (),
    target_builder=None,
    apply_fiducial: bool = True,
):
    """
    Stream selected calorimeter events into compact hit chunks

    The raw event limit precedes selection. Source units and ROOT entry
    numbers are preserved. Empty selected events remain in the chunks

    Args:
        path:  Input ROOT file
        tree_name:  ROOT event tree name
        max_hits:  Maximum retained hits per event
        max_events:  Maximum raw entries before selection
        chunk_size:  Raw entries per chunk
        e_min:  Minimum hit energy
        fiducial_eta_max:  Visible tau daughter pseudorapidity acceptance
        input_mode:  Calorimeter collection mode
        weight_branch:  Scalar signed event weights or unit weights
        target_branches:  Extra truth branches for vector regression
        target_builder:  Callable mapping selected arrays and entry IDs to targets
        apply_fiducial:  Apply MC acceptance, disabled for hit-only inference
    Yields:
        chunk: Flat retained hits, offsets, weights and ROOT entry numbers
    """
    for name, value in (("max_hits", max_hits), ("max_events", max_events), ("chunk_size", chunk_size)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(e_min) or e_min < 0:
        raise ValueError("e_min must be finite and nonnegative")
    if not np.isfinite(fiducial_eta_max) or fiducial_eta_max <= 0:
        raise ValueError("fiducial_eta_max must be finite and positive")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    branches = _ecal_vars_for_mode(input_mode)
    varlist = list(
        dict.fromkeys(
            list(branches)
            + (MC_PARTICLE_VARS + MC_RELATION_VARS if apply_fiducial else [])
            + list(target_branches)
            + ([weight_branch] if weight_branch is not None else [])
        )
    )
    before, after = _HitStats(), _HitStats()
    read_seconds = select_seconds = pack_seconds = 0.0
    processed = kept = 0
    # ROOT files are mounted locally, including EOS paths
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as root:
        tree = root[tree_name]
        target = min(tree.num_entries, max_events)
        print(
            f"Start streaming '{os.path.basename(path)}' "
            f"(tree='{tree_name}', mode='{input_mode}', max_events={target}, chunk_size={chunk_size})"
        )
        chunks = iter(tree.iterate(varlist, entry_stop=target, step_size=chunk_size, library="ak"))
        with tqdm(total=(target + chunk_size - 1) // chunk_size, desc=f"Reading {os.path.basename(path)}", leave=False) as progress:
            while processed < target:
                tick = time.perf_counter()
                arrays = next(chunks)
                read_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                size = len(arrays)
                weights = (
                    ak.to_numpy(arrays[weight_branch], allow_missing=False).astype(np.float64, copy=False)
                    if weight_branch is not None
                    else np.ones(size, dtype=np.float64)
                )
                if weights.shape != (size,) or not np.isfinite(weights).all():
                    raise ValueError("Event weights must be finite scalars")
                before.update(ak.to_numpy(ak.num(arrays[branches[0]], axis=1)))
                mask = (
                    _fiducial_event_mask_tau_decay_products(arrays, fiducial_eta_max)
                    if apply_fiducial
                    else ak.Array(np.ones(size, dtype=bool))
                )
                selected = ak.to_numpy(mask)
                entries = np.flatnonzero(selected) + processed
                try:
                    targets = target_builder(arrays[mask], entries) if target_builder else None
                except ValueError as error:
                    raise ValueError(f"Truth target selection failed in ROOT file {path!r}, tree {tree_name!r}: {error}") from error
                select_seconds += time.perf_counter() - tick
                tick = time.perf_counter()
                features, offsets, counts = _pack_hits(*(arrays[key][mask] for key in branches), max_hits, e_min)
                after.update(counts)
                chunk = Tau3Chunk(features, offsets, weights[selected], entries, targets)
                kept += len(chunk)
                processed += size
                pack_seconds += time.perf_counter() - tick
                progress.update(1)
                # Release ROOT buffers before the next chunk is decoded
                del arrays, weights, mask, selected
                if len(chunk):
                    yield chunk
                del chunk, features, offsets, counts, entries
    before.report(path, "BEFORE fiducial / energy cuts")
    after.report(path, "AFTER fiducial + energy cuts")
    print(
        f"Finished '{os.path.basename(path)}' (mode='{input_mode}'): "
        f"kept_events={kept} / {target}, "
        f"read={read_seconds:.2f}s, select={select_seconds:.2f}s, pack={pack_seconds:.2f}s"
    )

def read_tau3(path: str, tree_name: str, *, max_hits: int, **options):
    """
    Materialize padded arrays for callers requiring the dense reader API

    Training uses iter_tau3 directly and keeps only compact hit chunks

    Args:
        path:  Input ROOT file
        tree_name:  ROOT event tree name
        max_hits:  Maximum retained hits per event
        options:  Selection and streaming arguments for iter_tau3
    Returns:
        arrays: Features (N,max_hits,4), Boolean hit mask and event weights
    """
    chunks = list(iter_tau3(path, tree_name, max_hits=max_hits, **options))
    size = sum(len(chunk) for chunk in chunks)
    features = np.zeros((size, max_hits, 4), dtype=np.float32)
    mask = np.zeros((size, max_hits), dtype=bool)
    weights = np.empty(size, dtype=np.float64)
    event = 0
    for chunk in chunks:
        for start, stop in zip(chunk.offsets[:-1], chunk.offsets[1:]):
            count = stop - start
            features[event, :count] = chunk.features[start:stop]
            mask[event, :count] = True
            event += 1
        weights[event - len(chunk) : event] = chunk.weights
    return features, mask, weights
