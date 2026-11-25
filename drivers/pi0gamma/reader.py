# ROOT file reader
#
# mikael.mieskolainen@cern.ch (2026)

from contextlib import nullcontext
from dataclasses import dataclass
from glob import glob
from pathlib import Path
import os
import time
import numpy as np
import torch
from twistformer.logging import print
from twistformer.data.cutflow import Cutflow
from .features import cartesian_features, cell_features, angular_features
from .graphs.descriptors import BASES, RADIAL_SHOWER_BASES, SHOWER_FRAME_BASES, descriptor_rows
from .graphs.graph_inputs import shower_frame_hits

def source_paths(files, base_path=""):
    """
    Expand ordered ROOT files, directories and local file patterns

    Args:
        files:  Input paths or patterns
        base_path:  Shared directory for relative file entries
    Returns:
        paths: Unique ROOT paths in stable order
    """
    paths = []
    for value in files:
        if base_path and not Path(value).is_absolute() and "://" not in value:
            value = f"{base_path.rstrip('/')}/{value}"
        if "://" in value:
            found = [value]
        elif Path(value).is_dir():
            found = sorted(str(path.resolve()) for path in Path(value).rglob("*.root"))
        else:
            found = sorted(str(Path(path).resolve()) for path in glob(value))
        if not found:
            raise FileNotFoundError(f"No ROOT files found at {value}")
        paths.extend(found)
    return list(dict.fromkeys(paths))


def resolve_branches(tree, options, labeled):
    """
    Resolve split PODIO branches with optional explicit branch names

    Args:
        tree:  Uproot event tree
        options:  Driver reader options
        labeled:  Read MC labels and evaluation observables
    Returns:
        branches: Logical names mapped to ROOT branches
    """
    c, h, m = options.clusters, options.cell_collection, options.mc_particles
    relation = options.cluster_hits or f"_{c}_hits"
    fields = {
        "cluster_energy": (c, "energy"),
        "cell_energy": (h, "energy"),
        **{f"cell_{axis}": (h, f"position.{axis}") for axis in "xyz"},
    }
    if options.input_mode == "leading_cluster_cone":
        fields.update({f"cluster_{axis}": (c, f"position.{axis}") for axis in "xyz"})
    else:
        fields.update({"hits_begin": (c, "hits_begin"), "hits_end": (c, "hits_end"),
                       "hit_index": (relation, "index")})
    if options.feature_mode == "extended":
        fields["cell_id"] = (h, "cellID")
    if labeled:
        fields.update({
            "pdg": (m, "PDG"), "status": (m, "generatorStatus"),
            "parents_begin": (m, "parents_begin"), "parents_end": (m, "parents_end"),
            "mass": (m, "mass"),
            **{f"mc_{axis}": (m, f"momentum.{axis}") for axis in "xyz"},
        })
        fields.update(daughters_begin=(m, "daughters_begin"), daughters_end=(m, "daughters_end"),
                      daughter_index=(f"_{m}_daughters", "index"))
        if options.truth_links:
            fields["parent_index"] = (f"_{m}_parents", "index")
            for name in ("link_cell", "link_particle", "link_weight"):
                if name not in options.branches:
                    raise ValueError(f"truth_links requires driver.branches.{name}")
                fields[name] = ("", "")
    result = {}
    keys = set(tree.keys())
    for name, (collection, field) in fields.items():
        candidates = [options.branches[name]] if name in options.branches else [
            f"{collection}/{collection}.{field}", f"{collection}.{field}",
        ]
        found = next((key for key in candidates if key in keys), None)
        if found is None:
            # Photon-only files can omit MC daughter relations
            if name in {"daughters_begin", "daughters_end", "daughter_index"} and name not in options.branches:
                continue
            raise ValueError(f"Missing ROOT branch for {name}, tried {candidates}")
        result[name] = found
    for index, collection in (
        ("hit_index", "hit_collection"), ("parent_index", "parent_collection"), ("daughter_index", "daughter_collection"),
        ("link_cell", "link_cell_collection"), ("link_particle", "link_particle_collection"),
    ):
        if index in result and result[index].endswith(".index"):
            key = result[index][:-len(".index")] + ".collectionID"
            if key in keys:
                result[collection] = key
    return result


def collection_ids(root, tree_name):
    """
    Read PODIO collection identifiers without opening event truth branches

    Args:
        root:  Open ROOT file
        tree_name:  Event tree name
    Returns:
        ids: Collection names mapped to integer IDs, empty without PODIO metadata
    """
    if "podio_metadata" not in root:
        return {}
    metadata = root["podio_metadata"]
    prefix = f"{tree_name}___CollectionTypeInfo"
    name_key, id_key = f"{prefix}/{prefix}.name", f"{prefix}/{prefix}.collectionID"
    arrays = metadata.arrays([name_key, id_key], library="ak", how=dict)
    names, ids = arrays[name_key][0], arrays[id_key][0]
    return {str(name): int(value) for name, value in zip(names, ids, strict=True)}


def validate_relation_collection(event, field, expected, ids, indices=None):
    """
    Reject relation indices that refer to a different PODIO collection

    Args:
        event:  Event arrays including optional relation collection IDs
        field:  Logical collection ID branch name
        expected:  Required destination collection name
        ids:  PODIO collection identifiers
        indices:  Optional relation slice for the selected cluster
    """
    if not ids:
        return
    if field not in event or expected not in ids:
        raise ValueError(f"Cannot verify {field} against PODIO collection {expected}")
    index_field = {
        "hit_collection": "hit_index", "parent_collection": "parent_index", "daughter_collection": "daughter_index",
        "link_cell_collection": "link_cell", "link_particle_collection": "link_particle",
    }[field]
    if event[field].shape != event[index_field].shape:
        raise ValueError(f"{field} indices and collection IDs must align")
    values = event[field] if indices is None else event[field][indices]
    if (values != ids[expected]).any():
        raise ValueError(f"{field} must reference {expected}, found collection IDs {np.unique(values).tolist()}")


def pi0_daughters(event, primary):
    """
    Resolve the direct MC daughters of the selected pion

    Args:
        event:  MC particle arrays and daughter relations
        primary:  Index of the associated neutral pion
    Returns:
        daughters:  Indices in the MC particle collection
    """
    if not {"daughters_begin", "daughters_end", "daughter_index"} <= event.keys():
        raise ValueError("Pion truth selection requires MC daughter relations for neutral pions")
    begin, end = event["daughters_begin"], event["daughters_end"]
    pdg, indices = event["pdg"], event["daughter_index"]
    if begin.shape != pdg.shape or end.shape != pdg.shape:
        raise ValueError("MC daughter ranges must align with the particle collection")
    start, stop = int(begin[primary]), int(end[primary])
    if not 0 <= start <= stop <= len(indices):
        raise ValueError("Invalid MC daughter relation range")
    daughters = indices[start:stop]
    if ((daughters < 0) | (daughters >= len(pdg))).any() or len(np.unique(daughters)) != len(daughters):
        raise ValueError("Invalid MC daughter particle indices")
    return daughters


def pi0_decay_asymmetry(event, primary, daughters=None):
    """
    Measure the lab energy asymmetry of two direct photon daughters

    Args:
        event:  MC particle arrays
        primary:  Associated pion index
        daughters:  Validated direct daughter indices when available
    Returns:
        asymmetry:  Absolute energy difference over total energy or NaN for other decays
    """
    if daughters is None:
        daughters = pi0_daughters(event, primary)
    if len(daughters) != 2 or not (event["pdg"][daughters] == 22).all():
        return np.nan
    momentum = np.column_stack([event[f"mc_{axis}"][daughters] for axis in "xyz"]).astype(np.float64)
    energies = np.linalg.norm(momentum, axis=1)
    if not np.isfinite(energies).all():
        raise ValueError("MC daughter photon energies must be finite")
    total = energies.sum()
    return abs(energies[0] - energies[1]) / total if total > 0 else np.nan


def passes_pi0_decay_cut(event, primary, limit):
    """Apply the strict upper bound on the generator photon energy asymmetry"""
    return pi0_decay_asymmetry(event, primary) < limit


def truth_label(event, selected_cells, class_pdgs, use_links, max_pi0_decay_asymmetry=None, photon_theta_range_deg=None):
    """
    Label selected cells from primary particles and optional cell truth links

    Args:
        event:  MC arrays kept outside the feature builder
        selected_cells:  Retained indices in the configured cell collection
        class_pdgs:  PDG codes in output class order
        use_links:  Resolve the primary by summed nonnegative link weights
        max_pi0_decay_asymmetry:  Optional upper bound for the associated pion decay
        photon_theta_range_deg:  Strict polar angle bounds for each MC photon in degrees
    Returns:
        result: Class index and truth observables, or None without a PID primary
    """
    pdg, status = event["pdg"], event["status"]
    begin, end = event["parents_begin"], event["parents_end"]
    candidates = np.flatnonzero(np.isin(pdg, class_pdgs) & (status != 0) & (begin == end))
    if not len(candidates):
        return None
    primary = int(candidates[0])
    if use_links:
        cells, particles, weights = (event[key] for key in ("link_cell", "link_particle", "link_weight"))
        if not (cells.shape == particles.shape == weights.shape) or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Truth links require aligned indices and nonnegative finite weights")
        if (cells < 0).any() or ("cell_energy" in event and (cells >= len(event["cell_energy"])).any()):
            raise ValueError("Truth link cell index is outside the configured cell collection")
        if ((particles < 0) | (particles >= len(pdg))).any():
            raise ValueError("Truth link particle index is outside MCParticles")
        scores = np.zeros(len(candidates))
        parent_index = event["parent_index"]
        selected = np.isin(cells, selected_cells)
        for particle, weight in zip(particles[selected], weights[selected]):
            pending, ancestors = [int(particle)], set()
            while pending:
                index = pending.pop()
                if index in ancestors:
                    continue
                ancestors.add(index)
                a, b = int(begin[index]), int(end[index])
                if not 0 <= a <= b <= len(parent_index):
                    raise ValueError("Invalid MC parent relation range")
                parents = parent_index[a:b]
                if ((parents < 0) | (parents >= len(pdg))).any():
                    raise ValueError("Invalid MC parent particle index")
                pending.extend(int(parent) for parent in parents)
            matches = np.isin(candidates, list(ancestors))
            if matches.sum() > 1:
                raise ValueError("A truth link has multiple PID primary ancestors")
            scores += weight * matches
        if scores.max() <= 0:
            return None
        winners = np.flatnonzero(scores == scores.max())
        if len(winners) != 1:
            raise ValueError("Ambiguous cluster truth association")
        primary = int(candidates[winners[0]])
    elif len(candidates) != 1:
        raise ValueError("Multiple PID primaries require truth_links and explicit link branches")
    px, py, pz = (float(event[f"mc_{axis}"][primary]) for axis in "xyz")
    energy = np.sqrt(px * px + py * py + pz * pz + float(event["mass"][primary]) ** 2)
    theta = np.degrees(np.arctan2(np.hypot(px, py), pz))
    if not np.isfinite([energy, theta]).all():
        raise ValueError("MC evaluation observables must be finite")
    pion = pdg[primary] == 111
    asymmetry = np.nan
    if pion and (max_pi0_decay_asymmetry is not None or photon_theta_range_deg is not None
                 or {"daughters_begin", "daughters_end", "daughter_index"} <= event.keys()):
        daughters = pi0_daughters(event, primary)
        asymmetry = pi0_decay_asymmetry(event, primary, daughters)
    truth = {"truth_energy": float(energy), "truth_theta": float(theta), "pi0_decay_asymmetry": asymmetry}
    if max_pi0_decay_asymmetry is not None:
        truth["passes_decay_cut"] = not pion or asymmetry < max_pi0_decay_asymmetry
    if photon_theta_range_deg is not None and not pion:
        low, high = photon_theta_range_deg
        truth["passes_photon_theta"] = (px != 0 or py != 0 or pz != 0) and low < theta < high
    elif photon_theta_range_deg is not None:
        photons = daughters[pdg[daughters] == 22]
        px, py, pz = (event[f"mc_{axis}"][photons].astype(np.float64) for axis in "xyz")
        pt = np.hypot(px, py)
        angles = np.degrees(np.arctan2(pt, pz))
        low, high = photon_theta_range_deg
        valid = np.isfinite(pt) & np.isfinite(pz) & ((pt > 0) | (pz != 0))
        truth["passes_photon_theta"] = len(photons) == 2 and bool(np.all(valid & (low < angles) & (angles < high)))
    return class_pdgs.index(int(pdg[primary])), truth


@dataclass
class ClusterChunk:
    features: np.ndarray | torch.Tensor
    offsets: np.ndarray
    entries: np.ndarray
    targets: np.ndarray | None
    weights: np.ndarray
    observables: dict[str, np.ndarray]
    global_features: np.ndarray | None = None
    passes_decay_cut: np.ndarray | None = None
    passes_photon_theta: np.ndarray | None = None

    def __len__(self):
        """
        Count retained events

        Returns:
            size: Retained event count
        """
        return len(self.entries)


def event_buffers(column):
    """
    Expose a ROOT branch as a flat NumPy array with event boundaries

    Args:
        column:  Cell, particle or relation lists for one ROOT chunk
    Returns:
        result: Flat values and offsets in source event order
    """
    import awkward as ak

    layout = ak.to_layout(column)
    if isinstance(layout, ak.contents.ListOffsetArray) and isinstance(layout.content, ak.contents.NumpyArray):
        offsets = np.asarray(layout.offsets, dtype=np.int64)
        values = layout.content.data[offsets[0]:offsets[-1]]
        if offsets[0]:
            offsets = offsets - offsets[0]
        return values, offsets

    counts = ak.to_numpy(ak.num(column, axis=1))
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    return ak.to_numpy(ak.flatten(column, axis=1)), offsets


def leading_cells(event, options, ids):
    """
    Select cells around the leading reconstructed ECAL cluster

    Args:
        event:  Detector arrays for one ROOT entry
        options:  Cluster and cell collection steering
        ids:  PODIO collection identifiers
    Returns:
        cells:  Indices in the configured cell collection
    """
    energies = event["cluster_energy"]
    if not np.isfinite(energies).all():
        raise ValueError("Reconstructed cluster energies must be finite")
    if not len(energies):
        return np.empty(0, dtype=np.int64)
    leading = int(np.argmax(energies))
    if options.input_mode == "leading_cluster_cone":
        coordinates = [event[f"cluster_{axis}"] for axis in "xyz"]
        if any(coordinate.shape != energies.shape for coordinate in coordinates):
            raise ValueError("Cluster energies and positions must align")
        axis = np.asarray([coordinate[leading] for coordinate in coordinates], dtype=np.float64)
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("The leading reconstructed cluster requires a finite nonzero position")
        x, y, z = (event[f"cell_{component}"].astype(np.float64, copy=False) for component in "xyz")
        radius = np.sqrt(x * x + y * y + z * z)
        if not np.isfinite(radius).all() or (radius == 0).any():
            raise ValueError("Positioned cells require finite nonzero directions")
        nx, ny, nz = axis / norm
        cosine = np.clip((x * nx + y * ny + z * nz) / radius, -1., 1.)
        return np.flatnonzero(cosine >= np.cos(options.cone_half_angle))
    if event["hits_begin"].shape != energies.shape or event["hits_end"].shape != energies.shape:
        raise ValueError("Cluster energies and hit relation ranges must align")
    begin, end = int(event["hits_begin"][leading]), int(event["hits_end"][leading])
    if not 0 <= begin <= end <= len(event["hit_index"]):
        raise ValueError("Invalid cluster hit relation range")
    cells = event["hit_index"][begin:end]
    validate_relation_collection(event, "hit_collection", options.cell_collection, ids, slice(begin, end))
    if ((cells < 0) | (cells >= len(event["cell_energy"]))).any() or len(np.unique(cells)) != len(cells):
        raise ValueError("Invalid or duplicate cluster cell index")
    return cells


def open_root(path, **settings):
    """
    Open a local memory map or a remote ROOT source

    Args:
        path:  ROOT file path or URL
        settings:  Uproot file options
    Returns:
        file:  Open ROOT file
    """
    import uproot

    if "://" not in str(path):
        settings["handler"] = uproot.source.file.MemmapSource
    return uproot.open(path, **settings)


def graph_selection(decay, theta):
    """
    Find the graph rows needed after the truth cuts

    Args:
        decay:  Decay cut mask or None
        theta:  Photon angle cut mask or None
    Returns:
        result:  Boolean selection or None when both cuts are disabled
    """
    if decay is None:
        return theta
    return decay if theta is None else decay & theta


def descriptor_batches(showers, basis, device, radius=None, *, keep=None):
    """
    Reduce shower graphs in batches of 32 events

    Args:
        showers:  Local cell coordinates and log energies
        basis:  Named graph basis or ordered tuple of components
        device:  Contraction device
        radius:  ECAL inner radius for the original detector-frame bases
        keep:  Mask of events that need graph contractions
    Returns:
        rows:  CPU tensor of graph columns in event order
    """
    if keep is None or keep.all():
        return torch.cat([descriptor_rows(showers[start:start + 32], basis, device, radius).cpu()
                          for start in range(0, len(showers), 32)])
    width = len(BASES[basis]) if isinstance(basis, str) else sum(len(BASES[name]) for name in basis)
    # Reserve the rejected row positions for the split index
    rows = torch.zeros((len(showers), width), dtype=torch.float32, device="cpu")
    for start in range(0, len(showers), 32):
        selected = np.flatnonzero(keep[start:start + 32]) + start
        if len(selected):
            rows[selected] = descriptor_rows([showers[index] for index in selected], basis, device, radius).cpu()
    return rows


def selection_cutflows(options, labeled):
    """
    Book event and cell counts for the active shower selection

    Args:
        options:  Detector and generator selection
        labeled:  Include MC association and active truth cuts
    Returns:
        cutflows:  Event flow and cell flow for events that pass the final event cut
    """
    event_stages = {"input": "Input ROOT events", "cluster": "Has reconstructed cluster",
                    "min_cells": f"min_cells >= {options.min_cells}"}
    mc_stages = {}
    if labeled:
        mc_stages["primary"] = "Associated MC primary"
        if options.max_pi0_decay_asymmetry is not None:
            mc_stages["asymmetry"] = f"max_pi0_decay_asymmetry: A < {options.max_pi0_decay_asymmetry:g}"
        if options.photon_theta_range_deg is not None:
            low, high = options.photon_theta_range_deg
            mc_stages["theta"] = f"photon_theta_range_deg: {low:g} < theta < {high:g} deg"
    cell_stages = {"input": f"Input cells ({options.cell_collection})"}
    cell_stages["cluster"] = (f"Leading cluster cone <= {options.cone_half_angle:g} rad"
                              if options.input_mode == "leading_cluster_cone" else "Linked to leading cluster")
    cell_stages["energy"] = (f"min_energy: E >= {options.min_energy:g} GeV"
                             if options.min_energy > 0 else "Cell E > 0")
    if options.max_cell_angle is not None:
        cell_stages["angle"] = f"max_cell_angle <= {options.max_cell_angle:g} rad"
    if options.max_cells is not None:
        cell_stages["cap"] = f"max_cells: first {options.max_cells} by energy"
    events = Cutflow(event_stages | mc_stages, groups=options.class_names if labeled else (), grouped_stages=mc_stages)
    return events, Cutflow(cell_stages)


def iter_clusters(path, options, geometry=None, labeled=True, descriptor_device="cpu", *, entry_start=0, entry_stop=None,
                  root_file=None, cutflows=None, defer_descriptors=False):
    """
    Stream selected cells around the highest-energy reconstructed ECAL cluster

    Args:
        path:  ROOT input file
        options:  Driver reader options
        geometry:  Detector readout constants, only the inner radius is used in local modes
        labeled:  Enable MC labels and truth evaluation observables
        descriptor_device:  Device for batched angular correlations
        entry_start:  First source entry for this reader
        entry_stop:  Exclusive source entry bound or None for the configured file limit
        root_file:  ROOT file already open in the calling process
        cutflows:  Event and cell counters shared across the input ranges of this read
        defer_descriptors:  Return local hits for graph contractions in the parent process
    Yields:
        chunk: Cell features, labels and source ROOT entry numbers
    """
    import awkward as ak

    options.validate_selection()
    event_flow, cell_flow = cutflows if cutflows is not None else selection_cutflows(options, labeled)
    if options.feature_mode not in {"extended", "angular", "cartesian"}:
        raise ValueError("driver.feature_mode must be extended, angular or cartesian")
    components = options.graph_components
    shower_frame = options.shower_basis in SHOWER_FRAME_BASES
    radial_shower = any(component in RADIAL_SHOWER_BASES for component in components)
    if not shower_frame:
        if geometry is None:
            raise ValueError("Cell features require detector readout constants")
        geometry.validate(local=options.feature_mode != "extended")
    radius = None if shower_frame else geometry.inner_radius_mm
    local_transform = cartesian_features if options.feature_mode == "cartesian" else angular_features
    cell_coordinates = ("cell_x", "cell_y", "cell_z")
    cell_fields = (*cell_coordinates, "cell_id") if options.feature_mode == "extended" else cell_coordinates
    angle_cosine = np.cos(options.max_cell_angle) if options.max_cell_angle is not None else None
    with (open_root(path) if root_file is None else nullcontext(root_file)) as root:
        tree = root[options.tree]
        verify_decay = labeled and options.max_pi0_decay_asymmetry is not None
        select_theta = labeled and options.photon_theta_range_deg is not None
        branches = resolve_branches(tree, options, labeled)
        verify_links = options.input_mode == "leading_cluster" or (labeled and options.truth_links) or "daughter_index" in branches
        ids = collection_ids(root, options.tree) if verify_links else {}
        expressions = list(dict.fromkeys(branches.values()))
        if options.weight_branch:
            expressions.append(options.weight_branch)
        processed = kept = raw_cells = kept_cells = 0
        minimum = maximum = 0
        stop = min(tree.num_entries, options.max_events)
        if entry_stop is not None:
            stop = min(stop, entry_stop)
        total = max(0, stop - entry_start)
        primary = int(os.environ.get("RANK", "0")) == 0
        started = time.perf_counter()
        if primary:
            source = f"{Path(path).name} [pid={os.getpid()}, entries {entry_start:,}:{stop:,}]"
            print(
                f"Reading {source}: {total:,} events, chunk_size={options.chunk_size:,}, "
                f"input_mode={options.input_mode}, cells={options.cell_collection}"
            )
        if entry_stop is None:
            chunks = tree.iterate(expressions, entry_start=entry_start, entry_stop=stop,
                                  step_size=options.chunk_size, library="ak", how=dict)
        else:
            # Each worker owns at most one chunk, with explicit TTree and RNTuple bounds
            chunks = (tree.arrays(expressions, entry_start=entry_start, entry_stop=stop, library="ak", how=dict),)
        for arrays in chunks:
            if processed >= total:
                break
            features, offsets, entries, targets, weights, observables = [], [0], [], [], [], {}
            shower_totals = []
            decay_mask = []
            theta_mask = []
            size = min(len(arrays[branches["cell_energy"]]), total - processed)
            event_flow.add("input", size)
            columns = {name: event_buffers(arrays[key][:size]) for name, key in branches.items()}
            event_weights = ak.to_numpy(arrays[options.weight_branch][:size]) if options.weight_branch else None
            for i in range(size):
                event = {name: values[edges[i]:edges[i + 1]] for name, (values, edges) in columns.items()}
                shape = event["cell_energy"].shape
                if len(shape) != 1 or any(event[name].shape != shape for name in cell_fields):
                    raise ValueError("Cell positions, energies and identifiers must align")
                cells = leading_cells(event, options, ids)
                if len(event["cluster_energy"]):
                    event_flow.add("cluster")
                raw_cells += len(cells)
                energy = event["cell_energy"][cells]
                if not np.isfinite(energy).all():
                    raise ValueError("Cell energies must be finite")
                passing = (energy > 0) & (energy >= options.min_energy)
                cells, passing_energy = cells[passing], energy[passing]
                above_threshold = len(cells)
                if angle_cosine is not None and len(cells):
                    position = np.stack([event[name][cells] for name in cell_coordinates], axis=1, dtype=np.float64)
                    cell_radius = np.linalg.norm(position, axis=1)
                    if not np.isfinite(cell_radius).all() or (cell_radius <= 0).any():
                        raise ValueError("A cell opening angle requires a finite nonzero position")
                    seed = np.argmax(passing_energy)
                    cosine = (position @ (position[seed] / cell_radius[seed])) / cell_radius
                    cosine[seed] = 1.
                    passing = cosine >= angle_cosine
                    cells, passing_energy = cells[passing], passing_energy[passing]
                order = np.argsort(-passing_energy, kind="stable")
                if options.max_cells is not None:
                    order = order[:options.max_cells]
                cells = cells[order]
                if len(cells) < options.min_cells:
                    continue
                position = np.column_stack([event[name][cells] for name in cell_coordinates])
                cell_energy = passing_energy[order]
                if shower_frame:
                    x = shower_frame_hits(position, cell_energy, options.feature_mode, include_radial=radial_shower)
                    obs = {"cluster_energy": float(cell_energy.sum(dtype=np.float64)), "n_cells": float(len(cells))}
                elif options.feature_mode == "extended":
                    x, obs = cell_features(position, cell_energy, event["cell_id"][cells], geometry)
                else:
                    x, obs = local_transform(position, cell_energy, geometry)
                if not len(x):
                    continue
                event_flow.add("min_cells")
                passes_event = True
                if labeled:
                    if options.truth_links:
                        validate_relation_collection(event, "parent_collection", options.mc_particles, ids)
                        validate_relation_collection(event, "link_cell_collection", options.cell_collection, ids)
                        validate_relation_collection(event, "link_particle_collection", options.mc_particles, ids)
                    if "daughter_index" in event:
                        validate_relation_collection(event, "daughter_collection", options.mc_particles, ids)
                    result = truth_label(event, cells, options.class_pdgs, options.truth_links,
                                         options.max_pi0_decay_asymmetry, options.photon_theta_range_deg)
                    if result is None:
                        continue
                    target, truth = result
                    group = options.class_names[target]
                    event_flow.add("primary", group=group)
                    if verify_decay:
                        passes_event = truth.pop("passes_decay_cut")
                        decay_mask.append(passes_event)
                        if passes_event:
                            event_flow.add("asymmetry", group=group)
                    if select_theta:
                        theta_mask.append(truth.pop("passes_photon_theta"))
                        passes_event = passes_event and theta_mask[-1]
                        if passes_event:
                            event_flow.add("theta", group=group)
                    targets.append(target)
                    obs.update(truth)
                if passes_event:
                    cell_flow.add("input", shape[0])
                    cell_flow.add("cluster", len(energy))
                    cell_flow.add("energy", above_threshold)
                    if options.max_cell_angle is not None:
                        cell_flow.add("angle", len(passing_energy))
                    if options.max_cells is not None:
                        cell_flow.add("cap", len(cells))
                weight = float(event_weights[i]) if event_weights is not None else 1.0
                if not np.isfinite(weight):
                    raise ValueError("Event weights must be finite scalars")
                if labeled:
                    obs["generator_weight"] = weight
                if options.global_features:
                    totals = {"log_n_cells": len(passing_energy), "log_sum_energy": passing_energy.sum(dtype=np.float64)}
                    shower_totals.append([totals[name] for name in options.global_features])
                features.append(x)
                offsets.append(offsets[-1] + len(x))
                entries.append(entry_start + processed + i)
                weights.append(weight)
                for name, value in obs.items():
                    observables.setdefault(name, []).append(value)
            processed += size
            if entries:
                counts = np.asarray(observables["n_cells"], dtype=np.int64)
                minimum = min(minimum, int(counts.min())) if kept else int(counts.min())
                maximum = max(maximum, int(counts.max()))
                kept += len(entries)
                kept_cells += int(counts.sum())
                decay = np.asarray(decay_mask, dtype=bool) if verify_decay else None
                theta = np.asarray(theta_mask, dtype=bool) if select_theta else None
                if components and not defer_descriptors:
                    features = descriptor_batches(features, components, descriptor_device, radius,
                                                  keep=graph_selection(decay, theta))
                    offsets = np.arange(len(entries) + 1)
                elif shower_frame:
                    features = torch.cat(features)
                else:
                    features = np.concatenate(features)
                yield ClusterChunk(
                    features, np.asarray(offsets, dtype=np.int64), np.asarray(entries, dtype=np.int64),
                    np.asarray(targets, dtype=np.int64) if labeled else None, np.asarray(weights),
                    {name: np.asarray(values) for name, values in observables.items()},
                    np.log(shower_totals).astype(np.float32) if options.global_features else None,
                    decay, theta,
                )
        if primary:
            mean = kept_cells / kept if kept else 0.0
            print(
                f"Finished {source}: retained {kept:,}/{processed:,} events, dropped {processed - kept:,}, "
                f"input_cells={raw_cells:,}, retained_cells={kept_cells:,}, "
                f"cells/event min={minimum}, mean={mean:.1f}, max={maximum}, "
                f"elapsed={time.perf_counter() - started:.1f} s"
            )
