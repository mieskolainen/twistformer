# Neutral pion and photon classification driver
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass, field, replace
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm
from drivers.base import BaseDriver
from twistformer.randomness import numpy_generator
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import ClassificationRows, InputSpec, Sample, TaskSpec
from twistformer.data.loading import fetch_samples
from twistformer.data.spectrum import normalize_energy_weights, report_energy_weights, spectrum_weights
from twistformer.logging import print
from .features import FEATURES, LOCAL_FEATURES, Geometry
from .graphs.descriptors import BASES, SHOWER_FRAME_BASES, shower_graph_components
from .parallel_reader import ordered_chunks
from .reader import selection_cutflows, source_paths


@dataclass
class Options:
    input_mode: str = "leading_cluster_cone"
    feature_mode: str = "extended"
    shower_basis: str | None = None
    shower_groups: list[str] | None = None
    shower_columns: list[int | list[int]] | None = None
    global_features: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    base_path: str = ""
    class_names: list[str] = field(default_factory=lambda: ["pi0", "gamma"])
    class_pdgs: list[int] = field(default_factory=lambda: [111, 22])
    signal_class: int | None = None
    split: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    energy_edges: list[float] = field(default_factory=list)
    energy_weights: list[float] | None = None
    balance_energy: bool = False
    seed: int = 101
    independent_energy_splits: bool = False
    tree: str = "events"
    max_events: int = 500000
    chunk_size: int = 2000
    root_workers: int = 1
    min_energy: float = 0.0
    max_pi0_decay_asymmetry: float | None = None
    photon_theta_range_deg: list[float] | None = None
    cone_half_angle: float = 0.30
    max_cell_angle: float | None = None
    min_cells: int = 1
    max_cells: int | None = None
    clusters: str = "AugmentedEMBCaloClusters"
    cells: str | None = None
    cluster_hits: str | None = None
    mc_particles: str = "MCParticles"
    truth_links: bool = False
    branches: dict[str, str] = field(default_factory=dict)
    geometry: dict | None = field(default_factory=dict)
    weight_branch: str | None = None

    @property
    def graph_components(self):
        """
        Resolve the feature groups selected by the steering card

        Returns:
            components:  Graph kernels in classifier order
        """
        return shower_graph_components(self.shower_basis, self.shower_groups)

    @property
    def cell_collection(self):
        """
        Select the positioned barrel collection or linked cluster members

        Returns:
            collection: Explicit override or collection for the input mode
        """
        if self.cells is not None:
            return self.cells
        return "ECalBarrelModuleThetaMergedPositioned" if self.input_mode == "leading_cluster_cone" else "EMBCaloClusterCells"

    def validate_selection(self):
        """
        Check the detector cone and the generator decay selection
        """
        if self.input_mode not in {"leading_cluster_cone", "leading_cluster"}:
            raise ValueError("pi0gamma requires leading_cluster_cone or leading_cluster as driver.input_mode")
        if type(self.root_workers) is not int or self.root_workers < 0:
            raise ValueError("root_workers must be a nonnegative integer")
        limit = self.max_pi0_decay_asymmetry
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 <= limit <= 1):
            raise ValueError("max_pi0_decay_asymmetry must lie in [0, 1] or be null")
        bounds = self.photon_theta_range_deg
        if bounds is not None and (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                                   or any(type(value) not in (int, float) for value in bounds)
                                   or not 0 <= bounds[0] < bounds[1] <= 180):
            raise ValueError("photon_theta_range_deg requires ordered angles in [0, 180] degrees or null")
        if self.input_mode == "leading_cluster_cone":
            angle = self.cone_half_angle
            if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not 0 < angle <= np.pi:
                raise ValueError("cone_half_angle must lie in (0, pi] radians")
            if self.max_cell_angle is not None:
                raise ValueError("Steer a leading-cluster cone with cone_half_angle")


class Pi0GammaDataset(Dataset):
    def __init__(self, blocks, indices):
        """
        Share compact cell buffers between event splits

        Args:
            blocks:  Source path and ClusterChunk pairs
            indices:  Chunk and event index pairs
        """
        self.blocks, self.indices = blocks, np.asarray(indices, dtype=np.int64).reshape(-1, 2)
        self.weights = np.array([blocks[block][1].weights[event] for block, event in self.indices], dtype=float)

    def __len__(self):
        """
        Count selected events

        Returns:
            size: Event count in this split
        """
        return len(self.indices)

    def __getitem__(self, index):
        """
        Copy one cluster with labels kept outside the input tensor

        Args:
            index:  Event index in this split
        Returns:
            sample: Cell features and event metadata
        """
        block, event = self.indices[index]
        path, chunk = self.blocks[block]
        begin, end = chunk.offsets[event:event + 2]
        return Sample(
            torch.as_tensor(chunk.features[begin:end]).clone(),
            target=None if chunk.targets is None else torch.tensor(int(chunk.targets[event])),
            global_features=None if chunk.global_features is None else torch.from_numpy(chunk.global_features[event].copy()),
            weight=float(self.weights[index]), id=f"{path}:{chunk.entries[event]}",
            observables={name: float(values[event]) for name, values in chunk.observables.items()},
        )

    def __getitems__(self, indices):
        """
        Gather cluster cells in one buffer without changing event order

        Args:
            indices:  Event indices in sampler order, including repeats
        Returns:
            samples: Independent event views with labels and observables
        """
        records = self.indices[indices]
        if len(records) == 0:
            return []
        blocks = [self.blocks[block] for block, _ in records]
        arrays, labels = [], []
        for (_, chunk), (_, event) in zip(blocks, records, strict=True):
            begin, end = chunk.offsets[event:event + 2]
            arrays.append(chunk.features[begin:end])
            labels.append(0 if chunk.targets is None else int(chunk.targets[event]))
        buffer = torch.cat(arrays) if torch.is_tensor(arrays[0]) else torch.from_numpy(np.concatenate(arrays))
        features = buffer.split([len(array) for array in arrays])
        targets = torch.tensor(labels, dtype=torch.long)
        return [
            Sample(
                cells,
                target=None if chunk.targets is None else target,
                global_features=None if chunk.global_features is None else torch.from_numpy(chunk.global_features[event].copy()),
                weight=float(weight), id=f"{path}:{chunk.entries[event]}",
                observables={name: float(values[event]) for name, values in chunk.observables.items()},
            )
            for (path, chunk), (_, event), cells, target, weight in zip(
                blocks, records, features, targets, self.weights[indices], strict=True
            )
        ]


class ShowerDescriptors(Pi0GammaDataset):
    """Single-row shower descriptors stored in physical ROOT chunks"""

    def classification_batch(self, indices, observable="", *, observables=()):
        """
        Gather selected shower rows from each chunk and restore sampler order

        Args:
            indices:     Slice or event indices with possible repeated views
            observable:  Optional scalar used for class-bin balancing
            observables:  Additional physical scalars in sampler order
        Returns:
            batch:  Detector inputs, labels and split-specific physical weights
        """
        records = self.indices[indices]
        template = self.blocks[0][1]
        size = len(records)
        features = torch.as_tensor(template.features).new_empty((size, template.features.shape[1]))
        globals_ = (np.empty((size, template.global_features.shape[1]), dtype=template.global_features.dtype)
                    if template.global_features is not None else None)
        labels = np.empty(size, dtype=np.int64)
        names = dict.fromkeys((observable, *observables) if observable else observables)
        values = {name: np.empty(size, dtype=np.float64) for name in names}
        order = records[:, 0].argsort(kind="stable")
        boundaries = np.flatnonzero(np.diff(records[order, 0])) + 1
        for positions in np.split(order, boundaries):
            if not len(positions):
                continue
            block = self.blocks[records[positions[0], 0]][1]
            events = records[positions, 1]
            features[positions] = torch.as_tensor(block.features)[events]
            labels[positions] = block.targets[events]
            if globals_ is not None:
                globals_[positions] = block.global_features[events]
            for name, column in values.items():
                if name not in block.observables:
                    raise ValueError(f"Missing event observable {name}")
                column[positions] = block.observables[name][events]
        columns = {name: torch.from_numpy(column) for name, column in values.items()}
        return ClassificationRows(features, torch.from_numpy(labels),
                                  torch.as_tensor(self.weights[indices]),
                                  torch.from_numpy(globals_) if globals_ is not None else None,
                                  columns.get(observable), {name: columns[name] for name in observables})


def reflect_x(sample):
    """
    Reflect all cells in the local theta or beam coordinate

    Args:
        sample:  Raw event in the local detector frame
    Returns:
        sample: Reflected cell features with the source event identity and weight
    """
    features = sample.features.clone()
    features[:, 0] *= -1
    return replace(sample, features=features)


class ReflectedTrainingDataset(Dataset):
    """Index each training shower and its reflection from the same cell buffers"""

    def __init__(self, dataset):
        """
        Store the physical training split

        Args:
            dataset:  Raw training events
        """
        self.dataset = dataset

    def __len__(self):
        """Count both views of every training event"""
        return 2 * len(self.dataset)

    def __getitem__(self, index):
        """
        Read the original or reflected event

        Args:
            index:  Training view index
        Returns:
            sample: Original for even indices and reflected for odd indices
        """
        sample = self.dataset[index // 2]
        return reflect_x(sample) if index % 2 else sample

    def __getitems__(self, indices):
        """
        Fetch both shower orientations in sampler order

        Args:
            indices:  Training view indices
        Returns:
            samples: Ordered original and reflected events
        """
        samples = fetch_samples(self.dataset, [index // 2 for index in indices])
        return [reflect_x(sample) if index % 2 else sample for index, sample in zip(indices, samples, strict=True)]


def selected_columns(selection, width):
    """
    Resolve individual graph indices and half-open column ranges in steering order

    Args:
        selection:  Indices or pairs of first and exclusive stop indices
        width:  Number of analytical shower descriptors
    Returns:
        columns:  Unique indices in classifier input order
    """
    if not isinstance(selection, list) or not selection:
        raise ValueError("shower_columns must be a nonempty list")
    columns = []
    for item in selection:
        if type(item) is int and 0 <= item < width:
            columns.append(item)
        elif (isinstance(item, list) and len(item) == 2 and all(type(i) is int for i in item)
              and 0 <= item[0] < item[1] <= width):
            columns.extend(range(*item))
        else:
            raise ValueError("shower_columns requires valid indices or half-open ranges")
    if len(set(columns)) != len(columns):
        raise ValueError("shower_columns must select unique indices")
    return columns


class Driver(BaseDriver):
    comparison_input_options = ("feature_mode", "geometry", "global_features", "shower_basis", "shower_groups",
                                "shower_columns")
    comparison_source_options = (
        *BaseDriver.comparison_source_options, "base_path", "energy_edges", "energy_weights", "balance_energy",
        "independent_energy_splits", "max_pi0_decay_asymmetry", "photon_theta_range_deg", "root_workers",
    )
    evaluation_observables = {
        "cluster_energy": {"label": "Total cell $E$", "unit": "GeV"},
        "n_cells": {"label": "$N$ cells", "unit": ""},
        "truth_energy": {"label": "True $E$", "unit": "GeV"},
        "truth_theta": {"label": "True $\\theta$", "unit": "deg"},
    }

    def __init__(self, options, task):
        """
        Declare the cluster features and binary PID labels

        Args:
            options:  Serializable reader and geometry settings
            task:  Shared classification settings
        """
        self.descriptor_device = "cpu"
        if "eta_max" in options and options["eta_max"] is None:
            # Older checkpoints store the disabled cell eta cut
            options = {name: value for name, value in options.items() if name != "eta_max"}
        self.options = strict_dataclass(Options, options)
        o = self.options
        o.validate_selection()
        components = o.graph_components
        if o.feature_mode not in {"extended", "angular", "cartesian"}:
            raise ValueError("driver.feature_mode must be extended, angular or cartesian")
        if set(o.global_features) - {"log_n_cells", "log_sum_energy"}:
            raise ValueError("driver.global_features accepts log_n_cells and log_sum_energy")
        if task.name != "classification" or set(o.class_pdgs) != {22, 111} or len(o.class_names) != 2 or len(o.class_pdgs) != 2:
            raise ValueError("Pi0Gamma requires two classes with PDG codes 22 and 111")
        if not o.files or min(o.max_events, o.chunk_size) < 1:
            raise ValueError("Pi0Gamma requires ROOT sources and positive event limits")
        if not np.isfinite(o.min_energy) or o.min_energy < 0:
            raise ValueError("min_energy must be finite and nonnegative in GeV")
        if o.max_cell_angle is not None and not 0 < o.max_cell_angle <= np.pi:
            raise ValueError("max_cell_angle must lie in (0, pi] radians, or be null")
        for name in ("min_cells", "max_cells"):
            value = getattr(o, name)
            if name == "max_cells" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if o.max_cells is not None and o.max_cells < o.min_cells:
            raise ValueError("max_cells must be at least min_cells")
        if len(o.split) != 3 or not np.isfinite(o.split).all() or min(o.split) <= 0 or abs(sum(o.split) - 1) > 1e-6:
            raise ValueError("Pi0Gamma split requires three positive fractions summing to one")
        if o.energy_edges and (
            len(o.energy_edges) < 2 or not np.isfinite(o.energy_edges).all() or (np.diff(o.energy_edges) <= 0).any()
        ):
            raise ValueError("energy_edges must contain finite increasing bin boundaries")
        if o.balance_energy and not o.energy_edges:
            raise ValueError("balance_energy requires energy_edges")
        self.energy_fractions = None
        if o.energy_weights is not None:
            self.energy_fractions = normalize_energy_weights(o.energy_weights, o.energy_edges)
            if o.balance_energy:
                raise ValueError("Set balance_energy to false when using energy_weights")
        self.evaluation_observables = {
            **self.evaluation_observables,
            "pi0_decay_asymmetry": {"label": r"Generated $\pi^0$ decay asymmetry (inclusive $\gamma$ reference)", "unit": "",
                                    "binned_class": o.class_names[o.class_pdgs.index(111)]},
            "truth_energy": {**self.evaluation_observables["truth_energy"], "gun_energy_edges": o.energy_edges},
            "generator_weight": {"label": "Generator weight", "unit": ""},
        }
        self.geometry = None if o.shower_basis in SHOWER_FRAME_BASES else strict_dataclass(Geometry, o.geometry)
        features = FEATURES if o.feature_mode == "extended" else LOCAL_FEATURES
        units = ("50 mm", "50 mm", "50 mm", "ln(E/GeV)", "", "", "50 mm", "50 mm")
        self.input_spec = InputSpec(features, units=units[:len(features)], global_features=tuple(o.global_features))
        self.event_descriptors = bool(components)
        self.shower_columns = None
        if self.event_descriptors:
            if o.feature_mode == "extended":
                raise ValueError("shower_basis requires angular or cartesian hits")
            features = tuple(name for component in components for name in BASES[component])
            if o.shower_columns is not None:
                self.shower_columns = selected_columns(o.shower_columns, len(features))
                features = tuple(features[index] for index in self.shower_columns)
            self.input_spec = InputSpec(features, global_features=tuple(o.global_features))
        elif o.shower_columns is not None:
            raise ValueError("shower_columns requires a shower_basis")
        # Older training cards used the neutral pion as signal
        signal_class = o.signal_class if "signal_class" in options else o.class_pdgs.index(111)
        self.task_spec = TaskSpec("classification", tuple(o.class_names), signal_class=signal_class)
        self._datasets = None

    def create_preprocessor(self, state=None, *, config=None):
        """
        Configure training-fitted energy weights without changing model inputs

        Args:
            state:  Saved tensor preprocessing state or None
            config:  Shared normalization settings
        Returns:
            preprocessor: Standard tensor preprocessor with optional energy balancing
        """
        preprocessor = super().create_preprocessor(state, config=config)
        if self.options.balance_energy:
            preprocessor.configure_class_bins("truth_energy", self.options.energy_edges)
        return preprocessor

    def comparison_identity(self, split, input_path=None):
        """
        Resolve directory inputs for shared event comparisons

        Args:
            split:  Requested event split
            input_path:  Optional input override
        Returns:
            identity: Reader options and expanded source paths
        """
        identity = super().comparison_identity(split, input_path)
        identity["sources"] = (
            source_paths([input_path]) if input_path else source_paths(self.options.files, self.options.base_path)
        )
        return identity

    def _read(self, paths, labeled):
        """
        Index compact ROOT chunks without copying cell buffers

        Args:
            paths:  Expanded ROOT paths
            labeled:  Read MC labels for training and evaluation
        Returns:
            result: Shared blocks and event index array
        """
        blocks, records = [], []
        cutflows = selection_cutflows(self.options, labeled)
        primary = int(os.environ.get("RANK", "0")) == 0
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        workers = min(self.options.root_workers or cpus, cpus)
        if primary:
            print(f"Loading ROOT data: {len(paths):,} files, up to {workers} readers on {cpus} available logical CPUs")
            if labeled and self.energy_fractions is not None:
                report_energy_weights("driver.energy_weights", self.options.energy_weights, self.energy_fractions)
        with tqdm(total=len(paths), desc="ROOT files ready", unit="file", dynamic_ncols=True, disable=not primary) as progress:
            for path, chunks, last in ordered_chunks(paths, self.options, self.geometry, labeled,
                                                     self.descriptor_device, workers, cutflows):
                for chunk in chunks:
                    if self.shower_columns is not None:
                        chunk.features = chunk.features[:, self.shower_columns].contiguous()
                    records.extend((len(blocks), event) for event in range(len(chunk)))
                    blocks.append((path, chunk))
                if last:
                    progress.update()
        if primary:
            events, cells = cutflows
            print(events.table("Event cutflow (all input files, unweighted)"))
            print(cells.table("Cell cutflow (events passing the final event selection, unweighted)"))
        return blocks, np.asarray(records, dtype=np.int64).reshape(-1, 2)

    def prediction_dataset(self, input_path):
        """
        Read detector clusters without opening any MC branches

        Args:
            input_path:  ROOT file, directory or pattern
        Returns:
            dataset: Unlabeled cluster samples
        """
        return Pi0GammaDataset(*self._read(source_paths([input_path]), labeled=False))

    def _labeled_dataset(self, blocks, indices, labels, bins):
        """
        Set the gun spectrum within each class of the selected sample

        Args:
            blocks:   Shared ROOT chunks
            indices:  Selected chunk and event indices
            labels:   Class of each selected event
            bins:     Truth energy bin of each selected event
        Returns:
            dataset: Selected events with physical spectrum weights
        """
        keep = np.ones(len(indices), dtype=bool)
        if self.options.max_pi0_decay_asymmetry is not None:
            # Apply the decay cut after splitting to preserve physical event identities
            keep = np.fromiter((blocks[b][1].passes_decay_cut[i] for b, i in indices), dtype=bool, count=len(indices))
        if self.options.photon_theta_range_deg is not None:
            keep &= np.fromiter((blocks[b][1].passes_photon_theta[i] for b, i in indices), dtype=bool, count=len(indices))
        dataset_type = ShowerDescriptors if self.event_descriptors else Pi0GammaDataset
        dataset = dataset_type(blocks, indices[keep])
        if self.energy_fractions is not None:
            dataset.weights = spectrum_weights(dataset.weights, labels[keep], bins[keep], self.energy_fractions)
        return dataset
    
    def dataset(self, split, input_path=None):
        """
        Make deterministic splits within each class and configured energy interval

        Args:
            split:  train, val or test
            input_path:  Optional labeled evaluation input
        Returns:
            dataset: Selected cluster samples sharing compact storage
        """
        if input_path and self.energy_fractions is None:
            blocks, indices = self._read(source_paths([input_path]), labeled=True)
            return self._labeled_dataset(blocks, indices, None, None)
        if not input_path and split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split: {split}")
        if input_path or self._datasets is None:
            paths = source_paths([input_path]) if input_path else source_paths(self.options.files, self.options.base_path)
            blocks, indices = self._read(paths, labeled=True)
            if not len(indices):
                raise ValueError("No labeled positive-energy leading clusters found")
            labels = np.asarray([blocks[block][1].targets[event] for block, event in indices])
            bins = np.zeros(len(indices), dtype=np.int64)
            bin_count = 1
            if self.options.energy_edges:
                energies = np.asarray([blocks[block][1].observables["truth_energy"][event] for block, event in indices])
                bins = np.searchsorted(self.options.energy_edges, energies, side="right") - 1
                bin_count = len(self.options.energy_edges) - 1
                if not np.isfinite(energies).all() or ((bins < 0) | (bins >= bin_count)).any():
                    raise ValueError("Truth energies must lie within energy_edges")
            if input_path:
                return self._labeled_dataset(blocks, indices, labels, bins)
            # Each energy reproduces its standalone class-stratified split
            if self.options.independent_energy_splits:
                generators = [numpy_generator(self.options.seed) for _ in range(bin_count)]
            else:
                generators = [numpy_generator(self.options.seed)] * bin_count
            records = {name: [] for name in ("train", "val", "test")}
            required_bins = np.full(bin_count, self.options.balance_energy)
            if self.energy_fractions is not None:
                # Both classes must cover each populated energy with positive target yield
                present = np.bincount(bins, minlength=bin_count) > 0
                required_bins = present & (self.energy_fractions > 0)
            for label in range(2):
                for energy_bin in range(bin_count):
                    selected = generators[energy_bin].permutation(np.flatnonzero((labels == label) & (bins == energy_bin)))
                    if not len(selected) and not required_bins[energy_bin]:
                        continue
                    if len(selected) < 3:
                        raise ValueError(f"Class {label}, energy bin {energy_bin} needs at least three events for splitting")
                    a = max(1, min(len(selected) - 2, round(len(selected) * self.options.split[0])))
                    b = max(a + 1, min(len(selected) - 1, a + round(len(selected) * self.options.split[1])))
                    for name, subset in zip(records, (selected[:a], selected[a:b], selected[b:])):
                        records[name].append(subset)
            datasets = {}
            for name, parts in records.items():
                subset = np.concatenate(parts)
                datasets[name] = self._labeled_dataset(blocks, indices[subset], labels[subset], bins[subset])
            self._datasets = datasets
            if int(os.environ.get("RANK", "0")) == 0:
                counts = ", ".join(f"{name}={len(dataset):,}" for name, dataset in self._datasets.items())
                print(f"Pi0Gamma splits: {counts} events")
        return self._datasets[split]

    def augment_dataset(self, dataset):
        """
        Pair each training shower with its local x reflection

        Args:
            dataset:  Physical training split
        Returns:
            dataset: Two orientations per source event
        """
        if self.options.shower_basis:
            # Each descriptor has even parity under beam reflection
            return Subset(dataset, np.arange(len(dataset)).repeat(2))
        return ReflectedTrainingDataset(dataset)

    def augment(self, sample):
        """
        Reflect local x with probability one half in the selected coordinate frame

        Args:
            sample:  Raw training sample
        Returns:
            sample: Independent reflected features or the original sample
        """
        if self.options.shower_basis:
            return sample
        return reflect_x(sample) if torch.rand(()) < 0.5 else sample
