# Driver data and physics definitions
#
# mikael.mieskolainen@cern.ch (2026)

"""ALLEGRO reader adapter. Raw features are fitted by the shared preprocessor"""

from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset
from twistformer.randomness import numpy_generator
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec, TaskSpec, Sample
from drivers.base import BaseDriver
from .geometry import gather_hits, hit_momenta, validate_proxy_options

@dataclass
class HitOptions:
    files: list[str] = field(default_factory=list)  # ordered by class
    split: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    seed: int = 101
    tree: str = "events"
    max_hits: int = 2048
    max_events: int = 20000
    chunk_size: int = 20000
    min_energy: float = 0.0005
    eta_max: float = 1.1
    input_mode: str = "sim"
    weight_branch: str | None = None
    momentum_proxy: bool = False
    proxy_origin: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

@dataclass
class Options(HitOptions):
    class_names: list[str] = field(default_factory=lambda: ["0pi0", "1pi0", "2pi0"])
    signal_class: int | None = None

class Tau3Dataset(Dataset):
    def __init__(self, blocks, indices, momentum_proxy=False, proxy_origin=(0.0, 0.0, 0.0)):
        """
        Share compact calorimeter chunks across dataset splits

        Args:
            blocks:  Path, class label and compact chunk tuples
            indices:  Chunk and event indices for this split
            momentum_proxy:  Attach null energy-flow geometry
            proxy_origin:  Nominal production point in mm
        """
        self.blocks = blocks
        self.indices = indices
        self.momentum_proxy, self.proxy_origin = momentum_proxy, tuple(proxy_origin)

    def __len__(self):
        """
        Count events in this split

        Returns:
            size: Selected nonempty event count
        """
        return len(self.indices)

    def __getitem__(self, i):
        """
        Copy retained hits into an independent training sample

        Args:
            i:  Event index in this split
        Returns:
            sample: Calorimeter hits, class, weight and original ROOT entry ID
        """
        block, event = self.indices[i]
        record = self.blocks[block]
        chunk = record[-1]
        start, stop = chunk.offsets[event : event + 2]
        features = torch.from_numpy(chunk.features[start:stop].copy())
        p4 = hit_momenta(features, self.proxy_origin) if self.momentum_proxy else None
        return self._sample(record, event, features, p4, features[:, 3].double().sum())

    def __getitems__(self, indices):
        """
        Gather a random batch from compact hit buffers

        Args:
            indices:  Event indices in sampler order
        Returns:
            samples: Independent events with batched energy-flow geometry
        """
        records = self.indices[indices]
        blocks = [self.blocks[block] for block, _ in records]
        hits = gather_hits([block[-1] for block in blocks], records[:, 1], self.momentum_proxy, self.proxy_origin)
        return [
            self._sample(block, event, features, p4, energy)
            for block, (_, event), (features, p4, energy) in zip(blocks, records, hits, strict=True)
        ]

    def _sample(self, block, event, features, p4, energy):
        """
        Attach class labels and calorimeter observables to one event

        Args:
            block:  Source path, class label and hit chunk
            event:  Event index within the chunk
            features:  Independent hit tensor
            p4:  Optional energy-flow momenta
            energy:  Selected calorimeter energy
        Returns:
            sample: Labeled calorimeter event
        """
        path, label, chunk = block
        return Sample(
            features,
            None if label is None else torch.tensor(label),
            p4=p4,
            id=f"{path}:{chunk.entries[event]}",
            observables={"visible_energy": float(energy), "n_hits": float(len(features))},
            weight=float(chunk.weights[event]),
        )

class Driver(BaseDriver):
    comparison_input_options = ("max_hits", "momentum_proxy", "proxy_origin")
    evaluation_observables = {
        "visible_energy": {"label": "Calorimeter energy", "unit": "GeV"},
        "n_hits": {"label": "Hits", "unit": ""},
    }

    def __init__(self, options, task):
        self.options = strict_dataclass(Options, options)
        o = self.options
        validate_proxy_options(o.momentum_proxy, o.proxy_origin)
        if task.name != "classification" or len(o.files) != len(o.class_names) or len(o.files) < 2:
            raise ValueError("Tau3 requires ordered files/class_names and classification")
        if len(o.split) != 3 or min(o.split) <= 0 or abs(sum(o.split) - 1) > 1e-6:
            raise ValueError("Tau3 split must have three positive fractions summing to one")
        if (
            min(o.max_hits, o.max_events, o.chunk_size) < 1
            or not np.isfinite(o.min_energy)
            or o.min_energy < 0
            or not np.isfinite(o.eta_max)
            or o.eta_max <= 0
        ):
            raise ValueError("Invalid tau3 selection parameters")
        if o.input_mode not in {"sim", "reco-cluster", "reco-topocluster"}:
            raise ValueError("Unknown tau3 input_mode")
        self.input_spec = InputSpec(
            ("x", "y", "z", "E"), geometry="p4" if o.momentum_proxy else "none", units=("mm", "mm", "mm", "GeV")
        )
        self.task_spec = TaskSpec("classification", tuple(o.class_names), signal_class=o.signal_class)
        self._datasets = None

    def _read(self, path, apply_fiducial=True):
        """
        Open a compact calorimeter chunk iterator

        Args:
            path:  ROOT class file
            apply_fiducial:  Apply MC acceptance for labeled training samples
        Returns:
            chunks: Selected calorimeter chunks
        """
        from .reader import iter_tau3

        o = self.options
        return iter_tau3(
            path,
            o.tree,
            max_hits=o.max_hits,
            max_events=o.max_events,
            chunk_size=o.chunk_size,
            e_min=o.min_energy,
            fiducial_eta_max=o.eta_max,
            input_mode=o.input_mode,
            weight_branch=o.weight_branch,
            apply_fiducial=apply_fiducial,
        )

    def _load_blocks(self, path, label, blocks, apply_fiducial=True):
        """
        Index compact chunks without copying their hit buffers

        Args:
            path:  ROOT class file
            label:  Class index or None for inference
            blocks:  Shared chunk storage extended in place
            apply_fiducial:  Apply MC acceptance outside hit-only prediction
        Returns:
            indexed: Chunk/event pairs and nonempty event mask
        """
        indices, nonempty = [], []
        chunks = self._read(path) if apply_fiducial else self._read(path, apply_fiducial=False)
        for chunk in chunks:
            indices.append(
                np.column_stack((np.full(len(chunk), len(blocks), dtype=np.int64), np.arange(len(chunk), dtype=np.int64)))
            )
            nonempty.append(np.diff(chunk.offsets) > 0)
            blocks.append((path, label, chunk))
        if not indices:
            return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=bool)
        return np.concatenate(indices), np.concatenate(nonempty)

    def prediction_dataset(self, input_path):
        """
        Read unlabeled calorimeter hits without MC acceptance branches

        Args:
            input_path:  ROOT file containing detector hits
        Returns:
            dataset: Nonempty events with optional null momentum proxies
        """
        blocks = []
        indices, nonempty = self._load_blocks(input_path, None, blocks, apply_fiducial=False)
        return Tau3Dataset(blocks, indices[nonempty], self.options.momentum_proxy, self.options.proxy_origin)

    def dataset(self, split, input_path=None):
        """
        Build deterministic splits sharing compact calorimeter storage

        Args:
            split:  train, val or test
            input_path:  Optional unlabeled ROOT file
        Returns:
            dataset: Indexed nonempty calorimeter events
        """
        if input_path:
            blocks = []
            indices, nonempty = self._load_blocks(input_path, None, blocks)
            return Tau3Dataset(blocks, indices[nonempty], self.options.momentum_proxy, self.options.proxy_origin)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split: {split}")
        if self._datasets is None:
            blocks = []
            records = {key: [] for key in ("train", "val", "test")}
            rng = numpy_generator(self.options.seed)
            for label, path in enumerate(self.options.files):
                indices, nonempty = self._load_blocks(path, label, blocks)
                order = rng.permutation(len(indices))
                a = round(len(indices) * self.options.split[0])
                b = min(len(indices), a + round(len(indices) * self.options.split[1]))
                for key, selected in zip(records, (order[:a], order[a:b], order[b:])):
                    records[key].append(indices[selected[nonempty[selected]]])
            self._datasets = {
                key: Tau3Dataset(blocks, np.concatenate(parts), self.options.momentum_proxy, self.options.proxy_origin)
                for key, parts in records.items()
            }
        return self._datasets[split]
