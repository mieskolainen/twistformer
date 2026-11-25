# Calorimeter hit regression for tau momentum and decay displacement
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass
import numpy as np
import torch
from drivers.base import BaseDriver
from twistformer.randomness import numpy_generator
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec, TaskSpec
from drivers.tau3_classification.geometry import validate_proxy_options
from drivers.tau3_classification.driver import HitOptions, Tau3Dataset

@dataclass
class Options(HitOptions):
    pass

class Tau3RegressionDataset(Tau3Dataset):
    def _sample(self, block, event, features, p4, energy):
        """
        Attach the tau six-vector and transverse momentum to one event

        Args:
            block:  Source path and hit chunk
            event:  Event index within the chunk
            features:  Independent hit tensor
            p4:  Optional energy-flow momenta
            energy:  Selected calorimeter energy
        Returns:
            sample: Calorimeter event with optional truth targets
        """
        path, chunk = block
        sample = super()._sample((path, None, chunk), event, features, p4, energy)
        sample.target = None if chunk.targets is None else torch.from_numpy(chunk.targets[event].copy())
        sample.observables["truth_pt"] = (
            float(np.linalg.norm(chunk.targets[event, :2])) if chunk.targets is not None else float("nan")
        )
        return sample

class Driver(BaseDriver):
    comparison_input_options = ("max_hits", "momentum_proxy", "proxy_origin")
    evaluation_observables = {
        "visible_energy": {"label": "Calorimeter energy", "unit": "GeV"},
        "n_hits": {"label": "Hits", "unit": ""},
        "truth_pt": {"label": "Truth tau pT", "unit": "GeV"},
    }

    def __init__(self, options, task):
        """
        Define hit inputs and six Cartesian regression outputs

        Args:
            options:  ROOT files, hit cuts and split settings
            task:  Gaussian or Gaussian mixture regression settings
        """
        self.options = strict_dataclass(Options, options)
        o = self.options
        validate_proxy_options(o.momentum_proxy, o.proxy_origin)
        if task.name not in {"gaussian_regression", "flow_matching", "flow_nll"} or not o.files:
            raise ValueError("Tau3 targets require ROOT files and a regression or flow task")
        if len(set(o.files)) != len(o.files):
            raise ValueError("Repeated ROOT files would leak events between splits")
        if len(o.split) != 3 or not np.isfinite(o.split).all() or min(o.split) <= 0 or abs(sum(o.split) - 1) > 1e-6:
            raise ValueError("Tau3 split requires three positive fractions summing to one")
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
        self.task_spec = TaskSpec(
            task.name, ("px", "py", "pz", "dx", "dy", "dz"), ("GeV", "GeV", "GeV", "mm", "mm", "mm")
        )
        self._datasets = None

    def _load(self, path, blocks, read_targets=True):
        """
        Index hit chunks while preserving ROOT entry and truth alignment

        Args:
            path:  ROOT file path
            blocks:  Shared chunk storage extended in place
            read_targets:  Enable truth labels and MC acceptance
        Returns:
            indexed: Chunk/event pairs and nonempty event mask
        """
        from .reader import iter_tau3_regression

        o = self.options
        indices, nonempty = [], []
        for chunk in iter_tau3_regression(
            path,
            o.tree,
            read_targets=read_targets,
            max_hits=o.max_hits,
            max_events=o.max_events,
            chunk_size=o.chunk_size,
            e_min=o.min_energy,
            fiducial_eta_max=o.eta_max,
            input_mode=o.input_mode,
            weight_branch=o.weight_branch,
        ):
            indices.append(np.column_stack((np.full(len(chunk), len(blocks)), np.arange(len(chunk)))))
            nonempty.append(np.diff(chunk.offsets) > 0)
            blocks.append((path, chunk))
        if not indices:
            return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=bool)
        return np.concatenate(indices), np.concatenate(nonempty)

    def prediction_dataset(self, input_path):
        """
        Read an unlabeled ROOT file using calorimeter branches only

        Args:
            input_path:  ROOT file for inference without MC acceptance
        Returns:
            dataset: Nonempty hit events without labels
        """
        blocks = []
        indices, nonempty = self._load(input_path, blocks, read_targets=False)
        return Tau3RegressionDataset(blocks, indices[nonempty], self.options.momentum_proxy, self.options.proxy_origin)

    def dataset(self, split, input_path=None):
        """
        Reconstruct seeded splits or read a labeled external ROOT file

        Args:
            split:  train, val or test
            input_path:  Optional labeled ROOT file bypassing the split
        Returns:
            dataset: Calorimeter hits and tau six-vectors
        """
        if input_path:
            blocks = []
            indices, nonempty = self._load(input_path, blocks)
            return Tau3RegressionDataset(blocks, indices[nonempty], self.options.momentum_proxy, self.options.proxy_origin)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split: {split}")
        if self._datasets is None:
            blocks = []
            records = {key: [] for key in ("train", "val", "test")}
            rng = numpy_generator(self.options.seed)
            for path in self.options.files:
                indices, nonempty = self._load(path, blocks)
                order = rng.permutation(len(indices))
                a = round(len(indices) * self.options.split[0])
                b = min(len(indices), a + round(len(indices) * self.options.split[1]))
                for key, selected in zip(records, (order[:a], order[a:b], order[b:])):
                    records[key].append(indices[selected[nonempty[selected]]])
            self._datasets = {
                key: Tau3RegressionDataset(blocks, np.concatenate(parts), self.options.momentum_proxy, self.options.proxy_origin)
                for key, parts in records.items()
            }
            if any(not len(dataset) for dataset in self._datasets.values()):
                raise ValueError("Tau3 regression requires nonempty train, val and test splits")
        return self._datasets[split]

    def prediction_extra(self, data):
        """
        Export cylindrical momentum for each joint hypothesis

        Args:
            data:  Predictions in physical Cartesian coordinates
        Returns:
            arrays: Derived observables and leading component predictions
        """
        from .evaluation import prediction_extra

        return prediction_extra(data)

    def evaluate_extra(self, data, output):
        """
        Write tau observables and ambiguity diagnostics

        Args:
            data:  Physical predictions and truth labels
            output:  Evaluation directory
        """
        from .evaluation import evaluate_extra

        evaluate_extra(data, output)

    def plot_extra(self, data, output):
        """
        Plot tau residuals and marginal mixture calibration

        Args:
            data:  Physical predictions and truth labels
            output:  Plot directory
        """
        from .evaluation import plot_extra

        plot_extra(data, output)
