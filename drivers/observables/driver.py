# Event observable classifier inputs
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass, field, replace
import torch
from torch.utils.data import Dataset, Subset
from drivers.base import BaseDriver, build_driver
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec
from twistformer.data.loading import fetch_samples


class ObservableDataset(Dataset):
    def __init__(self, dataset, variables, source_spec):
        """
        Select scalar observables as one classifier token per event

        Args:
            dataset:  Raw source events
            variables:  Observable names or global:name and parameter:name selectors
            source_spec:  Source input schema
        Returns:
            None: The source and scalar selectors are stored
        """
        self.dataset, self.variables, self.source_spec = dataset, variables, source_spec

    def __len__(self):
        """
        Count source events

        Returns:
            size: Source event count
        """
        return len(self.dataset)

    def project(self, sample):
        """
        Extract scalar inputs while retaining class labels and physical weights

        Args:
            sample:  Raw source event
        Returns:
            sample: One token with selected scalar inputs
        """
        values = []
        for variable in self.variables:
            if variable.startswith(("global:", "parameter:")):
                kind, name = variable.split(":", 1)
                field = "global_features" if kind == "global" else "parameters"
                names = getattr(self.source_spec, field)
                vector = getattr(sample, field)
                if name not in names or vector is None:
                    raise ValueError(f"Missing reweighting input: {variable}")
                value = float(vector[names.index(name)])
            else:
                if variable not in sample.observables:
                    raise ValueError(f"Missing reweighting observable: {variable}")
                value = float(sample.observables[variable])
            values.append(value)
        features = torch.tensor([values], dtype=torch.float32)
        if not torch.isfinite(features).all():
            raise ValueError("Reweighting inputs must be finite")
        return replace(sample, features=features, p4=None, token_weights=None, global_features=None, parameters=None)

    def __getitem__(self, index):
        """
        Project one source event

        Args:
            index:  Event index
        Returns:
            sample: Scalar classifier event
        """
        return self.project(self.dataset[index])

    def __getitems__(self, indices):
        """
        Preserve batched source reads

        Args:
            indices:  Event indices in sampler order
        Returns:
            samples: Projected events in the same order
        """
        return [self.project(sample) for sample in fetch_samples(self.dataset, indices)]


@dataclass
class Options:
    source: dict = field(default_factory=dict)
    variables: list[str] = field(default_factory=list)
    splits: dict[str, str] = field(default_factory=dict)
    event_ids: dict[str, list[str]] = field(default_factory=dict)


class Driver(BaseDriver):
    def __init__(self, options, task):
        """
        Build a scalar view of an existing physics driver

        Args:
            options:  Source driver, scalar selectors and optional fold selections
            task:  Classification task configuration
        Returns:
            None: Source reader and scalar input schema are configured
        """
        self.options = strict_dataclass(Options, options)
        if task.name != "classification":
            raise ValueError("The observables driver requires classification")
        self.source = build_driver(self.options.source, task)
        self.input_spec = InputSpec(tuple(self.options.variables))
        self.task_spec = self.source.task_spec
        self.evaluation_observables = self.source.evaluation_observables

    def project(self, dataset):
        """
        Apply the configured scalar view to raw source events

        Args:
            dataset:  Raw source dataset
        Returns:
            dataset: Classifier scalar view
        """
        return ObservableDataset(dataset, self.options.variables, self.source.input_spec)

    def dataset(self, split, input_path=None):
        """
        Read a source split and optional persisted fold selection

        Args:
            split:  Requested train, val or test split
            input_path:  Optional source path override
        Returns:
            dataset: Scalar events in source order
        """
        dataset = self.project(self.source.dataset(self.options.splits.get(split, split), input_path))
        if split in self.options.event_ids:
            selected = set(self.options.event_ids[split])
            indices = [i for i in range(len(dataset)) if dataset[i].id in selected]
            if len(indices) != len(selected):
                raise ValueError("Fold event IDs must match unique source events")
            dataset = Subset(dataset, indices)
        return dataset
