# External parameter hypotheses for conditional inference
#
# mikael.mieskolainen@cern.ch (2026)

from collections.abc import Mapping
from dataclasses import replace
import torch
from torch.utils.data import Dataset
from .loading import fetch_samples


class ParameterDataset(Dataset):
    def __init__(self, dataset, names, values):
        """
        Set named hypotheses without changing event features or labels

        Args:
            dataset:  Raw event dataset
            names:  Parameter names in model input order
            values:  Mapping from each name to a scalar or one value per event
        """
        if not names:
            raise ValueError("This model does not declare conditioning parameters")
        if not isinstance(values, Mapping) or set(values) != set(names):
            raise ValueError(f"Supply exactly these parameter names: {list(names)}")
        self.dataset = dataset
        columns = []
        for name in names:
            column = torch.as_tensor(values[name], dtype=torch.float32, device="cpu").detach()
            if column.ndim == 0:
                if not torch.isfinite(column):
                    raise ValueError(f"Parameter {name} must be finite")
                column = column.expand(len(dataset))
            if column.shape != (len(dataset),) or not torch.isfinite(column).all():
                raise ValueError(f"Parameter {name} must be finite and scalar or have one value per event")
            columns.append(column)
        self.values = torch.stack(columns, dim=-1)

    def __len__(self):
        """
        Count conditioned events

        Returns:
            size: Original event count
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Attach an independent hypothesis vector to one event

        Args:
            index:  Event index
        Returns:
            sample: Event with the supplied parameter values
        """
        return replace(self.dataset[index], parameters=self.values[index].clone())

    def __getitems__(self, indices):
        """
        Preserve compact reader batch access and event order

        Args:
            indices:  Event indices in sampler order
        Returns:
            samples: Events with independent parameter vectors
        """
        return [
            replace(sample, parameters=self.values[index].clone())
            for index, sample in zip(indices, fetch_samples(self.dataset, indices), strict=True)
        ]
