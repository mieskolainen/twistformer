# Synthetic, tabular and image flow data
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass, field
from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from drivers.base import BaseDriver
from twistformer.randomness import torch_generator
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec, TaskSpec, Sample


@dataclass
class Options:
    dataset: str = "gaussians"
    files: dict[str, str] = field(default_factory=dict)
    shape: list[int] = field(default_factory=lambda: [2])
    parameters: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=lambda: ["context"])
    outputs: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    seed: int = 101
    noise: float = 0.1
    root: str = "data/flow"
    download: bool = False
    dequantize: bool = True
    resolution_scale: int = 4
    allow_legacy_pickle: bool = False


class TensorEvents(Dataset):
    def __init__(self, arrays, split, shape, feature_count):
        """
        Store flow arrays with explicit event identities

        Args:
            arrays:  Target, source and condition arrays
            split:  Split or source file identifier
            shape:  Target event shape
            feature_count:  Particle condition feature count
        """
        self.arrays, self.split, self.shape = arrays, split, tuple(shape)
        self.feature_count = feature_count
        reference = arrays.get("targets", arrays.get("features"))
        if reference is None:
            raise ValueError("Flow arrays require targets or input features")
        self.count = len(reference)
        for key, value in arrays.items():
            if len(value) != self.count:
                raise ValueError(f"Flow array {key} has a different event count")
        if "targets" in arrays and tuple(arrays["targets"].shape[1:]) != self.shape:
            raise ValueError("Flow file target shape differs from driver.shape")
        if "source" in arrays and tuple(arrays["source"].shape[1:]) != self.shape:
            raise ValueError("Flow source shape differs from driver.shape")

    def __len__(self):
        """
        Return the number of events
        """
        return self.count

    def __getitem__(self, index):
        """
        Read one independent event from tensor storage

        Args:
            index:  Event index
        Returns:
            sample: Flow event with physical targets
        """
        arrays = self.arrays
        def tensor(name):
            return torch.as_tensor(arrays[name][index]).float().clone() if name in arrays else None
        features = tensor("features")
        features = torch.zeros(1, self.feature_count) if features is None else features
        if "mask" in arrays:
            features = features[torch.as_tensor(arrays["mask"][index]).bool()]
        return Sample(features=features, target=tensor("targets"), source=tensor("source"),
                      parameters=tensor("parameters"), flow_condition=tensor("flow_condition"),
                      weight=float(arrays["weights"][index]) if "weights" in arrays else 1.0,
                      id=str(arrays["ids"][index]) if "ids" in arrays else f"{self.split}:{index}")


class ImageEvents(Dataset):
    def __init__(self, dataset, indices, options, split):
        """
        Define deterministic image targets and measurement conditions

        Args:
            dataset:  Torchvision dataset with integer image storage
            indices:  Disjoint split indices
            options:  Driver options
            split:  Split identifier
        """
        self.dataset, self.indices, self.options, self.split = dataset, indices, options, split

    def __len__(self):
        """
        Return the image count
        """
        return len(self.indices)

    def __getitem__(self, index):
        """
        Construct a dequantized image and its paired measurement

        Args:
            index:  Split-local event index
        Returns:
            sample: Image flow event
        """
        source_index = int(self.indices[index])
        raw = torch.as_tensor(self.dataset.data[source_index]).float()
        raw = raw.unsqueeze(0) if raw.ndim == 2 else raw.permute(2, 0, 1)
        generator = torch_generator(self.options.seed + source_index + (1000000 if self.split == "test" else 0))
        noise = torch.rand(raw.shape, generator=generator) if self.options.dequantize else torch.zeros_like(raw)
        target = (raw + noise) / 256
        source = condition = None
        if "-SR" in self.options.dataset:
            condition = torch.nn.functional.avg_pool2d(target, self.options.resolution_scale)
        if "-DN" in self.options.dataset:
            source = (target + self.options.noise * torch.randn(target.shape, generator=generator)).clamp(0, 1)
        target = target.reshape(self.options.shape)
        source = None if source is None else source.reshape(self.options.shape)
        return Sample(torch.zeros(1, len(self.options.features)), target=target, source=source,
                      flow_condition=condition, id=f"{self.options.dataset}:{self.split}:{source_index}")


class Driver(BaseDriver):
    def __init__(self, options, task):
        """
        Define flow coordinate schemas and dataset options

        Args:
            options:  Driver configuration mapping
            task:  Flow task configuration
        """
        self.options = strict_dataclass(Options, options)
        o = self.options
        if not task.name.startswith("flow_"):
            raise ValueError("The flow driver requires a flow task")
        if not o.shape or min(o.shape) < 1 or min(o.train_size, o.val_size, o.test_size, o.resolution_scale) < 1:
            raise ValueError("Flow dataset shapes and counts must be positive")
        if o.noise < 0:
            raise ValueError("Flow data noise must be nonnegative")
        if not o.parameters and o.dataset in {"two-spirals", "radii-spirals"}:
            o.parameters = ["class" if o.dataset == "two-spirals" else "radius"]
        outputs = tuple(o.outputs or [f"x{index}" for index in range(math.prod(o.shape))])
        if len(outputs) != math.prod(o.shape):
            raise ValueError("Driver output names must match the target shape")
        self.input_spec = InputSpec(tuple(o.features), parameters=tuple(o.parameters))
        self.task_spec = TaskSpec(task.name, outputs, tuple(o.units))

    def dataset(self, split, input_path=None):
        """
        Read an explicit data file or construct a reproducible split

        Args:
            split:  Train, validation or test split
            input_path:  Optional data file override
        Returns:
            dataset: Raw flow events
        """
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown flow split")
        file = input_path or self.options.files.get(split)
        if file:
            return TensorEvents(self.read_file(file, split), str(Path(file).resolve()), self.options.shape, len(self.input_spec.features))
        if self.options.dataset.startswith(("MNIST", "CIFAR10")):
            return self.images(split)
        if self.options.dataset in {"BSDS300", "GAS", "HEPMASS", "MINIBOONE", "POWER"}:
            from .tabular import load_benchmark

            arrays = {"targets": load_benchmark(self.options.dataset, Path(self.options.root), split, self.options.seed)}
        else:
            arrays = self.synthetic(split)
        return TensorEvents(arrays, f"{self.options.dataset}:{split}", self.options.shape, len(self.input_spec.features))

    def read_file(self, file, split):
        """
        Read portable arrays or an explicitly enabled legacy benchmark file

        Args:
            file:  Input file path
            split:  Benchmark split key
        Returns:
            arrays: Flow tensors
        """
        file = Path(file)
        if file.suffix == ".npz":
            with np.load(file, allow_pickle=False) as result:
                arrays = {name: result[name] for name in result.files}
            split_key = {"train": "trn", "val": "val", "test": "tst"}[split]
            if split_key in arrays:
                return {"targets": arrays[split_key]}
            return arrays
        if file.suffix == ".npy":
            return {"targets": np.load(file, allow_pickle=False)}
        if file.suffix in {".pkl", ".pickle"} and self.options.allow_legacy_pickle:
            import pickle

            with file.open("rb") as stream:
                result = pickle.load(stream)
            key = {"train": "trn", "val": "val", "test": "tst"}[split]
            return {"targets": result[key]}
        raise ValueError("Use NPZ or NPY data, or explicitly enable allow_legacy_pickle for trusted benchmark files")

    def images(self, split):
        """
        Read disjoint image training, validation and test samples

        Args:
            split:  Dataset split
        Returns:
            dataset: Lazy image events
        """
        try:
            from torchvision.datasets import MNIST, CIFAR10
        except ImportError as error:
            raise ImportError("Image benchmarks require twistformer[flow-images]") from error
        factory = MNIST if self.options.dataset.startswith("MNIST") else CIFAR10
        dataset = factory(self.options.root, train=split != "test", download=self.options.download)
        if split == "test":
            indices = torch.arange(len(dataset))
        else:
            order = torch.randperm(len(dataset), generator=torch_generator(self.options.seed))
            boundary = int(0.9 * len(dataset))
            indices = order[:boundary] if split == "train" else order[boundary:]
        indices = indices[:getattr(self.options, f"{split}_size")]
        return ImageEvents(dataset, indices, self.options, split)

    def create_preprocessor(self, state=None, *, config=None):
        """
        Record the image quantization measure with fitted preprocessing

        Args:
            state:  Optional checkpoint state
            config:  Shared normalization configuration
        Returns:
            preprocessor: Flow normalization
        """
        preprocessor = super().create_preprocessor(state, config=config)
        preprocessor.quantization_levels = 256 if self.options.dequantize and self.options.dataset.startswith(("MNIST", "CIFAR10")) else 0
        return preprocessor

    def synthetic(self, split):
        """
        Generate independent synthetic density and transport examples

        Args:
            split:  Split identifier
        Returns:
            arrays: Target, source and optional condition tensors
        """
        o = self.options
        count = getattr(o, f"{split}_size")
        generator = torch_generator(o.seed + {"train": 0, "val": 100000, "test": 200000}[split])
        def normal(*shape):
            return torch.randn(*shape, generator=generator)
        def uniform(*shape):
            return torch.rand(*shape, generator=generator)
        if o.dataset == "normal":
            return {"targets": normal(count, *o.shape)}
        if o.shape != [2]:
            raise ValueError("Two-dimensional synthetic datasets require driver.shape=[2]")
        phase = uniform(count) * 2 * math.pi
        modes = torch.randint(8, (count,), generator=generator)
        gaussians = 2 * torch.stack(((modes * math.pi / 4).cos(), (modes * math.pi / 4).sin()), -1) + o.noise * normal(count, 2)
        theta = uniform(count) * math.pi
        labels = torch.randint(2, (count,), generator=generator)
        moons = torch.stack((theta.cos() * (1 - 2 * labels) + labels,
                             theta.sin() * (1 - 2 * labels) + 0.5 * labels), -1) + o.noise * normal(count, 2)
        radius = uniform(count).sqrt() * 3
        angle = radius * 2 * math.pi / 3
        spirals = torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
        arrays = {}
        if o.dataset == "gaussians":
            target = gaussians
        elif o.dataset in {"moons", "MOONS"}:
            target = moons
        elif o.dataset in {"spirals", "two-spirals", "radii-spirals", "spiral-to-spiral"}:
            target = spirals * (1 - 2 * labels[:, None]) + o.noise * normal(count, 2)
            if o.dataset == "two-spirals":
                arrays["parameters"] = labels[:, None].float()
            elif o.dataset == "radii-spirals":
                condition = 0.5 + uniform(count, 1)
                target = target * condition
                arrays["parameters"] = condition
            elif o.dataset == "spiral-to-spiral":
                arrays["source"] = target @ torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        elif o.dataset == "gaussians-to-moons":
            arrays["source"], target = gaussians, moons
        elif o.dataset == "OLYMPIC":
            centers = torch.tensor([[-2., 1.], [0., 1.], [2., 1.], [-1., 0.], [1., 0.]])
            target = centers[torch.randint(5, (count,), generator=generator)] + torch.stack((phase.cos(), phase.sin()), -1)
            target = target + o.noise * normal(count, 2)
        elif o.dataset == "CHESS":
            x = 4 * uniform(count) - 2
            y = uniform(count) + torch.randint(2, (count,), generator=generator) * 2 - 2 + x.floor().remainder(2)
            target = torch.stack((x, y), -1)
        elif o.dataset == "SWISS":
            angle = 1.5 * math.pi * (1 + 2 * uniform(count))
            target = torch.stack((angle * angle.cos(), angle * angle.sin()), -1) / 5 + o.noise * normal(count, 2)
        elif o.dataset == "TREE":
            target = torch.zeros(count, 2)
            direction = torch.zeros(count)
            for level in range(6):
                direction = direction + (torch.randint(2, (count,), generator=generator) * 2 - 1) * math.pi / 6
                target = target + 0.7**level * torch.stack((direction.sin(), direction.cos()), -1)
            target = target + o.noise * normal(count, 2)
        else:
            raise ValueError(f"Unknown flow dataset {o.dataset}")
        arrays["targets"] = target
        return arrays
