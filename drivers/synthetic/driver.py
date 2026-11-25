# Driver data and physics definitions
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass
import torch
from twistformer.randomness import torch_generator
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec, TaskSpec, Sample
from drivers.base import BaseDriver
from twistformer.geometry.lorentz import dot

@dataclass
class Options:
    train_size: int = 256
    val_size: int = 64
    test_size: int = 64
    max_tokens: int = 12
    seed: int = 13
    target_vectors: int = 1
    signal_class: int | None = None

class Driver(BaseDriver):
    evaluation_observables = {"energy": {"label": "Total energy", "unit": "arb"}}

    """Deterministic small physics sets. regression has known input-dependent noise"""

    def __init__(self, options, task):
        if task.name == "classification":
            options = {"signal_class": 1, **options}
        self.options = strict_dataclass(Options, options)
        if min(self.options.train_size, self.options.val_size, self.options.test_size, self.options.max_tokens) < 2:
            raise ValueError("Synthetic sizes and max_tokens must be >= 2")
        self.input_spec = InputSpec(("E", "px", "py", "pz"), "p4", ("arb",) * 4)
        if self.options.target_vectors < 1 or (task.name != "covariant_regression" and self.options.target_vectors != 1):
            raise ValueError("target_vectors must be positive and requires covariant_regression")
        outputs = ("low", "high") if task.name == "classification" else ("response",)
        if task.name == "covariant_regression":
            outputs = tuple(
                f"p{k}_{component}" for k in range(self.options.target_vectors) for component in ("E", "px", "py", "pz")
            )
        self.task_spec = TaskSpec(task.name, outputs, signal_class=self.options.signal_class)

    def dataset(self, split, input_path=None):
        if input_path is not None:
            raise ValueError("Synthetic driver has no input files. use --split")
        offsets = {"train": 0, "val": 1, "test": 2}
        if split not in offsets:
            raise ValueError(f"Unknown split {split}")
        g = torch_generator(self.options.seed + 10000 * offsets[split])
        samples = []
        for i in range(getattr(self.options, f"{split}_size")):
            n = int(torch.randint(2, self.options.max_tokens + 1, (), generator=g))
            p = torch.randn(n, 3, generator=g)
            p4 = torch.cat([p.norm(dim=-1, keepdim=True), p], -1)
            signal = p4[:, 0].mean()
            if self.task_spec.name == "classification":
                target = (signal > 1.6).long()
            elif self.task_spec.name == "covariant_regression":
                total = p4.sum(0)
                fraction = dot(p4, total) / dot(total, total).clamp_min(1e-8)
                powers = torch.arange(1, self.options.target_vectors + 1)
                coefficients = torch.sigmoid(fraction[:, None] * powers)
                target = (coefficients[..., None] * p4[:, None]).sum(0).flatten()
            else:
                target = (signal + (0.08 + 0.08 * signal) * torch.randn((), generator=g)).reshape(1)
            samples.append(
                Sample(p4.clone(), target, p4, id=f"{split}:{i}", observables={"energy": float(p4[:, 0].double().sum())})
            )
        return samples
