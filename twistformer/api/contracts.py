# Public experiment API
#
# mikael.mieskolainen@cern.ch (2026)

"""Public tensor contracts. Masks use True for valid entries"""

from dataclasses import dataclass, field, fields, replace
from typing import Optional

import torch

@dataclass(frozen=True)
class InputSpec:
    features: tuple[str, ...]
    geometry: str = "none"  # none | p4. p4 is (E, px, py, pz), metric (+---)
    units: tuple[str, ...] = ()
    global_features: tuple[str, ...] = ()
    global_units: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    parameter_units: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Input features must be nonempty and uniquely named")
        if self.geometry not in {"none", "p4"}:
            raise ValueError("Input geometry must be none or p4")
        if self.units and len(self.units) != len(self.features):
            raise ValueError("Input units must match the feature count")
        if len(set(self.global_features)) != len(self.global_features):
            raise ValueError("Global features must be uniquely named")
        if self.global_units and len(self.global_units) != len(self.global_features):
            raise ValueError("Global units must match the global feature count")
        if len(set(self.parameters)) != len(self.parameters) or any(not isinstance(name, str) or not name for name in self.parameters):
            raise ValueError("Conditioning parameters must have unique nonempty names")
        if self.parameter_units and len(self.parameter_units) != len(self.parameters):
            raise ValueError("Parameter units must match the parameter count")

@dataclass(frozen=True)
class TaskSpec:
    name: str  # classification | gaussian_regression | covariant_regression
    outputs: tuple[str, ...]
    units: tuple[str, ...] = ()
    signal_class: Optional[int] = None

    def __post_init__(self):
        if self.name not in {"classification", "gaussian_regression", "covariant_regression", "flow_matching", "flow_nll"}:
            raise ValueError("Unknown task")
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("Task outputs must be nonempty and uniquely named")
        if self.name == "classification" and len(self.outputs) < 2:
            raise ValueError("Classification requires at least two classes")
        if self.units and len(self.units) != len(self.outputs):
            raise ValueError("Target units must match the output count")
        if self.name == "covariant_regression":
            if len(self.outputs) % 4:
                raise ValueError("Covariant targets must be complete (E,px,py,pz) groups")
            if self.units and any(len(set(self.units[i : i + 4])) != 1 for i in range(0, len(self.outputs), 4)):
                raise ValueError("All components of a four-vector must have the same unit")
        if self.signal_class is not None and (self.name != "classification" or not 0 <= self.signal_class < len(self.outputs)):
            raise ValueError("signal_class must index a classification output")

@dataclass
class ClassificationRows:
    """One detector descriptor vector, label and physical weight per event"""

    features: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    global_features: torch.Tensor | None = None
    balance_values: torch.Tensor | None = None
    observables: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class Sample:
    features: torch.Tensor  # (N,F), unpadded
    target: Optional[torch.Tensor] = None
    p4: Optional[torch.Tensor] = None  # (N,4), separate from learned features
    weight: float = 1.0
    id: str = ""
    target_mask: Optional[torch.Tensor] = None
    token_weights: Optional[torch.Tensor] = None
    global_features: Optional[torch.Tensor] = None  # (G), event scalars
    observables: dict[str, float] = field(default_factory=dict)
    parameters: Optional[torch.Tensor] = None  # (P), external hypothesis values
    source: Optional[torch.Tensor] = None
    flow_condition: Optional[torch.Tensor] = None
    flow_log_det: Optional[torch.Tensor] = None

@dataclass
class Batch:
    features: torch.Tensor  # (B,N,F)
    mask: torch.Tensor  # (B,N)
    sample_weights: torch.Tensor  # (B,)
    ids: tuple[str, ...]
    targets: Optional[torch.Tensor] = None
    p4: Optional[torch.Tensor] = None
    target_mask: Optional[torch.Tensor] = None
    token_weights: Optional[torch.Tensor] = None
    global_features: Optional[torch.Tensor] = None  # (B,G)
    observables: dict[str, torch.Tensor] = field(default_factory=dict)
    parameters: Optional[torch.Tensor] = None  # (B,P)
    source: Optional[torch.Tensor] = None
    flow_condition: Optional[torch.Tensor] = None
    flow_log_det: Optional[torch.Tensor] = None

    def pin_memory(self):
        """
        Pin particle tensors for asynchronous host-to-device transfers

        Returns:
            batch: New Batch with page-locked tensor storage
        """
        return replace(
            self,
            **{f.name: getattr(self, f.name).pin_memory() for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)},
        )

    def to(self, device, non_blocking=False):
        """
        Move batch tensors while preserving event identifiers

        Args:
            device:  Torch destination device
            non_blocking:  Allow asynchronous tensor transfers
        Returns:
            batch: New Batch on the requested device
        """
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
                for f in fields(self)
                if isinstance(getattr(self, f.name), torch.Tensor)
            },
        )

@dataclass
class EncoderOutput:
    features: torch.Tensor  # (B,N,D) or (B,D) for an invariant event readout

@dataclass
class Prediction:
    logits: Optional[torch.Tensor] = None  # (B,1) sigmoid or (B,C) softmax raw logits
    mean: Optional[torch.Tensor] = None  # (B,R)
    std: Optional[torch.Tensor] = None  # (B,R), strictly positive
    mixture_logits: Optional[torch.Tensor] = None  # (B,K), joint hypothesis weights
    component_mean: Optional[torch.Tensor] = None  # (B,K,R), component hypotheses
    component_std: Optional[torch.Tensor] = None  # (B,K,R), conditional widths
    flow_loss: Optional[torch.Tensor] = None
    flow_weights: Optional[torch.Tensor] = None
    log_prob: Optional[torch.Tensor] = None
    samples: Optional[torch.Tensor] = None

    @property
    def probabilities(self):
        """
        Convert raw logits to normalized class probabilities

        Returns:
            probabilities: Sigmoid pair (1-p,p) or multiclass softmax (B,C)
        """
        if self.logits is None:
            raise TypeError("Probabilities require classification predictions")
        logits = self.logits.float()
        if logits.shape[-1] == 1:
            return torch.cat((-logits, logits), dim=-1).sigmoid()
        return logits.softmax(-1)

    @property
    def log_std(self):
        if self.std is None:
            raise TypeError("log_std requires regression predictions")
        return self.std.log()

@dataclass
class LossResult:
    numerator: torch.Tensor
    normalizer: torch.Tensor

    @property
    def loss(self):
        """Return the signed weighted mean and retain zero for an empty loss"""
        empty = self.normalizer == 0
        mean = self.numerator / self.normalizer.masked_fill(empty, 1)
        return mean.masked_fill(empty & (self.numerator != 0), float("nan"))

# Expose event tensors to compiled graphs and DDP output traversal
torch.export.register_dataclass(Batch, serialized_type_name="twistformer.Batch")
torch.export.register_dataclass(Prediction, serialized_type_name="twistformer.Prediction")
