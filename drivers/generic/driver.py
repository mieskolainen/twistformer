# Driver data and physics definitions
#
# mikael.mieskolainen@cern.ch (2026)

"""Generic driver for new domains without writing a Python plug-in"""

from dataclasses import dataclass, field
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec, TaskSpec
from drivers.base import BaseDriver
from .reader import ArrayDataset

@dataclass
class Options:
    files: dict[str, str] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    target_units: list[str] = field(default_factory=list)
    global_features: list[str] = field(default_factory=list)
    global_units: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    parameter_units: list[str] = field(default_factory=list)
    geometry: str = "none"
    signal_class: int | None = None

class Driver(BaseDriver):
    def __init__(self, options, task):
        self.options = strict_dataclass(Options, options)
        o = self.options
        if not o.features or not o.outputs or o.geometry not in {"none", "p4"}:
            raise ValueError("Generic driver requires feature/output names and valid geometry")
        self.input_spec = InputSpec(
            tuple(o.features), o.geometry, tuple(o.units), tuple(o.global_features), tuple(o.global_units),
            tuple(o.parameters), tuple(o.parameter_units),
        )
        self.task_spec = TaskSpec(task.name, tuple(o.outputs), tuple(o.target_units), o.signal_class)

    def dataset(self, split, input_path=None):
        path = input_path or self.options.files.get(split)
        if not path:
            raise ValueError(f"No file configured for {split}")
        return ArrayDataset(path, self.input_spec, self.task_spec)

    def prediction_dataset(self, input_path):
        """
        Read prediction inputs with optional externally supplied hypotheses

        Args:
            input_path:  NPZ input path
        Returns:
            dataset: Events whose missing parameters must be supplied at inference
        """
        return ArrayDataset(input_path, self.input_spec, self.task_spec, allow_missing_parameters=True)
