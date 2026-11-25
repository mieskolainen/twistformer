# Driver data and physics definitions
#
# mikael.mieskolainen@cern.ch (2026)

"""Driver extension point: datasets and preprocessing, never training loops"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Protocol, TYPE_CHECKING
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from twistformer.api.contracts import InputSpec, TaskSpec, Sample
    from twistformer.data.preprocessing import Preprocessor

class Driver(Protocol):
    input_spec: InputSpec
    task_spec: TaskSpec
    evaluation_observables: dict[str, dict[str, str]]

    def comparison_identity(self, split: str, input_path: str | None = None) -> dict:
        """
        Identify reader settings and source files for a common event sample

        Args:
            split: Common data split
            input_path: Optional common input override
        Returns:
            identity: JSON mapping with reader options and a source path list
        """
        ...

    def comparison_dataset(self, reference: Driver, split: str, input_path: str | None = None) -> Dataset:
        """
        Construct model inputs for the reference event selection

        Args:
            reference: Driver defining common source events and targets
            split: Common data split
            input_path: Optional common input override
        Returns:
            dataset: Model inputs with targets in the reference output order
        """
        ...

    def dataset(self, split: str, input_path: str | None = None) -> Dataset:
        """
        Read a split or an explicitly supplied input file

        Args:
            split:  train, val or test
            input_path:  Optional driver-specific path override
        Returns:
            dataset: Indexable raw Sample records
        """
        ...

    def create_preprocessor(self, state=None, *, config=None) -> Preprocessor:
        """
        Construct train-fitted or checkpoint-restored normalization

        Args:
            state:  Saved tensor dictionary, or None before fitting
            config:  Shared normalization switches
        Returns:
            preprocessor: Driver preprocessing instance
        """
        ...

    def augment(self, sample: Sample) -> Sample:
        """
        Apply training augmentation to a raw event

        Args:
            sample:  Raw Sample with physical geometry
        Returns:
            augmented: Sample preserving targets and event identity
        """
        ...

class BaseDriver:
    # Set for binary classifiers that supply one feature row per event
    event_descriptors = False
    evaluation_observables = {}
    comparison_input_options = ()
    comparison_source_options = ("files", "split", "seed", "max_events", "chunk_size")

    def comparison_identity(self, split, input_path=None):
        """
        Describe the reader selection and source files for dataset verification

        Args:
            split: Common data split
            input_path: Optional common input override
        Returns:
            identity: JSON-compatible reader options and source paths
        """
        from dataclasses import asdict

        options = asdict(self.options)
        sources = options.get("files", [])
        if isinstance(sources, dict):
            sources = [sources[split]] if split in sources else []
        return {"options": options, "sources": [input_path] if input_path else sources}

    def comparison_dataset(self, reference, split, input_path=None):
        """
        Read common source events with this model's input representation

        Args:
            reference:  Driver defining common sources and event selection
            split:  Requested data split
            input_path:  Optional common input override
        Returns:
            dataset: Samples labeled in the reference class order
        """
        from dataclasses import asdict
        from twistformer.api.config import TaskConfig

        if type(self) is not type(reference):
            raise ValueError("Comparison requires the same driver reader")
        options, common = asdict(self.options), asdict(reference.options)
        for key in options:
            if key in self.comparison_input_options:
                continue
            if key in self.comparison_source_options or key in ("class_names", "outputs", "signal_class"):
                options[key] = common[key]
            elif options[key] != common[key]:
                raise ValueError(f"Incompatible comparison reader option: {key}")
        driver = type(self)(options, TaskConfig(name=self.task_spec.name))
        if hasattr(self, "descriptor_device"):
            driver.descriptor_device = self.descriptor_device
        return driver.dataset(split, input_path)

    def create_preprocessor(self, state=None, *, config=None):
        """
        Construct train-fitted or checkpoint-restored normalization

        Args:
            state:  Saved tensor dictionary, or None before fitting
            config:  Shared normalization switches
        Returns:
            preprocessor: Driver preprocessing instance
        """
        from twistformer.api.config import PreprocessingConfig
        from twistformer.data.preprocessing import Preprocessor

        covariant = self.task_spec.name == "covariant_regression"
        config = config or PreprocessingConfig(normalize_targets=not covariant)
        if covariant:
            if config.normalize_targets:
                raise ValueError("Covariant targets must retain physical four-vector components")
            if state is not None and (
                "target_mean" not in state
                or "target_std" not in state
                or not (state["target_mean"] == 0).all()
                or not (state["target_std"] == 1).all()
            ):
                raise ValueError("Covariant checkpoints require identity target preprocessing")
        if self.task_spec.name.startswith("flow_"):
            from twistformer.flows.preprocessing import FlowPreprocessor

            preprocessor_type = FlowPreprocessor
        else:
            preprocessor_type = Preprocessor
        return preprocessor_type(
            state,
            normalize_features=config.normalize_features,
            normalize_targets=config.normalize_targets,
            normalize_globals=config.normalize_globals,
            global_dim=len(self.input_spec.global_features),
            normalize_parameters=config.normalize_parameters,
            parameter_dim=len(self.input_spec.parameters),
            **({"flow_transforms": config.flow_transforms} if self.task_spec.name.startswith("flow_") else {}),
        )

    def augment(self, sample):
        """
        Apply training augmentation to a raw event

        Args:
            sample:  Raw Sample with physical geometry
        Returns:
            augmented: Sample preserving targets and event identity
        """
        return sample

def build_driver(config, task):
    """
    Resolve a built-in driver, module class or entry point

    Args:
        config:  Driver name and reader-specific options
        task:  Shared TaskConfig
    Returns:
        driver: Plugin supplying datasets, schemas and preprocessing
    """
    options = dict(config)
    name = options.pop("name")
    if ":" in name:
        module, cls = name.split(":", 1)
        factory = getattr(import_module(module), cls)
    else:
        matches = list(entry_points(group="twistformer.drivers", name=name))
        if len(matches) > 1:
            raise ValueError(f"Ambiguous driver: {name}")
        if matches:
            factory = matches[0].load()
        else:
            if not name.isidentifier():
                raise ValueError(f"Unknown driver: {name}")
            # Restore the photon-first defaults for earlier pion PID checkpoints
            if name == "gammapi0":
                name = "pi0gamma"
                options.setdefault("class_names", ["gamma", "pi0"])
                options.setdefault("class_pdgs", [22, 111])
            module = f"drivers.{name}.driver"
            try:
                factory = getattr(import_module(module), "Driver")
            except ModuleNotFoundError as error:
                if error.name in {f"drivers.{name}", module}:
                    raise ValueError(f"Unknown driver: {name}") from error
                raise
    app = factory(options, task)
    if app.task_spec.name != task.name:
        raise ValueError("Driver/task mismatch")
    return app
