# Driver data and physics definitions
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass, field, replace
import torch
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec, TaskSpec
from twistformer.geometry.lorentz_hypergraphs import REFERENCE_NAMES
from drivers.base import BaseDriver
from .graphs import BASES
from .reader import JetGraphs, Jets
from .features import KINEMATICS15, constituent_features

@dataclass
class Options:
    files: dict[str, str] = field(default_factory=dict)
    signal_class: int | None = 1
    max_particles: int = 200
    max_events: int = -1
    train_start: int = 0
    seed: int = 101
    rotate: bool = True
    feature_set: str = "p4"
    compile_graphs: bool = False
    lund_backend: str = "native"
    lund_threads: int = 0
    graph_features: list[str] = field(default_factory=list)
    reference_features: bool = False
    weight_column: str | None = None

class Driver(BaseDriver):
    comparison_input_options = ("max_particles", "feature_set", "rotate", "compile_graphs", "graph_features",
                                "lund_backend", "lund_threads", "reference_features")
    evaluation_observables = {
        "jet_pt": {"label": "Jet pT", "unit": "GeV"},
        "jet_energy": {"label": "Jet energy", "unit": "GeV"},
        "jet_mass": {"label": "Jet mass", "unit": "GeV"},
        "n_constituents": {"label": "Constituents", "unit": ""},
    }

    def __init__(self, options, task):
        self.options = strict_dataclass(Options, options)
        self.descriptor_device = "cpu"
        if type(self.options.compile_graphs) is not bool:
            raise ValueError("driver.compile_graphs must be true or false")
        if self.options.lund_backend not in ("native", "torch"):
            raise ValueError("driver.lund_backend must be native or torch")
        if type(self.options.lund_threads) is not int or not 0 <= self.options.lund_threads < 2**31:
            raise ValueError("driver.lund_threads must be a nonnegative CPU worker count")
        if task.name != "classification" or self.options.max_particles < 1:
            raise ValueError("Top tagging supplies classification targets and needs max_particles > 0")
        if self.options.max_events != -1 and self.options.max_events < 1:
            raise ValueError("max_events must be positive or -1 for all events")
        if type(self.options.train_start) is not int or self.options.train_start < 0:
            raise ValueError("driver.train_start must be a nonnegative row offset")
        feature_set = self.options.feature_set
        if feature_set in ("rest_hypergraphs", "lorentz_hypergraphs") and self.options.lund_backend != "native":
            raise ValueError("Rest-frame hypergraphs require driver.lund_backend=native")
        if type(self.options.reference_features) is not bool:
            raise ValueError("driver.reference_features must be true or false")
        if self.options.reference_features and (
            self.options.lund_backend != "native"
            or feature_set not in ("lund_graphs", "rest_hypergraphs", "lorentz_hypergraphs")
        ):
            raise ValueError("Reference features require a native Lund or hypergraph basis")
        self.event_descriptors = feature_set in BASES
        selected = self.options.graph_features
        if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
            raise ValueError("driver.graph_features must be a list of names")
        references = REFERENCE_NAMES if self.options.reference_features else ()
        if selected and (not self.event_descriptors or len(set(selected)) != len(selected)
                         or any(name not in BASES[feature_set] and name not in references for name in selected)):
            raise ValueError("driver.graph_features needs distinct names from the selected graph basis")
        if self.event_descriptors:
            names = tuple(selected) or BASES[feature_set]
            self.input_spec = InputSpec(names + tuple(name for name in references if name not in names))
        elif feature_set in {"p4", "kinematics15"}:
            features = KINEMATICS15 if feature_set == "kinematics15" else KINEMATICS15[:4]
            units = ("GeV",) * 4
            if feature_set == "kinematics15":
                units += ("GeV", "", "rad", "GeV", "", "rad", "", "", "", "", "")
            self.input_spec = InputSpec(features, "p4", units)
        else:
            raise ValueError(f"Unknown top-tag feature set: {feature_set}")
        self.task_spec = TaskSpec("classification", ("QCD", "top"), signal_class=self.options.signal_class)

    def dataset(self, split, input_path=None):
        path = input_path or self.options.files.get(split)
        if not path:
            raise ValueError(f"No top-tag file configured for {split}")
        start = self.options.train_start if split == "train" and input_path is None else 0
        if self.event_descriptors:
            jets = Jets(path, replace(self.options, feature_set="p4"), start)
            return JetGraphs(jets, self.options.feature_set, self.descriptor_device, self.options.compile_graphs,
                             self.input_spec.features, self.options.lund_backend, self.options.lund_threads)
        return Jets(path, self.options, start)

    def augment(self, sample):
        if not self.options.rotate or self.event_descriptors:
            return sample
        p = sample.p4.clone()
        angle = torch.rand(()) * (2 * torch.pi)
        c, s = angle.cos(), angle.sin()
        px, py = p[:, 1].clone(), p[:, 2].clone()
        p[:, 1], p[:, 2] = c * px - s * py, s * px + c * py
        return replace(sample, features=constituent_features(p, self.options.feature_set), p4=p)
