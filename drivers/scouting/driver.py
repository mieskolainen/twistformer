# Scouting jet calibration and conditional generation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import dataclass, field
from pathlib import Path
import glob
import numpy as np
import torch
from drivers.base import BaseDriver
from twistformer.randomness import numpy_generator
from twistformer.api.config import strict_dataclass
from twistformer.api.contracts import InputSpec, TaskSpec, Sample


@dataclass
class Options:
    files: list[str] = field(default_factory=list)
    tree: str = "Events"
    mode: str = "jet"
    split: list[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    seed: int = 101
    max_events: int = 100000
    chunk_size: int = 10000
    min_pt: float = 15.0
    max_pt: float = 7500.0
    max_eta: float = 5.0
    max_delta_r: float = 0.4
    branches: dict[str, str] = field(default_factory=lambda: {
        "pt": "Jet_pt", "eta": "Jet_eta", "phi": "Jet_phi",
        "gen_pt": "Jet_genPt", "gen_eta": "Jet_genEta", "gen_phi": "Jet_genPhi",
    })


def cartesian(values):
    """
    Convert cylindrical three-momenta to Cartesian components

    Args:
        values:  Array with columns pt, eta and phi
    Returns:
        output: Array with columns px, py and pz
    """
    pt, eta, phi = values.T
    return np.stack((pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), -1).astype(np.float32)


class Driver(BaseDriver):
    evaluation_observables = {"pt": {"label": "Measured jet pT", "unit": "GeV"},
                              "multiplicity": {"label": "Selected jet count", "unit": ""}}

    def __init__(self, options, task):
        """
        Configure measured and generator-level jet transport

        Args:
            options:  ROOT reader settings
            task:  Flow task configuration
        """
        self.options = strict_dataclass(Options, options)
        o = self.options
        if not task.name.startswith("flow_") or o.mode not in {"jet", "dijet", "graph", "morph"}:
            raise ValueError("Scouting requires a flow task and jet, dijet, graph or morph mode")
        if len(o.split) != 3 or min(o.split) <= 0 or not np.isclose(sum(o.split), 1):
            raise ValueError("Scouting split fractions must be positive and sum to one")
        if min(o.max_events, o.chunk_size) < 1 or not 0 < o.min_pt < o.max_pt or min(o.max_eta, o.max_delta_r) <= 0:
            raise ValueError("Invalid scouting event limits or fiducial cuts")
        size = 6 if o.mode == "dijet" else 3
        outputs = tuple(f"jet{index // 3}_{('px', 'py', 'pz')[index % 3]}" for index in range(size))
        parameters = tuple(f"measured_{name}" for name in outputs) if o.mode in {"jet", "dijet"} else ()
        features = ("px", "py", "pz", "query") if o.mode == "graph" else ("px", "py", "pz")
        self.input_spec = InputSpec(features, units=("GeV", "GeV", "GeV", "") if o.mode == "graph" else ("GeV",) * 3,
                                    parameters=parameters, parameter_units=("GeV",) * len(parameters))
        self.task_spec = TaskSpec(task.name, outputs, ("GeV",) * size)

    def dataset(self, split, input_path=None):
        """
        Split physical events before expanding their selected jets

        Args:
            split:  Training, validation or test
            input_path:  Optional ROOT file override
        Returns:
            samples: Jet or dijet flow events
        """
        return self._read(split, input_path, read_targets=True)

    def prediction_dataset(self, input_path):
        """
        Read measured jets without generator branches or truth cuts

        Args:
            input_path:  ROOT file with measured jet branches
        Returns:
            samples: Conditional or morphing input events
        """
        return self._read("test", input_path, read_targets=False)

    def _read(self, split, input_path, *, read_targets):
        """
        Read jet events with optional truth and deterministic splitting

        Args:
            split:  Requested data split
            input_path:  Optional file that bypasses split selection
            read_targets:  Read generator branches and apply matching cuts
        Returns:
            samples: Selected jet events
        """
        import awkward as ak
        import uproot

        o = self.options
        split_index = {"train": 0, "val": 1, "test": 2}[split]
        patterns = [input_path] if input_path else o.files
        files = sorted({file for pattern in patterns for file in (glob.glob(pattern) if "://" not in pattern else [pattern])})
        if not files:
            raise FileNotFoundError("No scouting ROOT files matched driver.files")
        generator = numpy_generator(o.seed)
        samples, event_index = [], 0
        branches = list(o.branches.values()) if read_targets else [o.branches[key] for key in ("pt", "eta", "phi")]
        for file in files:
            open_options = {} if "://" in file else {"handler": uproot.source.file.MemmapSource}
            with uproot.open(file, decompression_executor=uproot.source.futures.TrivialExecutor(),
                             interpretation_executor=uproot.source.futures.TrivialExecutor(), **open_options) as source:
                for chunk in source[o.tree].iterate(branches, step_size=o.chunk_size, library="ak"):
                    for row in range(len(chunk)):
                        if event_index >= o.max_events:
                            return samples
                        event_id = event_index
                        event_index += 1
                        selected_split = int(np.searchsorted(np.cumsum(o.split), generator.random(), side="right"))
                        if input_path is None and selected_split != split_index:
                            continue
                        measured = np.stack([ak.to_numpy(chunk[o.branches[key]][row]) for key in ("pt", "eta", "phi")], -1)
                        truth = np.stack([ak.to_numpy(chunk[o.branches[key]][row]) for key in ("gen_pt", "gen_eta", "gen_phi")], -1) if read_targets else measured.copy()
                        if measured.shape != truth.shape:
                            raise ValueError("Measured and generator jet arrays must align")
                        angular = np.arctan2(np.sin(measured[:, 2] - truth[:, 2]), np.cos(measured[:, 2] - truth[:, 2]))
                        valid = np.isfinite(measured).all(-1) & np.isfinite(truth).all(-1)
                        valid &= np.hypot(measured[:, 1] - truth[:, 1], angular) < o.max_delta_r
                        for values in (measured, truth):
                            valid &= (values[:, 0] > o.min_pt) & (values[:, 0] < o.max_pt)
                            valid &= (np.abs(values[:, 1]) < o.max_eta) & (np.abs(values[:, 2]) < np.pi)
                        measured, truth = measured[valid], truth[valid]
                        if len(measured) < 2:
                            continue
                        order = np.argsort(-measured[:, 0], kind="stable")
                        measured, truth = measured[order], truth[order]
                        nodes, targets = torch.from_numpy(cartesian(measured)), torch.from_numpy(cartesian(truth))
                        identifier = f"{Path(file).name}:{event_id}"
                        if o.mode == "dijet":
                            samples.append(Sample(nodes.clone(), targets[:2].flatten() if read_targets else None, parameters=nodes[:2].flatten(), id=identifier,
                                                  observables={"pt": float(measured[0, 0]), "multiplicity": float(len(nodes))}))
                        else:
                            for jet in range(len(nodes)):
                                features = nodes.clone()
                                if o.mode == "graph":
                                    query = torch.zeros(len(nodes), 1)
                                    query[jet] = 1
                                    features = torch.cat((features, query), -1)
                                samples.append(Sample(features, targets[jet] if read_targets else None, id=f"{identifier}:jet{jet}",
                                                      parameters=nodes[jet] if o.mode == "jet" else None,
                                                      source=nodes[jet] if o.mode == "morph" else None,
                                                      observables={"pt": float(measured[jet, 0]), "multiplicity": float(len(nodes))}))
        return samples
