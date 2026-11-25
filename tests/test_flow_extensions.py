# Flow migration, optional backends and physical inference validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch
from twistformer import parse_config, build_experiment, Trainer, load_predictor
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples
from twistformer.flows.config import FlowConfig
from twistformer.flows.continuous import ContinuousFlow
from twistformer.flows.networks import VectorField


@pytest.mark.parametrize("family", ["transformer", "twistformer", "pelican", "gatr"])
def test_geometric_conditioner_gradients(family):
    """
    Propagate flow matching gradients through native event encoders

    Args:
        family:  Event encoder family
    """
    config = parse_config({"driver": {"name": "synthetic", "train_size": 8, "max_tokens": 4},
        "task": {"name": "flow_matching"},
        "model": {"family": family, "width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
        "flow": {"mode": "conditional", "condition": "encoder", "width": 8, "depth": 1, "steps": 2}})
    experiment = build_experiment(config)
    batch = collate_samples(experiment.driver.dataset("train")[:3])
    result = experiment.task.compute_loss(experiment.model(batch), batch)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in experiment.model.conditioner.parameters())


def test_unet_likelihood_higher_derivatives_under_autocast():
    """
    Keep image density gradients finite under an outer CPU autocast context
    """
    config = parse_config({"driver": {"name": "flow", "shape": [1, 4, 4]}, "task": {"name": "flow_nll"},
        "flow": {"shape": [1, 4, 4], "network": "unet", "width": 32, "depth": 1, "steps": 1}})
    experiment = build_experiment(config)
    batch = collate_samples([Sample(torch.zeros(1, 1), target=torch.randn(1, 4, 4)) for _ in range(2)])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = experiment.task.compute_loss(experiment.model(batch), batch)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    gradients = [parameter.grad for parameter in experiment.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def configuration(**flow):
    """
    Configure a small flow workflow

    Args:
        flow:  Flow algorithm overrides
    Returns:
        config: Native experiment configuration
    """
    return parse_config({"driver": {"name": "flow", "train_size": 16, "val_size": 8, "test_size": 8},
                         "task": {"name": "flow_nll" if flow.get("type") == "discrete" else "flow_matching"},
                         "flow": {"width": 8, "depth": 1, "steps": 2, "layers": 1, "trace": "exact", **flow},
                         "training": {"epochs": 1, "batch_size": 8, "device": "cpu", "scheduler": "none"}})


@pytest.mark.parametrize("path", ["FM", "CFM", "COS-CFM", "OT", "S-BRIDGE"])
def test_probability_path_velocity(path):
    """
    Compare analytic velocities with path time derivatives

    Args:
        path:  Probability path
    """
    config = FlowConfig(path=path, sigma=0.2, width=8, depth=1)
    flow = ContinuousFlow(VectorField((2,), 0, config), (2,), path=path, sigma=0.2)
    source, target, noise = torch.randn(3, 4, 2, dtype=torch.float64)
    time = torch.linspace(0.1, 0.9, 4, dtype=torch.float64)[:, None]
    velocity, _ = flow.path_sample(source, target, time, noise=noise)
    positive = flow.path_sample(source, target, time + 1e-5, noise=noise)[1]
    negative = flow.path_sample(source, target, time - 1e-5, noise=noise)[1]
    torch.testing.assert_close(velocity, (positive - negative) / 2e-5, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("method", ["exact", "sinkhorn", "partial", "unbalanced"])
def test_ot_backends_and_label_alignment(method):
    """
    Check positive transport plans and aligned label sampling

    Args:
        method:  POT solver
    """
    pytest.importorskip("ot")
    from twistformer.flows.ot import OTPlanSampler, wasserstein

    coordinates = torch.tensor([[0.0], [0.1], [0.2], [0.3]])
    sampler = OTPlanSampler(method, reg=1)
    plan = sampler.get_map(coordinates, coordinates + 0.5)
    torch.testing.assert_close(plan.sum(), torch.tensor(1., dtype=plan.dtype))
    source, target, source_labels, target_labels = sampler.sample_plan_with_labels(
        coordinates, coordinates + 0.5, torch.arange(4), torch.arange(4))
    torch.testing.assert_close(source, coordinates[source_labels])
    torch.testing.assert_close(target, (coordinates + 0.5)[target_labels])
    trajectory = sampler.sample_trajectory(torch.stack((coordinates, coordinates + 0.5), 1))
    assert trajectory.shape == (4, 2, 1)
    assert wasserstein(coordinates, coordinates + 0.5, power=1) == pytest.approx(0.5)
    assert wasserstein(coordinates, coordinates + 0.5, power=2) == pytest.approx(0.5)


def test_ot_auxiliary_weights_and_conditional_morphing(monkeypatch):
    """
    Keep auxiliary losses, event weights and conditional sources aligned

    Args:
        monkeypatch:  Controlled transport-index fixture
    """
    model = build_experiment(configuration(path="OT")).model
    permutation = torch.tensor([3, 1, 0, 2])
    monkeypatch.setattr(model.flow.ot, "indices", lambda a, b: (torch.arange(len(a)), permutation))
    target, weights = torch.randn(4, 2), torch.tensor([1., 2., 3., 4.])
    torch.manual_seed(30)
    loss, aligned = model.flow.matching_loss(target, weights=weights, auxiliary=weights * 10)
    torch.manual_seed(30)
    plain, _ = model.flow.matching_loss(target, weights=weights)
    torch.testing.assert_close(aligned, weights[permutation])
    torch.testing.assert_close(loss - plain, weights[permutation] * 10)
    config = FlowConfig(width=8, depth=1)
    flow = ContinuousFlow(VectorField((2,), 1, config), (2,), path="OT")
    calls = []
    def paired(source, target):
        """
        Record source groups and return reversed target indices
        """
        calls.append((source.clone(), target.clone()))
        return torch.arange(len(source)), torch.arange(len(target) - 1, -1, -1)
    monkeypatch.setattr(flow.ot, "indices", paired)
    flow.matching_loss(torch.arange(8.).reshape(4, 2), torch.arange(8.).reshape(4, 2), torch.tensor([[0.], [1.], [0.], [1.]]))
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], torch.tensor([[0., 1.], [4., 5.]]))


def test_graph_conditioner_training_and_checkpoint(tmp_path):
    """
    Train graph attention through the native task and checkpoint path

    Args:
        tmp_path:  Temporary run directory
    """
    pytest.importorskip("torch_geometric")
    config = configuration(mode="conditional", condition="graph")
    experiment = build_experiment(config)
    events = experiment.driver.dataset("train")
    batch = collate_samples([events[index] for index in range(4)])
    loss = experiment.task.compute_loss(experiment.model(batch), batch).loss
    loss.backward()
    assert all(parameter.grad is not None for parameter in experiment.model.conditioner.parameters())
    checkpoint = Trainer(tmp_path / "graph").fit(build_experiment(config))
    predictor = load_predictor(checkpoint, "cpu")
    assert predictor.sample(predictor.experiment.driver.dataset("test"), 2).shape == (8, 2, 2)


@pytest.mark.parametrize("kind", ["continuous", "affine", "rqs", "encoder"])
def test_onnx_sampling_from_independent_noise(tmp_path, kind):
    """
    Verify ONNX Runtime sampling on independent base coordinates

    Args:
        tmp_path:  Temporary output directory
        kind:  Flow or condition architecture
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from twistformer.api.export import export_onnx

    options = {"type": "discrete", "coupling": kind} if kind in {"affine", "rqs"} else {}
    config = configuration(**options)
    if kind == "encoder":
        config.flow.mode, config.flow.condition = "conditional", "encoder"
        config.model.width, config.model.heads, config.model.depth = 8, 2, 1
    checkpoint = Trainer(tmp_path / kind).fit(build_experiment(config))
    file = export_onnx(checkpoint, tmp_path / f"{kind}.onnx", max_tokens=2)
    assert file.is_file()


def test_legacy_lzmlp_import_against_original_field(tmp_path):
    """
    Verify migrated weights against the original Supraflow vector field

    Args:
        tmp_path:  Temporary migration files
    """
    from twistformer.flows.legacy import import_checkpoint

    package = types.ModuleType("_flow_reference")
    package.__path__ = [str(Path("supraflow-master/supraflow").resolve())]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location("_flow_reference.nnet", Path(package.__path__[0]) / "nnet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.VNET(module.LZMLP(6, 2, [8], activation="silu", layer_norm=True),
                           x_dim=2, theta_dim=0, freqs=2, DCT_x=False, DCT_theta=False).eval()
    config = configuration(network="lzmlp", frequencies=2, layer_norm=True)
    experiment = build_experiment(config)
    preprocessing = experiment.driver.create_preprocessor(config=config.preprocessing).fit(experiment.driver.dataset("train"))
    x, time = torch.randn(8, 2), torch.linspace(0, 1, 8)[:, None]
    reference = {"x": x, "t": time, "velocity": original(x, time).detach()}
    checkpoint = tmp_path / "legacy.pth"
    torch.save({"model_state": {"module.vnet." + name: value for name, value in original.state_dict().items()}}, checkpoint)
    output = import_checkpoint(checkpoint, tmp_path / "native.pt", config, preprocessing.state, reference)
    predictor = load_predictor(output, "cpu")
    torch.testing.assert_close(predictor.model.flow.vector_field(x, time), reference["velocity"])
    assert predictor.sample(events=3).shape == (3, 1, 2)
    with pytest.raises(ValueError, match="no portable optimizer"):
        Trainer(tmp_path / "resume", resume=output).fit(build_experiment(config))
    with pytest.raises(AssertionError):
        import_checkpoint(checkpoint, tmp_path / "wrong.pt", config, preprocessing.state,
                           {**reference, "velocity": reference["velocity"] + 1})
    assert not (tmp_path / "wrong.pt").exists()


@pytest.mark.parametrize("dataset,shape", [("MNIST-SR", (1, 8, 8)), ("CIFAR10-SR", (3, 8, 8)), ("MNIST-DN", (1, 8, 8))])
def test_image_dequantization_without_download(dataset, shape):
    """
    Validate image axes, source alignment and deterministic dequantization

    Args:
        dataset:  Image operation
        shape:  Channel and spatial dimensions
    """
    from drivers.flow.driver import ImageEvents, Options

    raw_shape = (8, 8) if shape[0] == 1 else (8, 8, 3)
    storage = types.SimpleNamespace(data=torch.full((4, *raw_shape), 127, dtype=torch.uint8))
    options = Options(dataset=dataset, shape=list(shape))
    events = ImageEvents(storage, [0, 2], options, "train")
    sample = events[0]
    assert sample.target.shape == shape
    torch.testing.assert_close(sample.target, events[0].target)
    assert (sample.target >= 127 / 256).all() and (sample.target < 128 / 256).all()
    if "SR" in dataset:
        assert sample.flow_condition.shape == (shape[0], 2, 2)
    else:
        assert sample.source.shape == shape


def test_tau_flow_physical_workflow(tmp_path):
    """
    Train a calorimeter-conditioned tau flow and export physical diagnostics

    Args:
        tmp_path:  ROOT fixture and output directory
    """
    pytest.importorskip("uproot")
    from test_tau3_regression import root_fixture
    from twistformer.evaluation.runner import evaluate

    file = tmp_path / "tau.root"
    root_fixture(file)
    config = parse_config({"driver": {"name": "tau3_regression", "files": [str(file)], "max_hits": 4},
                           "task": {"name": "flow_matching"}, "model": {"width": 8, "heads": 2, "depth": 1},
                           "flow": {"mode": "conditional", "condition": "encoder", "width": 8, "depth": 1,
                                    "steps": 2, "trace": "exact", "evaluation_samples": 8},
                           "training": {"epochs": 1, "batch_size": 8, "device": "cpu", "scheduler": "none"},
                           "evaluation": {"bootstrap_samples": 0}})
    checkpoint = Trainer(tmp_path / "run").fit(build_experiment(config))
    predictor = load_predictor(checkpoint, "cpu")
    events = predictor.experiment.driver.dataset("test")
    result = evaluate(predictor, events, tmp_path / "evaluation", batch_size=4)
    assert np.isfinite(result["density_nll"])
    with np.load(tmp_path / "evaluation/predictions.npz") as arrays:
        assert arrays["sample_observables"].shape[1:] == (8, 7)
