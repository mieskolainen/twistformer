# Flow data, preprocessing and runtime validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
from pathlib import Path
import json
import math
import numpy as np
import pytest
import torch
from twistformer import parse_config, build_experiment, Trainer, load_predictor
from twistformer.api.config import load_config
from twistformer.api.contracts import Sample, Prediction
from twistformer.data.batching import collate_samples
from twistformer.flows.preprocessing import FlowPreprocessor
from twistformer.flows.transforms import transform_chain
from twistformer.training.checkpoint import read_checkpoint


@pytest.mark.parametrize("config,value", [
    ({"name": "affine", "scale": 3.0, "shift": -2.0}, [0.2, 0.6, 0.8]),
    ({"name": "logit"}, [0.2, 0.6, 0.8]),
    ({"name": "cartesian"}, [30.0, 0.6, -0.8]),
    ({"name": "collider"}, [30.0, 0.6, -0.8]),
])
def test_physical_transform_determinants(config, value):
    """
    Compare physical Jacobians with automatic derivatives

    Args:
        config:  Coordinate transform settings
        value:  Physical test point
    """
    value = torch.tensor([value], dtype=torch.float64)
    output, determinant = transform_chain(value, [config])
    restored, reverse_determinant = transform_chain(output, [config], inverse=True)
    torch.testing.assert_close(restored, value)
    torch.testing.assert_close(determinant + reverse_determinant, torch.zeros_like(determinant))
    jacobian = torch.autograd.functional.jacobian(lambda x: transform_chain(x[None], [config])[0][0], value[0])
    torch.testing.assert_close(determinant[0], torch.linalg.slogdet(jacobian).logabsdet)


def test_preprocessing_density_measure_and_image_shape():
    """
    Preserve image shape and the original coordinate density measure
    """
    samples = [Sample(torch.zeros(1, 1), target=torch.rand(1, 4, 4) * 0.8 + 0.1) for _ in range(8)]
    preprocessor = FlowPreprocessor(flow_transforms=[{"name": "logit"}]).fit(samples)
    sample = samples[0]
    normalized = preprocessor.transform(sample)
    torch.testing.assert_close(preprocessor.inverse_target(normalized.target), sample.target, atol=2e-6, rtol=2e-6)
    prediction = preprocessor.inverse_prediction(Prediction(log_prob=normalized.flow_log_det[None]))
    jacobian = torch.autograd.functional.jacobian(
        lambda x: preprocessor.transform(replace(sample, target=x.reshape_as(sample.target))).target.flatten(),
        sample.target.flatten())
    torch.testing.assert_close(prediction.log_prob[0], torch.linalg.slogdet(jacobian).logabsdet, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dataset", ["gaussians", "moons", "two-spirals", "radii-spirals", "spiral-to-spiral",
                                      "gaussians-to-moons", "MOONS", "OLYMPIC", "CHESS", "SWISS", "TREE"])
def test_synthetic_splits_and_native_objectives(dataset):
    """
    Exercise every synthetic workflow with disjoint reproducible splits

    Args:
        dataset:  Synthetic workflow name
    """
    mode = "conditional" if dataset in {"two-spirals", "radii-spirals"} else "morph" if "-to-" in dataset else "unconditional"
    config = parse_config({"driver": {"name": "flow", "dataset": dataset, "train_size": 16, "val_size": 8, "test_size": 8},
                           "task": {"name": "flow_matching"}, "flow": {"mode": mode, "width": 8, "depth": 1, "steps": 2}})
    experiment = build_experiment(config)
    events = experiment.driver.dataset("train")
    assert {event.id for event in events}.isdisjoint(event.id for event in experiment.driver.dataset("test"))
    preprocessor = experiment.driver.create_preprocessor(config=config.preprocessing).fit(events)
    batch = collate_samples(preprocessor.transform_many([events[index] for index in range(4)]))
    loss = experiment.task.compute_loss(experiment.model(batch), batch).loss
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("codec", ["ae", "vae"])
def test_latent_workflow_and_density_domain(tmp_path, codec):
    """
    Train latent codecs and require an observation density definition

    Args:
        tmp_path:  Temporary run directory
        codec:  Deterministic or variational codec
    """
    config = parse_config({"driver": {"name": "flow", "train_size": 16, "val_size": 8, "test_size": 8},
                           "task": {"name": "flow_matching"},
                           "flow": {"space": "latent", "codec": codec, "latent_dim": 2, "width": 8, "depth": 1, "steps": 2},
                           "training": {"epochs": 1, "batch_size": 8, "device": "cpu", "scheduler": "none"}})
    checkpoint = Trainer(tmp_path / codec).fit(build_experiment(config))
    predictor = load_predictor(checkpoint, "cpu")
    generated = predictor.sample(events=5, count=2)
    assert generated.shape == (5, 2, 2)
    with pytest.raises(ValueError, match="no exact observation density"):
        predictor.log_prob(predictor.experiment.driver.dataset("test"))


def test_flow_resume_and_fresh_cli(tmp_path):
    """
    Restore complete optimizer and RNG state and run the sampling CLI

    Args:
        tmp_path:  Temporary run directory
    """
    config = parse_config({"driver": {"name": "flow", "train_size": 16, "val_size": 8, "test_size": 8},
                           "task": {"name": "flow_matching"}, "flow": {"width": 8, "depth": 1, "steps": 2},
                           "training": {"epochs": 2, "batch_size": 8, "device": "cpu", "save_every": 1, "scheduler": "none"}})
    Trainer(tmp_path / "full").fit(build_experiment(config))
    checkpoint = tmp_path / "full/checkpoints/epoch_00000.pt"
    Trainer(tmp_path / "resumed", resume=checkpoint).fit(build_experiment(config))
    reference, restored = [read_checkpoint(tmp_path / name / "checkpoints/last.pt") for name in ("full", "resumed")]
    for key, value in reference["model"].items():
        torch.testing.assert_close(value, restored["model"][key], atol=0, rtol=0)
    from twistformer.cli.flow import main

    output = tmp_path / "samples.npz"
    main(["sample", "--checkpoint", str(checkpoint), "--output", str(output), "--events", "7", "--device", "cpu"])
    with np.load(output) as arrays:
        assert arrays["samples"].shape == (7, 1, 2)


def test_flow_presets_build_without_external_data():
    """
    Resolve all flow presets without reading production datasets
    """
    files = list(Path("configs/flow").glob("*.yml")) + [Path("configs/tau3_regression/flow.yml")]
    for file in files:
        experiment = build_experiment(load_config(file))
        assert experiment.task.spec.name.startswith("flow_")


def test_unet_image_condition_gradients():
    """
    Train the image vector field with low-resolution conditions
    """
    config = parse_config({"driver": {"name": "flow", "dataset": "MNIST-SR", "shape": [1, 8, 8]},
                           "task": {"name": "flow_matching"},
                           "flow": {"shape": [1, 8, 8], "network": "unet", "mode": "conditional", "condition": "image",
                                    "width": 32, "depth": 1, "steps": 2}})
    experiment = build_experiment(config)
    events = [Sample(torch.zeros(1, 1), target=torch.randn(1, 8, 8), flow_condition=torch.randn(1, 2, 2)) for _ in range(2)]
    batch = collate_samples(events)
    result = experiment.task.compute_loss(experiment.model(batch), batch)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert experiment.model.sample(batch, count=1).shape == (2, 1, 1, 8, 8)


@pytest.mark.parametrize("mode", ["jet", "dijet", "graph", "morph"])
def test_scouting_splits_preserve_event_groups(tmp_path, mode):
    """
    Keep jets from one ROOT event in exactly one data split

    Args:
        tmp_path:  Temporary ROOT fixture directory
        mode:  Scouting representation
    """
    import awkward as ak
    import uproot

    file = tmp_path / "jets.root"
    count = 80
    with uproot.recreate(file) as output:
        output.mktree("Events", {name: ak.Array([values] * count) for name, values in {
            "Jet_pt": [30., 25., 10.], "Jet_eta": [0.2, -0.3, 1.], "Jet_phi": [0.1, -1., 2.],
            "Jet_genPt": [32., 26., 11.], "Jet_genEta": [0.21, -0.31, 1.], "Jet_genPhi": [0.11, -1.01, 2.],
        }.items()})
    from drivers.scouting.driver import Driver
    from twistformer.api.config import TaskConfig

    driver = Driver({"files": [str(file)], "mode": mode}, TaskConfig(name="flow_matching"))
    splits = [driver.dataset(split) for split in ("train", "val", "test")]
    identifiers = [{sample.id.rsplit(":jet", 1)[0] for sample in split} for split in splits]
    assert sum(map(len, splits)) == (count if mode == "dijet" else 2 * count)
    assert not (identifiers[0] & identifiers[1] or identifiers[0] & identifiers[2] or identifiers[1] & identifiers[2])
    measured = tmp_path / "measured.root"
    with uproot.recreate(measured) as output:
        output.mktree("Events", {"Jet_pt": ak.Array([[30., 25.]]), "Jet_eta": ak.Array([[0.1, -0.2]]),
                                 "Jet_phi": ak.Array([[0.2, -1.]])})
    events = driver.prediction_dataset(str(measured))
    assert len(events) == (1 if mode == "dijet" else 2)
    assert all(sample.target is None for sample in events)
