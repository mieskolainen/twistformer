# Parametric model contracts and conditional predictions
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import numpy as np
import pytest
import torch
from twistformer import InputSpec, Sample, parse_config, build_experiment, Trainer, load_predictor
from twistformer.api.export import InferenceGraph, example_inputs
from twistformer.data.batching import collate_samples, drop_tokens
from twistformer.data.parameters import ParameterDataset
from twistformer.data.preprocessing import Preprocessor
from twistformer.evaluation.runner import collect_predictions, evaluate


def parameter_config(tmp_path, family="transformer", fusion="readout", task="classification"):
    """
    Write a small sample with independent measured and hypothesis inputs

    Args:
        tmp_path:  Input directory
        family:  Model family
        fusion:  Parameter injection location
        task:  Supervised task
    Returns:
        config: Validated experiment steering
    """
    files = {}
    rng = np.random.default_rng(103)
    for split, shift in (("train", 0), ("val", 20), ("test", 40)):
        spatial = rng.normal(size=(12, 3, 3)).astype(np.float32)
        p4 = np.concatenate((np.sqrt(1 + (spatial ** 2).sum(-1, keepdims=True)), spatial), -1)
        parameters = np.column_stack((np.arange(12) + shift, np.ones(12) * 7)).astype(np.float32)
        target = np.arange(12) % 2
        if task == "gaussian_regression":
            target = (p4[:, :, 0].sum(1) + parameters[:, 0])[:, None]
        elif task == "covariant_regression":
            target = p4.sum(1)
        path = tmp_path / f"{split}.npz"
        np.savez(path, features=p4, p4=p4, parameters=parameters, targets=target, global_features=rng.normal(size=(12, 1)))
        files[split] = str(path)
    model = {"family": family, "width": 8, "heads": 2, "depth": 1, "dropout": 0.0, "parameter_fusion": fusion}
    if family == "gatr":
        model.update(gatr_mv_channels=2, gatr_algebra="lorentz")
    return parse_config({
        "driver": {
            "name": "generic", "files": files, "features": ["E", "px", "py", "pz"], "geometry": "p4",
            "parameters": ["mass", "coupling"], "parameter_units": ["GeV", ""],
            "global_features": ["measured"],
            "outputs": ["background", "signal"] if task == "classification" else (
                ["E", "px", "py", "pz"] if task == "covariant_regression" else ["response"]
            ),
        },
        "model": model, "task": {"name": task},
        "preprocessing": {"normalize_targets": task != "covariant_regression"},
        "training": {"device": "cpu", "workers": 0, "epochs": 1, "batch_size": 4, "eval_batch_size": 4},
    })


@pytest.mark.parametrize("family", ["transformer", "triformer", "twistformer", "pelican", "gatr"])
@pytest.mark.parametrize("fusion", ["readout", "tokens"])
def test_parameter_model_gradients(tmp_path, family, fusion):
    """
    Check that every encoder learns and responds to external hypotheses

    Args:
        tmp_path:  Input directory
        family:  Model family
        fusion:  Parameter injection location
    """
    cfg = parameter_config(tmp_path, family, fusion)
    cfg.model.global_fusion = "tokens"
    exp = build_experiment(cfg)
    data = exp.driver.dataset("train")
    pre = exp.driver.create_preprocessor(config=cfg.preprocessing).fit(data)
    batch = collate_samples(pre.transform_many([data[i] for i in range(3)]))
    batch.parameters.requires_grad_()
    output = exp.model(batch)
    exp.task.compute_loss(output, batch).loss.backward()
    assert torch.isfinite(batch.parameters.grad).all() and batch.parameters.grad.abs().sum() > 0
    layer = exp.model.parameter_fuse if fusion == "readout" else exp.model.parameter_embed
    assert layer[0].weight.grad.abs().sum() > 0
    changed = exp.model(replace(batch, parameters=batch.parameters + 2))
    assert not torch.allclose(output.logits, changed.logits)
    torch.testing.assert_close(drop_tokens(batch, 0.5).parameters, batch.parameters)
    torch.testing.assert_close(batch.to("cpu").parameters, batch.parameters)
    with pytest.raises(ValueError, match="Parameter shape"):
        exp.model(replace(batch, parameters=None))


@pytest.mark.parametrize("fusion", ["readout", "tokens"])
def test_covariant_parameter_conditioning(tmp_path, fusion):
    """
    Keep physical vector outputs under invariant parameter conditioning

    Args:
        tmp_path:  Input directory
        fusion:  Parameter injection location
    """
    exp = build_experiment(parameter_config(tmp_path, "pelican", fusion, "covariant_regression"))
    data = exp.driver.dataset("train")
    pre = exp.driver.create_preprocessor(config=exp.config.preprocessing).fit(data, regression=True)
    batch = collate_samples([pre.transform(data[0]), pre.transform(data[1])])
    output = exp.model(batch)
    exp.task.compute_loss(output, batch).loss.backward()
    assert output.mean.shape == (2, 4) and torch.isfinite(output.mean).all()


@pytest.mark.parametrize("normalize", [False, True])
def test_parameter_normalization_and_contracts(normalize):
    """
    Fit event moments independently of multiplicity and reject invalid hypotheses

    Args:
        normalize:  Fit affine scales or save an identity transform
    """
    samples = [Sample(torch.zeros(n, 1), parameters=torch.tensor([value, 7.0])) for n, value in ((1, 1.0), (9, 5.0))]
    pre = Preprocessor(normalize_parameters=normalize, parameter_dim=2).fit(samples)
    torch.testing.assert_close(pre.state["parameter_mean"], torch.tensor([3.0, 7.0]) if normalize else torch.zeros(2))
    torch.testing.assert_close(pre.state["parameter_std"], torch.tensor([2.0, 1.0]) if normalize else torch.ones(2))
    for sample, transformed in zip(samples, pre.transform_many(samples), strict=True):
        torch.testing.assert_close(transformed.parameters, pre.transform(sample).parameters)
        torch.testing.assert_close(pre.inverse_parameters(transformed.parameters), sample.parameters)
    for value in (None, torch.ones(3), torch.ones(1, 2), torch.tensor([float("nan"), 1.0])):
        bad = replace(samples[1], parameters=value)
        with pytest.raises(ValueError):
            collate_samples([samples[0], bad])
        with pytest.raises(ValueError):
            Preprocessor(parameter_dim=2).fit([samples[0], bad])
        with pytest.raises(ValueError):
            pre.transform(bad)
        with pytest.raises(ValueError):
            pre.transform_many([samples[0], bad])
    with pytest.raises(ValueError, match="unique"):
        InputSpec(("x",), parameters=("mass", "mass"))
    with pytest.raises(ValueError, match="units"):
        InputSpec(("x",), parameters=("mass",), parameter_units=("GeV", "GeV"))


def test_named_hypotheses_are_ordered_and_independent():
    """
    Broadcast named values in schema order without modifying source events
    """
    samples = [Sample(torch.ones(2, 1), id=str(i)) for i in range(3)]
    data = ParameterDataset(samples, ("mass", "coupling"), {"coupling": 0.5, "mass": [100, 125, 150]})
    torch.testing.assert_close(data[1].parameters, torch.tensor([125.0, 0.5]))
    assert [sample.id for sample in data.__getitems__([2, 0, 2])] == ["2", "0", "2"]
    data[1].parameters.zero_()
    assert data[1].parameters[0] == 125 and samples[1].parameters is None
    for values in ({"mass": 1}, {"mass": 1, "coupling": 2, "other": 3}, {"mass": [1, 2], "coupling": 2}, {"mass": float("nan"), "coupling": 2}):
        with pytest.raises(ValueError):
            ParameterDataset(samples, ("mass", "coupling"), values)


@pytest.mark.parametrize("task", ["classification", "gaussian_regression"])
def test_parametric_training_reload_and_named_inference(tmp_path, task):
    """
    Restore training normalization and evaluate alternative parameter hypotheses

    Args:
        tmp_path:  Input and run directory
        task:  Supervised task
    """
    cfg = parameter_config(tmp_path, fusion="tokens", task=task)
    exp = build_experiment(cfg)
    best = Trainer(tmp_path / "run").fit(exp)
    predictor = load_predictor(best, "cpu")
    assert predictor.experiment.driver.input_spec.parameters == ("mass", "coupling")
    torch.testing.assert_close(predictor.preprocessor.state["parameter_mean"], torch.tensor([5.5, 7.0]))
    events = predictor.experiment.driver.dataset("test")
    hypotheses = {"coupling": 7.0, "mass": np.arange(12, dtype=float)}
    expected = predictor.predict(predictor.condition(events, hypotheses))
    actual = predictor.predict(events, batch_size=3, parameters=hypotheses)
    name = "logits" if task == "classification" else "mean"
    torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(getattr(actual, name), getattr(predictor.predict(events), name))
    assert events[0].parameters[0] == 40
    data, _ = collect_predictions(predictor, events, parameters=hypotheses)
    np.testing.assert_allclose(data["parameter_mass"], np.arange(12), atol=1e-6)
    np.testing.assert_allclose(data["parameter_coupling"], 7.0)
    result = evaluate(predictor, events, tmp_path / "evaluation", plots=False, parameters=hypotheses, bootstrap_samples=0)
    assert np.isfinite(result["preprocessed_loss"])
    graph = InferenceGraph(predictor)
    graph.output_names = [name]
    inputs = example_inputs(predictor, 3)
    assert inputs[-1].shape == (2, 2) and torch.isfinite(graph(*inputs)[0]).all()


def test_npz_external_hypotheses_and_cli(tmp_path):
    """
    Predict from NPZ events without stored parameters using named CLI inputs

    Args:
        tmp_path:  Input and run directory
    """
    from twistformer.cli.evaluate import predict

    cfg = parameter_config(tmp_path)
    exp = build_experiment(cfg)
    best = Trainer(tmp_path / "run").fit(exp)
    source = tmp_path / "unlabeled.npz"
    with np.load(cfg.driver["files"]["test"]) as arrays:
        np.savez(source, **{name: arrays[name] for name in ("features", "p4", "global_features")})
    with pytest.raises(ValueError, match="requires parameters"):
        exp.driver.dataset("test", str(source))
    samples = exp.driver.prediction_dataset(str(source))
    predictor = load_predictor(best, "cpu")
    with pytest.raises(ValueError, match="Parameters"):
        predictor.predict(samples)
    output = tmp_path / "prediction.npz"
    predict([
        "--checkpoint", str(best), "--input", str(source), "--output", str(output), "--device", "cpu",
        "--parameter", "coupling=7", "--parameter", "mass=5",
    ])
    with np.load(output) as arrays:
        np.testing.assert_allclose(arrays["parameter_mass"], 5)
        np.testing.assert_allclose(arrays["parameter_coupling"], 7)
        expected = predictor.predict_prob(samples, parameters={"mass": 5, "coupling": 7})
        np.testing.assert_allclose(arrays["probabilities"], expected.numpy(), atol=1e-6)
