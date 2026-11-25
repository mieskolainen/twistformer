# Flow dataset measures, conditional semantics and comparison validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import math
import json
import numpy as np
import pytest
import torch
from twistformer import parse_config, build_experiment, Trainer, load_predictor
from twistformer.api.contracts import Sample, TaskSpec
from twistformer.flows.base import BaseDensity, constrained_sample
from twistformer.flows.evaluation import flow_metrics


@pytest.mark.parametrize("name,dimension", [("POWER", 6), ("MINIBOONE", 4), ("GAS", 4), ("HEPMASS", 4), ("BSDS300", 4)])
def test_raw_benchmark_readers(tmp_path, name, dimension):
    """
    Read benchmark fixtures without fitting normalization on held-out data

    Args:
        tmp_path:  Raw benchmark directory
        name:  Benchmark format
        dimension:  Expected retained feature count
    """
    from drivers.flow.tabular import load_benchmark

    generator = np.random.default_rng(2)
    data = generator.normal(size=(100, 8 if name == "POWER" else dimension))
    if name in {"POWER", "MINIBOONE"}:
        directory = tmp_path / name.lower()
        directory.mkdir()
        np.save(directory / "data.npy", data)
    elif name == "GAS":
        pd = pytest.importorskip("pandas")
        directory = tmp_path / "gas"
        directory.mkdir()
        frame = pd.DataFrame(data)
        for column in ("Meth", "Eth", "Time"):
            frame[column] = 0
        frame.to_pickle(directory / "ethylene_CO.pickle")
    elif name == "HEPMASS":
        pd = pytest.importorskip("pandas")
        directory = tmp_path / "hepmass"
        directory.mkdir()
        pd.DataFrame(np.column_stack((np.ones(100), data))).to_csv(directory / "1000_train.csv", index=False)
        pd.DataFrame(np.column_stack((np.ones(100), data + 1000, np.zeros(100)))).to_csv(directory / "1000_test.csv", index=False)
    else:
        h5py = pytest.importorskip("h5py")
        directory = tmp_path / "BSDS300"
        directory.mkdir()
        with h5py.File(directory / "BSDS300.hdf5", "w") as file:
            for split, values in (("train", data[:80]), ("validation", data[80:90]), ("test", data[90:])):
                file[split] = values
    parts = [load_benchmark(name, tmp_path, split) for split in ("train", "val", "test")]
    assert all(values.shape[1] == dimension and np.isfinite(values).all() for values in parts)
    assert not set(map(tuple, parts[0])) & set(map(tuple, parts[1]))
    assert not set(map(tuple, parts[0])) & set(map(tuple, parts[2]))
    if name == "HEPMASS":
        assert parts[2].mean() > 900


def test_rejection_preserves_conditions_and_temperature():
    """
    Redraw invalid coordinates without assigning another event condition
    """
    conditions = torch.arange(4.)[:, None]
    calls = []
    def draw(count, condition):
        """
        Reject the first attempt for alternate condition values
        """
        calls.append(condition.clone())
        return condition + (10 if len(calls) == 1 else 0)
    output = constrained_sample(draw, 4, conditions, (0, 4), max_attempts=1)
    torch.testing.assert_close(output, conditions)
    torch.testing.assert_close(calls[1], conditions)
    base = BaseDensity((2,))
    first = base.sample(6, generator=torch.Generator().manual_seed(4))
    second = base.sample(6, generator=torch.Generator().manual_seed(4), temperature=2)
    torch.testing.assert_close(second, first * 2)
    with pytest.raises(RuntimeError, match="constraints failed"):
        constrained_sample(lambda count, condition: torch.ones(count, 2) * 10, 2, constraints=(0, 1), max_attempts=1)


def test_reverse_generation_returns_base_coordinates():
    """
    Keep the inverse generative map in the base density measure
    """
    from twistformer.flows.predictor import FlowPredictor

    config = parse_config({"driver": {"name": "flow", "train_size": 8}, "task": {"name": "flow_matching"},
                           "flow": {"width": 8, "depth": 1, "steps": 2}})
    experiment = build_experiment(config)
    events = [replace(experiment.driver.dataset("train")[index], target=torch.tensor([10. + index, 30. + 2 * index])) for index in range(8)]
    preprocessor = experiment.driver.create_preprocessor(config=config.preprocessing).fit(events)
    with torch.no_grad():
        for parameter in experiment.model.parameters():
            parameter.zero_()
    predictor = FlowPredictor(experiment, preprocessor, "cpu")
    base = predictor.transport(events, reverse=True)
    expected = torch.stack([preprocessor.transform(sample).target for sample in events])
    torch.testing.assert_close(base, expected)
    assert not torch.allclose(base, torch.stack([sample.target for sample in events]))


def test_flow_calibration_bootstrap_and_signed_density():
    """
    Keep probability density scores separate from signed event yields
    """
    generator = np.random.default_rng(3)
    samples = generator.normal(size=(12, 32, 2))
    targets = generator.normal(size=(12, 2))
    data = {"samples": samples, "targets": targets, "weights": np.ones(12), "log_prob": -np.ones(12) * 3}
    spec = TaskSpec("flow_matching", ("x", "y"))
    metrics = flow_metrics(data, spec, bootstrap_samples=8)
    assert metrics["x"]["bias"] == pytest.approx(np.mean(targets[:, 0] - samples[:, :, 0].mean(1)))
    assert metrics["density_nll"] == 3
    assert metrics["uncertainties"]["density_nll"]["valid_resamples"] == 8
    signed = flow_metrics({**data, "weights": np.array([-1.] + [1.] * 11)}, spec, bootstrap_samples=8)
    assert signed["density_nll"] is None and signed["density_status"] == "signed_weights"
    assert signed["uncertainties"]["density_nll"]["valid_resamples"] == 0
    image = flow_metrics({**data, "dequantization_levels": np.full(12, 256)}, spec)
    assert image["bits_per_dimension"] == pytest.approx(8 + 3 / (2 * math.log(2)))


def test_likelihood_prediction_uses_one_density_estimate(monkeypatch):
    """Use the same stochastic trace for the flow objective and reported density"""
    from unittest.mock import Mock
    from twistformer.flows.predictor import FlowPredictor

    config = parse_config({"driver": {"name": "flow", "train_size": 8}, "task": {"name": "flow_nll"},
                           "flow": {"width": 8, "depth": 1, "steps": 2, "trace": "hutchinson"}})
    experiment = build_experiment(config)
    events = experiment.driver.dataset("train")
    preprocessor = experiment.driver.create_preprocessor(config=config.preprocessing).fit(events)
    predictor = FlowPredictor(experiment, preprocessor, "cpu")
    batch = next(iter(predictor.loader(events, batch_size=4)))
    estimate = Mock(wraps=experiment.model.flow.log_prob)
    monkeypatch.setattr(experiment.model.flow, "log_prob", estimate)
    prediction = predictor.forward(batch)
    estimate.assert_called_once()
    torch.testing.assert_close(prediction.log_prob, batch.flow_log_det - prediction.flow_loss)


def test_flow_comparison_and_dashboard(tmp_path):
    """
    Compare flow densities on shared events with paired uncertainty bands

    Args:
        tmp_path:  Training and comparison output directory
    """
    from twistformer.evaluation.comparison import compare, comparison_config
    from twistboard.server import list_comparisons

    config = parse_config({"driver": {"name": "flow", "train_size": 16, "val_size": 8, "test_size": 8},
                           "task": {"name": "flow_matching"}, "flow": {"width": 8, "depth": 1, "steps": 2, "trace": "exact"},
                           "training": {"epochs": 1, "batch_size": 8, "device": "cpu", "scheduler": "none"},
                           "evaluation": {"bootstrap_samples": 0}})
    checkpoint = Trainer(tmp_path / "run").fit(build_experiment(config))
    file = tmp_path / "comparison.yml"
    file.write_text(json.dumps({"comparison": {"name": "flows", "dataset": {"reference_checkpoint": str(checkpoint), "split": "test"},
                      "models": [{"label": "first", "checkpoint": str(checkpoint)}, {"label": "second", "checkpoint": str(checkpoint)}],
                      "bootstrap_samples": 4, "plots": False, "device": "cpu"}}))
    document = comparison_config(file)
    compare(document, output=tmp_path / "comparison")
    record = json.loads((tmp_path / "comparison/comparison.json").read_text())
    assert record["status"] == "completed"
    selection = json.loads((tmp_path / "comparison/selections/inclusive/comparison.json").read_text())
    assert selection["models"][0]["metrics"]["density_status"] == "available"
    assert list_comparisons(tmp_path)
    predictor = load_predictor(checkpoint, "cpu")
    events = list(predictor.experiment.driver.dataset("test"))
    events[-1] = replace(events[-1], weight=-1)
    assert predictor.predict(events, batch_size=4).samples.shape[0] == 8
