# Portable particle inference and checkpoint normalization
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import asdict

import pytest
import torch

from twistformer import build_experiment, parse_config, export_onnx
from twistformer.training.checkpoint import VERSION

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")


def test_signed_plane_export_compression():
    """
    Preserve signed contractions from zero to large physical scales
    """
    from twistformer.models.twistformer import _onnx_asinh

    positive = torch.tensor([0, 1e-300, 1e-12, 0.5, 1, 2, 1e12, 1e300], dtype=torch.float64)
    values = torch.cat((-positive, positive))
    torch.testing.assert_close(_onnx_asinh(values), values.asinh(), atol=1e-15, rtol=1e-14)


def checkpoint_file(tmp_path, model, task=None, ema=False, driver=None):
    """
    Save fitted inference state without a training run

    Args:
        tmp_path:  Temporary output directory
        model:  Model-specific steering
        task:  Optional supervised task steering
        ema:  Select different EMA weights for inference
        driver:  Optional raw input driver
    Returns:
        path: Versioned checkpoint file
    """
    config = parse_config(
        {
            "driver": driver
            or {"name": "synthetic", "train_size": 8, "max_tokens": 4},
            "model": {"width": 8, "heads": 2, "depth": 1, **model},
            "task": task or {},
            "preprocessing": {
                "normalize_targets": (task or {}).get("name") != "covariant_regression"
            },
            "training": {"device": "cpu"},
        }
    )
    experiment = build_experiment(config)
    preprocessor = experiment.driver.create_preprocessor(
        config=config.preprocessing
    )
    preprocessor.fit(
        experiment.driver.dataset("train"),
        regression=config.task.name != "classification",
    )
    weights = experiment.model.state_dict()
    averaged = {key: value.clone() for key, value in weights.items()}
    if ema:
        averaged["head.linear.bias"] += 0.75
    path = tmp_path / "saved.pt"
    torch.save(
        {
            "schema_version": VERSION,
            "config": config.to_dict(),
            "input_spec": asdict(experiment.driver.input_spec),
            "task_spec": asdict(experiment.driver.task_spec),
            "model": weights,
            "ema": {"model": averaged} if ema else None,
            "inference_weights": "ema" if ema else "model",
            "preprocessing": preprocessor.state,
        },
        path,
    )
    return path


@pytest.mark.parametrize(
    "model",
    [
        {"family": "transformer"},
        {"family": "transformer", "reference_token": "frame"},
        {"family": "triformer", "triangle_k": 2},
        {"family": "pelican"},
        {"family": "pelican", "auxiliary_features": ["E"]},
        {
            "family": "twistformer",
            "depth": 2,
            "pair_channels": 4,
            "representation_channels": 2,
        },
        {"family": "twistformer", "twistor_detector": True},
        {"family": "twistformer", "reference_token": "time"},
        {"family": "twistformer", "reference_token": "beam_plane"},
        {
            "family": "twistformer",
            "depth": 2,
            "twistor_precision": "float32",
            "twistor_transport": "frame",
            "reference_token": "event",
            "pair_channels": 4,
            "representation_channels": 2,
            "three_point": True,
            "twistor_qk": True,
            "pooling": "pma",
        },
        *[
            {"family": "gatr", "gatr_algebra": algebra, "gatr_mv_channels": 2}
            for algebra in ("euclidean", "projective", "conformal", "lorentz")
        ],
    ],
    ids=[
        "transformer",
        "transformer-frame",
        "triformer",
        "pelican",
        "pelican-auxiliary",
        "twistformer",
        "detector",
        "twistformer-time",
        "twistformer-beam_plane",
        "twistformer-toptag",
        "euclidean",
        "projective",
        "conformal",
        "lorentz",
    ],
)
def test_saved_model_export(tmp_path, model):
    """
    Match ONNX Runtime and saved inference across dynamic batch sizes

    Args:
        tmp_path:  Temporary output directory
        model:  Model family and algebra steering
    """
    import onnx

    checkpoint = checkpoint_file(tmp_path, model)
    output = export_onnx(checkpoint, tmp_path / "model.onnx", max_tokens=4)
    graph = onnx.load(output)
    assert {item.name for item in graph.graph.output} == {"logits"}
    assert all(
        item.data_location != onnx.TensorProto.EXTERNAL
        for item in graph.graph.initializer
    )
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param


@pytest.mark.parametrize("task", ["gaussian_regression", "covariant_regression"])
def test_physical_regression_export(tmp_path, task):
    """
    Include target scales and Gaussian mixture parameters in portable inference

    Args:
        tmp_path:  Temporary output directory
        task:  Gaussian mixture or covariant four-vector regression
    """
    model = {"family": "pelican" if task == "covariant_regression" else "transformer"}
    steering = {"name": task}
    if task == "gaussian_regression":
        steering["mixture_components"] = 3
    checkpoint = checkpoint_file(tmp_path, model, steering)
    export_onnx(checkpoint, tmp_path / "regression.onnx", max_tokens=4)


def test_verification_failure_preserves_output(tmp_path, monkeypatch):
    """
    Keep a previous export if runtime verification fails

    Args:
        tmp_path:  Temporary output directory
        monkeypatch:  Inject a verification error
    """
    import twistformer.api.export as exporter

    checkpoint = checkpoint_file(tmp_path, {"family": "transformer"}, ema=True)
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"previous export")

    def fail(*args):
        """
        Simulate a runtime verification failure
        """
        raise ValueError("verification failed")

    monkeypatch.setattr(exporter, "verify_onnx", fail)
    with pytest.raises(ValueError, match="verification failed"):
        export_onnx(checkpoint, destination, max_tokens=4)
    assert destination.read_bytes() == b"previous export"
    assert not list(tmp_path.glob(".onnx-*"))


@pytest.mark.parametrize("regression", [False, True])
def test_export_matches_public_predictor(tmp_path, regression):
    """
    Match the public raw-event prediction API including saved EMA weights

    Args:
        tmp_path:  Temporary output directory
        regression:  Select physical Gaussian mixture outputs
    """
    import onnxruntime as ort
    from torch.nn import functional as F
    from twistformer import load_predictor
    from twistformer.data.batching import collate_samples

    task = (
        {"name": "gaussian_regression", "mixture_components": 3} if regression else None
    )
    checkpoint = checkpoint_file(tmp_path, {"family": "transformer"}, task, ema=True)
    predictor = load_predictor(checkpoint, "cpu")
    samples = predictor.experiment.driver.dataset("test")[:3]
    expected = predictor.predict(samples)
    output = export_onnx(checkpoint, tmp_path / "predict.onnx", max_tokens=4)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(output), options, providers=["CPUExecutionProvider"]
    )
    batch = collate_samples(samples)
    padding = 4 - batch.mask.shape[1]
    feed = {
        "features": F.pad(batch.features, (0, 0, 0, padding)).numpy(),
        "mask": F.pad(batch.mask, (0, padding)).numpy(),
    }
    for name, actual in zip(session.get_outputs(), session.run(None, feed)):
        torch.testing.assert_close(
            torch.from_numpy(actual), getattr(expected, name.name), atol=3e-5, rtol=3e-4
        )


def test_global_features_and_weighted_pooling(tmp_path):
    """
    Export event normalization and externally supplied particle weights

    Args:
        tmp_path:  Temporary NPZ and export directory
    """
    import numpy as np
    import onnx

    path = tmp_path / "events.npz"
    rng = np.random.default_rng(7)
    np.savez(
        path,
        features=rng.normal(size=(4, 4, 4)).astype("float32"),
        global_features=rng.normal(10, 2, size=(4, 2)).astype("float32"),
        token_weights=rng.uniform(0.1, 2, size=(4, 4)).astype("float32"),
        targets=np.arange(4) % 2,
    )
    checkpoint = checkpoint_file(
        tmp_path,
        {"family": "transformer", "pooling": "weighted_mean"},
        driver={
            "name": "generic",
            "files": {split: str(path) for split in ("train", "val", "test")},
            "features": ["x", "y", "z", "e"],
            "outputs": ["a", "b"],
            "global_features": ["energy", "count"],
        },
    )
    output = export_onnx(checkpoint, tmp_path / "globals.onnx", max_tokens=4)
    graph = onnx.load(output)
    assert {item.name for item in graph.graph.input} == {
        "features",
        "mask",
        "global_features",
        "token_weights",
    }


@pytest.mark.parametrize("fusion", ["readout", "tokens"])
def test_parametric_export_matches_hypothesis_override(tmp_path, fusion):
    """
    Export physical hypotheses and compare changed parameters with the predictor

    Args:
        tmp_path:  Input and export directory
        fusion:  Parameter injection location
    """
    import numpy as np
    import onnxruntime as ort
    from twistformer import load_predictor
    from twistformer.data.batching import collate_samples

    rng = np.random.default_rng(13)
    path = tmp_path / "events.npz"
    np.savez(
        path, features=rng.normal(size=(6, 3, 2)).astype(np.float32),
        parameters=np.arange(6, dtype=np.float32)[:, None], targets=np.arange(6) % 2,
    )
    checkpoint = checkpoint_file(
        tmp_path, {"family": "transformer", "parameter_fusion": fusion},
        driver={
            "name": "generic", "files": {"train": str(path), "test": str(path)},
            "features": ["a", "b"], "outputs": ["background", "signal"],
            "parameters": ["mass"], "parameter_units": ["GeV"],
        },
    )
    output = export_onnx(checkpoint, tmp_path / "parameters.onnx", max_tokens=3)
    predictor = load_predictor(checkpoint, "cpu")
    samples = predictor.experiment.driver.dataset("test")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    assert "parameters" in {item.name for item in session.get_inputs()}
    for mass in (1.0, 4.0):
        selected = [samples[i] for i in range(3)]
        batch = collate_samples(selected)
        feed = {
            "features": batch.features.numpy(), "mask": batch.mask.numpy(),
            "parameters": np.full((3, 1), mass, dtype=np.float32),
        }
        expected = predictor.predict(selected, parameters={"mass": mass}).logits.numpy()
        actual = session.run(["logits"], feed)[0]
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
