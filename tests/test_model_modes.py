# Training and inference modes for particle models
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import pytest
import torch
from twistformer import Trainer, build_experiment, parse_config
from twistformer.api.predictor import Predictor
from twistformer.evaluation.runner import collect_predictions, evaluate
from twistformer.training import engine


def mode_config(**training):
    """
    Build a small PELICAN run with dropout and BatchNorm in the task head

    Args:
        training:  Training option overrides
    Returns:
        config: Two-epoch synthetic classification run
    """
    return parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 4,
                "val_size": 4,
                "test_size": 4,
                "max_tokens": 4,
            },
            "model": {
                "family": "pelican",
                "width": 8,
                "heads": 2,
                "depth": 1,
                "dropout": 0.3,
                "head_widths": [8],
                "head_hidden_norm": "batch",
                "head_dropout": 0.3,
            },
            "training": {
                "device": "cpu",
                "epochs": 2,
                "batch_size": 2,
                "eval_batch_size": 2,
                **training,
            },
        }
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("ema_eval", [None, False, True])
def test_training_and_validation_modes(tmp_path, monkeypatch, compiled, ema_eval):
    """
    Switch every submodule at epoch boundaries including compiled and EMA models

    Args:
        tmp_path:  Temporary run directory
        monkeypatch:  CPU compiler substitution
        compiled:  Wrap training in torch.compile
        ema_eval:  Disable EMA with None or select its validation policy
    Returns:
        None: Only training uses dropout, gradients and BatchNorm updates
    """
    cfg = mode_config(
        compile=compiled, ema=ema_eval is not None, ema_eval=bool(ema_eval)
    )
    exp = build_experiment(cfg)
    exp.model.eval()
    observations = []

    @torch.compiler.disable
    def record_mode(model, inputs):
        """
        Record the actual mode before each particle batch

        Args:
            model:  Ordinary or EMA model entering forward
            inputs:  Positional forward inputs
        Returns:
            None: Mode and gradient flags are recorded
        """
        assert all(layer.training == model.training for layer in model.modules())
        observations.append(
            (model.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )

    def compile_cpu(model, config):
        """
        Exercise compiled-module mode propagation without native code generation

        Args:
            model:  Training model
            config:  Training options
        Returns:
            model: Compiled wrapper or eager model
        """
        return torch.compile(model, backend="eager") if config.compile else model

    monkeypatch.setattr(engine, "compile_model", compile_cpu)
    handle = exp.model.register_forward_pre_hook(record_mode)
    try:
        Trainer(tmp_path / "run").fit(exp)
    finally:
        handle.remove()
    assert observations == ([(True, True, False)] * 2 + [(False, False, True)] * 2) * 2
    norms = [m for m in exp.model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert norms
    assert all(int(m.num_batches_tracked) == 4 for m in norms)


@pytest.mark.parametrize("entry", ["forward", "predict", "collect", "evaluate"])
def test_inference_reasserts_eval_mode(tmp_path, entry):
    """
    Protect inference when a shared model has returned to training mode

    Args:
        tmp_path:  Evaluation output directory
        entry:  Public prediction or evaluation path
    Returns:
        None: Predictions are deterministic and BatchNorm buffers remain fixed
    """
    exp = build_experiment(mode_config())
    samples = exp.driver.dataset("test")
    preprocessor = exp.driver.create_preprocessor().fit(
        exp.driver.dataset("train"), False
    )
    predictor = Predictor(exp, preprocessor, "cpu")
    original = {name: value.clone() for name, value in predictor.model.named_buffers()}

    def infer():
        """
        Run the selected public inference path

        Returns:
            logits: Detached CPU classification scores
        """
        if entry == "forward":
            batch = next(iter(predictor.loader(samples, batch_size=4)))
            logits = predictor.forward(batch).logits
        elif entry == "predict":
            logits = predictor.predict(samples, batch_size=4).logits
        elif entry == "collect":
            data, _ = collect_predictions(predictor, samples, batch_size=4)
            logits = torch.from_numpy(data["logits"])
        else:
            evaluate(
                predictor,
                samples,
                tmp_path,
                batch_size=4,
                plots=False,
                bootstrap_samples=0,
            )
            with np.load(tmp_path / "predictions.npz") as data:
                logits = torch.from_numpy(data["logits"])
        assert not logits.requires_grad
        assert not any(layer.training for layer in predictor.model.modules())
        return logits

    expected = infer()
    predictor.model.train()
    actual = infer()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, value in predictor.model.named_buffers():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_evaluation_progress(tmp_path, capsys):
    """
    Show inference batches and bootstrap replicas without changing predictions

    Args:
        tmp_path:  Temporary evaluation output
        capsys:  Captured terminal output
    """
    exp = build_experiment(mode_config())
    samples = exp.driver.dataset("test")
    preprocessor = exp.driver.create_preprocessor().fit(
        exp.driver.dataset("train")
    )
    predictor = Predictor(exp, preprocessor, "cpu")
    quiet, _ = collect_predictions(predictor, samples, batch_size=2)
    assert capsys.readouterr().err == ""
    evaluate(
        predictor,
        samples,
        tmp_path,
        batch_size=2,
        plots=False,
        bootstrap_samples=2,
        progress=True,
    )
    output = capsys.readouterr().err
    assert "Evaluation:" in output and "2/2" in output
    assert "ROC bootstrap:" in output
    assert "Calibration bootstrap:" in output
    with np.load(tmp_path / "predictions.npz") as saved:
        np.testing.assert_array_equal(saved["probabilities"], quiet["probabilities"])
