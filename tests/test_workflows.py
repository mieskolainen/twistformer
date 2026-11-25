# Physics and workflow validation
#
# mikael.mieskolainen@cern.ch (2026)

import os
import subprocess
import sys
import numpy as np
import torch
import pytest
from twistformer import parse_config, build_experiment, load_predictor, Trainer
from twistformer.training.checkpoint import read_checkpoint
from twistformer.evaluation.runner import evaluate
from twistformer.evaluation.metrics import classification_metrics, regression_metrics
from twistformer.api.contracts import TaskSpec
from twistboard.outputs import append_epoch_metrics, read_history


def config(task="classification", **training):
    return parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 24,
                "val_size": 8,
                "test_size": 8,
                "max_tokens": 4,
            },
            "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.1},
            "task": {"name": task},
            "training": {
                "epochs": 2,
                "batch_size": 5,
                "eval_batch_size": 4,
                "save_every": 1,
                "keep_checkpoints": 3,
                "scheduler": "cosine",
                **training,
            },
        }
    )


def test_resume_respects_early_stopping(tmp_path, monkeypatch):
    """Reopen an early-stopped run without taking another optimizer step"""
    from twistformer.training import engine

    cfg = config(epochs=5, patience=1, device="cpu")
    monkeypatch.setattr(engine, "evaluate_validation", lambda *args, **kwargs: {"val_loss": 1.0})
    root = tmp_path / "stopped"
    best = Trainer(root).fit(build_experiment(cfg))
    last = root / "checkpoints/last.pt"
    assert read_checkpoint(last)["epoch"] == 1
    history = (root / "metrics.csv").read_bytes()
    assert Trainer(root, resume=last).fit(build_experiment(cfg)) == best
    assert (root / "metrics.csv").read_bytes() == history
    assert read_checkpoint(last)["epoch"] == 1


def test_checkpoint_resume_exact_and_fresh_process(tmp_path):
    cfg = config(accumulation=2)
    full = tmp_path / "full"
    Trainer(full).fit(build_experiment(cfg))
    checkpoint = full / "checkpoints/epoch_00000.pt"
    saved = read_checkpoint(checkpoint)
    assert saved["training_elapsed_seconds"] > 0
    saved["training_elapsed_seconds"] = 1234.0
    torch.save(saved, checkpoint)
    resumed = tmp_path / "resumed"
    Trainer(resumed, resume=checkpoint).fit(build_experiment(cfg))
    reference = read_checkpoint(full / "checkpoints/last.pt")
    restored = read_checkpoint(resumed / "checkpoints/last.pt")
    assert restored["training_elapsed_seconds"] >= 1234.0
    for key, value in reference["model"].items():
        torch.testing.assert_close(value, restored["model"][key], atol=0, rtol=0)
    assert reference["step"] == restored["step"]
    predictor = load_predictor(full / "checkpoints/best.pt", "cpu")
    samples = predictor.experiment.driver.dataset("test")
    prediction = predictor.predict(samples, batch_size=3)
    output = tmp_path / "fresh.npz"
    subprocess.run(
        [
            sys.executable,
            "scripts/predict.py",
            "--checkpoint",
            str(full / "checkpoints/best.pt"),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "3",
        ],
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
    )
    with np.load(output) as f:
        np.testing.assert_allclose(
            f["logits"], prediction.logits.numpy(), atol=0, rtol=0
        )


@pytest.mark.parametrize("task", ["classification", "gaussian_regression"])
def test_train_evaluate_plots(tmp_path, task):
    run = tmp_path / task
    best = Trainer(run).fit(build_experiment(config(task)))
    predictor = load_predictor(best, "cpu")
    samples = predictor.experiment.driver.dataset("val")
    output = tmp_path / f"eval_{task}"
    result = evaluate(predictor, samples, output, batch_size=3)
    assert (output / "metrics.json").exists() and (output / "predictions.npz").exists()
    assert list((output / "plots").glob("*.png")) and list(
        (output / "plots").glob("*.pdf")
    )
    assert np.isfinite(result["preprocessed_loss"])
    epoch = read_checkpoint(best)["epoch"]
    row = read_history(run)[epoch]
    if task == "classification":
        assert row["val_auc"] == pytest.approx(result["macro_auc"])
    else:
        assert "val_auc" not in row


def test_validation_auc_extends_old_history(tmp_path):
    """
    Preserve earlier loss and learning rate when a classifier resumes

    Args:
        tmp_path:  Training output directory
    """
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,step,val_loss,lr\n0,3,0.7,0.001\n")
    row = {"epoch": 1, "step": 6, "val_loss": 0.6, "val_auc": 0.85, "lr": 0.0005}
    append_epoch_metrics(path, row)
    append_epoch_metrics(path, {**row, "epoch": 2, "val_auc": float("nan")})
    history = read_history(tmp_path)
    assert history[0] == {"epoch": 0, "step": 3, "val_loss": 0.7, "lr": 0.001}
    assert history[1] == row
    assert history[2]["val_loss"] == 0.6 and "val_auc" not in history[2]


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_weighted_accumulation_equals_large_batch(tmp_path, checkpoint_blocks):
    """
    Preserve weighted gradients across microbatches and block recomputation

    Args:
        tmp_path:  Temporary training output directory
        checkpoint_blocks:  Recompute block activations during backward
    """
    # SGD isolates reduction correctness. Adam can amplify rounding noise in the
    # analytically zero gradient of attention's key bias
    a = config(
        epochs=1, batch_size=12, accumulation=1, scheduler="none", optimizer="sgd"
    )
    a.model.dropout = 0
    b = config(
        epochs=1, batch_size=4, accumulation=3, scheduler="none", optimizer="sgd"
    )
    b.model.dropout = 0
    b.model.checkpoint_blocks = checkpoint_blocks
    # Same dataset order, two windows of twelve examples
    Trainer(tmp_path / "a").fit(build_experiment(a))
    Trainer(tmp_path / "b").fit(build_experiment(b))
    ma = read_checkpoint(tmp_path / "a/checkpoints/last.pt")["model"]
    mb = read_checkpoint(tmp_path / "b/checkpoints/last.pt")["model"]
    for key in ma:
        torch.testing.assert_close(ma[key], mb[key], atol=2e-6, rtol=2e-5)


def test_roc_missing_class_and_zero_background():
    data = {
        "targets": np.array([0, 0, 1, 1]),
        "probabilities": np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]),
        "weights": np.ones(4),
    }
    result, _ = classification_metrics(
        data, TaskSpec("classification", ("a", "b"), signal_class=1)
    )
    assert result["macro_auc"] == 1
    assert result["signal_operating_points"][0]["zero_background"]
    data["targets"] = np.zeros(4, dtype=int)
    result, _ = classification_metrics(
        data, TaskSpec("classification", ("a", "b"), signal_class=1)
    )
    assert result["macro_auc"] is None


def test_metrics_validate_weights_and_assign_calibration_boundaries():
    confidence = np.linspace(0.5, 1, 6)
    data = {
        "targets": np.ones(6, dtype=int),
        "probabilities": np.stack([1 - confidence, confidence], -1),
        "weights": np.ones(6),
    }
    spec = TaskSpec("classification", ("a", "b"))
    metrics, _ = classification_metrics(data, spec)
    assert sum(row["weight"] for row in metrics["calibration"]) == 6
    data["weights"][0] = -1
    signed, _ = classification_metrics(data, spec)
    assert signed["weight_summary"]["negative_events"] == 1
    with pytest.raises(ValueError, match="positive finite"):
        regression_metrics(
            {
                "targets": np.zeros((1, 1)),
                "mean": np.zeros((1, 1)),
                "std": np.zeros((1, 1)),
                "weights": np.ones(1),
            },
            TaskSpec("gaussian_regression", ("response",)),
        )


@pytest.mark.parametrize("normalize_features", [False, True])
@pytest.mark.parametrize("normalize_targets", [False, True])
def test_checkpoint_normalization_switches(
    tmp_path, normalize_features, normalize_targets
):
    cfg = config("gaussian_regression", epochs=1)
    cfg.preprocessing.normalize_features = normalize_features
    cfg.preprocessing.normalize_targets = normalize_targets
    experiment = build_experiment(cfg)
    checkpoint = Trainer(tmp_path / "run").fit(experiment)
    predictor = load_predictor(checkpoint, "cpu")
    saved = read_checkpoint(checkpoint)
    assert saved["config"]["preprocessing"] == {
        "normalize_features": normalize_features,
        "normalize_targets": normalize_targets,
        "normalize_globals": True,
        "normalize_parameters": True,
        "flow_transforms": [],
    }
    for key, value in saved["preprocessing"].items():
        torch.testing.assert_close(
            predictor.preprocessor.state[key], value, atol=0, rtol=0
        )
    sample = experiment.driver.dataset("test")[0]
    prepared = predictor.preprocessor.transform(sample)
    if not normalize_features:
        torch.testing.assert_close(prepared.features, sample.features, atol=0, rtol=0)
    if not normalize_targets:
        torch.testing.assert_close(prepared.target, sample.target, atol=0, rtol=0)
    torch.testing.assert_close(
        predictor.preprocessor.inverse_target(prepared.target), sample.target
    )
    prediction = predictor.predict([sample])
    from twistformer.evaluation.runner import collect_predictions

    exported, _ = collect_predictions(predictor, [sample], require_targets=True)
    np.testing.assert_allclose(exported["mean"], prediction.mean.numpy())
    np.testing.assert_allclose(exported["std"], prediction.std.numpy())
