# Automatic restoration of sigmoid and softmax classifier checkpoints
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from twistformer import build_experiment, load_predictor, parse_config
from twistformer.api.predictor import Predictor
from twistformer.evaluation.runner import collect_predictions
from twistformer.training.checkpoint import VERSION


@pytest.mark.parametrize(
    "family", ["transformer", "triformer", "twistformer", "pelican", "gatr"]
)
@pytest.mark.parametrize("outputs", [1, 2])
@pytest.mark.parametrize("use_ema", [False, True])
def test_classifier_checkpoint_detects_saved_head(tmp_path, family, outputs, use_ema):
    """
    Restore raw logits and weighted evaluation loss from saved classifier heads

    Args:
        tmp_path:  Temporary checkpoint directory
        family:  Model family with a shared classification head
        outputs:  Saved binary logit count
        use_ema:  Select averaged checkpoint weights
    Returns:
        None: Predictions and losses match the saved classifier
    """
    cfg = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 8,
                "test_size": 4,
                "max_tokens": 4,
            },
            "model": {
                "family": family,
                "width": 8,
                "heads": 2,
                "depth": 1,
                "dropout": 0.0,
            },
            "task": {"class_weights": [1.0, 3.0], "label_smoothing": 0.1},
        }
    )
    experiment = build_experiment(cfg)
    assert experiment.model.head.linear.out_features == 1
    if outputs == 2:
        experiment.model.head.linear = torch.nn.Linear(
            experiment.model.head.linear.in_features, 2
        )
    pre = experiment.driver.create_preprocessor(config=cfg.preprocessing)
    pre.fit(experiment.driver.dataset("train"))
    samples = experiment.driver.dataset("test")
    reference = Predictor(experiment, pre, "cpu")
    expected = reference.predict(samples, batch_size=2)
    weights = experiment.model.state_dict()
    state = {
        "schema_version": VERSION,
        "config": cfg.to_dict(),
        "input_spec": asdict(experiment.driver.input_spec),
        "task_spec": asdict(experiment.driver.task_spec),
        "model": weights,
        "preprocessing": pre.state,
        "inference_weights": "ema" if use_ema else "model",
    }
    if use_ema:
        state["ema"] = {"model": weights}
        state["model"] = {key: value.clone() for key, value in weights.items()}
        state["model"]["head.linear.bias"].add_(5)
    path = tmp_path / "classifier.pt"
    torch.save(state, path)

    restored = load_predictor(path, "cpu")
    assert restored.model.head.linear.out_features == outputs
    actual = restored.predict(samples, batch_size=2)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    probabilities = (
        expected.logits.softmax(-1)
        if outputs == 2
        else torch.cat((-expected.logits, expected.logits), -1).sigmoid()
    )
    torch.testing.assert_close(actual.probabilities, probabilities)

    data, loss = collect_predictions(
        restored, samples, batch_size=2, require_targets=True
    )
    logits = torch.from_numpy(data["logits"])
    targets = torch.from_numpy(data["targets"]).long()
    event_weights = torch.from_numpy(data["weights"])
    weights = event_weights * torch.tensor(cfg.task.class_weights)[targets]
    per_event = (
        F.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1)
        if outputs == 2
        else F.binary_cross_entropy_with_logits(
            logits[:, 0], targets.float() * 0.9 + 0.05, reduction="none"
        )
    )
    assert loss == pytest.approx(float((weights * per_event).sum() / weights.sum()))

    state["input_spec"]["features"] = ("wrong",)
    torch.save(state, path)
    with pytest.raises(ValueError, match="schema"):
        load_predictor(path, "cpu")

    state["input_spec"] = asdict(experiment.driver.input_spec)
    selected = state["ema"]["model"] if use_ema else state["model"]
    selected["head.linear.weight"] = torch.zeros(outputs, 1)
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_predictor(path, "cpu")
