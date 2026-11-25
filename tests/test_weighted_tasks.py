# Shared classifier outputs, class balancing and weighted likelihoods
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from twistformer import load_config, parse_config
from twistformer.api.contracts import InputSpec, TaskSpec, Sample, Prediction
from twistformer.data.batching import collate_samples
from twistformer.data.preprocessing import Preprocessor
from twistformer.models.factory import Model
from twistformer.tasks.supervised import SupervisedTask


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize(
    "family", ["transformer", "triformer", "twistformer", "pelican", "gatr"]
)
def test_classification_contract_across_models(family, classes):
    """
    Check raw output dimensions, probability transforms and weighted gradients
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "model": {
                "family": family,
                "width": 8,
                "heads": 2,
                "depth": 1,
                "dropout": 0.0,
            },
        }
    )
    spec = TaskSpec("classification", tuple(str(i) for i in range(classes)))
    model = Model(InputSpec(("E", "px", "py", "pz"), "p4"), spec, cfg.model, cfg.task)
    p4 = torch.tensor([[2.0, 1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0]])
    batch = collate_samples(
        [Sample(p4, torch.tensor(i), p4=p4, weight=i + 1.0) for i in range(classes)]
    )
    prediction = model(batch)
    assert prediction.logits.shape == (classes, 1 if classes == 2 else classes)
    expected = (
        prediction.logits.sigmoid()[:, 0]
        if classes == 2
        else prediction.logits.softmax(-1)
    )
    torch.testing.assert_close(
        prediction.probabilities[:, 1] if classes == 2 else prediction.probabilities,
        expected,
    )
    task = SupervisedTask(spec, cfg.task)
    task.compute_loss(prediction, batch).loss.backward()
    assert all(
        p.grad.isfinite().all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("smoothing", [0.0, 0.15])
def test_weighted_classification_balancing_and_smoothing(classes, smoothing):
    """
    Match weighted log-likelihoods with one balance convention for all classes
    """
    labels = torch.arange(classes).repeat_interleave(torch.arange(1, classes + 1))
    samples = [
        Sample(torch.ones(1, 1), y, weight=float(i + 1)) for i, y in enumerate(labels)
    ]
    pre = Preprocessor(normalize_features=False).fit(samples, class_count=classes)
    batch = collate_samples([pre.transform(s) for s in samples])
    for c in range(classes):
        torch.testing.assert_close(
            batch.sample_weights[labels == c].sum(),
            batch.sample_weights.sum() / classes,
        )
    manual = torch.arange(1, classes + 1).float()
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "task": {
                "lossfunc": "cross_entropy",
                "label_smoothing": smoothing,
                "class_weights": manual.tolist(),
                "balance_classes": True,
            },
        }
    )
    task = SupervisedTask(
        TaskSpec("classification", tuple(map(str, range(classes)))), cfg.task
    )
    logits = torch.randn(
        len(labels), 1 if classes == 2 else classes, requires_grad=True
    )
    prediction = Prediction(logits=logits)
    target = F.one_hot(labels, classes).float() * (1 - smoothing) + smoothing / classes
    log_probs = (
        torch.cat((F.logsigmoid(-logits), F.logsigmoid(logits)), -1)
        if classes == 2
        else logits.log_softmax(-1)
    )
    per_event = -(target * log_probs).sum(-1)
    weights = batch.sample_weights * manual[labels]
    result = task.compute_loss(prediction, batch)
    torch.testing.assert_close(result.normalizer, weights.sum())
    torch.testing.assert_close(result.loss, (weights * per_event).sum() / weights.sum())
    scaled = task.compute_loss(
        prediction, replace(batch, sample_weights=batch.sample_weights * 7)
    ).loss
    torch.testing.assert_close(scaled, result.loss)
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "task_name,components",
    [
        ("gaussian_regression", 1),
        ("gaussian_regression", 3),
        ("covariant_regression", 1),
    ],
)
def test_event_weighted_regression_with_missing_targets(task_name, components):
    """
    Compare weighted regression with replicated events and masked observations
    """
    outputs = (
        ("E", "px", "py", "pz") if task_name == "covariant_regression" else ("x", "y")
    )
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "model": (
                {"family": "pelican"} if task_name == "covariant_regression" else {}
            ),
            "task": {
                "name": task_name, "mixture_components": components,
                "lossfunc": "mse" if task_name == "covariant_regression" else "gaussian_nll",
            },
            "preprocessing": {"normalize_targets": task_name != "covariant_regression"},
        }
    )
    task = SupervisedTask(TaskSpec(task_name, outputs), cfg.task)
    targets = torch.randn(4, len(outputs))
    targets[1, -1] = float("nan")
    targets[3] = float("nan")
    batch = collate_samples(
        [
            Sample(torch.ones(1, 1), y, weight=w)
            for y, w in zip(targets, [2.0, 3.0, 0.0, 5.0])
        ]
    )
    mean = torch.randn_like(targets, requires_grad=True)
    prediction = Prediction(mean=mean, std=torch.ones_like(mean))
    if components > 1:
        prediction.component_mean = (
            mean[:, None].expand(-1, components, -1)
            + torch.arange(components)[None, :, None]
        )
        prediction.component_std = torch.ones_like(prediction.component_mean)
        prediction.mixture_logits = torch.randn(4, components)
    result = task.compute_loss(prediction, batch)
    unit = task.compute_loss(
        prediction, replace(batch, sample_weights=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    ).loss
    second = task.compute_loss(
        prediction, replace(batch, sample_weights=torch.tensor([0.0, 1.0, 0.0, 0.0]))
    ).loss
    torch.testing.assert_close(result.loss, (2 * unit + 3 * second) / 5)
    assert result.normalizer.item() == 5
    result.loss.backward()
    assert torch.isfinite(mean.grad).all()
    assert not mean.grad[2:].count_nonzero()
    assert mean.grad[1, -1] == 0
    scaled = task.compute_loss(prediction, replace(batch, sample_weights=batch.sample_weights * 7))
    torch.testing.assert_close(scaled.loss, result.loss)


@pytest.mark.parametrize("name,lossfunc", [
    ("classification", "mse"), ("classification", "gaussian_nll"),
    ("gaussian_regression", "cross_entropy"), ("gaussian_regression", "mse"),
    ("covariant_regression", "gaussian_nll"), ("covariant_regression", "cross_entropy"),
    ("classification", "unknown"),
])
def test_incompatible_task_losses(name, lossfunc):
    """
    Reject losses that do not match the configured prediction type

    Args:
        name:  Task type
        lossfunc:  Incompatible loss name
    """
    with pytest.raises(ValueError, match="task.lossfunc"):
        parse_config({"driver": {"name": "synthetic"}, "task": {"name": name, "lossfunc": lossfunc}})


def test_all_driver_cards_declare_loss():
    """
    Check explicit or inherited loss names and resolved config serialization
    """
    from twistformer.api.config import load_yaml_mapping

    expected = {"classification": "cross_entropy", "gaussian_regression": "gaussian_nll", "covariant_regression": "mse",
                "flow_matching": "flow_matching", "flow_nll": "flow_nll"}
    for path in Path("configs").glob("*/*.yml"):
        data = load_yaml_mapping(path)
        if "driver" not in data:
            continue
        cfg = load_config(path)
        assert data["task"]["lossfunc"] == expected[cfg.task.name], str(path)
        assert parse_config(cfg.to_dict()).task.lossfunc == cfg.task.lossfunc


def test_all_classifier_cards_enable_balancing():
    """
    Enforce shared balancing defaults in every shipped classifier card
    """
    from twistformer.api.config import load_yaml_mapping

    for path in Path("configs").glob("*/*.yml"):
        if "driver" not in load_yaml_mapping(path):
            continue
        cfg = load_config(path)
        if cfg.task.name == "classification":
            assert cfg.task.balance_classes, str(path)


def test_binary_head_rejects_two_logit_weights():
    """
    Keep direct head loading strict outside checkpoint restoration
    """
    from twistformer.api.config import ModelConfig, TaskConfig
    from twistformer.tasks.supervised import Head

    head = Head(8, TaskSpec("classification", ("a", "b")), TaskConfig(), ModelConfig())
    state = head.state_dict()
    state["linear.weight"] = torch.randn(2, 8)
    state["linear.bias"] = torch.randn(2)
    with pytest.raises(RuntimeError, match="size mismatch"):
        head.load_state_dict(state)
