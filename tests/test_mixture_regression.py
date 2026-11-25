# Joint multimodal likelihood and physical prediction validation
#
# mikael.mieskolainen@cern.ch (2026)

import math
import numpy as np
import pytest
import torch
from twistformer.api.config import TaskConfig, ModelConfig, parse_config
from twistformer.api.contracts import TaskSpec, Sample, Prediction
from twistformer.data.batching import collate_samples
from twistformer.data.preprocessing import Preprocessor
from twistformer.tasks.supervised import Head, SupervisedTask
from twistformer.evaluation.metrics import regression_metrics


def test_joint_mixture_loss_masks_and_permutation():
    """
    Compare joint likelihood with Torch and preserve missing-coordinate marginals
    """
    spec = TaskSpec("gaussian_regression", ("dx", "dy", "dz"))
    config = TaskConfig(name="gaussian_regression", mixture_components=2)
    task = SupervisedTask(spec, config)
    means = torch.tensor([[[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]], requires_grad=True)
    stds = torch.full_like(means, 0.3)
    logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
    prediction = Prediction(
        mean=torch.zeros(1, 3),
        std=torch.ones(1, 3),
        mixture_logits=logits,
        component_mean=means,
        component_std=stds,
    )
    batch = collate_samples([Sample(torch.ones(1, 1), torch.tensor([2.0, 2.0, 2.0]))])
    reference = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(means, stds), 1),
    )
    loss = task.compute_loss(prediction, batch).loss
    torch.testing.assert_close(loss, -reference.log_prob(batch.targets).mean() / 3)
    loss.backward()
    assert torch.isfinite(means.grad).all() and torch.isfinite(logits.grad).all()
    prediction.mixture_logits = logits.flip(1)
    prediction.component_mean = means.flip(1)
    prediction.component_std = stds.flip(1)
    torch.testing.assert_close(task.compute_loss(prediction, batch).loss, loss)
    batch.targets[0, 1:] = float("nan")
    marginal = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Normal(means[:, :, 0], stds[:, :, 0]),
    )
    torch.testing.assert_close(
        task.compute_loss(prediction, batch).loss,
        -marginal.log_prob(batch.targets[:, 0]).mean(),
    )
    batch.targets[:] = float("nan")
    assert task.compute_loss(prediction, batch).loss == 0


def test_mixture_initialization_normalization_and_calibration():
    """
    Keep distinct hypotheses and evaluate the density instead of moment Gaussian NLL
    """
    spec = TaskSpec("gaussian_regression", ("dx",))
    cfg = TaskConfig(name="gaussian_regression", mixture_components=2)
    head = Head(4, spec, cfg, ModelConfig(width=4, head_zero_init=True))
    prediction = head(torch.ones(3, 4))
    assert not torch.equal(
        prediction.component_mean[:, 0], prediction.component_mean[:, 1]
    )
    weights = prediction.mixture_logits.softmax(-1)[..., None]
    torch.testing.assert_close(
        prediction.mean, (weights * prediction.component_mean).sum(1)
    )
    pre = Preprocessor(
        {"target_mean": torch.tensor([10.0]), "target_std": torch.tensor([3.0])}
    )
    restored = pre.inverse_prediction(prediction)
    torch.testing.assert_close(
        restored.component_mean, prediction.component_mean * 3 + 10
    )
    torch.testing.assert_close(restored.component_std, prediction.component_std * 3)
    data = dict(
        targets=np.array([[-2.0], [2.0]]),
        mean=np.zeros((2, 1)),
        std=np.full((2, 1), np.sqrt(4.01)),
        component_mean=np.array([[[-2.0], [2.0]]] * 2),
        component_std=np.full((2, 2, 1), 0.1),
        mixture_logits=np.zeros((2, 2)),
        weights=np.ones(2),
    )
    metrics = regression_metrics(data, spec)
    assert metrics["dx"]["nll"] == pytest.approx(
        math.log(2) + math.log(0.1) + 0.5 * math.log(2 * math.pi)
    )
    assert metrics["mixture"]["joint_nll"] == pytest.approx(metrics["dx"]["nll"])
    assert metrics["dx"]["pit_mean"] == pytest.approx(0.5)
    assert metrics["dx"]["coverage_1sigma"] == 1
    assert [row["coverage_1sigma"] for row in metrics["dx"]["bins"]] == [1, 1]


def test_mixture_learns_two_vertex_solutions():
    """
    Resolve two displaced vertices for identical measured inputs
    """
    spec = TaskSpec("gaussian_regression", ("dx", "dy", "dz"))
    cfg = TaskConfig(name="gaussian_regression", mixture_components=2, min_std=0.05)
    head = Head(4, spec, cfg, ModelConfig(width=4, head_zero_init=True))
    task = SupervisedTask(spec, cfg)
    samples = [
        Sample(torch.ones(1, 1), torch.full((3,), sign * 2.0)) for sign in [-1, 1] * 8
    ]
    batch = collate_samples(samples)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.04)
    x = torch.ones(len(samples), 4)
    for _ in range(250):
        optimizer.zero_grad()
        loss = task.compute_loss(head(x), batch).loss
        loss.backward()
        optimizer.step()
    prediction = head(x[:1])
    modes = prediction.component_mean[0].detach().numpy()
    assert np.linalg.norm(modes - 2, axis=-1).min() < 0.2
    assert np.linalg.norm(modes + 2, axis=-1).min() < 0.2
    assert prediction.component_std.max() < 0.2
    assert prediction.mixture_logits.softmax(-1).min() > 0.3


@pytest.mark.parametrize(
    "task,components", [("classification", 2), ("gaussian_regression", 0)]
)
def test_mixture_config_validation(task, components):
    """
    Reject invalid component counts and classification mixtures
    """
    with pytest.raises(ValueError, match="mixture_components"):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "task": {"name": task, "mixture_components": components},
            }
        )
