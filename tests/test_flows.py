# Flow transport and density validation
#
# mikael.mieskolainen@cern.ch (2026)

import math
import pytest
import torch
from torch import nn
from twistformer.flows.base import BaseDensity
from twistformer.flows.continuous import ContinuousFlow
from twistformer.flows.discrete import DiscreteFlow, InvertibleLinear
from twistformer.flows.config import FlowConfig
from twistformer.flows.networks import DCTFeatureMapping
from twistformer import parse_config, build_experiment, Trainer, load_predictor
from twistformer.data.batching import collate_samples


class LinearField(nn.Module):
    def __init__(self, matrix):
        """
        Construct a linear field with analytic transport

        Args:
            matrix:  Coordinate mixing matrix
        """
        super().__init__()
        self.matrix = nn.Parameter(matrix.clone())

    def forward(self, x_t, t, theta=None):
        """
        Evaluate a linear vector field

        Args:
            x_t:  Coordinates
            t:  Unused time
            theta:  Optional scalar translation
        Returns:
            output: Velocity
        """
        return x_t @ self.matrix.T + (0 if theta is None else theta)


class TimeField(nn.Module):
    def forward(self, x_t, t, theta=None):
        """
        Return the time coordinate as velocity

        Args:
            x_t:  Coordinates
            t:  Time
            theta:  Unused conditions
        Returns:
            output: Velocity
        """
        return t.expand_as(x_t)


@pytest.mark.parametrize("kind,bound", [("uniform", math.sqrt(3)), ("triangle", math.sqrt(6))])
def test_compact_base_support_and_temperature(kind, bound):
    """
    Check compact support, density normalization and temperature
    """
    base = BaseDensity((1,), kind).double()
    grid = torch.linspace(-bound, bound, 10001, dtype=torch.float64).unsqueeze(-1)
    integral = torch.trapezoid(base.log_prob(grid).exp(), grid[:, 0])
    torch.testing.assert_close(integral, torch.tensor(1.0, dtype=torch.float64), atol=1e-7, rtol=1e-7)
    assert torch.isneginf(base.log_prob(torch.tensor([[bound + 1]], dtype=torch.float64))).all()
    one = base.sample(50, generator=torch.Generator().manual_seed(8))
    two = base.sample(50, generator=torch.Generator().manual_seed(8), temperature=2)
    torch.testing.assert_close(two, 2 * one)


def test_heun_uses_current_time_and_returns_all_endpoints():
    """
    Integrate a time-dependent field in both directions
    """
    flow = ContinuousFlow(TimeField(), (2,), steps=8)
    source = torch.zeros(3, 2)
    output = flow.transport(source, trajectory=True)
    assert output.shape == (9, 3, 2)
    torch.testing.assert_close(output[-1], source + 0.5)
    torch.testing.assert_close(flow.transport(output[-1], reverse=True), source, atol=1e-7, rtol=0)


@pytest.mark.parametrize("trace", ["exact", "vmap", "hutchinson", "finite_difference", "finite_hutchinson"])
def test_trace_estimators_and_likelihood_gradients(trace):
    """
    Compare divergence and density gradients with linear transport

    Args:
        trace:  Divergence algorithm
    """
    matrix = torch.diag(torch.tensor([0.2, -0.1], dtype=torch.float64))
    flow = ContinuousFlow(LinearField(matrix), (2,), steps=40, trace=trace, finite_difference=1e-4).double()
    target = torch.tensor([[0.3, -0.7], [1.2, 0.2]], dtype=torch.float64)
    density = flow.log_prob(target, create_graph=True)
    source = target @ torch.matrix_exp(-matrix).T
    expected = flow.base.log_prob(source) - matrix.trace()
    torch.testing.assert_close(density, expected, atol=3e-6, rtol=3e-6)
    density.sum().backward()
    assert flow.vector_field.matrix.grad is not None
    expected_gradient = (source.square().sum(0) - len(source))
    torch.testing.assert_close(flow.vector_field.matrix.grad.diag(), expected_gradient, atol=2e-5, rtol=2e-5)


def test_finite_difference_uses_diagonal_entries():
    """
    Exclude off-diagonal derivatives from the divergence
    """
    matrix = torch.tensor([[0.1, 3.0], [-2.0, 0.4]], dtype=torch.float64)
    flow = ContinuousFlow(LinearField(matrix), (2,), trace="finite_difference").double()
    value = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    _, trace = flow.divergence(value, torch.zeros(4, 1, dtype=torch.float64))
    torch.testing.assert_close(trace, torch.full((4,), 0.5, dtype=torch.float64))


@pytest.mark.parametrize("coupling", ["affine", "kan", "rqs"])
@pytest.mark.parametrize("dimension", [1, 3, 4])
def test_discrete_inverse_and_analytic_jacobian(coupling, dimension):
    """
    Check coupling inversion and exact density determinants

    Args:
        coupling:  Coupling family
        dimension:  Even, odd or scalar dimension
    """
    config = FlowConfig(type="discrete", coupling=coupling, width=8, depth=1, layers=2)
    flow = DiscreteFlow((dimension,), 2, config).double().eval()
    value = torch.randn(3, dimension, dtype=torch.float64) * 0.3
    condition = torch.randn(3, 2, dtype=torch.float64)
    output, determinant = flow(value, condition)
    restored, inverse_determinant = flow(output, condition, inverse=True)
    torch.testing.assert_close(restored, value, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(determinant + inverse_determinant, torch.zeros_like(determinant), atol=2e-7, rtol=0)
    jacobian = torch.autograd.functional.jacobian(lambda x: flow(x[None], condition[:1])[0][0], value[0])
    torch.testing.assert_close(determinant[0], torch.linalg.slogdet(jacobian).logabsdet, atol=2e-7, rtol=2e-7)
    loss = -flow.log_prob(value, condition).mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in flow.parameters() if parameter.requires_grad)


def test_linear_inverse_changes_with_parameters():
    """
    Recompute the inverse after optimizer changes
    """
    layer = InvertibleLinear(3).double()
    value = torch.randn(5, 3, dtype=torch.float64)
    layer(value, inverse=True)
    with torch.no_grad():
        layer.log_diagonal.add_(0.2)
    output, _ = layer(value)
    restored, _ = layer(output, inverse=True)
    torch.testing.assert_close(restored, value)


def test_dct_roundtrip():
    """
    Preserve norms and invert the cosine feature transform
    """
    transform = DCTFeatureMapping(7).double()
    value = torch.randn(6, 7, dtype=torch.float64)
    torch.testing.assert_close(transform.inverse(transform(value)), value, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("task,flow_type", [("flow_matching", "continuous"), ("flow_nll", "continuous"), ("flow_nll", "discrete")])
def test_flow_training_with_event_conditioner(tmp_path, task, flow_type):
    """
    Train native conditional flow tasks and save portable checkpoints

    Args:
        tmp_path:  Local temporary output directory
        task:  Flow objective
        flow_type:  Continuous or discrete architecture
    """
    config = parse_config({
        "driver": {"name": "synthetic", "train_size": 16, "val_size": 8, "test_size": 8, "max_tokens": 4},
        "task": {"name": task},
        "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
        "flow": {"type": flow_type, "mode": "conditional", "condition": "encoder", "width": 8, "depth": 1,
                 "steps": 2, "layers": 1, "trace": "exact"},
        "training": {"epochs": 1, "batch_size": 8, "device": "cpu", "scheduler": "none", "seed": 41},
    })
    experiment = build_experiment(config)
    batch = collate_samples(experiment.driver.dataset("train")[:3])
    result = experiment.task.compute_loss(experiment.model(batch), batch)
    result.loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in experiment.model.conditioner.parameters())
    checkpoint = Trainer(tmp_path / task / flow_type).fit(build_experiment(config))
    assert checkpoint.exists()
    predictor = load_predictor(checkpoint, "cpu")
    events = predictor.experiment.driver.dataset("test")
    prediction = predictor.predict(events, batch_size=4)
    assert prediction.samples.shape == (8, config.flow.evaluation_samples, 1)
    assert torch.isfinite(prediction.log_prob).all()
    torch.testing.assert_close(predictor.sample(events, count=2), predictor.sample(events, count=2, seed=41))
    torch.testing.assert_close(predictor.sample(events, count=2, seed=20), predictor.sample(events, count=2, seed=20))
    from twistformer.evaluation.runner import evaluate

    result = evaluate(predictor, events, tmp_path / task / flow_type / "evaluation", batch_size=4)
    assert math.isfinite(result["density_nll"])
