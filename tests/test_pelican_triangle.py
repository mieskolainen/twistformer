# PELICAN triangle contractions and Lorentz symmetry
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import pytest
import torch
from twistformer import build_experiment, parse_config
from twistformer.api.config import load_config
from twistformer.data.batching import collate_samples
from twistformer.layers.pelican import TriangleInteraction
from tests.test_pelican_covariant import lorentz_matrix


def test_triangle_matches_explicit_triplets_and_gradients():
    """
    Compare the matrix contraction against an explicit sum over ordered triplets

    Returns:
        None: Values and all gradients match with noncontiguous particle masks
    """
    torch.manual_seed(14)
    layer = TriangleInteraction(3, 2, norm="layer").double()
    x = torch.randn(2, 4, 4, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
    valid = mask[:, :, None] & mask[:, None, :]
    clean = x.masked_fill(~valid[..., None], 0)
    normalized = layer.norm(clean)
    left = layer.activation(layer.left(normalized)).masked_fill(~valid[..., None], 0)
    right = layer.activation(layer.right(normalized)).masked_fill(~valid[..., None], 0)
    triplets = left[:, :, :, None] * right[:, None, :, :]
    product = triplets.sum(2) / mask.sum(1)[:, None, None, None]
    expected = (clean + layer.scale * layer.out(product)).masked_fill(
        ~valid[..., None], 0
    )
    actual = layer(x, mask)
    torch.testing.assert_close(actual, expected)
    params = (x, *layer.parameters())
    gradients = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    reference = torch.autograd.grad(expected.square().sum(), params)
    for found, wanted in zip(gradients, reference):
        torch.testing.assert_close(found, wanted, atol=1e-10, rtol=1e-9)
    assert not gradients[0][~valid].count_nonzero()


def test_triangle_connects_i_k_and_k_j():
    """
    Require both edges of a directed path to update the remaining pair

    Returns:
        None: The contraction couples two distinct edges through their shared particle
    """
    layer = TriangleInteraction(1, 1, norm="none", activation="relu", scale=1)
    with torch.no_grad():
        layer.left.weight.fill_(1)
        layer.right.weight.fill_(1)
        layer.left.bias.zero_()
        layer.right.bias.zero_()
        layer.out.weight.fill_(1)
    x = torch.tensor(
        [[[[0.0], [2.0], [0.0]], [[0.0], [0.0], [3.0]], [[0.0], [0.0], [0.0]]]]
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected = x.clone()
    expected[0, 0, 2, 0] = 2
    torch.testing.assert_close(layer(x, mask), expected)
    x[0, 1, 2] = 0
    torch.testing.assert_close(layer(x, mask), x)


def test_triangle_padding_permutation_and_singletons():
    """
    Exclude dirty padding and preserve equivariance across variable multiplicities

    Returns:
        None: Pair outputs are independent of padding and other events
    """
    layer = TriangleInteraction(4, 3).eval()
    x = torch.randn(2, 4, 4, 4)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
    valid = mask[:, :, None] & mask[:, None, :]
    expected = layer(x, mask)
    dirty = x.masked_fill(~valid[..., None], float("nan"))
    torch.testing.assert_close(layer(dirty, mask), expected)
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(
        layer(x[:, order][:, :, order], mask[:, order]), expected[:, order][:, :, order]
    )
    for b in range(2):
        indices = mask[b].nonzero().flatten()
        compact = x[b : b + 1][:, indices][:, :, indices]
        actual = layer(compact, torch.ones(1, len(indices), dtype=torch.bool))
        torch.testing.assert_close(
            actual, expected[b : b + 1][:, indices][:, :, indices]
        )


@pytest.mark.parametrize(
    "task", ["classification", "gaussian_regression", "covariant_regression"]
)
@pytest.mark.parametrize("amp", [False, True])
def test_triangle_model_lorentz_symmetry_and_backward(task, amp):
    """
    Preserve scalar invariance and four-vector covariance through triangle updates

    Args:
        task:  Event prediction task
        amp:  Enable BF16 scalar layers
    Returns:
        None: Every active parameter has finite gradients and outputs transform correctly
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 8, "max_tokens": 5},
            "model": {
                "family": "pelican",
                "width": 8,
                "heads": 2,
                "depth": 2,
                "norm": "rms",
                "dropout": 0.0,
                "pelican_triangle": True,
                "pelican_triangle_channels": 3,
            },
            "task": {"name": task},
            "preprocessing": {"normalize_targets": False},
        }
    )
    exp = build_experiment(cfg)
    batch = collate_samples(exp.driver.dataset("train")[:3])
    batch = replace(batch, p4=batch.p4.double())
    transform = lorentz_matrix()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        prediction = exp.model(batch)
        changed = exp.model(replace(batch, p4=batch.p4 @ transform.T))
        key = "logits" if task == "classification" else "mean"
        expected = getattr(prediction, key)
        if task == "covariant_regression":
            expected = expected @ transform.T
        tolerance = 0.02 if amp else 2e-5
        torch.testing.assert_close(
            getattr(changed, key), expected, atol=tolerance, rtol=tolerance
        )
        loss = exp.task.compute_loss(prediction, batch).loss
    loss.backward()
    for name, p in exp.model.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), name
        if "triangles" in name:
            assert p.grad.abs().sum() > 0, name
    assert len(exp.model.backbone.triangles) == cfg.model.depth


@pytest.mark.parametrize(
    "path,enabled",
    [("configs/toptag/pelican.yml", False), ("configs/synthetic/pelican_covariant.yml", True)],
)
def test_pelican_cards_select_triangle_layers(path, enabled):
    """
    Build the PELICAN baseline and the covariant triangle extension

    Args:
        path:  Configuration card
        enabled:  Expected triangle setting
    Returns:
        None: Only the extension includes triangle interactions
    """
    cfg = load_config(path)
    exp = build_experiment(cfg)
    assert cfg.model.pelican_triangle == enabled
    assert len(exp.model.backbone.triangles) == (cfg.model.depth if enabled else 0)


@pytest.mark.parametrize(
    "options",
    [
        {"family": "transformer", "pelican_triangle": True},
        {"pelican_triangle": True, "pelican_triangle_channels": 0},
        {"pelican_triangle": True, "pelican_triangle_scale": 0},
        {"pelican_triangle_channels": 7},
        {"pelican_triangle_scale": 0.5},
    ],
)
def test_invalid_triangle_configuration(options):
    """
    Reject unused or invalid triangle settings

    Args:
        options:  Invalid model options
    Returns:
        None: Configuration rejects the incompatible settings
    """
    with pytest.raises(ValueError):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "model": {"family": "pelican", **options},
            }
        )


def test_old_pelican_configuration_retains_checkpoint_layout():
    """
    Preserve strict weight loading when triangle options are absent

    Returns:
        None: Legacy pair encoders retain their parameter keys and predictions
    """
    data = {
        "driver": {"name": "synthetic", "train_size": 8, "max_tokens": 4},
        "model": {
            "family": "pelican",
            "width": 8,
            "heads": 2,
            "depth": 1,
            "dropout": 0,
        },
    }
    exp = build_experiment(parse_config(data))
    state = exp.model.state_dict()
    assert not any("triangles" in key for key in state)
    data["model"]["pelican_triangle"] = False
    restored = build_experiment(parse_config(data))
    restored.model.load_state_dict(state, strict=True)
    batch = collate_samples(exp.driver.dataset("train")[:2])
    torch.testing.assert_close(
        exp.model(batch).logits, restored.model(batch).logits, rtol=0, atol=0
    )
