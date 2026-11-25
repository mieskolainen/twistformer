# Configurable architecture and numerical validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from twistformer import parse_config, build_experiment
from twistformer.data.batching import collate_samples
from twistformer.layers.common import mlp, normalization
from twistformer.layers.physics import JetBias


@pytest.mark.parametrize("norm", ["none", "layer", "rms"])
def test_geglu_reference_and_gradients(norm):
    layer = mlp(4, 7, 4, norm=norm, activation="geglu").double()
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    value, gate = F.linear(x, layer[0].weight, layer[0].bias).chunk(2, -1)
    hidden = value * F.gelu(gate)
    if norm == "layer":
        hidden = F.layer_norm(hidden, (7,), layer[2].weight, layer[2].bias, 1e-5)
    elif norm == "rms":
        hidden = (
            hidden
            * (hidden.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * layer[2].weight
        )
    expected = F.linear(hidden, layer[-1].weight, layer[-1].bias)
    actual = layer(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.square().sum(), x)[0],
    )
    small = torch.zeros(2, 4, dtype=torch.bfloat16)
    rms = normalization("rms", 4)
    assert rms(small).dtype == small.dtype and rms(small).isfinite().all()


@pytest.mark.parametrize(
    "family", ["transformer", "triformer", "twistformer", "pelican"]
)
@pytest.mark.parametrize("task", ["classification", "gaussian_regression"])
def test_deep_head_training_and_architecture_invariance(family, task):
    options = dict(
        head_norm="rms",
        head_layers=3,
        head_expansion=2,
        head_residual=True,
        head_zero_init=True,
    )
    if family in {"transformer", "triformer"}:
        options.update(
            norm="rms",
            embedding="two_stage",
            embedding_norm="layer",
            ffn_activation="geglu",
            ffn_norm="layer",
        )
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 8, "max_tokens": 4},
            "model": dict(
                family=family, width=8, heads=2, depth=2, dropout=0.0, **options
            ),
            "task": {"name": task},
        }
    )
    exp = build_experiment(cfg)
    batch = collate_samples(exp.driver.dataset("train")[:3])
    optimizer = torch.optim.SGD(exp.model.parameters(), lr=0.01)
    assert not exp.model.head.linear.weight.count_nonzero()
    for _ in range(2):
        optimizer.zero_grad()
        loss = exp.task.compute_loss(exp.model(batch), batch).loss
        loss.backward()
        assert loss.isfinite()
        assert all(
            p.grad is None or p.grad.isfinite().all() for p in exp.model.parameters()
        )
        optimizer.step()
    first = exp.model.backbone.embed[0]
    if family in {"transformer", "triformer"}:
        first = first[0]
    assert first.weight.grad.abs().sum() > 0
    exp.model.eval()
    key = "logits" if task == "classification" else "mean"
    expected = getattr(exp.model(batch), key)
    permutation = torch.arange(batch.mask.shape[1] - 1, -1, -1)
    changed = replace(
        batch,
        features=batch.features[:, permutation],
        p4=batch.p4[:, permutation],
        mask=batch.mask[:, permutation],
    )
    torch.testing.assert_close(
        getattr(exp.model(changed), key), expected, atol=2e-6, rtol=2e-5
    )
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 2)),
        p4=F.pad(batch.p4, (0, 0, 0, 2)),
        mask=F.pad(batch.mask, (0, 2)),
    )
    torch.testing.assert_close(
        getattr(exp.model(padded), key), expected, atol=2e-6, rtol=2e-5
    )
    if family == "transformer":
        compiled = torch.compile(exp.model, backend="eager", fullgraph=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = getattr(compiled(batch), key)
            reference = getattr(exp.model(batch), key)
        torch.testing.assert_close(actual, reference)


def test_jet_bias_mask_centering_and_gradients():
    bias = JetBias(2)
    with torch.no_grad():
        bias.to_bias.weight.normal_(std=0.05)
    p = torch.tensor(
        [[[5.0, 3.0, 0.0, 4.0], [5.0, 0.0, 4.0, 3.0], [2.0, 0.0, 0.0, 0.0]]]
    )
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = bias(p, mask)
    assert result.isfinite().all()
    assert result.abs().max() > 0
    torch.testing.assert_close(result.sum(-1), torch.zeros(1, 2, 3), atol=1e-6, rtol=0)
    assert not result.diagonal(dim1=-2, dim2=-1).count_nonzero()
    padded = bias(F.pad(p, (0, 0, 0, 2), value=float("nan")), F.pad(mask, (0, 2)))
    torch.testing.assert_close(padded[..., :3, :3], result)
    assert not padded[..., 3:, :].count_nonzero()
    assert not padded[..., :, 3:].count_nonzero()
    assert not bias(p[:, :1], mask[:, :1]).count_nonzero()
    result.square().sum().backward()
    assert bias.network[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("family", ["transformer", "triformer"])
@pytest.mark.parametrize("zero_scale", [None, "jet_scale_init", "jet_block_scale_init"])
def test_jet_bias_learns_from_zero_initial_attention(family, zero_scale):
    """
    Train jet pair scores from a zero projection or one zero scale

    Args:
        family:  Scalar or triangle attention encoder
        zero_scale:  Optional scale initialized at zero
    Returns:
        None: Attention starts unbiased and the pair network learns
    """
    from tests.test_models import setup

    torch.manual_seed(17)
    options = {zero_scale: 0.0} if zero_scale is not None else {}
    exp, batch = setup(family, pair_bias="jet", **options)
    backbone = exp.model.backbone
    initial = backbone.bias(batch.p4, batch.mask)
    assert all(
        not (block.edge_scale * initial).count_nonzero() for block in backbone.blocks
    )
    optimizer = torch.optim.SGD(exp.model.parameters(), lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        exp.task.compute_loss(exp.model(batch), batch).loss.backward()
        assert all(
            p.grad is None or p.grad.isfinite().all() for p in exp.model.parameters()
        )
        optimizer.step()
    assert backbone.bias.to_bias.weight.grad.abs().sum() > 0
    assert backbone.bias.network[0].weight.grad.abs().sum() > 0
    learned = backbone.bias(batch.p4, batch.mask)
    assert any(
        (block.edge_scale * learned).count_nonzero() for block in backbone.blocks
    )


def test_jet_bias_rejects_two_zero_scales():
    """
    Reject two zero factors that prevent jet pair learning

    Returns:
        None: Configuration and direct layer construction reject dead gates
    """
    from twistformer.api.config import ModelConfig

    options = dict(pair_bias="jet", jet_scale_init=0.0, jet_block_scale_init=0.0)
    with pytest.raises(ValueError, match="both be zero"):
        parse_config({"driver": {"name": "synthetic"}, "model": options})
    with pytest.raises(ValueError, match="both be zero"):
        JetBias(2, ModelConfig(**options))


@pytest.mark.parametrize("family", ["transformer", "triformer"])
@pytest.mark.parametrize("widths", [[], [8], [8, 8, 16], [5, 13]])
def test_embedding_widths_training_and_reload(family, widths):
    """
    Train arbitrary embedding widths and restore their saved configuration

    Args:
        family:  Attention encoder family
        widths:  Embedding hidden widths
    """
    cfg = parse_config({
        "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 4},
        "model": {
            "family": family, "width": 8, "heads": 2, "depth": 1,
            "embedding": widths, "embedding_norm": "layer", "dropout": 0.0,
        },
    })
    exp = build_experiment(cfg)
    model = exp.model.eval()
    batch = collate_samples(exp.driver.dataset("train")[:3])
    layers = [
        layer for layer in model.backbone.embed.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in layers] == list(
        zip([4, *widths], [*widths, 8])
    )
    prediction = model(batch)
    exp.task.compute_loss(prediction, batch).loss.backward()
    assert all(layer.weight.grad.isfinite().all() for layer in layers)
    assert all(layer.weight.grad.abs().sum() > 0 for layer in layers)
    restored = build_experiment(parse_config(cfg.to_dict())).model.eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(batch).logits, prediction.logits)
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 2), value=float("nan")),
        p4=F.pad(batch.p4, (0, 0, 0, 2)),
        mask=F.pad(batch.mask, (0, 2)),
    )
    torch.testing.assert_close(model(padded).logits, prediction.logits)


@pytest.mark.parametrize("kind", ["mlp", "two_stage"])
def test_legacy_embedding_weights_and_outputs(kind):
    """
    Load legacy embedding weights without changing normalization or activations

    Args:
        kind:  Saved embedding architecture name
    """
    cfg = parse_config({
        "driver": {"name": "synthetic"},
        "model": {
            "width": 8, "heads": 2, "embedding": kind,
            "embedding_norm": "layer", "dropout": 0.0,
        },
    })
    legacy = mlp(4, 8, 8, norm="layer")
    if kind == "two_stage":
        legacy = torch.nn.Sequential(legacy, mlp(8, 16, 8, norm="layer"))
    embedding = build_experiment(cfg).model.backbone.embed
    embedding.load_state_dict(legacy.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 4)
    torch.testing.assert_close(embedding(inputs), legacy(inputs))


@pytest.mark.parametrize(
    "options",
    [
        {"norm": "batch"},
        {"embedding": "unknown"},
        {"embedding": [0]},
        {"embedding": [64, -1]},
        {"embedding": [64.0]},
        {"embedding": [True]},
        {"embedding": ["64"]},
        {"embedding": 64},
        {"ffn_activation": "unknown"},
        {"ffn_norm": "batch"},
        {"head_layers": -1},
        {"head_expansion": 0},
        {"head_dropout": 1.0},
        {"embedding_dropout": -0.1},
        {"residual_dropout": 1.0},
        {"family": "pelican", "norm": "batch"},
        {"family": "twistformer", "embedding": "two_stage"},
        {"family": "twistformer", "embedding": [64]},
    ],
)
def test_architecture_configuration_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        parse_config({"driver": {"name": "synthetic"}, "model": options})
