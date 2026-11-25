# Projective point distances and scalar transformer validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from twistformer import build_experiment, load_config, parse_config
from twistformer.api.contracts import InputSpec, Sample
from twistformer.data.batching import collate_samples
from twistformer.layers.attention import ProjectiveAttention, point_distance_features
from twistformer.models.transformer import Transformer


def test_point_metric_and_rigid_motion():
    """
    Recover squared distances and preserve common rotations and translations
    """
    q, k = torch.randn(2, 4, 3, dtype=torch.float64)
    expected = -(q[:, None] - k[None]).square().sum(-1)
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=q.dtype)).Q
    shift = torch.tensor([12.0, -4.0, 8.0], dtype=q.dtype)
    for left, right in ((q, k), (q @ rotation + shift, k @ rotation + shift)):
        actual = (
            point_distance_features(left, True)
            @ point_distance_features(right, False).T
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("reinsert", [False, True])
def test_attention_matches_explicit_distances_and_gradients(reinsert):
    """
    Match lifted attention to direct pair distances and parameter derivatives

    Args:
        reinsert:  Include fixed detector points and learned offsets
    """
    layer = ProjectiveAttention(8, 2, channels=2, reinsert=reinsert).double()
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True, True], [True, False, False, False]])
    coordinates = torch.randn(2, 4, 3, dtype=x.dtype, requires_grad=True)
    bias = torch.randn(2, 2, 4, 4, dtype=x.dtype, requires_grad=True)
    actual = layer(x, mask, bias, coordinates)
    clean = x.masked_fill(~mask[..., None], 0)
    q, k, v = [layer.split(t) for t in layer.qkv(clean).chunk(3, -1)]
    qp, kp = [
        t.reshape(2, 4, 2, 2, 3).transpose(1, 2)
        for t in layer.points(clean).chunk(2, -1)
    ]
    if reinsert:
        raw = coordinates[:, None, :, None].expand(-1, 2, -1, -1, -1)
        qp, kp = torch.cat((qp + raw, raw), -2), torch.cat((kp + raw, raw), -2)
    distance = (qp[:, :, :, None] - kp[:, :, None, :]).square().sum(-1)
    logits = (
        q @ k.transpose(-1, -2)
        - (distance * layer.distance_weight.exp()[None, :, None, None]).sum(-1)
    ) / layer.dim**0.5 + bias
    weights = logits.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
    result = (weights @ v).transpose(1, 2).reshape_as(x)
    expected = layer.out(result).masked_fill(~mask[..., None], 0)
    torch.testing.assert_close(actual, expected)
    parameters = (x, bias, *layer.parameters()) + ((coordinates,) if reinsert else ())
    actual_grad = torch.autograd.grad(
        actual.square().sum(), parameters, retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e)
        assert a.isfinite().all()


@pytest.mark.parametrize("reference", ["none", "frame"])
@pytest.mark.parametrize("pair_bias", ["none", "mandelstam", "jet"])
def test_projective_transformer_mask_permutation_and_amp(reference, pair_bias):
    """
    Preserve set operations and finite mixed-precision training with physical inputs

    Args:
        reference:  Optional laboratory axis tokens
        pair_bias:  Additional physical pair scores
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 5},
            "model": {
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "attention": "projective",
                "projective_coordinates": ["px", "py", "pz"],
                "reference_token": reference,
                "pair_bias": pair_bias,
            },
            "preprocessing": {"normalize_features": False},
        }
    )
    exp = build_experiment(cfg)
    model = exp.model.eval()
    samples = exp.driver.dataset("train")[:3]
    batch = collate_samples(samples)
    expected = model(batch).logits
    individual = torch.cat(
        [model(collate_samples([sample])).logits for sample in samples]
    )
    torch.testing.assert_close(individual, expected)
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 2), value=float("nan")),
        p4=F.pad(batch.p4, (0, 0, 0, 2), value=float("nan")),
        mask=F.pad(batch.mask, (0, 2)),
    )
    torch.testing.assert_close(model(padded).logits, expected)
    order = torch.arange(batch.mask.shape[1] - 1, -1, -1)
    permuted = replace(
        batch,
        features=batch.features[:, order],
        p4=batch.p4[:, order],
        mask=batch.mask[:, order],
    )
    torch.testing.assert_close(model(permuted).logits, expected)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = model(padded)
        loss = exp.task.compute_loss(prediction, padded).loss
    loss.backward()
    assert loss.isfinite()
    for block in model.backbone.blocks:
        assert block.attention.points.weight.grad.abs().sum() > 0
        assert (block.attention.distance_weight.grad.abs().sum(-1) > 0).all()
    assert all(
        p.grad.isfinite().all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tau_projective_presets_and_reload(task):
    """
    Run tau task heads with scaled detector points and restore the model state

    Args:
        task:  Tau prediction task
    """
    cfg = load_config(f"configs/tau3_{task}/transformer_projective.yml")
    exp = build_experiment(cfg)
    assert cfg.model.family == "transformer"
    assert cfg.model.attention == "projective"
    assert cfg.model.projective_coordinates == ["x", "y", "z"]
    assert not cfg.preprocessing.normalize_features
    assert cfg.model.projective_input_scale == 0.001
    samples = []
    for n in (1, 4):
        features = torch.randn(n, len(exp.driver.input_spec.features))
        for name in ("x", "y", "z"):
            features[:, exp.driver.input_spec.features.index(name)] *= 1000
        target = (
            torch.tensor(0)
            if task == "classification"
            else torch.zeros(len(exp.driver.task_spec.outputs))
        )
        samples.append(Sample(features, target))
    batch = collate_samples(samples)
    model = exp.model.eval()
    prediction = model(batch)
    loss = exp.task.compute_loss(prediction, batch).loss
    loss.backward()
    assert loss.isfinite()
    assert all(
        p.grad.isfinite().all() for p in model.parameters() if p.grad is not None
    )
    restored = build_experiment(parse_config(cfg.to_dict())).model.eval()
    restored.load_state_dict(model.state_dict())
    name = "logits" if task == "classification" else "mean"
    torch.testing.assert_close(
        getattr(restored(batch), name), getattr(prediction, name)
    )


def test_coordinate_scaling_and_feature_names():
    """
    Apply one length scale in the declared feature order without scaling energy
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "model": {
                "attention": "projective",
                "projective_coordinates": ["x", "y", "z"],
                "projective_input_scale": 0.001,
            },
            "preprocessing": {"normalize_features": False},
        }
    ).model
    model = Transformer(InputSpec(("E", "z", "x", "y")), cfg)
    torch.testing.assert_close(
        model.feature_scale, torch.tensor([1.0, 0.001, 0.001, 0.001])
    )
    assert model.coordinates == [2, 3, 1]
    captured = []
    hook = model.embed.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    features = torch.tensor([[2.0, 3000.0, 1000.0, -2000.0]])
    model(collate_samples([Sample(features)]))
    hook.remove()
    torch.testing.assert_close(captured[0], torch.tensor([[[2.0, 3.0, 1.0, -2.0]]]))
    torch.testing.assert_close(features, torch.tensor([[2.0, 3000.0, 1000.0, -2000.0]]))
    with pytest.raises(ValueError, match="declared input features"):
        Transformer(InputSpec(("E", "eta", "phi")), cfg)


@pytest.mark.parametrize(
    "options,normalize",
    [
        ({"projective_channels": 0}, False),
        ({"projective_coordinates": ["x", "x", "z"]}, False),
        ({"projective_coordinates": ["x", "y"]}, False),
        ({"projective_input_scale": 0.0}, False),
        ({"projective_input_scale": float("nan")}, False),
        ({"projective_input_scale": 0.001}, False),
        ({"projective_coordinates": ["x", "y", "z"]}, True),
        ({"family": "gatr"}, False),
        ({"attention": "standard", "projective_channels": 3}, False),
    ],
)
def test_invalid_projective_options(options, normalize):
    """
    Reject invalid point metrics and componentwise coordinate normalization

    Args:
        options:  Invalid model steering
        normalize:  Enable componentwise feature normalization
    """
    with pytest.raises(ValueError):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "model": {"attention": "projective", **options},
                "preprocessing": {"normalize_features": normalize},
            }
        )
