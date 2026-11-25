# Twistor precision and invariant transport validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace

import pytest
import torch

from twistformer import build_experiment, parse_config
from twistformer.data.batching import collate_samples
from twistformer.geometry.lorentz import from_matrix, matrix
from twistformer.geometry.twistors import (
    pairwise_features,
    rest_frame_coordinates,
    spin_frame,
    three_point_features,
    twistors_from_p4,
)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_factorized_cycles_and_gradients(dtype):
    """
    Compare rank-four cycles and derivatives with dense matrix multiplication

    Args:
        dtype:  Complex geometry precision
    Returns:
        None: Masked cycles and coordinate derivatives agree
    """
    torch.manual_seed(91)
    z = torch.randn(2, 11, 4, dtype=dtype, requires_grad=True)
    metric = torch.eye(4, dtype=dtype).expand(2, -1, -1)
    mask = torch.arange(11)[None] < torch.tensor([11, 6])[:, None]
    _, overlap = pairwise_features(z, metric, mask)
    dense = three_point_features(overlap, mask)
    factored = three_point_features(overlap, mask, z, metric)
    torch.testing.assert_close(factored, dense)
    weight = torch.randn_like(dense)
    first = torch.autograd.grad((dense * weight).sum(), z, retain_graph=True)[0]
    second = torch.autograd.grad((factored * weight).sum(), z)[0]
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("kind", ["penrose", "momentum"])
@pytest.mark.parametrize("boost", [1.0, 30.0])
def test_rest_frame_pair_invariants(kind, boost):
    """
    Preserve geometric invariants under rest-frame conditioning and rounding

    Args:
        kind:  Incidence anchor convention
        boost:  Longitudinal SL(2,C) boost parameter
    Returns:
        None: Complex64 pair features agree with complex128 laboratory features
    """
    torch.manual_seed(42)
    xyz = torch.randn(2, 8, 3, dtype=torch.float64)
    p = torch.cat([(xyz.square().sum(-1) + 0.1).sqrt()[..., None], xyz], -1)
    s = torch.diag(torch.tensor([boost, 1 / boost], dtype=torch.complex128))
    p = from_matrix(s @ matrix(p) @ s.mH)
    mask = torch.arange(8)[None] < torch.tensor([8, 5])[:, None]
    tw, metric = twistors_from_p4(p, mask, kind)
    expected, overlap = pairwise_features(tw.Z, metric, mask)
    z, euclidean = rest_frame_coordinates(tw.Z, metric)
    actual, reduced = pairwise_features(z, euclidean, mask)
    assert z.dtype == torch.complex64
    torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(reduced.to(overlap.dtype), overlap, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("kind", ["penrose", "momentum"])
@pytest.mark.parametrize("transport", ["phase", "frame"])
@pytest.mark.parametrize(
    "reference", ["none", "event", "beam", "frame", "time", "beam_plane"]
)
def test_model_precision_and_bf16_backward(kind, transport, reference):
    """
    Compare model precisions and backpropagate through BF16 scalar layers

    Args:
        kind:  Incidence anchor convention
        transport:  Covariant transport rule
        reference:  Reference token convention
    Returns:
        None: Predictions agree and learned geometry retains finite gradients
    """
    torch.manual_seed(12)
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 6},
            "model": {
                "family": "twistformer",
                "width": 8,
                "heads": 2,
                "depth": 3,
                "dropout": 0.0,
                "norm": "rms",
                "twistor_kind": kind,
                "twistor_transport": transport,
                "reference_token": reference,
                "representation_channels": 3,
                "pair_channels": 4,
                "three_point": True,
                "twistor_qk": True,
            },
        }
    )
    exp = build_experiment(cfg)
    samples = exp.driver.dataset("train")
    batch = collate_samples([samples[i] for i in range(3)])
    model = exp.model.eval()
    with torch.no_grad():
        for block in model.backbone.blocks[:-1]:
            block.step.bias.fill_(0.3)
    expected = model(batch).logits
    cfg.model.twistor_precision = "float32"
    reduced = build_experiment(cfg).model.eval()
    reduced.load_state_dict(model.state_dict(), strict=True)
    actual = reduced(batch).logits
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for first, second in zip(model.parameters(), reduced.parameters(), strict=True):
        if first.grad is not None:
            torch.testing.assert_close(second.grad, first.grad, atol=3e-4, rtol=3e-3)
    reduced.zero_grad(set_to_none=True)
    padded = replace(
        batch,
        p4=torch.nn.functional.pad(batch.p4, (0, 0, 0, 2)),
        features=torch.nn.functional.pad(batch.features, (0, 0, 0, 2)),
        mask=torch.nn.functional.pad(batch.mask, (0, 2)),
    )
    torch.testing.assert_close(reduced(padded).logits, actual, atol=2e-5, rtol=2e-5)
    if kind == "penrose":
        order = torch.randperm(batch.mask.shape[1])
        permuted = replace(
            batch,
            p4=batch.p4[:, order],
            features=batch.features[:, order],
            mask=batch.mask[:, order],
        )
        torch.testing.assert_close(
            reduced(permuted).logits, actual, atol=2e-5, rtol=2e-5
        )
    if reference in {"none", "event"}:
        s = torch.diag(torch.tensor([2.0, 0.5], dtype=torch.complex128))
        boosted = from_matrix(s @ matrix(batch.p4.double()) @ s.mH)
        transformed = replace(batch, p4=boosted)
        torch.testing.assert_close(
            reduced(transformed).logits, actual, atol=2e-4, rtol=2e-4
        )
    batch = replace(batch, p4=batch.p4.double().requires_grad_())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = reduced(batch).logits
    loss = logits.float().square().sum()
    loss.backward()
    assert logits.isfinite().all()
    assert batch.p4.grad.isfinite().all()
    assert all(
        p.grad.isfinite().all() for p in reduced.parameters() if p.grad is not None
    )


def test_float32_near_singular_frames():
    """
    Exclude dyads with determinants below complex64 relative resolution

    Returns:
        None: Singular inverses vanish and their derivatives stay finite
    """
    a = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0]], dtype=torch.complex64, requires_grad=True
    )
    b = torch.tensor([[1.0, 1.0 + 1e-7], [0.0, 1.0]], dtype=torch.complex64)
    _, inverse, valid = spin_frame(a, b)
    assert valid.tolist() == [False, True]
    inverse.abs().square().sum().backward()
    assert not inverse[0].count_nonzero()
    assert a.grad.isfinite().all()


@pytest.mark.parametrize(
    "momenta",
    [
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0]],
        [[2.0, 0.0, 0.0, -2.0], [3.0, 0.0, 0.0, 3.0]],
        [[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    ],
)
def test_rest_frame_degenerate_input_gradients(momenta):
    """
    Keep null events, chart poles and padding finite after conditioning

    Args:
        momenta:  Physical four-momenta including degenerate configurations
    Returns:
        None: Features and input derivatives are finite
    """
    p = torch.tensor([momenta], dtype=torch.float64, requires_grad=True)
    mask = p[..., 0] > 0
    tw, metric = twistors_from_p4(p, mask)
    z, metric = rest_frame_coordinates(tw.Z, metric)
    pairs, _ = pairwise_features(z, metric, mask)
    pairs.square().sum().backward()
    assert pairs.isfinite().all()
    assert p.grad.isfinite().all()


def test_precision_config_validation():
    """
    Retain historical checkpoint precision and reject unsupported geometry dtypes

    Returns:
        None: Old configurations default to float64 and invalid choices fail
    """
    base = {"driver": {"name": "synthetic"}, "model": {"family": "twistformer"}}
    assert parse_config(base).model.twistor_precision == "float64"
    base["model"]["twistor_precision"] = "bf16"
    with pytest.raises(ValueError, match="twistor_precision"):
        parse_config(base)


@pytest.mark.parametrize("activation", ["gelu", "swiglu"])
def test_training_preset_bf16(activation):
    """
    Exercise the GPU training preset through scalar and geometric backward paths

    Args:
        activation:  Feed-forward activation candidate
    Returns:
        None: Gated FFNs and separate pre-norms support finite BF16 training
    """
    from twistformer import load_config

    cfg = load_config("configs/toptag/twistformer.yml")
    cfg.driver = {"name": "synthetic", "train_size": 4, "max_tokens": 12}
    cfg.evaluation.categories = []
    cfg.model.ffn_activation = activation
    cfg.model.attention_dropout = 0.1
    exp = build_experiment(parse_config(cfg.to_dict()))
    samples = exp.driver.dataset("train")
    batch = collate_samples([samples[i] for i in range(3)])
    block = exp.model.backbone.blocks[0]
    assert block.norm.weight is not block.ff_norm.weight
    assert block.ff[0].out_features == cfg.model.width * cfg.model.expansion * (
        2 if activation == "swiglu" else 1
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = exp.task.compute_loss(exp.model(batch), batch)
    result.loss.backward()
    assert result.loss.isfinite()
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )
