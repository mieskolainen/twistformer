# Physical four-momenta through preprocessing and geometric encoders
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import io

import pytest
import torch

from twistformer import build_experiment, parse_config
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples
from twistformer.data.preprocessing import Preprocessor, PreparedDataset
from twistformer.geometry.lorentz import (
    dot,
    from_matrix,
    matrix,
    null_projection,
    validate_p4,
)
from twistformer.geometry.twistors import (
    pairwise_features,
    spinor_from_null,
    twistors_from_p4,
)
from twistformer.layers.physics import MandelstamBias


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
def test_preprocessing_and_checkpoint_preserve_physical_momenta(dtype, normalize):
    """
    Preserve physical geometry through fitted and restored feature transforms

    Args:
        dtype:  Input momentum precision
        normalize:  Enable componentwise learned-feature standardization
    Returns:
        None: Geometry and source tensors remain unchanged
    """
    p = torch.tensor(
        [[10.00000001, 1.0, 2.0, 3.0], [5.0, -2.0, 1.0, -1.0]], dtype=dtype
    )
    original = p.clone()
    samples = [Sample(p, p4=p), Sample(p[:1], p4=p[:1])]
    pre = Preprocessor(normalize_features=normalize).fit(samples)
    buffer = io.BytesIO()
    torch.save(pre.state, buffer)
    buffer.seek(0)
    restored = Preprocessor(torch.load(buffer, weights_only=True))
    batch = collate_samples([restored.transform(s) for s in samples]).to("cpu")
    trained = collate_samples(list(PreparedDataset(samples, pre)))
    assert batch.p4.dtype == dtype
    assert torch.equal(batch.p4, trained.p4)
    assert torch.equal(batch.p4[0], original)
    assert torch.equal(p, original)
    assert not batch.p4[1, 1:].count_nonzero()
    assert torch.equal(batch.features, trained.features)
    if normalize:
        assert not torch.equal(batch.features[0], original.float())
    validate_p4(batch.p4, batch.mask)


@pytest.mark.parametrize("kind", ["penrose", "momentum"])
def test_twistor_predictions_ignore_component_normalization(kind):
    """
    Keep invariant-model predictions unchanged by learned-feature normalization

    Args:
        kind:  Incidence anchor convention
    Returns:
        None: Normalized and raw feature streams give identical logits
    """
    exp = build_experiment(
        parse_config(
            {
                "driver": {"name": "synthetic", "train_size": 8, "max_tokens": 5},
                "model": {
                    "family": "twistformer",
                    "twistor_kind": kind,
                    "width": 8,
                    "heads": 2,
                    "depth": 2,
                    "dropout": 0.0,
                    "pair_channels": 4,
                    "representation_channels": 2,
                    "twistor_transport": "frame",
                    "reference_token": "event",
                },
            }
        )
    )
    samples = exp.driver.dataset("train")
    pre = Preprocessor().fit(samples)
    raw = collate_samples(samples[:3])
    normalized = collate_samples([pre.transform(s) for s in samples[:3]])
    exp.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            exp.model(raw).logits, exp.model(normalized).logits, rtol=0, atol=0
        )


def test_mandelstam_features_after_normalization_and_lorentz_transform():
    """
    Compare nonzero Mandelstam bias with analytic invariants after preprocessing

    Returns:
        None: Raw geometry fixes the bias under boosts, rotations and CPU AMP
    """
    p = torch.tensor(
        [[6.0, 1.0, 2.0, 3.0], [4.0, -2.0, 1.0, -1.0]], dtype=torch.float64
    )
    samples = [Sample(p, p4=p), Sample(p[:1], p4=p[:1])]
    pre = Preprocessor().fit(samples)
    batch = collate_samples([pre.transform(s) for s in samples])
    layer = MandelstamBias(2)
    with torch.no_grad():
        layer.network[-1].weight.fill_(0.1)
    captured = []
    hook = layer.network.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach())
    )
    bias = layer(batch.p4, batch.mask)
    hook.remove()
    plus = batch.p4[:, :, None] + batch.p4[:, None]
    minus = batch.p4[:, :, None] - batch.p4[:, None]
    expected = torch.stack([dot(plus, plus), dot(minus, minus)], -1).asinh().float()
    torch.testing.assert_close(captured[0], expected)
    rotation = torch.tensor([[0.8, 0.6j], [0.6j, 0.8]], dtype=torch.complex128)
    s = rotation @ torch.diag(torch.tensor([1.7, 1 / 1.7], dtype=torch.complex128))
    boosted = from_matrix(s @ matrix(batch.p4) @ s.mH)
    torch.testing.assert_close(layer(boosted, batch.mask), bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hook = layer.network.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach())
        )
        mixed = layer(batch.p4, batch.mask)
        hook.remove()
    torch.testing.assert_close(captured[-1], expected, atol=0, rtol=0)
    assert mixed.isfinite().all()
    assert not bias[1, :, 1:].count_nonzero()


@pytest.mark.parametrize("kind", ["penrose", "momentum"])
@pytest.mark.parametrize(
    "momenta",
    [
        [[2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0]],
        [[2.0, 1.0, 0.0, 0.0], [3.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0, -2.0], [3.0, 0.0, 0.0, 3.0]],
    ],
)
def test_twistor_input_gradients_at_degenerate_geometry(kind, momenta):
    """
    Keep input derivatives finite at rest, collinearity, poles and padding

    Args:
        kind:  Incidence anchor convention
        momenta:  Future non-spacelike event including optional padding
    Returns:
        None: Geometric features and input gradients are finite
    """
    p = torch.tensor([momenta], dtype=torch.float64, requires_grad=True)
    mask = p[..., 0] > 0
    tw, metric = twistors_from_p4(p, mask, kind)
    features, _ = pairwise_features(tw.Z, metric, mask)
    features.square().sum().backward()
    assert features.isfinite().all()
    assert p.grad.isfinite().all()
    assert not p.grad[~mask].count_nonzero()


def test_geometry_input_derivatives_match_finite_differences():
    """
    Check analytic input derivatives away from chart and light-cone boundaries

    Returns:
        None: Projection and complex spinor derivatives match finite differences
    """
    p = torch.tensor(
        [[[3.0, 1.0, 0.5, 0.2], [4.0, -0.3, 0.7, -0.5]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(lambda x: null_projection(x, x.sum(1)), (p,))
    assert torch.autograd.gradcheck(spinor_from_null, (p,))


def test_collation_preserves_mixed_momentum_precision():
    """
    Promote mixed momentum dtypes without truncating the precise event

    Returns:
        None: Double precision survives batching beside float32 inputs
    """
    p = torch.tensor([[10.00000001, 1.0, 2.0, 3.0]], dtype=torch.float64)
    sample = Sample(p.float(), p4=p)
    batch = collate_samples([replace(sample, p4=p.float()), sample])
    assert batch.p4.dtype == torch.float64
    assert torch.equal(batch.p4[1], p)


@pytest.mark.parametrize(
    "p",
    [
        torch.tensor([[[1.0, 2.0, 0.0, 0.0]]]),
        torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]]),
        torch.tensor([[[float("nan"), 0.0, 0.0, 0.0]]]),
        torch.ones(1, 1, 4, dtype=torch.complex64),
        torch.ones(1, 1, 3),
    ],
)
def test_invalid_four_momenta_rejected(p):
    """
    Reject nonphysical momenta before entering the geometric encoders

    Args:
        p:  Invalid momentum tensor
    Returns:
        None: Validation raises a descriptive error
    """
    with pytest.raises(ValueError):
        validate_p4(p, torch.ones(1, 1, dtype=torch.bool))
