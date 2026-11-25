# Laboratory frame references and detector orientation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace

import pytest
import torch

from twistformer import build_experiment, parse_config
from twistformer.data.batching import collate_samples
from twistformer.geometry.lorentz import from_matrix
from twistformer.geometry.twistors import (
    append_reference,
    laboratory_reference_features,
    twistors_from_p4,
)


def test_frame_rays_and_degenerate_mask():
    """
    Recover the six laboratory directions and mask a null event total
    """
    p = torch.tensor(
        [[[2, 1, 0, 0], [2, 0, 1, 0]], [[1, 0, 0, 1], [1, 0, 0, 1]]],
        dtype=torch.float64,
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    tw, metric = twistors_from_p4(p, mask)
    z, extended = append_reference(p, tw.Z, metric, mask, "frame", [0, 0, 1])
    assert z.shape == (2, 8, 4)
    torch.testing.assert_close(z[:, :2], tw.Z)
    torch.testing.assert_close(extended[:, :2], mask)
    assert extended[0, 2:].all()
    assert not extended[1, 2:].any()
    assert not z[1, 2:].count_nonzero()
    lam = z[0, 2:, :2]
    rays = from_matrix(lam[..., :, None] @ lam.conj()[..., None, :])
    expected = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(rays[:, 1:] / rays[:, :1], expected)
    norms = torch.einsum("ni,ij,nj->n", z[0, 2:].conj(), metric[0], z[0, 2:])
    torch.testing.assert_close(norms, torch.ones_like(norms))


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_frame_labels_distinguish_detector_axes(precision):
    """
    Distinguish detector axes through reference labels with auxiliary scalars

    Args:
        precision:  Twistor arithmetic precision
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 5},
            "model": {
                "family": "twistformer",
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "reference_token": "frame",
                "twistor_precision": precision,
                "auxiliary_features": ["E"],
            },
        }
    )
    exp = build_experiment(cfg)
    batch = collate_samples(exp.driver.dataset("train")[:3])
    model = exp.model.eval()
    rotated = batch.p4.clone()
    rotated[..., 1] = -batch.p4[..., 2]
    rotated[..., 2] = batch.p4[..., 1]
    other = replace(batch, p4=rotated)
    expected = model(batch).logits
    actual = model(other).logits
    assert (actual - expected).abs().max() > 1e-7
    expected.square().sum().backward()
    labels = model.backbone.frame_embedding.weight
    assert labels.grad.isfinite().all()
    assert (labels.grad.abs().sum(-1) > 0).all()
    assert model.backbone(batch).features.shape[:2] == batch.mask.shape
    with torch.no_grad():
        labels.zero_()
    # Unlabeled axis rays form the same set after a quarter turn
    torch.testing.assert_close(
        model(batch).logits, model(other).logits, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize("kind", ["time", "beam_plane"])
def test_laboratory_contractions_and_scalar_tokens(kind):
    """
    Recover laboratory energies and oriented pair areas without null factorization

    Args:
        kind:  Laboratory reference convention
    """
    p = torch.tensor(
        [[[2, 1, 0, 0], [3, 0, 2, 0], [float("nan")] * 4]], dtype=torch.float64
    )
    p.requires_grad_()
    mask = torch.tensor([[True, True, False]])
    scalars, pairs = laboratory_reference_features(p, mask, kind, [0, 0, 2])
    expected = torch.tensor(
        [[2, 3, 0, 5] + ([0] if kind == "beam_plane" else [])]
    ).double()
    torch.testing.assert_close(scalars[..., 0], expected.log1p())
    loss = scalars.square().sum()
    if pairs is not None:
        assert pairs.shape == (1, 5, 5, 1)
        torch.testing.assert_close(
            pairs[0, 0, 1, 0], torch.tensor(2.0).double().asinh()
        )
        torch.testing.assert_close(pairs, -pairs.transpose(1, 2))
        assert not pairs[:, 2:].count_nonzero()
        _, reversed_pairs = laboratory_reference_features(p, mask, kind, [0, 0, -2])
        torch.testing.assert_close(reversed_pairs, -pairs)
        loss = loss + pairs.square().sum()
    loss.backward()
    assert p.grad.isfinite().all()
    assert not p.grad[:, 2:].count_nonzero()
    # A null event still has well-defined time and plane contractions
    null = torch.tensor([[[1, 0, 0, 1]]], dtype=torch.float64)
    valid = torch.ones(1, 1, dtype=torch.bool)
    tw, metric = twistors_from_p4(null, valid)
    z, extended = append_reference(null, tw.Z, metric, valid, kind, [0, 0, 1])
    assert extended.all()
    assert z.shape[1] == (3 if kind == "beam_plane" else 2)
    assert not z.count_nonzero()


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["time", "beam_plane"])
def test_laboratory_reference_symmetry_and_gradients(kind, precision):
    """
    Preserve observer rotations and detect boosts or plane tilts

    Args:
        kind:  Laboratory reference convention
        precision:  Twistor arithmetic precision
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 5},
            "model": {
                "family": "twistformer",
                "width": 8,
                "heads": 2,
                "depth": 3,
                "dropout": 0.0,
                "reference_token": kind,
                "twistor_precision": precision,
                "representation_channels": 2,
                "pair_channels": 4,
                "twistor_qk": True,
            },
        }
    )
    exp = build_experiment(cfg)
    batch = collate_samples(exp.driver.dataset("train")[:3])
    model = exp.model.eval()
    expected = model(batch).logits
    expected.square().sum().backward()
    labels = model.backbone.spurion_embedding.weight
    assert labels.grad.isfinite().all()
    assert (labels.grad.abs().sum(-1) > 0).all()
    assert model.backbone.embed[0].weight.grad[:, -1].abs().sum() > 0
    assert model.backbone(batch).features.shape[:2] == batch.mask.shape
    angle = torch.tensor(0.73, dtype=batch.p4.dtype)
    c, s = angle.cos(), angle.sin()
    rotated = batch.p4.clone()
    rotated[..., 1] = c * batch.p4[..., 1] - s * batch.p4[..., 2]
    rotated[..., 2] = s * batch.p4[..., 1] + c * batch.p4[..., 2]
    torch.testing.assert_close(
        model(replace(batch, p4=rotated)).logits, expected, atol=3e-6, rtol=3e-5
    )
    tilted = batch.p4.clone()
    tilted[..., 1] = c * batch.p4[..., 1] - s * batch.p4[..., 3]
    tilted[..., 3] = s * batch.p4[..., 1] + c * batch.p4[..., 3]
    actual = model(replace(batch, p4=tilted)).logits
    if kind == "time":
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    else:
        assert (actual - expected).abs().max() > 1e-6
    boosted = batch.p4.clone()
    c, s = angle.cosh(), angle.sinh()
    boosted[..., 0] = c * batch.p4[..., 0] + s * batch.p4[..., 3]
    boosted[..., 3] = s * batch.p4[..., 0] + c * batch.p4[..., 3]
    assert (model(replace(batch, p4=boosted)).logits - expected).abs().max() > 1e-6


def test_plane_normal_and_model_rotate_together():
    """
    Rotate the oriented plane normal with the particle momenta
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 5},
            "model": {
                "family": "twistformer",
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "reference_token": "beam_plane",
                "beam_direction": [1.0, 2.0, 3.0],
            },
        }
    )
    exp = build_experiment(cfg)
    batch = collate_samples(exp.driver.dataset("train")[:3])
    model = exp.model.eval()
    expected = model(batch).logits
    rotated = batch.p4[..., [0, 3, 1, 2]]
    model.backbone.beam_direction = [3.0, 1.0, 2.0]
    torch.testing.assert_close(
        model(replace(batch, p4=rotated)).logits, expected, atol=3e-6, rtol=3e-5
    )


@pytest.mark.parametrize("kind", ["time", "beam_plane"])
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_laboratory_references_at_rest_and_null_events(kind, precision):
    """
    Keep observer features and derivatives finite at degenerate event geometry

    Args:
        kind:  Laboratory reference convention
        precision:  Twistor arithmetic precision
    """
    from twistformer.api.contracts import Sample

    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "model": {
                "family": "twistformer",
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "reference_token": kind,
                "twistor_precision": precision,
                "representation_channels": 2,
                "twistor_transport": "frame",
            },
        }
    )
    exp = build_experiment(cfg)
    events = (
        [[1, 0, 0, 1], [2, 0, 0, 2]],
        [[2, 0, 0, 0]],
        [[2, 1, 0, 0], [2, -1, 0, 0]],
    )
    samples = []
    for event in events:
        p = torch.tensor(event, dtype=torch.float64)
        samples.append(Sample(p.float(), p4=p))
    batch = collate_samples(samples)
    batch.p4.requires_grad_()
    output = exp.model(batch).logits
    output.square().sum().backward()
    assert output.isfinite().all()
    assert batch.p4.grad.isfinite().all()
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("reference", ["time", "beam_plane"])
def test_laboratory_reference_validation(reference):
    """
    Restrict new contractions to Twistformer and reject invalid plane normals

    Args:
        reference:  Laboratory reference convention
    """
    for family in ("transformer", "triformer", "gatr", "pelican"):
        with pytest.raises(ValueError, match="reference_token"):
            parse_config(
                {
                    "driver": {"name": "synthetic"},
                    "model": {"family": family, "reference_token": reference},
                }
            )
    with pytest.raises(ValueError, match="beam_direction"):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "model": {
                    "family": "twistformer",
                    "reference_token": reference,
                    "beam_direction": [0.0, 0.0, 0.0],
                },
            }
        )
