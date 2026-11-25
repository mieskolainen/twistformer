# Laboratory axis tokens and particle attention
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from twistformer import build_experiment, load_config, parse_config
from twistformer.api.contracts import InputSpec
from twistformer.data.batching import collate_samples
from twistformer.models.transformer import Transformer


@pytest.mark.parametrize("attention", ["standard", "hyperbolic"])
@pytest.mark.parametrize("pair_bias", ["none", "mandelstam", "jet"])
def test_frame_attention_gradients_padding_and_permutation(attention, pair_bias):
    """
    Propagate axis labels through masked particle attention

    Args:
        attention:  Scalar attention metric
        pair_bias:  Physical particle pair bias
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 5},
            "model": {
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                "reference_token": "frame",
                "attention": attention,
                "pair_bias": pair_bias,
            },
        }
    )
    exp = build_experiment(cfg)
    batch = collate_samples(exp.driver.dataset("train")[:3])
    model = exp.model.eval()
    expected = model(batch).logits
    expected.square().sum().backward()
    labels = model.backbone.frame_embedding.weight
    assert labels.grad.isfinite().all()
    assert (labels.grad.abs().sum(-1) > 0).all()
    assert model.backbone(batch).features.shape[:2] == batch.mask.shape
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 3), value=float("nan")),
        p4=F.pad(batch.p4, (0, 0, 0, 3), value=float("nan")),
        mask=F.pad(batch.mask, (0, 3)),
    )
    torch.testing.assert_close(model(padded).logits, expected, atol=2e-6, rtol=2e-5)
    order = torch.arange(batch.mask.shape[1] - 1, -1, -1)
    permuted = replace(
        batch,
        features=batch.features[:, order],
        p4=batch.p4[:, order],
        mask=batch.mask[:, order],
    )
    torch.testing.assert_close(model(permuted).logits, expected, atol=2e-6, rtol=2e-5)
    with torch.no_grad():
        labels.add_(torch.linspace(-0.5, 0.5, labels.shape[-1]))
    assert (model(batch).logits - expected).abs().max() > 1e-7
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert model(batch).logits.isfinite().all()


def test_frame_features_follow_named_axes():
    """
    Embed signed detector axes in the declared feature order
    """
    cfg = parse_config(
        {
            "driver": {"name": "synthetic"},
            "model": {"reference_token": "frame"},
        }
    ).model
    model = Transformer(InputSpec(("E", "z", "x", "y")), cfg)
    expected = torch.tensor(
        [
            [0, 0, 1, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, -1],
            [0, 1, 0, 0],
            [0, -1, 0, 0],
        ]
    ).float()
    torch.testing.assert_close(model.frame_features, expected)
    with pytest.raises(ValueError, match="require x,y,z or px,py,pz"):
        Transformer(InputSpec(("E", "eta", "phi")), cfg)


@pytest.mark.parametrize(
    "family,reference",
    [
        ("transformer", "event"),
        ("transformer", "beam"),
        ("triformer", "frame"),
        ("gatr", "frame"),
        ("pelican", "frame"),
    ],
)
def test_invalid_reference_family(family, reference):
    """
    Reject references without an implementation in the selected encoder

    Args:
        family:  Encoder family
        reference:  Unsupported reference kind
    """
    with pytest.raises(ValueError, match="reference_token"):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "model": {"family": family, "reference_token": reference},
            }
        )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_tau_frame_presets(task):
    """
    Resolve frame references through tau preset inheritance

    Args:
        task:  Tau prediction task
    """
    cfg = load_config(f"configs/tau3_{task}/transformer.yml")
    assert cfg.model.reference_token == "frame"
    for algebra in ("euclidean", "projective", "conformal", "lorentz"):
        cfg = load_config(f"configs/tau3_{task}/gatr_{algebra}.yml")
        assert cfg.model.gatr_spurions == ("frame" if algebra == "lorentz" else "none")
        assert not cfg.model.gatr_reference_tokens
