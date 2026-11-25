# Research architecture covariance and optional readouts
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import copy
import pytest
import torch
from torch.nn import functional as F
from twistformer import parse_config, build_experiment
from twistformer.data.batching import collate_samples
from twistformer.geometry.lorentz import matrix, from_matrix
from twistformer.geometry.twistors import (
    transport_channels,
    projective_normalize,
    pairwise_features,
)
from twistformer.layers.common import ResidualMLP
from twistformer.layers.attention import TriangleAttention


def experiment(**options):
    cfg = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 12, "max_tokens": 5},
            "model": {"width": 8, "heads": 2, "depth": 2, "dropout": 0.0, **options},
        }
    )
    exp = build_experiment(cfg)
    return exp, collate_samples(exp.driver.dataset("train")[:3])


@pytest.mark.parametrize("transport", ["phase", "frame"])
@pytest.mark.parametrize(
    "reference", ["none", "event", "beam", "frame", "time", "beam_plane"]
)
def test_twistor_research_gradients_and_symmetries(transport, reference):
    exp, batch = experiment(
        family="twistformer",
        pair_channels=4,
        representation_channels=3,
        twistor_transport=transport,
        reference_token=reference,
        three_point=True,
        twistor_qk=True,
        twistor_geometry_scale=0.2,
        twistor_representation_scale=0.05,
        twistor_residual_scaling="depth",
    )
    exp.model.eval()
    prediction = exp.model(batch)
    exp.task.compute_loss(prediction, batch).loss.backward()
    assert all(
        p.grad is None or p.grad.isfinite().all() for p in exp.model.parameters()
    )
    block = exp.model.backbone.blocks[0]
    assert sum(m.weight.grad.abs().sum() for m in block.pair.maps) > 0
    assert block.qk.weight.grad.abs().sum() > 0
    assert block.rep_mix.real.grad.abs().sum() > 0
    assert block.rep_mix.imag.grad.abs().sum() > 0
    assert exp.model.backbone.rep_seed.weight.grad.abs().sum() > 0
    permutation = torch.arange(batch.mask.shape[1] - 1, -1, -1)
    shuffled = replace(
        batch,
        features=batch.features[:, permutation],
        p4=batch.p4[:, permutation],
        mask=batch.mask[:, permutation],
    )
    torch.testing.assert_close(
        exp.model(shuffled).logits, prediction.logits, atol=3e-6, rtol=3e-5
    )
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 3)),
        p4=F.pad(batch.p4, (0, 0, 0, 3)),
        mask=F.pad(batch.mask, (0, 3)),
    )
    torch.testing.assert_close(
        exp.model(padded).logits, prediction.logits, atol=3e-6, rtol=3e-5
    )
    if reference in {"none", "event"}:
        with torch.no_grad():
            block.step.bias.fill_(0.4)
        s = torch.tensor([[1.2, 0.3j], [0.0, 1 / 1.2]], dtype=torch.complex128)
        boosted = from_matrix(s @ matrix(batch.p4.double()) @ s.mH).float()
        torch.testing.assert_close(
            exp.model(replace(batch, p4=boosted)).logits,
            exp.model(batch).logits,
            atol=1e-5,
            rtol=1e-4,
        )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = exp.model(batch).logits
    assert result.isfinite().all()


def test_frame_transport_covariance_gauge_and_singular_fallback():
    torch.manual_seed(23)
    z = torch.randn(2, 4, 4, dtype=torch.complex128)
    g = torch.eye(4, dtype=torch.complex128).expand(2, -1, -1)
    mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
    z = projective_normalize(z.masked_fill(~mask[..., None], 0), g)
    _, overlap = pairwise_features(z, g, mask)
    v = (
        torch.randn(2, 4, 4, 3, dtype=torch.complex128)
        .masked_fill(~mask[..., None, None], 0)
        .requires_grad_()
    )
    a = (
        torch.randn(2, 4, 4, dtype=torch.float64)
        .masked_fill(~mask[:, None], -torch.inf)
        .softmax(-1)
    )
    result = transport_channels(z, overlap, a, v, mask, "frame")
    s = torch.tensor([[1.3, 0.2j], [0.1, (1 + 0.02j) / 1.3]], dtype=torch.complex128)
    transform = torch.block_diag(s, torch.linalg.inv(s).mH)
    inverse = torch.linalg.inv(transform)
    new_g = inverse.mH @ g @ inverse
    new_z = z @ transform.T
    new_v = torch.einsum("ij,bnjc->bnic", transform, v)
    _, new_overlap = pairwise_features(new_z, new_g, mask)
    transformed = transport_channels(new_z, new_overlap, a, new_v, mask, "frame")
    torch.testing.assert_close(
        transformed,
        torch.einsum("ij,bnjc->bnic", transform, result),
        atol=2e-10,
        rtol=2e-9,
    )
    phase = torch.exp(1j * torch.randn(2, 4, dtype=torch.float64))
    _, phased_overlap = pairwise_features(z * phase[..., None], g, mask)
    phased = transport_channels(
        z * phase[..., None],
        phased_overlap,
        a,
        v * phase[..., None, None],
        mask,
        "frame",
    )
    torch.testing.assert_close(
        phased, result * phase[..., None, None], atol=2e-10, rtol=2e-9
    )
    result.abs().square().sum().backward()
    assert v.grad.isfinite().all()
    singular = z[:, :1].expand(-1, 4, -1).masked_fill(~mask[..., None], 0)
    _, c = pairwise_features(singular, g, mask)
    torch.testing.assert_close(
        transport_channels(singular, c, a, v, mask, "frame"),
        transport_channels(singular, c, a, v, mask, "phase"),
    )


@pytest.mark.parametrize("pool", ["pma", "cls"])
@pytest.mark.parametrize("family", ["transformer", "triformer", "twistformer"])
def test_attention_pooling_padding_permutation_and_gradients(pool, family):
    exp, batch = experiment(
        family=family,
        pooling=pool,
        pooling_depth=2 if pool == "cls" else 1,
        pooling_queries=3 if pool == "pma" else 1,
    )
    exp.model.eval()
    predicted = exp.model(batch).logits
    predicted.square().sum().backward()
    assert exp.model.pool.seed.grad.abs().sum() > 0
    assert all(
        p.grad is None or p.grad.isfinite().all() for p in exp.model.parameters()
    )
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 3)),
        p4=F.pad(batch.p4, (0, 0, 0, 3)),
        mask=F.pad(batch.mask, (0, 3)),
    )
    torch.testing.assert_close(
        exp.model(padded).logits, predicted, atol=2e-6, rtol=2e-5
    )
    order = torch.randperm(batch.mask.shape[1])
    shuffled = replace(
        batch,
        features=batch.features[:, order],
        p4=batch.p4[:, order],
        mask=batch.mask[:, order],
    )
    torch.testing.assert_close(
        exp.model(shuffled).logits, predicted, atol=2e-6, rtol=2e-5
    )


@pytest.mark.parametrize(
    "activation",
    [
        "gelu",
        "silu",
        "relu",
        "elu",
        "tanh",
        "sine",
        "sine_adaptive",
        "glu",
        "geglu",
        "swiglu",
    ],
)
def test_reusable_mlp_projected_residuals(activation):
    network = ResidualMLP(
        4,
        2,
        [7, 3, 5],
        activation=activation,
        norm="rms",
        residual=True,
        init="xavier_normal",
    )
    x = torch.randn(6, 4, requires_grad=True)
    output = network(x)
    output.square().sum().backward()
    assert output.shape == (6, 2) and x.grad.isfinite().all()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in network.parameters()
    )
    zero = ResidualMLP(
        4,
        2,
        [7, 3],
        activation=activation,
        residual=True,
        init="xavier_normal",
        zero_init_last=True,
    )
    assert not zero(x).count_nonzero()


def test_masked_batchnorm_ignores_padding():
    a = ResidualMLP(4, 2, [5, 3], norm="batch", residual=True)
    b = copy.deepcopy(a)
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    result = a(x, mask)
    padded = b(F.pad(x, (0, 0, 0, 4), value=float("nan")), F.pad(mask, (0, 4)))
    torch.testing.assert_close(result, padded[:, :3])
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name])
    with pytest.raises(ValueError, match="validity mask"):
        a(x)
    assert a(torch.ones(1, 4)).isfinite().all()


@pytest.mark.parametrize(
    "shared,distinct,kj,kk",
    [
        (False, False, 2, 3),
        (True, False, 3, 3),
        (True, True, 3, 3),
        (False, True, 1, 1),
    ],
)
def test_triangle_candidate_options(shared, distinct, kj, kk):
    layer = TriangleAttention(
        8, 2, kj=kj, kk=kk, shared=shared, distinct=distinct
    ).eval()
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[True] * 5, [True, False, False, False, False]])
    y = layer(x, mask)
    assert y.isfinite().all()
    order = torch.tensor([2, 4, 0, 1, 3])
    torch.testing.assert_close(
        layer(x[:, order], mask[:, order]), y[:, order], atol=2e-6, rtol=2e-5
    )
    y.square().sum().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in layer.parameters())


def test_jet_and_graph_options():
    exp, batch = experiment(
        pair_bias="jet",
        jet_features=["p4dot", "dR", "eec"],
        jet_eec_betas=[0.5, 2.0],
        jet_radius=0.4,
        jet_head_mode="shared",
        jet_hidden=8,
    )
    with torch.no_grad():
        exp.model.backbone.bias.to_bias.weight.normal_(std=0.1)
    loss = exp.task.compute_loss(exp.model(batch), batch).loss
    loss.backward()
    assert exp.model.backbone.bias.head_gain.grad.isfinite().all()
    exp, batch = experiment(
        family="pelican",
        norm="rms",
        pelican_activation="silu",
        pelican_message_width=5,
        pelican_average_nobj=4.0,
        pelican_momentum_scale=0.5,
        pelican_factorize=False,
        pelican_multiplicity=False,
    )
    exp.model.eval()
    s = torch.tensor([[1.3, 0.0], [0.0, 1 / 1.3]], dtype=torch.complex128)
    boosted = from_matrix(s @ matrix(batch.p4.double()) @ s.mH).float()
    torch.testing.assert_close(
        exp.model(replace(batch, p4=boosted)).logits,
        exp.model(batch).logits,
        atol=2e-5,
        rtol=2e-5,
    )
    exp.task.compute_loss(exp.model(batch), batch).loss.backward()
    assert all(
        p.grad is None or p.grad.isfinite().all() for p in exp.model.parameters()
    )


@pytest.mark.parametrize(
    "values",
    [
        {"family": "twistformer", "pair_channels": -1},
        {"family": "twistformer", "twistor_transport": "invalid"},
        {
            "family": "twistformer",
            "reference_token": "beam",
            "beam_direction": [0.0, 0.0, 0.0],
        },
        {"representation_channels": 2},
        {"pooling": "cls", "pooling_queries": 2},
        {"pooling": "pma", "pooling_heads": 3},
        {
            "family": "triformer",
            "triangle_kj": 2,
            "triangle_kk": 3,
            "triangle_shared_candidates": True,
        },
        {"pair_bias": "jet", "jet_features": ["invalid"]},
        {"pair_bias": "jet", "jet_eec_betas": []},
        {"head_widths": [8], "head_layers": 2},
    ],
)
def test_research_config_validation(values):
    with pytest.raises(ValueError):
        parse_config({"driver": {"name": "synthetic"}, "model": values})


def test_jet_geometry_gradients_at_zero_transverse_momentum():
    exp, batch = experiment(pair_bias="jet", jet_grad_geometry=True)
    p = torch.tensor(
        [[[2.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True, False]])
    layer = exp.model.backbone.bias
    with torch.no_grad():
        layer.to_bias.weight.normal_(std=0.1)
    layer(p, mask).square().sum().backward()
    assert p.grad.isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("pool", ["pma", "cls"])
def test_cuda_research_attention_and_transport(pool):
    exp, batch = experiment(
        family="twistformer",
        pair_channels=4,
        representation_channels=3,
        twistor_transport="frame",
        reference_token="event",
        pooling=pool,
    )
    exp.model.cuda()
    batch = batch.to("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        prediction = exp.model(batch)
    loss = exp.task.compute_loss(prediction, batch).loss
    loss.backward()
    assert loss.isfinite()
    assert all(
        p.grad is None or p.grad.isfinite().all() for p in exp.model.parameters()
    )
