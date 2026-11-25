# Geometric-algebra identities, symmetries and particle workflows
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import math

import pytest
import torch
from torch.nn import functional as F

from twistformer import (
    build_experiment,
    load_config,
    load_predictor,
    parse_config,
    Trainer,
)
from twistformer.api.contracts import Sample
from twistformer.data.batching import collate_samples
from twistformer.evaluation.runner import evaluate
from twistformer.geometry.clifford import CliffordAlgebra, SIGNATURES
from twistformer.layers.gatr import EquivariantLinear, distance_features


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("kind", SIGNATURES)
def test_attention_amp_preserves_geometric_precision(monkeypatch, kind, checkpoint):
    """
    Protect conformal attention and geometry while retaining AMP scalar heads

    Args:
        monkeypatch:  Attention call recorder
        kind:  Clifford algebra
        checkpoint:  Recompute geometric blocks during backward
    """
    from twistformer.layers import gatr

    exp = build_experiment(
        configuration(kind, gatr_multi_query=True, gatr_checkpoint=checkpoint)
    )
    batch = collate_samples(exp.driver.dataset("train")[:2])
    original = gatr.scaled_attention
    calls = []

    def record(q, k, v, mask, **kwargs):
        """
        Record attention precision and call the shared dispatcher

        Args:
            q:  Query features
            k:  Key features
            v:  Value features
            mask:  Valid particles
            kwargs:  Attention dropout and scale
        Returns:
            context: Shared attention output
        """
        calls.append((q.dtype, k.dtype, v.dtype, q.shape[1], k.shape[1]))
        return original(q, k, v, mask, **kwargs)

    monkeypatch.setattr(gatr, "scaled_attention", record)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoded = exp.model.backbone(batch).features
        prediction = exp.model.head(exp.model.pool(encoded, batch))
    assert encoded.dtype == torch.float32
    dtype = torch.float32 if kind == "conformal" else torch.bfloat16
    assert calls == [(dtype,) * 3 + (2, 1)] * 2
    assert prediction.logits.dtype == torch.bfloat16
    if kind == "conformal":
        torch.testing.assert_close(
            encoded, exp.model.backbone(batch).features, rtol=0, atol=0
        )
    prediction.logits.float().square().sum().backward()
    assert all(call[:3] == (dtype,) * 3 for call in calls)
    assert all(p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None)


def configuration(kind, **model):
    """
    Build small geometric transformer steering for CPU checks

    Args:
        kind:  Geometric algebra
        model:  Model overrides
    Returns:
        config: Validated experiment steering
    """
    return parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 8,
                "val_size": 4,
                "test_size": 4,
                "max_tokens": 5,
            },
            "model": {
                "family": "gatr",
                "gatr_algebra": kind,
                "width": 8,
                "heads": 2,
                "depth": 2,
                "gatr_mv_channels": 4,
                "dropout": 0.0,
                **model,
            },
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "eval_batch_size": 4,
                "device": "cpu",
            },
        }
    )


@pytest.mark.parametrize(
    "kind,count",
    [("euclidean", 4), ("projective", 9), ("conformal", 20), ("lorentz", 10)],
)
def test_clifford_identities_and_linear_basis(kind, count):
    """
    Verify Clifford anticommutation, associativity and independent complete maps
    """
    algebra = CliffordAlgebra(kind).double()
    vectors = torch.eye(algebra.dim, dtype=torch.float64)[
        1 : 1 + len(algebra.signature)
    ]
    unit = torch.eye(algebra.dim, dtype=torch.float64)[0]
    for i, a in enumerate(vectors):
        for j, b in enumerate(vectors):
            expected = 2 * algebra.signature[i] * unit if i == j else unit * 0
            torch.testing.assert_close(
                algebra.product(a, b) + algebra.product(b, a), expected
            )
    x, y, z = torch.randn(3, 5, algebra.dim, dtype=torch.float64)
    torch.testing.assert_close(
        algebra.product(algebra.product(x, y), z),
        algebra.product(x, algebra.product(y, z)),
    )
    assert len(algebra.linear_basis) == count
    assert torch.linalg.matrix_rank(algebra.linear_basis.flatten(1)).item() == count


@pytest.mark.parametrize("kind", SIGNATURES)
def test_equivariant_linear_and_geometric_product(kind):
    """
    Check primitive covariance under rotations, translations or boosts
    """
    algebra = CliffordAlgebra(kind).double()
    eye = torch.eye(algebra.dim, dtype=torch.float64)
    bivector = torch.zeros(algebra.dim, dtype=torch.float64)
    if kind == "projective":
        bivector[algebra.index[3]] = 0.2
        bivector[algebra.index[6]] = 0.3
    elif kind == "conformal":
        bivector[algebra.index[17]] = 0.2
        bivector[algebra.index[9]] = -0.2
        bivector[algebra.index[3]] = 0.3
    else:
        bivector[algebra.index[3]] = 0.3
    generator = algebra.product(bivector, eye) - algebra.product(eye, bivector)
    action = torch.matrix_exp(generator)
    layer = EquivariantLinear(algebra, 3, 4, 5, 6).double()
    x, y = torch.randn(2, 7, 3, algebra.dim, dtype=torch.float64)
    scalars = torch.randn(7, 5, dtype=torch.float64)
    mv, s = layer(x, scalars)
    transformed, invariant = layer(x @ action, scalars)
    torch.testing.assert_close(transformed, mv @ action, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(invariant, s, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        algebra.product(x @ action, y @ action),
        algebra.product(x, y) @ action,
        atol=2e-7,
        rtol=2e-7,
    )
    torch.testing.assert_close(
        algebra.norm_squared(x @ action), algebra.norm_squared(x), atol=2e-7, rtol=2e-7
    )


@pytest.mark.parametrize("in_mv,out_mv", [(1, 3), (4, 2)])
def test_conformal_initialization_preserves_algebra_components(in_mv, out_mv):
    """
    Check common channel mixing across grades and trainable infinity maps

    Args:
        in_mv:  Input multivector channels
        out_mv:  Output multivector channels
    """
    torch.manual_seed(101)
    algebra = CliffordAlgebra("conformal")
    layer = EquivariantLinear(algebra, in_mv, out_mv, 5, 6)
    mv = torch.randn(2, 7, in_mv, algebra.dim)
    scalars = torch.randn(2, 7, 5)
    output, _ = layer(mv, scalars)
    offset, _ = layer(torch.zeros_like(mv), scalars)
    mixing = layer.weight[..., 0].detach()
    expected = torch.einsum("oi,...id->...od", mixing, mv)
    torch.testing.assert_close(output - offset, expected, atol=2e-7, rtol=2e-6)
    assert mixing.abs().max() <= in_mv**-0.5
    assert mixing.std() > 0
    assert layer.weight[..., 6:].count_nonzero() == 0
    (output * torch.randn_like(output)).sum().backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert (layer.weight.grad[..., 6:].abs().sum((0, 1)) > 0).all()


def test_projective_distance_and_join():
    """
    Check distance attention and nonzero line geometry from two projective points
    """
    algebra = CliffordAlgebra("projective").double()
    x, y = torch.randn(2, 7, 3, dtype=torch.float64)
    a, b = algebra.embed(x), algebra.embed(y)
    q = distance_features(a, 0.001, True)
    k = distance_features(b, 0.001, False)
    torch.testing.assert_close((q * k).sum(-1), -(x - y).square().sum(-1) / 1.001**4)
    line = algebra.join(a, b, torch.ones(7, 1, dtype=torch.float64))
    assert line.abs().sum() > 0
    torch.testing.assert_close(
        algebra.norm_squared(line).squeeze(-1), (x - y).square().sum(-1)
    )


@pytest.mark.parametrize("global_token,reference_tokens", [(False, False), (True, False), (True, True)])
def test_compiled_projective_frame_reinsertion(global_token, reference_tokens):
    """
    Compare compiled and eager projective attention with detector frame inputs

    Args:
        global_token:  Use a global readout token
        reference_tokens:  Store frame points as tokens
    """
    cfg = load_config("configs/pi0gamma/gatr_projective_angular.yml")
    cfg.driver["global_features"] = []
    cfg.model.depth = 1
    cfg.model.dropout = 0
    cfg.model.gatr_checkpoint = False
    cfg.model.gatr_global_token = global_token
    cfg.model.gatr_reference_tokens = reference_tokens
    cfg.model.head_zero_init = False
    model = build_experiment(cfg).model.double().eval()
    reference = deepcopy(model)
    model.backbone.blocks[0].attention = torch.compile(
        model.backbone.blocks[0].attention, backend="aot_eager", fullgraph=True
    )
    batch = collate_samples([Sample(torch.randn(count, 4, dtype=torch.float64)) for count in (3, 5)])
    batch.features.requires_grad_()
    actual, expected = model(batch).logits, reference(batch).logits
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    found = torch.autograd.grad(actual.square().sum(), (batch.features, *model.parameters()))
    expected = torch.autograd.grad(expected.square().sum(), (batch.features, *reference.parameters()))
    for a, b in zip(found, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("kind", ["euclidean", "projective"])
@pytest.mark.parametrize("multi_query", [True, False])
def test_learned_reinsertion(kind, multi_query):
    """
    Learn raw cell contributions to geometric queries and keys

    Args:
        kind:  Spatial algebra for cell coordinates
        multi_query:  Share keys and values across heads
    """
    config = load_config(f"configs/pi0gamma/gatr_{kind}_angular.yml")
    config.model.gatr_multi_query = multi_query
    encoder = build_experiment(config).model.backbone.double().eval()
    batch = collate_samples([Sample(torch.randn(count, 4, dtype=torch.float64)) for count in (3, 5)])
    raw, cell_scalars, mask, _ = encoder.embed(batch)
    block = encoder.blocks[0]
    with torch.no_grad():
        mv, scalars = block.norm1(*encoder.input(raw, cell_scalars))
    attention = block.attention
    original = (raw, cell_scalars)
    empty = (torch.zeros_like(raw), torch.zeros_like(cell_scalars))
    actual = attention(mv, scalars, mask, original)
    torch.testing.assert_close(actual, attention(mv, scalars, mask, empty), rtol=0, atol=0)

    optimizer = torch.optim.Adam(attention.parameters(), lr=3e-4)
    sum(stream.square().mean() for stream in actual).backward()
    energy_column = scalars.shape[-1] + 1 + config.model.auxiliary_features.index("log_energy")
    for projection in (attention.q, attention.k):
        assert projection.weight.grad[:, mv.shape[-2]].abs().sum() > 0
        assert projection.to_s.weight.grad[:, energy_column].abs().sum() > 0
    optimizer.step()
    with torch.no_grad():
        _, updated = attention(mv, scalars, mask, original)
        _, without_raw = attention(mv, scalars, mask, empty)
    assert not torch.allclose(updated, without_raw, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("mv_channels,expansion,product_channels,join_channels", [(8, 2, 8, 8), (3, 1, 2, 1), (1, 1, 1, 1)])
def test_projective_mlp_channel_split(mv_channels, expansion, product_channels, join_channels):
    """
    Share the expanded multivector width between the geometric product and join

    Args:
        mv_channels:  Input multivector channels
        expansion:  MLP width multiplier
        product_channels:  Expected geometric products per cell
        join_channels:  Expected joins per cell
    """
    config = configuration("projective", gatr_mv_channels=mv_channels, expansion=expansion)
    mlp = build_experiment(config).model.backbone.blocks[0].mlp
    assert mlp.left.out_mv == mlp.right.out_mv == product_channels
    assert mlp.join_left.out_mv == mlp.join_right.out_mv == join_channels
    assert mlp.bilinear.weight.shape[:2] == (mv_channels * expansion, product_channels + join_channels)
    mv = torch.randn(2, 3, mv_channels, mlp.algebra.dim)
    scalars = torch.randn(2, 3, config.model.width)
    output, hidden = mlp(mv, scalars, mv.new_ones(2, 1, 1, 1))
    (output.square().mean() + hidden.square().mean()).backward()
    for projection in (mlp.left, mlp.right, mlp.join_left, mlp.join_right):
        assert torch.isfinite(projection.weight.grad).all() and projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["euclidean", "projective"])
def test_spatial_initialization_retains_geometric_variance(kind):
    """
    Keep spatial components at channel fan-in variance across grade dimensions

    Args:
        kind:  Positive or degenerate spatial algebra
    """
    algebra = CliffordAlgebra(kind)
    layer = EquivariantLinear(algebra, 8, 256, 4, 0)
    mv = torch.randn(128, 8, algebra.dim)
    scalars = torch.zeros(128, 4)
    output, _ = layer(mv, scalars)
    offset, _ = layer(torch.zeros_like(mv), scalars)
    variance = (output - offset).var(dim=(0, 1))
    expected = torch.full_like(variance, 1 / 3)
    if kind == "projective":
        expected[[bool(blade & 1) for blade in algebra.blades]] *= 2
    torch.testing.assert_close(variance, expected, atol=0.04, rtol=0.1)


@pytest.mark.parametrize("kind", SIGNATURES)
def test_coordinate_embeddings_and_metrics(kind):
    """
    Check spatial point coordinates and the Lorentz energy-first convention

    Args:
        kind:  Clifford algebra
    """
    algebra = CliffordAlgebra(kind).double()
    dimension = 4 if kind == "lorentz" else 3
    x, y = torch.randn(2, 7, dimension, dtype=torch.float64)
    a, b = algebra.embed(x), algebra.embed(y)
    inner = (a * b * algebra.metric).sum(-1)
    if kind == "projective":
        torch.testing.assert_close(a[..., algebra.index[14]], torch.ones_like(x[..., 0]))
        decoded = torch.stack((-a[..., algebra.index[13]], a[..., algebra.index[11]], -a[..., algebra.index[7]]), -1)
        torch.testing.assert_close(decoded, x)
        expected = torch.ones_like(inner)
    elif kind == "conformal":
        torch.testing.assert_close((a.square() * algebra.metric).sum(-1), torch.zeros_like(inner), atol=1e-12, rtol=0)
        expected = -(x - y).square().sum(-1) / 2
    elif kind == "lorentz":
        torch.testing.assert_close(a[..., 1:5], x)
        expected = x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(-1)
    else:
        torch.testing.assert_close(a[..., 1:4], x)
        expected = (x * y).sum(-1)
    torch.testing.assert_close(inner, expected)


@pytest.mark.parametrize("kind", SIGNATURES)
@pytest.mark.parametrize("multi_query,global_token", [(False, False), (True, True)])
def test_model_backward_permutation_padding_and_amp(kind, multi_query, global_token):
    """
    Check scalar heads, BF16 training and masked particle-set behavior
    """
    exp = build_experiment(
        configuration(
            kind,
            gatr_multi_query=multi_query,
            gatr_global_token=global_token,
            gatr_checkpoint=True,
        )
    )
    batch = collate_samples(exp.driver.dataset("train")[:3])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = exp.model(batch)
        loss = exp.task.compute_loss(prediction, batch).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )
    exp.model.eval()
    expected = exp.model(batch).logits
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 3), value=float("nan")),
        p4=F.pad(batch.p4, (0, 0, 0, 3), value=float("nan")),
        mask=F.pad(batch.mask, (0, 3)),
    )
    torch.testing.assert_close(exp.model(padded).logits, expected, atol=2e-5, rtol=2e-5)
    order = torch.randperm(batch.mask.shape[1])
    permuted = replace(
        batch,
        features=batch.features[:, order],
        p4=batch.p4[:, order],
        mask=batch.mask[:, order],
    )
    torch.testing.assert_close(
        exp.model(permuted).logits, expected, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("kind", SIGNATURES)
def test_full_model_symmetry_and_sensitivity(kind):
    """
    Check model invariance while requiring a response to changed geometry
    """
    exp = build_experiment(
        configuration(
            kind,
            depth=4,
            width=16,
            heads=4,
            gatr_mv_channels=8,
            gatr_global_token=True,
            gatr_center=kind == "euclidean",
        )
    )
    exp.model.double().eval()
    batch = collate_samples(exp.driver.dataset("train")[:3])
    batch = replace(batch, features=batch.features.double(), p4=batch.p4.double())
    transform = torch.eye(4, dtype=torch.float64)
    angle = 0.7
    if kind == "lorentz":
        transform[0, 0] = transform[1, 1] = math.cosh(angle)
        transform[0, 1] = transform[1, 0] = math.sinh(angle)
    else:
        transform[1, 1] = transform[2, 2] = math.cos(angle)
        transform[1, 2], transform[2, 1] = -math.sin(angle), math.sin(angle)
    p4 = batch.p4 @ transform.T
    if kind != "lorentz":
        p4[..., 1:] += torch.tensor([0.3, -0.4, 0.2])
    expected = exp.model(batch).logits
    torch.testing.assert_close(
        exp.model(replace(batch, p4=p4)).logits, expected, atol=1e-7, rtol=1e-7
    )
    changed = exp.model(replace(batch, p4=batch.p4 * 2)).logits
    assert (expected - changed).abs().max() > 1e-8


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*/gatr*.yml")))
def test_physics_cards_and_native_tasks(path):
    """
    Exercise classification or regression from each physics card
    """
    from drivers.tau3_classification.geometry import hit_momenta

    config = load_config(path)
    if config.model.gatr_algebra == "conformal":
        assert not config.training.amp and not config.training.tf32
    config.model.width, config.model.heads, config.model.depth = 8, 2, 2
    config.model.gatr_mv_channels = 2
    exp = build_experiment(config)
    samples = []
    for i, count in enumerate((3, 5)):
        if config.driver["name"].startswith("tau3"):
            features = torch.randn(count, 4)
            features[:, :3] *= 1000
            features[:, 3] = torch.rand(count) + 0.01
            p4 = hit_momenta(features)
        elif config.driver["name"] == "pi0gamma":
            features = torch.randn(count, len(exp.driver.input_spec.features))
            p4 = None
        else:
            spatial = torch.randn(count, 3) * 10
            p4 = torch.cat((spatial.norm(dim=-1, keepdim=True), spatial), -1)
            features = p4.float()
        target = (
            torch.tensor(i)
            if config.task.name == "classification"
            else torch.randn(len(exp.task.spec.outputs))
        )
        global_dim = len(exp.driver.input_spec.global_features)
        samples.append(Sample(features, target, p4=p4, global_features=torch.randn(global_dim) if global_dim else None))
    batch = collate_samples(samples)
    loss = exp.task.compute_loss(exp.model(batch), batch).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )
    if config.driver["name"] == "pi0gamma":
        assert config.model.gatr_reinsert
        exp.model.eval()
        with torch.no_grad():
            backbone = exp.model.backbone
            points, cell_scalars, mask, _ = backbone.embed(batch)
            torch.testing.assert_close(cell_scalars[..., 1:][mask], batch.features[mask])
            torch.testing.assert_close(points[..., 0, :][mask], backbone.algebra.embed(batch.features[mask][:, :3]))
            cells = exp.model.backbone(batch).features
            assert cells.shape == (*batch.mask.shape, config.model.width)
            expected = torch.stack([event[mask].mean(0) for event, mask in zip(cells, batch.mask)])
            expected = expected + exp.model.global_fuse(torch.cat([expected, batch.global_features], -1))
            torch.testing.assert_close(exp.model.encode(batch), expected)


@pytest.mark.parametrize("kind", SIGNATURES)
def test_checkpoint_and_evaluation(tmp_path, kind):
    """
    Train, reload and evaluate each algebra through the public APIs
    """
    config = configuration(kind)
    best = Trainer(tmp_path / kind).fit(build_experiment(config))
    predictor = load_predictor(best, "cpu")
    samples = predictor.experiment.driver.dataset("test")
    metrics = evaluate(
        predictor, samples, tmp_path / "evaluation", plots=False, bootstrap_samples=0
    )
    assert math.isfinite(metrics["preprocessed_loss"])


def test_tau_conformal_detector_precision():
    """
    Compare FP32 logits and gradients with FP64 for wide detector hit clouds
    """
    config = load_config("configs/tau3_classification/gatr_conformal.yml")
    exp = build_experiment(config)
    exp.model.eval()
    reference = deepcopy(exp.model).double()
    features = torch.randn(2, 32, 4)
    features[..., :3] *= 1000
    features[..., 3] = features[..., 3].abs() * 0.01
    batch = collate_samples(
        [Sample(x, torch.tensor(i)) for i, x in enumerate(features)]
    )
    actual = exp.model(batch).logits
    expected = reference(batch).logits
    assert (actual.double() - expected).norm() / expected.norm() < 1e-4
    F.cross_entropy(actual, batch.targets).backward()
    F.cross_entropy(expected, batch.targets).backward()
    gradient = torch.cat(
        [p.grad.flatten() for p in exp.model.parameters() if p.grad is not None]
    ).double()
    reference_gradient = torch.cat(
        [p.grad.flatten() for p in reference.parameters() if p.grad is not None]
    )
    assert torch.isfinite(gradient).all()
    assert (gradient - reference_gradient).norm() / reference_gradient.norm() < 1e-4


def test_lorentz_covariant_head():
    """
    Transform deterministic four-vector predictions with the input boost
    """
    data = configuration("lorentz").to_dict()
    data["task"].update(name="covariant_regression", lossfunc="mse")
    data["preprocessing"]["normalize_targets"] = False
    exp = build_experiment(parse_config(data))
    exp.model.double().eval()
    batch = collate_samples(exp.driver.dataset("train")[:3])
    batch = replace(batch, p4=batch.p4.double(), features=batch.features.double())
    transform = torch.eye(4, dtype=torch.float64)
    transform[0, 0] = transform[1, 1] = math.cosh(0.4)
    transform[0, 1] = transform[1, 0] = math.sinh(0.4)
    expected = exp.model(batch).mean @ transform.T
    actual = exp.model(replace(batch, p4=batch.p4 @ transform.T)).mean
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "options",
    [
        {"gatr_algebra": "unknown"},
        {"gatr_mv_channels": 0},
        {"gatr_input_scale": -1.0},
        {"gatr_norm_epsilon": float("nan")},
        {"gatr_algebra": "lorentz", "gatr_center": True},
        {"gatr_algebra": "lorentz", "gatr_input": "coordinates"},
        {"gatr_input": "coordinates"},
        {"gatr_global_token": True, "pooling": "sum"},
        {"gatr_spurions": "beam"},
        {"gatr_algebra": "euclidean", "gatr_projective_improved": False},
    ],
)
def test_invalid_gatr_steering(options):
    """
    Reject invalid geometry, scaling and silently ignored steering
    """
    with pytest.raises(ValueError):
        configuration("projective", **options)


def test_published_lorentz_tagging_inputs_and_recipe():
    """
    Check 20 GeV scaling, beam-plane tokens and published optimizer steering
    """
    cfg = load_config("configs/toptag/gatr_lorentz.yml")
    assert (
        cfg.model.width,
        cfg.model.gatr_mv_channels,
        cfg.model.heads,
        cfg.model.depth,
    ) == (32, 16, 8, 12)
    assert cfg.training.optimizer == "lion" and cfg.training.batch_size == 128
    assert (
        cfg.training.max_steps == 200000 and cfg.training.scheduler_interval == "step"
    )
    assert cfg.training.learning_rate == 3e-4 and cfg.training.weight_decay == 0.2
    assert not cfg.training.amp and not cfg.training.token_dropout
    assert not cfg.driver["rotate"] and cfg.driver["max_particles"] == 200
    cfg.model.width, cfg.model.gatr_mv_channels, cfg.model.heads, cfg.model.depth = (
        8,
        2,
        2,
        1,
    )
    exp = build_experiment(cfg)
    p4 = torch.tensor([[40.0, 10.0, 20.0, 30.0], [20.0, 5.0, 2.0, 1.0]])
    batch = collate_samples([Sample(p4, torch.tensor(1), p4=p4)])
    mv, scalars, mask, reference = exp.model.backbone.embed(batch)
    assert mv.shape == (1, 5, 1, 16) and mask.all()
    assert scalars[0, 0, 0] == 1 and scalars[0, 1:].count_nonzero() == 0
    assert mv[0, 0].count_nonzero() == 0
    assert mv[0, 1, 0, 8] == 1 and mv[0, 1].count_nonzero() == 1
    assert mv[0, 2, 0, 1] == 1 and mv[0, 2].count_nonzero() == 1
    torch.testing.assert_close(mv[0, 3:, 0, 1:5], p4 / 20)
    prediction = exp.model(batch)
    assert prediction.logits.shape == (1, 1)
    expected = F.binary_cross_entropy_with_logits(
        prediction.logits[:, 0], batch.targets.float()
    )
    torch.testing.assert_close(exp.task.compute_loss(prediction, batch).loss, expected)
    exp.model.eval()
    padded = replace(
        batch,
        p4=F.pad(p4[None], (0, 0, 0, 2), value=float("nan")),
        features=F.pad(p4[None], (0, 0, 0, 2), value=float("nan")),
        mask=F.pad(batch.mask, (0, 2)),
    )
    torch.testing.assert_close(exp.model(padded).logits, exp.model(batch).logits)
