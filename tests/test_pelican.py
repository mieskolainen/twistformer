# PELICAN basis, symmetry and training validation
#
# mikael.mieskolainen@cern.ch (2026)

from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from twistformer import Trainer, build_experiment, load_predictor, parse_config
from twistformer.layers.pelican import (
    PairAggregation,
    InvariantReadout,
    PowerEncoding,
    pair_statistics,
)
from twistformer.training.checkpoint import read_checkpoint, VERSION
from tests.test_models import setup


def contraction_basis(n):
    """
    Construct the 15 independent Kronecker contractions of a rank-two tensor

    Args:
        n:  Number of particles
    Returns:
        basis: Contraction tensors with axes map,i,j,k,l
    """
    i, j, k, l = torch.meshgrid(*[torch.arange(n)] * 4, indexing="ij")
    return torch.stack(
        [
            (i == k) & (j == l),
            (i == l) & (j == k),
            (i == j) & (i == k) & (i == l),
            (j == k) & (j == l),
            (i == k) & (i == l),
            (i == j) & (j == l),
            j == l,
            i == l,
            (i == j) & (i == k),
            j == k,
            i == k,
            k == l,
            (i == j) & (k == l),
            torch.ones_like(i, dtype=torch.bool),
            i == j,
        ]
    ).double()


@pytest.mark.parametrize("factorize", [False, True])
@pytest.mark.parametrize("multiplicity", [False, True])
def test_pair_maps_match_independent_contractions_and_gradients(
    factorize, multiplicity
):
    """
    Compare optimized maps with explicit Kronecker sums and their gradients

    Args:
        factorize:  Use factorized channel coefficients
        multiplicity:  Learn powers of the constituent count
    Returns:
        None: All maps and parameter gradients match the independent contraction
    """
    torch.manual_seed(7)
    layer = PairAggregation(3, 2, 2.5, factorize, multiplicity).double()
    x = torch.randn(2, 4, 4, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
    valid = mask[:, :, None] & mask[:, None, :]
    basis = contraction_basis(4)
    assert torch.linalg.matrix_rank(basis.flatten(1)) == 15
    operations = torch.einsum(
        "aijkl,bklc->baijc", basis, x.masked_fill(~valid[..., None], 0)
    )
    count = mask.sum(1).double()
    order = torch.tensor([0] * 5 + [1] * 8 + [2] * 2)
    operations = operations / count[:, None, None, None, None].pow(
        order[None, :, None, None, None]
    )
    if multiplicity:
        alpha = torch.cat([torch.zeros(5, 3), layer.alpha])
        operations = operations * (count[:, None, None, None, None] / 2.5).pow(
            alpha[None, :, None, None]
        )
    expected = torch.einsum("baijc,acd->bijd", operations, layer.coefficients())
    eye = torch.eye(4)[None, :, :, None]
    expected = (expected + layer.bias + eye * layer.diag_bias).masked_fill(
        ~valid[..., None], 0
    )
    actual = layer(x, mask)
    torch.testing.assert_close(actual, expected)
    params = (x, *layer.parameters())
    actual_grad = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), params)
    for found, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(found, reference, atol=1e-10, rtol=1e-9)


def test_invariant_readout_keeps_trace_and_total_separate():
    """
    Distinguish tensors with equal total contraction and different traces

    Returns:
        None: The two independent invariant maps retain distinct information
    """
    layer = InvariantReadout(1, 2, multiplicity=False)
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(2))
        layer.linear.bias.zero_()
    x = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    y = x.flip(1)
    mask = torch.ones(1, 2, dtype=torch.bool)
    torch.testing.assert_close(layer(x, mask), torch.tensor([[2.5, 2.5]]))
    y[0, 0, 0] += 2
    y[0, 0, 1] -= 2
    torch.testing.assert_close(layer(y, mask), torch.tensor([[2.5, 3.5]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pair_statistics_preserve_fp32_reductions(dtype):
    """
    Retain FP32 reductions when pair storage uses lower precision

    Args:
        dtype:  Pair storage dtype
    Returns:
        None: Dirty padding is excluded and reductions match explicit FP32 means
    """
    torch.manual_seed(27)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
    valid = mask[:, :, None] & mask[:, None, :]
    x = torch.randn(2, 4, 4, 3).to(dtype)
    clean = x.float().masked_fill(~valid[..., None], 0)
    values, diagonal, row, column, total, trace = pair_statistics(
        x.masked_fill(~valid[..., None], float("nan")), mask
    )
    count = mask.sum(1).float()
    expected_diagonal = clean.diagonal(dim1=1, dim2=2).transpose(1, 2)
    assert values.dtype == dtype
    torch.testing.assert_close(values.float(), clean)
    for actual, expected in (
        (diagonal, expected_diagonal),
        (row, clean.sum(2) / count[:, None, None]),
        (column, clean.sum(1) / count[:, None, None]),
        (total, clean.sum((1, 2)) / count[:, None].square()),
        (trace, expected_diagonal.sum(1) / count[:, None]),
    ):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("amp", [False, True])
def test_compiled_pair_and_triangle_backward(amp):
    """
    Compile the complete PELICAN gradient graph across particle multiplicities

    Args:
        amp:  Enable BF16 channel projections
    Returns:
        None: Compiled predictions and all parameter gradients match eager execution
    """
    torch.manual_seed(29)
    exp, batch = setup("pelican", norm="rms", pelican_triangle=True)
    reference = exp.model
    model = deepcopy(reference)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=True)
    for particles in (batch.mask.shape[1], 2):
        current = replace(
            batch,
            features=batch.features[:, :particles],
            p4=batch.p4[:, :particles],
            mask=batch.mask[:, :particles],
        )
        reference.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            expected = reference(current).logits
            actual = compiled(current).logits
        torch.testing.assert_close(actual, expected)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        for found, wanted in zip(model.parameters(), reference.parameters()):
            if amp:
                # Compare gradient norms at BF16 precision despite local cancellations
                error = (found.grad - wanted.grad).norm()
                scale = wanted.grad.norm().clamp_min(1e-8)
                assert error <= torch.finfo(torch.bfloat16).eps * scale
            else:
                torch.testing.assert_close(
                    found.grad, wanted.grad, atol=1e-6, rtol=1e-5
                )


def test_power_encoding_small_exponents_and_large_invariants():
    """
    Keep the signed power encoding finite near its logarithmic limit

    Returns:
        None: Forward values and learned-power gradients remain finite
    """
    layer = PowerEncoding(3)
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor([0.0, 0.05, 0.5]))
    x = torch.tensor([0.0, -1e-8, 1e-8, -1e4, 1e10])[:, None].expand(-1, 3)
    a = layer.alpha.double().square() + 1e-6
    expected = x.double().sign() * torch.expm1(a * x.double().abs().log1p()) / a
    actual = layer(x)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-8)
    actual.sum().backward()
    assert layer.alpha.grad.isfinite().all()


def test_power_encoding_input_derivative_at_null_contractions():
    """
    Preserve the unit derivative at zero Minkowski contractions

    Returns:
        None: Input gradients agree with the analytic slope and finite differences
    """
    layer = PowerEncoding(3)
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor([0.0, 0.05, 0.5]))
    x = torch.tensor([-2.0, -1e-8, 0.0, 1e-8, 2.0])[:, None].repeat(1, 3)
    x.requires_grad_()
    layer(x).sum().backward()
    exponent = layer.alpha.detach().square() + 1e-6
    expected = (1 + x.detach().abs()).pow(exponent - 1)
    torch.testing.assert_close(x.grad, expected)
    step = 1e-4
    zero = torch.zeros(1, 3)
    finite_difference = (layer(zero + step) - layer(zero - step)) / (2 * step)
    torch.testing.assert_close(x.grad[2:3], finite_difference, atol=1e-4, rtol=1e-4)
    assert layer.alpha.grad.isfinite().all()


@pytest.mark.parametrize("amp", [False, True])
def test_pair_encoder_dirty_padding_amp_and_all_gradients(amp):
    """
    Mask arbitrary padding before geometry and train every pair-encoder parameter

    Args:
        amp:  Enable BF16 channel projections
    Returns:
        None: Padding has no effect and every active parameter gets finite gradients
    """
    torch.manual_seed(12)
    exp, batch = setup("pelican", norm="rms")
    dirty = replace(
        batch, p4=batch.p4.masked_fill(~batch.mask[..., None], float("nan"))
    )
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        expected = exp.model(batch).logits
        actual = exp.model(dirty).logits
        loss = exp.task.compute_loss(exp.model(batch), batch).loss
    torch.testing.assert_close(actual, expected)
    loss.backward()
    for name, parameter in exp.model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name


@pytest.mark.parametrize(
    "options",
    [
        {"graph": "knn"},
        {"graph_gate_hidden": 4},
        {"graph_event_features": True},
        {"pelican_average_nobj": 0},
        {"pelican_momentum_scale": -1},
        {"pelican_message_width": -1},
        {"pelican_activation": "unknown"},
    ],
)
def test_invalid_and_legacy_options_are_rejected(options):
    """
    Reject graph approximations and invalid PELICAN parameters

    Args:
        options:  Invalid model settings
    Returns:
        None: Configuration raises a useful error
    """
    with pytest.raises(ValueError):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "model": {"family": "pelican", **options},
            }
        )


def test_legacy_graph_checkpoint_is_rejected(tmp_path):
    """
    Reject obsolete graph weights before reconstructing a pair encoder

    Args:
        tmp_path:  Temporary checkpoint directory
    Returns:
        None: The error identifies the incompatible graph architecture
    """
    path = tmp_path / "old.pt"
    torch.save(
        {
            "schema_version": VERSION,
            "preprocessing": {},
            "config": {"model": {"family": "pelican"}},
            "model": {"backbone.blocks.0.edge.0.weight": torch.ones(1)},
        },
        path,
    )
    with pytest.raises(ValueError, match="former graph model"):
        read_checkpoint(path)


@pytest.mark.parametrize("task", ["classification", "gaussian_regression"])
@pytest.mark.parametrize("triangles", [False, True])
def test_pelican_training_resume_and_prediction(tmp_path, task, triangles):
    """
    Train the pair encoder through the shared API and restore an exact continuation

    Args:
        tmp_path:  Temporary run directory
        task:  Classification or invariant scalar regression
        triangles:  Enable triangle interactions in each pair block
    Returns:
        None: Checkpoints resume exactly and produce finite predictions
    """
    cfg = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 12,
                "val_size": 6,
                "test_size": 4,
                "max_tokens": 4,
            },
            "model": {
                "family": "pelican",
                "width": 8,
                "heads": 2,
                "depth": 1,
                "norm": "rms",
                "dropout": 0.025,
                "pelican_triangle": triangles,
            },
            "task": {"name": task},
            "training": {
                "device": "cpu",
                "epochs": 2,
                "batch_size": 4,
                "eval_batch_size": 4,
                "save_every": 1,
            },
        }
    )
    experiment = build_experiment(cfg)
    best = Trainer(tmp_path / "full").fit(experiment)
    Trainer(
        tmp_path / "resumed", resume=tmp_path / "full/checkpoints/epoch_00000.pt"
    ).fit(build_experiment(cfg))
    full = read_checkpoint(tmp_path / "full/checkpoints/last.pt")
    resumed = read_checkpoint(tmp_path / "resumed/checkpoints/last.pt")
    for name, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
    predictor = load_predictor(best, "cpu")
    prediction = predictor.predict(experiment.driver.dataset("test"), batch_size=2)
    value = prediction.logits if task == "classification" else prediction.mean
    assert value.isfinite().all()
