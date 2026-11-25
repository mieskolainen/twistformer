# Compiled particle models and training modes
#
# mikael.mieskolainen@cern.ch (2026)

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch
from torch.nn import functional as F

from twistformer import build_experiment, parse_config
from twistformer.api.contracts import Prediction
from twistformer.data.batching import collate_samples
from twistformer.training.engine import mixed_precision_dtype

MODELS = [
    pytest.param({"family": "transformer"}, id="transformer"),
    pytest.param(
        {"family": "transformer", "attention": "projective"},
        id="transformer-projective",
    ),
    pytest.param(
        {
            "family": "transformer",
            "attention": "projective",
            "projective_coordinates": ["px", "py", "pz"],
            "reference_token": "frame",
        },
        id="transformer-projective-frame",
    ),
    pytest.param(
        {"family": "transformer", "reference_token": "frame"}, id="transformer-frame"
    ),
    pytest.param(
        {"family": "transformer", "pooling": "weighted_mean"}, id="weighted-pooling"
    ),
    pytest.param(
        {"family": "transformer", "attention": "hyperbolic", "pair_bias": "mandelstam"},
        id="hyperbolic",
    ),
    pytest.param({"family": "transformer", "pair_bias": "jet"}, id="jet"),
    pytest.param({"family": "triformer", "triangle_k": 2}, id="triformer"),
    pytest.param({"family": "twistformer"}, id="twistformer"),
    *[
        pytest.param(
            {"family": "twistformer", "reference_token": reference},
            id=f"twistformer-{reference}",
        )
        for reference in ("time", "beam_plane")
    ],
    pytest.param(
        {
            "family": "twistformer",
            "twistor_precision": "float32",
            "twistor_transport": "frame",
            "reference_token": "event",
            "representation_channels": 2,
            "pair_channels": 4,
            "three_point": True,
            "pooling": "pma",
        },
        id="twistformer-toptag",
    ),
    pytest.param({"family": "twistformer", "twistor_detector": True}, id="detector"),
    pytest.param({"family": "pelican"}, id="pelican"),
    pytest.param(
        {"family": "pelican", "pelican_triangle": True, "pelican_factorize": False},
        id="pelican-triangle",
    ),
    *[
        pytest.param(
            {
                "family": "gatr",
                "gatr_algebra": algebra,
                "gatr_multi_query": True,
                "gatr_checkpoint": True,
            },
            id=f"gatr-{algebra}",
        )
        for algebra in ("euclidean", "projective", "conformal", "lorentz")
    ],
]


def model_batch(options, dropout=0.0):
    """
    Construct a small particle model with padded physical events

    Args:
        options:  Model family and architecture options
        dropout:  Residual and attention dropout probability
    Returns:
        pair: Model and padded event batch
    """
    config = parse_config(
        {
            "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 4},
            "model": {
                "width": 8,
                "heads": 2,
                "depth": 2,
                **({"gatr_mv_channels": 2} if options["family"] == "gatr" else {}),
                "dropout": dropout,
                "attention_dropout": dropout if options["family"] != "pelican" else 0,
                **options,
            },
            "preprocessing": {
                "normalize_features": not bool(options.get("projective_coordinates"))
            },
        }
    )
    experiment = build_experiment(config)
    batch = collate_samples(experiment.driver.dataset("train")[:2])
    if options.get("pooling") == "weighted_mean":
        batch = replace(batch, token_weights=batch.mask.float() * 1.5)
    return experiment.model, batch


def prediction_tensors(prediction):
    """
    Select the tensors provided by a shared task head

    Args:
        prediction:  Classification or regression outputs
    Returns:
        tensors: Active prediction tensors
    """
    return [
        getattr(prediction, field.name)
        for field in fields(Prediction)
        if getattr(prediction, field.name) is not None
    ]


@pytest.mark.parametrize("options", MODELS)
def test_block_checkpointing_preserves_dropout_and_state(options):
    """
    Preserve dropout, gradients and saved parameter names under recomputation

    Args:
        options:  Model family and architecture options
    """
    options = {**options, "gatr_checkpoint": False}
    reference, batch = model_batch(options, dropout=0.2)
    model, _ = model_batch({**options, "checkpoint_blocks": True}, dropout=0.2)
    model.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(19)
    expected = reference(batch).logits
    expected.square().sum().backward()
    rng = torch.get_rng_state()
    torch.manual_seed(19)
    actual = model(batch).logits
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for actual_parameter, expected_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        assert (actual_parameter.grad is None) == (expected_parameter.grad is None)
        if actual_parameter.grad is not None:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
@pytest.mark.parametrize("options", MODELS)
def test_compiled_forward_backward_and_mode_transitions(options, checkpoint_blocks):
    """
    Match eager outputs and parameter gradients across modes and token lengths

    Args:
        options:  Model family and architecture options
        checkpoint_blocks:  Recompute block activations during backward
    """
    model, batch = model_batch(options)
    reference = deepcopy(model)
    if checkpoint_blocks:
        model, _ = model_batch({**options, "checkpoint_blocks": True})
        model.load_state_dict(reference.state_dict(), strict=True)
    # Tied triangle candidates have data-dependent cardinality
    compiled = torch.compile(
        model,
        backend="aot_eager",
        dynamic=True,
        fullgraph=options["family"] != "triformer",
    )
    padded = replace(
        batch,
        features=F.pad(batch.features, (0, 0, 0, 2), value=float("nan")),
        p4=F.pad(batch.p4, (0, 0, 0, 2), value=float("nan")),
        mask=F.pad(batch.mask, (0, 2)),
        token_weights=(
            F.pad(batch.token_weights, (0, 2))
            if batch.token_weights is not None
            else None
        ),
    )
    for training, sample in ((True, batch), (False, padded), (True, padded)):
        compiled.train(training)
        reference.train(training)
        assert all(module.training == training for module in compiled.modules())
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            actual = prediction_tensors(compiled(sample))
            expected = prediction_tensors(reference(sample))
            for a, e in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-4)
                assert torch.isfinite(a).all()
            if training:
                sum(value.square().sum() for value in actual).backward()
                sum(value.square().sum() for value in expected).backward()
                for a, e in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    assert (a.grad is None) == (e.grad is None)
                    if a.grad is not None:
                        assert torch.isfinite(a.grad).all()
                        torch.testing.assert_close(a.grad, e.grad, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("options", MODELS)
def test_compiled_dropout_and_inference_mode(options):
    """
    Disable dropout during inference and restore it when training resumes

    Args:
        options:  Model family and architecture options
    """
    model, batch = model_batch(options, dropout=0.4)
    compiled = torch.compile(model, backend="aot_eager")
    for training in (True, False, True):
        compiled.train(training)
        if training:
            first = compiled(batch).logits
            second = compiled(batch).logits
            assert not torch.equal(first, second)
            second.square().sum().backward()
            assert all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            model.zero_grad(set_to_none=True)
        else:
            with torch.inference_mode():
                first = compiled(batch).logits
                second = compiled(batch).logits
                torch.testing.assert_close(first, second, atol=0, rtol=0)
                torch.testing.assert_close(
                    first, model(batch).logits, atol=2e-5, rtol=2e-4
                )
            assert not first.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("options", MODELS)
def test_cuda_compiled_amp_matches_math(options):
    """
    Compare native CUDA compilation and AMP against unfused attention

    Args:
        options:  Model family and architecture options
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model, batch = model_batch(options)
    model = model.cuda()
    batch = batch.to("cuda")
    reference = deepcopy(model)
    compiled = torch.compile(
        model, dynamic=True, fullgraph=options["family"] != "triformer"
    )
    dtype = mixed_precision_dtype(torch.device("cuda"))
    for training in (True, False, True):
        compiled.train(training)
        reference.train(training)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=dtype):
            actual = compiled(batch).logits
            with sdpa_kernel(SDPBackend.MATH):
                expected = reference(batch).logits
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)
            assert torch.isfinite(actual).all()
        if training:
            actual.float().square().sum().backward()
            expected.float().square().sum().backward()
            for a, e in zip(model.parameters(), reference.parameters(), strict=True):
                assert (a.grad is None) == (e.grad is None)
                if a.grad is not None:
                    assert torch.isfinite(a.grad).all()
                    torch.testing.assert_close(a.grad, e.grad, atol=3e-2, rtol=5e-2)
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
