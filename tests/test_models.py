# Physics and workflow validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import pytest
import torch
from twistformer import parse_config, build_experiment
from twistformer.data.batching import collate_samples
from twistformer.geometry.lorentz import matrix, from_matrix
from twistformer.layers.attention import Attention, TriangleAttention


def setup(family="transformer", task="classification", **options):
    cfg = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 16,
                "val_size": 8,
                "test_size": 8,
                "max_tokens": 5,
            },
            "model": {
                "family": family,
                "width": 8,
                "heads": 2,
                "depth": 2,
                "dropout": 0.0,
                **options,
            },
            "task": {"name": task},
        }
    )
    exp = build_experiment(cfg)
    raw = exp.driver.dataset("train")
    pre = exp.driver.create_preprocessor().fit(raw, task == "gaussian_regression")
    batch = collate_samples([pre.transform(s) for s in raw[:3]])
    return exp, batch


@pytest.mark.parametrize(
    "family", ["transformer", "triformer", "twistformer", "pelican"]
)
@pytest.mark.parametrize("task", ["classification", "gaussian_regression"])
def test_model_backward_padding_permutation(family, task):
    exp, batch = setup(family, task)
    pred = exp.model(batch)
    loss = exp.task.compute_loss(pred, batch).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )
    exp.model.eval()
    key = "logits" if task == "classification" else "mean"
    expected = getattr(exp.model(batch), key)
    padded = replace(
        batch,
        features=torch.nn.functional.pad(batch.features, (0, 0, 0, 3)),
        p4=torch.nn.functional.pad(batch.p4, (0, 0, 0, 3)),
        mask=torch.nn.functional.pad(batch.mask, (0, 3)),
    )
    torch.testing.assert_close(
        getattr(exp.model(padded), key), expected, atol=1e-6, rtol=1e-5
    )
    permutation = torch.randperm(batch.mask.shape[1])
    permuted = replace(
        batch,
        features=batch.features[:, permutation],
        p4=batch.p4[:, permutation],
        mask=batch.mask[:, permutation],
    )
    torch.testing.assert_close(
        getattr(exp.model(permuted), key), expected, atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize("family", ["twistformer", "pelican"])
def test_end_to_end_lorentz(family):
    exp, batch = setup(family)
    exp.model.eval()
    if family == "twistformer":
        with torch.no_grad():
            exp.model.backbone.blocks[0].step.bias.fill_(0.3)
    s = torch.tensor([[1.4, 0], [0, 1 / 1.4]], dtype=torch.complex128)
    boosted = from_matrix(s @ matrix(batch.p4.double()) @ s.mH).float()
    transformed = replace(batch, p4=boosted, features=boosted)
    torch.testing.assert_close(
        exp.model(batch).logits, exp.model(transformed).logits, atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("kind", ["standard", "hyperbolic"])
def test_attention_mask_and_bias(kind):
    a = Attention(8, 2, kind=kind).eval()
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    y = a(x, mask)
    xp = torch.cat([x, torch.randn(2, 2, 8) * 100], 1)
    mp = torch.nn.functional.pad(mask, (0, 2))
    torch.testing.assert_close(a(xp, mp)[:, :4], y)
    torch.testing.assert_close(a(x, mask, torch.zeros(2, 2, 4, 4)), y)
    assert (y - a.out.bias).abs().max() > 0.01


def test_triangle_gamma_zero_and_single_token():
    a = TriangleAttention(8, 2, k=3, gamma=0).eval()
    x = torch.randn(1, 1, 8)
    mask = torch.ones(1, 1, dtype=torch.bool)
    y = a(x, mask)
    assert y.isfinite().all()


def test_dropout_and_mandelstam_hyperbolic():
    exp, batch = setup(dropout=0.6, attention="hyperbolic", pair_bias="mandelstam")
    exp.model.train()
    assert not torch.equal(exp.model(batch).logits, exp.model(batch).logits)
    loss = exp.task.compute_loss(exp.model(batch), batch).loss
    loss.backward()
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )


def test_amp_all_models():
    for family in ["transformer", "triformer", "twistformer", "pelican"]:
        exp, batch = setup(family)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = exp.model(batch)
        loss = exp.task.compute_loss(prediction, batch).loss
        loss.backward()
        assert loss.isfinite()


@pytest.mark.parametrize(
    "family", ["transformer", "triformer", "twistformer", "pelican"]
)
def test_compile_capture_matches_eager(family):
    exp, batch = setup(family)
    exp.model.eval()
    expected = exp.model(batch).logits
    compiled = torch.compile(exp.model, backend="eager")
    torch.testing.assert_close(compiled(batch).logits, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "family", ["transformer", "triformer", "twistformer", "pelican"]
)
def test_cuda_backward(family):
    exp, batch = setup(family)
    exp.model.cuda()
    batch = batch.to("cuda")
    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ):
        prediction = exp.model(batch)
    loss = exp.task.compute_loss(prediction, batch).loss
    loss.backward()
    assert loss.isfinite()
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize(
    "family,options",
    [
        ("transformer", {}),
        ("transformer", {"attention": "hyperbolic", "pair_bias": "mandelstam"}),
        ("transformer", {"pair_bias": "jet", "pooling": "attention_query"}),
        ("triformer", {"triangle_k": 2, "query_chunk": 1, "pair_bias": "mandelstam"}),
        ("twistformer", {"three_point": True}),
        ("twistformer", {"twistor_kind": "momentum", "three_point": True}),
        ("twistformer", {"geometry_update": False}),
        ("pelican", {}),
        ("pelican", {"norm": "rms", "pelican_factorize": False}),
    ],
)
def test_model_batch_independence_and_degenerate_events(family, options):
    from twistformer.api.contracts import Sample

    exp, _ = setup(family, **options)
    exp.model.eval()
    momenta = [
        [[2.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0]],
        [[2.0, 1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0]],
    ]
    samples = [
        Sample(torch.tensor(p), torch.tensor(i % 3), torch.tensor(p), id=str(i))
        for i, p in enumerate(momenta)
    ]
    batch = collate_samples(samples)
    predicted = exp.model(batch).logits
    individual = torch.cat(
        [exp.model(collate_samples([sample])).logits for sample in samples]
    )
    torch.testing.assert_close(predicted, individual, atol=2e-6, rtol=2e-5)
    predicted.square().sum().backward()
    assert all(
        p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None
    )


def test_mandelstam_uses_precise_invariants():
    from twistformer.layers.physics import MandelstamBias
    from twistformer.geometry.lorentz import dot

    layer = MandelstamBias(2)
    p = torch.tensor(
        [[[1e4, 1e4, 1.0, 0.0], [1e4, 1e4, -1.0, 0.0]]], dtype=torch.float32
    )
    captured = []
    hook = layer.network.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    layer(p, torch.ones(1, 2, dtype=torch.bool))
    hook.remove()
    pd = p.double()
    expected = (
        torch.stack(
            [
                dot(pd[:, :, None] + pd[:, None], pd[:, :, None] + pd[:, None]),
                dot(pd[:, :, None] - pd[:, None], pd[:, :, None] - pd[:, None]),
            ],
            -1,
        )
        .asinh()
        .float()
    )
    torch.testing.assert_close(captured[0], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["standard", "hyperbolic"])
def test_packed_flash_matches_masked_math(kind):
    from torch.nn.attention import sdpa_kernel, SDPBackend

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("FlashAttention requires Ampere or newer")
    layer = Attention(32, 4, kind=kind).cuda().half().eval()
    x = torch.randn(2, 6, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    mask = torch.tensor(
        [[1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 0]], device="cuda", dtype=torch.bool
    )
    actual = layer(x, mask)
    actual.square().sum().backward()
    gradient = x.grad.clone()
    x.grad = None
    with sdpa_kernel(SDPBackend.MATH):
        expected = layer(x, mask)
        expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(gradient, x.grad, atol=3e-3, rtol=3e-3)


def test_triangle_factorization_matches_joint_softmax_and_gradients():
    layer = TriangleAttention(8, 2, k=8, gamma=0, dropout=0).eval()
    x = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tensor([[True, False, True, True], [True, False, False, False]])
    actual = layer(x, mask)
    expected = layer.aggregate(layer.projections(x), mask, None)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual_grad = torch.autograd.grad(
        actual.square().sum(), (x, *layer.parameters()), retain_graph=True
    )
    expected_grad = torch.autograd.grad(
        expected.square().sum(), (x, *layer.parameters())
    )
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dim", [4, 8, 9, 16, 17])
@pytest.mark.parametrize("scale", [None, 0.25])
def test_fused_head_padding_preserves_attention(monkeypatch, dim, scale):
    """
    Check small-head dispatch, score scale, masks and QKV gradients on CPU

    Args:
        monkeypatch:  Scoped CUDA dispatch and kernel replacements
        dim:  Channels per attention head
        scale:  Optional score scale
    """
    from twistformer.layers import attention
    from torch.nn import functional as F

    def dense_kernel(q, k, v, mask, bias, scale):
        """
        Evaluate the fused kernel contract with dense attention

        Args:
            q:  Padded queries
            k:  Padded keys
            v:  Padded values
            mask:  Valid particle mask
            bias:  Geometry scores
            scale:  Score scale
        Returns:
            context: Dense attention output
        """
        assert q.shape[-1] >= 16
        return F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=bias.float().masked_fill(~mask[:, None, None, :], -torch.inf),
            scale=scale,
        )

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: True
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(attention, "flex_attention", dense_kernel)
    monkeypatch.setattr(attention, "fused_attention_kernel", lambda: dense_kernel)
    q, k, v = [
        torch.randn(2, 2, 5, dim, dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]
    bias = torch.randn(2, 2, 5, 5)
    mask = torch.tensor(
        [[True, False, True, True, False], [True, True, False, False, False]]
    )
    actual = attention.scaled_attention(q, k, v, mask, bias=bias, scale=scale, backend="auto")
    expected = F.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=bias.masked_fill(~mask[:, None, None, :], -torch.inf),
        scale=scale,
    ).masked_fill(~mask[:, None, :, None], 0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    gradients = torch.autograd.grad(actual.square().sum(), (q, k, v))
    reference = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for gradient, expected_gradient in zip(gradients, reference):
        torch.testing.assert_close(gradient, expected_gradient, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dim", [4, 8, 9, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_geometry_bias_matches_math_and_bias_gradients(dim, dtype):
    """
    Match fused geometry attention and gradients to dense attention

    Args:
        dim:  Channels per attention head
        dtype:  CUDA attention precision
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from twistformer.layers.attention import scaled_attention

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Fused attention requires Ampere or newer")
    q, k, v = [
        torch.randn(2, 2, 7, dim, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    bias = torch.randn(2, 2, 7, 7, device="cuda", dtype=dtype, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 0, 0]], device="cuda", dtype=torch.bool
    )
    actual = scaled_attention(q, k, v, mask, bias=bias)
    gradients = torch.autograd.grad(actual.square().sum(), (q, k, v, bias))
    with sdpa_kernel(SDPBackend.MATH):
        expected = scaled_attention(q, k, v, mask, bias=bias)
        reference_gradients = torch.autograd.grad(
            expected.square().sum(), (q, k, v, bias)
        )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for actual_gradient, expected_gradient in zip(gradients, reference_gradients):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=4 * tolerance,
            rtol=4 * tolerance,
        )
