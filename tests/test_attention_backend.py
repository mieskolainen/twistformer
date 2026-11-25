# Selectable self-attention kernels and masked gradients
#
# mikael.mieskolainen@cern.ch (2026)

import pytest
import torch
from torch.nn import functional as F

from twistformer import build_experiment, parse_config
from twistformer.data.batching import collate_samples
from twistformer.layers import attention, gatr


@pytest.mark.parametrize("with_bias", [False, True])
def test_masked_multi_query_uses_shared_head_views(monkeypatch, with_bias):
    """
    Retain masked SDPA backends and shared-key gradients for multi-query attention

    Args:
        monkeypatch:  Record SDPA dispatch
        with_bias:  Include trainable pair scores
    """
    original = F.scaled_dot_product_attention
    calls = []

    def record(q, k, v, **options):
        """
        Record head strides and grouped-query steering

        Args:
            q:  Query heads
            k:  Shared key heads
            v:  Shared value heads
            options:  SDPA steering
        Returns:
            output: Native attention output
        """
        calls.append((k.stride(1), v.stride(1), options["enable_gqa"]))
        return original(q, k, v, **options)

    q = torch.randn(2, 4, 7, 16, dtype=torch.float64, requires_grad=True)
    k, v = [torch.randn(2, 1, 7, 16, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    mask = torch.tensor([[1, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    bias = torch.randn(2, 1, 7, 7, dtype=torch.float64, requires_grad=True) if with_bias else None
    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    actual = attention.scaled_attention(q, k, v, mask, bias=bias, scale=0.3)
    keep = mask[:, None, None, :]
    expected = original(
        q, k, v, attn_mask=bias.masked_fill(~keep, -torch.inf) if with_bias else keep,
        scale=0.3, enable_gqa=True,
    ).masked_fill(~mask[:, None, :, None], 0)
    assert calls == [(0, 0, False)]
    torch.testing.assert_close(actual, expected)
    inputs = (q, k, v, bias) if with_bias else (q, k, v)
    for found, reference in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), inputs), strict=True,
    ):
        torch.testing.assert_close(found, reference)


@pytest.mark.parametrize("backend", ["sdpa", "auto", "flex"])
@pytest.mark.parametrize("compiling", [False, True])
def test_explicit_backend_dispatch(monkeypatch, backend, compiling):
    """
    Check backend selection, padding, grouped heads and masked gradients

    Args:
        monkeypatch:  CUDA dispatch replacements for CPU arithmetic
        backend:  Requested attention backend
        compiling:  Model graph capture state
    """
    calls = []

    def kernel(q, k, v, mask, bias, scale):
        """
        Record the Flex boundary and evaluate its equivalent score rule
        """
        calls.append(q.shape[-1])
        keep = mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep if bias is None else bias.masked_fill(~keep, -torch.inf),
            scale=scale, enable_gqa=True,
        )

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(attention, "fused_attention_available", lambda device: False)
    monkeypatch.setattr(attention, "flex_attention", kernel)
    monkeypatch.setattr(attention, "masked_flex_attention", kernel)
    monkeypatch.setattr(attention, "fused_attention_kernel", lambda: kernel)
    q = torch.randn(2, 4, 5, 9, requires_grad=True)
    k, v = [torch.randn(2, 1, 5, 9, requires_grad=True) for _ in range(2)]
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    actual = attention.scaled_attention(q, k, v, mask, backend=backend)
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None, None, :], enable_gqa=True,
    ).masked_fill(~mask[:, None, :, None], 0)
    assert calls == ([16] if backend == "flex" else [])
    torch.testing.assert_close(actual, expected)
    for a, e in zip(
        torch.autograd.grad(actual.square().sum(), (q, k, v)),
        torch.autograd.grad(expected.square().sum(), (q, k, v)),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("model,match", [
    ({"attention_backend": "unknown"}, "attention_backend"),
    ({"family": "twistformer"}, "encoders"),
    ({"family": "triformer"}, "encoders"),
    ({"family": "pelican"}, "encoders"),
    ({"pair_bias": "jet"}, "pair_bias"),
    ({"pair_bias": "mandelstam"}, "pair_bias"),
    ({"attention_dropout": 0.1}, "attention_dropout"),
])
def test_backend_config_rejects_incompatible_models(model, match):
    """
    Reject incompatible backend choices before loading events
    """
    with pytest.raises(ValueError, match=match):
        parse_config({"driver": {"name": "synthetic"}, "model": {"attention_backend": "flex", **model}})


@pytest.mark.parametrize("model", [
    {"attention": "standard"},
    {"attention": "hyperbolic"},
    {"attention": "projective"},
    *[{"family": "gatr", "gatr_algebra": kind, "gatr_mv_channels": 4} for kind in ("euclidean", "projective", "conformal", "lorentz")],
])
def test_model_propagates_backend(monkeypatch, model):
    """
    Pass the steering choice through every compatible encoder
    """
    calls = []
    original = attention.scaled_attention

    def record(*args, backend, **kwargs):
        """
        Record model dispatch while using CPU SDPA for the arithmetic
        """
        calls.append(backend)
        return original(*args, backend="sdpa", **kwargs)

    monkeypatch.setattr(attention, "scaled_attention", record)
    monkeypatch.setattr(gatr, "scaled_attention", record)
    cfg = parse_config({
        "driver": {"name": "synthetic", "train_size": 4, "max_tokens": 5},
        "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0,
                  "attention_backend": "flex", **model},
    })
    exp = build_experiment(cfg)
    batch = collate_samples(exp.driver.dataset("train")[:2])
    features = exp.model.backbone(batch).features
    features.square().sum().backward()
    assert calls and set(calls) == {"flex"}
    assert all(p.grad.isfinite().all() for p in exp.model.parameters() if p.grad is not None)


def test_flex_requires_cuda():
    """
    Report unsupported CPU execution instead of silently switching backend
    """
    q = torch.randn(1, 1, 2, 16)
    with pytest.raises(ValueError, match="requires CUDA"):
        attention.scaled_attention(q, q, q, torch.ones(1, 2, dtype=torch.bool), backend="flex")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_masked_multi_query_retains_efficient_sdpa():
    """
    Run masked shared-KV attention with the quadratic math backend disabled
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(2, 4, 65, 104, device="cuda", requires_grad=True)
    k, v = [torch.randn(2, 1, 65, 104, device="cuda", requires_grad=True) for _ in range(2)]
    mask = torch.arange(65, device="cuda")[None] < torch.tensor([65, 37], device="cuda")[:, None]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual = attention.scaled_attention(q, k, v, mask)
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :], enable_gqa=True)
        expected = expected.masked_fill(~mask[:, None, :, None], 0)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for found, reference in zip(
        torch.autograd.grad(actual.square().mean(), (q, k, v)),
        torch.autograd.grad(expected.square().mean(), (q, k, v)), strict=True,
    ):
        torch.testing.assert_close(found, reference, atol=2e-7, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_cuda_flex_matches_sdpa(dtype, kv_heads):
    """
    Compare real Flex outputs and gradients with native SDPA on CUDA
    """
    q = torch.randn(2, 2, 65, 24, device="cuda", dtype=dtype, requires_grad=True)
    k, v = [torch.randn(2, kv_heads, 65, 24, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)]
    mask = torch.arange(65, device="cuda")[None] < torch.tensor([65, 37], device="cuda")[:, None]
    actual = attention.scaled_attention(q, k, v, mask, backend="flex")
    expected = attention.scaled_attention(q, k, v, mask, backend="sdpa")
    tolerance = 0.04 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for a, e in zip(
        torch.autograd.grad(actual.float().square().mean(), (q, k, v)),
        torch.autograd.grad(expected.float().square().mean(), (q, k, v)),
        strict=True,
    ):
        torch.testing.assert_close(a, e, atol=tolerance * 0.001, rtol=tolerance)
