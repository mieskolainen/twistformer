# Torch backend API selection and attention fallbacks
#
# mikael.mieskolainen@cern.ch (2026)

from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from twistformer.api.config import TrainingConfig
from twistformer.training.engine import configure_precision, make_grad_scaler
from twistformer.layers import attention


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_grouped_attention_matches_repeated_heads(kv_heads):
    """
    Match multi-query outputs and gradients to explicit key and value heads

    Args:
        kv_heads:  Number of shared key and value heads
    """
    q = torch.randn(2, 4, 5, 8, requires_grad=True)
    k, v = [torch.randn(2, kv_heads, 5, 8, requires_grad=True) for _ in range(2)]
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    actual = attention.scaled_attention(q, k, v, mask, scale=0.3)
    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4 // kv_heads, dim=1),
        v.repeat_interleave(4 // kv_heads, dim=1),
        attn_mask=mask[:, None, None, :],
        scale=0.3,
    ).masked_fill(~mask[:, None, :, None], 0)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual.square().sum(), (q, k, v))
    reference = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(gradients, reference, strict=True):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compiled_plain_attention_uses_native_sdpa(monkeypatch, dtype):
    """
    Keep plain attention on native SDPA with padded and dropped tokens

    Args:
        monkeypatch:  Emulate CUDA dispatch and compiled model capture
        dtype:  Reduced precision attention dtype
    Returns:
        None: Outputs and zero-upstream gradients match native SDPA
    """
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(attention, "fused_attention_available", lambda device: True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def forbidden(*args, **kwargs):
        """
        Reject custom kernels for compiled plain attention
        """
        raise AssertionError("Plain compiled attention must use native SDPA")

    monkeypatch.setattr(attention, "flex_attention", forbidden)
    monkeypatch.setattr(attention, "masked_flex_attention", forbidden)
    monkeypatch.setattr(attention, "fused_attention_kernel", forbidden)
    monkeypatch.setattr(attention, "varlen_attn", forbidden)
    q = torch.randn(2, 4, 5, 9, dtype=dtype, requires_grad=True)
    k, v = [torch.randn(2, 1, 5, 9, dtype=dtype, requires_grad=True) for _ in range(2)]
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    actual = attention.scaled_attention(q, k, v, mask, backend="auto")
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1), attn_mask=mask[:, None, None, :]
    ).masked_fill(~mask[:, None, :, None], 0)
    torch.testing.assert_close(actual, expected)
    for upstream in (torch.zeros_like(actual), torch.randn_like(actual)):
        gradients = torch.autograd.grad(actual, (q, k, v), upstream, retain_graph=True)
        reference = torch.autograd.grad(expected, (q, k, v), upstream, retain_graph=True)
        for gradient, target in zip(gradients, reference, strict=True):
            assert torch.isfinite(gradient).all()
            assert not gradient.transpose(1, 2)[~mask].count_nonzero()
            torch.testing.assert_close(gradient, target)


@pytest.mark.parametrize("compiling", [False, True])
@pytest.mark.parametrize("shared_bias", [False, True])
def test_learned_bias_bypasses_flex_backward(monkeypatch, compiling, shared_bias):
    """
    Retain pair-score gradients without the FlexAttention backward compiler

    Args:
        monkeypatch:  CUDA dispatch and compiler substitutions
        compiling:  Eager or compiled model dispatch
        shared_bias:  Broadcast one learned pair score over all attention heads
    """
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(attention, "fused_attention_available", lambda device: True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)

    def forbidden(*args, **kwargs):
        """
        Reject FlexAttention for differentiable pair scores
        """
        raise AssertionError("Learned pair scores must use SDPA")

    monkeypatch.setattr(attention, "flex_attention", forbidden)
    monkeypatch.setattr(attention, "masked_flex_attention", forbidden)
    monkeypatch.setattr(attention, "fused_attention_kernel", forbidden)
    q, k, v = [
        torch.randn(2, 4, 7, 8, dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    scores = torch.randn(2, 7, 7, 1 if shared_bias else 4, requires_grad=True)
    bias = scores.to(q.dtype).permute(0, 3, 1, 2)
    mask = torch.tensor([[1, 1, 0, 1, 0, 0, 0], [1, 0, 1, 1, 1, 0, 1]], dtype=torch.bool)
    actual = attention.scaled_attention(q, k, v, mask, bias, scale=0.3)
    expected = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=bias.masked_fill(~mask[:, None, None, :], -torch.inf),
        scale=0.3,
    ).masked_fill(~mask[:, None, :, None], 0)
    inputs = (q, k, v, scores)
    actual_grad = torch.autograd.grad(actual.float().square().sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.float().square().sum(), inputs)
    torch.testing.assert_close(actual, expected)
    assert not actual.transpose(1, 2)[~mask].count_nonzero()
    assert actual_grad[-1].count_nonzero()
    for a, e in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_learned_bias_dynamic_backward():
    """
    Match BF16 compiled pair-score gradients across event multiplicities
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    compiled = torch.compile(attention.scaled_attention, dynamic=True, fullgraph=True)
    for batch, length in [(2, 13), (3, 19)]:
        q, k, v = [
            torch.randn(batch, 4, length, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        ]
        scores = torch.randn(batch, length, length, 4, device="cuda", requires_grad=True)
        bias = scores.to(q.dtype).permute(0, 3, 1, 2)
        mask = torch.rand(batch, length, device="cuda") > 0.3
        mask[:, 0] = True
        actual = compiled(q, k, v, mask, bias)
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=bias.masked_fill(~mask[:, None, None, :], -torch.inf),
            ).masked_fill(~mask[:, None, :, None], 0)
        inputs = (q, k, v, scores)
        actual_grad = torch.autograd.grad(actual.float().square().sum(), inputs, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.float().square().sum(), inputs)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=3e-2)
        for a, e in zip(actual_grad, expected_grad, strict=True):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, e, atol=5e-2, rtol=5e-2)


def test_packed_multi_query_restores_query_heads(monkeypatch):
    """
    Scatter packed multi-query outputs back to all query heads

    Args:
        monkeypatch:  CUDA capability and variable-length kernel substitutions
    """
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: True
    )
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)

    def packed(q, k, v, offsets_q, offsets_k, max_q, max_k, *, scale, enable_gqa):
        """
        Evaluate each packed event with the PyTorch attention reference

        Args:
            q:  Packed queries
            k:  Packed shared keys
            v:  Packed shared values
            offsets_q:  Query event offsets
            offsets_k:  Key event offsets
            max_q:  Maximum query length
            max_k:  Maximum key length
            scale:  Physical score normalization
            enable_gqa:  Grouped-query switch
        Returns:
            context: Packed values with the query head count
        """
        assert enable_gqa and q.shape[-1] == 16
        outputs = []
        for start, stop in zip(offsets_q[:-1], offsets_q[1:]):
            streams = [t[start:stop].transpose(0, 1)[None] for t in (q, k, v)]
            outputs.append(
                F.scaled_dot_product_attention(*streams, scale=scale, enable_gqa=True)[
                    0
                ].transpose(0, 1)
            )
        return torch.cat(outputs)

    monkeypatch.setattr(attention, "varlen_attn", packed)
    q = torch.randn(2, 4, 5, 9, dtype=torch.float16, requires_grad=True)
    k, v = [
        torch.randn(2, 1, 5, 9, dtype=torch.float16, requires_grad=True)
        for _ in range(2)
    ]
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    actual = attention.scaled_attention(q, k, v, mask, backend="auto")
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None, None, :], enable_gqa=True
    ).masked_fill(~mask[:, None, :, None], 0)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    gradients = torch.autograd.grad(actual.square().sum(), (q, k, v))
    reference = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(gradients, reference, strict=True):
        torch.testing.assert_close(a, e, atol=1e-2, rtol=1e-2)


def test_tf32_uses_modern_backend_api(monkeypatch):
    """
    Set matrix and convolution precision through the current backend API

    Args:
        monkeypatch:  Replace backend precision controls
    """
    matmul = SimpleNamespace(fp32_precision="ieee")
    conv = SimpleNamespace(fp32_precision="ieee")
    rnn = SimpleNamespace(fp32_precision="ieee")
    monkeypatch.setattr(
        torch,
        "backends",
        SimpleNamespace(
            cuda=SimpleNamespace(matmul=matmul), cudnn=SimpleNamespace(conv=conv, rnn=rnn)
        ),
    )
    for enabled in (True, False):
        configure_precision(TrainingConfig(tf32=enabled))
        assert (
            matmul.fp32_precision
            == conv.fp32_precision
            == rnn.fp32_precision
            == ("tf32" if enabled else "ieee")
        )


def test_scaler_uses_cuda_namespace_when_unified_api_is_missing(monkeypatch):
    calls = []
    sentinel = object()

    def construct(*, enabled):
        calls.append(enabled)
        return sentinel

    monkeypatch.setattr(torch, "amp", SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "amp", SimpleNamespace(GradScaler=construct))
    assert make_grad_scaler(torch.device("cuda"), True) is sentinel
    assert make_grad_scaler(torch.device("cpu"), False) is sentinel
    assert calls == [True, False]
    with pytest.raises(ValueError, match="only on CUDA"):
        make_grad_scaler(torch.device("cpu"), True)


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("availability_api", [False, True])
def test_attention_without_fused_kernels(monkeypatch, biased, availability_api):
    # Exercise CUDA dispatcher decisions with CPU tensors for the arithmetic check
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: availability_api
    )
    monkeypatch.setattr(attention, "varlen_attn", None)
    monkeypatch.setattr(attention, "flex_attention", None)
    q, k, v = [
        torch.randn(2, 2, 3, 8, dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]
    mask = torch.tensor([[True, True, False], [True, False, False]])
    bias = (
        torch.randn(2, 2, 3, 3, dtype=torch.float16, requires_grad=True)
        if biased
        else None
    )
    actual = attention.scaled_attention(q, k, v, mask, bias)
    keep = mask[:, None, None, :]
    attn_mask = bias.masked_fill(~keep, -torch.inf) if biased else keep
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).masked_fill(
        ~mask[:, None, :, None], 0
    )
    torch.testing.assert_close(actual, expected)
    inputs = (q, k, v, bias) if biased else (q, k, v)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    for a, e in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, e)


def test_compiled_dispatch_avoids_boolean_packing(monkeypatch):
    """
    Exercise the compiled CUDA dispatch without requiring a GPU
    """
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: True
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def forbidden(*args, **kwargs):
        """
        Reject data-dependent packing and host Boolean conversion
        """
        raise AssertionError("Compiled attention must retain masked tensor shapes")

    monkeypatch.setattr(attention, "varlen_attn", forbidden)
    monkeypatch.setattr(attention, "flex_attention", None)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    q, k, v = [
        torch.randn(2, 2, 5, 8, dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]
    mask = torch.tensor([[1, 0, 1, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    actual = attention.scaled_attention(q, k, v, mask)
    expected = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None, None, :]
    ).masked_fill(~mask[:, None, :, None], 0)
    actual_grad = torch.autograd.grad(actual.square().sum(), (q, k, v))
    expected_grad = torch.autograd.grad(expected.square().sum(), (q, k, v))
    monkeypatch.undo()
    torch.testing.assert_close(actual, expected)
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FlexAttention backward"
))])
def test_flex_attention_full_graph_and_geometry_gradients(device):
    """
    Check compiled FlexAttention scores and CUDA derivatives against dense attention

    Args:
        device:  Device for the compiled attention kernel
    """
    gradients = device == "cuda"
    q, k, v = [torch.randn(2, 2, 5, 16, device=device, requires_grad=gradients) for _ in range(3)]
    bias = torch.randn(2, 2, 5, 5, device=device, requires_grad=gradients)
    mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1]], dtype=torch.bool, device=device)
    compiled = torch.compile(
        attention.masked_flex_attention, backend="aot_eager", fullgraph=True
    )
    actual = compiled(q, k, v, mask, bias, 0.25)
    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=bias.masked_fill(~mask[:, None, None, :], -torch.inf),
        scale=0.25,
    )
    torch.testing.assert_close(actual, expected)
    if not gradients:
        return
    inputs = (q, k, v, bias)
    for a, e in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), inputs),
    ):
        torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("kind", ["standard", "hyperbolic"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attention_fullgraph_backward_variable_masks(kind, device):
    """
    Match eager outputs and gradients through full graph capture across lengths
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = torch.float16 if device == "cuda" else torch.float32
    layer = attention.Attention(32, 4, kind=kind).to(device=device, dtype=dtype).eval()
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True, dynamic=True)
    for size, length in [(2, 5), (3, 8), (1, 4)]:
        x = torch.randn(
            size, length, 32, device=device, dtype=dtype, requires_grad=True
        )
        mask = torch.rand(size, length, device=device) > 0.4
        mask[:, 0] = True
        actual = compiled(x, mask)
        expected = layer(x, mask)
        inputs = (x, *layer.parameters())
        actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
        expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
        tolerance = 5e-3 if device == "cuda" else 2e-5
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        for a, e in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, e, atol=tolerance, rtol=tolerance)
        assert torch.isfinite(actual).all()
        assert not actual[~mask].count_nonzero()


@pytest.mark.parametrize(
    "mode", ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
)
@pytest.mark.parametrize("dynamic", [True, False, None])
def test_compile_policy_preserves_mode_and_skips_dynamic_captures(monkeypatch, mode, dynamic):
    """
    Retain mode tuning while avoiding captures for changing event multiplicities
    """
    from twistformer.training.engine import compile_model

    model = torch.nn.Linear(4, 2)
    calls = []

    def capture(module, **options):
        """
        Record compiler options without building machine code

        Args:
            module:  Model passed to the compiler
            options:  Compiler keyword arguments
        Returns:
            model: Original model
        """
        calls.append(options)
        return module

    monkeypatch.setattr(torch, "compile", capture)
    assert compile_model(model, TrainingConfig()) is model
    assert not calls
    assert (
        compile_model(model, TrainingConfig(compile=True, compile_mode=mode, compile_dynamic=dynamic)) is model
    )
    expected = dict(torch._inductor.list_mode_options(mode))
    expected["triton.cudagraph_skip_dynamic_graphs"] = True
    assert calls == [{"dynamic": dynamic, "options": expected}]
