# Fast geometric contractions and independent derivatives
#
# mikael.mieskolainen@cern.ch (2026)

import pytest
import torch

from twistformer.geometry.clifford import BladeProduct, CliffordAlgebra, SIGNATURES, blade_product, sparse_product
from twistformer.layers.gatr import EquivariantLinear, project_group
from twistformer.layers.pelican import ScalarLift
from twistformer.training.engine import configure_precision, mixed_precision_dtype
from twistformer.api.config import TrainingConfig


@pytest.mark.parametrize("kind", SIGNATURES)
@pytest.mark.parametrize("compiled", [False, True])
def test_blade_derivatives_with_broadcasting(kind, compiled):
    """
    Match direct bilinear derivatives and higher derivatives to blade arithmetic

    Args:
        kind:  Clifford algebra
        compiled:  Capture the product and its backward graph
    """
    algebra = CliffordAlgebra(kind).double()
    left = torch.randn(2, 1, algebra.dim, dtype=torch.float64, requires_grad=True)
    right = torch.randn(3, algebra.dim, dtype=torch.float64, requires_grad=True)
    forward = torch.compile(BladeProduct.apply, backend="aot_eager", fullgraph=True) if compiled else BladeProduct.apply
    actual = forward(left, right, algebra.gp_right, algebra.gp_sign, algebra.gp_gradient_right, algebra.gp_gradient_sign)
    expected = sparse_product(left, right, algebra.gp_right, algebra.gp_sign)
    if kind == "projective":
        actual = actual + 0.7 * forward(
            left, right, algebra.join_right, algebra.join_sign, algebra.join_gradient_right, algebra.join_gradient_sign
        )
        expected = expected + 0.7 * torch.einsum("...i,...j,oij->...o", left, right, algebra.join_kernel)
    torch.testing.assert_close(actual, expected)
    found = torch.autograd.grad(actual.square().sum(), (left, right), create_graph=not compiled)
    reference = torch.autograd.grad(expected.square().sum(), (left, right), create_graph=not compiled)
    for a, b in zip(found, reference, strict=True):
        torch.testing.assert_close(a, b)
    if not compiled:
        found_second = torch.autograd.grad(sum(value.square().sum() for value in found), (left, right))
        reference_second = torch.autograd.grad(sum(value.square().sum() for value in reference), (left, right))
        for a, b in zip(found_second, reference_second, strict=True):
            torch.testing.assert_close(a, b)


def test_precision_policy_uses_modern_controls():
    """
    Keep matrix and convolution precision consistent with the training policy
    """
    for enabled in (True, False):
        configure_precision(TrainingConfig(tf32=enabled))
        assert torch.backends.cuda.matmul.fp32_precision == (
            "tf32" if enabled else "ieee"
        )
        assert torch.backends.cudnn.conv.fp32_precision == (
            "tf32" if enabled else "ieee"
        )


def test_auto_amp_uses_native_cuda_precision(monkeypatch):
    """
    Select FP16 when CUDA can only emulate BF16

    Args:
        monkeypatch:  Replace hardware precision detection
    """
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda including_emulation=True: including_emulation,
    )
    assert mixed_precision_dtype(torch.device("cuda")) == torch.float16
    assert mixed_precision_dtype(torch.device("cuda"), "bf16") == torch.bfloat16
    assert mixed_precision_dtype(torch.device("cpu")) == torch.bfloat16


@pytest.mark.parametrize("kind", SIGNATURES)
def test_grouped_geometric_projections(kind):
    """
    Match combined projections and derivatives to independent equivariant maps

    Args:
        kind:  Clifford algebra
    """
    algebra = CliffordAlgebra(kind).double()
    layers = [
        EquivariantLinear(algebra, 3, count, 5, scalar).double()
        for count, scalar in ((4, 7), (2, 0), (2, 3))
    ]
    mv = torch.randn(2, 6, 3, algebra.dim, dtype=torch.float64, requires_grad=True)
    scalars = torch.randn(2, 6, 5, dtype=torch.float64, requires_grad=True)
    actual = project_group(layers, mv, scalars)
    actual_loss, expected_loss = 0, 0
    for layer, (found_mv, found_s) in zip(layers, actual):
        kernel = torch.einsum("oib,bxy->oixy", layer.weight, algebra.linear_basis)
        expected_mv = torch.einsum("...iy,oixy->...ox", mv, kernel)
        constants = layer.to_mv(scalars).unflatten(-1, (layer.out_mv, -1))
        expected_mv = expected_mv + constants @ algebra.constants
        torch.testing.assert_close(found_mv, expected_mv)
        actual_loss = actual_loss + found_mv.square().sum()
        expected_loss = expected_loss + expected_mv.square().sum()
        if layer.to_s is None:
            assert found_s is None
        else:
            expected_s = layer.to_s(
                torch.cat((scalars, algebra.invariants(mv).flatten(-2)), -1)
            )
            torch.testing.assert_close(found_s, expected_s)
            actual_loss = actual_loss + found_s.square().sum()
            expected_loss = expected_loss + expected_s.square().sum()
    inputs = (
        mv,
        scalars,
        *(parameter for layer in layers for parameter in layer.parameters()),
    )
    found = torch.autograd.grad(actual_loss, inputs)
    expected = torch.autograd.grad(expected_loss, inputs)
    for a, b in zip(found, expected):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("compiled", [False, True])
def test_sparse_join_and_derivatives(compiled):
    """
    Match sparse joins to the dense exterior-algebra contraction

    Args:
        compiled:  Capture the sparse product and its backward graph
    """
    algebra = CliffordAlgebra("projective").double()
    left = torch.randn(2, 5, 3, 16, dtype=torch.float64, requires_grad=True)
    right = torch.randn_like(left, requires_grad=True)
    forward = (
        torch.compile(sparse_product, backend="aot_eager", fullgraph=True)
        if compiled
        else sparse_product
    )
    actual = forward(left, right, algebra.join_right, algebra.join_sign)
    blades = torch.tensor(algebra.blades)
    disjoint = (blades[:, None] & blades[None, :]) == 0
    wedge = algebra.gp_kernel * disjoint
    dual = torch.zeros_like(wedge[0])
    full = algebra.blades[-1]
    for i, blade in enumerate(algebra.blades):
        complement = full ^ blade
        dual[algebra.index[complement], i] = blade_product(blade, complement, algebra.signature)[1]
    join_kernel = torch.einsum("oa,abc,bi,cj->oij", dual, wedge, dual, dual)
    torch.testing.assert_close(algebra.join_kernel, join_kernel)
    expected = torch.einsum("...i,...j,oij->...o", left, right, join_kernel)
    torch.testing.assert_close(actual, expected)
    found = torch.autograd.grad(actual.square().sum(), (left, right))
    reference = torch.autograd.grad(expected.square().sum(), (left, right))
    for a, b in zip(found, reference):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", SIGNATURES)
def test_cuda_fused_blade_products(kind):
    """
    Match fused CUDA blade contractions and gradients to direct arithmetic

    Args:
        kind:  Clifford algebra
    """
    algebra = CliffordAlgebra(kind).cuda()
    left = torch.randn(2, 5, 3, algebra.dim, device="cuda", requires_grad=True)
    right = torch.randn_like(left, requires_grad=True)
    actual = algebra.product(left, right)
    expected = sparse_product(left, right, algebra.gp_right, algebra.gp_sign)
    if kind == "projective":
        actual = actual + algebra.join(left, right, 1.0)
        expected = expected + torch.einsum(
            "...i,...j,oij->...o", left, right, algebra.join_kernel
        )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    found = torch.autograd.grad(actual.square().sum(), (left, right))
    reference = torch.autograd.grad(expected.square().sum(), (left, right))
    for a, b in zip(found, reference):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
))])
@pytest.mark.parametrize("kind", SIGNATURES)
def test_mlp_operands(kind, device, compiled):
    """
    Check blade contractions on strided MLP operands

    Args:
        kind:  Clifford algebra
        device:  Device for the public algebra methods
        compiled:  Capture the forward and backward graphs
    """
    algebra = CliffordAlgebra(kind).to(device)

    def bilinears(left, right, reference):
        """Evaluate the geometric MLP branches"""
        product = algebra.product(left, right)
        if kind == "projective":
            return product, algebra.join(left, right, reference)
        return (product,)

    forward = torch.compile(bilinears, backend="aot_eager", fullgraph=True, dynamic=True) if compiled else bilinears
    for cells in (3, 8):
        left = torch.randn(2, cells, 32, algebra.dim, device=device)[:, :, 8:16].requires_grad_()
        right = torch.randn(2, cells, 8, 2 * algebra.dim, device=device)[..., ::2].requires_grad_()
        reference = torch.randn(2, 1, 1, 1, device=device, requires_grad=True)
        actual = forward(left, right, reference)
        expected = (sparse_product(left, right, algebra.gp_right, algebra.gp_sign),)
        inputs = (left, right)
        if kind == "projective":
            expected += (reference * sparse_product(left, right, algebra.join_right, algebra.join_sign),)
            inputs += (reference,)
        for value, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, target, rtol=2e-5, atol=4e-6)
        weights = tuple(torch.randn_like(value) for value in actual)
        for found, target in zip(torch.autograd.grad(actual, inputs, weights),
                                 torch.autograd.grad(expected, inputs, weights), strict=True):
            torch.testing.assert_close(found, target, rtol=1e-4, atol=2e-4)


@pytest.mark.parametrize("amp", [False, True])
def test_scalar_lift_matches_dense_diagonal(amp):
    """
    Preserve the five scalar lift maps with linear-size diagonal updates

    Args:
        amp:  Use BF16 scalar projections
    """
    layer = ScalarLift(5)
    x = torch.randn(3, 7, 5, requires_grad=True)
    mask = torch.arange(7)[None] < torch.tensor([7, 4, 1])[:, None]
    clean = x.masked_fill(~mask[..., None], 0)
    mean = clean.sum(1) / mask.sum(1)[:, None]
    eye = torch.eye(7)[None, :, :, None]
    valid = mask[:, :, None] & mask[:, None, :]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        expected = (
            layer.maps[0](clean)[:, :, None] * eye
            + layer.maps[1](clean)[:, :, None]
            + layer.maps[2](clean)[:, None]
            + layer.maps[3](mean)[:, None, None] * eye
            + layer.maps[4](mean)[:, None, None]
        ).masked_fill(~valid[..., None], 0)
        actual = layer(x, mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    inputs = (x, *layer.parameters())
    found = torch.autograd.grad(actual.square().sum(), inputs)
    reference = torch.autograd.grad(expected.square().sum(), inputs)
    for a, b in zip(found, reference):
        torch.testing.assert_close(
            a, b, atol=0.3 if amp else 1e-5, rtol=0.03 if amp else 1e-5
        )
