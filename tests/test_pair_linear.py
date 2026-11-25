# Equivariant pair contractions and derivatives
#
# mikael.mieskolainen@cern.ch (2026)

import pytest
import torch

from twistformer.geometry.pairs import PairLinear


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_reduced_pair_sums_match_dense_basis(normalize, amp, compiled):
    """
    Match all 15 pair basis terms and their parameter derivatives

    Args:
        normalize:  Normalize event and particle sums by valid multiplicity
        amp:  Use bf16 projections with float32 accumulation
        compiled:  Capture the pair layer and its backward graph
    """
    layer = PairLinear(3, 5, normalize=normalize)
    forward = (
        torch.compile(layer, backend="aot_eager", fullgraph=True, dynamic=True)
        if compiled
        else layer
    )
    x = torch.randn(3, 6, 6, 3, requires_grad=True)
    mask = torch.arange(6)[None] < torch.tensor([6, 3, 0])[:, None]
    valid = mask[:, :, None] & mask[:, None, :]
    clean = x.masked_fill(~valid[..., None], 0)
    diag = clean.diagonal(dim1=1, dim2=2).transpose(1, 2)
    row, col = clean.sum(2), clean.sum(1)
    total, trace = clean.sum((1, 2)), diag.sum(1)
    count = mask.sum(1).clamp_min(1).float()
    if normalize:
        total = total / count[:, None].square()
        trace = trace / count[:, None]
        row, col = row / count[:, None, None], col / count[:, None, None]
    eye = torch.eye(6)[None, :, :, None]
    basis = [
        total[:, None, None],
        total[:, None, None] * eye,
        trace[:, None, None],
        trace[:, None, None] * eye,
        row[:, :, None],
        col[:, None],
        col[:, :, None],
        row[:, None],
        clean,
        clean.transpose(1, 2),
        row[:, :, None] * eye,
        col[:, :, None] * eye,
        diag[:, :, None],
        diag[:, None],
        diag[:, :, None] * eye,
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = forward(x, mask)
        projected = [mapping(term) for mapping, term in zip(layer.maps, basis)]
        expected = projected[8] + projected[9] + layer.bias
        for index in (*range(8), *range(10, 15)):
            expected = expected + projected[index]
        expected = expected.masked_fill(~valid[..., None], 0)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
    weight = torch.randn_like(actual)
    inputs = (x, *layer.parameters())
    actual_gradients = torch.autograd.grad((actual * weight).sum(), inputs)
    expected_gradients = torch.autograd.grad((expected * weight).sum(), inputs)
    tolerance = 0.02 if amp else 3e-5
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance
        )
