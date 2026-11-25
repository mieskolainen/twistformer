# Physics and workflow validation
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from twistformer.geometry.lorentz import matrix, from_matrix, dot, null_projection
from twistformer.geometry.twistors import (
    twistor_inner,
    twistors_from_p4,
    pairwise_features,
    three_point_features,
    geometry_update,
    spin_frame,
    projective_normalize,
)
from twistformer.geometry.pairs import PairLinear


def momenta():
    spatial = torch.randn(2, 5, 3, dtype=torch.float64)
    energy = (spatial.square().sum(-1) + 0.4).sqrt()
    return torch.cat([energy[..., None], spatial], -1)


def transform():
    # General SL(2,C) element: rotation followed by noncompact boost
    rotation = torch.tensor([[0.8, 0.6j], [0.6j, 0.8]], dtype=torch.complex128)
    return rotation @ torch.diag(torch.tensor([1.5, 1 / 1.5], dtype=torch.complex128))


def test_incidence_basis_and_massive_projection():
    p = momenta()
    mask = torch.ones(2, 5, dtype=torch.bool)
    k = null_projection(p, p.sum(1))
    torch.testing.assert_close(
        dot(k, k), torch.zeros_like(k[..., 0]), atol=1e-12, rtol=1e-10
    )
    tw, g = twistors_from_p4(p, mask)
    torch.testing.assert_close(
        twistor_inner(tw.Z, tw.Z),
        torch.zeros(2, 5, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        matrix(k),
        tw.lam[..., :, None] @ tw.lam.conj()[..., None, :],
        atol=1e-12,
        rtol=1e-10,
    )
    s = transform()
    transformed = from_matrix(s @ matrix(p) @ s.mH)
    expected = from_matrix(s @ matrix(k) @ s.mH)
    torch.testing.assert_close(
        null_projection(transformed, transformed.sum(1)),
        expected,
        atol=1e-11,
        rtol=1e-10,
    )


def test_projective_and_lorentz_invariants_and_updates():
    p = momenta()
    mask = torch.ones(2, 5, dtype=torch.bool)
    tw, g = twistors_from_p4(p, mask, kind="momentum")
    features, overlap = pairwise_features(tw.Z, g, mask)
    factors = torch.randn(2, 5, 1, dtype=torch.complex128) + 0.5
    shifted, overlap2 = pairwise_features(tw.Z * factors, g, mask)
    torch.testing.assert_close(features, shifted, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(
        three_point_features(overlap, mask),
        three_point_features(overlap2, mask),
        atol=1e-11,
        rtol=1e-10,
    )
    a = torch.randn(2, 5, 5, dtype=torch.float64).softmax(-1)
    step = torch.full((2, 5), 0.05, dtype=torch.float64)
    update = geometry_update(tw.Z, g, overlap, a, step, mask)
    update2 = geometry_update(tw.Z * factors, g, overlap2, a, step, mask)
    torch.testing.assert_close(
        update2, update * factors / factors.abs(), atol=1e-11, rtol=1e-10
    )
    s = transform()
    representation = torch.block_diag(s, torch.linalg.inv(s).mH)
    inverse = torch.linalg.inv(representation)
    z2 = tw.Z @ representation.T
    g2 = inverse.mH @ g @ inverse
    f2, c2 = pairwise_features(z2, g2, mask)
    torch.testing.assert_close(features, f2, atol=1e-11, rtol=1e-10)
    u2 = geometry_update(z2, g2, c2, a, step, mask)
    torch.testing.assert_close(u2, update @ representation.T, atol=1e-11, rtol=1e-10)


def test_padding_and_degenerate_geometry():
    p = momenta()
    mask = torch.ones(2, 5, dtype=torch.bool)
    tw, g = twistors_from_p4(p, mask)
    f, c = pairwise_features(tw.Z, g, mask)
    padded = torch.nn.functional.pad(tw.Z, (0, 0, 0, 3))
    padded_mask = torch.nn.functional.pad(mask, (0, 3))
    f2, c2 = pairwise_features(padded, g, padded_mask)
    torch.testing.assert_close(f, f2[:, :5, :5])
    torch.testing.assert_close(
        three_point_features(c, mask), three_point_features(c2, padded_mask)[:, :5, :5]
    )
    p = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0]]], dtype=torch.float64
    )
    tw, g = twistors_from_p4(p, torch.ones(1, 2, dtype=torch.bool))
    assert projective_normalize(tw.Z, g).isfinite().all()
    assert tw.Z.abs().max() == 0


def test_frames_and_pair_basis():
    a, b = (
        torch.randn(2, 4, 2, dtype=torch.complex128),
        torch.randn(2, 4, 2, dtype=torch.complex128),
    )
    frame, inv, valid = spin_frame(a, b)
    s = transform()
    f2, i2, v2 = spin_frame(a @ s.T, b @ s.T)
    torch.testing.assert_close(f2, s @ frame)
    torch.testing.assert_close(i2, inv @ torch.linalg.inv(s))
    assert valid.all() and v2.all()
    _, inverse, valid = spin_frame(a, a)
    assert not valid.any() and inverse.abs().max() == 0
    layer = PairLinear(3, 4)
    x = torch.randn(2, 5, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    expected = layer(x, mask)[:, permutation][:, :, permutation]
    torch.testing.assert_close(
        layer(x[:, permutation][:, :, permutation], mask), expected
    )
    padded = torch.nn.functional.pad(x, (0, 0, 0, 2, 0, 2))
    pm = torch.nn.functional.pad(mask, (0, 2))
    torch.testing.assert_close(layer(padded, pm)[:, :5, :5], layer(x, mask))


def test_multiplicity_linear_commutes_with_group():
    from twistformer.layers.representations import RepresentationLinear

    layer = RepresentationLinear(3, 5).double()
    x = torch.randn(2, 4, 4, 3, dtype=torch.complex128)
    s = transform()
    g = torch.block_diag(s, torch.linalg.inv(s).mH)
    transformed = torch.einsum("ij,bnjc->bnic", g, x)
    expected = torch.einsum("ij,bnjc->bnic", g, layer(x))
    torch.testing.assert_close(layer(transformed), expected)


def test_spinor_chart_near_negative_z_pole():
    from twistformer.geometry.twistors import spinor_from_null

    p = torch.tensor(
        [[1e12, 1.0, 0.0, -1e12], [1e12, 1.0, 0.0, 1e12]], dtype=torch.float64
    )
    spinor = spinor_from_null(p)
    reconstructed = spinor[..., :, None] @ spinor.conj()[..., None, :]
    torch.testing.assert_close(reconstructed, matrix(p), atol=1e-12, rtol=1e-12)
def test_spherical_transport_preserves_tangents():
    """Carry a tangent across a quarter turn and back"""
    from twistformer.geometry.angular import spherical_transport

    source = torch.tensor([0., 0., 1.], dtype=torch.float64)
    target = torch.tensor([1., 0., 0.], dtype=torch.float64)
    tangent = torch.tensor([2., 3., 0.], dtype=torch.float64)
    carried = spherical_transport(source, target) @ tangent
    torch.testing.assert_close(carried, tangent.new_tensor([0., 3., -2.]))
    torch.testing.assert_close(spherical_transport(target, source) @ carried, tangent)
    torch.testing.assert_close(spherical_transport(source, source) @ tangent, tangent)
    assert not spherical_transport(source, -source).any()
