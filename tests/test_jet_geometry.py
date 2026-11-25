# Shared clustering and angular flow observables
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import pytest
import torch

from twistformer.geometry.jet_clustering import JetTree, cluster
from twistformer.geometry.lund_graphs import MassHypothesis, jet_feature_names, jet_rows


def test_shared_jet_rows_and_mass_hypothesis():
    """
    Check individual Torch blocks and a known pair and parent mass

    Returns:
        None:  Feature names and four-body reconstruction agree
    """
    pt = np.array([80., 50., 25., 14.])
    phi = np.array([.2, -.1, .45, -.62])
    eta = np.array([.5, -.11, .08, -.21])
    p = np.column_stack((pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)))
    momenta = np.stack((p, np.zeros_like(p)))
    pair = p[1] + p[3]
    parent = pair + p[2]
    pair_mass = np.sqrt(pair[0]**2 - pair[1:] @ pair[1:])
    parent_mass = np.sqrt(parent[0]**2 - parent[1:] @ parent[1:])
    fit = MassHypothesis(parent_mass, pair_mass, 1., 1.)

    for hypothesis, width in ((None, 796), (fit, 836)):
        names = jet_feature_names(mass_hypothesis=hypothesis)
        rows = jet_rows(momenta, mass_hypothesis=hypothesis)
        assert rows.shape == (2, width) and len(set(names)) == width
        assert not rows[1].any()
        blocks = ("ca", "kt", "flow")
        assert names == sum((jet_feature_names(block, hypothesis) for block in blocks), ())
        torch.testing.assert_close(rows, torch.cat(
            [jet_rows(momenta, block, hypothesis) for block in blocks], dim=1), rtol=0, atol=0)
    for algorithm in ("ca", "kt"):
        for field, expected in (("pair_mass", pair_mass), ("parent_mass", parent_mass)):
            column = names.index(f"lund_{algorithm}_n4_{field}")
            assert rows[0, column] == pytest.approx(expected, rel=1.e-6)


def test_equal_pt_axes_and_collinear_split():
    """Preserve physical zero-angle splittings and the order of equal-pt subjets"""
    p = torch.tensor([[[10., 10., 0., 0.], [10., 10., 0., 0.], [10., -10., 0., 0.],
                       [1., 0., 1., 0.], [1., 0., -1., 0.]]])
    for block in ("ca", "kt"):
        rows = jet_rows(p, block)
        names = jet_feature_names(block)
        for name, expected in (("split4_soft_count", 1.), ("split4_z", .5),
                               ("n5_axis4_dphi", -torch.pi / 2), ("n5_axis5_dphi", torch.pi / 2)):
            assert rows[0, names.index(f"lund_{block}_{name}")].item() == pytest.approx(expected)


def test_clustering_reuses_tree_type_across_batches():
    """Keep the Dynamo input type stable for both jet histories and angular flow"""
    p = torch.zeros(1, 16, 4, dtype=torch.float64)
    p[0, :2] = torch.tensor([[2., 2., 0., 0.], [1., 0., 1., 0.]])
    valid = p[..., 0] > 0
    for kt, history in ((False, True), (True, True), (True, False)):
        assert type(cluster(p, valid, kt, history)) is JetTree
