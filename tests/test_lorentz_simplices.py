# Reference contractions for Lorentz hypergraphs and spherical transport
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import numpy as np
import torch

from drivers.toptag.graphs.lund_graphs import GRAM_NAMES, SIMPLEX_NAMES, native_top_rows


def test_lorentz_volume_and_transport():
    """
    Compare exterior volumes and triangular holonomy with independent contractions

    Returns:
        None:  Common boosts and particle permutations preserve the intrinsic scalars
    """
    rng = np.random.default_rng(515)
    xyz = rng.normal(size=(6, 3))*12
    p = np.column_stack((np.sqrt(.2 + np.square(xyz).sum(1)), xyz))
    total = p.sum(0)
    mass = np.sqrt(total[0]**2 - total[1:] @ total[1:])
    energy = (p[:, 0]*total[0] - xyz @ total[1:])/mass
    rest = xyz - ((energy+p[:, 0])/(total[0]+mass))[:, None]*total[1:]
    directions = rest/np.linalg.norm(rest, axis=1)[:, None]
    strongest = np.argmax(energy)
    volume = sum(abs(np.linalg.det(p[list(face)])/mass**4) for face in combinations(range(6), 4) if strongest in face)
    curvature = 0.
    for triangle in combinations(directions, 3):
        angles = []
        for i, v in enumerate(triangle):
            a, b = triangle[(i+1)%3], triangle[(i+2)%3]
            a, b = a-v*(a@v), b-v*(b@v)
            angles.append(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1)))
        curvature += np.sin((sum(angles)-np.pi)/2)**2
    transformed = p.copy()
    transformed[:, 0] = np.cosh(.38)*p[:, 0]+np.sinh(.38)*p[:, 1]
    transformed[:, 1] = np.sinh(.38)*p[:, 0]+np.cosh(.38)*p[:, 1]
    batch = torch.from_numpy(np.stack((p, p[::-1], transformed)).astype('f4'))
    names = ('simplex_volume4_sum_v1', 'simplex_holonomy_count_sum', 'simplex_sheaf_1_energy_max_energy')
    with native_top_rows(names, 2) as project:
        result = project(batch).numpy()
    np.testing.assert_allclose(result[0, :2], [volume, curvature], rtol=2.e-5, atol=1.e-7)
    np.testing.assert_allclose(result[1:], np.repeat(result[:1], 2, axis=0), rtol=2.e-5, atol=2.e-7)


def test_constituent_gram_pair_counts():
    """
    Count edges for each channel and verify empty and low-multiplicity jets

    Returns:
        None:  Each unordered pair contributes exactly once per channel
    """
    rng = np.random.default_rng(400)
    xyz = rng.normal(size=(5, 9, 3))
    p = np.concatenate((np.sqrt(1 + np.square(xyz).sum(-1, keepdims=True)), xyz), axis=-1).astype('f4')
    counts = np.array([0, 1, 3, 7, 9])
    for jet, count in zip(p, counts):
        jet[count:] = 0
    with native_top_rows(GRAM_NAMES, 2) as project:
        result = project(torch.from_numpy(p)).numpy()
        permuted = project(torch.from_numpy(p[:, ::-1].copy())).numpy()
    np.testing.assert_array_equal(result, permuted)
    np.testing.assert_array_equal(result[:, :315].reshape(5, 3, 105).sum(-1),
                                  np.tile((counts*(counts-1)//2)[:, None], (1, 3)))
    assert len(set(SIMPLEX_NAMES)) == len(SIMPLEX_NAMES) == 400


def test_filtered_complex_circle_and_sphere():
    """
    Recover a persistent loop on a square and a cavity on an octahedral shell

    Returns:
        None:  Geometric flag filtrations give the analytic finite intervals
    """
    from drivers.toptag.graphs.lund_graphs import COMPLEX_NAMES

    triad = np.array([[1, 2, 2], [2, 1, -2], [-2, 2, -1]], dtype=np.float32)
    directions = np.concatenate((triad, -triad))
    octahedron = np.column_stack((np.full(6, 3, dtype=np.float32), directions))
    square = np.zeros_like(octahedron)
    square[:4] = octahedron[[0, 1, 3, 4]]
    jets = torch.from_numpy(np.stack((square, octahedron, octahedron[::-1])))
    for partition in ("kt", "rest"):
        for dim, index in ((1, 0), (2, 1)):
            names = tuple(f"complex_{partition}_angle_h{dim}_{stat}" for stat in
                          ("count", "birth", "death", "longest", "betti75"))
            with native_top_rows(names, 2) as project:
                rows = project(jets).numpy()
            np.testing.assert_allclose(rows[index], [1, .5, 1, .5, 1], atol=1.e-6)
            np.testing.assert_allclose(rows[1], rows[2], atol=1.e-6)
    assert len(set(COMPLEX_NAMES)) == len(COMPLEX_NAMES) == 280


def test_cascade_incidence_and_decay_planes():
    """
    Count adjacent splittings and recover orthogonal planes in a four-body decay

    Returns:
        None:  All internal vertices and their connecting edges enter once
    """
    from drivers.toptag.graphs.lund_graphs import CASCADE_NAMES

    # The two narrow pairs point along opposite x axes in the jet rest frame
    spatial = np.array([[20, 1, 0], [20, -1, 0], [-20, 0, 1], [-20, 0, -1]])
    p = np.column_stack((np.linalg.norm(spatial, axis=1), spatial))
    batch = torch.tensor(np.stack((p, p[::-1])), dtype=torch.float32)
    with native_top_rows(CASCADE_NAMES, 2) as project:
        rows = project(batch).numpy()
    np.testing.assert_array_equal(rows[0], rows[1])
    assert rows[0, :90].sum() == 3
    assert rows[0, 90:198].sum() == 2
    assert rows[0, 198:].sum() == 1
    assert rows[0, CASCADE_NAMES.index('cascade_planes_o0_4_0')] == 1


def test_intrinsic_complex_under_boost():
    """
    Check the intrinsic angular partition with more constituents than microjets

    Returns:
        None:  A boost and a permutation preserve the rest-frame complex
    """
    from drivers.toptag.graphs.lund_graphs import COMPLEX_NAMES

    spatial = np.random.default_rng(918).normal(size=(15, 3))
    p = np.column_stack((np.sqrt(.02+np.square(spatial).sum(1)), spatial))
    boosted = p.copy()
    boosted[:, 0] = np.cosh(.45)*p[:, 0] + np.sinh(.45)*p[:, 1]
    boosted[:, 1] = np.sinh(.45)*p[:, 0] + np.cosh(.45)*p[:, 1]
    jets = torch.tensor(np.stack((p, boosted, p[::-1])), dtype=torch.float32)
    names = tuple(name for name in COMPLEX_NAMES if name.startswith("complex_rest_"))
    with native_top_rows(names, 2) as project:
        rows = project(jets).numpy()
    np.testing.assert_allclose(rows[1:], np.broadcast_to(rows[0], rows[1:].shape), atol=1.e-5, rtol=2.e-5)


def test_constituent_connection_operator():
    """
    Compare the incidence operator with rotations between independent tangent frames

    Returns:
        None:  Transport preserves the section norm and the connection energy identity
    """
    from drivers.toptag.graphs.lund_graphs import SHEAF_NAMES

    spatial = np.array([[2., 1., 3.], [-1., 4., 1.], [2., -1., -5.], [-3., -4., 1.]])
    energy = np.linalg.norm(spatial, axis=1)
    p = np.column_stack((energy, spatial))
    order = np.argsort(-energy)
    directions = (spatial/energy[:, None])[order]
    energy = energy[order]/energy.sum()
    cosine = np.clip(directions @ directions.T, -1., 1.)
    weights = (.25/(.25+1.-cosine))**2
    weights /= weights.sum(axis=1, keepdims=True)
    edge = energy[:, None]*weights
    # Axes and constituents coincide for a four-body rest-frame event
    section = edge.T @ directions
    section -= np.sum(section*directions, axis=1)[:, None]*directions
    section /= edge.sum(axis=0)[:, None]

    def rotate(v, a, b):
        """Rotate a tangent vector along the geodesic from a to b"""
        cross = np.cross(a, b)
        return v + np.cross(cross, v) + np.cross(cross, np.cross(cross, v))/(1.+a@b)

    particle = np.stack([sum(weights[i,j]*rotate(section[j], directions[j], directions[i])
                             for j in range(4)) for i in range(4)])
    returned = np.stack([sum(edge[i,j]*rotate(particle[i], directions[i], directions[j])
                             for i in range(4))/edge[:,j].sum() for j in range(4)])
    batch = torch.tensor(np.stack((p, p[::-1])), dtype=torch.float32)
    with native_top_rows(SHEAF_NAMES, 2) as project:
        computed = project(batch).numpy()
    np.testing.assert_allclose(computed[0], computed[1], atol=1.e-6, rtol=1.e-5)
    row = dict(zip(SHEAF_NAMES, computed[0], strict=True))
    np.testing.assert_allclose([row[f"sheaf_wide_v{i+1}_return_norm2"] for i in range(4)],
                               np.sum(returned**2, axis=1), rtol=2.e-5, atol=1.e-8)
    np.testing.assert_allclose(row["sheaf_wide_mean_edge_residual"], row["sheaf_wide_mean_laplacian_product"],
                               atol=1.e-9, rtol=2.e-5)


def test_lab_reference_contractions():
    """
    Compare time and beam-plane contractions with independent tensor formulas

    Returns:
        None:  Rotations about the beam preserve the pooled and rooted contractions
    """
    from twistformer.geometry.lorentz_hypergraphs import REFERENCE_NAMES
    from twistformer.geometry.native_lund import native_lund_rows

    p4 = torch.tensor([[6., 5., 0., 1.], [5., 0., 4., 2.], [4., -3., 0., -1.], [3., 0., -2., 1.]])
    rotated = p4.clone()
    rotated[:, 1] = .6*p4[:, 1] - .8*p4[:, 2]
    rotated[:, 2] = .8*p4[:, 1] + .6*p4[:, 2]
    boosted = p4.clone()
    boosted[:, 0] = 1.25*p4[:, 0] + .75*p4[:, 3]
    boosted[:, 3] = .75*p4[:, 0] + 1.25*p4[:, 3]
    jets = torch.stack((p4, rotated, p4.flip(0), boosted, torch.zeros_like(p4)))
    with native_lund_rows(8192, None, 2) as project:
        rows = project(jets)
    assert rows.shape == (len(jets), len(REFERENCE_NAMES))
    assert torch.count_nonzero(rows[-1]) == 0
    torch.testing.assert_close(rows[1:3], rows[0].expand(2, -1))
    v = p4[:, 1:]/p4[:, :1]
    expected = {
        "reference_time_energy": p4[:, 0].sum()/20,
        "reference_time_concentration": (p4[:, 0]/p4[:, 0].sum()).square().sum(),
        "reference_pair12_beam_area2": torch.linalg.det(v[:2, :2]).square(),
        "reference_pair12_time_wedge_norm2": (v[0]-v[1]).square().sum(),
        "reference_triple123_beam_area2": torch.linalg.det(v[1:3, :2]-v[0, :2]).square(),
        "reference_triple123_time_volume2": torch.linalg.det(v[:3]).square(),
    }
    for name, value in expected.items():
        torch.testing.assert_close(rows[0, REFERENCE_NAMES.index(name)], value)
    torch.testing.assert_close(rows[3, 0], boosted[:, 0].sum()/20)
