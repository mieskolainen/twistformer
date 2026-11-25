# Calorimeter streaming and tau acceptance validation
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import pytest

ak = pytest.importorskip("awkward")
uproot = pytest.importorskip("uproot")

from drivers.tau3_classification.reader import (  # noqa: E402
    ECAL_BRANCHES_BY_MODE,
    MC_PARTICLE_VARS,
    MC_RELATION_VARS,
    _fiducial_event_mask_tau_decay_products,
    _pack_hits,
    iter_tau3,
    read_tau3,
)


def _truth(events):
    """
    Build generator records from explicit parent graphs

    Args:
        events:  Per-event PDG, status, eta and parent records
    Returns:
        arrays: Jagged MC branches
    """
    columns = {key: [] for key in MC_PARTICLE_VARS + MC_RELATION_VARS}
    for pdg, status, eta, parents in events:
        begin, end, links = [], [], []
        for indices in parents:
            begin.append(len(links))
            links.extend(indices)
            end.append(len(links))
        values = [
            pdg,
            [1] * len(pdg),
            status,
            [1.0] * len(pdg),
            [0.0] * len(pdg),
            np.sinh(eta).tolist(),
            begin,
            end,
            links,
        ]
        for key, value in zip(columns, values):
            columns[key].append(value)
    return {key: ak.Array(value) for key, value in columns.items()}


def test_tau_ancestry_selection_cases():
    """
    Check visible descendants, unrelated particles and simulation ancestry
    """
    sim = 1 << 30
    events = [
        ([15, 211], [0, 0], [0, 0], [[], [0]]),
        ([15, 211], [0, 0], [0, 2], [[], [0]]),
        ([15, 111, 22], [0, 0, 0], [0, 0, 2], [[], [0], [1]]),
        ([15, 16], [0, 0], [0, 2], [[], [0]]),
        ([15, 211], [0, 0], [0, 2], [[], []]),
        ([15, 211], [0, sim], [0, 2], [[], [0]]),
        ([15, 111, 22], [0, sim, 0], [0, 0, 2], [[], [0], [1]]),
        ([211], [0], [2], [[]]),
        ([], [], [], []),
        ([15, 211], [0, 0], [0, np.nan], [[], [0]]),
        ([15, 111, 22], [0, 0, 0], [0, 0, 2], [[], [], [1, 0]]),
    ]
    actual = _fiducial_event_mask_tau_decay_products(_truth(events), 0.9)
    np.testing.assert_array_equal(
        actual, [True, False, False, True, True, True, True, True, True, False, False]
    )


def test_tau_ancestry_random_graph_reference():
    """
    Compare acceptance with an independent full descendant graph scan
    """
    rng = np.random.default_rng(32)
    events, expected = [], []
    for _ in range(250):
        size = int(rng.integers(1, 30))
        pdg = rng.choice([15, -15, 211, 22, 111, 12, 14, 16], size).tolist()
        status = rng.choice([0, 1 << 30], size, p=[0.8, 0.2]).tolist()
        eta = rng.uniform(-3, 3, size)
        parents = [
            rng.choice(i, min(i, 2), replace=False).tolist() for i in range(size)
        ]
        descendant = [False] * size
        accepted = True
        for i in range(size):
            if status[i]:
                continue
            descendant[i] = abs(pdg[i]) == 15 or any(descendant[p] for p in parents[i])
            if descendant[i] and abs(pdg[i]) not in (12, 14, 16) and abs(eta[i]) >= 0.9:
                accepted = False
        events.append((pdg, status, eta, parents))
        expected.append(accepted)
    np.testing.assert_array_equal(
        _fiducial_event_mask_tau_decay_products(_truth(events), 0.9), expected
    )


def test_hit_packing_stable_order_threshold_and_empty_events():
    """
    Preserve tied-energy order, threshold equality and empty event offsets
    """
    x = ak.Array([[10, 20, 30, 40, 50], [], [60]])
    energy = ak.Array([[0.1, 2.0, 2.0, 0.5, 0.01], [], [0.01]])
    features, offsets, counts = _pack_hits(x, x * 2, x * 3, energy, 2, 0.1)
    np.testing.assert_array_equal(offsets, [0, 2, 2, 2])
    np.testing.assert_array_equal(counts, [4, 0, 0])
    np.testing.assert_array_equal(features, [[20, 40, 60, 2], [30, 60, 90, 2]])
    assert features.dtype == np.float32
    assert features.nbytes == 2 * 4 * 4


def _root_fixture(path, weights=None):
    """
    Write calorimeter events spanning fiducial and empty-hit selections

    Args:
        path:  Output ROOT path
        weights:  Optional signed event yields
    """
    events = [([15, 211], [0, 0], [0, eta], [[], [0]]) for eta in [0, 2, 0, 0, 0, 0]]
    arrays = _truth(events)
    branches = ECAL_BRANCHES_BY_MODE["sim"]
    for coordinate in ("x", "y", "z"):
        arrays[branches[coordinate]] = ak.Array([[i, i + 10] for i in range(6)])
    arrays[branches["e"]] = ak.Array(
        [[1, 2], [1, 2], [0.01, 0.01], [1, 2], [1, 2], [1, 2]]
    )
    arrays["weight"] = (
        np.arange(1, 7, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    with uproot.recreate(path) as root:
        root.mktree("events", arrays)


def test_compact_stream_matches_dense_and_preserves_entries(tmp_path):
    """
    Preserve raw entry IDs and weights across chunks and an incomplete cap
    """
    path = tmp_path / "events.root"
    _root_fixture(path)
    options = dict(max_hits=2048, max_events=5, e_min=0.1, weight_branch="weight")
    chunks = list(iter_tau3(str(path), "events", chunk_size=2, **options))
    np.testing.assert_array_equal(
        np.concatenate([c.entries for c in chunks]), [0, 2, 3, 4]
    )
    np.testing.assert_array_equal(
        np.concatenate([c.weights for c in chunks]), [1, 3, 4, 5]
    )
    compact_bytes = sum(c.features.nbytes for c in chunks)
    features, mask, weights = read_tau3(str(path), "events", chunk_size=3, **options)
    assert features.shape == (4, 2048, 4)
    assert compact_bytes < features.nbytes / 100
    np.testing.assert_array_equal(
        features[mask], np.concatenate([c.features for c in chunks])
    )
    np.testing.assert_array_equal(mask.sum(axis=1), [2, 0, 2, 2])
    np.testing.assert_array_equal(weights, [1, 3, 4, 5])


def test_dataset_chunk_independence_and_sample_ownership(tmp_path):
    """
    Keep splits independent of chunk size and protect shared hit buffers
    """
    from drivers.tau3_classification.driver import Driver
    from twistformer.api.config import TaskConfig

    paths = [tmp_path / f"class{i}.root" for i in range(2)]
    for path in paths:
        _root_fixture(path)
    options = dict(
        files=list(map(str, paths)),
        class_names=["a", "b"],
        min_energy=0.1,
        max_events=5,
        weight_branch="weight",
    )
    small = Driver(dict(options, chunk_size=1), TaskConfig())
    large = Driver(dict(options, chunk_size=4), TaskConfig())
    for split in ("train", "val", "test"):
        first, second = small.dataset(split), large.dataset(split)
        assert [s.id for s in first] == [s.id for s in second]
        assert [s.weight for s in first] == [s.weight for s in second]
        assert all(s.features.shape == (2, 4) for s in first)
    train = small.dataset("train")
    before = train[0].features.clone()
    train[0].features.zero_()
    np.testing.assert_array_equal(train[0].features, before)
    inference = small.dataset("test", str(paths[0]))
    assert [s.id for s in inference] == [f"{paths[0]}:{i}" for i in [0, 3, 4]]
    assert all(s.target is None for s in inference)


@pytest.mark.parametrize("option", ["e_min", "fiducial_eta_max"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_reader_rejects_invalid_cuts(option, value):
    """
    Reject nonphysical selection parameters before ROOT access
    """
    with pytest.raises(ValueError, match=option):
        list(iter_tau3("missing.root", "events", max_hits=2, **{option: value}))


def test_root_reader_preserves_negative_event_weights(tmp_path):
    """
    Preserve signed yields after fiducial selection and across chunk boundaries
    """
    path = tmp_path / "signed.root"
    _root_fixture(path, weights=[-0.5, 2, 3, -1, 5, 6])
    chunks = list(
        iter_tau3(
            str(path),
            "events",
            max_hits=3,
            max_events=5,
            chunk_size=2,
            weight_branch="weight",
        )
    )
    np.testing.assert_array_equal(
        np.concatenate([c.weights for c in chunks]), [-0.5, 3, -1, 5]
    )
