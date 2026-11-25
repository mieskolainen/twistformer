# Cluster PID geometry, ROOT input and training checks
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import re
import pytest
import torch
from drivers.pi0gamma.driver import Driver, Options, Pi0GammaDataset
from drivers.pi0gamma.features import Geometry, cartesian_features, cell_features, angular_features
from drivers.pi0gamma.reader import ClusterChunk, event_buffers, iter_clusters, truth_label
from twistformer import load_config, build_experiment, Trainer, load_predictor
from twistformer.api.config import load_yaml_mapping, parse_config
from twistformer.data.batching import collate_samples
from twistformer.models.transformer import neighbor_bias
from twistformer.training.checkpoint import read_checkpoint


@pytest.mark.parametrize("labeled", [False, True])
def test_batched_cluster_fetch(labeled):
    """
    Preserve shuffled cluster records and isolate repeated event buffers

    Args:
        labeled:  Attach classification labels when true
    Returns:
        None: Batched samples match compact source records
    """
    blocks = []
    for block in range(2):
        chunk = ClusterChunk(
            features=np.arange(48, dtype=np.float32).reshape(6, 8) + block * 100,
            offsets=np.array([0, 1, 4, 6]),
            entries=np.array([10, 20, 30]),
            targets=np.array([0, 1, 0]) if labeled else None,
            weights=np.array([0.5, 2.0, 3.0]),
            observables={"cluster_energy": np.array([5.0, 10.0, 20.0]) + block},
        )
        blocks.append((f"sample_{block}.root", chunk))
    dataset = Pi0GammaDataset(blocks, [(1, 2), (0, 1), (1, 0), (0, 2)])
    order = [2, 1, 0, 1, 3]
    samples = dataset.__getitems__(order)
    assert dataset.__getitems__([]) == []
    for index, sample in zip(order, samples, strict=True):
        block, event = dataset.indices[index]
        path, chunk = blocks[block]
        begin, end = chunk.offsets[event:event + 2]
        np.testing.assert_array_equal(sample.features.numpy(), chunk.features[begin:end])
        torch.testing.assert_close(dataset[index].features, sample.features)
        assert sample.id == f"{path}:{chunk.entries[event]}"
        assert sample.weight == chunk.weights[event]
        assert sample.observables == {"cluster_energy": chunk.observables["cluster_energy"][event]}
        if labeled:
            assert sample.target.dtype == torch.long
            assert sample.target.item() == chunk.targets[event]
        else:
            assert sample.target is None
    expected = samples[3].features.clone()
    samples[1].features.fill_(-1)
    torch.testing.assert_close(samples[3].features, expected)
    torch.testing.assert_close(dataset[1].features, expected)
    if labeled:
        samples[1].target.fill_(0)
        assert samples[3].target.item() == dataset[1].target.item() == 1


@pytest.fixture
def geometry():
    """
    Define a small test readout with strip and merged layers

    Returns:
        geometry: Synthetic detector constants
    """
    return Geometry(1000.0, 0.01, 0.02, np.pi / 2, 4, 3, [1, 2], [2, 1])


def write_root(path, truth=True, events=20, flat=False, ntuple=False, energy=10.0, metadata=False, cell_ids=True,
               input_mode="leading_cluster", event_weights=None, extra_cone_cells=False, decays=True, photon_angles=None):
    """
    Write two clusters with interleaved cell relations per event

    Args:
        path:  Output fixture path
        truth:  Include primary particle labels
        events:  Number of events
        flat:  Use unslashed PODIO branch names
        ntuple:  Write an RNTuple instead of a TTree
        energy:  Primary energy in GeV
        metadata:  Write PODIO collection IDs and cluster relation destinations
        cell_ids:  Include readout identifiers for geometry features
        input_mode:  Write cluster relations or positioned cells without clusters
        event_weights:  Optional generator weights in event order
        extra_cone_cells:  Add two unlinked barrel cells across the cone boundary
        decays:  Alternate balanced and asymmetric direct two-photon decays
        photon_angles:  Primary and daughter polar angles per event in degrees
    Returns:
        path: ROOT fixture path
    """
    ak = pytest.importorskip("awkward")
    uproot = pytest.importorskip("uproot")
    c, h, m = "AugmentedEMBCaloClusters", "EMBCaloClusterCells", "MCParticles"
    if input_mode == "all_cells":
        h = "ECalBarrelModuleThetaMergedPositioned"
    branches = {}
    if event_weights is not None:
        branches["event_weight"] = ak.Array(event_weights)

    def add(collection, field, values):
        """
        Add an event-jagged PODIO field

        Args:
            collection:  Collection name
            field:  Member name
            values:  Values repeated across events
        """
        key = f"{collection}.{field}" if flat else f"{collection}/{collection}.{field}"
        branches[key] = ak.Array([values] * events)

    if input_mode == "leading_cluster":
        add(c, "energy", [2.0, 10.0])
        add(c, "position.x", [1010., 1010.])
        add(c, "position.y", [0., 0.])
        add(c, "position.z", [500., 0.])
        add(c, "hits_begin", [0, 1])
        add(c, "hits_end", [1, 5])
        add(f"_{c}_hits", "index", [1, 4, 0, 3, 2])
        if metadata:
            add(f"_{c}_hits", "collectionID", [4059670135] * 5)
    add(h, "energy", [1.0, 100.0, -1.0, 0.0, 3.0])
    if cell_ids:
        add(h, "cellID", [0, 0, 16, 16, 16])
    add(h, "position.x", [1010.0] * 5)
    add(h, "position.y", [0.0, 0.0, 0.0, 0.0, 20.0])
    add(h, "position.z", [0.0, 500.0, 0.0, 0.0, 10.0])
    if input_mode == "leading_cluster":
        barrel = "ECalBarrelModuleThetaMergedPositioned"
        for field, values, extra in (
            ("energy", [1., 100., -1., 0., 3.], [5., 9.]),
            ("position.x", [1010.] * 5, [1010., 1010.]),
            ("position.y", [0., 0., 0., 0., 20.], [1010. * np.tan(.29), 1010. * np.tan(.31)]),
            ("position.z", [0., 500., 0., 0., 10.], [0., 0.]),
            ("cellID", [0, 0, 16, 16, 16], [0, 0]),
        ):
            if field != "cellID" or cell_ids:
                add(barrel, field, values + (extra if extra_cone_cells else []))
    if truth:
        add(m, "PDG", [22])
        key = f"{m}.PDG" if flat else f"{m}/{m}.PDG"
        branches[key] = ak.Array([[22 if i % 2 else 111] for i in range(events)])
        add(m, "generatorStatus", [2])
        add(m, "parents_begin", [0])
        add(m, "parents_end", [0])
        add(m, "mass", [0.135])
        add(m, "momentum.x", [np.sqrt(energy ** 2 - 0.135 ** 2)])
        add(m, "momentum.y", [0.0])
        add(m, "momentum.z", [0.0])
        if decays:
            branches[key] = ak.Array([[22 if i % 2 else 111, 22, 22] for i in range(events)])
            for name, values in (("generatorStatus", [2, 0, 0]), ("parents_begin", [0, 0, 1]),
                                 ("parents_end", [0, 1, 2]), ("mass", [.135, 0., 0.]),
                                 ("momentum.y", [0., 0., 0.]), ("momentum.z", [0., 0., 0.]),
                                 ("daughters_begin", [0, 2, 2]), ("daughters_end", [2, 2, 2])):
                add(m, name, values)
            momentum_key = key.removesuffix("PDG") + "momentum.x"
            branches[momentum_key] = ak.Array([[np.sqrt(energy**2 - .135**2), 1., 3. if i % 4 == 2 else 1.] for i in range(events)])
            if photon_angles is not None:
                momentum = ak.to_numpy(branches[momentum_key])
                angles = np.radians(photon_angles)
                branches[momentum_key] = ak.Array(momentum * np.sin(angles))
                branches[key.removesuffix("PDG") + "momentum.z"] = ak.Array(momentum * np.cos(angles))
            add(f"_{m}_daughters", "index", [2, 1])
            if metadata:
                add(f"_{m}_daughters", "collectionID", [2714477136] * 2)
    with uproot.recreate(path) as root:
        if ntuple:
            root.mkrntuple("events", branches)
        else:
            root.mktree("events", {key: values.type.content for key, values in branches.items()})
            root["events"].extend(branches)
        if metadata:
            prefix = "events___CollectionTypeInfo"
            root.mkrntuple("podio_metadata", {
                f"{prefix}/{prefix}.name": ak.Array([[h, m]]),
                f"{prefix}/{prefix}.collectionID": ak.Array([[4059670135, 2714477136]]),
            })
    return str(path)


def test_cell_features_geometry_and_phi_boundary(geometry):
    """
    Check local angles, cylindrical depth, readout sizes and energy fractions

    Args:
        geometry:  Synthetic readout fixture
    """
    theta = np.array([np.pi / 2 + 0.01, np.pi / 2 + 0.02])
    phi = np.array([np.pi - 0.01, -np.pi + 0.01])
    rho = np.array([1010.0, 1020.0])
    positions = np.column_stack((rho * np.cos(phi), rho * np.sin(phi), rho / np.tan(theta)))
    x, obs = cell_features(positions, [1.0, 3.0], [0, 16], geometry)
    np.testing.assert_allclose(x[:, 0], [-0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(x[:, 2], [0.2, 0.4])
    np.testing.assert_allclose(x[:, 3], np.log([1.0, 3.0]))
    np.testing.assert_allclose(x[:, 4:], [[1, 0.25, 0.2, 0.8], [0, 0.75, 0.4, 0.4]])
    assert np.max(np.abs(x[:, 1])) < 0.4
    assert obs == {"cluster_energy": 4.0, "n_cells": 2.0}
    rotated = positions.copy()
    angle = 1.2
    rotated[:, 0] = positions[:, 0] * np.cos(angle) - positions[:, 1] * np.sin(angle)
    rotated[:, 1] = positions[:, 0] * np.sin(angle) + positions[:, 1] * np.cos(angle)
    np.testing.assert_allclose(cell_features(rotated, [1, 3], [0, 16], geometry)[0], x, atol=1e-6)
    with pytest.raises(ValueError, match="readout"):
        Geometry().validate()
    with pytest.raises(ValueError, match="layer"):
        cell_features(positions, [1, 3], [0, 112], geometry)


@pytest.mark.parametrize("theta_centre", [0.8, 1.6, 2.3])
@pytest.mark.parametrize("draw", [0.0, 0.4999, 0.5, 0.9999])
@pytest.mark.parametrize("mode", ["extended", "angular", "cartesian"])
def test_theta_augmentation_matches_reflected_geometry(geometry, monkeypatch, theta_centre, draw, mode):
    """
    Compare feature augmentation with a reflection of detector cell positions

    Args:
        geometry:  Readout constants with fine and merged layers
        monkeypatch:  Select either branch of the reflection draw
        theta_centre:  Cluster direction in radians
        draw:  Uniform draw with reflection below one half
        mode:  Angular, Cartesian or extended features
    """
    from twistformer.api.contracts import Sample

    theta = theta_centre + np.array([-0.013, 0.004, 0.017])
    phi = np.array([np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02])
    rho = np.array([1010.0, 1020.0, 1040.0])
    energy, ids = np.array([1.0, 3.0, 2.0]), np.array([0, 16, 0])
    positions = np.column_stack((rho * np.cos(phi), rho * np.sin(phi), rho / np.tan(theta)))
    local_transform = cartesian_features if mode == "cartesian" else angular_features
    features, obs = (cell_features(positions, energy, ids, geometry) if mode == "extended"
                     else local_transform(positions, energy, geometry))
    original = features.copy()
    sample = Sample(torch.from_numpy(features), target=torch.tensor(1), weight=2.5,
                    id="cluster", observables={**obs, "truth_theta": 61.0})
    cfg = load_config("configs/pi0gamma/transformer_extended.yml")
    cfg.driver["feature_mode"] = mode
    driver = build_experiment(cfg).driver
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(draw))
    actual = driver.augment(sample)

    barycentre = np.average(theta, weights=energy)
    centre = geometry.theta_offset_rad + geometry.theta_pitch_rad * np.floor(
        (barycentre - geometry.theta_offset_rad) / geometry.theta_pitch_rad + 0.5
    ) if mode == "extended" else barycentre
    reflected_positions = positions.copy()
    if draw < 0.5:
        if mode == "cartesian":
            reflected_positions[:, 2] = 2 * np.average(positions[:, 2], weights=energy) - positions[:, 2]
        else:
            reflected_positions[:, 2] = rho / np.tan(2 * centre - theta)
    expected, reflected_obs = (cell_features(reflected_positions, energy, ids, geometry) if mode == "extended"
                               else local_transform(reflected_positions, energy, geometry))
    np.testing.assert_allclose(actual.features.numpy(), expected, atol=1e-6)
    np.testing.assert_array_equal(actual.features[:, 1:].numpy(), original[:, 1:])
    np.testing.assert_array_equal(sample.features.numpy(), original)
    assert reflected_obs == obs
    assert actual.target is sample.target and actual.observables is sample.observables
    assert actual.id == sample.id and actual.weight == sample.weight
    # Two reflections restore the event, with no change to the source buffers
    torch.testing.assert_close(driver.augment(actual).features, sample.features, rtol=0, atol=0)


def test_jagged_cell_id_buffers():
    """Preserve empty showers and unsigned cell IDs in a sliced ROOT chunk"""
    ak = pytest.importorskip("awkward")
    ids = np.arange(7, dtype=np.uint64) + np.uint64(1 << 63)
    column = ak.unflatten(ids, [2, 1, 0, 3, 1])[1:-1]
    values, offsets = event_buffers(column)
    np.testing.assert_array_equal(offsets, [0, 1, 1, 4])
    np.testing.assert_array_equal(values, ids[2:6])
    assert values.dtype == ids.dtype


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("ntuple", [False, True])
def test_root_relations_streaming_and_truth_free_prediction(tmp_path, geometry, flat, ntuple):
    """
    Check leading-cluster relations, chunk limits and truth-free inputs

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout fixture
        flat:  PODIO branch name layout
        ntuple:  ROOT storage format
    """
    path = write_root(tmp_path / "labeled.root", flat=flat, ntuple=ntuple)
    options = Options(input_mode="leading_cluster", files=[path], max_events=7, chunk_size=3, geometry=asdict(geometry))
    chunks = list(iter_clusters(path, options, geometry))
    assert [len(chunk) for chunk in chunks] == [3, 3, 1]
    assert np.concatenate([chunk.entries for chunk in chunks]).tolist() == list(range(7))
    assert np.concatenate([chunk.targets for chunk in chunks]).tolist() == [0, 1, 0, 1, 0, 1, 0]
    assert chunks[0].offsets.tolist() == [0, 2, 4, 6]
    np.testing.assert_allclose(chunks[0].features[:2, 5], [0.75, 0.25])
    assert chunks[0].observables["cluster_energy"].tolist() == [4.0] * 3
    detector_path = write_root(tmp_path / "detector.root", truth=False, flat=flat, ntuple=ntuple)
    unlabeled = list(iter_clusters(detector_path, options, geometry, labeled=False))
    np.testing.assert_array_equal(chunks[0].features, unlabeled[0].features)
    assert unlabeled[0].targets is None
    assert set(unlabeled[0].observables) == {"cluster_energy", "n_cells"}


def test_truth_links_resolve_primary_ancestors():
    """
    Select the pi0 primary through a linked secondary photon
    """
    event = {
        "pdg": np.array([22, 111, 22]), "status": np.array([1, 2, 1]),
        "parents_begin": np.array([0, 0, 0]), "parents_end": np.array([0, 0, 1]),
        "parent_index": np.array([1]), "mass": np.array([0, 0.135, 0]),
        "mc_x": np.array([1, 2, 3]), "mc_y": np.zeros(3), "mc_z": np.zeros(3),
        "link_cell": np.array([4, 1]), "link_particle": np.array([2, 0]), "link_weight": np.array([1.0, 100.0]),
    }
    assert truth_label(event, np.array([4]), [111, 22], True)[0] == 0
    with pytest.raises(ValueError, match="Multiple PID primaries"):
        truth_label(event, np.array([4]), [111, 22], False)


@pytest.mark.parametrize("labeled", [False, True])
def test_podio_collection_destinations(tmp_path, geometry, monkeypatch, labeled):
    """
    Check relation destinations even when indices fit the wrong collection

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout fixture
        monkeypatch:  Substitute the configured cell collection ID
        labeled:  Enable MC branches
    """
    from drivers.pi0gamma import reader

    path = write_root(tmp_path / "metadata.root", metadata=True)
    options = Options(input_mode="leading_cluster")
    assert sum(map(len, iter_clusters(path, options, geometry, labeled=labeled))) == 20
    monkeypatch.setattr(reader, "collection_ids", lambda *args: {"EMBCaloClusterCells": 7})
    with pytest.raises(ValueError, match="must reference EMBCaloClusterCells"):
        list(iter_clusters(path, options, geometry, labeled=labeled))


def test_truth_relation_collection_validation():
    """
    Reject wrong MC and cell relation destinations before using their indices
    """
    from drivers.pi0gamma.reader import validate_relation_collection

    for field, index, collection in (
        ("parent_collection", "parent_index", "MCParticles"),
        ("link_cell_collection", "link_cell", "EMBCaloClusterCells"),
        ("link_particle_collection", "link_particle", "MCParticles"),
    ):
        event = {field: np.array([9]), index: np.array([0])}
        validate_relation_collection(event, field, collection, {collection: 9})
        with pytest.raises(ValueError, match="must reference"):
            validate_relation_collection(event, field, collection, {collection: 8})
        event[field] = np.array([9, 9])
        with pytest.raises(ValueError, match="must align"):
            validate_relation_collection(event, field, collection, {collection: 9})


@pytest.mark.parametrize("cuts, selected", [
    ({"min_energy": 1.0}, [4, 0]),
    ({"min_energy": 3.0}, [4]),
    ({"min_energy": 3.01}, []),
    ({"max_cells": 1}, [4]),
    ({"min_cells": 3}, []),
])
def test_cell_cuts_and_truth_free_selection(tmp_path, geometry, monkeypatch, cuts, selected):
    """
    Apply the same cell cuts to features, truth links and detector-only prediction

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout fixture
        monkeypatch:  Capture cells passed to truth association
        cuts:  Cell selection options
        selected:  Expected retained ROOT cell indices
    """
    from drivers.pi0gamma import reader

    labeled = write_root(tmp_path / "labeled.root", events=2)
    detector = write_root(tmp_path / "detector.root", truth=False, events=2)
    options = Options(**cuts)
    associated = []
    original = reader.truth_label

    def label(event, cells, *args):
        """
        Record retained cell indices before primary association

        Args:
            event:  MC and detector arrays
            cells:  Selected detector indices
            args:  Label settings
        Returns:
            result: Primary label and truth observables
        """
        associated.append(cells.tolist())
        return original(event, cells, *args)

    monkeypatch.setattr(reader, "truth_label", label)
    chunks = list(iter_clusters(labeled, options, geometry))
    prediction = list(iter_clusters(detector, options, geometry, labeled=False))
    if not selected:
        assert chunks == prediction == associated == []
        return
    assert associated == [selected, selected]
    assert chunks[0].entries.tolist() == [0, 1]
    np.testing.assert_array_equal(chunks[0].features, prediction[0].features)
    positions = np.array([[1010, 0, 0], [1010, 0, 500], [1010, 0, 0], [1010, 0, 0], [1010, 20, 10]])
    energy = np.array([1, 100, -1, 0, 3])
    ids = np.array([0, 0, 16, 16, 16])
    expected, obs = cell_features(positions[selected], energy[selected], ids[selected], geometry)
    np.testing.assert_allclose(chunks[0].features[:len(selected)], expected)
    assert chunks[0].observables["cluster_energy"].tolist() == [obs["cluster_energy"]] * 2
    assert chunks[0].observables["n_cells"].tolist() == [len(selected)] * 2


@pytest.mark.parametrize("cuts", [
    {"min_energy": -1}, {"min_energy": float("nan")},
    {"min_cells": 0}, {"max_cells": 1.5}, {"max_cells": True},
    {"min_cells": 3, "max_cells": 2}, {"eta_max": 0}, {"eta_max": float("inf")},
])
def test_invalid_cell_cuts(cuts):
    """
    Reject invalid cell selection settings before reading ROOT files

    Args:
        cuts:  Invalid selection options
    """
    cfg = load_config("configs/pi0gamma/transformer_extended.yml")
    cfg.driver.update(cuts)
    with pytest.raises(ValueError, match="min_energy|min_cells|max_cells|eta_max"):
        build_experiment(cfg)


def test_neighbors_padding_small_graphs_and_reflection():
    """
    Check eight nonself neighbours and isolated-cell support
    """
    points = torch.zeros(2, 12, 3)
    points[0, :, 0] = torch.arange(12)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 1:] = False
    bias = neighbor_bias(points, mask, 8)
    assert torch.isfinite(bias[0, 0]).sum(-1).tolist() == [8] * 12
    assert not torch.isfinite(bias[0, 0].diagonal()).any()
    assert torch.isfinite(bias[1, 0, 0, 0])
    assert not torch.isfinite(bias[1, 0, 0, 1:]).any()
    points[:, :, 0] *= -1
    torch.testing.assert_close(neighbor_bias(points, mask, 8), bias)
    assert torch.isfinite(neighbor_bias(points[:, :1], mask[:, :1], 8)).all()
    with pytest.raises(ValueError, match="normalize_features"):
        parse_config({"driver": {"name": "synthetic"}, "model": {"attention_neighbors": 8}})


def test_neighbor_mask_ties_ignore_padding():
    """
    Keep the first equidistant cell when padding changes the sequence length
    """
    points = torch.zeros(1, 3, 3)
    points[0, :, 0] = torch.arange(3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    reference = neighbor_bias(points, mask, 1)
    assert reference[0, 0, 1].isfinite().tolist() == [True, False, False]
    indices = [0, 2, 4]
    padded = torch.full((1, 7, 3), 25.0)
    padded[:, indices] = points
    padded_mask = torch.zeros(1, 7, dtype=torch.bool)
    padded_mask[:, indices] = True
    bias = neighbor_bias(padded, padded_mask, 1)
    torch.testing.assert_close(bias[:, :, indices][:, :, :, indices], reference)
    assert not bias[:, :, ~padded_mask[0], :].isfinite().any()
    assert not bias[:, :, :, ~padded_mask[0]].isfinite().any()


def test_full_attention_model_padded_gradients():
    """
    Check finite gradients for full attention with variable cell counts
    """
    from twistformer.api.contracts import Sample

    cfg = load_config("configs/pi0gamma/transformer_extended.yml")
    cfg.model = replace(cfg.model, width=16, depth=1, heads=2, embedding=[16], checkpoint_blocks=False, head_zero_init=False)
    model = build_experiment(cfg).model
    samples = [Sample(torch.randn(size, 8), global_features=torch.randn(2)) for size in (1, 4, 12)]
    batch = collate_samples(samples)
    batch.features.requires_grad_()
    batch.global_features.requires_grad_()
    output = model(batch).logits
    output.square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(batch.features.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert not batch.features.grad[~batch.mask].count_nonzero()
    assert torch.isfinite(batch.global_features.grad).all() and batch.global_features.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["extended", "angular", "cartesian"])
def test_driver_splits_augmentation_and_training(tmp_path, geometry, mode):
    """
    Train and restore cluster PID with deterministic disjoint splits

    Args:
        tmp_path:  Input and checkpoint directory
        geometry:  Synthetic readout fixture
        mode:  Input representation
    """
    path = write_root(tmp_path / "events.root", events=40)
    card = f"transformer_{mode}"
    cfg = load_config(f"configs/pi0gamma/{card}.yml")
    cfg.driver.update(base_path=str(tmp_path), files=["events.root"], geometry=asdict(geometry), chunk_size=7,
                      energy_edges=[0, 120], energy_weights=None, balance_energy=True, max_cells=1,
                      split=[0.7, 0.15, 0.15], root_workers=1)
    cfg.training = replace(
        cfg.training, device="cpu", workers=0, epochs=1, warmup_epochs=0, batch_size=8, eval_batch_size=8,
        accumulation=1, amp=False, tf32=False, compile=False,
    )
    cfg.model = replace(cfg.model, width=16, heads=2, depth=1, embedding=[16], checkpoint_blocks=False)
    exp = build_experiment(cfg)
    driver = exp.driver
    assert driver.comparison_identity("train")["sources"] == [path]
    datasets = [driver.dataset(split) for split in ("train", "val", "test")]
    assert [len(dataset) for dataset in datasets] == [28, 6, 6]
    ids = [{sample.id for sample in dataset} for dataset in datasets]
    assert len(set.union(*ids)) == 40
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    other = Driver(
        {key: value for key, value in cfg.driver.items() if key != "name"}, cfg.task
    )
    assert {sample.id for sample in other.dataset("test")} == ids[2]
    views = driver.augment_dataset(datasets[0])
    assert len(views) == 2 * len(datasets[0])
    order = [*reversed(range(len(views))), 1, 0]
    for index, view in zip(order, views.__getitems__(order), strict=True):
        source = datasets[0][index // 2]
        expected = source.features.clone()
        if index % 2:
            expected[:, 0] *= -1
        torch.testing.assert_close(view.features, expected)
        torch.testing.assert_close(view.global_features, source.global_features)
        torch.testing.assert_close(views[index].features, expected)
        assert view.id == source.id and view.weight == source.weight
        assert view.target == source.target
        assert view.observables.keys() == source.observables.keys()
        np.testing.assert_equal(list(view.observables.values()), list(source.observables.values()))
    best = Trainer(tmp_path / "run").fit(exp)
    assert "Training after augmentation: 56 examples from 28 events" in (tmp_path / "run/logs/train.log").read_text()
    assert read_checkpoint(best)["step"] == 7
    predictor = load_predictor(best, "cpu")
    assert predictor.experiment.driver.options.feature_mode == mode
    assert predictor.experiment.driver.options.input_mode == "leading_cluster_cone"
    assert len(predictor.experiment.driver.input_spec.features) == (8 if mode == "extended" else 4)
    assert predictor.preprocessor.state["class_bin_weights"].shape == (2, 1)
    restored = predictor.experiment.driver.dataset("test")
    assert [sample.id for sample in restored] == [sample.id for sample in datasets[2]]
    from twistformer.evaluation.runner import evaluate

    result = evaluate(predictor, restored, tmp_path / "evaluation", plots=False)
    assert set(result["classes"]) == {"gamma", "pi0"}
    assert result["signal_operating_points"] == result["classes"]["gamma"]["operating_points"]
    with np.load(tmp_path / "evaluation/predictions.npz") as saved:
        np.testing.assert_array_equal(saved["weights"], [sample.weight for sample in datasets[2]])
    detector = write_root(tmp_path / "detector.root", truth=False, events=2)
    samples = predictor.experiment.driver.prediction_dataset(detector)
    batch = collate_samples([predictor.preprocessor.transform(samples[0])])
    with torch.no_grad():
        prediction = predictor.experiment.model(batch)
    assert torch.isfinite(prediction.logits).all()
    assert samples[0].p4 is None and samples[0].target is None
    expected_totals = [2, 4]
    np.testing.assert_allclose(samples[0].global_features, np.log(expected_totals))
    torch.testing.assert_close(samples[0].global_features, datasets[0][0].global_features)
    assert samples[0].observables["n_cells"] == 1


def test_all_energy_splits_weights_and_roc_bins(tmp_path, geometry, capsys):
    """
    Check the gun spectrum for both classes in every split and the evaluation bins

    Args:
        tmp_path:  ROOT fixtures and category output directory
        geometry:  Synthetic readout fixture
        capsys:  Captured spectrum normalization message
    """
    from twistformer.evaluation.categories import evaluate_categories

    energies = [5, 10, 20, 35, 50, 80, 100]
    for index, energy in enumerate(energies):
        events = 30 + 4 * index
        write_root(tmp_path / f"gun_{energy}.root", energy=energy, events=events, event_weights=1 + np.arange(events) % 5)
    cfg = load_config("configs/pi0gamma/transformer_extended.yml")
    cfg.driver.update(base_path=str(tmp_path), files=["gun_*.root"], geometry=asdict(geometry),
                      input_mode="leading_cluster", weight_branch="event_weight",
                      energy_weights=[26.1, 26.1, 24.3, 14.2, 8., 1.3, 0.], root_workers=1)
    driver = build_experiment(cfg).driver
    splits = [list(driver.dataset(name)) for name in ("train", "val", "test")]
    ids = [{sample.id for sample in samples} for samples in splits]
    assert sum(map(len, ids)) == len(set.union(*ids)) == sum(30 + 4 * i for i in range(7))
    assert {sample.id for sample in driver.comparison_dataset(driver, "test")} == ids[2]
    groups = {(label, energy) for label in (0, 1) for energy in energies}
    for samples in splits:
        assert {(int(sample.target), round(sample.observables["truth_energy"])) for sample in samples} == groups
    train = splits[0]
    prep = driver.create_preprocessor(config=cfg.preprocessing).fit(train, class_count=2)
    prepared = prep.transform_many(train)
    fractions = np.array(cfg.driver["energy_weights"]) / 100.
    np.testing.assert_allclose(driver.energy_fractions, fractions)
    output = capsys.readouterr().out
    assert "Normalized driver.energy_weights" in output and "(sum = 1)" in output
    for samples in [*splits, prepared]:
        for label in (0, 1):
            selected = [sample for sample in samples if int(sample.target) == label]
            yields, _ = np.histogram([sample.observables["truth_energy"] for sample in selected],
                                     bins=cfg.driver["energy_edges"], weights=[sample.weight for sample in selected])
            np.testing.assert_allclose(yields / yields.sum(), fractions, atol=1e-7)
            if samples is not prepared:
                assert yields.sum() == pytest.approx(sum(1 + int(sample.id.rsplit(":", 1)[1]) % 5 for sample in selected))
    test = driver.dataset("test")
    assert [sample.weight for sample in test.__getitems__(list(range(len(test))))] == [sample.weight for sample in splits[2]]
    single_energy = driver.dataset("test", str(tmp_path / "gun_50.root"))
    np.testing.assert_allclose([sample.weight for sample in single_energy], 1 + np.arange(len(single_energy)) % 5)
    restored = driver.create_preprocessor(prep.state, config=cfg.preprocessing)
    state = {key: value.clone() for key, value in prep.state.items()}
    for raw, transformed in zip(train, prepared):
        assert restored.transform(raw).weight == pytest.approx(transformed.weight)
        torch.testing.assert_close(transformed.features, raw.features)
        assert transformed.parameters is None
        torch.testing.assert_close(transformed.global_features, restored.transform(raw).global_features)
    for raw, transformed in zip(splits[1], restored.transform_many(splits[1])):
        assert transformed.weight == pytest.approx(prep.transform(raw).weight)
    for key, value in state.items():
        torch.testing.assert_close(restored.state[key], value)
    detector = replace(train[0], target=None, observables={})
    assert restored.transform(detector).weight == detector.weight
    assert restored.transform_many([detector])[0].weight == detector.weight

    samples = splits[2]
    labels = np.array([int(sample.target) for sample in samples])
    score = 0.1 + 0.8 * labels
    data = {
        "targets": labels, "weights": np.array([sample.weight for sample in samples]),
        "probabilities": np.column_stack([1 - score, score]),
        "observable_truth_energy": np.array([sample.observables["truth_energy"] for sample in samples]),
    }
    category = next(item for item in cfg.evaluation.categories if item["name"] == "truth_energy")
    assert category["edges"] == cfg.driver["energy_edges"]
    manifest = evaluate_categories(data, driver.task_spec, [category], driver.evaluation_observables, tmp_path / "roc", plots=False, bootstrap_samples=0)
    group = manifest["categories"][0]
    assert group["missing_events"] == group["out_of_range_events"] == 0
    assert len(group["bins"]) == 7
    assert [item["status"] for item in group["bins"]] == ["completed"] * 6 + ["unavailable"]


def test_evaluation_energy_override(tmp_path):
    """
    Restore a zero-weight gun bin in fresh and saved PID evaluations

    Args:
        tmp_path:  Evaluation steering and output directory
    """
    import json
    from twistformer.api.contracts import Prediction, Sample
    from twistformer.api.predictor import Predictor
    from twistformer.cli.evaluate import evaluation_override, main
    from twistformer.evaluation.runner import evaluate

    cfg = load_config("configs/pi0gamma/transformer_extended.yml")
    cfg.driver.update(energy_edges=[0, 50, 120], energy_weights=[1, 0], global_features=[])
    cfg.training = replace(cfg.training, workers=0, amp=False)
    cfg.model = replace(cfg.model, width=16, heads=2, depth=1, embedding=[16])
    category = {"name": "energy", "observable": "truth_energy", "edges": [0, 50, 120]}
    cfg.evaluation = replace(cfg.evaluation, bootstrap_samples=0, categories=[category])
    exp = build_experiment(cfg)
    samples = [
        Sample(torch.full((1, 8), energy / 100), target=torch.tensor(label), id=f"{label}-{energy}", weight=4 if energy == 5 else 0,
               observables={"truth_energy": energy, "generator_weight": 1 if energy == 5 else 3})
        for label in (0, 1) for energy in (5, 100)
    ]
    prep = exp.driver.create_preprocessor(config=cfg.preprocessing).fit(samples, class_count=2)
    predictor = Predictor(exp, prep, "cpu")
    predictor.forward = lambda batch: Prediction(logits=batch.features[:, 0, :1])
    source = tmp_path / "source"
    original = evaluate(predictor, samples, source, plots=False)
    (source / "evaluation.json").write_text(json.dumps({"status": "completed"}))
    card = tmp_path / "evaluation.yml"
    card.write_text("driver:\n  energy_weights: [1, 3]\n")
    override = evaluation_override(card, cfg.evaluation)
    fresh = tmp_path / "fresh"
    metrics = evaluate(predictor, samples, fresh, plots=False, evaluation_config=override)
    assert metrics["cross_entropy"] > original["cross_entropy"]
    rebinned = tmp_path / "rebinned"
    main(["--from-evaluation", str(source), "--evaluation-config", str(card), "--output", str(rebinned), "--no-plots"])
    assert json.loads((rebinned / "metrics.json").read_text()) == metrics
    for output in (fresh, rebinned):
        bins = json.loads((output / "categories.json").read_text())["categories"][0]["bins"]
        assert [entry["support"]["sum_weights"] for entry in bins] == [2, 6]
        with np.load(output / "predictions.npz") as saved:
            np.testing.assert_array_equal(saved["weights"], [1, 3, 1, 3])
            np.testing.assert_array_equal(saved["physical_weights"], [4, 0, 4, 0])
    assert cfg.driver["energy_weights"] == [1, 0]
    card.write_text("evaluation:\n  driver:\n    energy_weights: null\n")
    reset = tmp_path / "reset"
    main(["--from-evaluation", str(rebinned), "--evaluation-config", str(card), "--output", str(reset), "--no-plots"])
    with np.load(reset / "predictions.npz") as saved:
        np.testing.assert_array_equal(saved["weights"], [4, 0, 4, 0])


def test_energy_balance_rejects_missing_support():
    """
    Reject absent class yields, invalid energies and changed checkpoint bins
    """
    from twistformer.api.contracts import Sample
    from twistformer.data.preprocessing import Preprocessor

    samples = [Sample(torch.ones(2, 8), target=torch.tensor(label), observables={"truth_energy": energy}) for label in (0, 1) for energy in (5, 10)]
    prep = Preprocessor().configure_class_bins("truth_energy", [0, 7.5, 15])
    with pytest.raises(ValueError, match="positive training weight"):
        prep.fit(samples[:-1], class_count=2)
    prep.fit(samples, class_count=2)
    for energy in (-1, 15, float("nan")):
        with pytest.raises(ValueError, match="finite truth_energy"):
            prep.transform(replace(samples[0], observables={"truth_energy": energy}))
    with pytest.raises(ValueError, match="match checkpoint"):
        Preprocessor(prep.state).configure_class_bins("truth_energy", [0, 8, 15])


def test_angular_features_and_validation():
    """
    Centre local angles across the phi boundary using only the ECAL inner radius
    """
    geometry = Geometry(inner_radius_mm=1000)
    theta = np.array([1.50, 1.52, 1.55])
    phi = np.array([np.pi - .01, -np.pi + .01, 0])
    rho = np.array([1010, 1020, 1030])
    positions = np.column_stack((rho * np.cos(phi), rho * np.sin(phi), rho / np.tan(theta)))
    features, obs = angular_features(positions, [1, 3, -1], geometry)
    np.testing.assert_allclose(features[:, 0], [-.3, .1], atol=1e-6)
    np.testing.assert_allclose(features[:, 2], [.2, .4], atol=1e-6)
    np.testing.assert_allclose(features[:, 3], np.log([1, 3]))
    assert np.max(np.abs(features[:, 1])) < .4
    assert abs(np.average(features[:, 0], weights=[1, 3])) < 1e-7
    rotated = positions.copy()
    rotated[:, :2] *= -1
    np.testing.assert_allclose(angular_features(rotated, [1, 3, -1], geometry)[0], features, atol=1e-6)
    assert obs == {"cluster_energy": 4.0, "n_cells": 2.0}
    assert features.dtype == np.float32
    empty, obs = angular_features(positions, [0, -1, 0], geometry)
    assert empty.shape == (0, 4) and obs == {}
    with pytest.raises(ValueError, match="align"):
        angular_features(positions, [1], geometry)
    with pytest.raises(ValueError, match="finite"):
        angular_features(positions, [1, np.nan, 1], geometry)
    with pytest.raises(ValueError, match="finite"):
        angular_features([[np.inf, 0, 1]], [1], geometry)
    with pytest.raises(ValueError, match="inner_radius_mm"):
        angular_features(positions, [1, 3, -1], Geometry())


@pytest.mark.parametrize("angle", [0.0, 1.2, np.pi - 0.001, -np.pi + 0.001])
def test_cartesian_features_preserve_distances_and_detector_frame(angle):
    """
    Preserve detector distances under a rigid local frame across the phi boundary

    Args:
        angle:  Common rotation about the beam axis in radians
    """
    geometry = Geometry(inner_radius_mm=1000)
    position = np.array([[1010, -30, -100], [1030, 10, 20], [1050, 0, 200]], dtype=np.float64)
    energy = np.array([1, 3, 2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    position[:, :2] = position[:, :2] @ rotation.T
    position[:, 2] += 1800
    features, observables = cartesian_features(position, energy, geometry)
    np.testing.assert_allclose(features[:, :3], [[3.2, -0.6, 0.2], [0.8, 0.2, 0.6], [-2.8, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(features[:, 3], np.log(energy), atol=1e-6)
    offsets = position[:, None] - position[None, :]
    local_offsets = features[:, None, :3] - features[None, :, :3]
    np.testing.assert_allclose(np.linalg.norm(local_offsets, axis=-1) * 50, np.linalg.norm(offsets, axis=-1), atol=2e-5)
    np.testing.assert_allclose(np.average(features[:, :2], weights=energy, axis=0), 0, atol=1e-7)
    assert observables == {"cluster_energy": 6.0, "n_cells": 3.0}
    assert features.dtype == np.float32
    shifted, _ = cartesian_features(position, energy, Geometry(inner_radius_mm=900))
    np.testing.assert_allclose(shifted[:, :3], features[:, :3] + [0, 0, 2], atol=1e-6)
    order = [2, 0, 1]
    permuted, _ = cartesian_features(position[order], energy[order], geometry)
    np.testing.assert_allclose(permuted, features[order], atol=1e-6)


def test_cartesian_features_selection_and_validation():
    """
    Retain physical radial depth and reject undefined Cartesian barrel frames
    """
    geometry = Geometry(inner_radius_mm=1000)
    features, observables = cartesian_features([[606, 808, 300], [np.nan, 0, 0]], [2, -1], geometry)
    np.testing.assert_allclose(features, [[0, 0, 0.2, np.log(2)]], atol=1e-6)
    assert observables == {"cluster_energy": 2.0, "n_cells": 1.0}
    empty, observables = cartesian_features([[1010, 0, 0]], [0], geometry)
    assert empty.shape == (0, 4) and observables == {}
    with pytest.raises(ValueError, match="align"):
        cartesian_features([[1010, 0, 0]], [1, 2], geometry)
    with pytest.raises(ValueError, match="finite"):
        cartesian_features([[1010, 0, 0]], [np.nan], geometry)
    with pytest.raises(ValueError, match="finite"):
        cartesian_features([[np.inf, 0, 0]], [1], geometry)
    with pytest.raises(ValueError, match="cylindrical radius"):
        cartesian_features([[0, 0, 1000]], [1], geometry)
    with pytest.raises(ValueError, match="barycentre"):
        cartesian_features([[1010, 0, 0], [-1010, 0, 0]], [1, 1], geometry)
    with pytest.raises(ValueError, match="inner_radius_mm"):
        cartesian_features([[1010, 0, 0]], [1], Geometry())


@pytest.mark.parametrize("mode", ["angular", "cartesian"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("ntuple", [False, True])
def test_local_reader_without_segmentation_or_cell_ids(tmp_path, flat, ntuple, mode):
    """
    Check relation ordering and truth-free reading with no readout identifiers
    """
    labeled = write_root(tmp_path / "labeled.root", events=2, flat=flat, ntuple=ntuple, cell_ids=False)
    detector = write_root(tmp_path / "detector.root", events=2, truth=False, flat=flat, ntuple=ntuple, cell_ids=False)
    options = Options(feature_mode=mode)
    geometry = Geometry(inner_radius_mm=1000)
    chunks = list(iter_clusters(labeled, options, geometry))
    prediction = list(iter_clusters(detector, options, geometry, labeled=False))
    local_transform = cartesian_features if mode == "cartesian" else angular_features
    expected, _ = local_transform([[1010, 20, 10], [1010, 0, 0]], [3, 1], geometry)
    np.testing.assert_array_equal(chunks[0].features[:2], expected)
    np.testing.assert_array_equal(chunks[0].features, prediction[0].features)
    assert chunks[0].targets.tolist() == [0, 1] and prediction[0].targets is None
    with pytest.raises(ValueError, match="cell_id"):
        list(iter_clusters(labeled, Options(), Geometry(1000, .01, .02, 1, 4, 3, [1, 2], [2, 1])))


@pytest.mark.parametrize("mode", ["angular", "cartesian"])
def test_feature_modes_preserve_event_selection(tmp_path, geometry, mode):
    """
    Compare feature representations on identical cells, labels and event splits
    """
    path = write_root(tmp_path / "events.root", events=40)
    cfg = load_config("configs/pi0gamma/transformer_extended.yml")
    cfg.driver.update(files=[path], geometry=asdict(geometry), energy_edges=[0, 120], energy_weights=[1.0], min_energy=2,
                      input_mode="leading_cluster")
    geometric = build_experiment(cfg).driver
    cfg.driver.update(feature_mode=mode, geometry={"inner_radius_mm": geometry.inner_radius_mm})
    local = build_experiment(cfg).driver
    assert local.geometry.inner_radius_mm == geometry.inner_radius_mm
    assert local.input_spec.units == ("50 mm", "50 mm", "50 mm", "ln(E/GeV)")
    for split in ("train", "val", "test"):
        left, right = geometric.dataset(split), local.dataset(split)
        comparison = local.comparison_dataset(geometric, split)
        assert len(left) == len(right) == len(comparison)
        for a, b, c in zip(left, right, comparison, strict=True):
            assert a.id == b.id == c.id and a.target == b.target == c.target
            assert a.weight == b.weight and a.observables.keys() == b.observables.keys()
            np.testing.assert_equal(list(a.observables.values()), list(b.observables.values()))
            assert a.features.shape == (1, 8) and b.features.shape == (1, 4)
            torch.testing.assert_close(b.features, c.features)
    cfg.driver["feature_mode"] = "invalid"
    with pytest.raises(ValueError, match="feature_mode"):
        build_experiment(cfg)


@pytest.mark.parametrize("mode", ["extended", "angular", "cartesian"])
@pytest.mark.parametrize("flat, ntuple", [(False, False), (True, False), (False, True), (True, True)])
def test_reject_events_without_cluster_links(tmp_path, geometry, mode, flat, ntuple):
    """
    Require reconstructed cluster links in labeled and detector-only files

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout constants
        mode:  Cell feature representation
        flat:  Flat PODIO branch names
        ntuple:  RNTuple storage instead of TTree
    """
    settings = dict(events=7, flat=flat, ntuple=ntuple, input_mode="all_cells", cell_ids=mode == "extended",
                    metadata=True)
    labeled = write_root(tmp_path / "labeled.root", **settings)
    detector = write_root(tmp_path / "detector.root", truth=False, **settings)
    options = Options(feature_mode=mode, chunk_size=3)
    for file, truth in ((labeled, True), (detector, False)):
        with pytest.raises(ValueError, match="Missing ROOT branch"):
            list(iter_clusters(file, options, geometry, labeled=truth))
        with pytest.raises(ValueError, match="leading_cluster"):
            list(iter_clusters(file, replace(options, input_mode="all_cells"), geometry, labeled=truth))


@pytest.mark.parametrize("mode", ["angular", "cartesian"])
@pytest.mark.parametrize("cuts, selected", [
    ({}, [4, 0]),
    ({"min_energy": 3}, [4]),
    ({"max_cells": 2}, [4, 0]),
    ({"max_cells": 10}, [4, 0]),
    ({"max_cells": 1}, [4]),
    ({"min_cells": 4}, []),
    ({"min_energy": 101}, []),
])
def test_cluster_cell_cuts_sort_cap_and_label_indices(tmp_path, geometry, monkeypatch, cuts, selected, mode):
    """
    Apply cuts before energy sorting and the cap, preserving truth link indices

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout constants
        monkeypatch:  Capture indices passed to label association
        cuts:  Cell selection settings
        selected:  Expected collection indices after cuts, sorting and cap
        mode:  Angular or Cartesian local features
    """
    from drivers.pi0gamma import reader

    path = write_root(tmp_path / "cells.root", events=2)
    options = Options(feature_mode=mode, **cuts)
    associated = []
    original = reader.truth_label

    def label(event, cells, *args):
        """
        Record source indices before MC label selection

        Args:
            event:  Detector and MC arrays
            cells:  Retained collection indices
            args:  Label settings
        Returns:
            result: Primary label and truth observables
        """
        associated.append(cells.tolist())
        return original(event, cells, *args)

    monkeypatch.setattr(reader, "truth_label", label)
    chunks = list(iter_clusters(path, options, geometry))
    if not selected:
        assert chunks == associated == []
        return
    assert associated == [selected, selected]
    expected = np.array([1, 100, -1, 0, 3])[selected]
    np.testing.assert_allclose(np.exp(chunks[0].features[:len(selected), 3]), expected, rtol=1e-6)
    assert abs(np.average(chunks[0].features[:len(selected), 0], weights=expected)) < 1e-6


@pytest.mark.parametrize("names", [[], ["log_n_cells"], ["log_sum_energy"], ["log_sum_energy", "log_n_cells"]])
def test_global_feature_steering(tmp_path, geometry, names):
    """
    Select shower scalars after cell cuts and before truncation

    Args:
        tmp_path:  Detector ROOT file directory
        geometry:  Synthetic readout fixture
        names:  Enabled global features in model input order
    """
    file = write_root(tmp_path / "cells.root", events=2, truth=False)
    cfg = load_config("configs/pi0gamma/transformer_extended.yml")
    cfg.driver.update(global_features=names, geometry=asdict(geometry), max_cells=1)
    driver = build_experiment(cfg).driver
    assert driver.input_spec.global_features == tuple(names)
    dataset = driver.prediction_dataset(file)
    values = {"log_n_cells": np.log(2), "log_sum_energy": np.log(4)}
    for sample in [dataset[0], *dataset.__getitems__([1, 0, 1])]:
        assert sample.observables == {"cluster_energy": 3, "n_cells": 1}
        if names:
            np.testing.assert_allclose(sample.global_features, [values[name] for name in names])
        else:
            assert sample.global_features is None


@pytest.mark.parametrize("card", sorted(p for p in Path("configs/pi0gamma").glob("*.yml")
                                        if p.stem.startswith(("twistgraph", "transformer", "gatr"))))
def test_all_pi0gamma_model_cards_select_leading_cluster_cone(card):
    """
    Verify the cluster source of each training and inference card

    Args:
        card:  Pi0gamma steering card
    """
    common = load_yaml_mapping("configs/pi0gamma/common.yml")["driver"]
    cfg = load_config(card)
    driver = build_experiment(cfg).driver
    assert driver.options.input_mode == "leading_cluster_cone"
    assert driver.options.cone_half_angle == 0.30
    assert driver.options.photon_theta_range_deg == common["photon_theta_range_deg"]
    assert "eta_max" not in cfg.driver
    assert driver.options.cell_collection == "ECalBarrelModuleThetaMergedPositioned"
    assert driver.options.max_cells == common["max_cells"]
    if card.stem == "twistgraph":
        assert driver.input_spec.global_features == ("log_sum_energy",)
        assert len(driver.input_spec.features) == 1999
        assert driver.options.shower_basis == "twistgraph"
        assert driver.geometry is None
        assert driver.options.base_path == common["base_path"]
        assert (cfg.training.monitor, cfg.training.mode) == ("val_auc", "max")
    else:
        assert driver.input_spec.global_features == ("log_n_cells", "log_sum_energy")
    assert driver.task_spec.outputs == ("pi0", "gamma")
    assert driver.task_spec.signal_class == 1
    assert driver.options.class_pdgs == [111, 22]
    cfg.driver["input_mode"] = "all_cells"
    with pytest.raises(ValueError, match="input_mode"):
        build_experiment(cfg)


def test_legacy_driver_class_order():
    """
    Restore the output columns of checkpoints with implicit PID labels
    """
    from drivers.base import build_driver

    cfg = load_config("configs/pi0gamma/transformer_angular.yml")
    options = {key: value for key, value in cfg.driver.items() if key not in {"class_names", "class_pdgs", "signal_class"}}
    options.pop("global_features")
    options["name"] = "gammapi0"
    legacy = build_driver(options, cfg.task)
    assert legacy.input_spec.global_features == ()
    spec = legacy.task_spec
    assert spec.outputs == ("gamma", "pi0") and spec.signal_class == 1
    options.update(class_names=["pi0", "gamma"], class_pdgs=[111, 22])
    spec = build_driver(options, cfg.task).task_spec
    assert spec.outputs == ("pi0", "gamma") and spec.signal_class == 0


@pytest.mark.parametrize("device", ["cpu", "cuda", "auto"])
def test_shower_booster_checkpoint(tmp_path, device, monkeypatch):
    """
    Restore a weighted tree ensemble and score detector-only hits

    Args:
        tmp_path:  ROOT fixture and training output directory
        device:  Requested XGBoost training and inference device
        monkeypatch:  Inspect native XGBoost training parameters
    """
    import json

    xgb = pytest.importorskip("xgboost")
    from twistboard.events import active_events
    from twistboard.outputs import read_history
    from twistboard.server import run_detail, training_distribution
    cuda = torch.cuda.is_available() and xgb.build_info().get("USE_CUDA", False)
    if device == "cuda" and not cuda:
        pytest.skip("CUDA XGBoost is unavailable")
    file = write_root(tmp_path / "train.root", events=30, cell_ids=False,
                      event_weights=[2, 1] * 15)
    detector = write_root(tmp_path / "hits.root", events=3, cell_ids=False, truth=False)
    overrides = ["training.epochs=3", "training.patience=0", "training.augmentation=dataset",
                 "training.seed=43", f"training.device={device}"]
    scheduled = device != "auto"
    if scheduled:
        overrides.extend(["training.epochs=5", "training.max_steps=3", "training.scheduler=cosine", "training.warmup_epochs=2"])
    else:
        overrides.extend(["training.scheduler=none", "training.warmup_epochs=0"])
    config = load_config("configs/pi0gamma/twistgraph.yml", overrides)
    config.driver.update(files=[file], base_path="", energy_edges=[0, 120], energy_weights=[1.0],
                         weight_branch="event_weight", feature_mode="cartesian", root_workers=1)
    if device == "auto":
        config.driver["shower_groups"] = ["shower_radial_sampling_rank", "shower_radial_spectral_depth"]
        config.driver["shower_columns"] = None
    config.task.class_weights = [0.5, 2]
    experiment = build_experiment(config)
    train_booster = xgb.train

    def check_seed(parameters, *args, **kwargs):
        """
        Verify the run seed at the native XGBoost training call

        Args:
            parameters:  Native XGBoost training parameters
            args:  Positional training inputs
            kwargs:  Training options
        Returns:
            booster: Trained XGBoost ensemble
        """
        assert parameters["seed"] == 43
        return train_booster(parameters, *args, **kwargs)

    monkeypatch.setattr(xgb, "train", check_seed)
    checkpoint = Trainer(tmp_path / "run").fit(experiment)
    output = checkpoint.parent.parent
    saved_model = load_yaml_mapping(output / "config.yml")["model"]
    assert set(saved_model) == {"family", "depth", "xgboost_threads", "xgboost_sketch_rows",
                                "xgboost_sketch_seed", "xgboost_params"}
    assert saved_model["depth"] == config.model.depth
    assert saved_model["xgboost_params"] == config.model.xgboost_params
    assert run_detail(tmp_path, "run")["configuration"]["model"] == saved_model
    assert read_checkpoint(checkpoint)["config"]["model"] == saved_model
    for name in ("best", "last"):
        state = read_checkpoint(checkpoint.with_name(f"{name}.pt"))
        file = checkpoint.with_name(f"{name}_feature_importance.json")
        result = json.loads(file.read_text())
        assert result == json.loads(state["feature_importance_json"])
        assert result["checkpoint"] == f"{name}.pt" and result["boosting_rounds"] == state["step"]
        assert result["importance_type"] == "total_gain"
        features = result["features"]
        spec = experiment.driver.input_spec
        assert {entry["feature"] for entry in features} == set(spec.features + spec.global_features)
        assert [entry["rank"] for entry in features] == list(range(1, len(features) + 1))
        scores = [entry["score"] for entry in features]
        assert scores == sorted(scores, reverse=True)
    pdf = (output / "plots/best_feature_importance.pdf").read_bytes()
    assert pdf.startswith(b"%PDF-")
    assert len(re.findall(rb"/Type\s*/Page\b", pdf)) > 1
    history = read_history(output)
    distributions = json.loads((output / "plots/training_distributions.json").read_text())
    spec = experiment.driver.input_spec
    assert distributions["features"] == list(spec.features + spec.global_features)
    assert distributions["classes"] == ["pi0", "gamma"]
    assert distributions["category_groups"] == [{"name": "truth_energy", "observable": "truth_energy", "edges": [0, 120]}]
    histogram = training_distribution(tmp_path, "run", len(spec.features))
    assert len(histogram["edges"]) == 49
    np.testing.assert_array_equal(histogram["weights"][0], histogram["weights"][1])
    assert sum(distributions["events"][0]) == len(experiment.driver.dataset("train"))
    events = list(active_events(output / "events.jsonl"))
    assert len(history) == len(events) == 3
    rate = config.training.learning_rate
    np.testing.assert_allclose([row["lr"] for row in history], [rate / 2 if scheduled else rate, rate, rate])
    assert all(np.isclose(row["val_accuracy"], 0.5) for row in history)
    for row, event in zip(history, events, strict=True):
        assert set(event["values"]) == {"train_loss", "val_loss", "val_accuracy", "val_auc", "lr"}
        assert event["values"] == {name: row[name] for name in event["values"]}
        assert event["epoch"] == row["epoch"] + 1 and event["step"] == row["step"]
    assert read_checkpoint(checkpoint.with_name("last.pt"))["step"] == len(history)
    train = experiment.driver.dataset("train")
    views = experiment.driver.augment_dataset(train)
    assert len(views) == 2 * len(train)
    assert json.loads((output / "manifest.json").read_text())["train_events"] == len(views)
    for view in views.__getitems__([0, 1]):
        torch.testing.assert_close(view.features, train[0].features)
        assert view.id == train[0].id and view.weight == train[0].weight
    predictor = load_predictor(checkpoint, device)
    expected_device = "cuda" if device == "cuda" or (device == "auto" and cuda) else "cpu"
    assert predictor.device.type == expected_device
    trained = json.loads(experiment.model.booster.save_config())["learner"]["generic_param"]["device"]
    assert trained.split(":")[0] == expected_device
    assert predictor.model.booster.num_boosted_rounds() == read_checkpoint(checkpoint)["step"]
    spec = experiment.driver.input_spec
    assert spec.global_features == ("log_sum_energy",)
    assert predictor.experiment.driver.options.shower_groups == config.driver["shower_groups"]
    assert predictor.model.booster.feature_names == list(spec.features + spec.global_features)
    predictor.experiment.driver.options.weight_branch = None
    detector_data = predictor.experiment.driver.prediction_dataset(detector)
    assert detector_data[0].target is None
    batch = next(iter(predictor.loader(detector_data)))
    matrix = xgb.DMatrix(experiment.model.inputs(batch).numpy(), feature_names=predictor.model.booster.feature_names)
    np.testing.assert_allclose(predictor.predict_prob(detector_data)[:, 1], experiment.model.booster.predict(matrix))
    restored = json.loads(predictor.model.booster.save_config())["learner"]["generic_param"]["device"]
    assert restored.split(":")[0] == expected_device
    if device == "cpu":
        from twistformer.cli.calibrate import main as calibrate

        calibrated_file = tmp_path / "calibrated.pt"
        args = ["--checkpoint", str(checkpoint), "--output", str(calibrated_file), "--device", "cpu"]
        with pytest.raises(SystemExit):
            calibrate(args)
        assert not calibrated_file.exists()
        calibrate(args + ["--config", "configs/pi0gamma/temperature_calibration.yml"])
        calibrated = load_predictor(calibrated_file, "cpu")
        record = read_checkpoint(calibrated_file)["calibration"]
        assert record["split"] == "val" and record["events"] == len(experiment.driver.dataset("val"))
        assert record["cross_entropy_after"] <= record["cross_entropy_before"]
        assert calibrated.model.temperature.item() == pytest.approx(record["temperature"])
        assert "calibration" not in read_checkpoint(checkpoint)
        torch.testing.assert_close(calibrated.predict(detector_data).logits,
                                   predictor.predict(detector_data).logits / record["temperature"])


def test_complete_energy_flow_graphs():
    """
    Verify all connected and disconnected graphs against explicit hit assignments
    """
    from itertools import product
    from twistformer.geometry.graph_polynomials import GRAPH_COUNTS, GRAPH_NAMES, graph_catalog, spherical_graphs

    connected, graphs = graph_catalog()
    assert len(connected) == 156
    degrees = [sum(len(connected[k]) for k in graph) for graph in graphs]
    assert tuple(np.bincount(degrees)[1:]) == GRAPH_COUNTS
    assert len(set(GRAPH_NAMES)) == len(graphs) == 313
    assert len(set(connected)) == 156 and len(set(graphs)) == 313
    # Include the six-edge complete graph on four distinct vertices
    assert ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)) in connected
    points = np.array([[0., .7], [.2, -.4], [-.5, .1]])
    weights = np.array([.15, .55, .30])
    denominator = 1 + (points**2).sum(axis=1)
    kernel = ((points[:, None] - points)**2).sum(axis=2) / np.outer(denominator, denominator)
    values = []
    for edges in connected:
        n = 1 + max(max(edge) for edge in edges)
        labels = np.array(list(product(range(len(weights)), repeat=n)))
        terms = weights[labels].prod(axis=1)
        for i, j in edges:
            terms *= kernel[labels[:, i], labels[:, j]]
        values.append(terms.sum())
    explicit = [np.prod(np.take(values, graph)) for graph in graphs]
    np.testing.assert_allclose(spherical_graphs(points, weights), explicit, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("multiclass", [False, True])
def test_temperature_fit_weighted_logits(multiclass):
    """
    Recover the known temperature of weighted binary and multiclass logits
    """
    from twistformer.api.config import CalibrationConfig
    from twistformer.evaluation.temperature import fit_temperature

    y = np.array([0, 1, 2] if multiclass else [0, 1])
    logits = np.array([[0., 2., -1.]] if multiclass else [[2.]])
    expected = np.exp(logits.ravel()/2) if multiclass else np.exp(np.array([0., 1.]))
    result = fit_temperature(np.repeat(logits, len(y), axis=0), y, expected, CalibrationConfig())
    assert result["temperature"] == pytest.approx(2.)
    assert result["cross_entropy_after"] < result["cross_entropy_before"]


def test_cell_cone_before_cap_and_without_truth(tmp_path, geometry):
    """
    Keep barrel cells in the seed cone before calculating shower totals

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout constants
    """
    file = write_root(tmp_path / "cone.root", truth=False, events=1)
    opts = Options(input_mode="leading_cluster", files=[file], feature_mode="angular", max_cells=1,
                   global_features=["log_n_cells", "log_sum_energy"])
    full = next(iter_clusters(file, opts, geometry, labeled=False))
    np.testing.assert_allclose(full.global_features[0], np.log([2, 4]))
    opts.max_cell_angle = .01
    selected = next(iter_clusters(file, opts, geometry, labeled=False))
    np.testing.assert_allclose(selected.global_features[0], np.log([1, 3]))
    opts.max_cell_angle = np.pi
    restored = next(iter_clusters(file, opts, geometry, labeled=False))
    np.testing.assert_array_equal(restored.global_features, full.global_features)


@pytest.mark.parametrize("angle, expected", [(0.28, [3, 1]), (0.30, [5, 3, 1]), (0.32, [9, 5, 3, 1])])
def test_leading_cluster_cone_includes_unlinked_cells(tmp_path, geometry, angle, expected):
    """
    Select positioned cells around the leading cluster before building any features

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic detector geometry
        angle:  Three-dimensional half-angle in radians
        expected:  Passing cell energies in descending order
    """
    options = Options(feature_mode="angular", cone_half_angle=angle,
                      global_features=["log_n_cells", "log_sum_energy"])
    for labeled in (False, True):
        file = write_root(tmp_path / f"cone_{labeled}.root", events=1, truth=labeled, extra_cone_cells=True)
        chunk = next(iter_clusters(file, options, geometry, labeled=labeled))
        np.testing.assert_allclose(np.exp(chunk.features[:, 3]), expected)
        np.testing.assert_allclose(chunk.global_features[0], np.log([len(expected), sum(expected)]))
        assert chunk.observables["cluster_energy"][0] == sum(expected)


def test_cone_across_azimuth_wrap_and_empty_cluster():
    """
    Verify the opening angle across the phi branch cut and the empty seed case
    """
    from drivers.pi0gamma.reader import leading_cells

    phi = np.array([-np.pi + .02, np.pi - .28, np.pi - .34])
    event = {"cluster_energy": np.array([10.]), "cluster_x": np.cos([np.pi-.01]),
             "cluster_y": np.sin([np.pi-.01]), "cluster_z": np.zeros(1),
             "cell_x": np.cos(phi), "cell_y": np.sin(phi), "cell_z": np.zeros(3)}
    np.testing.assert_array_equal(leading_cells(event, Options(), {}), [0, 1])
    event["cluster_energy"] = np.array([])
    assert not len(leading_cells(event, Options(), {}))


def test_independent_gun_energy_splits(tmp_path, geometry):
    """
    Match the standalone 50 GeV event IDs in the mixed spectrum

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout constants
    """
    for energy in (5, 50):
        write_root(tmp_path / f"gun_{energy}.root", energy=energy)
    cfg = load_config("configs/pi0gamma/twistgraph.yml")
    cfg.driver.update(files=["gun_*.root"], base_path=str(tmp_path), geometry=asdict(geometry),
                      input_mode="leading_cluster", energy_edges=[0, 30, 65], energy_weights=[.5, .5])
    mixed = build_experiment(cfg).driver
    cfg.driver.update(files=["gun_50.root"])
    single = build_experiment(cfg).driver
    cfg.driver.update(energy_edges=[30, 65], energy_weights=[1.])
    single_bin = build_experiment(cfg).driver
    for split in ("train", "val", "test"):
        selected = [s.id for s in mixed.dataset(split) if round(s.observables["truth_energy"]) == 50]
        samples = single.dataset(split)
        assert selected == [s.id for s in samples] == [s.id for s in single_bin.dataset(split)]
        np.testing.assert_allclose(samples.weights, single_bin.dataset(split).weights)


def test_model_cards_share_event_splits(tmp_path, geometry):
    """
    Match all split IDs across model cards, training seeds and ROOT worker counts

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic readout constants
    """
    from twistformer.randomness import seed_all

    for energy in (5, 50):
        write_root(tmp_path / f"gun_{energy}.root", energy=energy)

    def split_ids(card, training_seed, workers, *, driver_seed=None):
        """
        Read all splits with the reader settings from one model card

        Args:
            card:  Model card name
            training_seed:  Global run seed
            workers:  ROOT reader processes
            driver_seed:  Optional event split seed override
        Returns:
            ids:  Ordered event IDs in each split
        """
        config = load_config(f"configs/pi0gamma/{card}.yml", [f"training.seed={training_seed}"])
        options = {name: value for name, value in config.driver.items() if name != "name"}
        options.update(base_path=str(tmp_path), files=["gun_*.root"], energy_edges=[0, 30, 65],
                       energy_weights=[0.5, 0.5], chunk_size=8, root_workers=workers)
        if not options.get("shower_basis"):
            options["geometry"] = asdict(geometry)
        else:
            options.update(shower_groups=["shower_radial_sampling_rank"], shower_columns=None)
        if driver_seed is not None:
            options["seed"] = driver_seed
        seed_all(config.training.seed)
        driver = Driver(options, config.task)
        return [[sample.id for sample in driver.dataset(split)] for split in ("train", "val", "test")]

    expected = split_ids("transformer_extended", 11, 1)
    assert len(set().union(*expected)) == sum(map(len, expected)) == 40
    assert split_ids("gatr_projective_cartesian", 23, 2) == expected
    assert split_ids("twistgraph", 31, 1) == expected
    assert split_ids("transformer_extended", 11, 1, driver_seed=102) != expected


def test_harmonic_graphs():
    """
    Check finite edge and triangle contractions, reflections and empty layers
    """
    from drivers.pi0gamma.graphs.descriptors import BASES, descriptor_rows

    rng = np.random.default_rng(101)
    hits = rng.normal(size=(60, 4))
    hits[:, 2] = rng.uniform(0, 5, len(hits))
    names = BASES["harmonic_graphs"]
    result = descriptor_rows([hits], "harmonic_graphs")[0]
    assert torch.isfinite(result).all() and len(result) == len(set(names))
    reflected = hits[::-1].copy()
    reflected[:, 0] *= -1
    np.testing.assert_allclose(result, descriptor_rows([reflected], "harmonic_graphs")[0], atol=1e-6)
    singleton = descriptor_rows([torch.tensor([[0., 0., 0., 1.]])], "harmonic_graphs")[0]
    single = dict(zip(names, singleton, strict=True))
    assert single["energy_edge_L0_k0"] == 1
    assert single["energy_triangle_re_L0_k0"] == 1
    assert single["energy_adjacent_re_L0_k0"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA harmonic contractions")
def test_harmonic_graph_device_parity():
    """
    Preserve the CPU readout phase, tied beam orientation and sparse layer values
    """
    from drivers.pi0gamma.graphs.descriptors import descriptor_rows

    rng = np.random.default_rng(54)
    symmetric = np.array([[-.5333, .18, .36, -.5], [.5333, .18, .36, -.5]])
    hits = rng.normal(size=(530, 4))
    hits[:, 2] = rng.uniform(0, 8, len(hits))
    showers = [symmetric, hits, hits[:1], symmetric[::-1]]
    rows = descriptor_rows(showers, "harmonic_graphs", "cuda")
    assert rows.is_cuda
    torch.testing.assert_close(rows.cpu(), descriptor_rows(showers, "harmonic_graphs"), atol=1e-6, rtol=1e-6)


def test_energy_scaled_profile_graph_card():
    """
    Verify the triplet determinant, peak ties and feature order of the production card
    """
    from drivers.pi0gamma.graphs.descriptors import BASES, descriptor_rows

    hits = np.array([[.12, .31, .5, .1], [.82, -.23, .5, -1.2], [-.62, -.1, .5, -.52]])
    reflected = hits[::-1].copy()
    reflected[:, 0] *= -1
    tied = np.array([[-1., 0., .5, 0.], [1., 0., .5, 0.]])
    basis = "energy_scaled_profile_graphs"
    rows = descriptor_rows([hits, reflected, tied, tied[::-1]], basis, inner_radius_mm=2172.8)
    names = BASES[basis]
    assert rows.shape == (4, len(set(names))) and torch.isfinite(rows).all()
    np.testing.assert_allclose(rows[0], rows[1], atol=1e-6)
    np.testing.assert_allclose(rows[2], rows[3], atol=1e-6)
    energy = np.exp(hits[:, 3])
    phase = 50*hits[:, 0] / np.hypot(2172.8*.06/energy.sum(), 2.5)
    waves = np.exp(1j * phase[:, None] * np.arange(3))
    gram = waves.conj().T @ ((energy / energy.sum())[:, None] * waves)
    assert rows[0, names.index("mass_energy_L0to11_gram3_theta_0.06GeV")] == pytest.approx(np.linalg.det(gram).real)


def test_selected_shower_columns(tmp_path):
    """
    Match detector-only graph rows and names after selection in steering order

    Args:
        tmp_path:  Detector ROOT file directory
    """
    path = write_root(tmp_path / "hits.root", events=2, truth=False)
    cfg = load_config("configs/pi0gamma/twistgraph.yml", ["driver.root_workers=1", "driver.shower_columns=null"])
    full = build_experiment(cfg).driver
    expected = full.prediction_dataset(path)[0]
    columns = [len(full.input_spec.features) - 1, 0, 1, 5]
    cfg.driver["shower_columns"] = [columns[0], [0, 2], 5]
    selected = build_experiment(cfg).driver
    dataset = selected.prediction_dataset(path)
    assert torch.is_tensor(dataset.blocks[0][1].features)
    sample = dataset[0]
    assert selected.input_spec.features == tuple(full.input_spec.features[i] for i in columns)
    torch.testing.assert_close(sample.features, expected.features[:, columns])
    torch.testing.assert_close(sample.global_features, expected.global_features)
    batch = collate_samples(dataset.__getitems__([0, 1]))
    mlp = torch.nn.Sequential(torch.nn.Linear(len(columns), 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    logits = mlp(batch.features[:, 0])
    assert logits.shape == (2, 1) and torch.isfinite(logits).all()
    logits.sum().backward()
    assert mlp[0].weight.grad is not None
    cfg.driver["shower_columns"] = [0, 0]
    with pytest.raises(ValueError, match="shower_columns"):
        build_experiment(cfg)


def test_selected_shower_groups_skip_disabled_kernels(tmp_path, monkeypatch):
    """
    Read a group subset in basis order and skip the other graph kernels

    Args:
        tmp_path:  Detector ROOT file directory
        monkeypatch:  Pytest function override fixture
    """
    from drivers.pi0gamma.graphs.descriptors import BASES, SAMPLING_RANK_BASIS, SHOWER_FRAME_KERNELS

    file = write_root(tmp_path / "grouped.root", events=2, truth=False, cell_ids=False)
    config = load_config("configs/pi0gamma/twistgraph.yml", ["driver.root_workers=1", "driver.shower_groups=null",
                                                           "driver.shower_columns=null"])
    full = build_experiment(config).driver
    expected = full.prediction_dataset(file)[0]
    profile = "shower_frame_projected_profiles"
    groups = [SAMPLING_RANK_BASIS, profile]
    config.driver["shower_groups"] = groups
    selected = build_experiment(config).driver
    names = BASES[profile] + BASES[SAMPLING_RANK_BASIS]
    indices = {name: index for index, name in enumerate(full.input_spec.features)}
    assert selected.options.graph_components == (profile, SAMPLING_RANK_BASIS)
    assert selected.input_spec.features == names

    def disabled_kernel(*args, **kwargs):
        """Fail if a disabled group reaches its graph kernel"""
        pytest.fail("A disabled shower group was evaluated")

    monkeypatch.setitem(SHOWER_FRAME_KERNELS, "shower_frame_axis_graphs", disabled_kernel)
    sample = selected.prediction_dataset(file)[0]
    torch.testing.assert_close(sample.features, expected.features[:, [indices[name] for name in names]], rtol=0, atol=0)
    torch.testing.assert_close(sample.global_features, expected.global_features, rtol=0, atol=0)

    for invalid in ([], [profile, profile], ["unknown_group"], profile):
        config.driver["shower_groups"] = invalid
        with pytest.raises(ValueError, match="shower_groups"):
            build_experiment(config)


def test_twistgraph_shower_frame_independent_of_readout(tmp_path):
    """
    Compare ROOT inputs with absent, sparse and all-strip readout settings

    Args:
        tmp_path:  ROOT fixture directory
    """
    file = write_root(tmp_path / "shower.root", events=2, cell_ids=False, truth=False)
    config = load_config("configs/pi0gamma/twistgraph.yml", ["driver.shower_columns=null"])
    assert config.driver["geometry"] is None
    driver = build_experiment(config).driver
    expected = driver.prediction_dataset(file)[0]
    names = driver.input_spec.features
    assert len(names) == len(set(names)) == expected.features.shape[1]
    assert expected.features[0, names.index("shower_frame_energy_edge_D5_k0")] > 0
    for merges in ([4, 1, 4, 1, 4, 4], [1] * 16):
        config.driver["geometry"] = {"inner_radius_mm": 4000., "theta_merge": merges}
        sample = build_experiment(config).driver.prediction_dataset(file)[0]
        torch.testing.assert_close(sample.features, expected.features, rtol=0, atol=0)
        torch.testing.assert_close(sample.global_features, expected.global_features, rtol=0, atol=0)


def test_shower_frame_coordinates():
    """
    Check angular offsets, tangent-plane offsets and projected depth
    """
    from drivers.pi0gamma.graphs.graph_inputs import ANGLE_UNIT_RAD, shower_frame_hits

    position = torch.tensor([[990., -20., 0.], [1010., 20., 0.]])
    energy = torch.ones(2)
    angular = shower_frame_hits(position, energy, "angular")
    cartesian = shower_frame_hits(position, energy, "cartesian")
    expected_angle = torch.atan2(position[:, 1], position[:, 0]) / ANGLE_UNIT_RAD
    expected_tangent = position[:, 1] / (1000 * ANGLE_UNIT_RAD)
    torch.testing.assert_close(angular[:, 1], expected_angle)
    torch.testing.assert_close(cartesian[:, 1], expected_tangent)
    for rows in (angular, cartesian):
        torch.testing.assert_close(rows[:, 2], torch.tensor([-.2, .2]))
        assert not rows[:, (0, 3)].any()


def test_radial_shower_graph_basis(tmp_path):
    """
    Retain the projective and radial graph features in the combined frame

    Args:
        tmp_path:  ROOT fixture directory
    """
    from drivers.pi0gamma.graphs.descriptors import (
        RADIAL_COMPONENTS, RADIAL_SHOWER_BASIS, SAMPLING_RANK_BASIS, SPECTRAL_DEPTH_BASIS, SPECTRAL_SHOWER_BASIS,
        TWISTGRAPH_BASIS,
    )

    file = write_root(tmp_path / "hits.root", events=2, truth=False, cell_ids=False)
    config = load_config("configs/pi0gamma/twistgraph.yml", ["driver.shower_columns=null"])
    radial_parts = ("shower_frame_shape_graphs", *RADIAL_COMPONENTS)
    combined = {
        RADIAL_SHOWER_BASIS: radial_parts,
        SPECTRAL_SHOWER_BASIS: (RADIAL_SHOWER_BASIS, SPECTRAL_DEPTH_BASIS),
        TWISTGRAPH_BASIS: (SPECTRAL_SHOWER_BASIS, SAMPLING_RANK_BASIS),
    }
    results, names = {}, {}
    for basis in (*radial_parts, SPECTRAL_DEPTH_BASIS, SAMPLING_RANK_BASIS, *combined):
        config.driver.update(shower_basis=basis, shower_groups=None, root_workers=1)
        driver = build_experiment(config).driver
        dataset = driver.prediction_dataset(file)
        results[basis] = dataset.blocks[0][1].features
        names[basis] = driver.input_spec.features
    for basis, parts in combined.items():
        assert names[basis] == sum((names[part] for part in parts), ())
        torch.testing.assert_close(results[basis], torch.cat([results[part] for part in parts], dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph reduction"))])
def test_lobe_graphs_require_independent_depth_support(device):
    """
    Distinguish a persistent second lobe from an isolated deposit

    Args:
        device:  Torch device for the graph contractions
    """
    from drivers.pi0gamma.graphs.descriptors import (
        LOBE_GRAPH_BASIS, LOBE_TRANSPORT_BASIS, descriptor_rows, shower_graph_components,
    )
    from drivers.pi0gamma.graphs.lobe_graphs import LOBE_GRAPH_NAMES, LOBE_TRANSPORT_NAMES

    primary = torch.tensor([[0., 0., -2., .7], [0., 0., 0., .7], [0., 0., 2., .7]], dtype=torch.float64)
    secondary = torch.tensor([[2.4, 0., -2., -1.], [2.4, 0., 0., -1.], [2.4, 0., 2., -1.]], dtype=torch.float64)
    coherent = torch.cat((primary, secondary))
    isolated = torch.cat((primary, torch.tensor([[2.4, 0., 0., -1. + np.log(3.)]], dtype=torch.float64)))
    reflected = coherent.flip(0).clone()
    reflected[:, 0] *= -1
    reflected[:, 2] = 1.7 * reflected[:, 2] + .8
    basis = shower_graph_components("twistgraph", [LOBE_GRAPH_BASIS])
    showers = [coherent, isolated, primary[:1], reflected]
    rows = descriptor_rows(showers, basis, device)
    assert rows.device.type == device and rows.shape == (4, len(LOBE_GRAPH_NAMES))
    assert torch.isfinite(rows).all()
    for scale in (6, 12):
        opening = LOBE_GRAPH_NAMES.index(f"shower_frame_lobe_{scale}mrad_opening_rad")
        coherence = LOBE_GRAPH_NAMES.index(f"shower_frame_lobe_{scale}mrad_secondary_coherence")
        torch.testing.assert_close(rows[:2, opening], rows.new_full((2,), .06), atol=5e-5, rtol=0)
        assert rows[0, coherence] > .99
        assert rows[1, coherence] == rows[2, coherence] == rows[2, opening] == 0
    torch.testing.assert_close(rows[0], rows[3], atol=1e-7, rtol=1e-6)
    combined = shower_graph_components("twistgraph", [LOBE_GRAPH_BASIS, LOBE_TRANSPORT_BASIS])
    lobe_rows, transport_rows = descriptor_rows(showers, combined, device).split(
        (len(LOBE_GRAPH_NAMES), len(LOBE_TRANSPORT_NAMES)), dim=1
    )
    torch.testing.assert_close(lobe_rows, rows, rtol=0, atol=0)
    torch.testing.assert_close(transport_rows, descriptor_rows(showers, LOBE_TRANSPORT_BASIS, device), rtol=0, atol=0)
    assert torch.isfinite(transport_rows).all() and not transport_rows[2].any()


def test_lobe_transport_resolves_direction_changes():
    """Resolve opposite energy flow and orthogonal quadrupoles across depth"""
    from drivers.pi0gamma.graphs.lobe_graphs import transport_descriptors

    centers = torch.zeros((2, 2, 3, 2), dtype=torch.float64)
    narrow_centers = centers.clone()
    narrow_centers[..., 0] = -.2
    narrow_centers[1, 0, -1, 0] = .2
    covariance = torch.diag_embed(centers.new_tensor([.4, .1]).expand_as(centers)).clone()
    covariance[1, 0, -1] = covariance[1, 0, -1].flip((0, 1))
    support = centers.new_ones((2, 2, 3))
    support[:, 1, (0, 2)] = 0
    rows = transport_descriptors(centers, narrow_centers, covariance, support).reshape(2, 2, 3, 4)
    front_back = rows[:, 0, -1]
    torch.testing.assert_close(front_back[:, 0], centers.new_tensor([1., -1.]))
    torch.testing.assert_close(front_back[:, 2], centers.new_tensor([1., -1.]))
    assert not front_back[0, (1, 3)].any() and (front_back[1, (1, 3)] > 0).all()
    assert not rows[:, 1].any()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph reduction"))])
def test_spectral_depth_graphs_use_measured_depth(device):
    """
    Preserve the angular correlations when measured depths shift or scale

    Args:
        device:  Device for the depth-window contractions
    """
    from drivers.pi0gamma.graphs.spectral_depth_graphs import SPECTRAL_DEPTH_NAMES, spectral_depth_rows

    generator = torch.Generator().manual_seed(53)
    showers, transformed = [], []
    for count in (1, 2, 5, 11, 24):
        depth = torch.linspace(-5, 4, count, dtype=torch.float64).repeat_interleave(8)
        hits = torch.randn((len(depth), 4), generator=generator, dtype=torch.float64)
        hits[:, 2] = depth
        shifted = hits.clone()
        shifted[:, 2] = .8 * depth + 3.7
        showers.append(hits)
        transformed.append(shifted)
    rows = spectral_depth_rows(showers, device)
    assert rows.shape == (len(showers), len(SPECTRAL_DEPTH_NAMES))
    assert torch.isfinite(rows).all()
    torch.testing.assert_close(rows, spectral_depth_rows(transformed, device), atol=2e-7, rtol=1e-6)
    individual = torch.cat([spectral_depth_rows([shower]) for shower in showers])
    torch.testing.assert_close(rows.cpu(), individual, atol=2e-7, rtol=1e-6)


def test_sampling_rank_graphs_preserve_order_across_depth_spacing():
    """
    Preserve the angular graphs when ordered sampling positions change
    """
    from drivers.pi0gamma.graphs.sampling_rank_graphs import SAMPLING_RANK_NAMES, sampling_rank_rows

    generator = torch.Generator().manual_seed(61)
    showers, moved = [], []
    for count in (1, 2, 7, 11, 35):
        order = torch.arange(count, dtype=torch.float64).repeat_interleave(4)
        hits = torch.randn((len(order), 4), generator=generator, dtype=torch.float64)
        hits[:, 2] = .05 * order
        shifted = hits.clone()
        shifted[:, 2] = -.7 + .06*order + .001*order.square()
        showers.append(hits)
        moved.append(shifted)
    rows = sampling_rank_rows(showers)
    assert rows.shape == (len(showers), len(set(SAMPLING_RANK_NAMES))) and torch.isfinite(rows).all()
    torch.testing.assert_close(rows, sampling_rank_rows(moved), rtol=0, atol=0)


def test_graph_batches_skip_truth_rejections(monkeypatch):
    """
    Keep selected graph rows in source order and skip rejected contractions

    Args:
        monkeypatch:  Pytest function override fixture
    """
    from drivers.pi0gamma import reader

    basis = ("shower_frame_harmonic_graphs", "shower_radial_energy_scaled_graphs")
    generator = torch.Generator().manual_seed(127)
    showers = [torch.randn((8, 8), generator=generator) for _ in range(5)]
    full = reader.descriptor_batches(showers, basis, "cpu")
    descriptor_rows = reader.descriptor_rows
    batches = []

    def record_batch(hits, *args):
        """
        Record the shower count for each contraction

        Args:
            hits:  Batch of retained shower hits
            args:  Graph component and device settings
        Returns:
            rows:  Batch of graph columns
        """
        batches.append(len(hits))
        return descriptor_rows(hits, *args)

    monkeypatch.setattr(reader, "descriptor_rows", record_batch)
    keep = reader.graph_selection(np.array([True, True, False, True, False]), np.array([False, True, True, True, True]))
    partial = reader.descriptor_batches(showers, basis, "cpu", keep=keep)
    torch.testing.assert_close(partial[keep], full[keep], rtol=0, atol=0)
    assert not partial[~keep].count_nonzero()
    empty = reader.descriptor_batches(showers, basis, "cpu", keep=np.zeros(len(showers), dtype=bool))
    assert empty.shape == full.shape and not empty.count_nonzero()
    assert batches == [2]


def test_shower_frame_graphs_omit_duplicate_axis_columns():
    """
    Check the canonical axis group and its distinct feature columns
    """
    from drivers.pi0gamma.graphs.axis_graphs import SCALES, axis_graph_rows
    from drivers.pi0gamma.graphs.descriptors import BASES, descriptor_rows, shower_graph_components

    basis = "shower_frame_shape_graphs"
    axis_group = "shower_frame_axis_graphs"
    assert shower_graph_components(basis, [axis_group]) == (axis_group,)
    with pytest.raises(ValueError, match="shower_groups"):
        shower_graph_components(basis, [axis_group + "_v2"])
    with pytest.raises(ValueError, match="shower_basis"):
        shower_graph_components(basis + "_v2")
    names = BASES[axis_group]
    assert len(names) == 760 and len(BASES[basis]) == 7938
    assert all(f"shower_frame_axes_s{scale}_D0to11_{term}" not in names
               for scale in SCALES for term in ("gain2", "two_root_mass"))
    generator = torch.Generator().manual_seed(829)
    showers = [torch.randn((count, 4), generator=generator) for count in (3, 14, 49)]
    rows = descriptor_rows(showers, axis_group)
    assert rows.shape == (len(showers), len(names))
    torch.testing.assert_close(rows, axis_graph_rows(showers, shower_frame=True), rtol=0, atol=0)


def test_longitudinal_graph_cumulants():
    """
    Resolve equal transverse prongs after removing the common shower slope
    """
    from drivers.pi0gamma.graphs.descriptors import BASES, descriptor_rows

    depth = np.repeat([.5, 1.2, 2.4], 2)
    side = np.tile([-.3, .3], 3)
    hits = np.column_stack((.4 + side, -.15 * depth, depth, np.zeros(6)))
    reflected = hits[::-1].copy()
    reflected[:, 0] *= -1
    rows = descriptor_rows([hits, reflected, hits[:1]], "transport_graphs", inner_radius_mm=2172.8)
    names = BASES["transport_graphs"]
    assert rows.shape == (3, len(set(names))) and torch.isfinite(rows).all()
    np.testing.assert_allclose(rows[0], rows[1], atol=1e-6)
    # A symmetric distribution on two points has excess kurtosis of minus two
    assert rows[0, names.index("transport_32mm_L0to11_excess_theta")] == pytest.approx(-2.)
    assert rows[0, names.index("transport_32mm_L0to11_variance_theta")] == pytest.approx(.09)
    with pytest.raises(ValueError, match="ECAL inner radius"):
        descriptor_rows([hits], "transport_graphs")


def test_cluster_axis_graphs():
    """
    Resolve separated deposits across depths with fixed two-root energy graphs
    """
    from drivers.pi0gamma.graphs.axis_graphs import AXIS_NAMES, axis_graph_rows

    narrow = np.array([[0., 0., .4, 0.]])
    wide = np.array([[-3., 0., .4, 0.], [3., 0., .4, -.1], [-3.1, .1, 1., -.2], [3.1, .1, 1., -.3]])
    reflected = wide[::-1].copy()
    reflected[:, 0] *= -1
    rows = axis_graph_rows([narrow, wide, reflected])
    assert rows.shape == (3, len(set(AXIS_NAMES))) and torch.isfinite(rows).all()
    np.testing.assert_array_equal(rows[0], axis_graph_rows([narrow])[0])
    np.testing.assert_allclose(rows[1], rows[2], atol=1e-5)
    result = dict(zip(AXIS_NAMES, rows[1], strict=True))
    assert result["axes_s0.4_opening"] > 5
    assert result["axes_s0.4_second_gain"] > .3
    assert result["axes_s0.4_tau21"] < .1


@pytest.mark.parametrize("shower_frame", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph reduction"))])
def test_axis_graph_hit_gradients(shower_frame, device):
    """
    Differentiate two-axis features for padded showers and zero-width depth bands

    Args:
        shower_frame:  Use shower-relative coordinates and depth bands
        device:  Device for the axis contractions
    """
    from drivers.pi0gamma.graphs.axis_graphs import SCALARS, axis_graph_rows

    generator = torch.Generator().manual_seed(1221)
    hits = torch.randn(15, 4, dtype=torch.float64, generator=generator).requires_grad_()
    singleton = torch.zeros(1, 4, dtype=hits.dtype, requires_grad=True)
    coincident = torch.zeros(3, 4, dtype=hits.dtype, requires_grad=True)
    showers = [hits, singleton, coincident]
    rows = axis_graph_rows(showers, device=device, shower_frame=shower_frame)
    gradients = torch.autograd.grad(rows.sum(), showers, retain_graph=True)
    assert all(gradient.isfinite().all() for gradient in gradients)

    # Check the energy radius away from seed ties and depth boundaries
    column = SCALARS.index("energy_radius")
    direction = torch.randn(hits.shape, dtype=hits.dtype, generator=generator)
    derivative = (torch.autograd.grad(rows[0, column], hits)[0] * direction).sum()
    step = 1e-4
    with torch.no_grad():
        plus, minus = [axis_graph_rows([hits + delta * direction], device=device, shower_frame=shower_frame)[0, column]
                       for delta in (step, -step)]
    torch.testing.assert_close(derivative, ((plus - minus) / (2 * step)).to(derivative), rtol=.005, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph reduction")
def test_rooted_graph_device_precision():
    """
    Resolve nearly coincident projective cells with unequal batch padding
    """
    from drivers.pi0gamma.graphs.axis_graphs import axis_graph_rows

    rng = np.random.default_rng(71)
    showers = []
    for n in (333, 701, 480):
        hits = rng.normal(size=(n, 4))
        hits[:, :2] = np.rint(hits[:, :2] * 3) / 3 + 1e-5 * hits[:, :2]
        hits[:, 2] = rng.uniform(0, 6, n)
        showers.append(hits)
    expected = torch.cat([axis_graph_rows([hits]) for hits in showers])
    rows = axis_graph_rows(showers, "cuda")
    assert rows.is_cuda
    torch.testing.assert_close(rows.cpu(), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("weights", [None, [1.]])
def test_pi0_decay_cut_preserves_splits_and_prediction(tmp_path, geometry, weights):
    """
    Apply the generator cut to the same pion identities in every labeled split

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic ECAL readout
        weights:  Optional gun spectrum for the evaluation path
    """
    file = write_root(tmp_path / "decays.root", decays=True, metadata=True, events=60)
    config = load_config("configs/pi0gamma/transformer_angular.yml")
    options = {k: v for k, v in config.driver.items() if k != "name"}
    options.update(files=[file], base_path="", geometry=asdict(geometry), energy_edges=[0., 100.], energy_weights=weights)
    nominal = Driver({**options, "max_pi0_decay_asymmetry": None}, config.task)
    selected = Driver({**options, "max_pi0_decay_asymmetry": .49}, config.task)
    for split in ("train", "val", "test"):
        original = nominal.dataset(split)
        passed = selected.dataset(split)
        expected = [sample.id for sample in original if int(sample.id.rsplit(":", 1)[1]) % 4 != 2]
        assert [sample.id for sample in passed] == expected
    labeled = selected.dataset("test", input_path=file)
    assert len(labeled) == 45
    comparison = nominal.comparison_dataset(selected, "test", input_path=file)
    assert [sample.id for sample in comparison] == [sample.id for sample in labeled]
    assert all("passes_decay_cut" not in sample.observables for sample in labeled)
    edge = Driver({**options, "max_pi0_decay_asymmetry": .5}, config.task)
    assert len(edge.dataset("test", input_path=file)) == 45
    above = Driver({**options, "max_pi0_decay_asymmetry": .5001}, config.task)
    assert len(above.dataset("test", input_path=file)) == 60
    detector = write_root(tmp_path / "detector.root", truth=False, events=3)
    assert len(selected.prediction_dataset(detector)) == 3
    missing = write_root(tmp_path / "missing.root", events=3, decays=False)
    with pytest.raises(ValueError, match="requires MC daughter relations"):
        selected.dataset("test", input_path=missing)


def test_pi0_decay_cut_requires_direct_photons():
    """
    Check decay topology, relation bounds and the strict balanced endpoint
    """
    from drivers.pi0gamma.reader import passes_pi0_decay_cut

    event = dict(pdg=np.array([111, 22, 22, 11]), daughters_begin=np.array([0, 2, 2, 2]),
                 daughters_end=np.array([2, 2, 2, 2]), daughter_index=np.array([2, 1]),
                 mc_x=np.array([0., 3., 0., 0.]), mc_y=np.array([0., 4., 0., 0.]),
                 mc_z=np.array([0., 0., 5., 0.]))
    assert passes_pi0_decay_cut(event, 0, .01)
    assert not passes_pi0_decay_cut(event, 0, 0.)
    assert not passes_pi0_decay_cut({**event, "daughter_index": np.array([1, 3])}, 0, 1.)
    assert not passes_pi0_decay_cut({**event, "daughters_end": np.array([0, 2, 2, 2])}, 0, 1.)
    with pytest.raises(ValueError, match="indices"):
        passes_pi0_decay_cut({**event, "daughter_index": np.array([-1, 1])}, 0, 1.)
    for invalid in (-.01, 1.01, np.nan, True, ".95"):
        with pytest.raises(ValueError, match="max_pi0_decay_asymmetry"):
            Options(max_pi0_decay_asymmetry=invalid).validate_selection()


def test_photon_theta_acceptance(tmp_path, geometry):
    """
    Select both pion daughters or the single primary photon in each labeled split

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic ECAL readout
    """
    angles = [[20, 60, 120], [90, 20, 160], [90, 40, 90], [40, 90, 90],
              [90, 90, 139], [139, 90, 90], [90, 20, 160], [100, 20, 160]] * 4
    file = write_root(tmp_path / "theta.root", events=len(angles), photon_angles=angles)
    card = load_config("configs/pi0gamma/transformer_angular.yml")
    options = {k: v for k, v in card.driver.items() if k != "name"}
    bounds = options["photon_theta_range_deg"]
    corner = np.degrees(np.arctan2(2100., 2350.))
    np.testing.assert_allclose(bounds, [corner, 180 - corner])
    assert "eta_max" not in options
    options.update(files=[file], base_path="", geometry=asdict(geometry), energy_edges=[], energy_weights=None,
                   max_pi0_decay_asymmetry=None)
    selected = Driver(options, card.task)
    inclusive = Driver({**options, "photon_theta_range_deg": None, "eta_max": None}, card.task)
    expected = {f"{file}:{i}" for i in range(len(angles)) if i % 8 in (0, 1, 7)}
    for split in ("train", "val", "test"):
        assert [s.id for s in selected.dataset(split)] == [s.id for s in inclusive.dataset(split) if s.id in expected]
    assert {s.id for s in selected.dataset("val", input_path=file)} == expected
    assert len(selected.prediction_dataset(file)) == len(angles)
    for invalid in ([60, 60], [-1, 90], [10, 181], [True, 90], [np.nan, 90], 90):
        with pytest.raises(ValueError, match="photon_theta_range_deg"):
            Options(photon_theta_range_deg=invalid).validate_selection()


def test_event_and_cell_cutflow(tmp_path, geometry, capsys):
    """
    Check cumulative event and cell yields for ordered parallel ROOT ranges

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic ECAL readout
        capsys:  Captured parent process tables
    """
    angles = [[20, 60, 120], [90, 20, 160], [90, 40, 90], [40, 90, 90],
              [90, 90, 139], [139, 90, 90], [90, 20, 160], [100, 20, 160]]
    file = write_root(tmp_path / "flow.root", events=8, photon_angles=angles, extra_cone_cells=True)
    card = load_config("configs/pi0gamma/transformer_angular.yml")
    options = dict(files=[file], base_path="", feature_mode="angular", geometry=asdict(geometry), chunk_size=2,
                   min_energy=2., max_cells=1, max_pi0_decay_asymmetry=.45, photon_theta_range_deg=[42., 138.])
    for workers in (1, 2):
        driver = Driver({**options, "root_workers": workers}, card.task)
        assert len(driver.dataset("test", input_path=file)) == 3
        output = capsys.readouterr().out
        assert output.count("Event cutflow") == output.count("Cell cutflow") == 1
        assert output.index("Event cutflow") < output.index("Cell cutflow")
        rows = {fields[0]: fields[1:] for line in output.splitlines() if len(fields := re.split(r"\s{2,}", line.strip())) >= 5}
        assert rows["Associated MC primary"] == ["8", "0", "100.00", "100.00", "4", "4"]
        assert rows["max_pi0_decay_asymmetry: A < 0.45"] == ["6", "2", "75.00", "75.00", "2", "4"]
        assert rows["photon_theta_range_deg: 42 < theta < 138 deg"] == ["3", "3", "50.00", "37.50", "1", "2"]
        assert rows["Input cells (ECalBarrelModuleThetaMergedPositioned)"][0] == "21"
        assert rows["Leading cluster cone <= 0.3 rad"][:2] == ["15", "6"]
        assert "Cell E > 0" not in rows
        assert rows["min_energy: E >= 2 GeV"] == ["6", "9", "40.00", "28.57"]
        assert rows["max_cells: first 1 by energy"][:2] == ["3", "3"]

    detector = Driver({**options, "min_energy": 1000.}, card.task)
    assert len(detector.prediction_dataset(file)) == 0
    output = capsys.readouterr().out
    assert "Cell cutflow" in output and "Associated MC primary" not in output
    assert re.search(r"min_cells >= 1\s+0\s+8\s+0.00\s+0.00", output)
    assert re.search(r"min_energy: E >= 1000 GeV\s+0\s+0\s+-\s+-", output)

    detector = Driver({**options, "min_energy": 0.}, card.task)
    assert len(detector.prediction_dataset(file)) == 8
    output = capsys.readouterr().out
    assert re.search(r"Cell E > 0\s+24\s+16\s+60.00", output)
    assert "min_energy:" not in output


@pytest.mark.parametrize(("labeled", "file_count", "basis"), [
    (True, 2, None), (False, 2, None), (True, 1, None), (True, 1, "shower_shape_graphs"),
    (True, 1, "shower_frame_shape_graphs"),
    (True, 1, "twistgraph"),
])
def test_parallel_root_entry_ranges(tmp_path, geometry, labeled, file_count, basis, monkeypatch):
    """
    Match serial shower selection and source entries across TTree and RNTuple ranges

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic ECAL readout
        labeled:  Read MC labels and apply the generated decay cut
        file_count:  Split one RNTuple or process both ROOT formats
        basis:  Optional graph basis for the retained cells
        monkeypatch:  Pytest function override fixture
    """
    paths = [write_root(tmp_path / f"showers_{index}.root", events=13, ntuple=bool(index) or file_count == 1,
                        truth=labeled, event_weights=np.arange(13) + 1. + index * 13)
             for index in range(file_count)]
    config = load_config("configs/pi0gamma/transformer_angular.yml")
    options = {k: v for k, v in config.driver.items() if k != "name"}
    options.update(geometry=asdict(geometry), chunk_size=3, max_events=10, files=paths, base_path="",
                   weight_branch="event_weight", energy_weights=None, energy_edges=[], max_pi0_decay_asymmetry=.49,
                   shower_basis=basis)
    if basis == "twistgraph":
        options["shower_groups"] = ["shower_radial_sampling_rank", "shower_frame_projected_profiles",
                                    "shower_radial_energy_scaled_graphs"]
    pattern = str(tmp_path / "showers_*.root")
    datasets = []
    for count in (1, 2):
        driver = Driver({**options, "root_workers": count}, config.task)
        datasets.append(driver.dataset("val", input_path=pattern) if labeled else driver.prediction_dataset(pattern))
    if basis == "twistgraph":
        from drivers.pi0gamma import parallel_reader

        descriptor_batches = parallel_reader.descriptor_batches

        def graph_batches_cpu(showers, components, device, radius, *, keep):
            """
            Run the CUDA component group on CPU to check the column order

            Args:
                showers:  Retained local shower hits
                components:  Graph components for the CUDA batch
                device:  Requested CUDA device
                radius:  ECAL inner radius
                keep:  Truth selection mask
            Returns:
                rows:  Graph components evaluated on CPU
            """
            assert torch.device(device).type == "cuda" and len(components) > 1
            assert components == ("shower_radial_energy_scaled_graphs", "shower_radial_sampling_rank")
            return descriptor_batches(showers, components, "cpu", radius, keep=keep)

        monkeypatch.setattr(parallel_reader, "descriptor_batches", graph_batches_cpu)
        driver = Driver({**options, "root_workers": 2}, config.task)
        driver.descriptor_device = "cuda"
        datasets.append(driver.dataset("val", input_path=pattern))

    expected, *parallel = datasets
    entries = [index for index in range(10) if not labeled or index % 4 != 2]
    for actual in parallel:
        if basis is not None:
            assert all(torch.is_tensor(chunk.features) for _, chunk in actual.blocks)
        assert [sample.id for sample in actual] == [f"{path}:{index}" for path in paths for index in entries]
        for left, right in zip(expected, actual, strict=True):
            assert left.id == right.id and left.weight == right.weight
            torch.testing.assert_close(left.target, right.target)
            torch.testing.assert_close(left.features, right.features)
            torch.testing.assert_close(left.global_features, right.global_features)
            np.testing.assert_equal(left.observables, right.observables)


def test_auto_root_workers_split_small_file(tmp_path, geometry, monkeypatch):
    """
    Fill the CPU pool when one ROOT file fits in a single configured chunk

    Args:
        tmp_path:  ROOT fixture directory
        geometry:  Synthetic ECAL readout
        monkeypatch:  Override the process CPU affinity
    """
    from drivers.pi0gamma import driver as pi0gamma_driver

    monkeypatch.setattr(pi0gamma_driver.os, "sched_getaffinity", lambda _: set(range(4)), raising=False)
    file = write_root(tmp_path / "showers.root", events=12, ntuple=True, truth=False)
    config = load_config("configs/pi0gamma/twistgraph.yml")
    assert config.driver["root_workers"] == 0
    options = dict(files=[file], geometry=asdict(geometry), feature_mode="angular", chunk_size=20, max_events=9,
                   root_workers=config.driver["root_workers"])
    dataset = Driver(options, config.task).prediction_dataset(file)
    assert [chunk.entries.tolist() for _, chunk in dataset.blocks] == [[0, 1], [2, 3], [4, 5], [6, 7], [8]]
    assert [sample.id for sample in dataset] == [f"{file}:{entry}" for entry in range(9)]


@pytest.mark.parametrize("reverse", [False, True])
def test_pi0_asymmetry_roc_with_common_photons(tmp_path, geometry, reverse):
    """
    Use the measured pion asymmetry and all primary photons in each ROC

    Args:
        tmp_path:  ROOT input and binned evaluation output
        geometry:  Synthetic ECAL readout
        reverse:  Reverse the classifier output order
    """
    from twistformer.evaluation.categories import evaluate_categories

    file = write_root(tmp_path / "asymmetry.root", decays=True, events=16)
    config = load_config("configs/pi0gamma/transformer_angular.yml")
    options = {k: v for k, v in config.driver.items() if k != "name"}
    options.update(files=[file], base_path="", geometry=asdict(geometry), energy_edges=[], energy_weights=None,
                   max_pi0_decay_asymmetry=None)
    if reverse:
        options.update(class_names=["gamma", "pi0"], class_pdgs=[22, 111], signal_class=0)
    driver = Driver(options, config.task)
    samples = driver.dataset("val", input_path=file)
    values = np.array([sample.observables["pi0_decay_asymmetry"] for sample in samples])
    np.testing.assert_equal(values, np.tile([0., np.nan, .5, np.nan], 4))
    labels = np.array([int(sample.target) for sample in samples])
    data = dict(weights=np.ones(16), targets=labels, probabilities=np.eye(2)[labels], observable_pi0_decay_asymmetry=values)
    category = next(c for c in config.evaluation.categories if c["name"] == "pi0_decay_asymmetry")
    result = evaluate_categories(data, driver.task_spec, [category], driver.evaluation_observables, tmp_path / "roc",
                                 plots=False, bootstrap_samples=0)["categories"][0]
    assert result["reference_events"] == 8 and result["missing_events"] == 0
    for index, entry in enumerate(result["bins"]):
        classes = entry["support"]["classes"]
        assert classes["gamma"]["events"] == 8
        assert classes["pi0"]["events"] == (4 if index in (0, 2) else 0)
        assert entry["status"] == ("completed" if index in (0, 2) else "unavailable")


@pytest.mark.parametrize("normalize", [False, True])
def test_packed_shower_histograms(normalize):
    """
    Preserve split weights, energy bins, global columns and repeated training views

    Args:
        normalize:  Fit nonidentity descriptor and global transforms
    Returns:
        None:  Packed shower rows and quantile cuts match the sample loader
    """
    pytest.importorskip("xgboost")
    from torch.utils.data import Subset
    from drivers.pi0gamma.driver import ShowerDescriptors
    from twistformer.data.preprocessing import PreparedDataset
    from twistformer.training.xgboost import _packed_reader, shower_matrix

    config = load_config("configs/pi0gamma/twistgraph.yml", ["training.eval_batch_size=9", "training.device=cpu"])
    config.driver.update(energy_edges=[0, 7.5, 15], energy_weights=None, balance_energy=True)
    config.preprocessing.normalize_features = config.preprocessing.normalize_globals = normalize
    experiment = build_experiment(config)
    width = len(experiment.driver.input_spec.features)
    global_width = len(experiment.driver.input_spec.global_features)
    rng = np.random.default_rng(62)
    blocks = []
    for block in range(3):
        events = 16
        features = rng.random((events, width), dtype=np.float32)
        features[:, 0] = np.float32(.41)
        blocks.append((str(block), ClusterChunk(
            features, np.arange(events + 1), np.arange(events), np.arange(events) % 2,
            np.ones(events), {"truth_energy": np.tile([5., 5., 10., 10.], 4)},
            rng.random((events, global_width), dtype=np.float32))))
    selection = rng.permutation([(block, event) for event in range(events) for block in range(len(blocks))])
    packed = ShowerDescriptors(blocks, selection)
    reference = Pi0GammaDataset(blocks, selection)
    packed.weights = reference.weights = rng.uniform(.01, 2., len(packed))
    packed.weights[::7] = 0.
    make = experiment.driver.create_preprocessor
    a, b = (make(config=config.preprocessing).fit(data, class_count=2) for data in (packed, reference))
    torch.testing.assert_close(a.state, b.state, rtol=0., atol=0.)
    for indices in (np.arange(len(packed)).repeat(2), np.array([3, 8, 4, 3, 8, 7, 0, 6, 10, 9])):
        fast = PreparedDataset(Subset(packed, indices), a)
        scalar = PreparedDataset(Subset(reference, indices), b)
        rows = _packed_reader(fast)(slice(None))
        batch = collate_samples(scalar.__getitems__(range(len(indices))))
        np.testing.assert_array_equal(rows[0], experiment.model.inputs(batch))
        np.testing.assert_array_equal(rows[1], batch.targets)
        np.testing.assert_array_equal(rows[2], batch.sample_weights)
        encoded, expected = (shower_matrix(experiment, data) for data in (fast, scalar))
        for actual, original in zip(encoded.get_quantile_cut(), expected.get_quantile_cut(), strict=True):
            np.testing.assert_array_equal(actual, original)
