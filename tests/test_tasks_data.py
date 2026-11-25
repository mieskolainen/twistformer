# Physics and workflow validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from twistformer.api.config import TaskConfig, parse_config
from twistformer.api.contracts import LossResult, TaskSpec, Sample, Prediction
from twistformer.tasks.supervised import SupervisedTask
from twistformer.data.batching import collate_samples
from twistformer.data.preprocessing import Preprocessor
from drivers.base import build_driver


def test_losses_reference_missing_and_weights():
    samples = [
        Sample(torch.ones(2, 1), torch.tensor([2.0, float("nan")]), weight=2, id="a"),
        Sample(torch.ones(1, 1), torch.tensor([1.0, 3.0]), weight=1, id="b"),
    ]
    batch = collate_samples(samples)
    mean = torch.zeros(2, 2, requires_grad=True)
    std = torch.ones(2, 2, requires_grad=True)
    task = SupervisedTask(
        TaskSpec("gaussian_regression", ("a", "b")),
        TaskConfig(name="gaussian_regression"),
    )
    result = task.compute_loss(Prediction(mean=mean, std=std), batch)
    constant = 0.5 * math.log(2 * math.pi)
    expected = (2 * (2 + constant) + (0.5 + 4.5) / 2 + constant) / 3
    assert result.loss.item() == pytest.approx(expected)
    result.loss.backward()
    assert mean.grad[0, 1] == 0 and std.grad[0, 1] == 0
    assert torch.isfinite(mean.grad).all()
    labels = replace(batch, targets=torch.tensor([0, 1]))
    logits = torch.tensor([[2.0, -1.0], [0.0, 1.0]], requires_grad=True)
    classification = SupervisedTask(
        TaskSpec("classification", ("a", "b")), TaskConfig()
    )
    expected = (
        torch.nn.functional.cross_entropy(logits, labels.targets, reduction="none")
        * batch.sample_weights
    ).sum() / 3
    torch.testing.assert_close(
        classification.compute_loss(
            Prediction(logits=logits[:, 1:] - logits[:, :1]), labels
        ).loss,
        expected,
    )


@pytest.mark.parametrize("weight", [-2.0, 1e-16])
def test_loss_normalization_preserves_event_weight_scale(weight):
    """Preserve the weighted mean for signed and small event yields"""
    value = torch.tensor(3.0, requires_grad=True)
    result = LossResult(value * weight, torch.tensor(weight))
    torch.testing.assert_close(result.loss, value)
    result.loss.backward()
    assert value.grad.item() == pytest.approx(1)
    assert LossResult(torch.tensor(0.), torch.tensor(0.)).loss.item() == 0
    assert torch.isnan(LossResult(torch.tensor(1.), torch.tensor(0.)).loss)


def test_scalar_regression_batch_and_schema():
    """Keep scalar targets and masks aligned and reject broadcast feature widths"""
    samples = [
        Sample(torch.ones(2, 3), torch.tensor(value), target_mask=torch.tensor(True))
        for value in (1.0, 3.0)
    ]
    pre = Preprocessor().fit(samples, regression=True)
    particle = samples[0].features[0]
    torch.testing.assert_close(pre.transform_inputs(particle)[0], pre.transform(samples[0]).features[0])
    batch = collate_samples(pre.transform_many(samples))
    single = collate_samples([pre.transform(sample) for sample in samples])
    assert batch.targets.shape == batch.target_mask.shape == (2, 1)
    torch.testing.assert_close(batch.targets, single.targets)
    torch.testing.assert_close(batch.target_mask, single.target_mask)
    invalid = replace(samples[0], features=torch.ones(2, 1))
    with pytest.raises(ValueError, match="feature widths"):
        Preprocessor().fit([samples[0], invalid])
    with pytest.raises(ValueError, match="Feature width"):
        pre.transform(invalid)
    with pytest.raises(ValueError, match="Feature width"):
        pre.transform_many([invalid])


def test_binary_probability_tails_and_class_indices():
    """Keep both classifier tails and reject labels outside the binary schema"""
    prediction = Prediction(logits=torch.tensor([[20.0], [-20.0]]))
    probabilities = prediction.probabilities
    assert probabilities[0, 0] == probabilities[1, 1] > 0
    batch = collate_samples([Sample(torch.ones(1, 1), torch.tensor(label)) for label in (-1, 2)])
    task = SupervisedTask(TaskSpec("classification", ("a", "b")), TaskConfig())
    with pytest.raises(RuntimeError, match="declared class"):
        task.compute_loss(prediction, batch)


def test_preprocessing_no_mutation_and_inverse():
    samples = [
        Sample(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([10.0]), id="a"),
        Sample(torch.tensor([[5.0, 6.0]]), torch.tensor([20.0]), id="b"),
    ]
    original = samples[0].features.clone()
    pre = Preprocessor().fit(samples, True)
    converted = pre.transform(samples[0])
    torch.testing.assert_close(samples[0].features, original)
    torch.testing.assert_close(pre.inverse_target(converted.target), samples[0].target)
    pred = pre.inverse_prediction(
        Prediction(mean=torch.zeros(1, 1), std=torch.ones(1, 1))
    )
    assert pred.mean.item() == 15 and pred.std.item() == 5
    # Validation data never refit training normalization
    before = {k: v.clone() for k, v in pre.state.items()}
    pre.transform(Sample(torch.full((1, 2), 10000.0), id="validation"))
    for key in before:
        torch.testing.assert_close(pre.state[key], before[key])


def test_config_rejects_unknown_and_incompatible():
    with pytest.raises(ValueError, match="Unknown"):
        parse_config({"driver": {"name": "synthetic"}, "model": {"widht": 8}})
    with pytest.raises(ValueError, match="attention"):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "model": {"family": "pelican", "attention": "hyperbolic"},
            }
        )
    with pytest.raises(ValueError, match="only meaningful"):
        parse_config(
            {
                "driver": {"name": "synthetic"},
                "model": {"family": "transformer", "three_point": True},
            }
        )


def test_npz_plugin_and_unlabeled_inference(tmp_path):
    file = tmp_path / "data.npz"
    np.savez(
        file,
        features=np.ones((3, 4, 2), dtype=np.float32),
        mask=np.array([[1, 1, 0, 0]] * 3, dtype=bool),
    )
    app = build_driver(
        {
            "name": "generic",
            "features": ["a", "b"],
            "outputs": ["target"],
            "files": {"test": str(file)},
        },
        TaskConfig(name="gaussian_regression"),
    )
    ds = app.dataset("test")
    assert ds[0].target is None and ds[0].features.shape == (2, 2)
    batch = collate_samples([ds[0], ds[1]])
    assert batch.targets is None


def test_npz_preserves_momenta_and_flow_coordinates(tmp_path):
    """Keep detector precision and image axes through NPZ input and batching"""
    file = tmp_path / "events.npz"
    momenta = np.array([[[1000.000001, 1000., 0., 0.]]])
    target = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    np.savez(file, features=np.ones((1, 1, 1)), p4=momenta, targets=target, target_mask=np.ones_like(target, dtype=bool))
    driver = build_driver(
        {"name": "generic", "features": ["e"], "geometry": "p4", "outputs": list("abcd"), "files": {"train": str(file)}},
        TaskConfig(name="flow_matching"),
    )
    dataset = driver.dataset("train")
    preprocessor = driver.create_preprocessor().fit(dataset)
    batch = collate_samples(preprocessor.transform_many([dataset[0]]))
    torch.testing.assert_close(batch.p4, torch.from_numpy(momenta), atol=0, rtol=0)
    assert batch.targets.shape == batch.target_mask.shape == target.shape
    torch.testing.assert_close(preprocessor.inverse_target(batch.targets), torch.from_numpy(target))


def test_reject_empty_and_negative_weights():
    with pytest.raises(ValueError, match="at least one"):
        collate_samples([Sample(torch.empty(0, 2), id="empty")])
    with pytest.raises(ValueError, match="nonnegative"):
        collate_samples([Sample(torch.ones(1, 2), weight=-1, id="bad")])


def test_external_driver_plugin(tmp_path, monkeypatch):
    path = tmp_path / "new_domain.py"
    path.write_text(
        "from drivers.synthetic.driver import Driver\nclass NewDomain(Driver):\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app = build_driver(
        {"name": "new_domain:NewDomain", "train_size": 8}, TaskConfig()
    )
    assert len(app.dataset("train")) == 8


def test_tau3_adapter_splits_and_preprocessing(monkeypatch):
    pytest.importorskip("awkward")
    pytest.importorskip("uproot")
    from drivers.tau3_classification.driver import Driver as Tau3
    from drivers.tau3_classification.reader import Tau3Chunk

    def reader(self, path):
        offset = float(path)
        x = np.arange(120, dtype=np.float32).reshape(10, 3, 4) + offset
        yield Tau3Chunk(
            x.reshape(-1, 4),
            np.arange(11) * 3,
            np.arange(1, 11, dtype=float),
            np.arange(10),
        )

    monkeypatch.setattr(Tau3, "_read", reader)
    app = Tau3(
        {"files": ["0", "100"], "class_names": ["a", "b"], "split": [0.6, 0.2, 0.2]},
        TaskConfig(),
    )
    train, val, test = (app.dataset(s) for s in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (12, 4, 4)
    ids = [{s.id for s in ds} for ds in (train, val, test)]
    for dataset in (train, val, test):
        for sample in dataset:
            assert sample.weight == int(sample.id.rsplit(":", 1)[1]) + 1
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    pre = app.create_preprocessor().fit(train)
    original = train[0].features.clone()
    pre.transform(train[0])
    torch.testing.assert_close(original, train[0].features)
    assert app.dataset("test", "200")[0].target is None


@pytest.mark.parametrize("storage", ["table", "fixed"])
def test_toptag_seeded_sampling_and_source_ids(tmp_path, storage):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    from drivers.toptag.driver import Driver as TopTag

    values = np.zeros((80, 800), dtype=np.float32)
    values[:, 0] = 2
    columns = [f"{name}_{i}" for name in ("E", "PX", "PY", "PZ") for i in range(200)]
    frame = pd.DataFrame(values, columns=columns)
    frame["is_signal_new"] = np.repeat([0, 1], 40)
    frame["event_weight"] = np.arange(80, dtype=float) / 10
    path = tmp_path / "jets.h5"
    frame.to_hdf(path, key="table", format=storage)
    app = TopTag(
        {
            "files": {"train": str(path)},
            "max_events": 20,
            "seed": 101,
            "weight_column": "event_weight",
        },
        TaskConfig(),
    )
    first, second = app.dataset("train"), app.dataset("train")
    assert len(first) == 20
    assert {int(s.target) for s in first} == {0, 1}
    assert [s.id for s in first] == [s.id for s in second]
    for sample in first:
        index = int(sample.id.rsplit(":", 1)[1])
        assert sample.target.item() == frame.iloc[index]["is_signal_new"]
        assert sample.weight == frame.iloc[index]["event_weight"]

    frame.loc[0, "E_0"] = 0
    frame.to_hdf(path, key="table", format=storage, mode="w")
    full = TopTag(
        {"files": {"train": str(path)}, "rotate": False}, TaskConfig()
    ).dataset("train")
    assert len(full) == 79
    assert full[0].id == f"{path}:1"
    assert full[-1].id == f"{path}:79"
    assert len(list(full)) == 79
    torch.testing.assert_close(full[0].p4, torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    full[0].p4.zero_()
    assert full[0].p4[0, 0] == 2
    assert full[0].weight == 1


@pytest.mark.parametrize("mode", ["sim", "reco-cluster", "reco-topocluster"])
@pytest.mark.parametrize("weighted", [False, True])
def test_tau3_root_reader_selection_and_energy_order(tmp_path, mode, weighted):
    ak = pytest.importorskip("awkward")
    uproot = pytest.importorskip("uproot")
    from drivers.tau3_classification.reader import ECAL_BRANCHES_BY_MODE, read_tau3

    branches = ECAL_BRANCHES_BY_MODE[mode]
    arrays = {
        "event_weight": np.array([0.5, 2.0, 3.0]),
        branches["x"]: ak.Array([[10.0, 20.0, 30.0]] * 3),
        branches["y"]: ak.Array([[0.0, 0.0, 0.0]] * 3),
        branches["z"]: ak.Array([[0.0, 0.0, 0.0]] * 3),
        branches["e"]: ak.Array([[0.5, 2.0, 0.01]] * 3),
        "MCParticles/MCParticles.PDG": ak.Array([[15, 211, 16]] * 3),
        "MCParticles/MCParticles.generatorStatus": ak.Array([[2, 1, 1]] * 3),
        "MCParticles/MCParticles.simulatorStatus": ak.Array([[0, 0, 0]] * 3),
        "MCParticles/MCParticles.momentum.x": ak.Array([[1.0, 1.0, 0.0]] * 3),
        "MCParticles/MCParticles.momentum.y": ak.Array([[0.0, 0.0, 0.0]] * 3),
        "MCParticles/MCParticles.momentum.z": ak.Array(
            [[0.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 100.0]]
        ),
        "MCParticles/MCParticles.parents_begin": ak.Array([[0, 0, 1]] * 3),
        "MCParticles/MCParticles.parents_end": ak.Array([[0, 1, 2]] * 3),
        "_MCParticles_parents/_MCParticles_parents.index": ak.Array([[0, 0]] * 3),
    }
    path = tmp_path / "tau3.root"
    with uproot.recreate(path) as output:
        output.mktree("events", arrays)
    features, mask, weights = read_tau3(
        str(path),
        "events",
        max_hits=3,
        max_events=2,
        chunk_size=1,
        e_min=0.1,
        fiducial_eta_max=1.1,
        input_mode=mode,
        weight_branch="event_weight" if weighted else None,
    )
    assert features.shape == (1, 3, 4)
    np.testing.assert_array_equal(mask, [[True, True, False]])
    np.testing.assert_allclose(features[0, :, 3], [2.0, 0.5, 0.0])
    np.testing.assert_allclose(features[0, :, 0], [20.0, 10.0, 0.0])
    np.testing.assert_allclose(weights, [0.5 if weighted else 1.0])


def test_driver_convention_discovery(tmp_path, monkeypatch):
    import drivers

    package = tmp_path / "my_domain"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "driver.py").write_text(
        "from drivers.synthetic.driver import Driver as Synthetic\n"
        "class Driver(Synthetic):\n    pass\n"
    )
    monkeypatch.setattr(
        drivers, "__path__", [*drivers.__path__, str(tmp_path)]
    )
    app = build_driver({"name": "my_domain", "train_size": 8}, TaskConfig())
    assert type(app).__module__ == "drivers.my_domain.driver"
    assert len(app.dataset("train")) == 8
    with pytest.raises(ValueError, match="Unknown driver"):
        build_driver({"name": "missing_domain"}, TaskConfig())
