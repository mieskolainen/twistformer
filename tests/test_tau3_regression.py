# ROOT tau six-vector regression and checkpoint workflow validation
#
# mikael.mieskolainen@cern.ch (2026)

import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from twistformer import load_config, build_experiment, Trainer, load_predictor
from twistformer.api.config import TaskConfig
from twistformer.evaluation.runner import evaluate, collect_predictions
from drivers.tau3_regression.driver import Driver
from drivers.tau3_regression.evaluation import observables, evaluate_extra

ak = pytest.importorskip("awkward")
uproot = pytest.importorskip("uproot")

from drivers.tau3_classification.reader import (  # noqa: E402
    ECAL_BRANCHES_BY_MODE,
    MC_PARTICLE_VARS,
    MC_RELATION_VARS,
    iter_tau3,
)
from drivers.tau3_regression.reader import (  # noqa: E402
    VERTEX_BRANCHES,
    ENDPOINT_BRANCHES,
    iter_tau3_regression,
    tau_targets,
)


def root_fixture(path, *, truth=True, mode="sim", size=32):
    """
    Write detector hits with displaced production vertices and truth labels

    Args:
        path:  ROOT output file
        truth:  Include generator labels and parent relations
        mode:  Calorimeter branch collection
        size:  Number of raw events
    Returns:
        arrays: ROOT column dictionary
    """
    arrays = {}
    branches = ECAL_BRANCHES_BY_MODE[mode]
    for axis in "xyz":
        arrays[branches[axis]] = ak.Array(
            [[float(i), float(i + 10)] for i in range(size)]
        )
    arrays[branches["e"]] = ak.Array(
        [[0.01, 0.01] if i == 2 else [1.0, 2.0] for i in range(size)]
    )
    arrays["weight"] = np.arange(size, dtype=float) + 1
    if truth:
        columns = [
            [[15, 211]] * size,
            [[2, 1]] * size,
            [[0, 0]] * size,
            [[i + 1.0, 1.0] for i in range(size)],
            [[-1.0, 0.0]] * size,
            [[0.5, np.sinh(2) if i == 1 else 0.0] for i in range(size)],
            [[0, 0]] * size,
            [[0, 1]] * size,
            [[0]] * size,
        ]
        arrays.update(
            {
                key: ak.Array(value)
                for key, value in zip(MC_PARTICLE_VARS + MC_RELATION_VARS, columns)
            }
        )
        for axis, (vertex, endpoint) in enumerate(
            zip(VERTEX_BRANCHES, ENDPOINT_BRANCHES)
        ):
            arrays[vertex] = ak.Array([[10.0 * (axis + 1), 0.0]] * size)
            arrays[endpoint] = ak.Array(
                [[10.0 * (axis + 1) + (axis + 1) * i / 10, 0.0] for i in range(size)]
            )
    with uproot.recreate(path) as root:
        root.mktree("events", arrays)
    return arrays


@pytest.mark.parametrize("mode", list(ECAL_BRANCHES_BY_MODE))
def test_hit_truth_alignment_and_chunk_independence(tmp_path, mode):
    """
    Preserve classification hits and align displacement targets across chunk sizes
    """
    path = tmp_path / "tau.root"
    root_fixture(path, mode=mode)
    options = dict(
        max_hits=2, max_events=31, e_min=0.1, input_mode=mode, weight_branch="weight"
    )
    chunks = list(iter_tau3_regression(str(path), "events", chunk_size=3, **options))
    reference = list(iter_tau3(str(path), "events", chunk_size=3, **options))
    for chunk, original in zip(chunks, reference):
        np.testing.assert_array_equal(chunk.features, original.features)
        np.testing.assert_array_equal(chunk.entries, original.entries)
        np.testing.assert_array_equal(chunk.weights, original.weights)
        np.testing.assert_allclose(chunk.targets[:, 0], chunk.entries + 1)
        np.testing.assert_allclose(
            chunk.targets[:, 3:],
            chunk.entries[:, None] * np.array([0.1, 0.2, 0.3]),
            atol=1e-6,
        )
    base = dict(
        files=[str(path)],
        input_mode=mode,
        min_energy=0.1,
        max_events=31,
        weight_branch="weight",
    )
    small = Driver(dict(base, chunk_size=1), TaskConfig(name="gaussian_regression"))
    large = Driver(dict(base, chunk_size=9), TaskConfig(name="gaussian_regression"))
    identities = []
    for split in ("train", "val", "test"):
        first, second = small.dataset(split), large.dataset(split)
        assert [s.id for s in first] == [s.id for s in second]
        identities.extend(s.id for s in first)
        for a, b in zip(first, second):
            torch.testing.assert_close(a.target, b.target)
            assert (
                a.features.shape == (2, 4)
                and a.p4 is None
                and a.global_features is None
            )
    assert len(set(identities)) == len(identities) == 29
    before = small.dataset("train")[0].target.clone()
    small.dataset("train")[0].target.zero_()
    torch.testing.assert_close(small.dataset("train")[0].target, before)


def test_truth_validation_and_hit_only_inference(tmp_path):
    """
    Reject ambiguous truth associations and permit MC-free calorimeter inference
    """
    path = tmp_path / "hits.root"
    root_fixture(path, truth=False, size=8)
    app = Driver(
        dict(files=[str(path)], min_energy=0.1), TaskConfig(name="gaussian_regression")
    )
    samples = app.prediction_dataset(str(path))
    assert len(samples) == 7 and all(s.target is None for s in samples)
    arrays = root_fixture(tmp_path / "truth.root", size=4)
    for pdgs in ([[211, 211]] * 4, [[15, -15]] * 4):
        arrays[MC_PARTICLE_VARS[0]] = ak.Array(pdgs)
        with pytest.raises(ValueError, match="exactly one generator tau"):
            tau_targets(arrays, np.arange(4))
    arrays[MC_PARTICLE_VARS[0]] = ak.Array([[15, 211]] * 4)
    arrays[ENDPOINT_BRANCHES[0]] = ak.Array([[np.nan, 0.0]] * 4)
    with pytest.raises(ValueError, match="Nonfinite tau targets"):
        tau_targets(arrays, np.arange(4))


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4], [2, 4, 0, 3, 1]])
def test_tau_copy_chain_targets(tmp_path, order):
    """
    Resolve linked tau copies independently of particle ordering and chunk size

    Args:
        tmp_path:  Temporary ROOT directory
        order:  Particle record permutation
    Returns:
        None: Terminal momentum and full production-to-decay displacement agree
    """
    path = tmp_path / "copies.root"
    arrays = root_fixture(path, size=4)
    inverse = np.argsort(order)
    columns = [
        [15, 15, 15, 211, 15],
        [3, 3, 2, 1, 0],
        [0, 0, 0, 0, 1 << 30],
        [10, 20, 30, 1, 99],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    for key, values in zip(MC_PARTICLE_VARS, columns):
        arrays[key] = ak.Array([np.asarray(values)[order].tolist()] * 4)
    links = [[], [0], [1], [2], [2]]
    begin, end, parents = [], [], []
    for old in order:
        begin.append(len(parents))
        parents.extend(int(inverse[parent]) for parent in links[old])
        end.append(len(parents))
    for key, values in zip(MC_RELATION_VARS, [begin, end, parents]):
        arrays[key] = ak.Array([values] * 4)
    for axis, (vertex, endpoint) in enumerate(zip(VERTEX_BRANCHES, ENDPOINT_BRANCHES)):
        arrays[vertex] = ak.Array(
            [(np.array([10, 20, 30, 0, 0]) * (axis + 1))[order].tolist()] * 4
        )
        arrays[endpoint] = ak.Array(
            [(np.array([11, 22, 45, 0, 99]) * (axis + 1))[order].tolist()] * 4
        )
    expected = np.array([30, 0, 0, 35, 70, 105], dtype=np.float32)
    np.testing.assert_allclose(
        tau_targets(arrays, np.arange(4)), np.tile(expected, (4, 1))
    )
    with uproot.recreate(path) as root:
        root.mktree("events", arrays)
    for step in (1, 4):
        chunks = list(
            iter_tau3_regression(
                str(path), "events", chunk_size=step, max_events=4, max_hits=2, e_min=0.1
            )
        )
        targets = np.concatenate([chunk.targets for chunk in chunks])
        np.testing.assert_allclose(targets, np.tile(expected, (4, 1)))
        np.testing.assert_array_equal(
            np.concatenate([chunk.entries for chunk in chunks]), [0, 1, 2, 3]
        )


@pytest.mark.parametrize("parents", [[0, 0], [1, 0], [0, 99]])
def test_invalid_tau_copy_links(tmp_path, parents):
    """
    Reject branching, cyclic and invalid parent links without arbitrary labels

    Args:
        tmp_path:  Temporary ROOT directory
        parents:  Ambiguous or malformed parent indices
    Returns:
        None: The error identifies the original ROOT entry
    """
    arrays = root_fixture(tmp_path / "invalid.root", size=1)
    arrays[MC_PARTICLE_VARS[0]] = ak.Array([[15, 15]])
    arrays[MC_RELATION_VARS[0]] = ak.Array([[0, 1]])
    arrays[MC_RELATION_VARS[1]] = ak.Array([[1, 2]])
    arrays[MC_RELATION_VARS[2]] = ak.Array([parents])
    with pytest.raises(ValueError, match="ROOT entries.*123"):
        tau_targets(arrays, np.array([123]))


@pytest.mark.parametrize("charge", [1, -1])
@pytest.mark.parametrize("order", [list(range(7)), [4, 6, 0, 3, 2, 5, 1]])
@pytest.mark.parametrize("spectator_radiation", [False, True])
def test_tau_pair_with_one_stored_decay(tmp_path, charge, order, spectator_radiation):
    """
    Select the recorded pion-neutrino decay from the inspected ALLEGRO tau pair

    Args:
        tmp_path:  Temporary ROOT directory
        charge:  Charge conjugation of the stored tau decay
        order:  Particle record permutation
        spectator_radiation:  Attach a generator photon to the spectator tau
    Returns:
        None: Correct momentum and zero stored displacement survive ROOT streaming
    """
    path = tmp_path / "pair.root"
    arrays = root_fixture(path, size=1)
    momentum = [-7.712581274860113, 33.34089368695029, -29.60458014248808]
    columns = [
        [11, -11, 22, charge * 15, -charge * 15, charge * 16, -charge * 211],
        [4, 4, 1, 2, 2, 1, 1],
        [0, 0, 0, 0, 0, 0x2000000, 0x5000000],
        [0, 0, 0, momentum[0], 7.712578, -6.51833, -1.19424],
        [0, 0, 0, momentum[1], -33.340920, 24.946735, 8.394158],
        [45.6, -45.6, 0, momentum[2], 29.985545, -22.00382, -7.60075],
    ]
    for key, values in zip(MC_PARTICLE_VARS, columns):
        arrays[key] = ak.Array([np.asarray(values)[order].tolist()])
    inverse = np.argsort(order)
    links = [[], [], [0, 1], [0, 1], [0, 1], [3], [3]]
    if spectator_radiation:
        links[2] = [4]
    begin, end, parents = [], [], []
    for old in order:
        begin.append(len(parents))
        parents.extend(int(inverse[parent]) for parent in links[old])
        end.append(len(parents))
    for key, values in zip(MC_RELATION_VARS, [begin, end, parents]):
        arrays[key] = ak.Array([values])
    for vertex, endpoint in zip(VERTEX_BRANCHES, ENDPOINT_BRANCHES):
        arrays[vertex] = ak.Array([[0.0] * 7])
        arrays[endpoint] = ak.Array(
            [np.array([0, 0, 0, 0, 0, 25000, 2800], float)[order].tolist()]
        )
    expected = np.array([momentum + [0, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(tau_targets(arrays, np.array([0])), expected)
    batch = {}
    for key in (
        MC_PARTICLE_VARS + MC_RELATION_VARS + list(VERTEX_BRANCHES + ENDPOINT_BRANCHES)
    ):
        original = ak.to_list(arrays[key][0])
        if key == MC_RELATION_VARS[2]:
            padded = original
        else:
            fill = len(parents) if key in MC_RELATION_VARS[:2] else 0
            padded = original + [fill] * 20
        batch[key] = [original, padded, original]
    selected = ak.Array(batch)[np.array([False, True, True])]
    np.testing.assert_allclose(
        tau_targets(selected, np.array([16, 24])), np.repeat(expected, 2, axis=0)
    )
    with uproot.recreate(path) as root:
        root.mktree("events", arrays)
    chunks = list(
        iter_tau3_regression(
            str(path), "events", max_hits=2, max_events=1, fiducial_eta_max=2.0
        )
    )
    np.testing.assert_allclose(chunks[0].targets, expected)
    # Move one daughter to the spectator tau to make the association ambiguous
    modified = parents.copy()
    modified[begin[int(inverse[6])]] = int(inverse[4])
    arrays[MC_RELATION_VARS[2]] = ak.Array([modified])
    with pytest.raises(
        ValueError, match="2 tau chains have non-photon generator daughters"
    ):
        tau_targets(arrays, np.array([0]))


def test_two_recorded_tau_decays_remain_ambiguous(tmp_path):
    """
    Reject a tau pair with two pion-neutrino decays

    Args:
        tmp_path:  Temporary ROOT directory
    Returns:
        None: Both decay PDG sets appear in the association error
    """
    arrays = root_fixture(tmp_path / "two_decays.root", size=1)
    columns = [
        [15, -15, 16, -211, -16, 211],
        [2, 2, 1, 1, 1, 1],
        [0] * 6,
        [1.0] * 6,
        [0.0] * 6,
        [0.0] * 6,
    ]
    for key, values in zip(MC_PARTICLE_VARS, columns):
        arrays[key] = ak.Array([values])
    for key, values in zip(
        MC_RELATION_VARS, [[0, 0, 0, 1, 2, 3], [0, 0, 1, 2, 3, 4], [0, 0, 1, 1]]
    ):
        arrays[key] = ak.Array([values])
    for key in VERTEX_BRANCHES + ENDPOINT_BRANCHES:
        arrays[key] = ak.Array([[0.0] * 6])
    with pytest.raises(
        ValueError, match="2 tau chains have non-photon generator daughters"
    ) as error:
        tau_targets(arrays, np.array([16]))
    assert "[-211, 16]" in str(error.value)
    assert "[-16, 211]" in str(error.value)


def test_truth_inspection_without_target_selection(tmp_path, capsys):
    """
    Inspect ambiguous tau records without invoking target selection or hit loading

    Args:
        tmp_path:  Temporary ROOT and configuration directory
        capsys:  JSON output capture
    Returns:
        None: Candidate statuses, parent links and kinematics are reported
    """
    from drivers.tau3_regression.inspect_truth import main, describe_event

    path = tmp_path / "inspect.root"
    arrays = root_fixture(path, size=1)
    arrays[MC_PARTICLE_VARS[0]] = ak.Array([[15, -15]])
    arrays[MC_PARTICLE_VARS[1]] = ak.Array([[2, 0]])
    with uproot.recreate(path) as root:
        root.mktree("events", arrays)
    config = tmp_path / "config.yml"
    config.write_text(f"driver:\n  files: [{path}]\n  tree: events\n")
    main(["--config", str(config), "--entries", "0"])
    result = json.loads(capsys.readouterr().out)
    event = result["events"][0]
    assert event["tau_indices"] == [0, 1]
    assert [p["generator_status"] for p in event["particles"]] == [2, 0]
    assert event["particles"][0]["children"] == [1]
    assert event["particles"][1]["parents"] == [0]
    assert event["particles"][0]["vertex_mm"] == [10, 20, 30]
    assert not event["relation_errors"]
    assert event["target"] is None and event["selection_error"]
    with pytest.raises(ValueError, match="inspect.root.*unambiguous decay"):
        list(iter_tau3_regression(str(path), "events", max_hits=2, max_events=1))
    good = tmp_path / "valid.root"
    root_fixture(good, size=1)
    config.write_text(f"driver:\n  files: [{path}, {good}]\n  tree: events\n")
    capsys.readouterr()
    main(["--config", str(config), "--entries", "0", "--all-files"])
    reports = json.loads(capsys.readouterr().out)["files"]
    assert [r["input"] for r in reports] == [str(path), str(good)]
    assert reports[0]["events"][0]["selection_error"]
    assert reports[1]["events"][0]["selection_error"] is None
    assert reports[1]["events"][0]["target"] == [1, -1, 0.5, 0, 0, 0]
    arrays[MC_RELATION_VARS[2]] = ak.Array([[99]])
    malformed = describe_event(arrays, 123)
    assert malformed["entry"] == 123
    assert "invalid parent index 99" in malformed["relation_errors"][0]


def test_tau_coordinates_and_wrapped_residuals(tmp_path):
    """
    Keep zero displacement regular and compare azimuth across its branch cut
    """
    values = observables(
        np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [3.0, 4.0, 5.0, 1.0, 2.0, 2.0]])
    )
    assert np.isnan(values[0, 1:3]).all() and values[0, -1] == 0
    np.testing.assert_allclose(values[1, [0, 1, 6]], [5, np.arcsinh(1), 3])
    data = dict(
        targets=np.array([[-1.0, 1e-5, 0.0, 0.0, 0.0, 0.0]]),
        mean=np.array([[-1.0, -1e-5, 0.0, 0.0, 0.0, 0.0]]),
        weights=np.ones(1),
    )
    evaluate_extra(data, tmp_path)
    metrics = json.loads((tmp_path / "tau3_metrics.json").read_text())
    assert metrics["observables"]["phi"]["rmse"] == pytest.approx(2e-5)
    assert metrics["vertex"]["leading_mean_distance_mm"] == 0


@pytest.mark.parametrize("family", ["transformer", "twistformer", "pelican"])
def test_tau3_regression_train_reload_evaluate_predict(tmp_path, family):
    """
    Train ROOT hits and restore joint hypotheses in evaluation and fresh CLI inference

    Args:
        tmp_path:  Temporary ROOT data and run directory
        family:  Regression model family
    """
    path = tmp_path / "truth.root"
    root_fixture(path)
    cfg = load_config(
        f"configs/tau3_regression/{family}.yml",
        [
            f'driver.files=["{path}"]',
            "driver.min_energy=0.1",
            # The reader fixture contains a hit at the coordinate origin
            "driver.proxy_origin=[0.0,0.0,-1000.0]",
            "model.width=8",
            "model.heads=2",
            "model.depth=1",
            "model.head_layers=1" if family == "transformer" else "model.head_widths=[8]",
            "training.device=cpu",
            "training.compile=false",
            "training.workers=0",
            "training.epochs=1",
            "training.warmup_epochs=0",
            "training.batch_size=8",
            "training.eval_batch_size=8",
        ],
    )
    experiment = build_experiment(cfg)
    assert experiment.driver.task_spec.outputs == (
        "px",
        "py",
        "pz",
        "dx",
        "dy",
        "dz",
    )
    checkpoint = Trainer(tmp_path / "run").fit(experiment)
    predictor = load_predictor(checkpoint, "cpu")
    samples = predictor.experiment.driver.dataset("test")
    prediction = predictor.predict(samples)
    assert prediction.component_mean.shape == (len(samples), 5, 6)
    assert (
        torch.isfinite(prediction.component_std).all()
        and (prediction.component_std > 0).all()
    )
    data, _ = collect_predictions(predictor, samples, require_targets=True)
    np.testing.assert_allclose(
        data["component_mean"], prediction.component_mean.numpy(), atol=1e-6
    )
    assert data["component_observables"].shape == (len(samples), 5, 7)
    result = evaluate(predictor, samples, tmp_path / "evaluation", plots=True)
    assert np.isfinite(result["mixture"]["joint_nll"])
    assert (tmp_path / "evaluation/tau3_metrics.json").is_file()
    assert (tmp_path / "evaluation/plots/tau3_mixture_pit.png").is_file()
    hits = tmp_path / "hits.root"
    root_fixture(hits, truth=False, size=5)
    output = tmp_path / "predictions.npz"
    subprocess.run(
        [
            sys.executable,
            "scripts/predict.py",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(hits),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    with np.load(output) as exported:
        assert len(exported["ids"]) == 4
        assert exported["component_mean"].shape == (4, 5, 6)
        assert "targets" not in exported and "leading_observables" in exported
        np.testing.assert_allclose(
            exported["mixture_probabilities"].sum(-1), 1, atol=1e-6
        )
