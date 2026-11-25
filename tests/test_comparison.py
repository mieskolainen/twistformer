# Common-event classifier comparison validation
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from twistformer import build_experiment, parse_config, load_predictor
from twistformer.api.contracts import TaskSpec
from twistformer.evaluation.comparison import (
    align_predictions,
    compare,
    comparison_config,
    ModelEvents,
)
from twistformer.evaluation.comparison_stats import paired_metrics
from twistformer.evaluation.runner import collect_predictions, evaluate
from twistformer.training.checkpoint import VERSION
from twistboard.server import list_comparisons, comparison_detail


def checkpoint(path, family="transformer", outputs=1):
    """
    Save a small fitted classifier for comparison workflow checks

    Args:
        path: Checkpoint output path
        family: Model family
        outputs: Binary logit count
    Returns:
        path: Saved checkpoint
    """
    cfg = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 20,
                "test_size": 12,
                "max_tokens": 4,
            },
            "model": {
                "family": family,
                "width": 8,
                "heads": 2,
                "depth": 1,
                "dropout": 0.0,
            },
            "task": {"balance_classes": True},
        }
    )
    exp = build_experiment(cfg)
    if outputs == 2:
        exp.model.head.linear = torch.nn.Linear(exp.model.head.linear.in_features, 2)
    pre = exp.driver.create_preprocessor(config=cfg.preprocessing)
    pre.fit(exp.driver.dataset("train"), class_count=2)
    torch.save(
        {
            "schema_version": VERSION,
            "config": cfg.to_dict(),
            "input_spec": asdict(exp.driver.input_spec),
            "task_spec": asdict(exp.driver.task_spec),
            "model": exp.model.state_dict(),
            "preprocessing": pre.state,
        },
        path,
    )
    return path


def test_comparison_workflow_and_reuse(tmp_path):
    """
    Compare mixed binary heads with bins, paired intervals and cached predictions

    Args:
        tmp_path: Temporary dataset and result directory
    Returns:
        None: Files and dashboard records agree across fresh and cached runs
    """
    first = checkpoint(tmp_path / "a.pt")
    second = checkpoint(tmp_path / "b.pt", "pelican", 2)
    card = tmp_path / "compare.yml"
    card.write_text(
        yaml.safe_dump(
            {
                "comparison": {
                    "name": "test",
                    "dataset": {"reference_checkpoint": str(first)},
                    "models": [
                        {"label": "A", "checkpoint": str(first)},
                        {"label": "B", "checkpoint": str(second)},
                    ],
                    "bootstrap_samples": 3,
                    "device": "cpu",
                    "plots": True,
                },
                "evaluation": {
                    "batch_size": 4,
                    "efficiencies": [0.3, 0.5],
                    "categories": [
                        {
                            "name": "energy",
                            "observable": "energy",
                            "edges": [-100, 0, 100],
                        }
                    ],
                },
            }
        )
    )
    doc = comparison_config(card)
    output = compare(doc, tmp_path / "results")
    record = json.loads((output / "comparison.json").read_text())
    assert record["status"] == "completed"
    assert record["events"] == 12
    assert len(record["selections"]) == 3
    assert record["selections"][1]["status"] == "unavailable"
    assert list((output / "selections/inclusive/plots").glob("*.pdf"))
    selection = comparison_detail(tmp_path, "results", "inclusive")["selection"]
    assert len(selection["models"]) == 2
    assert selection["paired"]["A"]["A"]["auc:high"]["estimate"] == 0
    assert selection["paired"]["A"]["A"]["auc:high"]["lower"] == 0
    assert list_comparisons(tmp_path)[0]["id"] == "results"
    with np.load(output / "models/model_000/predictions.npz") as data:
        np.testing.assert_equal(data["weights"], np.ones(12))
        np.testing.assert_equal(data["physical_weights"], np.ones(12))
    doc["comparison"]["models"][1] = {
        "label": "B",
        "evaluation": str(output / "models/model_001"),
    }
    doc["comparison"]["plots"] = False
    reused = compare(doc, tmp_path / "reused")
    cached = json.loads((reused / "selections/inclusive/comparison.json").read_text())
    assert cached["paired"] == selection["paired"]
    assert cached["models"] == selection["models"]
    with pytest.raises(ValueError, match="empty"):
        compare(doc, output)
    with pytest.raises(ValueError):
        comparison_detail(tmp_path, "../escape")
    with pytest.raises(ValueError):
        comparison_detail(tmp_path, "results", "../escape")
    doc["comparison"]["dataset"]["split"] = "val"
    with pytest.raises(ValueError, match="dataset does not match"):
        compare(doc, tmp_path / "wrong-cache")
    failed = json.loads((tmp_path / "wrong-cache/comparison.json").read_text())
    assert failed["status"] == "failed"


def test_alignment_class_order_and_rejections():
    """
    Match event IDs and class names while rejecting incompatible physics samples
    """
    target = TaskSpec("classification", ("background", "signal"), signal_class=1)
    source = TaskSpec("classification", ("signal", "background"), signal_class=0)
    common = {
        "ids": np.array(["a", "b"]),
        "targets": np.array([0, 1]),
        "weights": np.array([2.0, 3.0]),
    }
    data = {
        "ids": np.array(["b", "a"]),
        "targets": np.array([0, 1]),
        "physical_weights": np.array([3.0, 2.0]),
        "probabilities": np.array([[0.8, 0.2], [0.1, 0.9]]),
        "logits": np.array([[-2.0], [3.0]]),
    }
    aligned = align_predictions(data, common, source, target)
    np.testing.assert_allclose(aligned["probabilities"], [[0.9, 0.1], [0.2, 0.8]])
    np.testing.assert_equal(aligned["logits"], [[-3.0], [2.0]])
    for key, replacement, message in [
        ("ids", np.array(["a", "a"]), "event IDs"),
        ("targets", np.array([1, 0]), "labels"),
        ("physical_weights", np.array([3.0, 7.0]), "weights"),
    ]:
        with pytest.raises(ValueError, match=message):
            align_predictions({**data, key: replacement}, common, source, target)


def test_physical_weights_are_separate_from_balancing(tmp_path):
    """
    Keep physics metrics independent of fitted training balance factors

    Args:
        tmp_path: Temporary checkpoint and evaluation directory
    Returns:
        None: Physics metrics retain the raw event weights
    """
    path = checkpoint(tmp_path / "model.pt")
    predictor = load_predictor(path, "cpu")
    samples = [
        replace(sample, weight=float(i + 1))
        for i, sample in enumerate(predictor.experiment.driver.dataset("test"))
    ]
    data, loss = collect_predictions(
        predictor, samples, batch_size=4, require_targets=True
    )
    np.testing.assert_equal(data["physical_weights"], np.arange(1, 13))
    assert np.isfinite(loss)
    assert not np.allclose(data["weights"], data["physical_weights"])
    evaluate(
        predictor,
        samples,
        tmp_path / "evaluation",
        batch_size=4,
        plots=False,
        bootstrap_samples=0,
    )
    with np.load(tmp_path / "evaluation/predictions.npz") as arrays:
        np.testing.assert_equal(arrays["weights"], data["physical_weights"])
        np.testing.assert_equal(arrays["loss_weights"], data["weights"])


def test_example_card_and_strict_steering(tmp_path):
    """
    Resolve the shipped card and reject ambiguous model selection

    Args:
        tmp_path: Temporary invalid card directory
    Returns:
        None: Shared evaluation settings and strict validation are enforced
    """
    doc = comparison_config("configs/comparisons/toptag.yml")
    assert doc["evaluation"]["batch_size"] == 2048
    assert doc["evaluation"]["categories"][0]["observable"] == "jet_pt"
    doc["comparison"]["models"][1]["label"] = "PELICAN"
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="unique"):
        comparison_config(path)


@pytest.mark.parametrize("feature_set, width", [("kinematics15", 15), ("local_graphs", 35)])
def test_toptag_views_preserve_sources_and_model_inputs(tmp_path, feature_set, width):
    """
    Read identical HDF5 events with different feature and particle budgets

    Args:
        tmp_path: Temporary HDF5 directory
        feature_set: Constituent or jet graph inputs
        width: Expected feature count
    Returns:
        None: IDs, labels and physics weights agree across model views
    """
    h5py = pytest.importorskip("h5py")
    from drivers.toptag.driver import Driver
    from twistformer.api.config import TaskConfig

    path = tmp_path / "jets.h5"
    names = [f"{name}_{i}" for name in ("E", "PX", "PY", "PZ") for i in range(200)] + [
        "is_signal_new",
        "weight",
    ]
    values = np.zeros((6, len(names)), dtype=np.float32)
    values[:, :3] = [10, 6, 2]
    values[:, 200:203] = [8, 4, 1]
    values[:, -2] = np.arange(6) % 2
    values[:, -1] = np.arange(1, 7)
    with h5py.File(path, "w") as store:
        group = store.create_group("table")
        group["axis1"] = np.arange(6)
        group["block0_items"] = np.asarray(names, dtype="S")
        group["block0_values"] = values
    reference = Driver(
        {"files": {"test": str(path)}, "max_particles": 3, "weight_column": "weight"},
        TaskConfig(),
    )
    model = Driver(
        {
            "files": {"test": "unused.h5"},
            "max_particles": 1,
            "feature_set": feature_set,
            "max_events": 2,
            "weight_column": "weight",
        },
        TaskConfig(),
    )
    original = reference.dataset("test")
    view = model.comparison_dataset(reference, "test")
    assert len(view) == len(original) == 6
    for i in range(6):
        assert original[i].id == view[i].id
        assert original[i].target == view[i].target
        assert original[i].weight == view[i].weight
        assert original[i].observables == view[i].observables
        assert view[i].features.shape == (1, width)
        assert original[i].features.shape == (3, 4)


def test_tau3_views_keep_proxy_and_class_order(monkeypatch):
    """
    Keep common tau events with model geometry and reordered classifier labels

    Args:
        monkeypatch: Replace ROOT IO with compact hit chunks
    Returns:
        None: Common events map to the saved class order
    """
    from drivers.tau3_classification.driver import Driver
    from drivers.tau3_classification.reader import Tau3Chunk
    from twistformer.api.config import TaskConfig

    def read(self, path, apply_fiducial=True):
        """
        Supply deterministic nonempty calorimeter events

        Args:
            self: Tau driver
            path: Source class file
            apply_fiducial: Reader selection switch
        Returns:
            chunks: Fixed hit geometry and source entries
        """
        yield Tau3Chunk(
            features=np.tile([[1.0, 2.0, 3.0, 4.0]], (20, 1)).astype(np.float32),
            offsets=np.arange(21),
            weights=np.ones(20),
            entries=np.arange(20),
        )

    monkeypatch.setattr(Driver, "_read", read)
    reference = Driver(
        {"files": ["a.root", "b.root", "c.root"], "class_names": ["a", "b", "c"]},
        TaskConfig(),
    )
    model = Driver(
        {
            "files": ["c.root", "a.root", "b.root"],
            "class_names": ["c", "a", "b"],
            "momentum_proxy": True,
            "seed": 999,
        },
        TaskConfig(),
    )
    original = reference.dataset("test")
    view = ModelEvents(
        model.comparison_dataset(reference, "test"),
        reference.task_spec,
        model.task_spec,
    )
    assert len(view) == len(original)
    for index in range(len(view)):
        assert view[index].id == original[index].id
        assert view[index].p4 is not None
        assert original[index].p4 is None
        assert (
            model.task_spec.outputs[int(view[index].target)]
            == reference.task_spec.outputs[int(original[index].target)]
        )
    incompatible = Driver(
        {
            "files": ["a.root", "b.root", "c.root"],
            "class_names": ["a", "b", "c"],
            "min_energy": 0.1,
        },
        TaskConfig(),
    )
    with pytest.raises(ValueError, match="min_energy"):
        incompatible.comparison_dataset(reference, "test")


def test_paired_multiclass_bootstrap_and_undefined_rejection():
    """
    Identical multiclass scores have zero paired AUC uncertainty
    """
    spec = TaskSpec("classification", ("a", "b", "c"))
    data = {
        "targets": np.tile(np.arange(3), 8),
        "weights": np.arange(1, 25, dtype=float),
    }
    probabilities = np.eye(3)[data["targets"]] * 0.7 + 0.1
    settings = {
        "bootstrap_samples": 8,
        "bootstrap_seed": 42,
        "bootstrap_confidence": 0.68,
    }
    paired = paired_metrics(
        data, [probabilities, probabilities], spec, ["A", "B"], [0.3, 0.5], settings
    )
    for name in spec.outputs:
        interval = paired["A"]["B"][f"auc:{name}"]
        assert interval["estimate"] == interval["lower"] == interval["upper"] == 0
        assert interval["valid_resamples"] == 8
        assert paired["A"]["B"][f"rejection:{name}:0.3"]["estimate"] is None


def test_multiclass_cli_with_reordered_classes(tmp_path):
    """
    Compare multiclass checkpoints through the public CLI with class alignment

    Args:
        tmp_path: Temporary NPZ data, checkpoints and comparison card
    Returns:
        None: CLI files use the common class order
    """
    from twistformer.cli.compare import main

    source = tmp_path / "events.npz"
    labels = np.tile(np.arange(3), 4)
    np.savez(
        source,
        features=np.random.default_rng(7).normal(size=(12, 3, 2)).astype(np.float32),
        targets=labels,
    )
    paths = []
    for index, names in enumerate([["a", "b", "c"], ["c", "a", "b"]]):
        cfg = parse_config(
            {
                "driver": {
                    "name": "generic",
                    "files": {"test": str(source)},
                    "features": ["x", "y"],
                    "outputs": names,
                },
                "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
            }
        )
        exp = build_experiment(cfg)
        pre = exp.driver.create_preprocessor(config=cfg.preprocessing)
        pre.fit(exp.driver.dataset("test"))
        path = tmp_path / f"multi_{index}.pt"
        torch.save(
            {
                "schema_version": VERSION,
                "config": cfg.to_dict(),
                "input_spec": asdict(exp.driver.input_spec),
                "task_spec": asdict(exp.driver.task_spec),
                "model": exp.model.state_dict(),
                "preprocessing": pre.state,
            },
            path,
        )
        paths.append(path)
    doc = {
        "comparison": {
            "name": "multiclass",
            "dataset": {"reference_checkpoint": str(paths[0])},
            "models": [
                {"label": str(i), "checkpoint": str(path)}
                for i, path in enumerate(paths)
            ],
            "bootstrap_samples": 2,
            "plots": False,
            "device": "cpu",
        }
    }
    card = tmp_path / "multiclass.yml"
    card.write_text(yaml.safe_dump(doc))
    main(["--config", str(card), "--output", str(tmp_path / "result")])
    result = json.loads(
        (tmp_path / "result/selections/inclusive/comparison.json").read_text()
    )
    assert set(result["models"][0]["metrics"]["classes"]) == {"a", "b", "c"}
    expected = load_predictor(paths[1], "cpu").predict_prob(
        build_experiment(cfg).driver.dataset("test")
    )
    with np.load(tmp_path / "result/models/model_001/predictions.npz") as saved:
        np.testing.assert_allclose(
            saved["probabilities"], expected.numpy()[:, [1, 2, 0]], atol=1e-6
        )
        np.testing.assert_equal(saved["targets"], labels)
    dataset = json.loads((tmp_path / "result/dataset.json").read_text())
    assert dataset["sources"][0]["sha256"]
    assert (tmp_path / "result/selections/inclusive/paired.csv").is_file()


def test_comparison_http_routes_and_source_changes(tmp_path):
    """
    Serve contained comparison records and reject modified source files

    Args:
        tmp_path: Temporary dashboard root
    Returns:
        None: Routes serve comparison data without permitting path escape
    """
    from types import SimpleNamespace
    from twistboard.server import Handler
    from twistformer.evaluation.comparison import verify_sources

    source = tmp_path / "input.npz"
    source.write_bytes(b"first")
    stats = {str(source): source.stat()}
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        verify_sources(stats)
    comparison = tmp_path / "comparisons/example"
    comparison.mkdir(parents=True)
    (comparison / "config.json").write_text("{}")
    (comparison / "comparison.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "example",
                "status": "completed",
                "selections": [],
            }
        )
    )
    responses = []
    handler = SimpleNamespace(
        root=tmp_path,
        path="/api/comparisons",
        send_json=responses.append,
        send_error=responses.append,
    )
    Handler.do_GET(handler)
    assert responses[-1][0]["id"] == "comparisons/example"
    handler.path = "/api/comparisons/comparisons%2Fexample"
    Handler.do_GET(handler)
    assert responses[-1]["name"] == "example"
    handler.path = "/api/comparisons/..%2Fescape"
    Handler.do_GET(handler)
    assert responses[-1] == 404


def test_regression_comparison_with_mixture_and_masks(tmp_path):
    """
    Compare Gaussian and mixture regressors in physical units with missing targets

    Args:
        tmp_path: Temporary regression data and checkpoints
    Returns:
        None: Shared metrics, paired differences, plots and cached outputs agree
    """
    rng = np.random.default_rng(17)
    targets = rng.normal(size=(12, 2)).astype(np.float32)
    targets[0, 0] = np.nan
    mask = np.ones_like(targets, dtype=bool)
    mask[1, 1] = False
    source = tmp_path / "regression.npz"
    np.savez(
        source,
        features=rng.normal(size=(12, 3, 2)).astype(np.float32),
        targets=targets,
        target_mask=mask,
    )
    paths = []
    for index, names in enumerate([["x", "y"], ["y", "x"]]):
        cfg = parse_config(
            {
                "driver": {
                    "name": "generic",
                    "files": {"test": str(source)},
                    "features": ["a", "b"],
                    "outputs": names,
                    "target_units": ["GeV", "GeV"],
                },
                "task": {
                    "name": "gaussian_regression",
                    "mixture_components": index + 1,
                },
                "model": {"width": 8, "heads": 2, "depth": 1, "dropout": 0.0},
            }
        )
        exp = build_experiment(cfg)
        pre = exp.driver.create_preprocessor(config=cfg.preprocessing)
        pre.fit(exp.driver.dataset("test"), regression=True)
        path = tmp_path / f"reg_{index}.pt"
        torch.save(
            {
                "schema_version": VERSION,
                "config": cfg.to_dict(),
                "input_spec": asdict(exp.driver.input_spec),
                "task_spec": asdict(exp.driver.task_spec),
                "model": exp.model.state_dict(),
                "preprocessing": pre.state,
            },
            path,
        )
        paths.append(path)
    card = tmp_path / "regression.yml"
    card.write_text(
        yaml.safe_dump(
            {
                "comparison": {
                    "name": "regression",
                    "dataset": {"reference_checkpoint": str(paths[0])},
                    "models": [
                        {"label": str(i), "checkpoint": str(path)}
                        for i, path in enumerate(paths)
                    ],
                    "bootstrap_samples": 3,
                    "plots": True,
                    "device": "cpu",
                }
            }
        )
    )
    output = compare(comparison_config(card), tmp_path / "regression-result")
    result = json.loads((output / "selections/inclusive/comparison.json").read_text())
    assert result["models"][0]["metrics"]["x"]["rmse"] > 0
    assert "mixture" in result["models"][1]["metrics"]
    assert result["paired"]["0"]["0"]["rmse:x"]["upper"] == 0
    assert result["paired"]["0"]["1"]["nll:x"]["valid_resamples"] == 3
    assert "residual" in result["models"][0]["curves"]["0"]
    assert list((output / "selections/inclusive/plots").glob("residual*.pdf"))
    with np.load(output / "models/model_001/predictions.npz") as saved:
        np.testing.assert_allclose(saved["targets"], targets)
        np.testing.assert_array_equal(saved["target_mask"], mask & np.isfinite(targets))
        assert saved["component_mean"].shape == (12, 2, 2)
    doc = comparison_config(card)
    doc["comparison"]["models"][1] = {
        "label": "1",
        "evaluation": str(output / "models/model_001"),
    }
    doc["comparison"]["plots"] = False
    cached = compare(doc, tmp_path / "regression-cached")
    assert (
        json.loads((cached / "selections/inclusive/comparison.json").read_text())[
            "paired"
        ]
        == result["paired"]
    )


def test_regression_schema_and_point_comparison():
    """
    Permit point and Gaussian predictions for the same four-vector targets
    """
    from twistformer.evaluation.comparison import validate_schemas
    from twistformer.evaluation.comparison_stats import scores

    point = TaskSpec("covariant_regression", ("E", "px", "py", "pz"), ("GeV",) * 4)
    gaussian = replace(point, name="gaussian_regression")
    validate_schemas(point, gaussian)
    with pytest.raises(ValueError, match="units"):
        validate_schemas(point, replace(gaussian, units=("MeV",) * 4))
    with pytest.raises(ValueError, match="task"):
        validate_schemas(point, TaskSpec("classification", point.outputs))
    data = {"targets": np.ones((4, 4)), "weights": np.ones(4)}
    result = scores(data, {"mean": np.zeros((4, 4))}, point, [0.3, 0.5])
    assert result["rmse:E"] == 1
    assert np.isnan(result["nll:E"])


def test_paired_flow_with_sparse_signed_weights():
    """Use physical yields for paired flow bias and density uncertainties"""
    spec = TaskSpec("flow_nll", ("E",), ("GeV",))
    targets = np.arange(4, dtype=float)[:, None]
    data = {"targets": targets, "weights": np.array([2., -1., 0., 0.])}
    prediction = {"samples": targets[:, None], "log_prob": np.zeros(4)}
    settings = {"bootstrap_samples": 32, "bootstrap_seed": 17, "bootstrap_confidence": 0.68}
    result = paired_metrics(data, [prediction], spec, ["model"], [], settings)["model"]["model"]
    assert result["density_nll"]["estimate"] is None
    assert result["density_nll"]["valid_resamples"] == 0
    assert result["bias:E"]["estimate"] == 0
    assert 0 < result["bias:E"]["valid_resamples"] < 32


def test_dashboard_starts_real_comparison_job(tmp_path):
    """
    Run a selected-checkpoint comparison through the dashboard job launcher

    Args:
        tmp_path: Temporary experiment root
    Returns:
        None: The child process creates a completed comparison
    """
    from twistboard.comparisons import start_comparison, _jobs

    models = []
    for name in ("first", "second"):
        directory = tmp_path / name
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "experiment.json").write_text("{}")
        path = checkpoint(directory / "checkpoints/best.pt")
        models.append(
            {"label": name, "checkpoint": path.relative_to(tmp_path).as_posix()}
        )
    job = start_comparison(tmp_path, {"models": models})
    process = _jobs[str(tmp_path.resolve())]
    with pytest.raises(RuntimeError, match="already running"):
        start_comparison(tmp_path, {"models": models})
    assert process.wait(timeout=60) == 0
    record = json.loads((tmp_path / job["id"] / "comparison.json").read_text())
    assert record["status"] == "completed"
    assert len(record["models"]) == 2
    with pytest.raises(ValueError):
        start_comparison(
            tmp_path,
            {
                "models": [
                    {"label": "outside", "checkpoint": "../outside.pt"},
                    models[1],
                ]
            },
        )


def test_dashboard_starts_real_evaluation_job(tmp_path):
    """
    Run a selected checkpoint through the host evaluation CLI

    Args:
        tmp_path:  Temporary experiment root
    """
    from twistboard.evaluations import start_evaluation, _jobs
    from twistboard.server import run_detail

    run = tmp_path / "host-evaluation"
    (run / "checkpoints").mkdir(parents=True)
    (run / "experiment.json").write_text('{"status": "trained"}')
    path = checkpoint(run / "checkpoints/best.pt")
    job = start_evaluation(tmp_path, {"checkpoint": path.relative_to(tmp_path).as_posix()})
    process = _jobs[str(tmp_path.resolve())]
    assert process.wait(timeout=90) == 0
    output = tmp_path / job["id"]
    record = json.loads((output / "evaluation.json").read_text())
    assert record["status"] == "completed"
    assert record["checkpoint"] == str(path)
    assert (output / "metrics.json").is_file()
    assert run_detail(tmp_path, run.name)["evaluations"][0]["metrics"]


def test_comparison_post_requires_same_origin(tmp_path):
    """
    Reject cross-origin job launches before reading a request body

    Args:
        tmp_path: Temporary dashboard root
    Returns:
        None: Untrusted requests receive HTTP 403
    """
    from types import SimpleNamespace
    from twistboard.server import Handler

    responses = []
    handler = SimpleNamespace(
        root=tmp_path,
        path="/api/comparisons",
        headers={
            "X-Twistboard-Action": "compare",
            "Origin": "https://other.example",
            "Host": "localhost:8765",
        },
        send_json=lambda body, status: responses.append((body, status)),
    )
    Handler.do_POST(handler)
    assert responses[-1][1] == 403


def test_comparison_post_launches_selected_models(tmp_path, monkeypatch):
    """
    Forward a same-origin browser selection to the local job launcher

    Args:
        tmp_path: Temporary dashboard root
        monkeypatch: Isolated job launcher
    Returns:
        None: HTTP 202 identifies the launched comparison
    """
    import io
    from types import SimpleNamespace
    from twistboard.server import Handler
    from twistboard import comparisons

    models = [
        {"label": "A", "checkpoint": "a/checkpoints/best.pt"},
        {"label": "B", "checkpoint": "b/checkpoints/best.pt"},
    ]
    calls, responses = [], []

    def launch(root, payload):
        """
        Record a validated dashboard launch request

        Args:
            root: Dashboard root
            payload: Parsed model selection
        Returns:
            job: Queued comparison identity
        """
        calls.append((root, payload))
        return {"id": "comparisons/new", "status": "queued"}

    monkeypatch.setattr(comparisons, "start_comparison", launch)
    body = json.dumps({"models": models}).encode()
    handler = SimpleNamespace(
        root=tmp_path,
        path="/api/comparisons",
        headers={
            "X-Twistboard-Action": "compare",
            "Origin": "http://localhost:8765",
            "Host": "localhost:8765",
            "Content-Length": str(len(body)),
        },
        rfile=io.BytesIO(body),
        send_json=lambda value, status: responses.append((value, status)),
    )
    Handler.do_POST(handler)
    assert calls == [(tmp_path, {"models": models})]
    assert responses == [({"id": "comparisons/new", "status": "queued"}, 202)]
