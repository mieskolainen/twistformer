# Physical category evaluation and file integration
#
# mikael.mieskolainen@cern.ch (2026)

import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from twistformer import build_experiment, parse_config
from twistformer.api.config import EvaluationConfig
from twistformer.api.contracts import Sample, TaskSpec
from twistformer.api.predictor import Predictor
from twistformer.data.batching import collate_samples
from twistformer.data.preprocessing import Preprocessor
from twistformer.evaluation.categories import (
    category_bins,
    evaluate_categories,
    rebin_evaluation,
    validate_categories,
)
from twistformer.evaluation.metrics import classification_metrics
from twistformer.evaluation.runner import evaluate
from twistboard.server import file_list, category_views

SCHEMA = {
    "energy": {"label": "Energy", "unit": "GeV"}
}
CATEGORY = {
    "name": "energy",
    "observable": "energy",
    "edges": [0, 10, 20],
    "underflow": True,
    "overflow": True,
}


def predictions():
    """
    Supply weighted binary events spanning all energy bins

    Returns:
        data: Aligned predictions and a missing observable
    """
    y = np.array([0, 1] * 5)
    score = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.6, 0.4, 0.8, 0.5, 0.7])
    return {
        "ids": np.array([str(i) for i in range(10)]),
        "targets": y,
        "weights": np.arange(1, 11, dtype=float),
        "probabilities": np.stack([1 - score, score], axis=1),
        "observable_energy": np.array([-2, -1, 0, 9, 10, 19, 20, 21, np.nan, np.inf]),
    }


def test_half_open_bins_and_flow():
    """
    Assign each finite event to at most one half-open bin
    """
    bins = list(category_bins(CATEGORY, predictions()["observable_energy"]))
    assert [int(mask.sum()) for _, _, mask in bins] == [2, 2, 2, 2]
    np.testing.assert_array_equal(
        np.sum([mask for _, _, mask in bins], axis=0), [1] * 8 + [0, 0]
    )
    assert bins[2][2][4]
    assert bins[3][2][6]


@pytest.mark.parametrize("signed", [False, True])
def test_category_roc_legends_use_bin_weighted_auc(tmp_path, monkeypatch, signed):
    """
    Show each bin's weighted AUC on individual and overlaid ROC curves

    Args:
        tmp_path:  Category evaluation directory
        monkeypatch:  Capture figure legends without writing images
        signed:  Include a negative weight and preserve the signed ROC AUC
    """
    from twistformer.evaluation import plots

    score = np.array([.1, .8, .5, .9, .6, .2, .7, .3])
    data = {
        "targets": np.array([0, 0, 1, 1] * 2),
        "probabilities": np.column_stack((1 - score, score)),
        "weights": np.array([-.25 if signed else 1, 3, 2, 1, 2, 1, 3, 2]),
        "observable_energy": np.array([5] * 4 + [15] * 4),
    }
    spec = TaskSpec("classification", ("gamma", "pi0"), signal_class=1)
    category = {"name": "energy", "observable": "energy", "edges": [0, 10, 20, 30]}
    legends = {}

    def save(figure, path):
        """
        Capture class and category ROC legends

        Args:
            figure:  Diagnostic figure
            path:  Output figure path
        """
        if path.name.startswith(("roc", "rejection")):
            legends[path.relative_to(tmp_path).as_posix()] = [
                text.get_text() for text in figure.axes[0].get_legend().get_texts()
            ]
            axis = figure.axes[0]
            assert len(axis.collections) == len(axis.lines) - 1

    monkeypatch.setattr(plots, "save_figure", save)
    manifest = evaluate_categories(data, spec, [category], SCHEMA, tmp_path, bootstrap_samples=30)
    entries = manifest["categories"][0]["bins"]
    assert entries[-1]["status"] == "unavailable"
    expected, areas = [], []
    for index, entry in enumerate(entries[:2]):
        mask = data["observable_energy"] == (5 if index == 0 else 15)
        metrics, _ = classification_metrics({key: value[mask] for key, value in data.items()}, spec, bootstrap_samples=0)
        area = metrics["classes"]["pi0"]["auc"]
        areas.append(area)
        expected.append(f"[{index * 10},{(index + 1) * 10}) GeV (AUC = {area:.4f})")
        for extension in ("png", "pdf"):
            assert legends[f"{entry['directory']}/plots/roc.{extension}"] == [
                f"pi0 vs gamma (AUC = {area:.4f})", "random (AUC = 0.5000)"
            ]
    assert areas[0] != areas[1]
    for extension in ("png", "pdf"):
        assert legends[f"categories/energy/plots/roc_class_1.{extension}"] == expected + ["random (AUC = 0.5000)"]


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../escape"},
        {"edges": [1, 1]},
        {"edges": [1, float("inf")]},
        {"observable": "missing"},
        {"overflow": "yes"},
        {"edges": [True, 2]},
        {"unexpected": 1},
    ],
)
def test_invalid_categories(change):
    """
    Reject ambiguous bins and undeclared observables

    Args:
        change:  Invalid category steering override
    """
    with pytest.raises(ValueError):
        validate_categories([{**CATEGORY, **change}], SCHEMA)


def test_metadata_survives_preprocessing_and_batching():
    """
    Keep physical category values separate from learned features
    """
    sample = Sample(
        torch.tensor([[1.0, 2.0], [2.0, 4.0]]), observables={"energy": 123.0}
    )
    preprocessor = Preprocessor().fit([sample])
    prepared = preprocessor.transform(sample)
    assert prepared.observables == sample.observables
    batch = collate_samples([prepared]).to("cpu")
    assert batch.observables["energy"].dtype == torch.float64
    assert batch.observables["energy"].item() == 123
    with pytest.raises(ValueError, match="names must agree"):
        collate_samples([sample, replace(sample, observables={})])


def test_category_metrics_match_manual_subsets(tmp_path):
    """
    Reuse inclusive metric definitions for each weighted event subset

    Args:
        tmp_path:  Temporary evaluation directory
    """
    data = predictions()
    spec = TaskSpec("classification", ("QCD", "top"), signal_class=1)
    manifest = evaluate_categories(
        data, spec, [CATEGORY], SCHEMA, tmp_path, plots=False, efficiencies=[0.3, 0.5]
    )
    group = manifest["categories"][0]
    assert group["missing_events"] == 2
    assert group["out_of_range_events"] == 0
    for entry, (_, _, mask) in zip(
        group["bins"], category_bins(CATEGORY, data["observable_energy"])
    ):
        manual, _ = classification_metrics(
            {key: value[mask] for key, value in data.items()},
            spec,
            efficiencies=[0.3, 0.5],
        )
        actual = json.loads(
            (tmp_path / entry["directory"] / "metrics.json").read_text()
        )
        assert actual == manual
        assert entry["support"]["events"] == 2
        assert not (tmp_path / entry["directory"] / "predictions.npz").exists()
    views = category_views(tmp_path, tmp_path, file_list(tmp_path, tmp_path))
    assert len(views) == 4
    assert all(view["metrics"]["macro_auc"] is not None for view in views)


def test_unavailable_bins_and_signed_weights(tmp_path):
    """
    Report absent support while preserving signed-weight diagnostics

    Args:
        tmp_path:  Temporary evaluation directory
    """
    data = predictions()
    data["weights"][:2] = [1, -1]
    data["observable_energy"][2:4] = 11
    config = {**CATEGORY, "edges": [0, 5, 10, 20, 100, 200]}
    spec = TaskSpec("classification", ("QCD", "top"), signal_class=1)
    manifest = evaluate_categories(data, spec, [config], SCHEMA, tmp_path, plots=False)
    entries = manifest["categories"][0]["bins"]
    assert entries[0]["support"]["sum_weights"] == 0
    assert entries[1]["status"] == "unavailable"
    assert entries[1]["reason"] == "Empty bin"
    assert entries[3]["status"] == "completed"


def test_one_inference_pass_and_rebin_without_checkpoint(tmp_path, monkeypatch):
    """
    Export observables once and reuse them without reconstructing the model

    Args:
        tmp_path:  Temporary evaluation directory
        monkeypatch:  Isolated inference or reader replacement
    """
    config = parse_config(
        {
            "driver": {
                "name": "synthetic",
                "train_size": 16,
                "test_size": 16,
                "max_tokens": 4,
            },
            "model": {"width": 8, "heads": 2, "depth": 1},
            "training": {"device": "cpu"},
            "evaluation": {
                "categories": [
                    {
                        "name": "energy",
                        "observable": "energy",
                        "edges": [0, 5, 20],
                        "overflow": True,
                    }
                ]
            },
        }
    )
    experiment = build_experiment(config)
    preprocessor = experiment.driver.create_preprocessor().fit(
        experiment.driver.dataset("train")
    )
    predictor = Predictor(experiment, preprocessor, "cpu")
    calls = []
    original = predictor.forward

    def forward(batch):
        """
        Count model batches without changing their predictions

        Args:
            batch:  Prepared event batch
        """
        calls.append(len(batch.ids))
        return original(batch)

    monkeypatch.setattr(predictor, "forward", forward)
    source = tmp_path / "source"
    samples = experiment.driver.dataset("test")
    inclusive = evaluate(
        predictor, samples, source, batch_size=4, plots=False, bootstrap_samples=0
    )
    assert calls == [4, 4, 4, 4]
    assert "preprocessed_loss" in inclusive
    assert (source / "observables.json").exists()
    (source / "evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": "/unavailable/best.pt",
                "split": "test",
            }
        )
    )
    output = tmp_path / "rebinned"
    rebin_evaluation(
        source,
        output,
        EvaluationConfig(
            categories=[{**config.evaluation.categories[0], "edges": [0, 10, 30]}]
        ),
        plots=False,
        bootstrap_samples=0,
    )
    assert calls == [4, 4, 4, 4]
    assert json.loads((output / "metrics.json").read_text()) == inclusive
    assert json.loads((output / "evaluation.json").read_text())["status"] == "completed"
    assert json.loads((output / "categories.json").read_text())["categories"][0][
        "edges"
    ] == [0, 10, 30]


def test_category_regression_and_plots(tmp_path):
    """
    Render classification bins and evaluate regression with target masks

    Args:
        tmp_path:  Temporary evaluation directory
    """
    data = predictions()
    spec = TaskSpec("classification", ("QCD", "top"), signal_class=1)
    evaluate_categories(
        data,
        spec,
        [{**CATEGORY, "edges": [0, 20], "underflow": False, "overflow": False}],
        SCHEMA,
        tmp_path / "classification",
        plots=True,
        bootstrap_samples=0,
    )
    assert (
        tmp_path / "classification/categories/energy/bin_000/plots/roc.png"
    ).exists()
    assert (
        tmp_path / "classification/categories/energy/plots/roc_class_1.png"
    ).exists()
    data.update(
        targets=np.arange(10.0)[:, None],
        mean=np.arange(10.0)[:, None] + 1,
        std=np.ones((10, 1)),
    )
    data["target_mask"] = np.ones((10, 1), dtype=bool)
    data["target_mask"][2] = False
    evaluate_categories(
        data,
        TaskSpec("gaussian_regression", ("x",)),
        [CATEGORY],
        SCHEMA,
        tmp_path / "regression",
        plots=False,
    )
    metrics = json.loads(
        (tmp_path / "regression/categories/energy/bin_001/metrics.json").read_text()
    )
    assert metrics["x"]["rmse"] == 1


def test_toptag_observables_precede_truncation():
    """
    Define jet bins from all valid constituents before retaining hardest particles
    """
    from drivers.toptag.reader import Jets

    jets = Jets.__new__(Jets)
    jets.p4 = torch.tensor([[5.0, 3.0, 4.0, 0.0], [2.0, 0.0, 2.0, 0.0]])
    jets.offsets = [0, 2]
    jets.labels = torch.tensor([1])
    jets.ids = ["jet"]
    jets.weights = [1.0]
    jets.max_particles = 1
    jets.feature_set = "p4"
    sample = jets[0]
    assert len(sample.p4) == 1
    assert sample.observables["n_constituents"] == 2
    assert sample.observables["jet_energy"] == 7
    assert sample.observables["jet_pt"] == pytest.approx(np.sqrt(45))


def test_tau_observables_use_physical_selected_hits():
    """
    Expose hit energy and truth transverse momentum with explicit missing truth
    """
    from drivers.tau3_classification.reader import Tau3Chunk
    from drivers.tau3_classification.driver import Tau3Dataset
    from drivers.tau3_regression.driver import Tau3RegressionDataset

    chunk = Tau3Chunk(
        features=np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32),
        offsets=np.array([0, 2]),
        weights=np.array([1.0]),
        entries=np.array([42]),
        targets=np.array([[3, 4, 5, 0, 0, 0]], dtype=np.float32),
    )
    sample = Tau3Dataset([("test.root", 1, chunk)], [(0, 0)])[0]
    assert sample.observables == {"visible_energy": 11, "n_hits": 2}
    dataset = Tau3RegressionDataset([("test.root", chunk)], [(0, 0)])
    assert dataset[0].observables["truth_pt"] == 5
    chunk.targets = None
    assert np.isnan(dataset[0].observables["truth_pt"])


def test_configuration_and_preinference_validation(tmp_path, monkeypatch):
    """
    Validate overrides and reject unavailable observables before model execution

    Args:
        tmp_path:  Temporary evaluation directory
        monkeypatch:  Isolated inference or reader replacement
    """
    from twistformer.cli.evaluate import evaluation_override

    path = tmp_path / "evaluation.yml"
    path.write_text(
        "evaluation:\n  categories:\n    - name: energy\n      observable: energy\n      edges: [0, 10]\n"
    )
    config = evaluation_override(path, EvaluationConfig(efficiencies=[0.3]))
    assert config.efficiencies == [0.3]
    assert config.categories[0]["edges"] == [0, 10]
    path.write_text("training:\n  batch_size: 1\n")
    with pytest.raises(ValueError, match="only an evaluation"):
        evaluation_override(path, EvaluationConfig())
    exp = build_experiment(parse_config({"driver": {"name": "synthetic"}}))
    predictor = Predictor(
        exp,
        exp.driver.create_preprocessor().fit(exp.driver.dataset("train")),
        "cpu",
    )

    def forbidden(*args, **kwargs):
        """
        Reject inference if validation did not stop execution first
        """
        raise AssertionError("Inference ran before observable validation")

    monkeypatch.setattr(predictor, "forward", forbidden)
    with pytest.raises(ValueError, match="Unknown observable"):
        evaluate(
            predictor,
            [],
            tmp_path / "bad",
            evaluation_config=EvaluationConfig(
                categories=[{**CATEGORY, "observable": "unknown"}]
            ),
        )
    assert not (tmp_path / "bad").exists()


def test_category_manifest_cannot_reference_other_evaluation(tmp_path):
    """
    Keep category metric reads inside their parent evaluation

    Args:
        tmp_path:  Temporary evaluation directory
    """
    source = tmp_path / "evaluation"
    source.mkdir()
    secret = tmp_path / "other"
    secret.mkdir()
    (secret / "metrics.json").write_text('{"secret": 42}')
    (source / "categories").mkdir()
    (source / "categories/link").symlink_to(secret, target_is_directory=True)
    manifest = {
        "categories": [
            {
                "name": "test",
                "bins": [{"directory": "categories/link", "id": "bad", "label": "bad"}],
            }
        ]
    }
    (source / "categories.json").write_text(json.dumps(manifest))
    assert category_views(tmp_path, source, []) == []


def test_empty_weight_and_missing_class_bins(tmp_path):
    """
    Retain useful counts when ROC support is absent

    Args:
        tmp_path:  Temporary evaluation directory
    """
    data = predictions()
    data["observable_energy"] = np.array([1, 11, 1, 11, 1, 11, 21, 21, np.nan, np.nan])
    data["weights"][6:8] = 0
    spec = TaskSpec("classification", ("QCD", "top"), signal_class=1)
    manifest = evaluate_categories(
        data,
        spec,
        [{"name": "energy", "observable": "energy", "edges": [0, 10, 20, 30]}],
        SCHEMA,
        tmp_path,
        plots=False,
    )
    entries = manifest["categories"][0]["bins"]
    assert entries[0]["status"] == entries[1]["status"] == "partial"
    assert entries[2]["status"] == "unavailable"
    assert entries[2]["reason"] == "Zero absolute event weight"
    assert entries[0]["support"]["classes"]["top"]["events"] == 0


def test_no_categories_and_missing_arrays(tmp_path):
    """
    Support empty steering and reject stale observable exports explicitly

    Args:
        tmp_path:  Temporary evaluation directory
    """
    spec = TaskSpec("classification", ("QCD", "top"), signal_class=1)
    result = evaluate_categories(
        predictions(), spec, [], SCHEMA, tmp_path / "none", plots=False
    )
    assert result["categories"] == []
    data = predictions()
    del data["observable_energy"]
    with pytest.raises(ValueError, match="Missing or misaligned"):
        evaluate_categories(
            data, spec, [CATEGORY], SCHEMA, tmp_path / "missing", plots=False
        )


@pytest.mark.parametrize("configured, explicit, expected", [(2048, None, 2048), (2048, 3, 3), (None, None, 7)])
def test_evaluation_batch_size_precedence(tmp_path, monkeypatch, configured, explicit, expected):
    """
    Resolve evaluation batches independently from training validation

    Args:
        tmp_path:  Temporary evaluation directory
        monkeypatch:  Controlled prediction collector
        configured:  Saved evaluation batch size
        explicit:  Caller override
        expected:  Effective inference batch size
    """
    from types import SimpleNamespace
    from twistformer.evaluation import runner

    config = EvaluationConfig(batch_size=configured)
    spec = TaskSpec("classification", ("QCD", "top"), signal_class=1)
    predictor = SimpleNamespace(experiment=SimpleNamespace(
        config=SimpleNamespace(evaluation=config, training=SimpleNamespace(eval_batch_size=7)),
        driver=SimpleNamespace(task_spec=spec, evaluation_observables={}),
    ))
    def collect(predictor, samples, batch_size, **kwargs):
        """
        Check the resolved batch size before supplying saved predictions

        Args:
            predictor:  Inference configuration
            samples:  Unused sample placeholder
            batch_size:  Effective evaluation batch size
            kwargs:  Collector flags
        Returns:
            result: Prediction arrays and loss
        """
        assert batch_size == expected
        data = predictions()
        data["physical_weights"] = data["weights"].copy()
        return data, .1
    monkeypatch.setattr(runner, "collect_predictions", collect)
    evaluate(predictor, [], tmp_path, batch_size=explicit, plots=False, bootstrap_samples=0)
    assert json.loads((tmp_path / "evaluation_config.json").read_text())["batch_size"] == expected


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_evaluation_batch_size(value):
    """
    Reject invalid evaluation batch sizes

    Args:
        value:  Invalid configured size
    """
    with pytest.raises(ValueError, match="positive integer"):
        EvaluationConfig(batch_size=value)


@pytest.mark.parametrize("explicit,expected", [(None, 30), (0, 0), (7, 7)])
def test_bootstrap_card_and_api_override(tmp_path, monkeypatch, explicit, expected):
    """
    Apply the configured replica count unless the caller supplies an override
    """
    from types import SimpleNamespace
    from twistformer.evaluation import runner

    config = EvaluationConfig(bootstrap_samples=30, bootstrap_seed=711)
    spec = TaskSpec("classification", ("QCD", "top"), signal_class=1)
    predictor = SimpleNamespace(experiment=SimpleNamespace(
        config=SimpleNamespace(evaluation=config, training=SimpleNamespace(eval_batch_size=7)),
        driver=SimpleNamespace(task_spec=spec, evaluation_observables={}),
    ))
    data = predictions()
    data["physical_weights"] = data["weights"].copy()
    monkeypatch.setattr(runner, "collect_predictions", lambda *args, **kwargs: (data, .1))
    result = evaluate(predictor, [], tmp_path, plots=False, bootstrap_samples=explicit)
    assert result.get("roc_bootstrap", {}).get("n_resamples", 0) == expected
    saved = json.loads((tmp_path / "evaluation_config.json").read_text())
    assert saved["bootstrap_samples"] == expected
    assert saved["bootstrap_seed"] == 711


@pytest.mark.parametrize("explicit,expected", [(None, 30), (0, 0), (7, 7)])
def test_bootstrap_card_and_cli_override(tmp_path, monkeypatch, explicit, expected):
    """
    Use evaluation YAML for rebinning and preserve explicit CLI overrides
    """
    from twistformer.cli.evaluate import main
    from twistformer.evaluation import categories

    path = tmp_path / "evaluation.yml"
    path.write_text("evaluation:\n  bootstrap_samples: 30\n  bootstrap_seed: 711\n"
                    "  categories:\n    - name: energy\n      observable: energy\n      edges: [0, 10]\n")
    calls = []
    monkeypatch.setattr(categories, "rebin_evaluation", lambda *args, **kwargs: calls.append(kwargs))
    args = ["--from-evaluation", str(tmp_path), "--evaluation-config", str(path)]
    if explicit is not None:
        args += ["--bootstrap-samples", str(explicit), "--bootstrap-seed", "712"]
    main(args)
    assert calls[0]["bootstrap_samples"] == expected
    assert calls[0]["bootstrap_seed"] == (711 if explicit is None else 712)


@pytest.mark.parametrize("count", [-1, 1, True, 2.5])
def test_invalid_bootstrap_card_count(count):
    """
    Reject invalid replica counts in evaluation steering
    """
    with pytest.raises(ValueError, match="bootstrap_samples"):
        EvaluationConfig(bootstrap_samples=count)
