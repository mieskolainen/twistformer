# Signed event-yield metric validation
#
# mikael.mieskolainen@cern.ch (2026)

import json
import numpy as np
import pytest
from twistformer import metrics
from twistformer.api.contracts import TaskSpec
from twistformer.evaluation.metrics import classification_metrics, regression_metrics


def test_auc_unsigned_coordinates():
    """Integrate integer yields in either threshold order"""
    x = np.array([0, 1, 2], dtype=np.uint8)
    y = np.full_like(x, 200)
    assert metrics.auc(x, y) == metrics.auc(x[::-1], y[::-1]) == 400.0
    with pytest.raises(ValueError, match="monotonic"):
        metrics.auc(x[[0, 2, 1]], y)


def test_comparison_auc_with_absent_class():
    """Use the same supported classes for comparison and validation AUC"""
    from twistformer.evaluation.comparison_stats import scores

    spec = TaskSpec("classification", ("a", "b", "c"))
    events = {"targets": np.array([0, 1]), "weights": np.ones(2)}
    probabilities = np.array([[0.7, 0.1, 0.2], [0.1, 0.8, 0.1]])
    actual = scores(events, probabilities, spec, [0.5])
    assert actual["macro_auc"] == 1
    assert np.isnan(actual["auc:c"])


def test_signed_binary_metrics_against_pair_concordance():
    """
    Match signed AUC against an independent weighted pair count including ties
    """
    rng = np.random.default_rng(91)
    for signed in (False, True):
        y = np.repeat([0, 1], 35)
        score = rng.integers(0, 8, len(y)) / 8
        w = rng.uniform(0.2, 2, len(y))
        if signed:
            w[::3] *= -1
        positive, negative = y == 1, y == 0
        comparison = score[positive, None] - score[None, negative]
        concordance = (comparison > 0) + 0.5 * (comparison == 0)
        expected = np.sum(concordance * w[positive, None] * w[None, negative])
        expected /= w[positive].sum() * w[negative].sum()
        assert metrics.roc_auc_score(y, score, sample_weight=w) == pytest.approx(
            expected
        )
        for factor in (7, -3):
            assert metrics.roc_auc_score(
                y, score, sample_weight=factor * w
            ) == pytest.approx(expected)
        fpr, tpr, thresholds = metrics.roc_curve(
            y, score, sample_weight=w, drop_intermediate=False
        )
        for f, t, threshold in zip(fpr, tpr, thresholds):
            selected = score >= threshold
            assert f == pytest.approx(w[selected & negative].sum() / w[negative].sum())
            assert t == pytest.approx(w[selected & positive].sum() / w[positive].sum())


def test_signed_roc_and_average_precision_are_not_clipped():
    """
    Preserve negative increments and efficiencies above unity
    """
    y, score, w = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6], [2, 1, -1, 1]
    fpr, tpr, thresholds = metrics.roc_curve(
        y, score, sample_weight=w, drop_intermediate=False
    )
    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 2, 2, 1, 1])
    assert metrics.roc_auc_score(y, score, sample_weight=w) == pytest.approx(1.5)
    assert metrics.average_precision_score(y, score, sample_weight=w) == pytest.approx(
        1.5
    )
    with pytest.raises(ValueError, match="Partial AUC"):
        metrics.roc_auc_score(y, score, sample_weight=w, max_fpr=0.5)
    assert thresholds[0] == np.inf


def test_signed_cancellation_is_explicit():
    """
    Return undefined diagnostics for zero net normalizations
    """
    with pytest.warns(RuntimeWarning, match="zero net"):
        assert np.isnan(
            metrics.roc_auc_score([1, 1, 0], [0.8, 0.7, 0.1], sample_weight=[1, -1, 2])
        )
    with pytest.warns(RuntimeWarning, match="zero net"):
        precision, _, _ = metrics.precision_recall_curve(
            [1, 0, 1], [0.9, 0.8, 0.7], sample_weight=[1, -1, 2]
        )
    assert np.isnan(precision[1])
    with pytest.warns(RuntimeWarning):
        assert np.isnan(metrics.accuracy_score([0, 1], [0, 1], sample_weight=[1, -1]))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(metrics.roc_auc_score([1, 1], [0.9, 0.8]))


def test_signed_confusion_and_single_class_accuracy():
    """
    Support arbitrary class labels, negative cells and single-class samples
    """
    cm = metrics.confusion_matrix(
        ["a", "a", "b"], ["a", "b", "b"], labels=["b", "a"], sample_weight=[2, -1, 3]
    )
    np.testing.assert_array_equal(cm, [[3, 0], [-1, 2]])
    normalized = metrics.confusion_matrix(
        [0, 0, 1], [0, 1, 1], sample_weight=[2, -1, 3], normalize="true"
    )
    np.testing.assert_array_equal(normalized, [[2, -1], [0, 1]])
    assert (
        metrics.accuracy_score(["tau", "tau"], ["tau", "tau"], sample_weight=[2, -1])
        == 1
    )
    np.testing.assert_array_equal(metrics.confusion_matrix([1, 1], [1, 1]), [[2]])


def test_multiclass_signed_averages_and_column_order():
    """
    Match multiclass reductions to explicit binary class diagnostics
    """
    y = np.array(["b", "a", "c", "b", "a", "c"])
    p = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6],
        ]
    )
    labels, w = np.array(["b", "a", "c"]), np.array([2, 3, 2, -1, -1, -0.5])
    supports = np.array([w[y == label].sum() for label in labels])
    for function in (metrics.roc_auc_score, metrics.average_precision_score):
        extra = {"multi_class": "ovr"} if function is metrics.roc_auc_score else {}
        values = function(y, p, labels=labels, sample_weight=w, average=None, **extra)
        expected = [
            function(y == label, p[:, i], sample_weight=w)
            for i, label in enumerate(labels)
        ]
        np.testing.assert_allclose(values, expected)
        assert function(
            y, p, labels=labels, sample_weight=w, average="macro", **extra
        ) == pytest.approx(np.mean(expected))
        assert function(
            y, p, labels=labels, sample_weight=w, average="weighted", **extra
        ) == pytest.approx(np.average(expected, weights=supports))
        micro = function(
            (y[:, None] == labels).ravel(), p.ravel(), sample_weight=np.repeat(w, 3)
        )
        assert function(
            y, p, labels=labels, sample_weight=w, average="micro", **extra
        ) == pytest.approx(micro)
    assert metrics.roc_auc_score(
        y, p, labels=labels, sample_weight=w, multi_class="ovo"
    ) == pytest.approx(1)


def test_nonnegative_metrics_match_sklearn_when_available():
    """
    Compare public metric conventions with an optional reference installation
    """
    try:
        from sklearn import metrics as reference
    except (ImportError, ValueError):
        pytest.skip("A compatible optional scikit-learn reference is unavailable")
    rng = np.random.default_rng(82)
    y = np.tile(np.arange(3), 30)
    p = rng.integers(1, 8, (len(y), 3)).astype(float)
    p /= p.sum(axis=1, keepdims=True)
    w = rng.uniform(0.1, 2, len(y))
    w[::9] = 0
    for binary in (y == 0, y == 1):
        for name in ("roc_curve", "precision_recall_curve"):
            actual = getattr(metrics, name)(binary, p[:, 0], sample_weight=w)
            expected = getattr(reference, name)(binary, p[:, 0], sample_weight=w)
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e, atol=1e-14)
        for name in ("roc_auc_score", "average_precision_score"):
            assert getattr(metrics, name)(
                binary, p[:, 0], sample_weight=w
            ) == pytest.approx(
                getattr(reference, name)(binary, p[:, 0], sample_weight=w)
            )
        assert metrics.roc_auc_score(
            binary, p[:, 0], sample_weight=w, max_fpr=0.3
        ) == pytest.approx(
            reference.roc_auc_score(binary, p[:, 0], sample_weight=w, max_fpr=0.3)
        )
    for average in (None, "macro", "weighted", "micro"):
        np.testing.assert_allclose(
            metrics.roc_auc_score(
                y, p, sample_weight=w, multi_class="ovr", average=average
            ),
            reference.roc_auc_score(
                y, p, sample_weight=w, multi_class="ovr", average=average
            ),
        )
    for average in ("macro", "weighted"):
        assert metrics.roc_auc_score(
            y, p, multi_class="ovo", average=average
        ) == pytest.approx(
            reference.roc_auc_score(y, p, multi_class="ovo", average=average)
        )
    for normalize in (None, "true", "pred", "all"):
        np.testing.assert_allclose(
            metrics.confusion_matrix(
                y, p.argmax(1), sample_weight=w, normalize=normalize
            ),
            reference.confusion_matrix(
                y, p.argmax(1), sample_weight=w, normalize=normalize
            ),
        )


def test_signed_evaluation_json_and_nonmonotonic_operating_point():
    """
    Use the first score-ordered crossing without a monotonic ROC assumption
    """
    score = np.array([0.9, 0.8, 0.7, 0.6])
    data = dict(
        targets=np.array([1, 0, 1, 0]),
        probabilities=np.stack([1 - score, score], axis=1),
        weights=np.array([2, 1, -1, 1]),
    )
    with pytest.warns(RuntimeWarning, match="zero net"):
        result, curves = classification_metrics(
            data, TaskSpec("classification", ("b", "s"), signal_class=1)
        )
    assert result["signal_operating_points"][0]["threshold"] == 0.9
    assert result["signal_operating_points"][0]["efficiency"] == 2
    assert not result["signal_operating_points"][0]["physical"]
    json.dumps(result, allow_nan=False)
    data["weights"] = np.array([1, 1, -1, -1])
    result, _ = classification_metrics(data, TaskSpec("classification", ("b", "s")))
    assert result["accuracy"] is None
    assert result["macro_auc"] is None
    json.dumps(result, allow_nan=False)


def test_regression_keeps_negative_weights_and_undefined_variance():
    """
    Preserve signed residual moments and null negative mean-square roots
    """
    result = regression_metrics(
        dict(
            targets=np.array([[0.0], [10.0]]),
            mean=np.zeros((2, 1)),
            std=np.ones((2, 1)),
            weights=np.array([2.0, -1.0]),
        ),
        TaskSpec("gaussian_regression", ("mass",)),
    )
    assert result["mass"]["bias"] == -10
    assert result["mass"]["rmse"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("weights", [[1, np.nan], [1, np.inf], [1]])
def test_invalid_weight_arrays(weights):
    """
    Reject nonfinite or misaligned event weights
    """
    with pytest.raises(ValueError, match="sample_weight"):
        metrics.roc_auc_score([0, 1], [0.1, 0.9], sample_weight=weights)


def test_pairwise_metrics_survive_cancelled_rest_yield(tmp_path):
    """
    Retain class pairs when signed rest normalization cancels
    """
    from twistformer.evaluation.plots import plot_results

    data = dict(
        targets=np.array([0, 1, 2]),
        probabilities=np.eye(3),
        weights=np.array([2.0, 1.0, -1.0]),
    )
    result, curves = classification_metrics(
        data, TaskSpec("classification", ("a", "b", "c"))
    )
    assert result["classes"]["a"]["auc"] is None
    assert result["classes"]["a"]["background_auc"] == {"b": 1.0, "c": 1.0}
    assert "fpr" not in curves[0]
    plot_results(
        data, TaskSpec("classification", ("a", "b", "c")), result, curves, tmp_path
    )
    assert (tmp_path / "roc_class_0.png").exists()


def test_signed_prediction_and_evaluation_export(tmp_path):
    """
    Preserve signed sample weights through the inference and metric pipeline
    """
    from dataclasses import replace
    from twistformer import build_experiment, parse_config
    from twistformer.api.predictor import Predictor
    from twistformer.evaluation.runner import evaluate

    experiment = build_experiment(
        parse_config(
            {
                "driver": {
                    "name": "synthetic",
                    "train_size": 8,
                    "test_size": 8,
                    "max_tokens": 3,
                },
                "model": {"width": 8, "heads": 2, "depth": 1},
                "training": {"device": "cpu", "amp": False},
            }
        )
    )
    app = experiment.driver
    preprocessor = app.create_preprocessor().fit(app.dataset("train"))
    predictor = Predictor(experiment, preprocessor, "cpu")
    samples = app.dataset("test")
    samples[0] = replace(samples[0], weight=-0.5)
    result = evaluate(predictor, samples, tmp_path, batch_size=3, plots=False)
    assert result["weight_summary"]["negative_events"] == 1
    with np.load(tmp_path / "predictions.npz") as data:
        np.testing.assert_array_equal(data["weights"], [-0.5] + [1.0] * 7)
    assert np.isfinite(result["preprocessed_loss"])
    json.loads((tmp_path / "metrics.json").read_text())


def test_evaluation_import_does_not_load_sklearn():
    """
    Import evaluation with scikit-learn deliberately blocked
    """
    import subprocess
    import sys

    code = """
import sys
sys.modules['sklearn'] = None
from twistformer.metrics import roc_auc_score
from twistformer.cli.evaluate import main
assert roc_auc_score([0, 1], [0.1, 0.9]) == 1
"""
    subprocess.run([sys.executable, "-c", code], check=True)
