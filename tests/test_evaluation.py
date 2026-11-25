# Weighted classification diagnostics
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import pytest
from twistformer.api.contracts import TaskSpec
from twistformer.evaluation.metrics import classification_metrics
from twistformer.evaluation.plots import plot_results


def weighted_data():
    return {
        "targets": np.array([0, 0, 1, 1, 2, 2, 2]),
        "probabilities": np.array(
            [
                [0.6, 0.2, 0.2],
                [0.2, 0.5, 0.3],
                [0.4, 0.5, 0.1],
                [0.1, 0.7, 0.2],
                [0.3, 0.2, 0.5],
                [0.1, 0.3, 0.6],
                [0.95, 0.025, 0.025],
            ]
        ),
        "weights": np.array([1.0, 3.0, 2.0, 1.0, 4.0, 2.0, 0.0]),
    }


def test_weighted_multiclass_roc_and_pairwise():
    data = weighted_data()
    spec = TaskSpec("classification", ("0pi0", "1pi0", "2pi0"))
    metrics, curves = classification_metrics(data, spec)
    for signal, curve in curves.items():
        for background in (None, *[i for i in range(3) if i != signal]):
            prefix = "" if background is None else f"background_{background}_"
            targets, scores, weights = (
                data["targets"],
                data["probabilities"][:, signal],
                data["weights"],
            )
            positives = targets == signal
            negatives = ~positives if background is None else targets == background
            for threshold, fpr, tpr in zip(
                curve[prefix + "thresholds"],
                curve[prefix + "fpr"],
                curve[prefix + "tpr"],
            ):
                accepted = scores >= threshold
                assert tpr == pytest.approx(
                    weights[accepted & positives].sum() / weights[positives].sum()
                )
                assert fpr == pytest.approx(
                    weights[accepted & negatives].sum() / weights[negatives].sum()
                )
    assert metrics["classes"]["0pi0"]["background_auc"]["1pi0"] == pytest.approx(0.5)
    scaled = {**data, "weights": data["weights"] * 7}
    scaled_metrics, scaled_curves = classification_metrics(scaled, spec)
    assert scaled_metrics["macro_auc"] == pytest.approx(metrics["macro_auc"])
    for i in curves:
        for key in curves[i]:
            np.testing.assert_allclose(curves[i][key], scaled_curves[i][key])


def test_scalar_uncertainties_use_weighted_event_replicas():
    """Check scalar standard errors against the same explicit event resamples"""
    from twistformer.metrics import one_vs_rest_auc

    data = weighted_data()
    y, p, weights = (data[key] for key in ("targets", "probabilities", "weights"))
    weights[0] = -0.25
    loss_weights = weights * (1 + y)
    loss = np.linspace(0.1, 2, len(y)) * loss_weights
    data.update(evaluation_loss_numerator=loss, evaluation_loss_normalizer=loss_weights)
    spec = TaskSpec("classification", ("a", "b", "c"))
    count, seed = 40, 133
    metrics, _ = classification_metrics(data, spec, bootstrap_samples=count, bootstrap_seed=seed)
    rng = np.random.default_rng(seed)
    accuracy, loss_values, macro = [], [], []
    for _ in range(count):
        multiplicity = rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
        replica_weights = weights * multiplicity
        if replica_weights.sum() != 0:
            accuracy.append(np.average(p.argmax(-1) == y, weights=replica_weights))
        if (loss_weights @ multiplicity) != 0:
            loss_values.append((loss @ multiplicity) / (loss_weights @ multiplicity))
        areas, auc = one_vs_rest_auc(y, p, replica_weights)
        if len(areas) == len(spec.outputs):
            macro.append(auc)
    for key, values in (("accuracy", accuracy), ("preprocessed_loss", loss_values), ("macro_auc", macro)):
        item = metrics["uncertainties"][key]
        assert item["std"] == pytest.approx(np.std(values, ddof=1))
        assert item["valid_resamples"] == len(values)
    assert metrics["preprocessed_loss"] == pytest.approx(loss.sum() / loss_weights.sum())
    assert metrics["classes"]["a"]["uncertainties"]["average_precision"]["std"] > 0
    assert metrics["classes"]["a"]["uncertainties"]["background_auc"]["b"]["std"] > 0
    assert "uncertainties" not in classification_metrics(data, spec, bootstrap_samples=0)[0]


def test_rejection_intervals_include_zero_background_replicas():
    """Keep unbounded rejection in the central bootstrap interval"""
    scores = np.r_[np.linspace(.9, .6, 100), .91, .775, np.full(12, .69), np.full(86, .1)]
    data = {
        "targets": np.r_[np.ones(100, dtype=int), np.zeros(100, dtype=int)],
        "probabilities": np.column_stack((1 - scores, scores)),
        "weights": np.ones(len(scores)),
    }
    spec = TaskSpec("classification", ("pi0", "gamma"), signal_class=1)
    metrics, _ = classification_metrics(data, spec, (0.3, 0.5, 0.9), bootstrap_samples=200)
    points = metrics["signal_operating_points"]
    for point in points[:2]:
        interval = point["uncertainties"]["rejection"]
        assert interval["std"] is None
        assert 0 < interval["lower"] < point["rejection"]
        assert interval["zero_background_resamples"] > 0
        assert interval["valid_resamples"] == 200
    first = points[0]["uncertainties"]["rejection"]
    assert first["upper_unbounded"] and first["upper"] is None
    assert points[2]["uncertainties"]["rejection"]["std"] > 0


def test_log_roc_plots_preserve_exact_zero_background(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    data = weighted_data()
    spec = TaskSpec("classification", ("0pi0", "1pi0", "2pi0"))
    metrics, curves = classification_metrics(data, spec)
    captured = {}
    legends = {}
    original = plt.Figure.savefig

    def save(figure, path, **kwargs):
        if path.name.startswith(("roc", "rejection")):
            legends[path.name] = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        if path.name.startswith("roc"):
            axis = plt.gca()
            captured[path.name] = (
                axis.get_xscale(),
                axis.get_xlabel(),
                axis.get_ylabel(),
                len(axis.lines),
            )
        original(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", save)
    plot_results(data, spec, metrics, curves, tmp_path)
    for stem, count in [
        ("roc", 3),
        ("roc_class_0", 2),
        ("roc_class_1", 2),
        ("roc_class_2", 2),
    ]:
        for extension in ("png", "pdf"):
            name = f"{stem}.{extension}"
            assert (tmp_path / name).stat().st_size > 0
            assert captured[name] == (
                "log",
                r"Background efficiency $\epsilon_B$",
                r"Signal efficiency $\epsilon_S$",
                count + 1,
            )
    assert all(curve["fpr"][0] == 0 for curve in curves.values())
    for kind in ("roc", "rejection"):
        for extension in ("png", "pdf"):
            assert legends[f"{kind}.{extension}"] == [
                f"{name} vs rest (AUC = {metrics['classes'][name]['auc']:.4f})" for name in spec.outputs
            ] + ["random (AUC = 0.5000)"]
            for index, name in enumerate(spec.outputs):
                assert legends[f"{kind}_class_{index}.{extension}"] == [
                    f"{name} vs {background} (AUC = {metrics['classes'][name]['background_auc'][background]:.4f})"
                    for background in spec.outputs if background != name
                ] + ["random (AUC = 0.5000)"]


@pytest.mark.parametrize("selection_id", ["inclusive", "energy/bin_000"])
def test_comparison_roc_legends_use_each_model_and_class_auc(tmp_path, monkeypatch, selection_id):
    """
    Label comparison ROC and rejection curves with each selected model AUC

    Args:
        tmp_path:  Figure directory
        monkeypatch:  Capture exported legends
        selection_id:  Inclusive or category selection
    """
    from twistformer.evaluation import comparison_plots

    data = weighted_data()
    spec = TaskSpec("classification", ("a", "b", "c"))
    selection = {"id": selection_id, "label": "Selected events", "models": []}
    for label, probability in (("Model A", data["probabilities"]), ("Model B", data["probabilities"][:, ::-1])):
        metrics, curves = classification_metrics({**data, "probabilities": probability}, spec, bootstrap_samples=0)
        selection["models"].append({"label": label, "metrics": metrics, "curves": {str(i): c for i, c in curves.items()}})
    legends = {}

    def save(figure, path):
        """
        Capture the legend sent to each output format

        Args:
            figure:  Diagnostic figure
            path:  Output figure path
        """
        legends[path.name] = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]

    monkeypatch.setattr(comparison_plots, "save_figure", save)
    comparison_plots.plot_comparison(selection, spec, tmp_path)
    for index, name in enumerate(spec.outputs):
        expected = [
            f"{model['label']} (AUC = {model['metrics']['classes'][name]['auc']:.4f})"
            for model in selection["models"]
        ]
        for extension in ("png", "pdf"):
            assert legends[f"roc_class_{index}.{extension}"] == expected + ["random (AUC = 0.5000)"]
            assert legends[f"rejection_class_{index}.{extension}"] == expected
            assert legends[f"precision_recall_class_{index}.{extension}"] == ["Model A", "Model B"]


def test_zero_weight_background_is_undefined():
    data = weighted_data()
    data["weights"][data["targets"] == 2] = 0
    metrics, curves = classification_metrics(
        data, TaskSpec("classification", ("a", "b", "c"))
    )
    assert metrics["classes"]["c"]["auc"] is None
    assert metrics["classes"]["a"]["background_auc"]["c"] is None
    assert "background_2_fpr" not in curves[0]


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("class_count", [2, 3])
@pytest.mark.parametrize("bootstrap_samples", [0, 30])
def test_category_overlay_accepts_label_curve_pairs(tmp_path, monkeypatch, signed, class_count, bootstrap_samples):
    """
    Draw bin AUCs and bootstrap bands from full category curve records

    Args:
        tmp_path:  Figure directory
        monkeypatch:  Capture exported legend labels
        signed:  Check a nonmonotonic ROC from a negative event weight
        class_count:  Binary or multiclass PID
        bootstrap_samples:  Event replicas, zero disables bands
    """
    from twistformer.evaluation import plots
    from matplotlib.colors import to_rgba

    data = weighted_data()
    selected = data["targets"] < class_count
    data = {key: value[selected] for key, value in data.items()}
    data["probabilities"] = data["probabilities"][:, :class_count]
    data["probabilities"] /= data["probabilities"].sum(axis=1, keepdims=True)
    if signed:
        data["weights"][0] = -0.25
    spec = TaskSpec("classification", ("a", "b", "c")[:class_count], signal_class=1 if class_count == 2 else None)
    series, expected = [], {i: [] for i in range(class_count)}
    for label, probability in (("Bin A", data["probabilities"]), ("Bin B", data["probabilities"][:, ::-1])):
        metrics, curves = classification_metrics(
            {**data, "probabilities": probability}, spec, bootstrap_samples=bootstrap_samples,
        )
        series.append((label, curves))
        for index, name in enumerate(spec.outputs):
            expected[index].append(f"{label} (AUC = {metrics['classes'][name]['auc']:.4f})")
    if signed:
        assert any((np.diff(curve["fpr"]) < 0).any() for _, curves in series for curve in curves.values())
    legends = {}

    def save(figure, path):
        """
        Capture legend labels without writing figures

        Args:
            figure:  Category ROC figure
            path:  Output figure path
        """
        legends[path.name] = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        axis = figure.axes[0]
        assert axis.get_xscale() == ("linear" if signed else "log")
        assert len(axis.collections) == (len(series) if bootstrap_samples else 0)
        index = int(path.stem.rsplit("_", 1)[1])
        for line, band, (_, curves) in zip(axis.lines[:-1], axis.collections, series):
            np.testing.assert_allclose(band.get_facecolor()[0], to_rgba(line.get_color(), 0.2))
            curve = curves[index]
            expected_x = curve["band_fpr"] if signed else np.maximum(curve["band_fpr"], axis.get_xlim()[0])
            vertices = np.concatenate([result.vertices for result in band.get_paths()])
            for x, lower, upper in zip(expected_x, curve["tpr_lower"], curve["tpr_upper"]):
                if np.isfinite([lower, upper]).all():
                    assert np.any(np.all(np.isclose(vertices, [x, lower]), axis=1))
                    assert np.any(np.all(np.isclose(vertices, [x, upper]), axis=1))

    monkeypatch.setattr(plots, "save_figure", save)
    plots.plot_category_rocs(series, spec, tmp_path, signed=signed)
    for index in ([1] if class_count == 2 else range(class_count)):
        for extension in ("png", "pdf"):
            assert legends[f"roc_class_{index}.{extension}"] == expected[index] + ["random (AUC = 0.5000)"]
