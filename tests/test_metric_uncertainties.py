# Statistical uncertainties of weighted classifier diagnostics
#
# mikael.mieskolainen@cern.ch (2026)

import numpy as np
import pytest

from twistformer.metrics import (
    confusion_matrix_statistics,
    roc_auc_score,
    roc_curve_bootstrap,
    calibration_curve_bootstrap,
    precision_recall_curve_bootstrap,
)
from twistformer.api.contracts import TaskSpec
from twistformer.evaluation.metrics import classification_metrics
from twistformer.evaluation.plots import plot_results, format_yield_uncertainty


def test_wilson_intervals_and_weight_scaling():
    """
    Check equal-population cells and invariance under yield rescaling

    Returns:
        None: Wilson errors and empty-class handling are verified
    """
    y = [0, 0, 1, 1]
    pred = [0, 1, 0, 1]
    result = confusion_matrix_statistics(y, pred, labels=[0, 1, 2])
    scaled = confusion_matrix_statistics(
        y, pred, labels=[0, 1, 2], sample_weight=[7] * 4
    )
    for axis in ("row", "column"):
        np.testing.assert_allclose(result[axis + "_rate"][:2, :2], 0.5)
        np.testing.assert_allclose(
            result[axis + "_lower"][:2, :2], 0.5 - np.sqrt(1 / 12)
        )
        np.testing.assert_allclose(
            result[axis + "_upper"][:2, :2], 0.5 + np.sqrt(1 / 12)
        )
        for quantity in ("rate", "lower", "upper"):
            np.testing.assert_allclose(
                result[f"{axis}_{quantity}"], scaled[f"{axis}_{quantity}"]
            )
    assert np.isnan(result["row_rate"][2]).all()
    assert np.isnan(result["column_rate"][:, 2]).all()


def test_signed_ratio_covariance():
    """
    Compare ratio errors with independent-event derivatives

    Returns:
        None: Signed covariance propagation is verified
    """
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 0, 1])
    weights = np.array([3, -1, 2, 4])
    result = confusion_matrix_statistics(y, pred, sample_weight=weights)
    assert result["row_rate"][0, 0] == 1.5
    for prefix, labels in (("row", y), ("column", pred)):
        for row in range(2):
            for col in range(2):
                denominator = labels == (row if prefix == "row" else col)
                cell = (y == row) & (pred == col)
                rate = weights[cell].sum() / weights[denominator].sum()
                gradient = (
                    weights * (cell - rate * denominator) / weights[denominator].sum()
                )
                error = np.sqrt(np.sum(gradient**2))
                assert result[prefix + "_upper"][row, col] == pytest.approx(
                    rate + error
                )
                assert result[prefix + "_lower"][row, col] == pytest.approx(
                    rate - error
                )


@pytest.mark.parametrize("signed", [False, True])
def test_bootstrap_matches_explicit_event_resampling(signed):
    """
    Check paired bootstrap AUC quantiles with ties and zero-weight events

    Args:
        signed:  Include negative event weights
    Returns:
        None: Replica AUC bounds and weight scaling are verified
    """
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.5, 0.5, 0.3, 0.5, 0.9])
    weights = np.array([1, 2, 0, 1, -0.5 if signed else 0.5, 2])
    band = roc_curve_bootstrap(
        y, score, sample_weight=weights, n_resamples=80, random_state=12
    )
    rng = np.random.default_rng(12)
    areas = []
    for _ in range(80):
        indices = np.repeat(np.arange(6), rng.multinomial(6, np.full(6, 1 / 6)))
        yy, ww = y[indices], weights[indices]
        if ww[yy == 0].sum() == 0 or ww[yy == 1].sum() == 0:
            continue
        areas.append(roc_auc_score(yy, score[indices], sample_weight=ww))
    alpha = (1 - band["confidence_level"]) / 2
    np.testing.assert_allclose(
        [band["auc_lower"], band["auc_upper"]], np.quantile(areas, [alpha, 1 - alpha])
    )
    assert band["valid_resamples"] == len(areas)
    scaled = roc_curve_bootstrap(
        y, score, sample_weight=weights * 8, n_resamples=80, random_state=12
    )
    for key in ("band_fpr", "tpr_lower", "tpr_upper", "auc_lower", "auc_upper"):
        np.testing.assert_allclose(band[key], scaled[key])
    assert band["method"] == (
        "threshold_percentile" if signed else "fixed_fpr_percentile"
    )
    assert np.all(band["tpr_upper"] >= band["tpr_lower"])


@pytest.mark.parametrize("signed", [False, True])
def test_shaded_bands_and_auc_legends(tmp_path, monkeypatch, signed):
    """
    Render multiclass bands and combined efficiency/purity annotations

    Args:
        tmp_path:  Temporary figure directory
        monkeypatch:  Figure export inspection fixture
        signed:  Include negative-weight events
    Returns:
        None: Curve legends, colour bands and annotations are verified
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    rng = np.random.default_rng(42)
    y = np.tile(np.arange(3), 40)
    p = rng.dirichlet([1, 1, 1], len(y)) + np.eye(3)[y] * 0.4
    p /= p.sum(axis=1, keepdims=True)
    data = {"targets": y, "probabilities": p, "weights": np.ones(len(y))}
    if signed:
        data["weights"][::7] = -0.2
    spec = TaskSpec("classification", ("0pi0", "1pi0", "2pi0"))
    metrics, curves = classification_metrics(data, spec, bootstrap_samples=30)
    savefig = plt.Figure.savefig
    captured = []

    def capture(figure, path, **kwargs):
        """
        Inspect the plotted curves before exporting the figure

        Args:
            figure:  Matplotlib figure
            path:  Figure destination
            kwargs:  Matplotlib export arguments
        Returns:
            None: Figure is checked and exported
        """
        assert all("bootstrap" not in ax.get_title().lower() for ax in plt.gcf().axes)
        if path.stem.startswith("roc"):
            ax = plt.gca()
            random = ax.lines[-1]
            assert random.get_label() == "random (AUC = 0.5000)"
            assert random.get_color() == "gray"
            assert random.get_linestyle() == "--"
            np.testing.assert_allclose(random.get_xdata(), random.get_ydata())
            assert len(ax.collections) == len(ax.lines) - 1
            for line, band in zip(ax.lines[:-1], ax.collections):
                assert " (AUC = " in line.get_label()
                decimals = line.get_label().split("AUC = ")[1].rstrip(")").split(".")[1]
                assert len(decimals) == 4
                np.testing.assert_allclose(
                    band.get_facecolor()[0], to_rgba(line.get_color(), 0.2)
                )
            captured.append(path.stem)
        elif path.stem == "confusion_normalized":
            ax = plt.gcf().axes[0]
            assert ax.get_title() == ""
            assert len(ax.texts) == 18
            assert len(ax.patches) == 21
            assert len(plt.gcf().axes) == 3
            assert all("^{+" in text.get_text() for text in ax.texts)
        elif path.stem == "confusion":
            assert plt.gcf().axes[0].get_title() == ""
        elif path.stem in ("precision_recall", "calibration", "calibration_top_label"):
            ax = plt.gca()
            count = 1 if path.stem == "calibration_top_label" else 3
            assert len(ax.collections) == count
            lines = ax.lines if path.stem == "precision_recall" else ax.lines[1:]
            for line, band in zip(lines, ax.collections):
                np.testing.assert_allclose(
                    band.get_facecolor()[0], to_rgba(line.get_color(), 0.2)
                )
            if path.stem == "precision_recall":
                assert ax.get_xscale() == "linear"
            elif path.stem == "calibration":
                assert [line.get_label() for line in lines] == list(spec.outputs)
            if not signed or path.stem.startswith("calibration"):
                assert ax.get_xlim() == (0, 1)
                assert ax.get_ylim() == (0, 1)
        elif path.stem.startswith("rejection"):
            ax = plt.gca()
            assert not signed
            assert ax.get_xscale() == "linear"
            assert ax.get_yscale() == "log"
            assert ax.get_xlim() == (0, 1)
            random = ax.lines[-1]
            assert random.get_label() == "random (AUC = 0.5000)"
            assert random.get_color() == "gray"
            assert random.get_linestyle() == "--"
            np.testing.assert_allclose(random.get_ydata(), 1 / random.get_xdata())
            assert len(ax.collections) == len(ax.lines) - 1
            source = 0 if path.stem == "rejection" else int(path.stem.rsplit("_", 1)[1])
            prefix = (
                ""
                if path.stem == "rejection"
                else f"background_{next(j for j in range(3) if j != source)}_"
            )
            x = curves[source][prefix + "fpr"]
            np.testing.assert_allclose(
                ax.lines[0].get_xdata(), curves[source][prefix + "tpr"]
            )
            finite = x > 0
            np.testing.assert_allclose(ax.lines[0].get_ydata()[finite], 1 / x[finite])
            assert np.isnan(ax.lines[0].get_ydata()[~finite]).all()
        elif path.stem.startswith("score_"):
            ax = plt.gca()
            assert ax.get_xlabel() == r"$P$"
            assert ax.get_xlim() == (0, 1)
            assert ax.get_yscale() == (
                ("symlog" if signed else "log")
                if path.stem.endswith("_log")
                else "linear"
            )
        savefig(figure, path, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    plot_results(data, spec, metrics, curves, tmp_path)
    assert set(captured) == {"roc", "roc_class_0", "roc_class_1", "roc_class_2"}
    assert metrics["classes"]["0pi0"]["auc_intervals"]["rest"]["valid_resamples"] == 30
    assert (tmp_path / "confusion_normalized.pdf").stat().st_size > 0


def test_yield_mathtext():
    """
    Check shared scientific exponents and signed yield annotations

    Returns:
        None: Large, small and zero yields use valid mathtext
    """
    assert format_yield_uncertainty(12340, 400) == r"$(1.234 \pm 0.040)\times 10^{4}$"
    assert format_yield_uncertainty(-12340, 400) == r"$(-1.234 \pm 0.040)\times 10^{4}$"
    assert (
        format_yield_uncertainty(0.00012, 0.00001) == r"$(1.20 \pm 0.10)\times 10^{-4}$"
    )
    assert format_yield_uncertainty(0, 0) == r"$0.00 \pm 0.00$"


@pytest.mark.parametrize("signed", [False, True])
def test_calibration_bootstrap_explicit_resampling(signed):
    """
    Compare fixed-bin intervals with explicit event replicas

    Args:
        signed:  Include cancellations in the first bin
    Returns:
        None: Central values, interval bounds and valid counts are verified
    """
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.15, 0.5, 0.7, 0.9, 1.0])
    w = np.array([1, -1 if signed else 2, 1, 2, 1, 0])
    result = calibration_curve_bootstrap(
        y, p, sample_weight=w, n_bins=5, n_resamples=60, random_state=12
    )
    index = np.minimum(np.searchsorted(np.linspace(0, 1, 6), p, side="right") - 1, 4)
    rng = np.random.default_rng(12)
    replicas = np.full((60, 5), np.nan)
    for replica in range(60):
        events = np.repeat(
            np.arange(len(y)), rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
        )
        for i in range(5):
            selected = events[index[events] == i]
            if w[selected].sum() != 0:
                replicas[replica, i] = (
                    np.dot(w[selected], y[selected]) / w[selected].sum()
                )
    alpha = (1 - result["confidence_level"]) / 2
    for i in range(5):
        finite = replicas[:, i][np.isfinite(replicas[:, i])]
        assert result["valid_resamples"][i] == len(finite)
        if len(finite) >= 2:
            np.testing.assert_allclose(
                [result["prob_true_lower"][i], result["prob_true_upper"][i]],
                np.quantile(finite, [alpha, 1 - alpha]),
            )
        else:
            assert np.isnan(result["prob_true_lower"][i])
    scaled = calibration_curve_bootstrap(
        y, p, sample_weight=w * 4, n_bins=5, n_resamples=60, random_state=12
    )
    for key in ("prob_pred", "prob_true", "prob_true_lower", "prob_true_upper"):
        np.testing.assert_allclose(result[key], scaled[key])
    disabled = calibration_curve_bootstrap(
        y, p, sample_weight=w, n_bins=5, n_resamples=0
    )
    np.testing.assert_allclose(disabled["prob_true"], result["prob_true"])
    assert not disabled["valid_resamples"].any()


@pytest.mark.parametrize("signed", [False, True])
def test_precision_bootstrap_explicit_resampling(signed):
    """
    Compare precision bands with a direct threshold scan of event replicas

    Args:
        signed:  Include negative weights and undefined selected yields
    Returns:
        None: Precision quantiles and valid replica counts are verified
    """
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.3, 0.5, 0.9])
    w = np.array([1, 2, 0, 1, -1 if signed else 1, 2])
    band = precision_recall_curve_bootstrap(
        y, scores, sample_weight=w, n_resamples=60, random_state=12
    )
    thresholds = np.unique(scores[w != 0])[::-1]
    rng = np.random.default_rng(12)
    replicas = []
    for _ in range(60):
        events = np.repeat(
            np.arange(len(y)), rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
        )
        total = w[events][y[events] == 1].sum()
        if total == 0:
            continue
        precision, recall = [1.0], [0.0]
        for threshold in thresholds:
            selected = events[scores[events] >= threshold]
            tp = w[selected][y[selected] == 1].sum()
            denominator = w[selected].sum()
            precision.append(tp / denominator if denominator != 0 else np.nan)
            recall.append(tp / total)
        precision, recall = np.array(precision), np.array(recall)
        finite = np.isfinite(precision)
        replicas.append(
            precision
            if signed
            else np.interp(band["band_recall"], recall[finite], precision[finite])
        )
    replicas = np.array(replicas)
    alpha = (1 - band["confidence_level"]) / 2
    lower, upper = np.nanquantile(replicas, [alpha, 1 - alpha], axis=0)
    np.testing.assert_allclose(band["precision_lower"], lower)
    np.testing.assert_allclose(band["precision_upper"], upper)
    np.testing.assert_array_equal(
        band["valid_resamples"], np.isfinite(replicas).sum(axis=0)
    )
