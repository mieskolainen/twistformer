# Signed event-weight classification metrics
#
# mikael.mieskolainen@cern.ch (2026)

"""NumPy metrics with scikit-learn-style arguments and signed event yields"""

import warnings
import numpy as np
from twistformer.randomness import numpy_generator

__all__ = [
    "confusion_matrix_statistics",
    "roc_curve_bootstrap",
    "calibration_curve_bootstrap",
    "precision_recall_curve_bootstrap",
    "accuracy_score",
    "confusion_matrix",
    "roc_curve",
    "roc_auc_score",
    "one_vs_rest_auc",
    "precision_recall_curve",
    "average_precision_score",
    "auc",
]

def _weights(sample_weight, size):
    """
    Validate finite signed event weights

    Args:
        sample_weight:  Event weights or None for unit weights
        size:  Number of events
    Returns:
        weights: Float64 event weights
    """
    w = np.ones(size) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w.shape != (size,) or not np.isfinite(w).all():
        raise ValueError("sample_weight must be a finite vector matching y_true")
    return w

def _vector(values, name):
    """
    Validate a nonempty one-dimensional label or score array

    Args:
        values:  Input array
        name:  Argument name for diagnostics
    Returns:
        array: Validated NumPy array
    """
    a = np.asarray(values)
    if a.ndim != 1 or not len(a):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a

def _ratio(numerator, denominator):
    """
    Normalize signed yields without clipping or replacing cancellations

    Args:
        numerator:  Signed yield or array of yields
        denominator:  Corresponding signed normalization
    Returns:
        value: Ratio with NaN for zero denominators
    """
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    if np.any(b == 0):
        warnings.warn("Metric is undefined for zero net event weight", RuntimeWarning, stacklevel=2)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)

def _binary_counts(y_true, y_score, pos_label, sample_weight):
    """
    Accumulate signed signal and background yields at distinct score thresholds

    Args:
        y_true:  Binary event labels
        y_score:  Finite discriminant scores
        pos_label:  Signal label or None for labels 0/1 or -1/1
        sample_weight:  Signed event weights
    Returns:
        counts: Background yields, signal yields and descending thresholds
    """
    y, score = _vector(y_true, "y_true"), _vector(y_score, "y_score").astype(float)
    if score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("y_score must be finite and match y_true")
    classes = np.unique(y)
    if len(classes) > 2:
        raise ValueError("Binary curves require at most two classes")
    if pos_label is None:
        if not (np.isin(classes, [0, 1]).all() or np.isin(classes, [-1, 1]).all()):
            raise ValueError("Specify pos_label for labels other than 0/1 or -1/1")
        pos_label = 1
    w = _weights(sample_weight, len(y))
    active = w != 0
    y, score, w = y[active] == pos_label, score[active], w[active]
    order = np.argsort(-score, kind="stable")
    y, score, w = y[order], score[order], w[order]
    if not len(score):
        return np.zeros(1), np.zeros(1), np.array([np.inf])
    ends = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    tp = np.cumsum(w * y, dtype=float)[ends]
    fp = np.cumsum(w * ~y, dtype=float)[ends]
    return fp, tp, score[ends]

def roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True):
    """
    Compute a binary ROC from signed yields above each score threshold

    Args:
        y_true:  Binary labels
        y_score:  Signal scores
        pos_label:  Signal label, inferred as 1 for standard binary labels
        sample_weight:  Signed event weights or None
        drop_intermediate:  Remove redundant collinear threshold points
    Returns:
        curve: FPR, TPR and descending thresholds starting at infinity
    """
    fp, tp, thresholds = _binary_counts(y_true, y_score, pos_label, sample_weight)
    if drop_intermediate and len(fp) > 2:
        selected = np.r_[True, (np.diff(fp, 2) != 0) | (np.diff(tp, 2) != 0), True]
        fp, tp, thresholds = fp[selected], tp[selected], thresholds[selected]
    return (_ratio(np.r_[0, fp], fp[-1]), _ratio(np.r_[0, tp], tp[-1]), np.r_[np.inf, thresholds])

def precision_recall_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False):
    """
    Compute precision and recall from signed selected event yields

    Args:
        y_true:  Binary labels
        y_score:  Signal scores
        pos_label:  Signal label
        sample_weight:  Signed event weights or None
        drop_intermediate:  Remove interior points with unchanged signal yield
    Returns:
        curve: Precision, recall and ascending thresholds with terminal (1,0)
    """
    fp, tp, thresholds = _binary_counts(y_true, y_score, pos_label, sample_weight)
    if drop_intermediate and len(tp) > 2:
        selected = np.r_[True, (np.diff(tp)[:-1] != 0) | (np.diff(tp)[1:] != 0), True]
        fp, tp, thresholds = fp[selected], tp[selected], thresholds[selected]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp[-1])
    return np.r_[precision[::-1], 1.0], np.r_[recall[::-1], 0.0], thresholds[::-1]

def auc(x, y):
    """
    Integrate an ordered curve with the trapezoidal rule

    Args:
        x:  Monotonic coordinates, increasing or decreasing
        y:  Curve ordinates
    Returns:
        area: Trapezoidal area with positive coordinate orientation
    """
    x, y = _vector(x, "x").astype(float), _vector(y, "y").astype(float)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("x and y must match and contain at least two points")
    dx = np.diff(x)
    if not ((dx >= 0).all() or (dx <= 0).all()):
        raise ValueError("x must be monotonic, use roc_auc_score for signed ROC curves")
    direction = -1 if (dx <= 0).all() else 1
    return float(direction * np.sum(dx * (y[1:] + y[:-1]) / 2))

def _binary_auc(y, score, w, max_fpr):
    """
    Integrate a signed ROC in threshold order

    Args:
        y:  Boolean signal labels
        score:  Signal scores
        w:  Signed event weights
        max_fpr:  Optional partial AUC limit for nonnegative weights
    Returns:
        area: Signed pair-concordance AUC with half credit for ties
    """
    fp, tp, _ = roc_curve(y, score, sample_weight=w, drop_intermediate=False)
    if max_fpr is not None and max_fpr != 1:
        if not 0 < max_fpr <= 1:
            raise ValueError("max_fpr must be in (0,1]")
        if (w < 0).any():
            raise ValueError("Partial AUC is undefined for nonmonotonic signed ROC curves")
        if not np.isfinite(fp).all() or not np.isfinite(tp).all():
            return float("nan")
        stop = np.searchsorted(fp, max_fpr, side="right")
        endpoint = np.interp(max_fpr, fp[stop - 1 : stop + 1], tp[stop - 1 : stop + 1])
        fp, tp = np.r_[fp[:stop], max_fpr], np.r_[tp[:stop], endpoint]
        area = np.sum(np.diff(fp) * (tp[1:] + tp[:-1]) / 2)
        minimum = 0.5 * max_fpr**2
        return float(0.5 * (1 + (area - minimum) / (max_fpr - minimum)))
    return float(np.sum(np.diff(fp) * (tp[1:] + tp[:-1]) / 2))

def _class_inputs(y_true, y_score, labels, sample_weight):
    """
    Align single-label multiclass targets with probability columns

    Args:
        y_true:  One label per event
        y_score:  Class probability matrix
        labels:  Column labels or sorted observed classes
        sample_weight:  Signed event weights
    Returns:
        inputs: Targets, scores, ordered labels and weights
    """
    y = _vector(y_true, "y_true")
    score = np.asarray(y_score, dtype=float)
    classes = np.unique(y) if labels is None else _vector(labels, "labels")
    if len(np.unique(classes)) != len(classes) or not np.isin(y, classes).all():
        raise ValueError("labels must be unique and include every observed class")
    if (
        score.shape != (len(y), len(classes))
        or not np.isfinite(score).all()
        or (score < 0).any()
        or (score > 1).any()
        or not np.allclose(score.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Multiclass y_score must be normalized probabilities matching labels")
    return y, score, classes, _weights(sample_weight, len(y))

def _average(values, supports, average):
    """
    Reduce class diagnostics with explicit undefined-class propagation

    Args:
        values:  Per-class or per-pair diagnostics
        supports:  Corresponding signed event yields
        average:  None, macro or weighted
    Returns:
        result: Diagnostic array or scalar
    """
    values = np.asarray(values, float)
    if average is None:
        return values
    if average == "macro":
        return float(np.mean(values))
    if average == "weighted":
        return float(_ratio(np.dot(values, supports), np.sum(supports)))
    raise ValueError("average must be None, 'macro', 'weighted' or supported 'micro'")

def roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None, max_fpr=None, multi_class="raise", labels=None):
    """
    Compute binary or multiclass ROC AUC with signed event weights

    Negative weights define a threshold-ordered line integral, equivalently
    signed signal-background pair concordance. AUC need not lie in [0,1]

    Args:
        y_true:  Binary or single-label multiclass targets
        y_score:  Binary scores or multiclass probability columns
        average:  None, macro, weighted or micro for one-vs-rest
        sample_weight:  Signed event weights or None
        max_fpr:  Standardized partial AUC limit for nonnegative binary weights
        multi_class:  raise, ovr or ovo
        labels:  Multiclass probability-column labels
    Returns:
        area: Scalar AUC or one AUC per class for average=None
    """
    score = np.asarray(y_score)
    if score.ndim == 1:
        y = _vector(y_true, "y_true")
        w = _weights(sample_weight, len(y))
        score = _vector(score, "y_score").astype(float)
        if score.shape != y.shape or not np.isfinite(score).all():
            raise ValueError("y_score must be finite and match y_true")
        classes = np.unique(y)
        if len(classes) > 2:
            raise ValueError("Multiclass AUC requires probability columns")
        if len(classes) < 2:
            warnings.warn("ROC AUC requires two observed classes", RuntimeWarning, stacklevel=2)
            return float("nan")
        return _binary_auc(y == classes[-1], score, w, max_fpr)
    if multi_class not in {"ovr", "ovo"}:
        raise ValueError("Specify multi_class='ovr' or 'ovo' for probability matrices")
    if max_fpr not in (None, 1):
        raise ValueError("Partial AUC is supported for binary scores only")
    y, score, classes, w = _class_inputs(y_true, score, labels, sample_weight)
    if len(classes) < 2:
        warnings.warn("ROC AUC requires two classes", RuntimeWarning, stacklevel=2)
        return _average([np.nan], [w.sum()], average)
    if multi_class == "ovr":
        binary = y[:, None] == classes
        if average == "micro":
            return _binary_auc(binary.ravel(), score.ravel(), np.repeat(w, len(classes)), None)
        return _average(
            [_binary_auc(binary[:, i], score[:, i], w, None) for i in range(len(classes))],
            [w[y == label].sum() for label in classes],
            average,
        )
    if average not in {"macro", "weighted"}:
        raise ValueError("One-vs-one AUC supports macro or weighted averaging")
    areas, supports = [], []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            selected = (y == classes[i]) | (y == classes[j])
            if not selected.any():
                areas.append(np.nan)
            else:
                areas.append(
                    0.5
                    * (
                        _binary_auc(y[selected] == classes[i], score[selected, i], w[selected], None)
                        + _binary_auc(y[selected] == classes[j], score[selected, j], w[selected], None)
                    )
                )
            supports.append(w[selected].sum())
    return _average(areas, supports, average)

def one_vs_rest_auc(targets, probabilities, weights):
    """
    Compute weighted ROC areas for classes with signal and background yield

    Args:
        targets:  Event class indices
        probabilities:  Class probabilities in output order
        weights:  Event weights
    Returns:
        areas:  Defined one-vs-rest areas by class index
        macro:  Mean of defined areas or None
    """
    weights = np.asarray(weights, dtype=float)
    areas = {}
    for index, scores in enumerate(probabilities.T):
        signal = targets == index
        if weights[signal].sum() != 0 and weights[~signal].sum() != 0:
            areas[index] = float(roc_auc_score(signal, scores, sample_weight=weights))
    return areas, float(np.mean(list(areas.values()))) if areas else None

def average_precision_score(y_true, y_score, *, average="macro", pos_label=1, sample_weight=None, labels=None):
    """
    Integrate precision over signed recall increments without interpolation

    Args:
        y_true:  Binary or single-label multiclass targets
        y_score:  Binary scores or multiclass probability columns
        average:  None, macro, weighted or micro for multiclass targets
        pos_label:  Signal label for binary scores
        sample_weight:  Signed event weights or None
        labels:  Multiclass probability-column labels
    Returns:
        score: Average precision, possibly outside [0,1] for signed weights
    """
    if np.asarray(y_score).ndim == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        delta = -np.diff(recall)
        active = delta != 0
        return float(np.sum(delta[active] * precision[:-1][active]))
    y, score, classes, w = _class_inputs(y_true, y_score, labels, sample_weight)
    binary = y[:, None] == classes
    if average == "micro":
        return average_precision_score(binary.ravel(), score.ravel(), sample_weight=np.repeat(w, len(classes)))
    return _average(
        [average_precision_score(binary[:, i], score[:, i], sample_weight=w) for i in range(len(classes))],
        [w[y == label].sum() for label in classes],
        average,
    )

def confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None):
    """
    Accumulate signed event yields by true and predicted class

    Args:
        y_true:  True event labels
        y_pred:  Predicted event labels
        labels:  Ordered output labels or sorted observed union
        sample_weight:  Signed event weights or None
        normalize:  None, true, pred or all
    Returns:
        matrix: Signed counts or normalized yields with NaN for zero totals
    """
    y, pred = _vector(y_true, "y_true"), _vector(y_pred, "y_pred")
    if y.shape != pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    classes = np.unique(np.r_[y, pred]) if labels is None else _vector(labels, "labels")
    if len(np.unique(classes)) != len(classes):
        raise ValueError("labels must be unique")
    w = _weights(sample_weight, len(y))
    mapping = {label: i for i, label in enumerate(classes)}
    row = np.array([mapping.get(label, -1) for label in y])
    col = np.array([mapping.get(label, -1) for label in pred])
    selected = (row >= 0) & (col >= 0)
    result = np.zeros((len(classes), len(classes)), dtype=float)
    np.add.at(result, (row[selected], col[selected]), w[selected])
    if normalize is None:
        return result
    if normalize == "true":
        return _ratio(result, result.sum(axis=1, keepdims=True))
    if normalize == "pred":
        return _ratio(result, result.sum(axis=0, keepdims=True))
    if normalize == "all":
        return _ratio(result, result.sum())
    raise ValueError("normalize must be None, 'true', 'pred' or 'all'")

def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    """
    Compute signed correctly classified yield or its fraction

    Args:
        y_true:  True labels
        y_pred:  Predicted labels
        normalize:  Divide by the total signed yield
        sample_weight:  Signed event weights or None
    Returns:
        accuracy: Correct yield or signed accuracy
    """
    y, pred = _vector(y_true, "y_true"), _vector(y_pred, "y_pred")
    if y.shape != pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    w = _weights(sample_weight, len(y))
    correct = w[y == pred].sum()
    return float(_ratio(correct, w.sum())) if normalize else float(correct)

def confusion_matrix_statistics(y_true, y_pred, *, labels=None, sample_weight=None):
    """
    Compute efficiency and purity matrices with approximate one-sigma intervals

    Nonnegative weights use Wilson intervals with effective counts from sumw2
    Signed weights use independent-event ratio propagation including covariance

    Args:
        y_true:  True class labels
        y_pred:  Predicted class labels
        labels:  Ordered class labels
        sample_weight:  Signed event weights or None
    Returns:
        statistics: Signed counts, sumw2, rates, lower and upper interval bounds
    """
    y = _vector(y_true, "y_true")
    w = _weights(sample_weight, len(y))
    cm = confusion_matrix(y, y_pred, labels=labels, sample_weight=w)
    q = confusion_matrix(y, y_pred, labels=labels, sample_weight=w**2)
    signed = bool((w < 0).any())
    result = {
        "counts": cm,
        "sumw2": q,
        "interval_method": "signed_ratio_delta" if signed else "effective_count_wilson",
        "confidence_level": 0.6826894921370859,
    }
    for prefix, axis in (("row", 1), ("column", 0)):
        total = cm.sum(axis=axis, keepdims=True)
        total2 = q.sum(axis=axis, keepdims=True)
        rate = np.divide(cm, total, out=np.full_like(cm, np.nan), where=total != 0)
        if signed:
            variance = np.divide(
                (1 - rate) ** 2 * q + rate**2 * np.maximum(total2 - q, 0), total**2, out=np.full_like(cm, np.nan), where=total != 0
            )
            lower, upper = rate - np.sqrt(variance), rate + np.sqrt(variance)
        else:
            inverse = np.divide(total2, total**2, out=np.full_like(total, np.nan), where=total != 0)
            center = (rate + inverse / 2) / (1 + inverse)
            half = np.sqrt(rate * (1 - rate) * inverse + inverse**2 / 4) / (1 + inverse)
            lower, upper = np.maximum(0, center - half), np.minimum(1, center + half)
        result.update(
            {f"{prefix}_rate": rate, f"{prefix}_lower": lower, f"{prefix}_upper": upper, f"{prefix}_denominator": total.ravel()}
        )
    return result

def roc_curve_bootstrap(
    y_true,
    y_score,
    *,
    pos_label=None,
    sample_weight=None,
    n_resamples=200,
    confidence_level=0.6826894921370859,
    random_state=101,
    include_precision_recall=False,
    progress=False,
    replica_efficiencies=None,
):
    """
    Estimate pointwise ROC and AUC percentile intervals by resampling events

    Each replica draws N events with replacement, preserving signed weights
    Nonnegative weights use a common FPR grid. Signed weights use original
    score thresholds because their ROC coordinates need not be monotonic

    Args:
        y_true:  Binary class labels
        y_score:  Signal scores
        pos_label:  Signal label
        sample_weight:  Signed event weights or None
        n_resamples:  Number of independent bootstrap replicas, at least two
        confidence_level:  Central percentile coverage
        random_state:  NumPy generator or reproducible random seed
        include_precision_recall:  Also compute precision bands with the same replicas
        progress:  Show bootstrap replica progress
        replica_efficiencies:  Signal efficiencies for scalar replicas or None to omit replicas
    Returns:
        band: Curve bounds, AUC bounds and valid replica count
    """
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer >= 2")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0,1)")
    fp, tp, thresholds = _binary_counts(y_true, y_score, pos_label, sample_weight)
    y = np.asarray(y_true)
    w = _weights(sample_weight, len(y))
    signal = y == (1 if pos_label is None else pos_label)
    scores = np.asarray(y_score, dtype=float)
    order = np.argsort(-scores, kind="stable")
    # Evaluate replicas at the same score thresholds, grouping score ties
    ends = np.searchsorted(-scores[order], -thresholds, side="right") - 1
    signed = bool((w < 0).any())
    grid = (
        np.divide(np.r_[0, fp], fp[-1], out=np.full(len(fp) + 1, np.nan), where=fp[-1] != 0)
        if signed
        else np.unique(np.r_[0, np.geomspace(1e-6, 1, 201), np.linspace(0, 1, 201)])
    )
    rng = numpy_generator(random_state)
    bands, fprs, areas = [], [], []
    pr_bands = []
    recall_grid = (
        np.divide(np.r_[0, tp], tp[-1], out=np.full(len(tp) + 1, np.nan), where=tp[-1] != 0)
        if signed
        else np.unique(np.r_[0, np.geomspace(1e-6, 1, 201), np.linspace(0, 1, 201)])
    )
    probability = np.full(len(y), 1 / len(y))
    positive = signal[order]
    replicas = None
    if replica_efficiencies is not None:
        replicas = {"auc": np.full(n_resamples, np.nan)}
        if include_precision_recall:
            replicas["average_precision"] = np.full(n_resamples, np.nan)
        if replica_efficiencies:
            replicas["rejection"] = np.full((n_resamples, len(replica_efficiencies)), np.nan)
    from tqdm.auto import tqdm

    for index in tqdm(
        range(n_resamples),
        desc="ROC bootstrap",
        unit="replica",
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True,
        disable=not progress,
    ):
        multiplicity = rng.multinomial(len(y), probability)
        weights = (w * multiplicity)[order]
        btp = np.cumsum(weights * positive)[ends]
        bfp = np.cumsum(weights * ~positive)[ends]
        if btp[-1] == 0:
            continue
        tpr = np.r_[0, btp / btp[-1]]
        if include_precision_recall:
            precision = np.r_[1.0, np.divide(btp, btp + bfp, out=np.full_like(btp, np.nan), where=(btp + bfp) != 0)]
            if replicas is not None:
                steps = np.diff(tpr)
                active = steps != 0
                replicas["average_precision"][index] = np.sum(steps[active] * precision[1:][active])
            if signed:
                pr_bands.append(precision)
            else:
                finite = np.isfinite(precision)
                pr_bands.append(np.interp(recall_grid, tpr[finite], precision[finite]))
        if bfp[-1] == 0:
            continue
        fpr = np.r_[0, bfp / bfp[-1]]
        area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
        areas.append(area)
        if replicas is not None:
            replicas["auc"][index] = area
            for column, efficiency in enumerate(replica_efficiencies):
                point = np.argmax(tpr >= efficiency)
                background = fpr[point]
                if 0 <= background <= 1 and efficiency <= tpr[point] <= 1:
                    replicas["rejection"][index, column] = 1 / background if background > 0 else np.inf
        bands.append(tpr if signed else np.interp(grid, fpr, tpr))
        if signed:
            fprs.append(fpr)
    alpha = (1 - confidence_level) / 2
    valid = len(areas)
    if valid >= 2:
        lower, upper = np.quantile(bands, [alpha, 1 - alpha], axis=0)
        auc_lower, auc_upper = np.quantile(areas, [alpha, 1 - alpha])
    else:
        lower = upper = np.full(grid.shape, np.nan)
        auc_lower = auc_upper = np.nan
    result = dict(
        band_fpr=grid,
        tpr_lower=lower,
        tpr_upper=upper,
        auc_lower=float(auc_lower),
        auc_upper=float(auc_upper),
        valid_resamples=valid,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="threshold_percentile" if signed else "fixed_fpr_percentile",
    )
    if signed:
        result["fpr_lower"], result["fpr_upper"] = (
            np.quantile(fprs, [alpha, 1 - alpha], axis=0)
            if valid >= 2
            else (np.full(grid.shape, np.nan), np.full(grid.shape, np.nan))
        )
    if include_precision_recall:
        lower, upper, counts = _pointwise_intervals(pr_bands, len(recall_grid), confidence_level)
        result.update(pr_band_recall=recall_grid, pr_precision_lower=lower, pr_precision_upper=upper, pr_valid_resamples=counts)
    if replicas is not None:
        result["replicas"] = replicas
    return result

def _pointwise_intervals(replicas, size, confidence_level):
    """
    Extract percentile intervals excluding undefined replica coordinates

    Args:
        replicas:  Replica arrays with NaN for zero signed denominators
        size:  Number of curve coordinates
        confidence_level:  Central percentile coverage
    Returns:
        intervals: Lower bounds, upper bounds and valid counts per coordinate
    """
    values = np.asarray(replicas, dtype=float).reshape(-1, size)
    counts = np.isfinite(values).sum(axis=0)
    lower, upper = np.full(size, np.nan), np.full(size, np.nan)
    valid = counts >= 2
    if valid.any():
        alpha = (1 - confidence_level) / 2
        lower[valid], upper[valid] = np.nanquantile(values[:, valid], [alpha, 1 - alpha], axis=0)
    return lower, upper, counts

def precision_recall_curve_bootstrap(
    y_true, y_score, *, pos_label=None, sample_weight=None, n_resamples=200, confidence_level=0.6826894921370859, random_state=101
):
    """
    Estimate pointwise precision intervals by paired event resampling

    Args:
        y_true:  Binary class labels
        y_score:  Signal scores
        pos_label:  Signal label
        sample_weight:  Signed event weights or None
        n_resamples:  Number of bootstrap replicas, at least two
        confidence_level:  Central percentile coverage
        random_state:  NumPy generator or reproducible random seed
    Returns:
        band: Precision bounds at fixed recall or signed score thresholds
    """
    result = roc_curve_bootstrap(
        y_true,
        y_score,
        pos_label=pos_label,
        sample_weight=sample_weight,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=random_state,
        include_precision_recall=True,
    )
    return {
        **{key.removeprefix("pr_"): value for key, value in result.items() if key.startswith("pr_")},
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "method": ("threshold_percentile" if result["method"] == "threshold_percentile" else "fixed_recall_percentile"),
    }

def calibration_curve_bootstrap(
    y_true,
    y_prob,
    *,
    sample_weight=None,
    n_bins=10,
    n_resamples=200,
    confidence_level=0.6826894921370859,
    random_state=101,
    progress=False,
):
    """
    Estimate binary calibration and pointwise intervals in fixed probability bins

    Args:
        y_true:  Binary class indicators or top-label correctness
        y_prob:  Class probability or top-label confidence in [0,1]
        sample_weight:  Signed event weights or None
        n_bins:  Number of uniform probability bins
        n_resamples:  Paired event replicas, zero returns only central values
        confidence_level:  Central percentile coverage
        random_state:  NumPy generator or reproducible random seed
        progress:  Show bootstrap replica progress
    Returns:
        curve: Bin coordinates, yield, percentile bounds and valid replica counts
    """
    y = _vector(y_true, "y_true")
    p = _vector(y_prob, "y_prob").astype(float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Calibration requires binary labels and matching probabilities in [0,1]")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 0 or n_resamples == 1:
        raise ValueError("n_resamples must be zero or an integer >= 2")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0,1)")
    w = _weights(sample_weight, len(y))
    edges = np.linspace(0, 1, n_bins + 1)
    index = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)

    def coordinates(weights):
        """
        Normalize weighted class yield and probability in fixed bins

        Args:
            weights:  Replica event weights
        Returns:
            values: Bin yield, mean predicted probability and observed fraction
        """
        total = np.bincount(index, weights=weights, minlength=n_bins)
        means = [
            np.divide(
                np.bincount(index, weights=weights * v, minlength=n_bins), total, out=np.full(n_bins, np.nan), where=total != 0
            )
            for v in (p, y)
        ]
        return total, *means

    total, mean, fraction = coordinates(w)
    rng = numpy_generator(random_state)
    predicted, observed = [], []
    probability = np.full(len(y), 1 / len(y))
    from tqdm.auto import tqdm

    for _ in tqdm(
        range(n_resamples),
        desc="Calibration bootstrap",
        unit="replica",
        bar_format="{l_bar}{bar:20}{r_bar}",
        dynamic_ncols=True,
        disable=not progress,
    ):
        _, bp, by = coordinates(w * rng.multinomial(len(y), probability))
        predicted.append(bp)
        observed.append(by)
    lower, upper, counts = _pointwise_intervals(observed, n_bins, confidence_level)
    xlower, xupper, _ = _pointwise_intervals(predicted, n_bins, confidence_level)
    return dict(
        bin_edges=edges,
        prob_pred=mean,
        prob_true=fraction,
        weight=total,
        prob_true_lower=lower,
        prob_true_upper=upper,
        prob_pred_lower=xlower,
        prob_pred_upper=xupper,
        valid_resamples=counts,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="fixed_bin_percentile",
    )
