# Prediction diagnostics and evaluation
#
# mikael.mieskolainen@cern.ch (2026)

"""Metrics operate on exported predictions in original target units"""

import math
import numpy as np
from twistformer.randomness import numpy_generator
from twistformer.metrics import (
    roc_curve,
    roc_auc_score,
    one_vs_rest_auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    confusion_matrix_statistics,
    roc_curve_bootstrap,
    calibration_curve_bootstrap,
)

def event_weights(data):
    """
    Validate signed event weights for physics diagnostics

    Args:
        data:  Exported prediction arrays
    Returns:
        weights: Finite one-dimensional array with nonzero absolute weight
    """
    weights = np.asarray(data["weights"], dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.abs(weights).sum() == 0:
        raise ValueError("Evaluation weights must be finite and have nonzero absolute total")
    return weights

def _mean(values, weights):
    """
    Compute a signed mean with an explicit undefined zero normalization

    Args:
        values:  Event diagnostics
        weights:  Signed event weights
    Returns:
        mean: Finite signed mean or None
    """
    return _finite(np.dot(values, weights) / weights.sum()) if weights.sum() != 0 else None

def _finite(value):
    """
    Map undefined diagnostics to JSON null

    Args:
        value:  Floating diagnostic
    Returns:
        value: Finite float or None
    """
    return float(value) if value is not None and np.isfinite(value) else None

def _sqrt(value):
    """
    Preserve undefined square roots of negative signed moments

    Args:
        value:  Signed second moment
    Returns:
        root: Real square root or None
    """
    return math.sqrt(value) if value is not None and value >= 0 else None


def bootstrap_summary(replicas, confidence, *, allow_unbounded=False):
    """
    Summarize a scalar observable with event bootstrap statistics

    Args:
        replicas:  Scalar replicas with NaN for undefined estimates
        confidence:  Central percentile coverage
        allow_unbounded:  Include infinite rejection from zero background
    Returns:
        summary:  Standard error, percentile interval and replica counts
    """
    valid = replicas[~np.isnan(replicas) if allow_unbounded else np.isfinite(replicas)]
    unbounded = int(np.count_nonzero(np.isposinf(valid)))
    tail = (1 - confidence) / 2
    # Use empirical quantiles when interpolation can involve infinite rejection
    method = "inverted_cdf" if unbounded else "linear"
    lower, upper = np.quantile(valid, (tail, 1 - tail), method=method) if len(valid) >= 2 else (None, None)
    summary = {
        "std": _finite(valid.std(ddof=1)) if len(valid) >= 2 and not unbounded else None,
        "lower": _finite(lower),
        "upper": _finite(upper),
        "confidence": confidence,
        "valid_resamples": len(valid),
        "requested_resamples": len(replicas),
    }
    if unbounded:
        summary.update(
            zero_background_resamples=unbounded,
            lower_unbounded=bool(lower == np.inf),
            upper_unbounded=bool(upper == np.inf),
        )
    return summary


def classification_scalar_replicas(values, weights, bin_index, confidence, count, seed, loss_terms):
    """
    Resample the weighted means and top-label calibration error

    Args:
        values:  Event accuracy, cross entropy and Brier scores in column order
        weights:  Physical event weights
        bin_index:  Top-label calibration bin per event
        confidence:  Top-label probability per event
        count:  Number of event bootstrap replicas
        seed:  Bootstrap seed
        loss_terms:  Event loss numerator and normalizer or None
    Returns:
        replicas:  Columns for accuracy, cross entropy, Brier score, ECE and optional evaluation loss
    """
    rng = numpy_generator(seed)
    probability = np.full(len(weights), 1 / len(weights))
    residual = values[:, 0] - confidence
    signed = (weights < 0).any()
    replicas = np.full((count, 4 if loss_terms is None else 5), np.nan)
    for index in range(count):
        multiplicity = rng.multinomial(len(weights), probability)
        if loss_terms is not None:
            numerator, normalizer = loss_terms
            total_loss_weight = multiplicity @ normalizer
            if total_loss_weight != 0:
                replicas[index, 4] = (multiplicity @ numerator) / total_loss_weight
        sampled = weights * multiplicity
        total = sampled.sum()
        if total == 0:
            continue
        replicas[index, :3] = sampled @ values / total
        mass = np.bincount(bin_index, weights=sampled)
        if signed and np.any((mass == 0) & (np.bincount(bin_index, weights=np.abs(sampled)) > 0)):
            continue
        gap = np.bincount(bin_index, weights=sampled * residual)
        replicas[index, 3] = (np.abs(gap) @ np.sign(mass)) / total
    return replicas


def classification_metrics(
    data,
    spec,
    efficiencies=(0.5, 0.8),
    *,
    bootstrap_samples=0,
    bootstrap_seed=101,
    bootstrap_confidence=0.6826894921370859,
    progress=False,
):
    """
    Compute weighted one-vs-rest and class-versus-class diagnostics

    Args:
        data:  Prediction, label and event-weight arrays
        spec:  Classification TaskSpec
        efficiencies:  Signal efficiencies for ROC operating points
        bootstrap_samples:  Event bootstrap replicas, zero disables bands
        bootstrap_seed:  Reproducible event resampling seed
        bootstrap_confidence:  Central percentile coverage
        progress:  Show bootstrap replica progress
    Returns:
        result: Metrics dictionary and ROC/PR curve arrays
    """
    if not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 0 or bootstrap_samples == 1:
        raise ValueError("bootstrap_samples must be zero or an integer >= 2")
    if not 0 < bootstrap_confidence < 1:
        raise ValueError("bootstrap_confidence must be in (0,1)")
    y, p, w = data["targets"], data["probabilities"], event_weights(data)
    c = len(spec.outputs)
    if y.shape != w.shape or not np.isfinite(y).all() or (y != np.floor(y)).any() or (y < 0).any() or (y >= c).any():
        raise ValueError("Invalid classification targets")
    y = y.astype(int)
    if (
        p.shape != (len(y), c)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(p.sum(-1), 1, atol=1e-6)
    ):
        raise ValueError("Classification probabilities must be normalized finite arrays [events,classes]")
    if any(not 0 < v <= 1 for v in efficiencies):
        raise ValueError("Signal efficiencies must be in (0,1]")
    pred = p.argmax(-1)
    correct = pred == y
    cross_entropy = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    brier = ((p - np.eye(c)[y]) ** 2).sum(-1)
    results = {
        "accuracy": _mean(correct, w),
        "cross_entropy": _mean(cross_entropy, w),
        "brier": _mean(brier, w),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(c)), sample_weight=w).tolist(),
        "classes": {},
    }
    loss_terms = None
    if "evaluation_loss_numerator" in data and "evaluation_loss_normalizer" in data:
        numerator = np.asarray(data["evaluation_loss_numerator"], dtype=float)
        normalizer = np.asarray(data["evaluation_loss_normalizer"], dtype=float)
        if any(values.shape != w.shape or not np.isfinite(values).all() for values in (numerator, normalizer)):
            raise ValueError("Evaluation loss terms must be finite event vectors")
        loss_terms = numerator, normalizer
        total_loss_weight = normalizer.sum()
        results["preprocessed_loss"] = _finite(numerator.sum() / total_loss_weight) if total_loss_weight != 0 else None
    statistics = confusion_matrix_statistics(y, pred, labels=list(range(c)), sample_weight=w)
    results["confusion_statistics"] = {
        key: (np.where(np.isfinite(value), value, None).tolist() if isinstance(value, np.ndarray) else value)
        for key, value in statistics.items()
    }
    curves = {}
    aucs, results["macro_auc"] = one_vs_rest_auc(y, p, w)
    for index, name in enumerate(spec.outputs):
        binary = y == index
        if index not in aucs:
            results["classes"][name] = {"auc": None, "reason": "Zero net signal or background weight"}
            continue
        fpr, tpr, thresholds = roc_curve(binary, p[:, index], sample_weight=w, drop_intermediate=False)
        precision, recall, _ = precision_recall_curve(binary, p[:, index], sample_weight=w)
        results["classes"][name] = {
            "auc": aucs[index],
            "average_precision": _finite(average_precision_score(binary, p[:, index], sample_weight=w)),
        }
        curves[index] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "precision": precision, "recall": recall}
    if c > 2:
        for index, name in enumerate(spec.outputs):
            binary = y == index
            backgrounds = results["classes"][name]["background_auc"] = {}
            for background, background_name in enumerate(spec.outputs):
                if background == index:
                    continue
                selected = binary | (y == background)
                if w[binary].sum() == 0 or w[y == background].sum() == 0:
                    backgrounds[background_name] = None
                    continue
                pair_fpr, pair_tpr, pair_thresholds = roc_curve(
                    binary[selected], p[selected, index], sample_weight=w[selected], drop_intermediate=False
                )
                backgrounds[background_name] = float(roc_auc_score(binary[selected], p[selected, index], sample_weight=w[selected]))
                curves.setdefault(index, {}).update(
                    {
                        f"background_{background}_fpr": pair_fpr,
                        f"background_{background}_tpr": pair_tpr,
                        f"background_{background}_thresholds": pair_thresholds,
                    }
                )
    # Descriptive operating points use each class as the signal
    for index, curve in curves.items():
        if "tpr" not in curve:
            continue
        points = []
        for efficiency in efficiencies:
            # First threshold crossing in score order also handles signed ROC loops
            j = int(np.flatnonzero(curve["tpr"] >= efficiency)[0])
            fpr = float(curve["fpr"][j])
            points.append(
                {
                    "requested_efficiency": efficiency,
                    "efficiency": float(curve["tpr"][j]),
                    "background_efficiency": fpr,
                    "rejection": 1 / fpr if fpr > 0 else None,
                    "threshold": _finite(curve["thresholds"][j]),
                    "physical": 0 <= fpr <= 1 and 0 <= float(curve["tpr"][j]) <= 1,
                    "zero_background": fpr == 0,
                }
            )
        results["classes"][spec.outputs[index]]["operating_points"] = points
        if index == spec.signal_class:
            results["signal_operating_points"] = points
    confidence = p.max(-1)
    calibration = []
    bin_index = np.minimum(np.searchsorted(np.linspace(0, 1, 11), confidence, side="right") - 1, 9)
    for index in range(10):
        selected = bin_index == index
        if np.abs(w[selected]).sum() > 0:
            calibration.append(
                {
                    "confidence": _mean(confidence[selected], w[selected]),
                    "accuracy": _mean(correct[selected], w[selected]),
                    "weight": float(w[selected].sum()),
                }
            )
    results["calibration"] = calibration
    calibration_curves = {}
    for name, binary, probability in [(name, y == index, p[:, index]) for index, name in enumerate(spec.outputs)] + [
        (None, correct, confidence)
    ]:
        curve = calibration_curve_bootstrap(
            binary,
            probability,
            sample_weight=w,
            n_resamples=bootstrap_samples,
            random_state=bootstrap_seed,
            confidence_level=bootstrap_confidence,
            progress=progress,
        )
        calibration_curves[name] = {
            key: (np.where(np.isfinite(value), value, None).tolist() if isinstance(value, np.ndarray) else value)
            for key, value in curve.items()
        }
    results["calibration_top_label"] = calibration_curves.pop(None)
    results["calibration_classes"] = calibration_curves
    results["ece"] = (
        _finite(sum(abs(r["accuracy"] - r["confidence"]) * r["weight"] for r in calibration) / w.sum())
        if w.sum() != 0 and all(r["accuracy"] is not None and r["confidence"] is not None for r in calibration)
        else None
    )
    results["weight_summary"] = {
        "sum": float(w.sum()),
        "sum_abs": float(np.abs(w).sum()),
        "sum_squared": float(np.dot(w, w)),
        "negative_events": int((w < 0).sum()),
    }
    if bootstrap_samples:
        mean_replicas = classification_scalar_replicas(
            np.column_stack((correct, cross_entropy, brier)), w, bin_index, confidence, bootstrap_samples, bootstrap_seed,
            loss_terms,
        )
        mean_keys = ("accuracy", "cross_entropy", "brier", "ece")
        if loss_terms is not None:
            mean_keys += ("preprocessed_loss",)
        uncertainty = results["uncertainties"] = {
            key: bootstrap_summary(values, bootstrap_confidence)
            for key, values in zip(mean_keys, mean_replicas.T, strict=True)
            if results[key] is not None
        }
        auc_replicas = []
        results["roc_bootstrap"] = {
            "n_resamples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": bootstrap_confidence,
        }
        for index, curve in curves.items():
            subsets = [("", None, y == index, p[:, index], w)] if "fpr" in curve else []
            for background in range(c):
                if f"background_{background}_fpr" in curve:
                    selected = (y == index) | (y == background)
                    subsets.append((f"background_{background}_", spec.outputs[background],
                                    y[selected] == index, p[selected, index], w[selected]))
            class_result = results["classes"][spec.outputs[index]]
            intervals = class_result["auc_intervals"] = {}
            class_uncertainty = class_result["uncertainties"] = {}
            for prefix, background_name, binary, scores, weights in subsets:
                band = roc_curve_bootstrap(
                    binary,
                    scores,
                    sample_weight=weights,
                    n_resamples=bootstrap_samples,
                    random_state=bootstrap_seed,
                    confidence_level=bootstrap_confidence,
                    include_precision_recall=not prefix,
                    progress=progress,
                    replica_efficiencies=tuple(efficiencies) if not prefix else (),
                )
                replicas = band["replicas"]
                auc_uncertainty = bootstrap_summary(replicas["auc"], bootstrap_confidence)
                if prefix:
                    class_uncertainty.setdefault("background_auc", {})[background_name] = auc_uncertainty
                else:
                    auc_replicas.append(replicas["auc"])
                    class_uncertainty["auc"] = auc_uncertainty
                    if class_result["average_precision"] is not None:
                        class_uncertainty["average_precision"] = bootstrap_summary(replicas["average_precision"], bootstrap_confidence)
                    if efficiencies:
                        for point, rejection in zip(class_result["operating_points"], replicas["rejection"].T, strict=True):
                            if point["physical"]:
                                point["uncertainties"] = {
                                    "rejection": bootstrap_summary(rejection, bootstrap_confidence, allow_unbounded=True)
                                }
                for key in (
                    "band_fpr",
                    "tpr_lower",
                    "tpr_upper",
                    "fpr_lower",
                    "fpr_upper",
                    "pr_band_recall",
                    "pr_precision_lower",
                    "pr_precision_upper",
                    "pr_valid_resamples",
                ):
                    if key in band:
                        curve[prefix + key] = band[key]
                intervals[prefix.rstrip("_") or "rest"] = {
                    key: (_finite(band[key]) if key in {"auc_lower", "auc_upper"} else band[key])
                    for key in ("auc_lower", "auc_upper", "valid_resamples", "n_resamples", "confidence_level", "method")
                }
        if auc_replicas:
            uncertainty["macro_auc"] = bootstrap_summary(np.mean(auc_replicas, axis=0), bootstrap_confidence)
    return results, curves

def regression_metrics(data, spec):
    """
    Evaluate regression residuals and optional Gaussian uncertainty diagnostics

    Args:
        data:  Physical targets, means, optional standard deviations and event weights
        spec:  Regression TaskSpec
    Returns:
        metrics: Per-target residual, pull and coverage diagnostics
    """
    y, mu, weights = (data["targets"], data["mean"], event_weights(data))
    std = data.get("std")
    shape = (len(weights), len(spec.outputs))
    if y.shape != shape or mu.shape != shape or (std is not None and std.shape != shape):
        raise ValueError("Regression arrays must match [events,outputs]")
    if not np.isfinite(mu).all():
        raise ValueError("Regression predictions require finite means")
    if spec.name == "gaussian_regression" and (std is None or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Gaussian predictions require finite means and positive finite standard deviations")
    if spec.name == "covariant_regression" and std is not None:
        raise ValueError("Covariant regression supplies mean four-vectors without Gaussian widths")
    if "target_mask" in data and (data["target_mask"].shape != shape or data["target_mask"].dtype != np.bool_):
        raise ValueError("target_mask must be Boolean and match regression targets")
    valid = np.isfinite(y) & data.get("target_mask", np.ones_like(y, dtype=bool))
    results = {}
    pit = None
    if "mixture_logits" in data:
        from .mixture import mixture_diagnostics

        nll, pit, _, results["mixture"] = mixture_diagnostics(data)
    for index, name in enumerate(spec.outputs):
        take = valid[:, index] & (weights != 0)
        if not take.any():
            results[name] = {"reason": "No observed targets"}
            continue
        w = weights[take]
        residual = y[take, index] - mu[take, index]
        sigma = std[take, index] if std is not None else None
        pull = residual / sigma if sigma is not None else None
        covered = None if pull is None else abs(pull) <= 1
        if pit is not None:
            u = pit[take, index]
            alpha = 0.5 * math.erfc(1 / math.sqrt(2))
            covered = (u >= alpha) & (u <= 1 - alpha)

        def average(values):
            """
            Average one target diagnostic with its signed event yields

            Args:
                values:  Per-event diagnostic
            Returns:
                mean: Signed mean or None for zero net yield
            """
            return _mean(values, w)

        bins = []
        values = y[take, index]
        boundaries = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
        for j, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            chosen = (values >= lo) & ((values < hi) if j < len(boundaries) - 2 else (values <= hi))
            if not chosen.any():
                continue
            bins.append(
                {
                    "target": _mean(values[chosen], w[chosen]),
                    "rmse": _sqrt(_mean(residual[chosen] ** 2, w[chosen])),
                    **({"coverage_1sigma": _mean(covered[chosen], w[chosen])} if covered is not None else {}),
                }
            )
        results[name] = {
            "mae": average(abs(residual)),
            "rmse": _sqrt(average(residual**2)),
            "bias": average(residual),
            "bins": bins,
        }
        if pull is not None:
            results[name].update(
                {
                    "nll": average(0.5 * pull**2 + np.log(sigma) + 0.5 * math.log(2 * math.pi)),
                    "pull_mean": average(pull),
                    "pull_std": (_sqrt(average((pull - average(pull)) ** 2)) if average(pull) is not None else None),
                    "coverage_1sigma": average(abs(pull) <= 1),
                    "coverage_2sigma": average(abs(pull) <= 2),
                }
            )
        if pit is not None:
            row = results[name]
            row["nll"] = _mean(nll[take, index], w)
            row["pit_mean"] = _mean(u, w)
            for sigma in (1, 2):
                alpha = 0.5 * math.erfc(sigma / math.sqrt(2))
                row[f"coverage_{sigma}sigma"] = _mean((u >= alpha) & (u <= 1 - alpha), w)
    return results
