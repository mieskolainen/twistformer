# Paired model uncertainties on common event samples
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import numpy as np
from tqdm.auto import tqdm
from twistformer.randomness import numpy_generator
from twistformer.metrics import roc_curve, one_vs_rest_auc

def scores(data, probabilities, spec, efficiencies):
    """
    Compute AUC and physical rejection for paired event replicas

    Args:
        data: Common targets and physical weights
        probabilities: Class probabilities in the common order
        spec: Class schema
        efficiencies: Requested signal efficiencies
    Returns:
        values: Finite scalar metrics, with NaN for undefined values
    """
    if spec.name != "classification":
        from .results import compute_metrics

        values, _ = compute_metrics({**data, **probabilities}, prediction_spec(spec, probabilities))
        result = {
            f"{key}:{name}": float(row[key]) if row.get(key) is not None else np.nan
            for name in spec.outputs
            for row in [values[name]]
            for key in ("mae", "rmse", "bias", "nll", "coverage_1sigma", "coverage_2sigma")
        }
        if "log_prob" in probabilities:
            value = values.get("density_nll")
            result["density_nll"] = np.nan if value is None else value
        return result
    if isinstance(probabilities, dict):
        probabilities = probabilities["probabilities"]
    values = {}
    y, w = data["targets"], data["weights"]
    aucs, macro = one_vs_rest_auc(y, probabilities, w)
    for index, name in enumerate(spec.outputs):
        binary = y == index
        rejection = [np.nan] * len(efficiencies)
        if index in aucs:
            fpr, tpr, _ = roc_curve(binary, probabilities[:, index], sample_weight=w, drop_intermediate=False)
            for j, efficiency in enumerate(efficiencies):
                crossing = np.flatnonzero(tpr >= efficiency)
                if len(crossing):
                    k = crossing[0]
                    if 0 < fpr[k] <= 1 and 0 <= tpr[k] <= 1:
                        rejection[j] = 1 / fpr[k]
        values[f"auc:{name}"] = aucs.get(index, np.nan)
        for efficiency, value in zip(efficiencies, rejection):
            values[f"rejection:{name}:{efficiency:g}"] = value
    values["macro_auc"] = macro if macro is not None else np.nan
    return values

def paired_metrics(data, predictions, spec, labels, efficiencies, config):
    """
    Estimate model differences with the same bootstrap event replicas

    Args:
        data: Common event metadata for this selection
        predictions: Class probability arrays per model
        spec: Class schema
        labels: Ordered model labels
        efficiencies: Requested signal efficiencies
        config: Bootstrap sample count, seed and confidence
    Returns:
        result: Reference/model metric differences and rejection ratios
    """
    observed = [scores(data, p, spec, efficiencies) for p in predictions]
    keys = list(dict.fromkeys(key for row in observed for key in row))
    point = np.asarray([[row.get(key, np.nan) for key in keys] for row in observed])
    count = config["bootstrap_samples"]
    replicas = np.full((count, len(labels), len(keys)), np.nan)
    signed_density = "density_nll" in keys and np.any(data["weights"] < 0)
    generator = numpy_generator(config["bootstrap_seed"])
    for replica in tqdm(
        range(count), desc="Paired bootstrap", unit="replica", disable=count == 0, bar_format="{l_bar}{bar:20}{r_bar}"
    ):
        indices = generator.integers(len(data["targets"]), size=len(data["targets"]))
        selected = {key: data[key][indices] for key in ("targets", "weights", "target_mask") if key in data}
        if not np.any(selected["weights"]):
            continue
        for model, p in enumerate(predictions):
            view = {key: value[indices] for key, value in p.items()} if isinstance(p, dict) else p[indices]
            row = scores(selected, view, spec, efficiencies)
            if signed_density:
                row["density_nll"] = np.nan
            replicas[replica, model] = [row.get(key, np.nan) for key in keys]
    tail = (1 - config["bootstrap_confidence"]) / 2
    result = {}
    for reference, reference_label in enumerate(labels):
        result[reference_label] = {}
        for model, label in enumerate(labels):
            values = {}
            for column, key in enumerate(keys):
                ratio = key.startswith("rejection:")
                if ratio:
                    estimate = point[model, column] / point[reference, column]
                    samples = replicas[:, model, column] / replicas[:, reference, column]
                else:
                    estimate = point[model, column] - point[reference, column]
                    samples = replicas[:, model, column] - replicas[:, reference, column]
                valid = samples[np.isfinite(samples)]
                interval = np.quantile(valid, [tail, 1 - tail]) if len(valid) >= 2 else [None, None]
                values[key] = {
                    "kind": "ratio" if ratio else "difference",
                    "estimate": float(estimate) if np.isfinite(estimate) else None,
                    "lower": interval[0],
                    "upper": interval[1],
                    "valid_resamples": len(valid),
                    "requested_resamples": count,
                    "confidence": config["bootstrap_confidence"],
                }
            result[reference_label][label] = values
    return result

def prediction_spec(spec, data):
    """
    Select point or probabilistic regression diagnostics from physical outputs

    Args:
        spec: Common task schema
        data: Model prediction arrays
    Returns:
        schema: Compatible task schema for shared metrics
    """
    if spec.name == "classification":
        return spec
    if "samples" in data:
        return replace(spec, name="flow_matching")
    return replace(spec, name="gaussian_regression" if "std" in data else "covariant_regression")

def regression_curves(data, spec, metrics):
    """
    Build common residual histograms and target-dependent regression diagnostics

    Args:
        data: Aligned physical prediction arrays per model
        spec: Common output schema
        metrics: Per-model regression summaries
    Returns:
        curves: Shared-bin residual density and target profiles per model
    """
    result = [{} for _ in data]
    for index, name in enumerate(spec.outputs):
        valid = (
            np.isfinite(data[0]["targets"][:, index])
            & data[0].get("target_mask", np.ones_like(data[0]["targets"], dtype=bool))[:, index]
        )
        valid &= data[0]["weights"] != 0
        residuals = [row["mean"][valid, index] - row["targets"][valid, index] for row in data]
        if not valid.any():
            continue
        low = min(values.min() for values in residuals)
        high = max(values.max() for values in residuals)
        if low == high:
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, 41)
        weights = data[0]["weights"][valid]
        for model, residual in enumerate(residuals):
            density = np.histogram(residual, bins=edges, weights=weights)[0] / (np.abs(weights).sum() * np.diff(edges))
            bins = metrics[model][name].get("bins", [])
            curve = {"residual": {"x": ((edges[:-1] + edges[1:]) / 2).tolist(), "y": density.tolist()}}
            for key in ("rmse", "coverage_1sigma"):
                points = [(row["target"], row[key]) for row in bins if row.get("target") is not None and row.get(key) is not None]
                curve[key] = {"x": [point[0] for point in points], "y": [point[1] for point in points]}
            result[model][str(index)] = curve
    return result
