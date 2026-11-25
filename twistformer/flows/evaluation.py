# Flow density and sample diagnostics
#
# mikael.mieskolainen@cern.ch (2026)

from pathlib import Path
import math
import numpy as np
from twistformer.randomness import numpy_generator


def flow_metrics(data, spec, *, bootstrap_samples=0, bootstrap_seed=101, bootstrap_confidence=0.6826894921370859):
    """
    Evaluate densities and empirical conditional calibration

    Args:
        data:  Physical prediction arrays
        spec:  Flow target schema
        bootstrap_samples:  Number of event resamples
        bootstrap_seed:  Event resampling seed
        bootstrap_confidence:  Central interval probability
    Returns:
        result: JSON-compatible diagnostics
    """
    if bootstrap_samples < 0 or bootstrap_samples == 1 or not 0 < bootstrap_confidence < 1:
        raise ValueError("Invalid flow bootstrap settings")
    weights = np.asarray(data.get("physical_weights", data["weights"]), dtype=float)
    if not np.isfinite(weights).all() or np.abs(weights).sum() == 0:
        raise ValueError("Flow diagnostics require finite weights with nonzero absolute yield")
    target = np.asarray(data["targets"]).reshape(len(weights), -1)
    samples = np.asarray(data["samples"]).reshape(len(weights), data["samples"].shape[1], -1)
    denominator = weights.sum()
    def mean(value):
        take = weights != 0
        result = np.dot(weights[take], value[take]) / denominator if denominator != 0 else np.nan
        return float(result) if np.isfinite(result) else None
    result = {"events": len(weights), "weight_sum": float(denominator), "density_nll": None,
              "bits_per_dimension": None, "density_status": "unavailable", "sample_count": samples.shape[1]}
    result["coverage_definition"] = "equal-tailed empirical marginal intervals"
    result["density_definition"] = "untempered density before sampling constraints"
    if "log_prob" in data and (weights >= 0).all():
        value = mean(-data["log_prob"])
        result.update(density_nll=value, density_status="available" if value is not None else "nonfinite")
        if "dequantization_levels" in data and value is not None:
            result["bits_per_dimension"] = value / (target.shape[1] * math.log(2)) + math.log2(float(data["dequantization_levels"][0]))
    elif (weights < 0).any():
        result["density_status"] = "signed_weights"
    prediction = samples.mean(1)
    for index, name in enumerate(spec.outputs):
        residual = target[:, index] - prediction[:, index]
        low, high = np.quantile(samples[:, :, index], [0.158655, 0.841345], axis=1)
        low2, high2 = np.quantile(samples[:, :, index], [0.02275, 0.97725], axis=1)
        second = mean(residual**2)
        result[name] = {
            "bias": mean(residual), "mae": mean(np.abs(residual)),
            "rmse": math.sqrt(second) if second is not None and second >= 0 else None,
            "nll": None,
            "coverage_1sigma": mean((target[:, index] >= low) & (target[:, index] <= high)),
            "coverage_2sigma": mean((target[:, index] >= low2) & (target[:, index] <= high2)),
            "pit_mean": mean((samples[:, :, index] <= target[:, None, index]).mean(1)),
        }
        edges = np.unique(np.quantile(target[:, index], np.linspace(0, 1, 11)))
        bins = []
        for left, right in zip(edges[:-1], edges[1:]):
            take = (target[:, index] >= left) & (target[:, index] <= right if right == edges[-1] else target[:, index] < right)
            mass = weights[take].sum()
            if not take.any() or mass == 0:
                continue
            second = float(np.dot(weights[take], residual[take]**2) / mass)
            bins.append({"target": float(np.dot(weights[take], target[take, index]) / mass),
                         "rmse": math.sqrt(second) if second >= 0 else None,
                         "coverage_1sigma": float(np.dot(weights[take], (target[take, index] >= low[take]) &
                                                          (target[take, index] <= high[take])) / mass)})
        result[name]["bins"] = bins
    if bootstrap_samples:
        keys = [(None, "density_nll"), (None, "bits_per_dimension")]
        keys += [(name, metric) for name in spec.outputs for metric in ("bias", "mae", "rmse", "coverage_1sigma", "coverage_2sigma", "pit_mean")]
        replicas = {key: [] for key in keys}
        generator = numpy_generator(bootstrap_seed)
        for _ in range(bootstrap_samples):
            indices = generator.integers(len(weights), size=len(weights))
            if np.abs(weights[indices]).sum() == 0:
                continue
            selected = {key: value[indices] for key, value in data.items()
                        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(weights)}
            metrics = flow_metrics(selected, spec)
            for name, key in keys:
                value = metrics[key] if name is None else metrics[name][key]
                if name is None and result["density_status"] == "signed_weights":
                    continue
                if value is not None and np.isfinite(value):
                    replicas[(name, key)].append(value)
        tail = (1 - bootstrap_confidence) / 2
        for (name, key), values in replicas.items():
            lower, upper = np.quantile(values, [tail, 1 - tail]) if len(values) >= 2 else (None, None)
            destination = result if name is None else result[name]
            destination.setdefault("uncertainties", {})[key] = {"lower": lower, "upper": upper,
                "confidence": bootstrap_confidence, "valid_resamples": len(values), "requested_resamples": bootstrap_samples}
    return result


def plot_flow_results(data, spec, output, selection_label=None):
    """
    Plot generated distributions and image samples

    Args:
        data:  Physical target and sample arrays
        spec:  Flow target schema
        output:  Plot directory
        selection_label:  Optional event selection label
    Returns:
        None: PNG and PDF figures are written
    """
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples, targets = data["samples"], data["targets"]
    if targets.ndim == 4:
        count = min(8, len(targets))
        figure, axes = plt.subplots(2, count, figsize=(2 * count, 4), squeeze=False)
        for index in range(count):
            for row, value in enumerate([targets[index], samples[index, 0]]):
                image = value[0] if value.shape[0] == 1 else np.moveaxis(value, 0, -1)
                axes[row, index].imshow(np.clip(image, 0, 1), cmap="gray")
                axes[row, index].axis("off")
        axes[0, 0].set_title("Target")
        axes[1, 0].set_title("Generated")
    else:
        target = targets.reshape(len(targets), -1)
        generated = samples.reshape(len(samples), samples.shape[1], -1)
        count = min(target.shape[-1], 12)
        figure, axes = plt.subplots(math.ceil(count / 3), min(count, 3), figsize=(4 * min(count, 3), 3 * math.ceil(count / 3)), squeeze=False)
        weights = data.get("physical_weights", data["weights"])
        for index, axis in enumerate(axes.flat):
            if index >= count:
                axis.axis("off")
                continue
            values = generated[:, :, index].reshape(-1)
            edges = np.histogram_bin_edges(np.concatenate((target[:, index], values)), bins=40)
            axis.hist(target[:, index], bins=edges, weights=weights, histtype="step", label="Target")
            axis.hist(values, bins=edges, weights=np.repeat(weights / samples.shape[1], samples.shape[1]),
                      histtype="step", label="Generated")
            axis.set_xlabel(spec.outputs[index])
            axis.set_ylabel("Weighted events")
            axis.legend()
    if selection_label:
        figure.suptitle(selection_label)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"flow_distributions.{extension}")
    plt.close(figure)
