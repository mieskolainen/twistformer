# Tau momentum, displacement and multiple-solution diagnostics
#
# mikael.mieskolainen@cern.ch (2026)

import json
from pathlib import Path
import numpy as np
from twistboard.plotting import plot_style, save_figure
from twistformer.evaluation.metrics import event_weights

OBSERVABLES = ("pt", "eta", "phi", "dx", "dy", "dz", "flight_length")
UNITS = ("GeV", "1", "rad", "mm", "mm", "mm", "mm")

def observables(cartesian):
    """
    Convert Cartesian tau vectors without assigning angles at zero pt

    Args:
        cartesian:  Array with final axis px,py,pz,dx,dy,dz
    Returns:
        values: pt,eta,phi,dx,dy,dz,flight_length with NaN for undefined angles
    """
    x = np.asarray(cartesian, dtype=np.float64)
    if x.shape[-1] != 6:
        raise ValueError("Tau coordinates require six Cartesian components")
    pt = np.hypot(x[..., 0], x[..., 1])
    eta = np.arcsinh(np.divide(x[..., 2], pt, out=np.full_like(pt, np.nan), where=pt > 0))
    phi = np.where(pt > 0, np.arctan2(x[..., 1], x[..., 0]), np.nan)
    return np.concatenate((np.stack((pt, eta, phi), axis=-1), x[..., 3:], np.linalg.norm(x[..., 3:], axis=-1)[..., None]), axis=-1)

def prediction_extra(data):
    """
    Preserve all joint hypotheses and identify the highest-weight component

    Args:
        data:  Cartesian means, widths and optional mixture parameters
    Returns:
        arrays: Named derived observables, validity masks and leading hypothesis
    """
    components = data.get("component_mean", data["mean"][:, None])
    probabilities = data.get("mixture_probabilities", np.ones(components.shape[:2]))
    leading = probabilities.argmax(-1)
    values = observables(components)
    result = {
        "observable_names": np.asarray(OBSERVABLES),
        "observable_units": np.asarray(UNITS),
        "component_observables": values,
        "component_observables_valid": np.isfinite(values),
        "leading_component": leading,
        "leading_mean": components[np.arange(len(components)), leading],
        "leading_observables": values[np.arange(len(values)), leading],
    }
    if "targets" in data:
        targets = np.where(data.get("target_mask", True), data["targets"], np.nan)
        result["target_observables"] = observables(targets)
        result["target_observables_valid"] = np.isfinite(result["target_observables"])
    if "samples" in data:
        result["sample_observables"] = observables(data["samples"])
        result["sample_observables_valid"] = np.isfinite(result["sample_observables"])
    return result

def _average(values, weights):
    """
    Compute signed-yield means with undefined zero-yield results

    Args:
        values:  Finite per-event values
        weights:  Signed event weights
    Returns:
        value: Weighted mean or None
    """
    total = weights.sum()
    value = float(np.dot(values, weights) / total) if total != 0 else np.nan
    return value if np.isfinite(value) else None

def evaluate_extra(data, output):
    """
    Compare leading hypotheses with truth and expose vertex ambiguities

    Args:
        data:  Exported physical predictions and truth vectors
        output:  Evaluation directory
    """
    data = {**data, **prediction_extra(data)}
    weights = event_weights(data)
    truth, prediction = data["target_observables"], data["leading_observables"]
    residuals = truth - prediction
    residuals[:, 2] = np.arctan2(np.sin(residuals[:, 2]), np.cos(residuals[:, 2]))
    metrics = {
        "prediction": "empirical sample mean" if "samples" in data else "highest-weight joint component mean, not mixture MAP",
        "residual": "truth minus prediction, phi wrapped to [-pi,pi]",
        "observables": {},
    }
    for index, name in enumerate(OBSERVABLES):
        take = np.isfinite(residuals[:, index]) & (weights != 0)
        residual, w = residuals[take, index], weights[take]
        mse = _average(residual**2, w)
        metrics["observables"][name] = {
            "unit": UNITS[index],
            "events": int(take.sum()),
            "bias": _average(residual, w),
            "mae": _average(abs(residual), w),
            "rmse": np.sqrt(mse).item() if mse is not None and mse >= 0 else None,
        }
    components = data.get("component_mean", data["mean"][:, None])
    vertex_error = np.linalg.norm(components[:, :, 3:] - truth[:, None, 3:6], axis=-1)
    valid = np.isfinite(vertex_error).all(-1) & (weights != 0)
    leading_error = vertex_error[np.arange(len(components)), data["leading_component"]]
    metrics["vertex"] = {
        "leading_mean_distance_mm": _average(leading_error[valid], weights[valid]),
        "oracle_nearest_component_distance_mm": _average(vertex_error[valid].min(-1), weights[valid]),
        "oracle_definition": "truth-selected diagnostic only, unavailable at inference",
    }
    if "samples" in data:
        distance = np.linalg.norm(data["samples"][:, :, 3:] - truth[:, None, 3:6], axis=-1)
        metrics["vertex"]["sample_mean_distance_mm"] = _average(distance.mean(1), weights)
        metrics["vertex"]["oracle_nearest_sample_distance_mm"] = _average(distance.min(1), weights)
    output = Path(output)
    (output / "tau3_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))

@plot_style
def plot_extra(data, output):
    """
    Plot leading tau residuals and probability integral transforms

    Args:
        data:  Exported physical mixture predictions and targets
        output:  Plot output directory
    """
    import matplotlib.pyplot as plt
    from twistformer.evaluation.mixture import mixture_diagnostics

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = prediction_extra(data)
    residual = arrays["target_observables"] - arrays["leading_observables"]
    residual[:, 2] = np.arctan2(np.sin(residual[:, 2]), np.cos(residual[:, 2]))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for index, (name, unit) in enumerate(zip(OBSERVABLES, UNITS)):
        valid = np.isfinite(residual[:, index])
        axes.flat[index].hist(residual[valid, index], weights=data["weights"][valid], bins=40, histtype="step")
        axes.flat[index].set_xlabel(f"True - leading {name} [{unit}]")
        axes.flat[index].set_ylabel("Weighted events")
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        save_figure(fig, output / f"tau3_leading_residuals.{extension}")
    plt.close(fig)
    if "mixture_logits" not in data:
        return
    _, pit, valid, _ = mixture_diagnostics(data)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for index, name in enumerate(("px", "py", "pz", "dx", "dy", "dz")):
        take = valid[:, index]
        axes.flat[index].hist(pit[take, index], weights=data["weights"][take], bins=np.linspace(0, 1, 21), histtype="step")
        axes.flat[index].set_xlabel(f"{name} mixture CDF at truth")
        axes.flat[index].set_ylabel("Weighted events")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        save_figure(fig, output / f"tau3_mixture_pit.{extension}")
    plt.close(fig)
