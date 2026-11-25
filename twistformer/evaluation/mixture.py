# Joint Gaussian mixture likelihood and marginal calibration
#
# mikael.mieskolainen@cern.ch (2026)

import math
import numpy as np
import torch

def mixture_diagnostics(data):
    """
    Evaluate mixture densities and probability integral transforms

    Args:
        data:  Physical component parameters, targets and event weights
    Returns:
        diagnostics: Marginal NLL, PIT, target mask and joint summaries
    """
    from .metrics import event_weights, _mean

    y = np.asarray(data["targets"], dtype=np.float64)
    means = np.asarray(data["component_mean"], dtype=np.float64)
    stds = np.asarray(data["component_std"], dtype=np.float64)
    logits = np.asarray(data["mixture_logits"], dtype=np.float64)
    if means.ndim != 3 or means.shape[0] != len(y) or means.shape[2] != y.shape[1] or not means.shape[1]:
        raise ValueError("Mixture means must have shape [events,components,outputs]")
    if stds.shape != means.shape or logits.shape != means.shape[:2]:
        raise ValueError("Mixture widths and logits must match component means")
    if not all(np.isfinite(a).all() for a in (means, stds, logits)) or (stds <= 0).any():
        raise ValueError("Mixture parameters must be finite with positive widths")
    valid = np.isfinite(y) & data.get("target_mask", np.ones_like(y, dtype=bool))
    log_weights = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    z = (np.where(valid, y, 0)[:, None] - means) / stds
    log_density = -0.5 * z**2 - np.log(stds) - 0.5 * math.log(2 * math.pi)
    marginal_nll = -np.logaddexp.reduce(log_weights[:, :, None] + log_density, axis=1)
    pit = (np.exp(log_weights[:, :, None]) * torch.special.ndtr(torch.from_numpy(z)).numpy()).sum(1)
    joint_nll = -np.logaddexp.reduce(log_weights + np.where(valid[:, None], log_density, 0).sum(-1), axis=1)
    observed = valid.sum(-1)
    weights = event_weights(data)
    take = (observed > 0) & (weights != 0)
    probabilities = np.exp(log_weights)
    summary = {
        "components": means.shape[1],
        "joint_nll": _mean(joint_nll[take], weights[take]),
        "joint_nll_per_observed_target": _mean(joint_nll[take] / observed[take], weights[take]),
        "absolute_weight_component_usage": ((abs(weights[:, None]) * probabilities).sum(0) / abs(weights).sum()).tolist(),
        "point_summary": "mean and std are mixture moments, pulls are moment standardized",
        "coverage_definition": "equal-tailed marginal mixture intervals",
    }
    return marginal_nll, pit, valid, summary
