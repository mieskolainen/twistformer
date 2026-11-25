# Post-training temperature fit for classification logits
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace
import math

import numpy as np
import torch
from torch import nn


class CalibratedClassifier(nn.Module):
    def __init__(self, model, calibration):
        """
        Attach the fitted temperature and its validation provenance to the classifier

        Args:
            model:  Restored classifier
            calibration:  Saved post-training fit record
        """
        super().__init__()
        temperature = float(calibration["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Saved classifier temperature must be positive and finite")
        self.classifier = model
        self.requires_p4 = model.requires_p4
        self.calibration = dict(calibration)
        self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float32))

    def forward(self, batch):
        """
        Scale classifier logits before sigmoid or softmax

        Args:
            batch:  Prepared classifier inputs
        Returns:
            prediction:  Prediction with fitted logits
        """
        prediction = self.classifier(batch)
        return replace(prediction, logits=prediction.logits.float() / self.temperature)


def fit_temperature(logits, targets, weights, steering):
    """
    Minimize weighted cross entropy in the inverse temperature by bisection

    Args:
        logits:  Raw validation margins or class logits
        targets:  Integer class labels
        weights:  Nonnegative validation event weights
        steering:  Allowed temperature interval
    Returns:
        result:  Fitted temperature and validation diagnostics
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if logits.ndim == 1:
        logits = logits[:, None]
    if logits.ndim != 2 or logits.shape[1] == 0 or len(logits) != len(targets) or targets.shape != weights.shape:
        raise ValueError("Validation logits, labels and weights must align")
    if logits.shape[1] == 1:
        logits = np.column_stack((np.zeros(len(logits)), logits[:, 0]))
    if (not np.isfinite(logits).all() or not np.isfinite(weights).all() or (weights < 0).any()
            or weights.sum() <= 0 or targets.dtype.kind not in "iu" or (targets < 0).any()
            or (targets >= logits.shape[1]).any()):
        raise ValueError("Temperature fitting requires finite logits, valid classes and positive total weight")
    logits = logits - logits.max(axis=1, keepdims=True)
    truth = logits[np.arange(len(targets)), targets]
    weights = weights / weights.sum()

    def objective(beta):
        """
        Evaluate cross entropy and its derivative in inverse temperature

        Args:
            beta:  Positive inverse temperature
        Returns:
            result:  Weighted cross entropy and slope
        """
        exp = np.exp(beta * logits)
        partition = exp.sum(axis=1)
        loss = weights @ (np.log(partition) - beta*truth)
        slope = weights @ ((exp*logits).sum(axis=1)/partition - truth)
        return float(loss), float(slope)

    lo, hi = 1 / steering.max_temperature, 1 / steering.min_temperature
    for _ in range(60):
        beta = (lo + hi)/2
        if objective(beta)[1] < 0:
            lo = beta
        else:
            hi = beta
    raw_loss = objective(1.)[0]
    beta = (lo + hi)/2
    fitted_loss = objective(beta)[0]
    if fitted_loss >= raw_loss:
        beta, fitted_loss = 1., raw_loss
    return {"method": "temperature_scaling", "temperature": 1/beta, "split": "val",
            "events": len(targets), "cross_entropy_before": raw_loss, "cross_entropy_after": fitted_loss}
