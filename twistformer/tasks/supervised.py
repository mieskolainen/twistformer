# Supervised heads and likelihoods
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch
from torch import nn
from torch.nn import functional as F
from twistformer.api.contracts import Prediction, LossResult
from twistformer.layers.common import normalization, ResidualMLP

class Head(nn.Module):
    def __init__(self, width, spec, cfg, model_cfg):
        super().__init__()
        self.regression = spec.name == "gaussian_regression"
        self.size, self.min_std = len(spec.outputs), cfg.min_std
        self.components = cfg.mixture_components if self.regression else 1
        output_size = 1 if not self.regression and self.size == 2 else self.size
        self.norm = normalization(model_cfg.head_norm, width)
        widths = model_cfg.head_widths or [width * model_cfg.head_expansion] * model_cfg.head_layers
        network = ResidualMLP(
            width,
            self.components * (2 * self.size + 1) if self.components > 1 else output_size * (2 if self.regression else 1),
            widths,
            norm=model_cfg.head_hidden_norm,
            activation=model_cfg.head_activation,
            dropout=model_cfg.head_dropout,
            residual=model_cfg.head_residual,
            zero_init_last=model_cfg.head_zero_init,
            init=model_cfg.head_init,
        )
        self.hidden = nn.Sequential(*list(network.children())[:-1])
        self.linear = network[-1]
        if self.components > 1 and model_cfg.head_zero_init:
            # Break component symmetry while retaining the zero-weight head
            with torch.no_grad():
                bias = self.linear.bias.view(self.components, 2 * self.size + 1)
                bias[:, 1 : 1 + self.size].normal_(std=0.2)

    def forward(self, x):
        """
        Map event features to logits or joint Gaussian mixture parameters

        Args:
            x:  Pooled event features (B,D)
        Returns:
            prediction: Shared Prediction record
        """
        raw = self.linear(self.hidden(self.norm(x)))
        if not self.regression:
            return Prediction(logits=raw)
        if self.components > 1:
            raw = raw.float().reshape(-1, self.components, 2 * self.size + 1)
            logits = raw[:, :, 0]
            means, scales = raw[:, :, 1:].chunk(2, -1)
            stds = F.softplus(scales) + self.min_std
            weights = logits.softmax(-1).unsqueeze(-1)
            mean = (weights * means).sum(1)
            variance = (weights * (stds.square() + (means - mean[:, None]).square())).sum(1)
            return Prediction(mean=mean, std=variance.sqrt(), mixture_logits=logits, component_mean=means, component_std=stds)
        mean, scale = raw.chunk(2, -1)
        return Prediction(mean=mean.float(), std=F.softplus(scale.float()) + self.min_std)

class SupervisedTask:
    def __init__(self, spec, config):
        self.spec, self.config = spec, config
        if spec.name != config.name:
            raise ValueError("Task configuration and output schema must agree")
        if config.class_weights and len(config.class_weights) != len(spec.outputs):
            raise ValueError("One class weight is required per class")

    def classification_loss_terms(self, prediction, batch):
        """
        Compute event loss and weights for the classification objective

        Args:
            prediction:  Binary or multiclass logits
            batch:  Class labels and event weights
        Returns:
            loss:  Cross entropy per event
            weights:  Event weights including class factors
        """
        if batch.target_mask is not None:
            raise ValueError("target_mask is supported for regression only")
        y, weights = batch.targets, batch.sample_weights.float()
        if y is None or y.shape != weights.shape or y.dtype != torch.long:
            raise ValueError("Classification labels must be int64 [B]")
        logits = prediction.logits.float()
        classes = len(self.spec.outputs)
        torch._assert_async(((y >= 0) & (y < classes)).all(), "Classification labels must index a declared class")
        scalar_binary = classes == 2 and logits.shape == (len(y), 1)
        if not scalar_binary and logits.shape != (len(y), classes):
            expected = f"({len(y)}, 1) or ({len(y)}, 2)" if classes == 2 else f"({len(y)}, {classes})"
            raise ValueError(f"Classification logits must have shape {expected}")
        if scalar_binary:
            target = y.float() * (1 - self.config.label_smoothing) + 0.5 * self.config.label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="none")
        else:
            loss = F.cross_entropy(logits, y, reduction="none", label_smoothing=self.config.label_smoothing)
        if self.config.class_weights:
            factors = weights.new_tensor(self.config.class_weights)
            weights = weights * factors[y]
        return loss, weights

    def compute_loss(self, prediction, batch):
        """
        Reduce weighted classification, Gaussian likelihood or four-vector MSE

        Args:
            prediction:  Logits, Gaussian parameters or covariant mean vectors
            batch:  Targets, observation masks and event weights, signed during evaluation
        Returns:
            loss: Differentiable numerator and event-weight normalizer
        """
        if batch.targets is None:
            raise ValueError("Targets are required for training/evaluation")
        if self.config.lossfunc == "cross_entropy":
            loss, weights = self.classification_loss_terms(prediction, batch)
            # Balance factors and supplied event weights share one reduction
            return LossResult((loss * weights).sum(), weights.sum())
        weights = batch.sample_weights.float()
        y, mean = (batch.targets.float(), prediction.mean.float())
        if y.shape != mean.shape:
            raise ValueError(f"Regression targets must have shape {tuple(mean.shape)}")
        valid = torch.isfinite(y)
        if batch.target_mask is not None:
            if batch.target_mask.shape != y.shape:
                raise ValueError("target_mask must match regression targets")
            valid &= batch.target_mask
        target = y.masked_fill(~valid, 0)
        if self.config.lossfunc == "mse":
            loss = (target - mean).square().masked_fill(~valid, 0)
            count = valid.sum(-1)
            weights = weights * (count > 0)
            per_event = loss.sum(-1) / count.clamp_min(1)
            return LossResult((per_event * weights).sum(), weights.sum())
        if self.config.lossfunc != "gaussian_nll":
            raise ValueError(f"Unknown task.lossfunc: {self.config.lossfunc}")
        std = prediction.std.float()
        if prediction.mixture_logits is not None:
            # Marginalize missing coordinates before summing joint hypotheses
            component_loss = (
                0.5 * ((target[:, None] - prediction.component_mean) / prediction.component_std).square()
                + prediction.component_std.log()
                + 0.5 * math.log(2 * math.pi)
            )
            component_loss = component_loss.masked_fill(~valid[:, None], 0).sum(-1)
            count = valid.sum(-1)
            per_event = -torch.logsumexp(prediction.mixture_logits.log_softmax(-1) - component_loss, dim=-1) / count.clamp_min(1)
            weights = weights * (count > 0)
            return LossResult((per_event * weights).sum(), weights.sum())
        loss = 0.5 * ((target - mean) / std).square() + std.log() + 0.5 * math.log(2 * math.pi)
        count = valid.sum(-1)
        per_event = loss.masked_fill(~valid, 0).sum(-1) / count.clamp_min(1)
        weights = weights * (count > 0)
        return LossResult((per_event * weights).sum(), weights.sum())
