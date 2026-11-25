# Particle batches and training normalization
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from dataclasses import replace
from twistformer.api.contracts import Batch, Sample

def drop_tokens(batch, probability, min_keep=1):
    """
    Apply training token dropout while retaining event support

    Args:
        batch:  Padded Batch with physical geometry and optional token weights
        probability:  Independent removal probability per valid token
        min_keep:  Minimum retained count, capped by the original multiplicity
    Returns:
        batch: New Batch with aligned masks, features and geometry
    """
    if probability == 0:
        return batch
    scores = torch.rand(batch.mask.shape, device=batch.mask.device).masked_fill(~batch.mask, -1)
    keep = (scores >= probability) & batch.mask
    indices = scores.topk(min(min_keep, scores.shape[1]), dim=1).indices
    minimum = torch.zeros_like(keep).scatter_(1, indices, True) & batch.mask
    keep |= minimum
    if batch.token_weights is not None:
        positive = batch.mask & (batch.token_weights > 0)
        first = scores.masked_fill(~positive, -2).argmax(1, keepdim=True)
        support = torch.zeros_like(keep).scatter_(1, first, True) & positive
        keep |= support
    values = {"mask": keep, "features": batch.features.masked_fill(~keep[..., None], 0)}
    if batch.p4 is not None:
        values["p4"] = batch.p4.masked_fill(~keep[..., None], 0)
    if batch.token_weights is not None:
        values["token_weights"] = batch.token_weights.masked_fill(~keep, 0)
    return replace(batch, **values)

def collate_samples(samples: list[Sample], *, allow_negative_weights=False) -> Batch:
    """
    Validate and pad variable-multiplicity events

    Args:
        samples:  Nonempty list of raw or preprocessed Sample records
        allow_negative_weights:  Permit signed event yields for inference and evaluation
    Returns:
        batch: Batch with True for valid particles
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if any(s.features.ndim != 2 or s.features.shape[0] == 0 for s in samples):
        raise ValueError("Each sample must contain at least one token of shape (N,F)")
    width = samples[0].features.shape[1]
    if any(s.features.shape[1] != width or not torch.isfinite(s.features).all() for s in samples):
        raise ValueError("Feature dimensions must agree and features must be finite")
    weights = torch.tensor([s.weight for s in samples], dtype=torch.float32)
    if not torch.isfinite(weights).all() or (not allow_negative_weights and (weights < 0).any()):
        raise ValueError("Sample weights must be finite and nonnegative unless signed evaluation is enabled")
    b, n = len(samples), max(len(s.features) for s in samples)
    x = torch.zeros(b, n, width)
    mask = torch.zeros(b, n, dtype=torch.bool)
    optional = {}
    names = set(samples[0].observables)
    if any(set(sample.observables) != names for sample in samples):
        raise ValueError("Evaluation observable names must agree within a batch")
    optional["observables"] = {
        name: torch.tensor([sample.observables[name] for sample in samples], dtype=torch.float64) for name in sorted(names)
    }
    if any(value.shape != (b,) for value in optional["observables"].values()):
        raise ValueError("Evaluation observables must be event scalars")
    for name in ("global_features", "parameters"):
        values = [getattr(sample, name) for sample in samples]
        present = [value is not None for value in values]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"Inconsistent presence of {name}")
        shape = values[0].shape
        if (
            len(shape) != 1
            or shape[0] == 0
            or any(value.shape != shape or not torch.isfinite(value).all() for value in values)
        ):
            raise ValueError(f"{name} must be finite vectors of equal width")
        optional[name] = torch.stack([value.float() for value in values])
    for name in ("source", "flow_condition", "flow_log_det"):
        values = [getattr(sample, name) for sample in samples]
        if any(value is not None for value in values):
            if any(value is None for value in values):
                raise ValueError(f"Inconsistent presence of {name}")
            if any(value.shape != values[0].shape or not torch.isfinite(value).all() for value in values):
                raise ValueError(f"{name} requires matching finite event tensors")
            optional[name] = torch.stack([value.float() for value in values])
    for name in ("p4", "token_weights", "target", "target_mask"):
        present = [getattr(s, name) is not None for s in samples]
        if any(present) and not all(present):
            raise ValueError(f"Inconsistent presence of {name}")
        if all(present):
            if name in {"target", "target_mask"}:
                if name == "target_mask" and any(s.target_mask.dtype != torch.bool for s in samples):
                    raise ValueError("target_mask must be Boolean")
                optional[{"target": "targets", "target_mask": "target_mask"}[name]] = torch.stack(
                    [torch.as_tensor(getattr(s, name)) for s in samples]
                )
            else:
                dtype = torch.float32
                if name == "p4":
                    for sample in samples:
                        if not sample.p4.is_floating_point():
                            raise ValueError("p4 must contain real floating-point momenta")
                        dtype = torch.promote_types(dtype, sample.p4.dtype)
                optional[name] = torch.zeros((b, n, 4) if name == "p4" else (b, n), dtype=dtype)
    for i, s in enumerate(samples):
        k = len(s.features)
        x[i, :k] = s.features
        mask[i, :k] = True
        for name in ("p4", "token_weights"):
            if name in optional:
                v = getattr(s, name)
                expected = (k, 4) if name == "p4" else (k,)
                if tuple(v.shape) != expected or not torch.isfinite(v).all():
                    raise ValueError(f"Invalid {name}, expected {expected}")
                if name == "token_weights" and (v < 0).any():
                    raise ValueError("Token weights must be nonnegative")
                optional[name][i, :k] = v
    return Batch(x, mask, weights, tuple(s.id for s in samples), **optional)
