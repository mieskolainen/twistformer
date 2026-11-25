# Invertible physical coordinate transforms
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch


def transform_coordinates(value, config, *, inverse=False):
    """
    Transform coordinates with an explicit change of density measure

    Args:
        value:  Batched coordinates
        config:  Serializable transform settings
        inverse:  Restore physical coordinates
    Returns:
        result: Transformed tensor and per-event log determinant
    """
    name = config.get("name")
    allowed = {"affine": {"name", "scale", "shift"}, "logit": {"name", "alpha"},
               "cartesian": {"name"}, "collider": {"name", "min_pt"}}
    if name not in allowed or set(config) - allowed[name]:
        raise ValueError("Unknown flow coordinate transform or options")
    dimension = value[0].numel()
    if name == "affine":
        scale, shift = float(config.get("scale", 2)), float(config.get("shift", -1))
        if not math.isfinite(scale) or scale == 0 or not math.isfinite(shift):
            raise ValueError("Affine transforms need a nonzero finite scale and finite shift")
        output = (value - shift) / scale if inverse else scale * value + shift
        determinant = value.new_full((len(value),), (-1 if inverse else 1) * dimension * math.log(abs(scale)))
    elif name == "logit":
        alpha = float(config.get("alpha", 1e-5))
        if not 0 <= alpha < 1:
            raise ValueError("Logit alpha must lie in [0,1)")
        if inverse:
            sigmoid = value.sigmoid()
            output = (sigmoid - alpha / 2) / (1 - alpha)
            determinant = (-torch.nn.functional.softplus(value) - torch.nn.functional.softplus(-value)
                           - math.log(1 - alpha)).flatten(1).sum(-1)
        else:
            if (value < 0).any() or (value > 1).any():
                raise ValueError("Logit inputs must lie in [0,1]")
            scaled = alpha / 2 + (1 - alpha) * value
            output = scaled.log() - torch.log1p(-scaled)
            determinant = (math.log(1 - alpha) - scaled.log() - torch.log1p(-scaled)).flatten(1).sum(-1)
    else:
        if dimension % 3:
            raise ValueError("Momentum coordinate transforms require complete triples")
        vectors = value.reshape(len(value), -1, 3)
        a, b, c = vectors.unbind(-1)
        if name == "cartesian":
            if inverse:
                pt = torch.hypot(a, b)
                if (pt <= 0).any():
                    raise ValueError("Cylindrical coordinates are singular at zero transverse momentum")
                eta = torch.asinh(c / pt)
                output = torch.stack((pt, eta, torch.atan2(b, a)), -1)
                determinant = -2 * pt.log() - eta.cosh().log()
            else:
                if (a <= 0).any() or (c < -math.pi).any() or (c > math.pi).any():
                    raise ValueError("Cylindrical momenta require positive pt and a principal phi angle")
                output = torch.stack((a * c.cos(), a * c.sin(), a * b.sinh()), -1)
                determinant = 2 * a.log() + b.cosh().log()
        else:
            minimum = float(config.get("min_pt", 15))
            if inverse:
                if (c.abs() >= math.pi / 2).any():
                    raise ValueError("Inverse collider angle is outside the arctangent domain")
                output = torch.stack((a.exp() + minimum, b, math.pi * c.tan()), -1)
                determinant = a + math.log(math.pi) - 2 * c.cos().log()
            else:
                if (a <= minimum).any():
                    raise ValueError("Collider transform requires pt above min_pt")
                output = torch.stack(((a - minimum).log(), b, torch.atan(c / math.pi)), -1)
                determinant = -(a - minimum).log() - math.log(math.pi) - torch.log1p((c / math.pi).square())
        output = output.reshape_as(value)
        determinant = determinant.sum(-1)
    return output, determinant


def transform_chain(value, transforms, *, inverse=False):
    """
    Compose physical transforms and their log determinants

    Args:
        value:  Batched event coordinates
        transforms:  Ordered serializable transform settings
        inverse:  Reverse transform order and direction
    Returns:
        result: Coordinates and summed log determinant
    """
    determinant = value.new_zeros(len(value))
    for config in reversed(transforms) if inverse else transforms:
        value, delta = transform_coordinates(value, config, inverse=inverse)
        determinant = determinant + delta
    return value, determinant


def dequantize(value, levels=256, *, generator=None):
    """
    Add uniform quantization noise in the unit interval

    Args:
        value:  Integer-valued measurements in [0, levels)
        levels:  Number of quantization levels
        generator:  Optional random generator
    Returns:
        result: Continuous coordinates and log volume correction
    """
    if levels < 2 or (value < 0).any() or (value >= levels).any() or (value != value.round()).any():
        raise ValueError("Dequantization requires valid integer quantization bins")
    value = value.float()
    output = (value + torch.rand(value.shape, device=value.device, dtype=value.dtype, generator=generator)) / levels
    return output, value.new_full((len(value),), -value[0].numel() * math.log(levels))
