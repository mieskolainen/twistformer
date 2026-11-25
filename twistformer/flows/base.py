# Base densities and constrained sampling
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch
from torch import nn


class BaseDensity(nn.Module):
    def __init__(self, shape, kind="gaussian"):
        """
        Define independent unit-variance base coordinates

        Args:
            shape:  Event tensor shape
            kind:  Gaussian, uniform or triangular density
        """
        super().__init__()
        self.shape = tuple(shape)
        self.kind = kind
        if not self.shape or any(d < 1 for d in self.shape) or kind not in {"gaussian", "uniform", "triangle"}:
            raise ValueError("Invalid base density or event shape")
        self.register_buffer("reference", torch.empty(0))

    def sample(self, count, *, generator=None, temperature=1.0):
        """
        Draw base coordinates with a common temperature

        Args:
            count:  Event count
            generator:  Optional Torch random generator
            temperature:  Positive coordinate scale
        Returns:
            output: Base events
        """
        if count < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling needs positive count and temperature")
        options = dict(device=self.reference.device, dtype=self.reference.dtype, generator=generator)
        if self.kind == "gaussian":
            output = torch.randn((count, *self.shape), **options)
        else:
            uniform = torch.rand((count, *self.shape), **options)
            if self.kind == "uniform":
                output = math.sqrt(3) * (2 * uniform - 1)
            else:
                output = math.sqrt(6) * torch.where(uniform < 0.5, (2 * uniform).sqrt() - 1,
                                                    1 - (2 * (1 - uniform)).sqrt())
        return temperature * output

    def log_prob(self, value):
        """
        Evaluate the normalized base density

        Args:
            value:  Base events
        Returns:
            output: Log density per event
        """
        if tuple(value.shape[1:]) != self.shape:
            raise ValueError("Base coordinates do not match the event shape")
        if self.kind == "gaussian":
            output = -0.5 * (value.square() + math.log(2 * math.pi))
        elif self.kind == "uniform":
            output = torch.full_like(value, -math.log(2 * math.sqrt(3)))
            output = output.masked_fill(value.abs() > math.sqrt(3), -torch.inf)
        else:
            probability = (math.sqrt(6) - value.abs()) / 6
            output = probability.clamp_min(torch.finfo(value.dtype).tiny).log().masked_fill(probability <= 0, -torch.inf)
        return output.flatten(1).sum(-1)


def expand_condition(condition, count):
    """
    Broadcast a shared condition across events

    Args:
        condition:  Optional batched condition tensor
        count:  Event count
    Returns:
        output: Aligned conditions
    """
    if condition is None:
        return None
    if condition.ndim == 1:
        condition = condition.unsqueeze(0)
    if condition.shape[0] != 1 and condition.shape[0] != count:
        raise ValueError("Condition count must be one or match the event count")
    return condition.expand(count, *condition.shape[1:])


def constrained_sample(draw, count, condition=None, constraints=None, max_attempts=10):
    """
    Resample rejected events without changing their conditions

    Args:
        draw:  Function of event count and conditions
        count:  Event count
        condition:  Optional conditions
        constraints:  Optional lower and upper tensor bounds
        max_attempts:  Maximum redraw count
    Returns:
        output: Events inside the requested bounds
    """
    condition = expand_condition(condition, count)
    output = draw(count, condition)
    if constraints is None:
        return output
    lower, upper = (torch.as_tensor(bound, device=output.device, dtype=output.dtype) for bound in constraints)
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower >= upper).any():
        raise ValueError("Sampling constraints must be finite ordered bounds")
    for attempt in range(max_attempts + 1):
        invalid = ((output < lower) | (output > upper) | ~torch.isfinite(output)).flatten(1).any(-1)
        if not invalid.any():
            return output
        if attempt < max_attempts:
            output[invalid] = draw(int(invalid.sum()), None if condition is None else condition[invalid])
    raise RuntimeError(f"Sampling constraints failed for {int(invalid.sum())} events after {max_attempts} redraws")
