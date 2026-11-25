# Continuous normalizing flows and flow matching
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch
from torch import nn
from .base import BaseDensity, expand_condition, constrained_sample
from .ot import OTPlanSampler


class ContinuousFlow(nn.Module):
    def __init__(self, vector_field, shape, *, steps=32, integrator="heun", base="gaussian",
                 path="CFM", sigma=0.0, warp=1.0, trace="hutchinson", trace_samples=1,
                 finite_difference=1e-3, ot_method="exact", ot_regularization=0.05,
                 ot_mass_regularization=1.0, custom_path=None):
        """
        Construct a vector-field transport with density evaluation

        Args:
            vector_field:  Network with arguments x_t, t and theta
            shape:  Event tensor shape
            steps:  Number of integration intervals
            integrator:  Euler or Heun scheme
            base:  Base density name
            path:  Training probability path
            sigma:  Path noise scale
            warp:  Cubic time sampling slope at the midpoint
            trace:  Divergence estimator
            trace_samples:  Number of stochastic probe vectors
            finite_difference:  Central difference displacement
            ot_method:  Transport solver
            ot_regularization:  Entropy regularization
            ot_mass_regularization:  Marginal regularization
            custom_path:  Optional function returning target velocity and coordinates
        """
        super().__init__()
        if steps < 1 or integrator not in {"euler", "heun"}:
            raise ValueError("Invalid integration scheme or interval count")
        if path not in {"FM", "CFM", "COS-CFM", "OT", "S-BRIDGE"}:
            raise ValueError("Unknown probability path")
        if sigma < 0 or not 0 <= warp <= 1.5 or trace_samples < 1 or finite_difference <= 0:
            raise ValueError("Invalid path or trace parameters")
        if path == "S-BRIDGE" and sigma <= 0:
            raise ValueError("S-BRIDGE requires positive path noise")
        if trace not in {"exact", "vmap", "hutchinson", "finite_difference", "finite_hutchinson"}:
            raise ValueError("Unknown divergence estimator")
        self.vector_field = vector_field
        self.base = BaseDensity(shape, base)
        self.steps, self.integrator = steps, integrator
        self.path, self.sigma, self.warp = path, sigma, warp
        self.trace, self.trace_samples, self.finite_difference = trace, trace_samples, finite_difference
        self.custom_path = custom_path
        self.ot = OTPlanSampler("sinkhorn" if path == "S-BRIDGE" else ot_method,
                                reg=2 * sigma**2 if path == "S-BRIDGE" else ot_regularization,
                                reg_m=ot_mass_regularization)

    def path_sample(self, source, target, time, *, noise=None):
        """
        Evaluate a probability path and its conditional velocity

        Args:
            source:  Source events
            target:  Target events
            time:  One time value per event
            noise:  Optional standard normal path noise
        Returns:
            result: Velocity and interpolated coordinates
        """
        if source.shape != target.shape:
            raise ValueError("Flow source and target shapes must agree")
        t = time.reshape(-1, *([1] * (target.ndim - 1)))
        if self.custom_path is not None:
            return self.custom_path(source, target, t)
        noise = torch.randn_like(target) if noise is None else noise
        if self.path == "FM":
            scale = 1 - (1 - self.sigma) * t
            value = t * target + scale * noise
            velocity = target - (1 - self.sigma) * noise
        elif self.path == "COS-CFM":
            angle = math.pi * t / 2
            value = angle.cos() * source + angle.sin() * target + self.sigma * noise
            velocity = math.pi / 2 * (angle.cos() * target - angle.sin() * source)
        else:
            mean = (1 - t) * source + t * target
            scale = self.sigma
            velocity = target - source
            if self.path == "S-BRIDGE":
                variance = (t * (1 - t)).clamp_min(torch.finfo(target.dtype).eps)
                scale = self.sigma * variance.sqrt()
                velocity = velocity + self.sigma * (1 - 2 * t) / (2 * variance.sqrt()) * noise
            value = mean + scale * noise
        return velocity, value

    def matching_loss(self, target, source=None, condition=None, weights=None, *, loss=None, auxiliary=None):
        """
        Match conditional velocities with aligned event weights

        Args:
            target:  Observed target coordinates
            source:  Optional paired source coordinates
            condition:  Optional target conditions
            weights:  Optional event weights
            loss:  Optional component loss function
            auxiliary:  Optional per-target auxiliary loss
        Returns:
            result: Per-event losses and aligned weights
        """
        if source is not None and self.path == "FM":
            raise ValueError("FM cannot transport an arbitrary supplied source")
        source = self.base.sample(len(target)).to(target) if source is None else source
        condition = expand_condition(condition, len(target))
        if self.path in {"OT", "S-BRIDGE"}:
            if condition is not None:
                _, groups = torch.unique(condition.detach().flatten(1), dim=0, return_inverse=True)
                i = torch.arange(len(target), device=target.device)
                j = i.clone()
                for group in groups.unique():
                    indices = (groups == group).nonzero().flatten()
                    left, right = self.ot.indices(source[indices], target[indices])
                    i[indices], j[indices] = indices[left], indices[right]
            else:
                i, j = self.ot.indices(source, target)
            source, target = source[i], target[j]
            condition = None if condition is None else condition[j]
            weights = None if weights is None else weights[j]
            auxiliary = None if auxiliary is None else auxiliary[j]
        t = torch.rand(len(target), 1, device=target.device, dtype=target.dtype)
        t = 4 * (1 - self.warp) * t**3 + 6 * (self.warp - 1) * t**2 + (3 - 2 * self.warp) * t
        velocity, value = self.path_sample(source, target, t)
        prediction = self.vector_field(x_t=value, t=t, theta=condition)
        result = (prediction - velocity).square() if loss is None else loss(prediction, velocity)
        result = result.reshape(len(target), -1).mean(-1)
        return result if auxiliary is None else result + auxiliary, weights

    def divergence(self, value, time, condition=None, *, create_graph=False, probes=None):
        """
        Evaluate the field and trace of its coordinate Jacobian

        Args:
            value:  Differentiable event coordinates
            time:  Batched integration times
            condition:  Optional conditions
            create_graph:  Retain derivative graphs for likelihood training
            probes:  Optional fixed Rademacher vectors for the full solve
        Returns:
            result: Vector field and divergence per event
        """
        field = self.vector_field(x_t=value, t=time, theta=condition)
        width = value[0].numel()
        if self.trace == "vmap":
            from torch.func import jacrev, vmap

            def single(x, t, c=None):
                return self.vector_field(x_t=x.unsqueeze(0), t=t.unsqueeze(0),
                                         theta=None if c is None else c.unsqueeze(0)).flatten()

            if condition is None:
                jacobian = vmap(jacrev(single))(value, time)
            else:
                jacobian = vmap(jacrev(single))(value, time, condition)
            result = jacobian.reshape(len(value), width, width).diagonal(dim1=-2, dim2=-1).sum(-1)
        elif self.trace == "finite_difference":
            result = value.new_zeros(len(value))
            for index in range(width):
                direction = torch.zeros_like(value)
                direction.reshape(len(value), -1)[:, index] = self.finite_difference
                positive = self.vector_field(x_t=value + direction, t=time, theta=condition)
                negative = self.vector_field(x_t=value - direction, t=time, theta=condition)
                result = result + (positive - negative).flatten(1)[:, index] / (2 * self.finite_difference)
        elif self.trace in {"hutchinson", "finite_hutchinson"}:
            if probes is None:
                probes = self.probes(value)
            result = value.new_zeros(len(value))
            for probe in probes:
                if self.trace == "finite_hutchinson":
                    h = self.finite_difference
                    positive = self.vector_field(x_t=value + h * probe, t=time, theta=condition)
                    negative = self.vector_field(x_t=value - h * probe, t=time, theta=condition)
                    product = (positive - negative) / (2 * h)
                elif field.requires_grad:
                    product = torch.autograd.grad(field, value, probe, create_graph=create_graph,
                                                  retain_graph=True, allow_unused=True)[0]
                    product = torch.zeros_like(value) if product is None else product
                else:
                    product = torch.zeros_like(value)
                result = result + (product * probe).flatten(1).sum(-1)
            result = result / len(probes)
        else:
            result = value.new_zeros(len(value))
            for index in range(width):
                component = field.flatten(1)[:, index]
                if component.requires_grad:
                    derivative = torch.autograd.grad(component.sum(), value, create_graph=create_graph,
                                                     retain_graph=True, allow_unused=True)[0]
                    if derivative is not None:
                        result = result + derivative.flatten(1)[:, index]
        return field, result if create_graph else result.detach()

    def probes(self, value):
        """
        Draw coordinate-independent Rademacher trace probes

        Args:
            value:  Reference event tensor
        Returns:
            output: Probe tensor with a leading replica axis
        """
        return (2 * torch.randint(2, (self.trace_samples, *value.shape), device=value.device) - 1).to(value)

    def integrate(self, value, condition=None, *, reverse=False, density=False, create_graph=False, trajectory=False):
        """
        Integrate coordinates and the divergence with common stages

        Args:
            value:  Initial events
            condition:  Optional conditions
            reverse:  Integrate from target time one to base time zero
            density:  Integrate the divergence
            create_graph:  Retain likelihood gradients
            trajectory:  Return every integration endpoint
        Returns:
            result: Final coordinates or trajectory, and integrated divergence
        """
        if tuple(value.shape[1:]) != self.base.shape:
            raise ValueError("Flow coordinates do not match the configured shape")
        condition = expand_condition(condition, len(value))
        if density and not value.requires_grad:
            value = value.clone().requires_grad_(True)
        probes = self.probes(value) if density and "hutchinson" in self.trace else None
        integral = value.new_zeros(len(value))
        output = [value]
        h = (-1.0 if reverse else 1.0) / self.steps
        for index in range(self.steps):
            time = value.new_full((len(value), 1), (1.0 if reverse else 0.0) + index * h)
            if density:
                velocity, trace = self.divergence(value, time, condition, create_graph=create_graph, probes=probes)
            else:
                velocity = self.vector_field(x_t=value, t=time, theta=condition)
                trace = 0.0
            endpoint = value + h * velocity
            if self.integrator == "heun":
                if density:
                    velocity_next, trace_next = self.divergence(endpoint, time + h, condition,
                                                               create_graph=create_graph, probes=probes)
                else:
                    velocity_next = self.vector_field(x_t=endpoint, t=time + h, theta=condition)
                    trace_next = 0.0
                value = value + h * (velocity + velocity_next) / 2
                integral = integral + h * (trace + trace_next) / 2
            else:
                value = endpoint
                integral = integral + h * trace
            if density and not create_graph:
                value = value.detach().requires_grad_(True)
                integral = integral.detach()
            if trajectory:
                output.append(value)
        return (torch.stack(output) if trajectory else value), integral

    def log_prob(self, target, condition=None, *, create_graph=False):
        """
        Evaluate the target density by reverse integration

        Args:
            target:  Target coordinates
            condition:  Optional conditions
            create_graph:  Retain gradients for likelihood optimization
        Returns:
            output: Log density per event
        """
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(target.device.type, enabled=False):
            target = target.to(self.base.reference).clone()
            condition = None if condition is None else condition.to(target).clone()
            source, integral = self.integrate(target, condition, reverse=True, density=True, create_graph=create_graph)
            output = self.base.log_prob(source) + integral
            return output if create_graph else output.detach()

    def transport(self, source, condition=None, *, reverse=False, trajectory=False):
        """
        Transport supplied events between source and target spaces

        Args:
            source:  Initial coordinates
            condition:  Optional conditions
            reverse:  Transport from target to source
            trajectory:  Return all integration endpoints
        Returns:
            output: Transported events
        """
        return self.integrate(source, condition, reverse=reverse, trajectory=trajectory)[0]

    def sample(self, count, condition=None, *, temperature=1.0, generator=None, constraints=None,
               max_attempts=10, trajectory=False):
        """
        Sample the learned density or its temperature-scaled variant

        Args:
            count:  Event count
            condition:  Optional conditions
            temperature:  Base coordinate scale
            generator:  Optional random generator
            constraints:  Optional lower and upper bounds
            max_attempts:  Maximum constraint redraw count
            trajectory:  Return full trajectories without rejection
        Returns:
            output: Generated events or trajectories
        """
        if trajectory and constraints is not None:
            raise ValueError("Trajectory sampling cannot use rejection constraints")
        def draw(size, theta):
            source = self.base.sample(size, temperature=temperature, generator=generator)
            return self.transport(source, theta, trajectory=trajectory)
        if trajectory:
            return draw(count, expand_condition(condition, count))
        return constrained_sample(draw, count, condition, constraints, max_attempts)
