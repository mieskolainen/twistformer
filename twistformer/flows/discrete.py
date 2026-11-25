# Invertible coupling flows
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch
from torch import nn
from .base import BaseDensity, expand_condition, constrained_sample
from .networks import make_network
from .splines import unconstrained_RQS


class AffineNorm(nn.Module):
    def __init__(self, dimension, batchnorm=False, momentum=0.1):
        """
        Construct invertible channel normalization

        Args:
            dimension:  Coordinate count
            batchnorm:  Update detached running moments during training
            momentum:  Running moment update fraction
        """
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dimension))
        self.log_scale = nn.Parameter(torch.zeros(dimension))
        self.batchnorm, self.momentum = batchnorm, momentum
        self.register_buffer("running_mean", torch.zeros(dimension))
        self.register_buffer("running_variance", torch.ones(dimension))

    def forward(self, value, condition=None, *, inverse=False):
        """
        Apply an affine map and its exact log determinant

        Args:
            value:  Coordinate vectors
            condition:  Unused external conditions
            inverse:  Apply the inverse map with the same running moments
        Returns:
            result: Transformed coordinates and log determinant
        """
        if self.batchnorm and self.training and not inverse:
            with torch.no_grad():
                self.running_mean.lerp_(value.mean(0), self.momentum)
                self.running_variance.lerp_(value.var(0, unbiased=False), self.momentum)
        mean = self.running_mean.clone()
        std = (self.running_variance.clone() + 1e-5).sqrt() if self.batchnorm else torch.ones_like(mean)
        log_scale = self.log_scale - std.log()
        if inverse:
            output = (value - self.shift) * (-log_scale).exp() + mean
        else:
            output = (value - mean) * log_scale.exp() + self.shift
        return output, ((-1 if inverse else 1) * log_scale.sum()).expand(len(value))


class Permutation(nn.Module):
    def __init__(self, dimension, mode="reverse"):
        """
        Construct a fixed coordinate permutation

        Args:
            dimension:  Coordinate count
            mode:  Reverse or random ordering
        """
        super().__init__()
        permutation = torch.randperm(dimension) if mode == "random" else torch.arange(dimension - 1, -1, -1)
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", permutation.argsort())

    def forward(self, value, condition=None, *, inverse=False):
        """
        Permute coordinates with unit absolute Jacobian

        Args:
            value:  Coordinate vectors
            condition:  Unused conditions
            inverse:  Restore the original order
        Returns:
            result: Permuted coordinates and zero log determinant
        """
        return value[:, self.inverse_permutation if inverse else self.permutation], value.new_zeros(len(value))


class InvertibleLinear(nn.Module):
    def __init__(self, dimension):
        """
        Construct a nonsingular LU map with fixed diagonal signs

        Args:
            dimension:  Coordinate count
        """
        super().__init__()
        orthogonal = torch.linalg.qr(torch.randn(dimension, dimension)).Q
        permutation, lower, upper = torch.linalg.lu(orthogonal)
        self.register_buffer("permutation", permutation)
        self.register_buffer("identity", torch.eye(dimension))
        self.register_buffer("sign", upper.diag().sign())
        self.lower = nn.Parameter(lower.tril(-1))
        self.upper = nn.Parameter(upper.triu(1))
        self.log_diagonal = nn.Parameter(upper.diag().abs().log())

    def forward(self, value, condition=None, *, inverse=False):
        """
        Mix coordinates without caching parameter-dependent inverses

        Args:
            value:  Coordinate vectors
            condition:  Unused conditions
            inverse:  Solve the inverse map
        Returns:
            result: Mixed coordinates and exact log determinant
        """
        matrix = self.permutation @ (self.lower.tril(-1) + self.identity) @ (
            self.upper.triu(1) + torch.diag(self.sign * self.log_diagonal.exp()))
        output = torch.linalg.solve(matrix.T, value.T).T if inverse else value @ matrix
        return output, ((-1 if inverse else 1) * self.log_diagonal.sum()).expand(len(value))


class Coupling(nn.Module):
    def __init__(self, dimension, condition_dim, config):
        """
        Construct two affine or rational-quadratic coupling maps

        Args:
            dimension:  Coordinate count
            condition_dim:  Condition width
            config:  Flow configuration
        """
        super().__init__()
        self.split = dimension // 2
        self.dimension, self.condition_dim, self.config = dimension, condition_dim, config
        parameters = 3 * config.bins - 1 if config.coupling == "rqs" else 2
        network = "fastkan" if config.coupling == "kan" else config.network
        self.first = make_network(network, self.split + condition_dim,
                                  (dimension - self.split) * parameters, config.width, config.depth, config=config)
        self.second = make_network(network, dimension - self.split + condition_dim,
                                   self.split * parameters, config.width, config.depth, config=config) if self.split else None

    def transform(self, value, context, network, inverse):
        """
        Transform one partition conditioned on the other

        Args:
            value:  Coordinates to transform
            context:  Fixed coordinates and external conditions
            network:  Parameter network
            inverse:  Apply the inverse map
        Returns:
            result: Transformed partition and log determinant
        """
        raw = network(context)
        if self.config.coupling == "rqs":
            raw = raw.reshape(len(value), value.shape[-1], 3 * self.config.bins - 1)
            width, height, derivative = torch.split(raw, [self.config.bins, self.config.bins, self.config.bins - 1], -1)
            output, determinant = unconstrained_RQS(value, width, height, derivative,
                                                    inverse=inverse, tail_bound=self.config.bound)
        else:
            shift, scale = raw.chunk(2, -1)
            scale = self.config.scale_limit * torch.tanh(scale / self.config.scale_limit)
            output = (value - shift) * (-scale).exp() if inverse else value * scale.exp() + shift
            determinant = -scale if inverse else scale
        return output, determinant.sum(-1)

    def forward(self, value, condition=None, *, inverse=False):
        """
        Apply the two coupling maps in their required order

        Args:
            value:  Coordinate vectors
            condition:  Optional external conditions
            inverse:  Reverse the map order and direction
        Returns:
            result: Coordinates and summed log determinant
        """
        lower, upper = value[:, :self.split], value[:, self.split:]
        determinant = value.new_zeros(len(value))
        def context(part):
            return part if condition is None else torch.cat((part, condition), -1)
        if inverse:
            if self.second is not None:
                lower, delta = self.transform(lower, context(upper), self.second, True)
                determinant = determinant + delta
            upper, delta = self.transform(upper, context(lower), self.first, True)
        else:
            upper, delta = self.transform(upper, context(lower), self.first, False)
            determinant = determinant + delta
            if self.second is not None:
                lower, delta = self.transform(lower, context(upper), self.second, False)
            else:
                delta = value.new_zeros(len(value))
        return torch.cat((lower, upper), -1), determinant + delta


class DiscreteFlow(nn.Module):
    def __init__(self, shape, condition_dim, config):
        """
        Stack invertible maps from observations to base coordinates

        Args:
            shape:  Observation tensor shape
            condition_dim:  External condition width
            config:  Flow configuration
        """
        super().__init__()
        self.base = BaseDensity(shape, config.base)
        self.condition_dim = condition_dim
        dimension = math.prod(shape)
        layers = []
        for _ in range(config.layers):
            if config.normalization != "none":
                layers.append(AffineNorm(dimension, batchnorm=config.normalization == "batchnorm"))
            if config.linear:
                layers.append(InvertibleLinear(dimension))
            if config.permutation != "none":
                layers.append(Permutation(dimension, config.permutation))
            layers.append(Coupling(dimension, condition_dim, config))
        self.layers = nn.ModuleList(layers)

    def forward(self, value, condition=None, *, inverse=False, trajectory=False):
        """
        Transform events and accumulate exact log determinants

        Args:
            value:  Observation or base coordinates
            condition:  Optional external conditions
            inverse:  Map base coordinates to observations
            trajectory:  Return every layer output
        Returns:
            result: Coordinates or trajectory, and log determinant
        """
        if tuple(value.shape[1:]) != self.base.shape:
            raise ValueError("Discrete flow event shape mismatch")
        condition = expand_condition(condition, len(value))
        if self.condition_dim:
            if condition is None or condition.shape != (len(value), self.condition_dim):
                raise ValueError("Discrete flow condition width mismatch")
        elif condition is not None:
            raise ValueError("This discrete flow is unconditional")
        shape = value.shape
        value = value.flatten(1)
        determinant = value.new_zeros(len(value))
        outputs = [value.reshape(shape)]
        for layer in reversed(self.layers) if inverse else self.layers:
            value, delta = layer(value, condition, inverse=inverse)
            determinant = determinant + delta
            if trajectory:
                outputs.append(value.reshape(shape))
        return (torch.stack(outputs) if trajectory else value.reshape(shape)), determinant

    def log_prob(self, target, condition=None, *, create_graph=False):
        """
        Evaluate the observation density

        Args:
            target:  Observation coordinates
            condition:  Optional conditions
            create_graph:  Shared continuous-flow interface flag
        Returns:
            output: Log density per event
        """
        with torch.autocast(target.device.type, enabled=False):
            target = target.to(self.base.reference)
            condition = None if condition is None else condition.to(target)
            source, determinant = self(target, condition)
            return self.base.log_prob(source) + determinant

    def transport(self, source, condition=None, *, reverse=False, trajectory=False):
        """
        Map base coordinates to observations or reverse

        Args:
            source:  Initial coordinates
            condition:  Optional conditions
            reverse:  Map observations to base coordinates
            trajectory:  Return every layer output
        Returns:
            output: Transformed coordinates
        """
        return self(source, condition, inverse=not reverse, trajectory=trajectory)[0]

    def sample(self, count, condition=None, *, temperature=1.0, generator=None, constraints=None,
               max_attempts=10, trajectory=False):
        """
        Generate events with optional rejection constraints

        Args:
            count:  Event count
            condition:  Optional conditions
            temperature:  Base coordinate scale
            generator:  Optional random generator
            constraints:  Optional lower and upper bounds
            max_attempts:  Maximum redraw count
            trajectory:  Return every layer output
        Returns:
            output: Generated events or trajectories
        """
        if trajectory and constraints is not None:
            raise ValueError("Trajectory sampling cannot use rejection constraints")
        def draw(size, theta):
            return self.transport(self.base.sample(size, temperature=temperature, generator=generator),
                                  theta, trajectory=trajectory)
        if trajectory:
            return draw(count, expand_condition(condition, count))
        return constrained_sample(draw, count, condition, constraints, max_attempts)
