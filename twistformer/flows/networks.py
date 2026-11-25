# Time-dependent vector fields and latent codecs
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch
from torch import nn
from .dmlp import DMLP
from .lzmlp import LZMLP
from .fastkan import FastKAN


def make_network(name, input_dim, output_dim, width, depth, *, config=None):
    """
    Construct a vector network without batch-dependent density terms

    Args:
        name:  MLP, Lipschitz MLP or FastKAN
        input_dim:  Input coordinate count
        output_dim:  Output coordinate count
        width:  Hidden width
        depth:  Hidden layer count
        config:  Optional activation and normalization settings
    Returns:
        network: Torch module
    """
    factory = {"mlp": DMLP, "lzmlp": LZMLP, "fastkan": FastKAN}.get(name)
    if factory is None:
        raise ValueError(f"Unknown vector network {name}")
    options = {} if name == "fastkan" else dict(
        activation=config.activation if config is not None else "silu",
        layer_norm=config.layer_norm if config is not None else False,
        skip_connections=config.skip_connections if config is not None else True,
        dropout=0.0)
    return factory(input_dim, output_dim, [width] * depth, **options)


class DCTFeatureMapping(nn.Module):
    def __init__(self, dimension):
        """
        Construct an orthonormal discrete cosine feature transform

        Args:
            dimension:  Feature count
        """
        super().__init__()
        index = torch.arange(dimension, dtype=torch.float64)
        matrix = torch.cos(math.pi / dimension * (index[:, None] + 0.5) * index[None]) * math.sqrt(2 / dimension)
        matrix[:, 0] /= math.sqrt(2)
        self.register_buffer("matrix", matrix.float())

    def forward(self, value):
        """
        Rotate features to cosine coordinates

        Args:
            value:  Batched vectors
        Returns:
            output: Cosine coefficients
        """
        return value @ self.matrix

    def inverse(self, value):
        """
        Restore vectors from cosine coefficients

        Args:
            value:  Cosine coefficients
        Returns:
            output: Original vectors
        """
        return value @ self.matrix.T


class VectorField(nn.Module):
    def __init__(self, shape, condition_dim, config):
        """
        Construct a vector field with sinusoidal time coordinates

        Args:
            shape:  Event tensor shape
            condition_dim:  External condition width
            config:  FlowConfig record
        """
        super().__init__()
        self.shape = tuple(shape)
        self.condition_dim = condition_dim
        dimension = math.prod(shape)
        self.register_buffer("frequencies", math.pi * torch.arange(1, config.frequencies + 1))
        self.mapping = DCTFeatureMapping(dimension) if config.dct else nn.Identity()
        time_width = 2 * config.frequencies if config.frequencies else 1
        self.network = make_network(config.network, dimension + condition_dim + time_width,
                                    dimension, config.width, config.depth, config=config)

    def forward(self, x_t, t, theta=None):
        """
        Predict the transport velocity

        Args:
            x_t:  Current event coordinates
            t:  Batched time values
            theta:  Optional external conditions
        Returns:
            output: Velocity with the event shape
        """
        coordinates = self.mapping(x_t.flatten(1))
        time = t.reshape(len(x_t), 1)
        if len(self.frequencies):
            angle = time * self.frequencies
            time = torch.cat((angle.cos(), angle.sin()), -1)
        values = [coordinates, time]
        if self.condition_dim:
            if theta is None or theta.shape != (len(x_t), self.condition_dim):
                raise ValueError("Vector-field conditions do not match the configured width")
            values.append(theta)
        elif theta is not None:
            raise ValueError("This vector field is unconditional")
        return self.network(torch.cat(values, -1)).reshape(len(x_t), *self.shape)

    def get_lipschitz_loss(self):
        """
        Return the product of learned linear Lipschitz bounds

        Returns:
            output: Scalar regularization term
        """
        if not hasattr(self.network, "get_lipschitz_loss"):
            raise ValueError("Lipschitz regularization requires the lzmlp network")
        return self.network.get_lipschitz_loss().sum()


class Codec(nn.Module):
    def __init__(self, shape, config):
        """
        Construct an AE or VAE for latent flow matching

        Args:
            shape:  Observation tensor shape
            config:  Flow configuration
        """
        super().__init__()
        self.shape, self.variational = tuple(shape), config.codec == "vae"
        self.encoder_network = make_network("mlp", math.prod(shape), config.latent_dim * (2 if self.variational else 1),
                                            config.width, config.depth, config=config)
        self.decoder_network = make_network("mlp", config.latent_dim, math.prod(shape), config.width, config.depth, config=config)

    def encode(self, value, *, stochastic=False):
        """
        Encode observations and retain the VAE KL contribution

        Args:
            value:  Observation tensor
            stochastic:  Draw VAE coordinates during training
        Returns:
            result: Latent coordinates and KL loss per event
        """
        encoded = self.encoder_network(value.flatten(1))
        if not self.variational:
            return encoded, encoded.new_zeros(len(value))
        mean, log_variance = encoded.chunk(2, -1)
        log_variance = log_variance.clamp(-20, 20)
        latent = mean + torch.randn_like(mean) * (0.5 * log_variance).exp() if stochastic else mean
        kl = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(-1)
        return latent, kl

    def decode(self, value):
        """
        Decode latent vectors with arbitrary leading sample axes

        Args:
            value:  Latent vectors
        Returns:
            output: Reconstructed observations
        """
        return self.decoder_network(value).reshape(*value.shape[:-1], *self.shape)
