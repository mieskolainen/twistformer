# Lorentz-invariant PELICAN pair encoder
# PELICAN: https://arxiv.org/abs/2211.00454
# PELICAN architecture: https://arxiv.org/abs/2307.16506
#
# mikael.mieskolainen@cern.ch (2026)

"""PELICAN scalar outputs from the complete rank-two equivariant basis"""

import torch
from torch import nn
from twistformer.api.contracts import EncoderOutput, Prediction
from twistformer.layers.common import AuxiliaryEmbedding, ResidualMLP, normalization
from twistformer.layers.pelican import (
    PairAggregation,
    PairMessage,
    PowerEncoding,
    ScalarLift,
    InvariantReadout,
    ParticleReadout,
    TriangleInteraction,
    pelican_activation,
)

class Pelican(nn.Module):
    def __init__(self, input_spec, cfg, *, covariant=False):
        super().__init__()
        self.momentum_scale = cfg.pelican_momentum_scale
        self.embed = nn.Sequential(nn.Linear(1, cfg.width, bias=False), PowerEncoding(cfg.width))
        nn.init.ones_(self.embed[0].weight)
        self.auxiliary = AuxiliaryEmbedding(input_spec, cfg.auxiliary_features, cfg.width) if cfg.auxiliary_features else None
        self.scalar_lift = ScalarLift(cfg.width) if self.auxiliary is not None else None
        hidden = cfg.pelican_message_width or cfg.width * cfg.expansion
        self.messages = nn.ModuleList([PairMessage(cfg.width, hidden, cfg) for _ in range(cfg.depth)])
        self.blocks = nn.ModuleList(
            [
                PairAggregation(hidden, cfg.width, cfg.pelican_average_nobj, cfg.pelican_factorize, cfg.pelican_multiplicity)
                for _ in range(cfg.depth)
            ]
        )
        self.activation = pelican_activation(cfg.pelican_activation)
        self.final_message = PairMessage(cfg.width, hidden, cfg)
        self.readout = (ParticleReadout if covariant else InvariantReadout)(
            hidden, cfg.width, cfg.pelican_average_nobj, cfg.pelican_multiplicity
        )
        self.triangles = nn.ModuleList(
            [
                TriangleInteraction(
                    cfg.width, cfg.pelican_triangle_channels, cfg.norm, cfg.pelican_activation, cfg.pelican_triangle_scale
                )
                for _ in range(cfg.depth)
            ]
            if cfg.pelican_triangle
            else []
        )

    def prepare_onnx(self, module, inputs):
        """
        Prepare the inference graph for ONNX conversion

        Args:
            module:  Shared inference module with preprocessing and task outputs
            inputs:  Example tensor inputs
        Returns:
            module: Exportable inference module
        """
        for block in self.blocks:
            block.export_diagonal = True
        if self.scalar_lift is not None:
            self.scalar_lift.export_diagonal = True
        return module

    def forward(self, batch, global_context=None):
        """
        Encode Minkowski Gram matrices into invariant event or particle channels

        Args:
            batch:  Padded particle Batch with physical four-momenta
            global_context:  Optional invariant event channels (B,D)
        Returns:
            encoded: Invariant features (B,D) or equivariant particle scalars (B,N,D)
        """
        mask = batch.mask
        valid = mask[:, :, None] & mask[:, None, :]
        # Only the initial contraction needs protection from collinear cancellation
        with torch.autocast(device_type=batch.p4.device.type, enabled=False):
            p = batch.p4.double().masked_fill(~mask[..., None], 0) * self.momentum_scale
            gram = (p[..., :1] @ p[..., :1].transpose(1, 2) - p[..., 1:] @ p[..., 1:].transpose(1, 2)).float()
            x = self.embed(gram[..., None])
        if self.auxiliary is not None:
            x = x + self.scalar_lift(self.auxiliary(batch.features, mask), mask)
        if global_context is not None:
            x = x + global_context[:, None, None]
        x = x.masked_fill(~valid[..., None], 0)
        for index, (message, block) in enumerate(zip(self.messages, self.blocks)):
            x = self.activation(block(message(x, valid), mask))
            if self.triangles:
                x = self.triangles[index](x, mask)
        x = self.final_message(x, valid)
        return EncoderOutput(self.activation(self.readout(x, mask)))

class CovariantHead(nn.Module):
    def __init__(self, width, vectors, cfg):
        super().__init__()
        self.norm = normalization(cfg.head_norm, width)
        self.network = ResidualMLP(
            width,
            vectors,
            cfg.head_widths or [width * cfg.head_expansion] * cfg.head_layers,
            norm=cfg.head_hidden_norm,
            activation=cfg.head_activation,
            dropout=cfg.head_dropout,
            residual=cfg.head_residual,
            zero_init_last=cfg.head_zero_init,
            init=cfg.head_init,
        )

    def forward(self, features, p4, mask):
        """
        Sum physical four-momenta with learned Lorentz-scalar coefficients

        Args:
            features:  Invariant particle channels (B,N,D)
            p4:  Physical input momenta (B,N,4), ordered E,px,py,pz
            mask:  Valid particles (B,N)
        Returns:
            prediction: Flattened covariant four-vectors in mean (B,4*K)
        """
        weights = self.network(self.norm(features), mask if self.network.batch_norm else None)
        weights = weights.masked_fill(~mask[..., None], 0)
        # Use physical momenta, independent of the scale used for Gram inputs
        with torch.autocast(device_type=p4.device.type, enabled=False):
            momenta = p4 if p4.dtype == torch.float64 else p4.float()
            momenta = momenta.masked_fill(~mask[..., None], 0)
            vectors = (weights.to(momenta.dtype)[..., None] * momenta[:, :, None]).sum(1)
        return Prediction(mean=vectors.flatten(1))
