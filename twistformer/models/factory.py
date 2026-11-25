# Particle model encoders
#
# mikael.mieskolainen@cern.ch (2026)

from torch import nn
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from twistformer.api.contracts import Prediction
from twistformer.layers.common import Pool, mlp
from twistformer.layers.gatr import GATrBlock
from twistformer.layers.pelican import PairMessage, PairAggregation, TriangleInteraction
from twistformer.tasks.supervised import Head
from .transformer import Transformer, Block
from .twistformer import Twistformer, DetectorTwistformer, TwistorBlock
from .pelican import Pelican, CovariantHead
from .gatr import GATr

class Model(nn.Module):
    def __init__(self, input_spec, task_spec, model_config, task_config):
        super().__init__()
        cfg = model_config
        self.direct_logits = cfg.family == "gatr" and cfg.gatr_readout == "logits"
        if self.direct_logits and input_spec.global_features:
            raise ValueError("Direct GATr logits do not use shared global feature fusion")
        self.covariant = task_spec.name == "covariant_regression"
        if self.covariant and cfg.family not in {"pelican", "gatr"}:
            raise ValueError("Covariant regression requires PELICAN or Lorentz GATr")
        physical = (
            cfg.family in {"twistformer", "pelican"} or cfg.pair_bias != "none" or (cfg.family == "gatr" and cfg.gatr_input == "p4")
        )
        if physical and input_spec.geometry != "p4":
            raise ValueError(f"{cfg.family}/{cfg.pair_bias} requires explicit p4 geometry (E,px,py,pz)")
        self.requires_p4 = physical
        factory = {
            "transformer": Transformer,
            "triformer": Transformer,
            "twistformer": DetectorTwistformer if cfg.twistor_detector else Twistformer,
            "pelican": Pelican,
            "gatr": GATr,
        }[cfg.family]
        self.backbone = (
            GATr(input_spec, cfg, 1 if len(task_spec.outputs) == 2 else len(task_spec.outputs))
            if self.direct_logits
            else (
                Pelican(input_spec, cfg, covariant=True) if self.covariant and cfg.family == "pelican" else factory(input_spec, cfg)
            )
        )
        if cfg.checkpoint_blocks:
            block_types = (Block, TwistorBlock, PairMessage, PairAggregation, TriangleInteraction)
            if not cfg.gatr_checkpoint:
                block_types += (GATrBlock,)
            # Preserve BatchNorm running statistics and avoid nested GATr checkpoints
            apply_activation_checkpointing(
                self.backbone,
                check_fn=lambda block: isinstance(block, block_types)
                and not any(isinstance(layer, nn.BatchNorm1d) for layer in block.modules()),
            )
        self.pool = Pool(
            cfg.width, cfg.pooling, cfg.pooling_heads or cfg.heads, cfg.pooling_depth, cfg.pooling_dropout, cfg.pooling_queries
        )
        self.global_dim = len(input_spec.global_features)
        self.global_mode = cfg.global_fusion
        self.parameter_dim = len(input_spec.parameters)
        if cfg.parameter_fusion == "tokens" and not self.parameter_dim:
            raise ValueError("Token parameter fusion requires declared parameters")
        if self.direct_logits and self.parameter_dim and cfg.parameter_fusion != "tokens":
            raise ValueError("Direct GATr logits require parameter_fusion=tokens for conditioning")
        self.parameter_embed = (
            mlp(self.parameter_dim, cfg.width, cfg.width)
            if self.parameter_dim and cfg.parameter_fusion == "tokens" else None
        )
        self.parameter_fuse = (
            mlp(cfg.width + self.parameter_dim, cfg.width * cfg.expansion, cfg.width)
            if self.parameter_dim and cfg.parameter_fusion == "readout" else None
        )
        if self.global_mode == "tokens" and not self.global_dim:
            raise ValueError("Token global fusion requires declared global features")
        self.global_embed = mlp(self.global_dim, cfg.width, cfg.width) if self.global_dim and self.global_mode == "tokens" else None
        self.global_fuse = (
            mlp(cfg.width + self.global_dim, cfg.width * cfg.expansion, cfg.width)
            if self.global_dim and self.global_mode == "readout"
            else None
        )
        self.head = (
            nn.Identity()
            if self.direct_logits
            else (
                CovariantHead(cfg.width, len(task_spec.outputs) // 4, cfg)
                if self.covariant
                else Head(cfg.width, task_spec, task_config, cfg)
            )
        )

    def forward(self, batch):
        """
        Encode events and apply the supervised head

        Args:
            batch:  Preprocessed event batch
        Returns:
            prediction: Supervised Prediction record
        """
        return self._forward(batch)

    def encode(self, batch):
        """
        Return invariant event features for conditional density models

        Args:
            batch:  Preprocessed event batch
        Returns:
            output: Pooled event features
        """
        if self.direct_logits or self.covariant:
            raise ValueError("Event encoding requires an invariant feature readout")
        return self._forward(batch, encoded_only=True)

    def _forward(self, batch, encoded_only=False):
        """
        Encode particle events and apply the shared task head

        Args:
            batch:  Validated Batch with fitted feature preprocessing
        Returns:
            prediction: Classification logits or Gaussian parameters
        """
        if self.requires_p4 and batch.p4 is None:
            raise ValueError("This model requires p4 geometry")
        if self.global_dim:
            if batch.global_features is None or batch.global_features.shape != (batch.mask.shape[0], self.global_dim):
                raise ValueError("Global feature shape must match the input schema")
        elif batch.global_features is not None:
            raise ValueError("This model does not declare global features")
        if self.parameter_dim:
            if batch.parameters is None or batch.parameters.shape != (batch.mask.shape[0], self.parameter_dim):
                raise ValueError("Parameter shape must match the input schema")
        elif batch.parameters is not None:
            raise ValueError("This model does not declare parameters")
        context = self.global_embed(batch.global_features) if self.global_embed is not None else None
        if self.parameter_embed is not None:
            embedded = self.parameter_embed(batch.parameters)
            context = embedded if context is None else context + embedded
        encoded = self.backbone(batch, global_context=context)
        if self.direct_logits:
            return Prediction(logits=encoded.features)
        if self.covariant:
            features = encoded.features
            if self.global_fuse is not None:
                global_features = batch.global_features[:, None].expand(-1, features.shape[1], -1)
                features = features + self.global_fuse(torch.cat([features, global_features], -1))
            if self.parameter_fuse is not None:
                parameters = batch.parameters[:, None].expand(-1, features.shape[1], -1)
                features = features + self.parameter_fuse(torch.cat([features, parameters], -1))
            return self.head(features, batch.p4, batch.mask)
        pooled = self.pool(encoded.features, batch)
        if self.global_fuse is not None:
            pooled = pooled + self.global_fuse(torch.cat([pooled, batch.global_features], -1))
        if self.parameter_fuse is not None:
            pooled = pooled + self.parameter_fuse(torch.cat([pooled, batch.parameters], -1))
        return pooled if encoded_only else self.head(pooled)
