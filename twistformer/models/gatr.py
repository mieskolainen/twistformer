# Geometric-algebra transformers for particle and detector inputs
# GATr: https://arxiv.org/abs/2305.18415
# E/P/C-GATr: https://arxiv.org/abs/2311.04744
# L-GATr: https://arxiv.org/abs/2405.14806
# L-GATr physics: https://arxiv.org/abs/2411.00446
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from twistformer.api.contracts import EncoderOutput
from twistformer.geometry.clifford import CliffordAlgebra
from twistformer.layers.gatr import EquivariantLinear, GATrBlock, GeometricNorm

class GATr(nn.Module):
    def __init__(self, input_spec, cfg, output_channels=None):
        """
        Construct E-GATr, P-GATr, improved P-GATr, C-GATr or L-GATr

        Args:
            input_spec:  Named particle or detector input features
            cfg:  Serializable model steering
            output_channels:  Optional direct classification logit count
        """
        super().__init__()
        self.cfg = cfg
        self.algebra = CliffordAlgebra(cfg.gatr_algebra, cfg.gatr_lorentz_proper)
        required = cfg.gatr_coordinates if cfg.gatr_input == "coordinates" else []
        for name in [*required, *cfg.auxiliary_features]:
            if name not in input_spec.features:
                raise ValueError(f"GATr input feature {name} is not declared")
        if cfg.gatr_input == "p4" and input_spec.geometry != "p4":
            raise ValueError("GATr p4 input requires explicit four-momenta")
        self.coordinates = [input_spec.features.index(name) for name in required]
        self.auxiliary = [input_spec.features.index(name) for name in cfg.auxiliary_features]
        self.context_dim = cfg.width if cfg.global_fusion == "tokens" or cfg.parameter_fusion == "tokens" else 0
        self.input_scalars = 1 + len(self.auxiliary) + self.context_dim
        dimension = 4 if cfg.gatr_algebra == "lorentz" else 3
        references = torch.empty(0, dimension)
        if cfg.gatr_spurions == "frame":
            references = torch.eye(dimension)
            if cfg.gatr_algebra != "lorentz":
                references = torch.cat((torch.zeros(1, dimension), references))
        elif cfg.gatr_spurions == "time":
            references = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        elif cfg.gatr_spurions == "beam":
            references = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        references = self.algebra.embed(references)
        if cfg.gatr_spurions == "beam_plane":
            references = torch.zeros(2, self.algebra.dim)
            references[0, self.algebra.index[6]] = 1
            references[1, self.algebra.index[1]] = 1
        self.register_buffer("references", references)
        channels = 1 if cfg.gatr_reference_tokens else 1 + len(references)
        self.input = EquivariantLinear(self.algebra, channels, cfg.gatr_mv_channels, self.input_scalars, cfg.width)
        self.blocks = nn.ModuleList([GATrBlock(self.algebra, cfg, channels, self.input_scalars) for _ in range(cfg.depth)])
        self.norm = GeometricNorm(self.algebra, cfg.width, cfg.gatr_norm_epsilon) if cfg.final_norm else None
        self.output = (
            EquivariantLinear(self.algebra, cfg.gatr_mv_channels, cfg.gatr_mv_channels, cfg.width, cfg.width)
            if cfg.gatr_readout == "components"
            else nn.Linear(cfg.width + cfg.gatr_mv_channels * len(self.algebra.functionals), cfg.width)
        )
        self.component_readout = (
            nn.Linear(cfg.gatr_mv_channels * self.algebra.dim, cfg.width) if cfg.gatr_readout == "components" else None
        )
        if cfg.gatr_readout == "logits":
            if output_channels is None:
                raise ValueError("Direct GATr readout requires a classification output count")
            self.output = EquivariantLinear(self.algebra, cfg.gatr_mv_channels, output_channels, cfg.width, 0)

    def embed(self, batch, global_context=None):
        """
        Attach geometric inputs, invariant scalars and explicit reference channels

        Args:
            batch:  Padded physical input batch
            global_context:  Optional event scalar embedding (B,S)
        Returns:
            streams: Input multivectors, scalar stream, mask and join reference
        """
        cfg, algebra = self.cfg, self.algebra
        dtype = self.input.weight.dtype
        mask = batch.mask
        if cfg.gatr_input == "p4":
            coordinates = batch.p4.to(dtype)
            if cfg.gatr_algebra != "lorentz":
                coordinates = coordinates[..., 1:]
        else:
            coordinates = batch.features[..., self.coordinates].to(dtype)
        coordinates = coordinates.masked_fill(~mask[..., None], 0) * cfg.gatr_input_scale
        if cfg.gatr_center:
            center = coordinates.sum(1, keepdim=True) / mask.sum(1)[:, None, None].clamp_min(1)
            coordinates = coordinates - center
        mv = algebra.embed(coordinates).unsqueeze(-2)
        if len(self.references) and not cfg.gatr_reference_tokens:
            refs = self.references.to(dtype)
            mv = torch.cat((mv, refs[None, None].expand(*mask.shape, -1, -1)), -2)
        mv = mv.masked_fill(~mask[..., None, None], 0)
        scalars = [torch.zeros(*mask.shape, 1, dtype=dtype, device=mv.device)]
        if self.auxiliary:
            auxiliary = batch.features[..., self.auxiliary].to(dtype).masked_fill(~mask[..., None], 0)
            scalars.append(auxiliary)
        if global_context is not None:
            scalars.append(global_context.to(dtype)[:, None].expand(-1, mask.shape[1], -1))
        scalars = torch.cat(scalars, -1).masked_fill(~mask[..., None], 0)
        # The join reference excludes padding and the global token
        reference = (
            (mv[..., algebra.index[14]].sum((1, 2)) / (mask.sum(1).clamp_min(1) * mv.shape[-2]))[:, None, None, None]
            if cfg.gatr_algebra == "projective"
            else None
        )
        if cfg.gatr_reference_tokens:
            count = len(self.references)
            refs = self.references.to(dtype)[None, :, None].expand(mask.shape[0], -1, -1, -1)
            mv = torch.cat((refs, mv), 1)
            scalars = torch.cat((scalars.new_zeros(mask.shape[0], count, self.input_scalars), scalars), 1)
            mask = torch.cat((mask.new_ones(mask.shape[0], count), mask), 1)
        if cfg.gatr_global_token:
            mv = torch.cat((torch.zeros_like(mv[:, :1]), mv), 1)
            token = torch.zeros_like(scalars[:, :1])
            token[..., 0] = 1
            scalars = torch.cat((token, scalars), 1)
            mask = torch.cat((torch.ones_like(mask[:, :1]), mask), 1)
        return mv, scalars, mask, reference

    def prepare_onnx(self, module, inputs):
        """
        Prepare the inference graph for ONNX conversion

        Args:
            module:  Shared inference module with preprocessing and task outputs
            inputs:  Example tensor inputs
        Returns:
            module: Exportable inference module
        """
        return module

    def forward(self, batch, global_context=None):
        """
        Encode geometric tokens with protected metric contractions

        Args:
            batch:  Particle or detector batch
            global_context:  Optional invariant event channels
        Returns:
            encoded: Invariant or explicitly frame-dependent task features
        """
        device_type = batch.features.device.type
        # Conformal metric cancellation needs FP32 even when the scalar head uses AMP
        attention_dtype = (
            torch.get_autocast_dtype(device_type)
            if torch.is_autocast_enabled(device_type) and self.algebra.kind != "conformal"
            else None
        )
        # Keep geometric products and norms in FP32
        with torch.autocast(device_type=batch.features.device.type, enabled=False):
            mv, scalars, mask, reference = self.embed(batch, global_context)
            original = (mv, scalars)
            mv, scalars = self.input(mv, scalars)
            for block in self.blocks:
                if self.cfg.gatr_checkpoint and self.training:
                    mv, scalars = checkpoint(block, mv, scalars, mask, original, reference, attention_dtype, use_reentrant=False)
                else:
                    mv, scalars = block(mv, scalars, mask, original, reference, attention_dtype)
            if self.norm is not None:
                mv, scalars = self.norm(mv, scalars)
            if self.cfg.gatr_readout == "logits":
                mv, _ = self.output(mv[:, :1], scalars[:, :1])
                return EncoderOutput(mv[:, 0, :, 0])
            if self.component_readout is not None:
                mv, scalars = self.output(mv, scalars)
                scalars = scalars + self.component_readout(mv.flatten(-2))
            else:
                scalars = self.output(torch.cat((scalars, self.algebra.invariants(mv).flatten(-2)), -1))
            scalars = scalars.masked_fill(~mask[..., None], 0)
            return EncoderOutput(scalars[:, 0] if self.cfg.gatr_global_token else scalars)
