# Particle model encoders
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from torch import nn

from twistformer.api.contracts import EncoderOutput
from twistformer.layers.attention import Attention, ProjectiveAttention, TriangleAttention
from twistformer.layers.physics import MandelstamBias, JetBias
from twistformer.layers.common import mlp, normalization, ResidualMLP


def neighbor_bias(coordinates, mask, neighbors):
    """
    Restrict each cell to its nearest neighbours in physical coordinates

    Args:
        coordinates:  Cell positions in common length units, shape (B,N,3)
        mask:  Valid cell mask, shape (B,N)
        neighbors:  Maximum number of other cells per query
    Returns:
        bias: Additive graph mask with a self edge only for isolated cells
    """
    with torch.autocast(device_type=coordinates.device.type, enabled=False):
        points = coordinates.float()
        distance = (points[:, :, None] - points[:, None, :]).square().sum(-1)
        diagonal = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        valid = mask[:, :, None] & mask[:, None, :]
        distance = distance.masked_fill(~valid | diagonal, torch.inf)
        # Resolve equal distances in cell order, independent of batch padding
        indices = distance.argsort(dim=-1, stable=True)[..., :neighbors]
        keep = torch.zeros_like(distance, dtype=torch.bool).scatter(-1, indices, torch.isfinite(distance.gather(-1, indices)))
        keep |= diagonal & valid & ~keep.any(-1, keepdim=True)
        return torch.zeros_like(distance).masked_fill(~keep, -torch.inf)[:, None]


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.width
        self.norm1, self.norm2 = normalization(cfg.norm, d), normalization(cfg.norm, d)
        self.attention = (
            TriangleAttention(
                d,
                cfg.heads,
                cfg.triangle_k,
                cfg.triangle_gamma,
                cfg.attention_dropout,
                cfg.query_chunk,
                kj=cfg.triangle_kj,
                kk=cfg.triangle_kk,
                shared=cfg.triangle_shared_candidates,
                distinct=cfg.triangle_distinct,
                backend=cfg.attention_backend,
            )
            if cfg.family == "triformer"
            else (
                ProjectiveAttention(d, cfg.heads, cfg.attention_dropout, cfg.projective_channels, bool(cfg.projective_coordinates), backend=cfg.attention_backend)
                if cfg.attention == "projective"
                else Attention(d, cfg.heads, cfg.attention_dropout, cfg.attention, backend=cfg.attention_backend)
            )
        )
        self.ff = mlp(d, d * cfg.expansion, d, cfg.dropout, norm=cfg.ffn_norm, activation=cfg.ffn_activation)
        self.drop = nn.Dropout(cfg.dropout if cfg.residual_dropout is None else cfg.residual_dropout)
        self.edge_scale = nn.Parameter(torch.tensor(cfg.jet_block_scale_init)) if cfg.pair_bias == "jet" else None

    def forward(self, x, mask, bias, coordinates=None):
        """
        Apply masked attention and scalar feature updates

        Args:
            x:  Learned token features (B,N,D)
            mask:  Valid particle mask (B,N)
            bias:  Optional per-head pair bias
            coordinates:  Optional common-scale input points (B,N,3)
        Returns:
            features: Updated token features (B,N,D)
        """
        if bias is not None and self.edge_scale is not None:
            bias = self.edge_scale * bias
        normalized = self.norm1(x)
        attended = (
            self.attention(normalized, mask, bias, coordinates)
            if isinstance(self.attention, ProjectiveAttention)
            else self.attention(normalized, mask, bias)
        )
        x = x + self.drop(attended)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x.masked_fill(~mask[..., None], 0)

class Transformer(nn.Module):
    def __init__(self, input_spec, cfg):
        """
        Construct particle attention with optional laboratory axis tokens

        Args:
            input_spec:  Named particle or detector input features
            cfg:  Serializable model steering
        """
        super().__init__()
        self.coordinates = []
        self.neighbors = cfg.attention_neighbors
        self.neighbor_coordinates = []
        if self.neighbors:
            if any(name not in input_spec.features for name in ("x", "y", "z")):
                raise ValueError("Cell neighbours require x,y,z input features")
            self.neighbor_coordinates = [input_spec.features.index(name) for name in ("x", "y", "z")]
        self.register_buffer("feature_scale", None)
        if cfg.attention == "projective" and cfg.projective_coordinates:
            if any(name not in input_spec.features for name in cfg.projective_coordinates):
                raise ValueError("Projective coordinates must be declared input features")
            self.coordinates = [input_spec.features.index(name) for name in cfg.projective_coordinates]
            scale = torch.ones(len(input_spec.features))
            scale[self.coordinates] = cfg.projective_input_scale
            self.feature_scale = scale
        dropout = cfg.dropout if cfg.embedding_dropout is None else cfg.embedding_dropout
        self.embed = ResidualMLP(
            len(input_spec.features),
            cfg.width,
            cfg.embedding if isinstance(cfg.embedding, list) else [cfg.width],
            dropout=dropout,
            norm=cfg.embedding_norm,
        )
        if cfg.embedding == "two_stage":
            self.embed = nn.Sequential(
                self.embed, mlp(cfg.width, cfg.width * cfg.expansion, cfg.width, dropout, norm=cfg.embedding_norm)
            )
        self.frame_embedding = None
        self.register_buffer("frame_features", None)
        if cfg.reference_token == "frame":
            axes = next(
                (names for names in (("x", "y", "z"), ("px", "py", "pz")) if all(name in input_spec.features for name in names)),
                None,
            )
            if axes is None:
                raise ValueError("Transformer frame tokens require x,y,z or px,py,pz")
            # Unit axes use model input units after feature preprocessing
            references = torch.zeros(6, len(input_spec.features))
            directions = torch.eye(3).repeat_interleave(2, dim=0)
            directions[1::2] *= -1
            references[:, [input_spec.features.index(name) for name in axes]] = directions
            self.frame_features = references
            self.frame_embedding = nn.Embedding(6, cfg.width)
            nn.init.normal_(self.frame_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm = normalization(cfg.norm, cfg.width) if cfg.final_norm else nn.Identity()
        self.bias = (
            JetBias(cfg.heads, cfg)
            if cfg.pair_bias == "jet"
            else (MandelstamBias(cfg.heads) if cfg.pair_bias == "mandelstam" else None)
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
            if isinstance(block.attention, TriangleAttention):
                block.attention.fixed_candidates = True
        return module

    def forward(self, batch, global_context=None):
        """
        Encode physical events into scalar features

        Args:
            batch:  Validated, padded particle Batch
            global_context:  Optional embedded invariant event scalars (B,D)
        Returns:
            encoded: EncoderOutput with token features or graph readouts
        """
        features = batch.features.masked_fill(~batch.mask[..., None], 0)
        if self.feature_scale is not None:
            features = features * self.feature_scale.to(features.dtype)
        mask = batch.mask
        if self.frame_embedding is not None:
            references = self.frame_features.to(features.dtype)
            features = torch.cat((features, references[None].expand(features.shape[0], -1, -1)), dim=1)
            mask = torch.nn.functional.pad(mask, (0, 6), value=True)
        coordinates = features[..., self.coordinates] if self.coordinates else None
        x = self.embed(features)
        if self.frame_embedding is not None:
            # Labels distinguish +x, -x, +y, -y, +z and -z references
            references = x[:, -6:] + self.frame_embedding.weight.to(x.dtype)
            x = torch.cat((x[:, :-6], references), dim=1)
        if global_context is not None:
            x = x + global_context[:, None]
        x = x.masked_fill(~mask[..., None], 0)
        bias = self.bias(batch.p4, batch.mask) if self.bias is not None else None
        if self.neighbors:
            bias = neighbor_bias(batch.features[..., self.neighbor_coordinates], batch.mask, self.neighbors)
        if bias is not None and self.frame_embedding is not None:
            # Reference tokens have no physical four-momentum pair bias
            bias = torch.nn.functional.pad(bias, (0, 6, 0, 6))
        for block in self.blocks:
            x = block(x, mask, bias, coordinates)
        x = x[:, : batch.mask.shape[1]]
        return EncoderOutput(self.norm(x).masked_fill(~batch.mask[..., None], 0))
