# Equivariant geometric-algebra attention and bilinear interactions
# GATr: https://arxiv.org/abs/2305.18415
# E/P/C-GATr: https://arxiv.org/abs/2311.04744
# L-GATr: https://arxiv.org/abs/2405.14806
# L-GATr physics: https://arxiv.org/abs/2411.00446
#
# mikael.mieskolainen@cern.ch (2026)

import math
import torch
from torch import nn
from torch.nn import functional as F
from .attention import scaled_attention

def project_streams(algebra, mv, scalars, weight, mv_weight, mv_bias, s_weight, s_bias):
    """
    Project geometric and scalar streams with dense matrix kernels

    Args:
        algebra:  Clifford algebra and equivariant basis
        mv:  Multivector input (...,C,D)
        scalars:  Scalar input (...,S)
        weight:  Equivariant channel coefficients
        mv_weight:  Scalar-to-multivector weights
        mv_bias:  Scalar-to-multivector bias
        s_weight:  Optional invariant scalar projection
        s_bias:  Optional scalar output bias
    Returns:
        streams: Projected multivectors and optional scalars
    """
    out_mv, in_mv = weight.shape[:2]
    dim = algebra.dim
    kernel = weight.flatten(0, 1) @ algebra.linear_basis.flatten(1)
    kernel = kernel.reshape(out_mv, in_mv, dim, dim).permute(0, 2, 1, 3)
    result = F.linear(mv.flatten(-2), kernel.reshape(out_mv * dim, in_mv * dim))
    result = result.unflatten(-1, (out_mv, dim))
    invariant = F.linear(scalars, mv_weight, mv_bias).unflatten(-1, (out_mv, -1))
    result = result + invariant @ algebra.constants
    scalar_output = None
    if s_weight is not None:
        inputs = torch.cat((scalars, algebra.invariants(mv).flatten(-2)), -1)
        scalar_output = F.linear(inputs, s_weight, s_bias)
    return result, scalar_output

def project_group(layers, mv, scalars):
    """
    Combine projections that share the same geometric and scalar inputs

    Args:
        layers:  Equivariant maps with matching input streams
        mv:  Multivector input (...,C,D)
        scalars:  Scalar input (...,S)
    Returns:
        streams: One projected stream pair for each input layer
    """
    scalar_maps = [layer.to_s for layer in layers if layer.to_s is not None]
    projected, scalar_output = project_streams(
        layers[0].algebra,
        mv,
        scalars,
        torch.cat([layer.weight for layer in layers]),
        torch.cat([layer.to_mv.weight for layer in layers]),
        torch.cat([layer.to_mv.bias for layer in layers]),
        torch.cat([layer.weight for layer in scalar_maps]) if scalar_maps else None,
        torch.cat([layer.bias for layer in scalar_maps]) if scalar_maps else None,
    )
    multivectors = projected.split([layer.out_mv for layer in layers], dim=-2)
    scalar_sizes = [layer.to_s.out_features if layer.to_s is not None else 0 for layer in layers]
    scalars = (
        [part if size else None for part, size in zip(scalar_output.split(scalar_sizes, dim=-1), scalar_sizes)]
        if scalar_maps
        else [None] * len(layers)
    )
    return list(zip(multivectors, scalars))

class EquivariantLinear(nn.Module):
    def __init__(self, algebra, in_mv, out_mv, in_s, out_s):
        """
        Couple multivectors through the complete equivariant map basis

        Args:
            algebra:  Clifford algebra and its symmetry group
            in_mv:  Input multivector channels
            out_mv:  Output multivector channels
            in_s:  Input scalar channels
            out_s:  Output scalar channels
        """
        super().__init__()
        self.algebra = algebra
        basis = len(algebra.linear_basis)
        self.weight = nn.Parameter(torch.randn(out_mv, in_mv, basis) / math.sqrt(in_mv * basis))
        self.to_mv = nn.Linear(in_s, out_mv * len(algebra.constants))
        self.to_s = nn.Linear(in_s + in_mv * len(algebra.functionals), out_s) if out_s else None
        self.out_mv = out_mv
        if algebra.kind in {"euclidean", "projective"}:
            self.reset_spatial_parameters()
        elif algebra.kind == "conformal":
            self.reset_conformal_parameters()
        elif algebra.kind == "lorentz":
            self.reset_lorentz_parameters()

    @torch.no_grad()
    def reset_spatial_parameters(self):
        """
        Set spatial map variance from channel fan-in and blade support

        Returns:
            None: Grade projectors and null maps have variance 1 / (3 * in_mv)
        """
        # Frobenius-normalized grade maps do not add independent fan-in channels
        support = self.algebra.linear_basis.abs().amax((-2, -1)).reciprocal()
        self.weight.normal_(std=1 / math.sqrt(3 * self.weight.shape[1])).mul_(support)

    @torch.no_grad()
    def reset_conformal_parameters(self):
        """
        Initialize channel mixing with the identity on conformal components

        Returns:
            None: Multivector weights are reset, scalar maps are retained
        """
        mixing = torch.empty_like(self.weight[..., 0])
        nn.init.kaiming_uniform_(mixing, a=math.sqrt(5))
        self.weight.zero_()
        # Compensate the Frobenius normalization of each grade projector
        grades = self.algebra.grade_mask.sum(-1).sqrt()
        self.weight[..., :len(grades)].copy_(mixing[..., None] * grades)

    @torch.no_grad()
    def reset_lorentz_parameters(self, scale=1.0, shift=0.0):
        """
        Initialize the Lorentz maps with the public implementation fan-in rule

        Args:
            scale:  Weight scale for default, small or almost-unit maps
            shift:  Constant offset for the almost-unit bilinear operand
        """
        out_mv, in_mv, _ = self.weight.shape
        in_s = self.to_mv.in_features
        self.weight.uniform_(-scale / math.sqrt(in_mv), scale / math.sqrt(in_mv))
        scalar_maps = self.algebra.linear_basis[:, 0].abs().sum(-1) > 0
        self.weight[..., scalar_maps] /= math.sqrt(2)
        self.to_mv.weight.uniform_(-scale / math.sqrt(2 * max(in_s, 1)), scale / math.sqrt(2 * max(in_s, 1)))
        bound = 1 / math.sqrt(in_s + in_mv)
        self.to_mv.bias.uniform_(shift - bound, shift + bound)
        if self.to_s is not None:
            mv_inputs = in_mv * len(self.algebra.functionals)
            self.to_s.weight[:, :in_s].uniform_(-scale / math.sqrt(2 * max(in_s, 1)), scale / math.sqrt(2 * max(in_s, 1)))
            self.to_s.weight[:, in_s:].uniform_(-scale / math.sqrt(2 * mv_inputs), scale / math.sqrt(2 * mv_inputs))
            self.to_s.bias.uniform_(-scale / math.sqrt(in_s + mv_inputs), scale / math.sqrt(in_s + mv_inputs))

    def forward(self, mv, scalars):
        """
        Mix geometric channels and invariant scalar features

        Args:
            mv:  Input multivectors (...,C,D)
            scalars:  Scalar stream (...,S)
        Returns:
            outputs: Multivector and scalar streams
        """
        return project_streams(
            self.algebra,
            mv,
            scalars,
            self.weight,
            self.to_mv.weight,
            self.to_mv.bias,
            self.to_s.weight if self.to_s is not None else None,
            self.to_s.bias if self.to_s is not None else None,
        )

class GeometricNorm(nn.Module):
    def __init__(self, algebra, scalar_channels, epsilon):
        """
        Normalize geometric norms and the auxiliary scalar stream

        Args:
            algebra:  Clifford algebra
            scalar_channels:  Scalar channel count
            epsilon:  Floor for null or near-null multivectors
        """
        super().__init__()
        self.algebra, self.epsilon = algebra, epsilon
        self.scalar_norm = nn.LayerNorm(scalar_channels, elementwise_affine=algebra.kind != "lorentz")

    def forward(self, mv, scalars):
        """
        Apply grade-wise absolute-norm normalization

        Args:
            mv:  Multivector stream (...,C,D)
            scalars:  Scalar stream (...,S)
        Returns:
            normalized: Geometric and scalar streams
        """
        norm = self.algebra.norm_squared(mv).mean(-2, keepdim=True)
        denominator = norm.clamp_min(self.epsilon) if self.algebra.kind == "lorentz" else norm + self.epsilon
        return mv * denominator.rsqrt(), self.scalar_norm(scalars)

def distance_features(mv, epsilon, query):
    """
    Lift projective trivectors into negative squared-distance dot products

    Args:
        mv:  Projective multivectors (...,C,16)
        epsilon:  Homogeneous-coordinate regularization
        query:  Select the query or key factorization
    Returns:
        features: Five distance features per channel (...,C,5)
    """
    tri = mv[..., 11:15]
    tri = tri * (tri[..., 3:4] / (tri[..., 3:4].square() + epsilon))
    xyz, weight = tri[..., :3], tri[..., 3:4]
    square = xyz.square().sum(-1, keepdim=True)
    if query:
        return torch.cat((square, weight.square(), xyz * weight), -1)
    return torch.cat((-weight.square(), -square, 2 * xyz * weight), -1)

class GeometricAttention(nn.Module):
    def __init__(self, algebra, cfg, input_channels, input_scalars):
        """
        Build multi-head or multi-query geometric self-attention

        Args:
            algebra:  Clifford algebra
            cfg:  Model configuration
            input_channels:  Reinserted multivector channel count
            input_scalars:  Reinserted scalar channel count
        """
        super().__init__()
        self.algebra, self.heads = algebra, cfg.heads
        self.channels = max(1, cfg.gatr_mv_channels * cfg.gatr_attention_ratio // cfg.heads)
        self.scalars = max(4, cfg.width * cfg.gatr_attention_ratio // cfg.heads)
        self.kv_heads = 1 if cfg.gatr_multi_query else cfg.heads
        self.reinsert = cfg.gatr_reinsert
        self.improved = algebra.kind == "projective" and cfg.gatr_projective_improved
        self.epsilon = cfg.gatr_distance_epsilon
        self.dropout = cfg.attention_dropout
        self.backend = cfg.attention_backend
        qk_mv = cfg.gatr_mv_channels + (input_channels if self.reinsert else 0)
        qk_scalars = cfg.width + (input_scalars if self.reinsert else 0)
        self.q = EquivariantLinear(algebra, qk_mv, self.heads * self.channels, qk_scalars, self.heads * self.scalars)
        self.k = EquivariantLinear(
            algebra, qk_mv, self.kv_heads * self.channels, qk_scalars, self.kv_heads * self.scalars
        )
        if self.reinsert:
            # Learn raw geometry and scalar contributions from zero
            invariant_offset = qk_scalars + cfg.gatr_mv_channels * len(algebra.functionals)
            with torch.no_grad():
                for layer in (self.q, self.k):
                    layer.weight[:, cfg.gatr_mv_channels:].zero_()
                    layer.to_mv.weight[:, cfg.width:].zero_()
                    layer.to_s.weight[:, cfg.width:qk_scalars].zero_()
                    layer.to_s.weight[:, invariant_offset:].zero_()
        self.v = EquivariantLinear(
            algebra, cfg.gatr_mv_channels, self.kv_heads * self.channels, cfg.width, self.kv_heads * self.scalars
        )
        self.out = EquivariantLinear(
            algebra, self.heads * self.channels, cfg.gatr_mv_channels, self.heads * self.scalars, cfg.width
        )
        if algebra.kind == "lorentz":
            self.out.reset_lorentz_parameters(scale=0.1)
        self.head_scale = nn.Parameter(torch.ones(self.heads)) if cfg.gatr_head_scale else None
        self.distance_weight = (
            nn.Parameter(torch.zeros(self.heads, 1, self.channels, 1))
            if self.improved
            else None
        )

    def _split(self, streams, heads):
        """
        Separate attention heads from projected channels

        Args:
            streams:  Projected multivector and scalar pair
            heads:  Number of query or key/value heads
        Returns:
            split: Streams ordered B,H,N,C,D and B,H,N,S
        """
        mv, s = streams
        return (mv.unflatten(-2, (heads, self.channels)).transpose(1, 2), s.unflatten(-1, (heads, self.scalars)).transpose(1, 2))

    def forward(self, mv, scalars, mask, original, attention_dtype=None):
        """
        Attend with invariant logits and covariant values

        Args:
            mv:  Multivector stream (B,N,C,D)
            scalars:  Scalar stream (B,N,S)
            mask:  Valid tokens (B,N)
            original:  Input streams for optional query/key reinsertion
            attention_dtype:  Optional AMP dtype for the attention contraction
        Returns:
            attended: Multivector and scalar residual updates
        """
        if self.reinsert:
            raw, raw_s = original
            query, key = project_group((self.q, self.k), torch.cat((mv, raw), -2), torch.cat((scalars, raw_s), -1))
            value = self.v(mv, scalars)
        else:
            query, key, value = project_group((self.q, self.k, self.v), mv, scalars)
        q, qs = self._split(query, self.heads)
        k, ks = self._split(key, self.kv_heads)
        v, vs = self._split(value, self.kv_heads)
        if self.improved:
            indices = [0, 2, 3, 4, 8, 9, 10]
            qf = torch.cat(
                (
                    q[..., indices].flatten(-2),
                    (distance_features(q, self.epsilon, True) * self.distance_weight.exp()).flatten(-2),
                    qs,
                ),
                -1,
            )
            kf = torch.cat((k[..., indices].flatten(-2), distance_features(k, self.epsilon, False).flatten(-2), ks), -1)
        else:
            qf = torch.cat(((q * self.algebra.metric).flatten(-2), qs), -1)
            kf = torch.cat((k.flatten(-2), ks), -1)
        vf = torch.cat((v.flatten(-2), vs), -1)
        size = 8 * math.ceil(max(qf.shape[-1], vf.shape[-1]) / 8)
        scale = qf.shape[-1] ** -0.5
        streams = [F.pad(t, (0, size - t.shape[-1])) for t in (qf, kf, vf)]
        if attention_dtype is not None and qf.dtype != torch.float64:
            streams = [t.to(attention_dtype) for t in streams]
        result = scaled_attention(*streams, mask, dropout=self.dropout if self.training else 0.0, scale=scale, backend=self.backend).to(vf.dtype)
        if self.head_scale is not None:
            result = result * self.head_scale[None, :, None, None]
        size_mv = self.channels * self.algebra.dim
        out_mv = result[..., :size_mv].unflatten(-1, (self.channels, self.algebra.dim))
        out_s = result[..., size_mv : size_mv + self.scalars]
        return self.out(out_mv.transpose(1, 2).flatten(-3, -2), out_s.transpose(1, 2).flatten(-2))

class GeometricMLP(nn.Module):
    def __init__(self, algebra, cfg):
        """
        Form geometric products, projective joins and scalar-gated nonlinearities

        Args:
            algebra:  Clifford algebra
            cfg:  Channel widths and algebra options
        """
        super().__init__()
        self.algebra = algebra
        self.use_join = algebra.kind == "projective" and cfg.gatr_projective_improved
        channels = cfg.gatr_mv_channels * cfg.expansion
        # Split the hidden width with at least one channel per active bilinear
        join_channels = max(1, channels // 2) if self.use_join else 0
        product_channels = max(1, channels - join_channels)
        self.left = EquivariantLinear(algebra, cfg.gatr_mv_channels, product_channels, cfg.width, 0)
        self.right = EquivariantLinear(algebra, cfg.gatr_mv_channels, product_channels, cfg.width, 0)
        if self.use_join:
            self.join_left = EquivariantLinear(algebra, cfg.gatr_mv_channels, join_channels, cfg.width, 0)
            self.join_right = EquivariantLinear(algebra, cfg.gatr_mv_channels, join_channels, cfg.width, 0)
        self.bilinear = EquivariantLinear(
            algebra, product_channels + join_channels, channels, cfg.width, cfg.width * cfg.expansion
        )
        self.norm = GeometricNorm(algebra, cfg.width * cfg.expansion, cfg.gatr_norm_epsilon) if cfg.gatr_bilinear_norm else None
        self.out = EquivariantLinear(algebra, channels, cfg.gatr_mv_channels, cfg.width * cfg.expansion, cfg.width)
        with torch.no_grad():
            self.right.weight.mul_(0.1)
            self.right.to_mv.weight.mul_(0.1)
            self.right.to_mv.bias.zero_()
            self.right.to_mv.bias.view(product_channels, -1)[:, 0] = 1
        if algebra.kind == "lorentz":
            self.right.reset_lorentz_parameters(scale=0.5, shift=1.0)

    def forward(self, mv, scalars, reference):
        """
        Evaluate channelwise bilinears followed by invariant gates

        Args:
            mv:  Multivector stream
            scalars:  Scalar stream
            reference:  Masked input orientation for the projective join
        Returns:
            updates: Geometric and scalar residual updates
        """
        layers = (self.left, self.right)
        if self.use_join:
            layers += (self.join_left, self.join_right)
        operands = project_group(layers, mv, scalars)
        (left, _), (right, _) = operands[:2]
        product = self.algebra.product(left, right)
        if self.use_join:
            (a, _), (b, _) = operands[2:]
            product = torch.cat((product, self.algebra.join(a, b, reference)), -2)
        product, hidden = self.bilinear(product, scalars)
        if self.norm is not None:
            product, hidden = self.norm(product, hidden)
        approximation = "tanh" if self.algebra.kind == "lorentz" else "none"
        return self.out(product * F.gelu(product[..., :1], approximate=approximation), F.gelu(hidden))

class GATrBlock(nn.Module):
    def __init__(self, algebra, cfg, input_channels, input_scalars):
        """
        Combine pre-normalized geometric attention and bilinear residuals

        Args:
            algebra:  Clifford algebra
            cfg:  Model configuration
            input_channels:  Raw multivector channel count
            input_scalars:  Raw scalar channel count
        """
        super().__init__()
        self.algebra = algebra
        self.norm1 = GeometricNorm(algebra, cfg.width, cfg.gatr_norm_epsilon)
        self.norm2 = GeometricNorm(algebra, cfg.width, cfg.gatr_norm_epsilon)
        self.attention = GeometricAttention(algebra, cfg, input_channels, input_scalars)
        self.mlp = GeometricMLP(algebra, cfg)
        self.dropout = cfg.dropout
        self.scale = (2 * cfg.depth) ** -0.5 if cfg.gatr_residual_scaling else 1.0

    def _residual(self, mv, scalars, update, mask):
        """
        Apply grade dropout and exclude padding from each residual

        Args:
            mv:  Current multivectors
            scalars:  Current scalar stream
            update:  Residual multivector and scalar streams
            mask:  Valid token mask
        Returns:
            streams: Masked residual streams
        """
        dm, ds = update
        if self.training and self.dropout:
            grade_keep = F.dropout(
                torch.ones(*dm.shape[:-1], len(self.algebra.grade_mask), device=dm.device, dtype=dm.dtype), self.dropout, True
            )
            dm = dm * grade_keep[..., self.algebra.grades]
        return (
            (mv + self.scale * dm).masked_fill(~mask[..., None, None], 0),
            (scalars + self.scale * F.dropout(ds, self.dropout, self.training)).masked_fill(~mask[..., None], 0),
        )

    def forward(self, mv, scalars, mask, original, reference, attention_dtype=None):
        """
        Update multivector and scalar streams for one transformer block

        Args:
            mv:  Current multivectors
            scalars:  Current scalar stream
            mask:  Valid token mask
            original:  Raw input streams
            reference:  Projective join orientation
            attention_dtype:  Optional AMP dtype for the attention contraction
        Returns:
            streams: Updated multivector and scalar streams
        """
        update = self.attention(*self.norm1(mv, scalars), mask, original, attention_dtype)
        mv, scalars = self._residual(mv, scalars, update, mask)
        return self._residual(mv, scalars, self.mlp(*self.norm2(mv, scalars), reference), mask)
