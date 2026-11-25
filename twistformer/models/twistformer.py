# Twistor and detector-frame particle encoders
#
# mikael.mieskolainen@cern.ch (2026)

from dataclasses import replace

import torch
from torch import nn
from twistformer.api.contracts import EncoderOutput
from twistformer.geometry.lorentz import invariant_features
from twistformer.geometry.twistors import (
    twistors_from_p4,
    pairwise_features,
    three_point_features,
    geometry_update,
    projective_normalize,
    transport_channels,
    append_reference,
    laboratory_reference_features,
    rest_frame_coordinates,
)
from twistformer.layers.common import mlp, normalization, AuxiliaryEmbedding
from twistformer.geometry.pairs import PairLinear
from twistformer.layers.representations import RepresentationLinear
from twistformer.layers.attention import scaled_attention
from .transformer import Transformer

class TwistorBlock(nn.Module):
    def __init__(self, cfg):
        """
        Construct twistor attention with optional fixed plane contractions

        Args:
            cfg:  Serializable model steering
        """
        super().__init__()
        self.heads, self.width = cfg.heads, cfg.width
        self.geometry_scale = cfg.twistor_geometry_scale
        self.representation_scale = cfg.twistor_representation_scale
        self.residual_scale = (2 * cfg.depth) ** -0.5 if cfg.twistor_residual_scaling == "depth" else 1.0
        self.qk = nn.Linear(cfg.width, 2 * cfg.width, bias=False) if cfg.twistor_qk else None
        self.three_point, self.update = cfg.three_point, cfg.geometry_update
        pair_dim = 6 if cfg.three_point else 4
        pair_dim += int(cfg.reference_token == "beam_plane")
        self.pair = PairLinear(pair_dim, cfg.pair_channels, normalize=True) if cfg.pair_channels else None
        self.edge = mlp(cfg.pair_channels or pair_dim, cfg.width, cfg.heads)
        self.transport = cfg.twistor_transport
        self.rep_mix = (
            RepresentationLinear(cfg.representation_channels, cfg.representation_channels) if cfg.representation_channels else None
        )
        self.rep_readout = mlp(2 * cfg.representation_channels, cfg.width, cfg.width) if cfg.representation_channels else None
        self.rep_gate = nn.Linear(cfg.width, cfg.representation_channels) if cfg.representation_channels else None
        self.norm = normalization(cfg.norm, cfg.width)
        self.ff_norm = normalization(cfg.norm, cfg.width) if cfg.twistor_separate_norm else None
        self.value = nn.Linear(cfg.width, cfg.width)
        self.ff = mlp(
            cfg.width, cfg.width * cfg.expansion, cfg.width, cfg.dropout, norm=cfg.ffn_norm, activation=cfg.ffn_activation
        )
        self.attention_dropout = cfg.attention_dropout
        self.attention_backend = cfg.attention_backend
        self.drop = nn.Dropout(cfg.dropout)
        self.step = nn.Linear(cfg.width, 1) if self.update else None
        if self.step is not None:
            nn.init.zeros_(self.step.weight)
            nn.init.zeros_(self.step.bias)

    def forward(self, h, z, metric, mask, channels=None, reference_pairs=None):
        """
        Update invariant scalar channels and covariant twistor coordinates

        Args:
            h:  Scalar token channels (B,N,D)
            z:  Complex twistor coordinates (B,N,4)
            metric:  Reference metric (B,4,4)
            mask:  Valid particle mask (B,N)
            channels:  Optional complex fields (B,N,4,C)
            reference_pairs:  Optional fixed laboratory plane contractions (B,N,N,1)
        Returns:
            updated: Scalar channels, twistor coordinates and representation fields
        """
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = projective_normalize(z, metric)
            features, overlap = pairwise_features(z, metric, mask, normalized=True)
            if self.three_point:
                features = torch.cat([features, three_point_features(overlap, mask, z, metric, normalized=True)], -1)
            if reference_pairs is not None:
                features = torch.cat((features, reference_pairs.to(features.dtype)), -1)
        features = features.float()
        if self.pair is not None:
            features = torch.nn.functional.gelu(self.pair(features, mask))
        logits = self.edge(features).permute(0, 3, 1, 2)
        b, n, d = h.shape
        normalized_h = self.norm(h)
        v = self.value(normalized_h).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        if self.qk is None:
            q = k = torch.zeros_like(v)
        else:
            q, k = [
                value.reshape(b, n, self.heads, d // self.heads).transpose(1, 2) for value in self.qk(normalized_h).chunk(2, -1)
            ]
        context = scaled_attention(q, k, v, mask, bias=logits, dropout=self.attention_dropout if self.training else 0.0, backend=self.attention_backend)
        context = context.transpose(1, 2).reshape_as(h)
        h = h + self.residual_scale * self.drop(context)
        weights = None
        if self.update or channels is not None:
            with torch.autocast(device_type=z.device.type, enabled=False):
                scores = logits.float()
                if self.qk is not None:
                    scores = scores + (q.float() @ k.float().transpose(-1, -2)) / (d // self.heads) ** 0.5
            weights = scores.masked_fill(~mask[:, None, None, :], -torch.inf).softmax(-1).to(z.real.dtype).mean(1)
        if channels is not None:
            gate = self.representation_scale * self.rep_gate(h).sigmoid().to(z.real.dtype)
            with torch.autocast(device_type=z.device.type, enabled=False):
                mixed = self.rep_mix(channels)
                transported = transport_channels(z, overlap, weights, mixed, mask, self.transport)
                channels = (channels + gate[..., None, :] * transported).masked_fill(~mask[..., None, None], 0)
                contraction = ((z.conj() @ metric).unsqueeze(-2) @ channels).squeeze(-2)
                invariants = torch.cat([contraction.real, contraction.imag], -1).float()
            h = h + self.residual_scale * self.rep_readout(invariants).masked_fill(~mask[..., None], 0)
        ff_norm = self.ff_norm if self.ff_norm is not None else self.norm
        h = (h + self.residual_scale * self.ff(ff_norm(h))).masked_fill(~mask[..., None], 0)
        if self.update:
            # Heads are combined as scalar weights before applying to all coordinates
            step = self.geometry_scale * self.step(h).squeeze(-1).tanh()
            with torch.autocast(device_type=z.device.type, enabled=False):
                z = geometry_update(z, metric, overlap, weights, step.to(z.real.dtype), mask, self.transport, normalized=True)
        return h, z, channels

class Twistformer(nn.Module):
    def __init__(self, input_spec, cfg):
        """
        Construct twistor channels and optional event or laboratory references

        Args:
            input_spec:  Named particle features and physical geometry
            cfg:  Serializable model steering
        """
        super().__init__()
        self.kind = cfg.twistor_kind
        self.precision = cfg.twistor_precision
        self.reference, self.beam_direction = cfg.reference_token, cfg.beam_direction
        self.reference_count = {"frame": 6, "beam_plane": 2, "none": 0}.get(self.reference, 1)
        self.laboratory_reference = self.reference in {"time", "beam_plane"}
        self.frame_embedding = nn.Embedding(6, cfg.width) if self.reference == "frame" else None
        if self.frame_embedding is not None:
            nn.init.normal_(self.frame_embedding.weight, std=0.02)
        self.spurion_embedding = nn.Embedding(self.reference_count, cfg.width) if self.laboratory_reference else None
        if self.spurion_embedding is not None:
            nn.init.normal_(self.spurion_embedding.weight, std=0.02)
        self.rep_seed = nn.Linear(cfg.width, cfg.representation_channels) if cfg.representation_channels else None
        self.embed = mlp(3 + int(self.laboratory_reference), cfg.width, cfg.width, cfg.dropout)
        self.auxiliary = AuxiliaryEmbedding(input_spec, cfg.auxiliary_features, cfg.width) if cfg.auxiliary_features else None
        self.blocks = nn.ModuleList(
            [TwistorBlock(replace(cfg, geometry_update=cfg.geometry_update and i < cfg.depth - 1)) for i in range(cfg.depth)]
        )
        self.norm = normalization(cfg.norm, cfg.width) if cfg.final_norm else nn.Identity()

    def prepare_onnx(self, module, inputs):
        """
        Lower complex twistor operations to real tensor operations

        Args:
            module:  Shared inference module with preprocessing and task outputs
            inputs:  Example tensor inputs
        Returns:
            graph: Real tensor graph for the standard ONNX exporter
        """
        from torch._subclasses.complex_tensor import ComplexTensor
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_map, tree_leaves

        for block in self.blocks:
            if block.pair is not None:
                block.pair.export_diagonal = True

        class RealComplexMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                """
                Keep PyTorch complex decompositions in separate real channels

                Args:
                    func:  Dispatched tensor operation
                    types:  Tensor subclass types
                    args:  Positional tensor arguments
                    kwargs:  Tensor factory and dtype arguments
                Returns:
                    output: Real tensors or PyTorch ComplexTensor wrappers
                """
                kwargs = dict(kwargs or {})
                if func == torch.ops.aten.complex.default:
                    return ComplexTensor(*args)
                if func == torch.ops.aten.view_as_complex.default:
                    return ComplexTensor(args[0][..., 0], args[0][..., 1])
                dtype = kwargs.get("dtype")
                if func == torch.ops.aten.ones_like.default and args[0].is_complex():
                    if dtype is not None and not dtype.is_complex:
                        return func(args[0].real, **kwargs)
                    if dtype is not None:
                        kwargs["dtype"] = torch.float32 if dtype == torch.complex64 else torch.float64
                    real = func(args[0].real, **kwargs)
                    return ComplexTensor(real, torch.zeros_like(real))
                if dtype in (torch.complex64, torch.complex128) and not any(
                    isinstance(x, torch.Tensor) and x.is_complex() for x in tree_leaves(args)
                ):
                    kwargs["dtype"] = torch.float32 if dtype == torch.complex64 else torch.float64
                    if func == torch.ops.aten.scalar_tensor.default and isinstance(args[0], complex):
                        return ComplexTensor(func(args[0].real, **kwargs), func(args[0].imag, **kwargs))
                    real = func(*args, **kwargs)
                    return ComplexTensor(real, torch.zeros_like(real))
                scalar_dtype = next(
                    (
                        x.real.dtype
                        for x in tree_leaves(args)
                        if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex())
                    ),
                    torch.float64,
                )

                def convert(value):
                    """
                    Wrap complex coefficients without changing their precision

                    Args:
                        value:  Tensor or scalar argument
                    Returns:
                        value: Argument with complex storage split by PyTorch
                    """
                    if isinstance(value, complex):
                        return ComplexTensor(
                            torch.tensor(value.real, dtype=scalar_dtype), torch.tensor(value.imag, dtype=scalar_dtype)
                        )
                    if isinstance(value, torch.Tensor) and value.is_complex() and not isinstance(value, ComplexTensor):
                        return ComplexTensor.from_interleaved(value)
                    return value

                return func(*tree_map(convert, args), **kwargs)

        with RealComplexMode():
            graph = make_fx(
                module,
                decomposition_table={
                    torch.ops.aten.hypot.default: _onnx_hypot,
                    torch.ops.prims.hypot.default: _onnx_hypot,
                    torch.ops.aten.asinh.default: _onnx_asinh,
                },
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
            )(*inputs)
        for node in graph.graph.nodes:
            if node.target == torch.ops.aten._neg_view.default:
                node.target = torch.ops.aten.neg.default
        graph.graph.eliminate_dead_code()
        graph.recompile()
        return graph.eval()

    def forward(self, batch, global_context=None):
        """
        Encode physical events into scalar features

        Args:
            batch:  Validated, padded particle Batch
            global_context:  Optional embedded invariant event scalars (B,D)
        Returns:
            encoded: EncoderOutput with token features or graph readouts
        """
        with torch.autocast(device_type=batch.features.device.type, enabled=False):
            # Keep cancellation-sensitive input kinematics in float64 at O(BN) cost
            p = batch.p4.double()
            tw, metric = twistors_from_p4(p, batch.mask, self.kind)
            features = invariant_features(p, batch.mask).float()
        z, mask = tw.Z, batch.mask
        reference_pairs = None
        if self.reference != "none":
            with torch.autocast(device_type=p.device.type, enabled=False):
                z, mask = append_reference(p, z, metric, mask, self.reference, self.beam_direction)
            features = torch.nn.functional.pad(features, (0, 0, 0, self.reference_count))
        if self.laboratory_reference:
            with torch.autocast(device_type=p.device.type, enabled=False):
                scalars, reference_pairs = laboratory_reference_features(p, batch.mask, self.reference, self.beam_direction)
            features = torch.cat((features, scalars.float()), -1)
        if self.precision == "float32":
            with torch.autocast(device_type=z.device.type, enabled=False):
                z, metric = rest_frame_coordinates(z, metric)
        h = self.embed(features)
        if self.frame_embedding is not None:
            # Scalar labels distinguish +x, -x, +y, -y, +z and -z references
            references = h[:, -self.reference_count :] + self.frame_embedding.weight.to(h.dtype)
            h = torch.cat((h[:, : -self.reference_count], references), dim=1)
        if self.spurion_embedding is not None:
            labels = self.spurion_embedding.weight.to(h.dtype)
            references = h[:, -self.reference_count :] + labels
            h = torch.cat((h[:, : -self.reference_count], references), dim=1)
        if self.auxiliary is not None:
            auxiliary = self.auxiliary(batch.features, batch.mask)
            if self.reference != "none":
                auxiliary = torch.nn.functional.pad(auxiliary, (0, 0, 0, self.reference_count))
            h = h + auxiliary
        if global_context is not None:
            h = h + global_context[:, None]
        h = h.masked_fill(~mask[..., None], 0)
        channels = None
        if self.rep_seed is not None:
            coefficients = self.rep_seed(h).to(z.real.dtype)
            with torch.autocast(device_type=z.device.type, enabled=False):
                channels = projective_normalize(z, metric)[..., None] * coefficients[..., None, :]
        for block in self.blocks:
            h, z, channels = block(h, z, metric, mask, channels, reference_pairs)
        h = h[:, : batch.mask.shape[1]]
        return EncoderOutput(self.norm(h).masked_fill(~batch.mask[..., None], 0))

class DetectorTwistformer(nn.Module):
    def __init__(self, input_spec, cfg):
        """
        Combine detector-frame attention with invariant twistor features

        Args:
            input_spec:  Learned detector features and separate p4 geometry
            cfg:  Shared width and depth for each encoder branch
        """
        super().__init__()
        self.detector = Transformer(input_spec, replace(cfg, family="transformer", reference_token="none"))
        self.twistor = Twistformer(input_spec, cfg)
        self.fuse = mlp(2 * cfg.width, cfg.width, cfg.width, cfg.dropout)
        self.norm = normalization(cfg.norm, cfg.width) if cfg.final_norm else nn.Identity()

    def prepare_onnx(self, module, inputs):
        """
        Prepare both detector and twistor branches for export

        Args:
            module:  Shared inference module with preprocessing and task outputs
            inputs:  Example tensor inputs
        Returns:
            graph: Real tensor graph for both encoder branches
        """
        self.detector.prepare_onnx(module, inputs)
        return self.twistor.prepare_onnx(module, inputs)

    def forward(self, batch, global_context=None):
        """
        Fuse aligned detector and twistor tokens before task pooling

        Args:
            batch:  Standardized features with unstandardized energy-flow p4
            global_context:  Optional embedded event context (B,D)
        Returns:
            encoded: Detector-frame token features (B,N,D)
        """
        detector = self.detector(batch, global_context).features
        twistor = self.twistor(batch, global_context).features
        fused = detector + self.fuse(torch.cat((detector, twistor), dim=-1))
        return EncoderOutput(self.norm(fused).masked_fill(~batch.mask[..., None], 0))

def _onnx_asinh(x):
    """
    Express signed plane compression with real ONNX operations

    Args:
        x:  Real signed pair contractions
    Returns:
        compressed: Stable inverse hyperbolic sine at zero and large magnitudes
    """
    magnitude = x.abs()
    small = magnitude.clamp(max=1)
    low = torch.log1p(small + small.square() / (torch.sqrt(1 + small.square()) + 1))
    large = magnitude.clamp(min=1)
    high = large.log() + torch.log1p(torch.sqrt(1 + large.reciprocal().square()))
    compressed = torch.where(magnitude <= 1, low, high)
    return torch.where(x < 0, -compressed, compressed)

def _onnx_hypot(x, y):
    """
    Express the complex modulus with real ONNX operations

    Args:
        x:  Real component
        y:  Imaginary component
    Returns:
        modulus: Stable nonnegative complex modulus
    """
    scale = torch.maximum(x.abs(), y.abs())
    denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale * ((x / denominator).square() + (y / denominator).square()).sqrt()
