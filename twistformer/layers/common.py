# Shared attention and representation layers
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from torch import nn
from torch.nn import functional as F

class BatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        """
        Normalize compact observations with running statistics for singletons

        Args:
            x:  Valid observations (N,C)
        Returns:
            features: Batch-normalized observations
        """
        if self.training and len(x) < 2:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias, training=False, momentum=0.0, eps=self.eps
            )
        return super().forward(x)

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        if not self.normalized_shape or any(d < 1 for d in self.normalized_shape):
            raise ValueError("RMSNorm requires positive normalized dimensions")
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        """
        Normalize RMS while retaining the activation dtype under autocast

        Args:
            x:  Feature tensor with channels on the last axis
        Returns:
            normalized: RMS-normalized features
        """
        axes = tuple(range(-len(self.normalized_shape), 0))
        if tuple(x.shape[-len(self.normalized_shape) :]) != self.normalized_shape:
            raise ValueError(f"RMSNorm expected trailing dimensions {self.normalized_shape}")
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        normalized = (values * torch.rsqrt(values.square().mean(axes, keepdim=True) + eps)).to(x.dtype)
        if self.weight is not None:
            normalized = normalized * self.weight.to(x.dtype)
        return normalized

def normalization(kind, width):
    """
    Construct channel normalization with fixed numerical tolerances

    Args:
        kind:  none, layer, rms or batch
        width:  Channel count
    Returns:
        norm: Normalization module
    """
    if kind == "none":
        return nn.Identity()
    if kind == "layer":
        return nn.LayerNorm(width, eps=1e-5)
    if kind == "rms":
        return RMSNorm(width, eps=1e-6)
    if kind == "batch":
        return BatchNorm(width)
    raise ValueError(f"Unknown normalization: {kind}")

class Sine(nn.Module):
    def __init__(self, adaptive=False):
        super().__init__()
        self.frequency = nn.Parameter(torch.ones(())) if adaptive else 1.0
        self.phase = nn.Parameter(torch.zeros(())) if adaptive else 0.0

    def forward(self, x):
        """
        Apply a fixed or learned sinusoidal activation

        Args:
            x:  Real scalar features
        Returns:
            features: Elementwise sinusoidal features
        """
        return torch.sin(self.frequency * x + self.phase)

def activation_layer(kind):
    """
    Construct a scalar activation without mixing token or event axes

    Args:
        kind:  Scalar activation name
    Returns:
        activation: Elementwise activation module
    """
    factories = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "sine": Sine,
        "sine_adaptive": lambda: Sine(True),
    }
    if kind not in factories:
        raise ValueError(f"Unknown activation: {kind}")
    return factories[kind]()

class GatedActivation(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        """
        Multiply value channels by sigmoid, GELU or SiLU gate channels

        Args:
            x:  Concatenated value and gate projections
        Returns:
            features: Gated value channels
        """
        value, gate = x.chunk(2, -1)
        gate = {"glu": torch.sigmoid, "geglu": F.gelu, "swiglu": F.silu}[self.kind](gate)
        return value * gate

class HiddenBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, residual, norm="none", activation="gelu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = normalization(norm, out_dim)
        self.activation = activation_layer(activation)
        self.dropout = nn.Dropout(dropout)
        self.skip = (nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)) if residual else None

    def forward(self, x):
        """
        Apply a normalized hidden layer and optional projected residual

        Args:
            x:  Hidden scalar features
        Returns:
            features: Updated hidden features
        """
        y = self.dropout(self.activation(self.norm(self.linear(x))))
        return y if self.skip is None else self.skip(x) + y

class ResidualMLP(nn.Sequential):
    """Shared MLP with arbitrary hidden widths and optional projected shortcuts"""

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden=(),
        *,
        dropout=0.0,
        norm="none",
        activation="gelu",
        residual=False,
        zero_init_last=False,
        init="default",
    ):
        widths = list(hidden)
        if min([in_dim, out_dim, *widths]) < 1 or not 0 <= dropout < 1:
            raise ValueError("MLP dimensions must be positive and dropout in [0,1)")
        if init not in {"default", "xavier_normal"}:
            raise ValueError("Unknown MLP initialization")
        layers = []
        gated = activation in {"glu", "geglu", "swiglu"}
        if gated and not widths:
            raise ValueError("Gating requires a hidden layer")
        if widths:
            layers.append(nn.Linear(in_dim, widths[0] * (2 if gated else 1)))
            if gated:
                layers.append(GatedActivation(activation))
            if norm != "none":
                layers.append(normalization(norm, widths[0]))
            if not gated:
                layers.append(activation_layer(activation))
            layers.append(nn.Dropout(dropout))
            for first, second in zip(widths, widths[1:]):
                layers.append(HiddenBlock(first, second, dropout, residual, norm, "gelu" if gated else activation))
        layers.append(nn.Linear(widths[-1] if widths else in_dim, out_dim))
        super().__init__(*layers)
        self.batch_norm = norm == "batch"
        if init == "xavier_normal":
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        if zero_init_last:
            nn.init.zeros_(self[-1].weight)
            nn.init.zeros_(self[-1].bias)

    def forward(self, x, mask=None):
        """
        Apply the MLP, excluding padded tokens from BatchNorm statistics

        Args:
            x:  Features with channels on the last axis
            mask:  Valid observations on leading axes, required for padded BatchNorm
        Returns:
            features: MLP outputs with masked observations zeroed
        """
        if self.batch_norm and x.ndim > 2 and mask is None:
            raise ValueError("BatchNorm on token tensors requires a validity mask")
        if mask is not None:
            if mask.shape != x.shape[:-1] or mask.dtype != torch.bool:
                raise ValueError("MLP mask must match the observation axes")
            compact = super().forward(x[mask])
            output = compact.new_zeros((*x.shape[:-1], self[-1].out_features))
            output[mask] = compact
            return output
        return super().forward(x)

def mlp(in_dim, hidden, out_dim, dropout=0.0, *, norm="none", activation="gelu"):
    """
    Construct a single-hidden-layer instance of the shared ResidualMLP

    Args:
        in_dim:  Input width
        hidden:  Hidden width before gating, if enabled
        out_dim:  Output width
        dropout:  Hidden dropout probability
        norm:  Hidden normalization
        activation:  Scalar activation or gating choice
    Returns:
        network: Shared MLP implementation
    """
    return ResidualMLP(in_dim, out_dim, [hidden], dropout=dropout, norm=norm, activation=activation)

class AuxiliaryEmbedding(nn.Module):
    """Embed explicitly selected particle scalars using the driver schema"""

    def __init__(self, input_spec, names, width):
        super().__init__()
        unknown = set(names) - set(input_spec.features)
        if unknown:
            raise ValueError(f"Unknown auxiliary_features: {sorted(unknown)}")
        self.indices = tuple(input_spec.features.index(name) for name in names)
        self.embed = mlp(len(names), width, width)

    def forward(self, features, mask):
        """
        Embed scalar features with padding excluded

        Args:
            features:  Preprocessed particle features (B,N,F)
            mask:  Valid particle mask (B,N)
        Returns:
            embedded: Masked auxiliary channels (B,N,D)
        """
        selected = features[..., self.indices].masked_fill(~mask[..., None], 0)
        return self.embed(selected).masked_fill(~mask[..., None], 0)

class Pool(nn.Module):
    def __init__(self, width, mode, heads=1, depth=1, dropout=0.0, queries=1):
        super().__init__()
        self.validate_weights = True
        self.mode = mode
        if mode == "pma":
            self.seed = nn.Parameter(torch.randn(1, queries, width))
            self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(width)
        elif mode == "cls":
            self.seed = nn.Parameter(torch.randn(1, 1, width))
            layer = nn.TransformerEncoderLayer(
                width, heads, 2 * width, dropout, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.score = nn.Linear(width, 1) if mode == "attention" else None
        if mode == "attention_query":
            self.query = nn.Parameter(torch.randn(width))
            self.key = nn.Linear(width, width, bias=False)

    def forward(self, x, batch):
        """
        Reduce masked token features to event features

        Args:
            x:  Token features (B,N,D) or graph readouts (B,D)
            batch:  Validity mask and optional nonnegative token weights
        Returns:
            events: Pooled features (B,D)
        """
        if x.ndim == 2:
            return x
        mask = batch.mask
        x = x.masked_fill(~mask[..., None], 0)
        if self.mode == "pma":
            query = self.seed.expand(x.shape[0], -1, -1)
            pooled, _ = self.attention(query, x, x, key_padding_mask=~mask, need_weights=False)
            return self.norm(pooled).mean(1)
        if self.mode == "cls":
            x = torch.cat([self.seed.expand(x.shape[0], -1, -1), x], 1)
            mask = F.pad(mask, (1, 0), value=True)
            return self.encoder(x, src_key_padding_mask=~mask)[:, 0]
        if self.mode == "sum":
            return x.sum(1)
        if self.mode == "attention_query":
            scores = (self.key(x) * self.query).sum(-1) / x.shape[-1] ** 0.5
            weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        elif self.mode == "attention":
            weights = self.score(x).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
        elif self.mode == "weighted_mean":
            if batch.token_weights is None:
                raise ValueError("weighted_mean pooling requires token_weights")
            weights = batch.token_weights.masked_fill(~mask, 0)
            if self.validate_weights:
                positive = (weights.sum(1) > 0).all()
                if torch.compiler.is_compiling():
                    torch._assert_async(positive, "Each event needs positive total token weight")
                elif not positive:
                    raise ValueError("Each event needs positive total token weight")
            weights = weights / weights.sum(1, keepdim=True)
        else:
            weights = mask.to(x.dtype) / mask.sum(1, keepdim=True)
        return (x * weights[..., None]).sum(1)
