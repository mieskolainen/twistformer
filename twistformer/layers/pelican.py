# Permutation-equivariant PELICAN aggregations
# PELICAN: https://arxiv.org/abs/2211.00454
# PELICAN architecture: https://arxiv.org/abs/2307.16506
#
# mikael.mieskolainen@cern.ch (2026)

"""Complete 2->2, 1->2 and 2->0 maps with masked multiplicity scaling"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from twistformer.layers.common import activation_layer, normalization

def pelican_activation(kind):
    """
    Construct the scalar activation for PELICAN messages

    Args:
        kind:  Activation name
    Returns:
        activation: Elementwise scalar module
    """
    return nn.LeakyReLU(0.01) if kind == "leaky_relu" else activation_layer(kind)

class PowerEncoding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.alpha = nn.Parameter(torch.linspace(0.05, 0.5, width))

    def forward(self, x):
        """
        Apply the learned signed power encoding with a stable logarithmic limit

        Args:
            x:  Projected Minkowski contractions (...,C)
        Returns:
            encoded: sign(x) times ((1+abs(x))**a-1)/a with a=alpha**2+1e-6
        """
        x = x.float()
        a = self.alpha.float().square() + 1e-6
        encoded = x.sign() * torch.expm1(a * torch.log1p(x.abs())) / a
        # Null contractions have unit slope in the signed power limit
        return torch.where(x == 0, x, encoded)

def pair_statistics(x, mask):
    """
    Reduce pair channels with valid constituent counts and FP32 accumulation

    Args:
        x:  Pair channels (B,N,N,C)
        mask:  Valid particles (B,N)
    Returns:
        statistics: Masked pairs, diagonal, row mean, column mean, total mean, trace mean
    """
    valid = mask[:, :, None] & mask[:, None, :]
    values = x.masked_fill(~valid[..., None], 0)
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    count = mask.sum(1).clamp_min(1).to(dtype)
    diag = values.diagonal(dim1=1, dim2=2).transpose(1, 2).to(dtype)
    row = values.sum(2, dtype=dtype) / count[:, None, None]
    col = values.sum(1, dtype=dtype) / count[:, None, None]
    total = row.sum(1) / count[:, None]
    trace = diag.sum(1) / count[:, None]
    return values, diag, row, col, total, trace

class PairAggregation(nn.Module):
    def __init__(self, in_dim, out_dim, average_nobj=49.0, factorize=True, multiplicity=True):
        super().__init__()
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.export_diagonal = False
        if factorize:
            self.input_scale = nn.Parameter(torch.randn(15, in_dim) / math.sqrt(15))
            self.output_scale = nn.Parameter(torch.randn(15, out_dim) / math.sqrt(15))
            self.mix_input = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(in_dim))
            self.mix_output = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(in_dim))
        else:
            self.weight = nn.Parameter(torch.randn(15, in_dim, out_dim) * math.sqrt(2 / (15 * in_dim)))
        self.alpha = nn.Parameter(torch.rand(10, in_dim)) if multiplicity else None
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.diag_bias = nn.Parameter(torch.zeros(out_dim))

    def coefficients(self):
        """
        Form channel coefficients without expanding particle axes

        Returns:
            weights: Independent or factorized map coefficients (15,C,D)
        """
        if not self.factorize:
            return self.weight
        return self.input_scale[..., None] * self.mix_input + self.output_scale[:, None, :] * self.mix_output

    def forward(self, x, mask):
        """
        Apply all 15 rank-two maps including independent diagonal and global biases

        Args:
            x:  Pair channels (B,N,N,C)
            mask:  Valid particles (B,N)
        Returns:
            pairs: Equivariant pair channels (B,N,N,D)
        """
        values, diag, row, col, total, trace = pair_statistics(x, mask)
        weights = self.coefficients()
        dtype = diag.dtype
        scale = None
        if self.alpha is not None:
            log_count = (mask.sum(1).clamp_min(1).to(dtype) / self.average_nobj).log()
            scale = (log_count[:, None, None] * self.alpha).exp()

        # Project both pair maps together before transposing particle indices
        pair_weights = weights[:2].transpose(1, 2).flatten(0, 1)
        if not torch.is_autocast_enabled(x.device.type):
            values = values.to(weights.dtype)
        direct, reverse = F.linear(values, pair_weights).chunk(2, -1)
        out = direct.to(dtype) + reverse.transpose(1, 2).to(dtype)

        # Combine the remaining maps on particle or event axes before expansion
        diag_weights = weights[2:5].transpose(1, 2).flatten(0, 1)
        projected = list(F.linear(diag, diag_weights).to(dtype).chunk(3, -1))
        particles = torch.stack((col, col, col, row, row, row), dim=0)
        events = torch.stack((trace, trace, total, total), dim=0)
        if scale is not None:
            particles = particles * scale[:, :6].transpose(0, 1)[:, :, None]
            events = events * scale[:, 6:].transpose(0, 1)
        particle_output = torch.bmm(particles.flatten(1, 2), weights[5:11]).reshape(6, *row.shape[:2], -1)
        event_output = torch.bmm(events, weights[11:15])
        projected.extend(particle_output.to(dtype).unbind(0))
        projected.extend(event_output.to(dtype).unbind(0))
        diagonal = projected[0] + projected[3] + projected[6]
        columns = projected[1] + projected[4] + projected[7]
        rows = projected[2] + projected[5] + projected[8]
        global_map = projected[9] + projected[11] + self.bias
        global_diagonal = projected[10] + projected[12] + self.diag_bias
        rows = rows + global_map[:, None]
        diagonal = diagonal + global_diagonal[:, None]
        out.add_(rows[:, :, None])
        out.add_(columns[:, None])
        # Add diagonal maps on N entries instead of multiplying N x N tensors
        if self.export_diagonal:
            out = out + torch.diag_embed(diagonal.transpose(1, 2)).movedim(1, -1)
        else:
            out.diagonal(dim1=1, dim2=2).add_(diagonal.transpose(1, 2))
        valid = mask[:, :, None] & mask[:, None, :]
        return out.masked_fill_(~valid[..., None], 0).to(x.dtype)

class TriangleInteraction(nn.Module):
    def __init__(self, width, channels, norm="rms", activation="leaky_relu", scale=0.1):
        super().__init__()
        self.norm = normalization(norm, width)
        self.left = nn.Linear(width, channels)
        self.right = nn.Linear(width, channels)
        self.activation = pelican_activation(activation)
        self.out = nn.Linear(channels, width, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x, mask):
        """
        Add joint i-k-j interactions using channelwise matrix products

        Args:
            x:  Lorentz-scalar pair channels (B,N,N,D)
            mask:  Valid particles (B,N)
        Returns:
            pairs: Residual pair features with the masked mean over k
        """
        valid = mask[:, :, None] & mask[:, None, :]
        x = x.masked_fill(~valid[..., None], 0)
        normalized = self.norm(x)
        weights = torch.cat((self.left.weight, self.right.weight))
        biases = torch.cat((self.left.bias, self.right.bias))
        branches = self.activation(F.linear(normalized, weights, biases))
        branches = branches.masked_fill(~valid[..., None], 0)
        left, right = branches.chunk(2, -1)
        # Contract k without an N x N x N intermediate and retain FP32 reductions
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            left = left.permute(0, 3, 1, 2).contiguous().to(dtype)
            right = right.permute(0, 3, 1, 2).contiguous().to(dtype)
            count = mask.sum(1).clamp_min(1).to(dtype)
            product = (left @ right) / count[:, None, None, None]
            product = product.permute(0, 2, 3, 1)
        update = self.out(product)
        result = x.to(dtype) + self.scale * update.to(dtype)
        return result.masked_fill(~valid[..., None], 0).to(x.dtype)

class ScalarLift(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.export_diagonal = False
        self.maps = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(5)])

    def forward(self, x, mask):
        """
        Lift invariant constituent scalars through the full five-map 1->2 basis

        Args:
            x:  Constituent scalars (B,N,C)
            mask:  Valid particles (B,N)
        Returns:
            pairs: Lifted scalar pair channels (B,N,N,C)
        """
        x = x.float().masked_fill(~mask[..., None], 0)
        mean = x.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        weights = torch.cat([layer.weight for layer in self.maps[:3]])
        diagonal, receiver, sender = F.linear(x, weights).to(x.dtype).chunk(3, -1)
        weights = torch.cat([layer.weight for layer in self.maps[3:]])
        trace, total = F.linear(mean, weights).to(x.dtype).chunk(2, -1)
        out = receiver[:, :, None] + sender[:, None] + total[:, None, None]
        diagonal = diagonal + trace[:, None]
        if self.export_diagonal:
            out = out + torch.diag_embed(diagonal.transpose(1, 2)).movedim(1, -1)
        else:
            out.diagonal(dim1=1, dim2=2).add_(diagonal.transpose(1, 2))
        valid = mask[:, :, None] & mask[:, None, :]
        return out.masked_fill(~valid[..., None], 0)

class PairMessage(nn.Module):
    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = pelican_activation(cfg.pelican_activation)
        self.norm = normalization(cfg.norm, out_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, valid):
        """
        Mix pair channels without coupling particle indices

        Args:
            x:  Pair channels (B,N,N,C)
            valid:  Valid particle pairs (B,N,N)
        Returns:
            messages: Masked nonlinear pair channels (B,N,N,D)
        """
        x = self.dropout(self.norm(self.activation(self.linear(x))))
        return x.masked_fill(~valid[..., None], 0)

class InvariantReadout(nn.Module):
    def __init__(self, in_dim, out_dim, average_nobj=49.0, multiplicity=True):
        super().__init__()
        self.average_nobj = average_nobj
        self.alpha = nn.Parameter(torch.rand(2, in_dim)) if multiplicity else None
        self.linear = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x, mask):
        """
        Combine the two independent rank-two invariant reductions

        Args:
            x:  Pair channels (B,N,N,C)
            mask:  Valid particles (B,N)
        Returns:
            features: Invariant event channels (B,D)
        """
        valid = mask[:, :, None] & mask[:, None, :]
        values = x.masked_fill(~valid[..., None], 0)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        count = mask.sum(1).clamp_min(1).to(dtype)
        total = values.sum((1, 2), dtype=dtype) / count[:, None].square()
        trace = values.diagonal(dim1=1, dim2=2).sum(-1, dtype=dtype) / count[:, None]
        reduced = torch.stack([total, trace], 1)
        if self.alpha is not None:
            log_count = (mask.sum(1).clamp_min(1).to(reduced.dtype) / self.average_nobj).log()
            reduced = reduced * (log_count[:, None, None] * self.alpha).exp()
        return self.linear(reduced.flatten(1))

class ParticleReadout(nn.Module):
    def __init__(self, in_dim, out_dim, average_nobj=49.0, multiplicity=True):
        super().__init__()
        self.average_nobj = average_nobj
        self.alpha = nn.Parameter(torch.rand(5, in_dim)) if multiplicity else None
        self.weight = nn.Parameter(torch.randn(5, out_dim, in_dim) * math.sqrt(2 / (5 * in_dim)))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, mask):
        """
        Apply all five 2->1 equivariant maps to produce particle scalars

        Args:
            x:  Pair channels (B,N,N,C)
            mask:  Valid particles (B,N)
        Returns:
            features: Masked particle channels (B,N,D)
        """
        _, diag, row, col, total, trace = pair_statistics(x, mask)
        scale = None
        if self.alpha is not None:
            log_count = (mask.sum(1).clamp_min(1).to(diag.dtype) / self.average_nobj).log()
            scale = (log_count[:, None, None] * self.alpha).exp()
        out = self.bias
        for index, values in enumerate((diag, row, col, trace, total)):
            if scale is not None:
                multiplier = scale[:, index]
                values = values * (multiplier[:, None] if values.ndim == 3 else multiplier)
            projected = F.linear(values, self.weight[index]).to(diag.dtype)
            out = out + (projected[:, None] if values.ndim == 2 else projected)
        return out.masked_fill(~mask[..., None], 0).to(x.dtype)
