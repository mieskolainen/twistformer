# Shared attention and representation layers
#
# mikael.mieskolainen@cern.ch (2026)

"""Standard, hyperbolic and triangle attention share mask and bias semantics"""

import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from .common import mlp

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None

try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None

@torch.compiler.assume_constant_result
def fused_attention_available(device):
    """
    Select the CUDA attention backend when the graph is traced

    Args:
        device:  Query device
    Returns:
        available: Flash kernels enabled on supported CUDA hardware
    """
    # Backend policy stays fixed for the lifetime of a compiled graph
    return (
        torch.cuda.get_device_capability(device)[0] >= 8
        and torch.backends.cuda.is_flash_attention_available()
        and torch.backends.cuda.flash_sdp_enabled()
    )

def masked_flex_attention(q, k, v, mask, bias, scale):
    """
    Capture masked attention and differentiable geometry scores in one graph

    Args:
        q:  Padded queries
        k:  Padded keys
        v:  Padded values
        mask:  Valid particle mask
        bias:  Optional pair scores
        scale:  Attention normalization
    Returns:
        context: Fused attention output
    """

    def score_mod(score, batch, head, query, key):
        if bias is not None:
            score = score + bias[batch, head, query, key]
        return torch.where(mask[batch, key], score, -torch.inf)

    options = None
    dtype = q.dtype
    if q.is_cuda and torch.version.hip is None and torch.cuda.get_device_capability(q.device)[0] < 8:
        # Small tiles fit Turing shared memory in forward and backward passes
        options = {"BLOCK_M": 16, "BLOCK_N": 16, "bwd_BLOCK_M1": 16, "bwd_BLOCK_N1": 16,
                   "bwd_BLOCK_M2": 16, "bwd_BLOCK_N2": 16, "num_stages": 1}
        if dtype == torch.bfloat16:
            q, k, v = q.float(), k.float(), v.float()
    return flex_attention(q, k, v, score_mod=score_mod, scale=scale, enable_gqa=q.shape[1] != k.shape[1],
                          kernel_options=options).to(dtype)

@lru_cache(maxsize=1)
def fused_attention_kernel():
    """
    Compile a standalone attention kernel for eager model execution

    Returns:
        kernel: Compiled masked attention
    """
    return torch.compile(masked_flex_attention, dynamic=True)

def scaled_attention(q, k, v, mask, bias=None, dropout=0.0, scale=None, backend="sdpa"):
    """
    Dispatch dense or packed variable-length scaled dot-product attention

    Args:
        q:  Queries (B,H,N,D)
        k:  Keys (B,Hkv,N,D)
        v:  Values (B,Hkv,N,D)
        mask:  Valid token mask (B,N)
        bias:  Optional additive pair bias
        dropout:  Attention dropout probability, zero during evaluation
        scale:  Score scale before additive bias
        backend:  sdpa, flex or auto dispatch
    Returns:
        context: Attention output with zero padded queries
    """
    if backend not in {"sdpa", "flex", "auto"}:
        raise ValueError("attention_backend must be sdpa, flex or auto")
    if backend == "flex":
        if flex_attention is None or not q.is_cuda:
            raise ValueError("FlexAttention requires CUDA and torch.nn.attention.flex_attention")
        if dropout:
            raise ValueError("FlexAttention requires attention_dropout=0")
        if bias is not None and torch.is_grad_enabled() and bias.requires_grad:
            raise ValueError("FlexAttention does not support learned pair bias in this dispatcher")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("FlexAttention requires float16, bfloat16 or float32 queries")
        if not (q.shape[-1] == k.shape[-1] == v.shape[-1] <= 256):
            raise ValueError("FlexAttention requires equal QKV head widths of at most 256")
    fused = (
        backend == "auto"
        and q.is_cuda
        and torch.version.hip is None
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] <= 256
        and q.shape[-1] == k.shape[-1] == v.shape[-1]
        and dropout == 0
        and fused_attention_available(q.device)
    )
    if backend == "flex" or (
        fused
        and flex_attention is not None
        # Plain compiled attention uses the native SDPA backward
        and bias is not None
        # Dynamic FlexAttention backward cannot reliably lower learned pair scores
        and not (bias is not None and torch.is_grad_enabled() and bias.requires_grad)
    ):
        dim = q.shape[-1]
        # FlexAttention requires at least 16 channels per head
        padding = next(size for size in (16, 32, 64, 128, 256) if size >= dim) - dim
        q, k, v = [F.pad(t, (0, padding)) for t in (q, k, v)]
        attend = masked_flex_attention if torch.compiler.is_compiling() else fused_attention_kernel()
        context = attend(
            q,
            k,
            v,
            mask,
            (bias.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2]) if bias is not None else None),
            scale if scale is not None else dim**-0.5,
        )[..., :dim]
        return context.masked_fill(~mask[:, None, :, None], 0)
    # Boolean packing has data-dependent shapes unsupported by compiled backward
    if fused and bias is None and varlen_attn is not None and not torch.compiler.is_compiling():
        dim = q.shape[-1]
        padding = (-dim) % 8
        packed = [F.pad(t.transpose(1, 2)[mask], (0, padding)) for t in (q, k, v)]
        lengths = mask.sum(1, dtype=torch.int32)
        offsets = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        y = varlen_attn(
            *packed,
            offsets,
            offsets,
            mask.shape[1],
            mask.shape[1],
            scale=scale if scale is not None else dim**-0.5,
            enable_gqa=q.shape[1] != k.shape[1],
        )[..., :dim]
        context = v.new_zeros((q.shape[0], q.shape[2], q.shape[1], v.shape[-1]))
        context[mask] = y
        return context.transpose(1, 2)
    keep = mask[:, None, None, :]
    attn_mask = keep if bias is None else bias.to(q.dtype).masked_fill(~keep, -torch.inf)
    if q.is_cuda and bias is None and not torch.compiler.is_compiling() and bool(mask.all()):
        attn_mask = None
    if k.shape[1] == v.shape[1] == 1 and q.shape[1] > 1 and attn_mask is not None:
        # Shared KV views retain masked SDPA kernels without grouped-head restrictions
        k = k.expand(-1, q.shape[1], -1, -1)
        v = v.expand(-1, q.shape[1], -1, -1)
    # SDPA takes a Python bool when event and particle counts are symbolic
    grouped_heads = False if q.shape[1] == k.shape[1] else True
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout, scale=scale, enable_gqa=grouped_heads
    ).masked_fill(~mask[:, None, :, None], 0)

class Attention(nn.Module):
    def __init__(self, width, heads, dropout=0.0, kind="standard", backend="sdpa"):
        super().__init__()
        self.heads, self.dim, self.kind, self.dropout = (heads, width // heads, kind, dropout)
        self.backend = backend
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def split(self, x):
        """
        Split token channels into attention heads

        Args:
            x:  Token features (B,N,D)
        Returns:
            heads: Tensor (B,H,N,D/H)
        """
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.dim).transpose(1, 2)

    def forward(self, x, mask, bias=None):
        """
        Aggregate valid tokens with standard, hyperbolic or triangle scores

        Args:
            x:  Learned token features (B,N,D)
            mask:  Valid particle mask (B,N)
            bias:  Optional finite pair bias (B,H,N,N)
        Returns:
            features: Masked token features (B,N,D)
        """
        q, k, v = [self.split(t) for t in self.qkv(x).chunk(3, -1)]
        if self.kind == "hyperbolic":
            # Accumulate squared lengths in float32 under AMP
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            q0 = (1 + q.to(dtype).square().sum(-1, keepdim=True)).sqrt().to(q.dtype)
            k0 = (1 + k.to(dtype).square().sum(-1, keepdim=True)).sqrt().to(k.dtype)
            q, k = torch.cat([-q0, q], -1), torch.cat([k0, k], -1)
            v = F.pad(v, (1, 0))
        y = scaled_attention(q, k, v, mask, bias, dropout=self.dropout if self.training else 0.0, backend=self.backend)
        if self.kind == "hyperbolic":
            y = y[..., 1:]
        y = y.transpose(1, 2).reshape_as(x)
        return self.out(y).masked_fill(~mask[..., None], 0)

def point_distance_features(points, query):
    """
    Lift affine points so query-key products give negative squared distance

    Args:
        points:  Cartesian point channels (...,C,3)
        query:  Select query or key factors
    Returns:
        features: Five factors per point (...,C,5)
    """
    square = points.square().sum(-1, keepdim=True)
    unit = torch.ones_like(square)
    if query:
        return torch.cat((square, unit, points), -1)
    return torch.cat((-unit, -square, 2 * points), -1)

class ProjectiveAttention(Attention):
    def __init__(self, width, heads, dropout=0.0, channels=2, reinsert=False, backend="sdpa"):
        """
        Add the affine point metric from improved projective GATr

        Args:
            width:  Scalar token width
            heads:  Attention head count
            dropout:  Attention dropout probability
            channels:  Learned point channels per head
            reinsert:  Include a fixed input point channel in every head
            backend:  sdpa, flex or auto dispatch
        """
        super().__init__(width, heads, dropout, backend=backend)
        self.channels, self.reinsert = channels, reinsert
        self.points = nn.Linear(width, 2 * heads * channels * 3)
        nn.init.normal_(self.points.weight, std=0.02)
        nn.init.zeros_(self.points.bias)
        self.distance_weight = nn.Parameter(torch.zeros(heads, channels + int(reinsert)))

    def forward(self, x, mask, bias=None, coordinates=None):
        """
        Combine scalar dot products with learned and reinserted point distances

        Args:
            x:  Scalar token features (B,N,D)
            mask:  Valid token mask (B,N)
            bias:  Optional pair scores (B,H,N,N)
            coordinates:  Common-scale input points (B,N,3)
        Returns:
            features: Masked scalar token updates (B,N,D)
        """
        if self.reinsert and coordinates is None:
            raise ValueError("Projective reinsertion requires input coordinates")
        x = x.masked_fill(~mask[..., None], 0)
        q, k, v = [self.split(t) for t in self.qkv(x).chunk(3, -1)]
        points = self.points(x)
        # Point norms use FP32 under AMP and retain FP64 for precision checks
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
            qp, kp = [t.to(dtype).reshape(*x.shape[:2], self.heads, self.channels, 3).transpose(1, 2) for t in points.chunk(2, -1)]
            if self.reinsert:
                raw = coordinates.to(dtype).masked_fill(~mask[..., None], 0)
                # A common origin reduces cancellation without changing distances
                center = raw.sum(1, keepdim=True) / mask.sum(1)[:, None, None].clamp_min(1)
                raw = (raw - center).masked_fill(~mask[..., None], 0)
                raw = raw[:, None, :, None].expand(-1, self.heads, -1, -1, -1)
                qp = torch.cat((qp + raw, raw), -2)
                kp = torch.cat((kp + raw, raw), -2)
            # exp(a) sets a positive inverse squared length for each point channel
            weight = self.distance_weight.to(dtype).exp()[None, :, None, :, None]
            qf = (point_distance_features(qp, True) * weight).flatten(-2)
            kf = point_distance_features(kp, False).flatten(-2)
        q = torch.cat((q, qf.to(q.dtype)), -1)
        k = torch.cat((k, kf.to(k.dtype)), -1)
        # Equal QKV widths retain the shared fused attention path
        v = F.pad(v, (0, q.shape[-1] - self.dim))
        y = scaled_attention(q, k, v, mask, bias, dropout=self.dropout if self.training else 0.0, scale=self.dim**-0.5, backend=self.backend)[
            ..., : self.dim
        ]
        y = y.transpose(1, 2).reshape_as(x)
        return self.out(y).masked_fill(~mask[..., None], 0)

def tied_topk(scores, count, fixed_width=False, *, query_mask=None):
    """
    Retain all finite candidates tied at the Kth score

    Args:
        scores:  Candidate scores with invalid entries set to negative infinity
        count:  Requested minimum candidate budget
        fixed_width:  Retain a static candidate axis for export
        query_mask:  Optional valid queries, broadcast over candidate counts
    Returns:
        selected: Scores and indices padded to the largest selected set
    """
    detached = scores.detach()
    threshold = detached.topk(min(count, scores.shape[-1]), -1).values[..., -1:]
    keep = torch.isfinite(detached) & (detached >= threshold)
    # ONNX retains masked candidates with a fixed particle dimension
    if fixed_width:
        width = scores.shape[-1]
    else:
        counts = keep.sum(-1)
        if query_mask is not None:
            counts = counts.masked_fill(~query_mask, 0)
        # Padded queries retain finite candidates for a finite softmax backward
        width = max(1, int(counts.max()))
    indices = detached.masked_fill(~keep, -torch.inf).topk(width, -1).indices
    selected = scores.gather(-1, indices).masked_fill(~keep.gather(-1, indices), -torch.inf)
    return selected, indices

class TriangleAttention(nn.Module):
    def __init__(
        self, width, heads, k=16, gamma=0.5, dropout=0.0, query_chunk=32, *, kj=None, kk=None, shared=False, distinct=False, backend="sdpa"
    ):
        super().__init__()
        self.heads, self.dim = heads, width // heads
        self.k, self.gamma, self.dropout, self.query_chunk = (k, gamma, dropout, query_chunk)
        self.kj, self.kk = kj or k, kk or k
        self.shared, self.distinct = shared, distinct
        self.fixed_candidates = False
        self.backend = backend
        self.projections = nn.Linear(width, 6 * width, bias=False)
        self.combine = mlp(4 * self.dim + 1, 2 * self.dim, self.dim, dropout)
        self.out = nn.Linear(width, width)

    def forward(self, x, mask, bias=None):
        """
        Aggregate valid tokens with standard, hyperbolic or triangle scores

        Args:
            x:  Learned token features (B,N,D)
            mask:  Valid particle mask (B,N)
            bias:  Optional finite pair bias (B,H,N,N)
        Returns:
            features: Masked token features (B,N,D)
        """
        projected = self.projections(x)
        if (
            self.gamma == 0
            and bias is None
            and min(self.kj, self.kk) >= x.shape[1]
            and not self.distinct
            and (not self.training or self.dropout == 0)
        ):
            b, n, _ = x.shape
            parts = [t.reshape(b, n, self.heads, self.dim).transpose(1, 2) for t in projected.chunk(6, -1)]
            q, kj, kk, kp, vj, vk = parts
            contexts = [
                scaled_attention(
                    F.pad(q, (0, self.dim)), F.pad(key, (0, self.dim)), torch.cat([value, kp], -1), mask, scale=self.dim**-0.5, backend=self.backend
                )
                for key, value in ((kj, vj), (kk, vk))
            ]
            first, second = contexts
            features = torch.cat(
                [
                    first[..., : self.dim],
                    second[..., : self.dim],
                    first[..., self.dim :],
                    second[..., self.dim :],
                    first.new_zeros((*first.shape[:-1], 1)),
                ],
                -1,
            )
            y = self.combine(features).transpose(1, 2).reshape_as(x)
            return self.out(y).masked_fill(~mask[..., None], 0)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if projected.dtype in (torch.float16, torch.bfloat16):
                projected = projected.float()
            return self.aggregate(projected, mask, bias)

    def aggregate(self, projected, mask, bias):
        """
        Accumulate triangle scores and expectations outside reduced precision

        Args:
            projected:  Six projected token streams (B,N,6D)
            mask:  Valid particle mask (B,N)
            bias:  Optional finite pair bias (B,H,N,N)
        Returns:
            features: Updated masked token features (B,N,D)
        """
        b, n, channels = projected.shape
        d = channels // 6

        def split(t):
            """
            Split token channels into attention heads

            Args:
                x:  Token features (B,N,D)
            Returns:
                heads: Tensor (B,H,N,D/H)
            """
            return t.reshape(b, n, self.heads, self.dim).transpose(1, 2)

        q, kj, kk, kp, vj, vk = map(split, projected.chunk(6, -1))
        sj = (q @ kj.transpose(-1, -2)) / math.sqrt(self.dim)
        sk = (q @ kk.transpose(-1, -2)) / math.sqrt(self.dim)
        sj = sj.masked_fill(~mask[:, None, None, :], -torch.inf)
        sk = sk.masked_fill(~mask[:, None, None, :], -torch.inf)
        outputs = []
        for start in range(0, n, self.query_chunk):
            rows_j = sj[:, :, start : start + self.query_chunk]
            rows_k = sk[:, :, start : start + self.query_chunk]
            query_mask = mask[:, None, start : start + self.query_chunk]
            if self.shared:
                selected, ij = tied_topk(rows_j + rows_k, self.kj, self.fixed_candidates, query_mask=query_mask)
                ik = ij
                invalid = ~torch.isfinite(selected)
                a = rows_j.gather(-1, ij).masked_fill(invalid, -torch.inf)
                c = rows_k.gather(-1, ik).masked_fill(invalid, -torch.inf)
            else:
                a, ij = tied_topk(rows_j, self.kj, self.fixed_candidates, query_mask=query_mask)
                c, ik = tied_topk(rows_k, self.kk, self.fixed_candidates, query_mask=query_mask)
            k = ij.shape[-1]
            length = a.shape[2]

            def gather(t, idx):
                return torch.gather(
                    t[:, :, None].expand(-1, -1, length, -1, -1), 3, idx[..., None].expand(-1, -1, -1, -1, self.dim)
                )

            pj, pk = gather(kp, ij), gather(kp, ik)
            compatibility = self.gamma * (pj @ pk.transpose(-1, -2)) / math.sqrt(self.dim)
            if bias is not None:
                bb = bias.expand(b, self.heads, n, n)
                rows = torch.gather(bb[:, :, None].expand(-1, -1, length, -1, -1), 3, ij[..., None].expand(-1, -1, -1, -1, n))
                compatibility = compatibility + torch.gather(rows, 4, ik[..., None, :].expand(-1, -1, -1, k, -1))
            logits = a[..., None] + c[..., None, :] + compatibility
            if self.distinct:
                keep = (ij[..., :, None] != ik[..., None, :]) & torch.isfinite(logits)
                # A singleton candidate set retains its valid diagonal as a fallback
                keep = torch.where(keep.any((-1, -2), keepdim=True), keep, torch.isfinite(logits))
                logits = logits.masked_fill(~keep, -torch.inf)
            weights = logits.flatten(-2).softmax(-1).reshape_as(logits)
            weights = F.dropout(weights, self.dropout, self.training)
            row, col = weights.sum(-1), weights.sum(-2)
            features = torch.cat(
                [
                    (row[..., None] * gather(vj, ij)).sum(-2),
                    (col[..., None] * gather(vk, ik)).sum(-2),
                    (row[..., None] * pj).sum(-2),
                    (col[..., None] * pk).sum(-2),
                    (weights * compatibility).sum((-1, -2))[..., None],
                ],
                -1,
            )
            outputs.append(self.combine(features))
        y = torch.cat(outputs, 2).transpose(1, 2).reshape(b, n, d)
        return self.out(y).masked_fill(~mask[..., None], 0)
