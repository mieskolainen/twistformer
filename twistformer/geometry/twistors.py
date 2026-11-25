# Lorentz and twistor algebra
#
# mikael.mieskolainen@cern.ch (2026)

"""Lorentz-covariant twistors in the incidence basis Z=(lambda, mu)

lambda -> S lambda, mu -> S^{-dagger} mu, det S=1
H is OFF-DIAGONAL in this basis. Projective normalization uses a transforming
timelike reference, never the null incidence norm Z^dagger H Z
"""

from dataclasses import dataclass
import torch
from .lorentz import dot, matrix, null_projection

@dataclass
class TwistorTensor:
    lam: torch.Tensor
    mu: torch.Tensor

    @property
    def Z(self):
        """
        Pack the incidence representation
        Returns:
            z: Complex coordinates (...,4), ordered lambda,mu
        """
        return torch.cat([self.lam, self.mu], -1)

    @classmethod
    def from_Z(cls, z):
        """
        Split complex coordinates into Weyl spinors

        Args:
            z:  Complex incidence coordinates (...,4)
        Returns:
            twistor: Paired lambda and mu tensors
        """
        if z.shape[-1] != 4 or not z.is_complex():
            raise ValueError("Z must have four complex coordinates")
        return cls(z[..., :2], z[..., 2:])

def spinor_bracket(a, b):
    """
    Contract two Weyl spinors with epsilon_ab

    Args:
        a:  Complex spinors (...,2)
        b:  Broadcast-compatible spinors
    Returns:
        bracket: SL(2,C)-invariant antisymmetric contraction
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def twistor_inner(z, w):
    """
    Contract twistors with the off-diagonal Hermitian form H

    Args:
        z:  Complex incidence coordinates (...,4)
        w:  Broadcast-compatible incidence coordinates
    Returns:
        inner: Lorentz-invariant complex contraction
    """
    return (z[..., :2].conj() * w[..., 2:]).sum(-1) + (z[..., 2:].conj() * w[..., :2]).sum(-1)

def incidence_mu(x, lam):
    """
    Apply the incidence relation mu=i Xbar lambda

    Args:
        x:  Real anchors (...,4), transforming as four-vectors
        lam:  Complex Weyl spinors (...,2)
    Returns:
        mu: Dual Weyl spinors (...,2)
    """
    return 1j * (matrix(x, dual=True) @ lam[..., None]).squeeze(-1)

def spinor_from_null(p, eps=1e-12):
    """
    Factor future null P=lambda lambda^dagger using two charts

    Args:
        p:  Future null four-vectors (...,4), or zero
        eps:  Pole and zero-energy tolerance
    Returns:
        lam: Weyl spinors (...,2), zero for zero momentum
    """
    e, px, py, pz = p.unbind(-1)
    a, b = (e + pz).clamp_min(eps).sqrt(), (e - pz).clamp_min(eps).sqrt()
    alt = pz < 0
    transverse = torch.complex(px, py)
    lam = torch.stack(
        [torch.where(alt, transverse.conj() / b, a.to(transverse.dtype)), torch.where(alt, b.to(transverse.dtype), transverse / a)],
        -1,
    )
    return lam.masked_fill((e <= eps)[..., None], 0)

def region_momenta(p):
    """
    Build ordered region momenta from exclusive cumulative sums

    Args:
        p:  Ordered particle momenta (B,N,4)
    Returns:
        x: Region anchors (B,N,4)
    """
    return p.cumsum(1) - p

def reference_metric(total, eps=1e-12):
    """
    Construct a positive twistor metric from the timelike event total

    Args:
        total:  Future event totals (B,4)
        eps:  Absolute lower bound on total mass squared
    Returns:
        reference: Tuple (unit total, metric G, timelike mask)
    """
    mass2 = dot(total, total)
    # Collinear energy flow can acquire positive mass squared from roundoff
    roundoff = 64 * torch.finfo(total.dtype).eps * total[..., 0].square()
    valid = mass2 > roundoff.clamp_min(eps)
    u = total / mass2.clamp_min(eps).sqrt()[..., None]
    u = u.masked_fill(~valid[..., None], 0)
    zero = torch.zeros_like(matrix(u))
    g = torch.cat([torch.cat([matrix(u, dual=True), zero], -1), torch.cat([zero, matrix(u)], -1)], -2)
    return u, g, valid

def twistors_from_p4(p, mask, kind="penrose"):
    """
    Build incidence twistors from massive or null particle momenta

    Args:
        p:  Future non-spacelike momenta (B,N,4)
        mask:  Valid particle mask (B,N)
        kind:  penrose uses the event total, momentum uses ordered regions
    Returns:
        geometry: Tuple (TwistorTensor, reference metric)
    """
    p = p.masked_fill(~mask[..., None], 0)
    total = p.sum(1)
    u, metric, valid = reference_metric(total)
    lam = spinor_from_null(null_projection(p, total))
    if kind == "penrose":
        x = u[:, None].expand_as(p)
    elif kind == "momentum":
        scale = dot(total, total).clamp_min(1e-12).sqrt()
        x = region_momenta(p) / scale[:, None, None]
    else:
        raise ValueError("kind must be penrose or momentum")
    tw = TwistorTensor(lam, incidence_mu(x, lam))
    z = tw.Z.masked_fill(~(mask & valid[:, None])[..., None], 0)
    return TwistorTensor.from_Z(z), metric

def projective_normalize(z, metric, eps=1e-12):
    """
    Normalize projective magnitude with the transforming metric G

    Args:
        z:  Complex twistor coordinates (B,N,4)
        metric:  Positive reference metric (B,4,4)
        eps:  Lower bound on the Hermitian norm squared
    Returns:
        z: Unit-norm representatives, zero for degenerate norms
    """
    norm2 = ((z.conj() @ metric) * z).sum(-1).real
    normalized = z / norm2.clamp_min(eps).sqrt()[..., None]
    return normalized.masked_fill((norm2 <= eps)[..., None], 0)

def rest_frame_coordinates(z, metric):
    """
    Condition twistors in the event rest frame before complex64 transport

    Args:
        z:  Complex128 incidence coordinates (B,N,4)
        metric:  Timelike reference metric (B,4,4)
    Returns:
        geometry: Complex64 coordinates and Euclidean reference metric
    """
    eye = torch.eye(4, dtype=metric.dtype, device=metric.device)
    trace = metric.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    valid = trace > 0
    # Each Weyl block has determinant one, so sqrt(G)=(G+I)/sqrt(tr(G)/2+2)
    boost = (metric + eye) / (trace / 2 + 2).sqrt()[:, None, None]
    coordinates = z @ boost.transpose(-1, -2)
    euclidean = eye.expand_as(metric).masked_fill(~valid[:, None, None], 0)
    coordinates = projective_normalize(coordinates, euclidean)
    return coordinates.to(torch.complex64), euclidean.to(torch.complex64)

def pairwise_features(z, metric, mask, normalized=False):
    """
    Construct real gauge-invariant pair features and complex overlaps

    C_ij transforms by exp(i(theta_j-theta_i)). Its conjugate transports
    the phase of sender j into receiver i

    Args:
        z:  Complex twistor coordinates (B,N,4)
        metric:  Reference metric (B,4,4)
        mask:  Valid particle mask (B,N)
        normalized:  Input already has unit projective norm
    Returns:
        pairs: Tuple (four real pair features, complex overlap C)
    """
    if not normalized:
        z = projective_normalize(z, metric)
    c = (z.conj() @ metric) @ z.transpose(1, 2)
    dual = torch.cat((z[..., 2:], z[..., :2]), -1)
    h = z.conj() @ dual.transpose(1, 2)
    ll = spinor_bracket(z[:, :, None, :2], z[:, None, :, :2]).abs().square()
    mm = spinor_bracket(z[:, :, None, 2:], z[:, None, :, 2:]).abs().square()
    valid = mask[:, :, None] & mask[:, None, :]
    c = c.masked_fill(~valid, 0)
    feats = torch.stack([c.abs().square(), h.abs().square(), ll, mm], -1)
    return feats.masked_fill(~valid[..., None], 0), c

def three_point_features(overlap, mask, z=None, metric=None, normalized=False):
    """
    Reduce closed cycles C_ij C_jk C_ki without a cubic tensor

    Args:
        overlap:  Complex pair overlaps (B,N,N)
        mask:  Valid particle mask (B,N)
        z:  Optional coordinates for a rank-four overlap factorization
        metric:  Reference metric required with coordinates
        normalized:  Coordinates already have unit projective norm
    Returns:
        features: Real and imaginary cycle averages (B,N,N,2)
    """
    valid = mask[:, :, None] & mask[:, None, :]
    c = overlap.masked_fill(~valid, 0)
    if z is None:
        square = c @ c
    else:
        if metric is None:
            raise ValueError("Three-point coordinates require a reference metric")
        if not normalized:
            z = projective_normalize(z, metric)
        z = z.masked_fill(~mask[..., None], 0)
        left = z.conj() @ metric
        right = z.transpose(1, 2)
        # C=LR gives C squared=L(RL)R with O(N squared) arithmetic
        square = (left @ (right @ left)) @ right
    cycles = c * square.transpose(1, 2)
    cycles = cycles / mask.sum(1).clamp_min(1)[:, None, None]
    return torch.stack([cycles.real, cycles.imag], -1).masked_fill(~valid[..., None], 0)

def geometry_update(z, metric, overlap, attention, step, mask, transport="phase", normalized=False):
    """
    Transport whole twistor representations with aligned phases

    Args:
        z:  Complex twistor coordinates (B,N,4)
        metric:  Reference metric (B,4,4)
        overlap:  Gauge-covariant overlaps (B,N,N)
        attention:  Real invariant weights (B,N,N)
        step:  Real bounded update factors (B,N)
        mask:  Valid particle mask (B,N)
        transport:  phase alignment or covariant spin-frame transport
        normalized:  Coordinates already have unit projective norm
    Returns:
        updated: Projectively normalized coordinates (B,N,4)
    """
    if not normalized:
        z = projective_normalize(z, metric)
    context = transport_channels(z, overlap, attention, z[..., None], mask, transport).squeeze(-1)
    updated = projective_normalize(z + step[..., None] * context, metric)
    return updated.masked_fill(~mask[..., None], 0)

def transport_channels(z, overlap, attention, values, mask, mode="phase"):
    """
    Transport representation channels with receiver phase and Lorentz covariance

    Args:
        z:  Projectively normalized coordinates (B,N,4)
        overlap:  Metric overlaps (B,N,N)
        attention:  Invariant real weights (B,N,N)
        values:  Complex channel fields (B,N,4,C)
        mask:  Valid particle mask (B,N)
        mode:  phase or frame, with phase fallback for singular dyads
    Returns:
        context: Covariant receiver fields (B,N,4,C)
    """
    valid = mask[:, :, None] & mask[:, None, :]
    weights = attention.to(overlap.dtype).masked_fill(~valid, 0)
    if mode == "phase":
        return torch.einsum("bnm,bmrc->bnrc", weights * overlap.conj(), values)
    if mode != "frame":
        raise ValueError("Unknown representation transport")
    lam = z[..., :2]
    partner = (overlap.conj() @ lam) / mask.sum(1).clamp_min(1)[:, None, None]
    frame, inverse, nonsingular = spin_frame(lam, partner, eps=1e-8)
    nonsingular = nonsingular & mask
    pairs = nonsingular[:, :, None] & nonsingular[:, None, :]
    frame_weights = weights.masked_fill(~pairs, 0)
    local_lam = inverse @ values[..., :2, :]
    local_mu = frame.mH @ values[..., 2:, :]
    lam_context = frame @ torch.einsum("bnm,bmrc->bnrc", frame_weights, local_lam)
    mu_context = inverse.mH @ torch.einsum("bnm,bmrc->bnrc", frame_weights, local_mu)
    fallback = torch.einsum("bnm,bmrc->bnrc", (weights * overlap.conj()).masked_fill(pairs, 0), values)
    return torch.cat([lam_context, mu_context], -2) + fallback

def laboratory_reference_features(p, mask, kind, beam_direction):
    """
    Contract momenta with laboratory time and an oriented spatial plane

    Args:
        p:  Physical momenta (B,N,4)
        mask:  Valid particle mask (B,N)
        kind:  time or beam_plane
        beam_direction:  Oriented plane normal
    Returns:
        features: Laboratory energies and optional signed pair areas with reference padding
    """
    if kind not in {"time", "beam_plane"}:
        raise ValueError("Laboratory contractions require time or beam_plane")
    p = p.masked_fill(~mask[..., None], 0)
    # t dot p = E and t dot P = total laboratory energy
    energy = p[..., 0].clamp_min(0)
    scalars = torch.cat((energy, energy.sum(1, keepdim=True)), dim=1).log1p()
    if kind == "time":
        return scalars[..., None], None
    # F_ij = n dot (p_i cross p_j), with F_xy = +1 for n = +z
    normal = p.new_tensor(beam_direction)
    normal = normal / normal.norm()
    x, y, z = p[..., 1:].unbind(-1)
    nx, ny, nz = normal.unbind(-1)
    normal_cross_p = torch.stack((ny * z - nz * y, nz * x - nx * z, nx * y - ny * x), -1)
    area = normal_cross_p @ p[..., 1:].transpose(1, 2)
    pairs = area.asinh().masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0)
    pairs = torch.nn.functional.pad(pairs, (0, 2, 0, 2))
    scalars = torch.nn.functional.pad(scalars, (0, 1))
    return scalars[..., None], pairs[..., None]

def append_reference(p, z, metric, mask, kind, beam_direction):
    """
    Append null reference twistors or scalar time and plane reference tokens

    Args:
        p:  Physical momenta (B,N,4)
        z:  Incidence coordinates (B,N,4)
        metric:  Event reference metric (B,4,4)
        mask:  Valid particle mask (B,N)
        kind:  event, beam, time, beam_plane or frame
        beam_direction:  Nonzero laboratory spatial direction
    Returns:
        geometry: Extended coordinates and mask, with degenerate null references masked
    """
    p = p.masked_fill(~mask[..., None], 0)
    if kind in {"time", "beam_plane"}:
        # Timelike vectors and plane bivectors enter through scalar contractions
        count = 2 if kind == "beam_plane" else 1
        reference = z.new_zeros(p.shape[0], count, 4)
        valid = mask.any(dim=1, keepdim=True).expand(-1, count)
        return torch.cat((z, reference), 1), torch.cat((mask, valid), 1)
    total = p.sum(1)
    anchor, _, timelike = reference_metric(total)
    if kind == "event":
        weights = dot(p, total[:, None]).clamp_min(0)
        average = (weights[..., None] * p).sum(1) / weights.sum(1).clamp_min(1e-12)[:, None]
        null = null_projection(average[:, None], total)
    elif kind == "beam":
        direction = p.new_tensor(beam_direction)
        direction = direction / direction.norm()
        null = torch.cat([direction.new_ones(1), direction]).expand(p.shape[0], 1, -1)
    elif kind == "frame":
        # Labeled future null rays span the laboratory time and spatial axes
        axes = torch.eye(3, dtype=p.dtype, device=p.device)
        directions = torch.stack((axes, -axes), dim=1).reshape(6, 3)
        null = torch.cat([directions.new_ones(6, 1), directions], -1)
        null = null[None].expand(p.shape[0], -1, -1)
    else:
        raise ValueError("Reference kind must be event, beam, time, beam_plane or frame")
    valid = timelike[:, None] & (null[..., 0] > 1e-12)
    null = null.masked_fill(~valid[..., None], 0)
    reference = reference_twistor(null, anchor[:, None], validate=False).Z
    reference = projective_normalize(reference, metric)
    reference = reference.masked_fill(~valid[..., None], 0)
    return torch.cat([z, reference], 1), torch.cat([mask, valid], 1)

def spin_frame(a, b, eps=1e-12):
    """
    Construct the raw covariant dyad F=[a,b]

    Singular frames have zero inverse and must be excluded from transport
    Complex64 also excludes determinants below the relative roundoff threshold

    Args:
        a:  First Weyl spinors (...,2)
        b:  Second Weyl spinors (...,2)
        eps:  Lower bound on the invariant determinant magnitude
    Returns:
        frame: Tuple (F, inverse, valid mask)
    """
    f = torch.stack([a, b], -1)
    determinant = spinor_bracket(a, b)
    threshold = determinant.real.new_tensor(eps)
    if f.dtype == torch.complex64:
        scale = a.norm(dim=-1) * b.norm(dim=-1)
        threshold = torch.maximum(threshold, 32 * torch.finfo(a.real.dtype).eps * scale)
    valid = determinant.abs() > threshold
    safe = torch.where(valid, determinant, torch.ones_like(determinant))
    adjugate = torch.stack([b[..., 1], -b[..., 0], -a[..., 1], a[..., 0]], -1)
    inv = (adjugate.reshape_as(f) / safe[..., None, None]).masked_fill(~valid[..., None, None], 0)
    return f, inv, valid

def frame_transport(receiver, sender_inverse):
    """
    Transport lambda with F_i F_j^{-1}. Mu uses the inverse adjoint

    Args:
        receiver:  Receiver dyads (...,2,2)
        sender_inverse:  Inverse sender dyads (...,2,2), from valid frames
    Returns:
        transport: Complex maps (...,2,2)
    """
    return receiver @ sender_inverse

def reference_twistor(null_p4, anchor, *, validate=True):
    """
    Build an explicit reference twistor with a transforming anchor

    A fixed laboratory beam does not define a Lorentz-invariant reference

    Args:
        null_p4:  Future null momentum (...,4), or zero
        anchor:  Transforming incidence anchors (...,4)
        validate:  Check external momenta, internally projected references are null
    Returns:
        twistor: Reference TwistorTensor
    """
    if validate and (
        (dot(null_p4, null_p4).abs() > 1e-8 * null_p4[..., 0].square().clamp_min(1)).any() or (null_p4[..., 0] < 0).any()
    ):
        raise ValueError("Reference momentum must be future null")
    lam = spinor_from_null(null_p4)
    return TwistorTensor(lam, incidence_mu(anchor, lam))
