# Lorentz and twistor algebra
#
# mikael.mieskolainen@cern.ch (2026)

"""Metric (+---). P = p^mu sigma_mu and Pbar = p^0 I - p^i sigma_i"""

import torch

def dot(p, q):
    """
    Contract four-vectors with metric (+---)

    Args:
        p:  Four-vectors (...,4), ordered E,px,py,pz
        q:  Broadcast-compatible four-vectors
    Returns:
        value: Minkowski contraction (...)
    """
    return p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(-1)

def matrix(p, dual=False):
    """
    Map a real four-vector to its Hermitian Pauli matrix

    Args:
        p:  Four-vectors (...,4)
        dual:  Use the conjugate Pauli basis for covectors
    Returns:
        matrix: Complex Hermitian tensors (...,2,2)
    """
    e, x, y, z = p.unbind(-1)
    if dual:
        x, y, z = -x, -y, -z
    return torch.stack([torch.stack([e + z, x - 1j * y], -1), torch.stack([x + 1j * y, e - z], -1)], -2)

def from_matrix(p):
    """
    Recover four-vectors from Hermitian Pauli matrices

    Args:
        p:  Hermitian tensors (...,2,2)
    Returns:
        p4: Real four-vectors (...,4)
    """
    return torch.stack(
        [
            (p[..., 0, 0].real + p[..., 1, 1].real) / 2,
            p[..., 0, 1].real,
            -p[..., 0, 1].imag,
            (p[..., 0, 0].real - p[..., 1, 1].real) / 2,
        ],
        -1,
    )

def validate_p4(p, mask):
    """
    Require finite future non-spacelike particle momenta

    Args:
        p:  Particle momenta (B,N,4)
        mask:  Valid particle mask (B,N)
    Returns:
        None: Invalid momenta raise ValueError
    """
    if p.shape != (*mask.shape, 4) or mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("Expected p4 (B,N,4) and a Boolean mask (B,N)")
    if not p.is_floating_point():
        raise ValueError("p4 must contain real floating-point momenta")
    valid = p[mask].double()
    if not torch.isfinite(valid).all() or (valid[:, 0] <= 0).any():
        raise ValueError("Four-momenta must be finite with positive energy")
    tolerance = 1e-5 * valid[:, 0].square().clamp_min(1)
    if (dot(valid, valid) < -tolerance).any():
        raise ValueError("Spacelike four-momenta are unsupported")

def null_projection(p, total, eps=1e-12):
    """
    Project to null momenta covariantly relative to the event total

    In the total rest frame k=(|p_vec|,p_vec). Rest particles map to zero
    Mass information must be retained separately

    Args:
        p:  Particle momenta (B,N,4)
        total:  Future event total (B,4)
        eps:  Lower bound for the scalar denominator
    Returns:
        k: Projected null momenta (B,N,4)
    """
    m2 = dot(p, p).clamp_min(0)
    pp = dot(p, total[:, None])
    total2 = dot(total, total).clamp_min(0)[:, None]
    discriminant = pp.square() - m2 * total2
    positive = discriminant > 0
    # Zero subgradient at rest and on padding avoids sqrt backward singularities
    root = discriminant.masked_fill(~positive, 1).sqrt().masked_fill(~positive, 0)
    alpha = m2 / (pp + root).clamp_min(eps)
    return p - alpha[..., None] * total[:, None]

def invariant_features(p, mask):
    """
    Construct particle mass, event-energy and multiplicity scalars

    Args:
        p:  Particle momenta (B,N,4)
        mask:  Valid particle mask (B,N)
    Returns:
        features: Log-transformed Lorentz scalars (B,N,3)
    """
    p = p.masked_fill(~mask[..., None], 0)
    total = p.sum(1)
    mass = dot(p, p).clamp_min(0)
    energy = dot(p, total[:, None]).clamp_min(0)
    count = mask.sum(1).clamp_min(1).to(p.dtype)
    features = torch.stack([mass.log1p(), energy.log1p(), count.log()[:, None].expand_as(mass)], -1)
    return features.masked_fill(~mask[..., None], 0)
