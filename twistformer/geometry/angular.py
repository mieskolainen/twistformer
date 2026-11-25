# Angular distances and transport on the direction sphere
#
# mikael.mieskolainen@cern.ch (2026)

import torch


def wrap_azimuth(phi):
    """
    Wrap azimuthal differences to the half-open interval [-pi, pi)

    Args:
        phi:  Angles in radians
    Returns:
        result:  Wrapped angles in radians
    """
    return (phi + torch.pi).remainder(2 * torch.pi) - torch.pi


def angular_distance_squared(first, second, *, periodic_azimuth=False):
    """
    Compute squared distance in a two-coordinate angular chart

    Args:
        first:  Broadcastable angular coordinates with two components
        second:  Broadcastable angular coordinates with two components
        periodic_azimuth:  Wrap the final coordinate over 2 pi radians
    Returns:
        result:  Squared angular separation
    """
    x = first[..., 0] - second[..., 0]
    y = first[..., 1] - second[..., 1]
    if periodic_azimuth:
        y = wrap_azimuth(y)
    return x.square().add_(y.square_())


def projective_tangent(coordinates):
    """
    Lift directions and their coordinate derivatives to the unit sphere

    Args:
        coordinates:  Two coordinates of a ray with third component one
    Returns:
        direction:  Unit ray direction
        jacobian:  Map from local coordinate displacements to tangent vectors
        projector:  Orthogonal projector onto the tangent plane
    """
    ray = torch.cat((coordinates, torch.ones_like(coordinates[..., :1])), dim=-1)
    length = torch.linalg.vector_norm(ray, dim=-1, keepdim=True)
    direction = ray / length
    identity = torch.eye(3, dtype=ray.dtype, device=ray.device)
    projector = identity - direction[..., :, None] * direction[..., None, :]
    return direction, projector[..., :2] / length[..., None], projector


def spherical_transport(source, target):
    """
    Transport tangents along the short great-circle arc

    Args:
        source:  Source unit directions with three components
        target:  Destination unit directions with three components
    Returns:
        operator:  Linear map on source tangents, zero at the antipodal ambiguity
    """
    denominator = 1 + (source * target).sum(-1)[..., None, None]
    identity = torch.eye(3, dtype=source.dtype, device=source.device)
    correction = (source + target)[..., :, None] * target[..., None, :]
    return torch.where(denominator > 1e-9, identity - correction / denominator.clamp_min(1e-9), 0.)


def transported_moment_edges(coordinates, offsets, covariance, edges):
    """
    Compare dipoles and traceless quadrupoles in parallel tangent frames

    Args:
        coordinates:  Projective ray coordinates with shape [..., nodes, 2]
        offsets:  Dipole offsets in the same coordinates
        covariance:  Local second central moments with shape [..., nodes, 2, 2]
        edges:  Nonempty sequence of source and target index pairs
    Returns:
        result:  Dipole alignment and residual, quadrupole alignment and residual per edge
    """
    directions, jacobian, projector = projective_tangent(coordinates)
    dipole = (jacobian @ offsets[..., None]).squeeze(-1)
    quadrupole = jacobian @ covariance @ jacobian.transpose(-1, -2)
    trace = quadrupole.diagonal(dim1=-2, dim2=-1).sum(-1)
    quadrupole = quadrupole - .5 * trace[..., None, None] * projector

    source, target = zip(*edges, strict=True)
    operator = spherical_transport(directions[..., source, :], directions[..., target, :])
    carried_dipole = (operator @ dipole[..., source, :, None]).squeeze(-1)
    carried_quadrupole = operator @ quadrupole[..., source, :, :] @ operator.transpose(-1, -2)
    fields = []
    for first, second in ((carried_dipole, dipole[..., target, :]),
                          (carried_quadrupole.flatten(-2), quadrupole[..., target, :, :].flatten(-2))):
        norms = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)
        alignment = ((first * second).sum(-1) / norms.clamp_min(1e-24)).clamp(-1, 1)
        residual = torch.linalg.vector_norm(first - second, dim=-1)
        fields.extend((alignment, residual))
    return torch.stack(fields, dim=-1)
