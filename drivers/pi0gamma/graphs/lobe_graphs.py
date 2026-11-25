# Separated photon lobes across the measured shower depth
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from torch.nn.utils.rnn import pad_sequence

from twistformer.geometry.angular import angular_distance_squared, transported_moment_edges
from twistformer.geometry.graph_polynomials import point_tensor
from .graph_inputs import ANGLE_UNIT_RAD
from .spectral_depth_graphs import normalized_depth


PION_MASS_GEV = .1349768
CORE_WIDTHS = (.24, .48)
DEPTH_EDGES = (.32, .62)
DEPTH_NAMES = ("front", "middle", "back")
DEPTH_PAIRS = ((0, 1), (1, 2), (0, 2))
SOURCE_DEPTHS, TARGET_DEPTHS = zip(*DEPTH_PAIRS, strict=True)
TERMS = ("opening_rad", "core_mass_GeV", "split_mass_GeV", "secondary_fraction", "profile_overlap",
         "primary_drift_rad", "secondary_drift_rad", "secondary_coherence")
DEPTH_TERMS = ("opening_rad", "primary_core_fraction", "secondary_core_fraction", "secondary_share")
TRANSPORT_TERMS = ("dipole_alignment", "dipole_residual", "quadrupole_alignment", "quadrupole_residual")
LOBE_GRAPH_NAMES = tuple(
    f"shower_frame_lobe_{round(scale * ANGLE_UNIT_RAD * 1e3)}mrad_{name}"
    for scale in CORE_WIDTHS for name in (*TERMS, *(f"{depth}_{term}" for depth in DEPTH_NAMES for term in DEPTH_TERMS))
)
LOBE_TRANSPORT_NAMES = tuple(
    f"shower_frame_lobe_transport_{round(scale * ANGLE_UNIT_RAD * 1e3)}mrad_{lobe}_{DEPTH_NAMES[a]}_{DEPTH_NAMES[b]}_{name}"
    for scale in CORE_WIDTHS for lobe in ("primary", "secondary") for a, b in DEPTH_PAIRS for name in TRANSPORT_TERMS
)


def lobe_anchors(position, marks, count):
    """
    Seed directions from energetic cells in each depth region

    Args:
        position:  Projective cell coordinates per shower
        marks:  Cell energy marks in each depth region
        count:  Maximum seed count per depth region
    Returns:
        result:  Candidate directions and a validity mask
    """
    energy = marks.transpose(1, 2)
    order = energy.argsort(dim=-1, descending=True, stable=True)[..., :count]
    valid = energy.gather(2, order).flatten(1) > 0
    anchors = position.gather(1, order.reshape(len(position), -1, 1).expand(-1, -1, 2))
    return anchors, valid


def longitudinal_coherence(energy):
    """
    Measure persistence from energy on distinct depth vertices

    Args:
        energy:  Local support fractions in the three measured depth regions
    Returns:
        coherence:  Normalized strength of cross-depth edges
    """
    front, middle, back = energy.unbind(-1)
    edges = front * middle + front * back + middle * back
    return edges / energy.square().sum(-1).clamp_min(1e-24)


def transport_descriptors(centers, narrow_centers, covariance, support):
    """
    Compare energy-flow moments in parallel frames along each shower lobe

    Args:
        centers:  Lobe centroids in projective coordinates per depth region
        narrow_centers:  Centroids with a stronger core weight
        covariance:  Local second central moments in projective coordinates
        support:  Lobe energy fraction in each depth region
    Returns:
        rows:  Dipole and quadrupole alignments and residuals on depth edges
    """
    result = transported_moment_edges(ANGLE_UNIT_RAD * centers, ANGLE_UNIT_RAD * (centers - narrow_centers),
                                      ANGLE_UNIT_RAD**2 * covariance, DEPTH_PAIRS)
    valid = (support[:, :, SOURCE_DEPTHS] > 0) & (support[:, :, TARGET_DEPTHS] > 0)
    return (result * valid[..., None]).flatten(1)


def lobe_graph_rows(showers, device="cpu", *, include_lobes=True, include_transport=False):
    """
    Reconstruct two angular lobes with support at independent depths

    Args:
        showers:  Projective cell coordinates, depth and log energy per shower
        device:  Torch device for graph contractions
        include_lobes:  Include the separation and energy sharing columns
        include_transport:  Include the directional alignment and residual columns
    Returns:
        rows:  Selected lobe columns followed by the selected transport columns
    """
    hits = [point_tensor(shower)[..., :4] for shower in showers]
    depths = [normalized_depth(shower) for shower in hits]
    data = pad_sequence(hits, batch_first=True).to(device)
    depth = pad_sequence(depths, batch_first=True).to(device)
    indices = torch.arange(len(hits), device=device)
    counts = torch.tensor([len(shower) for shower in hits], device=device)
    valid = torch.arange(data.shape[1], device=device) < counts[:, None]
    position = data[..., :2]
    energy = data[..., 3].exp() * valid
    total = energy.sum(1)
    weight = energy / total[:, None]
    region = torch.bucketize(depth, data.new_tensor(DEPTH_EDGES), right=True)
    marks = weight[..., None] * (region[..., None] == torch.arange(len(DEPTH_NAMES), device=device))
    position_marks = (marks[..., None] * position[:, :, None]).flatten(2)
    if include_transport:
        products = (position[..., :, None] * position[..., None, :]).flatten(-2)
        product_marks = (marks[..., None] * products[:, :, None]).flatten(2)
    band_energy = marks.sum(1)
    normalization = band_energy.clamp_min(.04)
    primary, primary_valid = lobe_anchors(position, marks, 6)
    primary_distance = angular_distance_squared(position[:, None], primary[:, :, None])
    # Search from half the symmetric pion opening angle
    separation_cut = PION_MASS_GEV / total / ANGLE_UNIT_RAD
    fields, transport_fields = [], []
    for scale in CORE_WIDTHS:
        kernel = (1 + primary_distance / scale**2).reciprocal().square()
        support = kernel @ marks
        strength = support.sum(-1) * (1 + longitudinal_coherence(support / normalization[:, None]))
        first = strength.masked_fill(~primary_valid, -1).argmax(1)
        first_weight = kernel[indices, first] * weight
        first_center = (first_weight[..., None] * position).sum(1) / first_weight.sum(1, keepdim=True)
        first_distance = angular_distance_squared(position, first_center[:, None])
        first_basis = (1 + first_distance / scale**2).reciprocal().square()

        separated = first_distance >= separation_cut.clamp_min(2 * scale)[:, None].square()
        residual = (1 - first_basis) * separated
        secondary_marks = marks * residual[..., None]
        secondary, secondary_valid = lobe_anchors(position, secondary_marks, 10)
        distance = angular_distance_squared(position[:, None], secondary[:, :, None])
        kernel = (1 + distance / scale**2).reciprocal().square()
        support = kernel @ secondary_marks
        coherence = longitudinal_coherence(support / normalization[:, None])
        second = (support.sum(-1) * (1 + coherence)).masked_fill(~secondary_valid, -1).argmax(1)
        second_basis = kernel[indices, second] * residual

        cores = torch.stack((first_basis, second_basis), dim=1)
        band_support = cores @ marks
        band_centers = (cores @ position_marks).reshape(len(hits), 2, len(DEPTH_NAMES), 2)
        band_centers = band_centers / band_support.clamp_min(1e-24)[..., None]
        if include_transport:
            narrow = cores.square()
            narrow_centers = (narrow @ position_marks).reshape_as(band_centers)
            narrow_centers = narrow_centers / (narrow @ marks).clamp_min(1e-24)[..., None]
            second_moment = (cores @ product_marks).reshape(*band_centers.shape, 2)
            covariance = second_moment / band_support.clamp_min(1e-24)[..., None, None]
            covariance = covariance - band_centers[..., :, None] * band_centers[..., None, :]
            transport_fields.append(transport_descriptors(band_centers, narrow_centers, covariance, band_support))
        if not include_lobes:
            continue
        core_fraction = band_support.sum(-1)
        centers = (band_centers * band_support[..., None]).sum(2) / core_fraction.clamp_min(1e-24)[..., None]
        drift = ((band_centers - centers[:, :, None]).square().sum(-1) * band_support).sum(-1)
        drift = (drift / core_fraction.clamp_min(1e-24)).sqrt() * ANGLE_UNIT_RAD
        first_fraction, second_fraction = core_fraction.unbind(1)
        active = second_fraction > 0
        opening = (centers[:, 0] - centers[:, 1]).norm(dim=-1) * ANGLE_UNIT_RAD * active
        share = second_basis / (first_basis + second_basis).clamp_min(1e-24)
        assigned = (weight * share).sum(1)
        profile = band_support / core_fraction.clamp_min(1e-24)[..., None]
        overlap = (profile[:, 0] * profile[:, 1]).sqrt().sum(1)
        fields.append(torch.stack((opening, total * opening * (first_fraction * second_fraction).sqrt(),
                                   total * opening * (assigned * (1 - assigned)).clamp_min(0).sqrt(), assigned,
                                   overlap, drift[:, 0], drift[:, 1], coherence[indices, second] * active), dim=1))
        band_opening = (band_centers[:, 0] - band_centers[:, 1]).norm(dim=-1) * ANGLE_UNIT_RAD
        band_opening = band_opening * (band_support > 0).all(1)
        fractions = band_support / band_energy.clamp_min(1e-24)[:, None]
        shares = (share[..., None] * marks).sum(1) / band_energy.clamp_min(1e-24)
        fields.append(torch.stack((band_opening, fractions[:, 0], fractions[:, 1], shares), dim=-1).flatten(1))
    return torch.cat(fields + transport_fields, dim=1).float()
