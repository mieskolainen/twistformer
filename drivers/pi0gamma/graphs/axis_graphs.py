# Cluster-local rooted shower graphs and energy sharing between two axes
#
# mikael.mieskolainen@cern.ch (2026)

import torch
from torch.nn.utils.rnn import pad_sequence

from twistformer.geometry.angular import angular_distance_squared
from twistformer.geometry.graph_polynomials import point_tensor
from .graph_inputs import ANGLE_UNIT_RAD, N_LAYERS, SHOWER_DEPTH_REGIONS, depth_indices, oriented_core


SCALES = (.18, .4, .85, 1.7)
BANDS = ((0, 2), (1, 5), (2, 7), (4, 11), (0, 11))
SCALARS = ("opening", "opening_x", "opening_y", "energy_opening", "balance", "root1", "root2",
           "second_gain", "overlap", "centre_bridge", "hard_bridge", "soft_gain", "tau21", "energy_radius")
LAYER_TERMS = ("mass", "core1", "core2", "gain2", "overlap", "four_star", "second_share", "bridge", "distance2",
               "soft_gain", "hard_gain", "root2_x", "root2_y")
BAND_TERMS = ("second_share", "gain2", "two_root_mass", "depth_mean1", "depth_mean2", "depth_rms2", "root2_concentration")
FULL_SHOWER_DUPLICATES = ("gain2", "two_root_mass")


def axis_names(bands, n_layers, *, shower_frame=False):
    """
    Name the two-axis features in contraction order

    Args:
        bands:  Half-open depth regions
        n_layers:  Number of depth bins
        shower_frame:  Omit redundant full-shower terms in the shower frame
    Returns:
        names:  Two-axis feature columns
    """
    return tuple(f"axes_s{s}_{field}" for s in SCALES for field in
                 (*SCALARS, *(f"{term}_L{l}" for term in LAYER_TERMS for l in range(n_layers)),
                  *(f"L{lo}to{hi}_{term}" for lo, hi in bands for term in BAND_TERMS
                    if not shower_frame or (lo, hi) != (0, n_layers) or term not in FULL_SHOWER_DUPLICATES)))


AXIS_NAMES = axis_names(BANDS, N_LAYERS)


def _sqrt_zero_slope(value):
    """
    Take the square root with a zero slope at vanishing support

    Args:
        value:  Nonnegative weight, squared distance or variance
    Returns:
        root:  Square root with the usual derivative at positive values
    """
    zero = value == 0
    return value.masked_fill(zero, 1).sqrt().masked_fill(zero, 0)


def axis_graph_rows(showers, device="cpu", *, shower_frame=False):
    """
    Contract energy graphs on the selected cluster about one or two calorimeter axes

    Args:
        showers:  Local transverse coordinates, depth and log energy per cell
        device:  Device for the event batches
        shower_frame:  Use measured angular offsets and relative depth
    Returns:
        result:  Root separations, shared-hit hyperedges and depth profiles
    """
    count = len(showers)
    width = max(map(len, showers))
    # Stable detector order fixes energy ties before choosing candidate roots
    # Resolve nearly coincident candidate roots before rounding the graph outputs
    cells = []
    for shower in showers:
        hits = point_tensor(shower)
        depth = hits[:, 2]
        energy = hits[:, 3].exp()
        fraction = energy / energy.sum()
        xy = oriented_core(hits[:, :2], depth, fraction, (2., .8, .4))
        row = torch.column_stack((xy, depth, energy))
        # The ROOT reader already orders cells by descending energy
        if energy.device.type != "cpu" or not torch.all(energy[:-1] > energy[1:]):
            order = depth.argsort(stable=True)
            for key in (xy[:, 1], xy[:, 0], -energy):
                order = order[key[order].argsort(stable=True)]
            row = row[order]
        cells.append(row)
    data = pad_sequence(cells, batch_first=True).to(device)
    xy, depth, energy = data[:, :, :2], data[:, :, 2], data[:, :, 3]
    total = energy.sum(1, keepdim=True)
    w = energy / total
    soft, hard = _sqrt_zero_slope(w), w.square()
    soft = soft / soft.sum(1, keepdim=True)
    hard = hard / hard.sum(1, keepdim=True)
    layer, n_layers = depth_indices(depth, shower_frame)
    bands = SHOWER_DEPTH_REGIONS if shower_frame else BANDS
    transverse_unit = ANGLE_UNIT_RAD if shower_frame else 50
    ids = torch.arange(count, device=device)
    layer_ids = (n_layers*ids[:, None]+layer).ravel()
    def pool(value):
        out = value.new_zeros(count*n_layers)
        return out.index_add_(0, layer_ids, value.ravel()).reshape(count, n_layers)
    mass = pool(w)
    norm = mass.clamp_min(1e-20)
    soft_norm = pool(soft).clamp_min(1e-20)
    hard_norm = pool(hard).clamp_min(1e-20)
    depth_bins = torch.arange(n_layers, device=device)
    def weighted_center(q, fallback):
        weight = q.sum(1)
        center = (q[:, :, None]*xy).sum(1)/weight.clamp_min(1e-20)[:, None]
        return torch.where(weight[:, None]>1e-12, center, fallback)
    def distance(center):
        return angular_distance_squared(xy, center[:, None])

    radius2 = xy.square().sum(-1)
    total_energy = total[:, 0]
    energy_radius = transverse_unit * total_energy * (w * _sqrt_zero_slope(radius2)).sum(1)
    nseed = min(24, width)
    leading = xy[:, :nseed]
    leading_distance2 = angular_distance_squared(xy[:, None], leading[:, :, None])
    result = []
    for scale in SCALES:
        s2 = scale*scale
        residual = w * radius2 / (s2 + radius2)
        remote = torch.argsort(residual, dim=1, descending=True, stable=True)[:, :nseed]
        remote_roots = xy[ids[:, None], remote]
        candidates = torch.cat((leading, remote_roots), dim=1)
        valid = torch.cat((w[:, :nseed], w[ids[:, None], remote]), dim=1) > 0
        # At most 48 detector roots interact with every retained cluster cell
        dr2 = torch.cat((leading_distance2, angular_distance_squared(xy[:, None], remote_roots[:, :, None])), dim=1)
        if dr2.requires_grad:
            kernel = (1 + dr2/s2).reciprocal().square()
        else:
            kernel = dr2.div_(s2).add_(1).reciprocal_().square_()
        first = (kernel @ w.unsqueeze(-1)).squeeze(-1).masked_fill(~valid, -1).argmax(1)
        k1 = kernel[ids, first]
        residual_weight = w * (1-k1)
        second = (kernel @ residual_weight.unsqueeze(-1)).squeeze(-1).masked_fill(~valid, -1).argmax(1)
        k2 = kernel[ids, second]
        root1, root2 = candidates[ids, first], candidates[ids, second]
        # One energy-barycentre update inside each half of the two-root graph
        share = k2 / (k1+k2).clamp_min(1e-20)
        root1 = weighted_center(w*k1*(1-share), root1)
        root2 = weighted_center(w*k2*share, root2)
        d1, d2 = distance(root1), distance(root2)
        k1, k2 = (1+d1/s2).reciprocal().square(), (1+d2/s2).reciprocal().square()
        share = k2 / (k1+k2).clamp_min(1e-20)
        gain = k2*(1-k1)
        overlap = k1*k2
        bridge = (1+distance((root1+root2)/2)/s2).reciprocal().square()
        delta = root2-root1
        opening = delta.norm(dim=1)
        second_energy = (w*share).sum(1)
        balance = (second_energy*(1-second_energy)).clamp_min(0)
        def mean(value):
            return (w*value).sum(1)
        energy_opening = transverse_unit * total_energy * opening
        tau21 = mean(_sqrt_zero_slope(torch.minimum(d1, d2))) / mean(_sqrt_zero_slope(d1)).clamp_min(1e-12)
        result.extend((opening, delta[:, 0].abs(), delta[:, 1].abs(), energy_opening * _sqrt_zero_slope(balance),
                       balance, mean(k1), mean(k2), mean(gain), mean(overlap), mean(bridge), (hard*bridge).sum(1),
                       (soft*gain).sum(1), tau21, energy_radius))
        core1, core2 = pool(w*k1), pool(w*k2)
        gain2, assigned = pool(w*gain), pool(w*share)
        offset = (xy-root2[:, None]) / (1+d2/s2).sqrt()[:, :, None]
        result.extend((mass, core1/norm, core2/norm, gain2/norm, pool(w*overlap)/norm,
                       pool(w*overlap.square())/norm, assigned/norm, pool(w*bridge)/norm,
                       pool(w*share*d2/(s2+d2))/norm, pool(soft*gain)/soft_norm,
                       pool(hard*gain)/hard_norm,
                       pool(w*k2*offset[:, :, 0])/core2.clamp_min(1e-20), pool(w*k2*offset[:, :, 1])/core2.clamp_min(1e-20)))
        for lo, hi in bands:
            region = slice(lo, hi)
            z = depth_bins[region]
            charge = mass[:, region].sum(1)
            fraction = assigned[:, region].sum(1)/charge.clamp_min(1e-20)
            p1 = core1[:, region]/core1[:, region].sum(1, keepdim=True).clamp_min(1e-20)
            p2 = core2[:, region]/core2[:, region].sum(1, keepdim=True).clamp_min(1e-20)
            mu1, mu2 = (p1*z).sum(1), (p2*z).sum(1)
            result.append(fraction)
            if not shower_frame or (lo, hi) != (0, n_layers):
                result.extend((gain2[:, region].sum(1)/charge.clamp_min(1e-20),
                               energy_opening * _sqrt_zero_slope((fraction * (1 - fraction)).clamp_min(0))))
            variance = (p2*(z-mu2[:, None]).square()).sum(1)
            result.extend((mu1, mu2, _sqrt_zero_slope(variance), p2.square().sum(1)))
    return torch.column_stack(result).float()
