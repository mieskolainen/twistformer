# Harmonic multigraphs and mixed-layer shower hyperedges
#
# mikael.mieskolainen@cern.ch (2026)

from itertools import combinations

import torch

from twistformer.geometry.graph_polynomials import harmonic_products, point_tensor, unit_phase
from .graph_inputs import ANGULAR_GRID_STEP, N_LAYERS, depth_indices, oriented_core

# Transverse wave vectors in inverse coordinate units
WAVES = tuple((k * a, k * b) for k in (1.5, 3., 6.)
              for a, b in ((1., 0.), (0., 1.), (0.7, 0.7), (0.7, -0.7)))
MARKS = ("energy", "sqrt_energy")
MOMENTS = ("x", "y", "xx", "xy", "yy", "xxx", "xxy", "xyy", "yyy", "xxxx", "xxyy", "yyyy")
BANDS = ("all", "core", "inner")

def harmonic_names(n_layers):
    """
    Name harmonic graph terms in depth-bin order

    Args:
        n_layers:  Number of depth bins
    Returns:
        names:  Harmonic feature columns
    """
    layers = tuple(range(n_layers))
    pairs = tuple(combinations(layers, 2))
    return (
        tuple(f"patch_L{l}_x{x}_y{y}" for l in layers for x in range(-3, 4) for y in range(-3, 4))
        + tuple(f"{mark}_{band}_{quantity}_L{l}" for mark in MARKS for band in BANDS
                for quantity in ("mass", *MOMENTS) for l in layers)
        + tuple(f"{mark}_{quantity}_L{l}_k{k}" for mark in MARKS
                for quantity, bins, count in (("edge", layers, 12), ("triangle_re", layers, 8),
                                              ("triangle_im", layers, 8), ("adjacent_re", layers[:-1], 12),
                                              ("adjacent_im", layers[:-1], 12), ("depth_wedge", layers[:-2], 12))
                for l in bins for k in range(count))
        + tuple(f"{mark}_{quantity}_L{i}_L{j}" for mark in MARKS
                for quantity in ("centroid_edge", "tensor_edge", "double_edge") for i, j in pairs)
    )


HARMONIC_NAMES = harmonic_names(N_LAYERS)


def harmonic_rows(showers, device="cpu", *, shower_frame=False, angular_grid_step=ANGULAR_GRID_STEP):
    """
    Reduce shower moments and phase closures on the selected device

    Args:
        showers:  Local cell coordinates and log energies, in event order
        device:  Device for the harmonic contractions
        shower_frame:  Use shower-relative depth bins and a common angular patch grid
        angular_grid_step:  Patch pitch in transverse units for shower-frame rows
    Returns:
        rows:  Shower summaries followed by HARMONIC_NAMES in card order
    """
    summaries, coordinates, fractions, depths = [], [], [], []
    for shower in showers:
        hits = point_tensor(shower)
        log_energy = hits[:, 3]
        maximum = log_energy.max()
        energy = (log_energy - maximum).exp()
        total = energy.sum()
        fraction = energy / total
        depth = hits[:, 2]
        if shower_frame:
            rms = (fraction @ hits[:, :3].square()).sqrt()
            summary = (rms[2], maximum + total.log(), rms[0], rms[1], fraction @ depth.abs())
        else:
            summary = (hits.new_tensor(len(hits)).log(), maximum + total.log(), fraction.max(),
                       fraction @ fraction, fraction @ depth)
        summaries.append(torch.stack(summary))
        coordinates.append(oriented_core(hits[:, :2], depth, fraction, (1.6, .6, .35)))
        fractions.append(fraction)
        depths.append(depth)

    count = len(showers)
    depth = torch.cat(depths).to(device)
    counts = torch.tensor([len(shower) for shower in showers], device=device)
    event = torch.repeat_interleave(torch.arange(count, device=device), counts, output_size=len(depth))
    layer, n_layers = depth_indices(depth, shower_frame)
    index = event * n_layers + layer
    weight = torch.cat(fractions).to(device)
    x, y = torch.cat(coordinates).to(device).T

    def pool(values):
        """
        Sum hit values by shower and sampling
        """
        shape = values.shape[1:]
        output = values.new_zeros((count * n_layers, *shape))
        return output.index_add_(0, index, values).reshape(count, n_layers, *shape)

    if shower_frame:
        theta_pitch = phi_pitch = angular_grid_step
    else:
        theta_pitch = torch.where(layer == 1, weight.new_tensor(5.333 / 50), weight.new_tensor(21.33 / 50))
        phi_pitch = 17.78 / 50
    ix, iy = (x / theta_pitch).round().long(), (y / phi_pitch).round().long()
    inside = (ix.abs() <= 3) & (iy.abs() <= 3)
    cell = 49 * index[inside] + 7 * (ix[inside] + 3) + iy[inside] + 3
    patch = weight.new_zeros(count * n_layers * 49).index_add_(0, cell, weight[inside])
    output = [torch.stack(summaries).to(device), patch.reshape(count, -1)]

    radius2 = x.square() + y.square()
    inverse = (.16 + radius2).rsqrt()
    u, v = x * inverse, y * inverse
    uu, uv, vv = u*u, u*v, v*v
    monomials = torch.stack((u, v, uu, uv, vv, uu*u, uu*v, vv*u, vv*v, uu*uu, uu*vv, vv*vv), dim=1)
    windows = ((1 + radius2/.64).square().reciprocal(), (1 + radius2/.09).square().reciprocal())
    full_moments = []
    for mark in (weight, weight.sqrt()):
        mass = pool(mark)
        total = mass.sum(1, keepdim=True)
        q = mark / mass.reshape(-1)[index]
        moments = pool(q[:, None] * monomials)
        full_moments.append((mass, q, moments))
        output.extend((mass / total, moments.transpose(1, 2)))
        for window in windows:
            charge = mark * window
            mass = pool(charge)
            q = charge / mass.reshape(-1)[index]
            output.extend((mass / total, pool(q[:, None] * monomials).transpose(1, 2)))

    waves = weight.new_tensor(WAVES)
    phase = x[:, None] * waves[:, 0] + y[:, None] * waves[:, 1]
    harmonic = unit_phase(phase)
    for _, q, _ in full_moments:
        f = pool(q[:, None] * harmonic)
        edge, triangle = harmonic_products(f, slice(0, 8), slice(4, None))
        adjacent = f[:, :-1] * f[:, 1:].conj()
        wedge = f[:, :-2] * f[:, 1:-1].conj().square() * f[:, 2:]
        output.extend((edge, triangle.real, triangle.imag, adjacent.real, adjacent.imag, wedge.real))

    i, j = torch.triu_indices(n_layers, n_layers, offset=1, device=device)
    for mass, _, moments in full_moments:
        a, b, aa, ab, bb = moments[..., :5].unbind(-1)
        xx, cross, yy = aa - a*a, ab - a*b, bb - b*b
        valid = (mass[:, i] > 0) & (mass[:, j] > 0)
        distance = (a[:, i] - a[:, j]).square() + (b[:, i] - b[:, j]).square()
        alignment = (xx[:, i]-yy[:, i]) * (xx[:, j]-yy[:, j]) + 4*cross[:, i]*cross[:, j]
        square = xx[:, i]*xx[:, j] + 2*cross[:, i]*cross[:, j] + yy[:, i]*yy[:, j]
        output.extend((distance * valid, alignment * valid, square * valid))
    return torch.column_stack([field.reshape(count, -1) for field in output]).float()
