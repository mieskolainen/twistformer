# Weighted graph polynomials on point clouds
#
# mikael.mieskolainen@cern.ch (2026)

from functools import lru_cache
from itertools import permutations, product

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .angular import angular_distance_squared


GRAPHS = ("edge", "double_edge", "wedge", "triangle")
MAX_EDGES = 6

# Number of loopless multigraphs without isolated vertices at each edge order
GRAPH_COUNTS = (1, 3, 8, 23, 66, 212)
GRAPH_NAMES = tuple(
    f"d{degree}_g{index:03d}" for degree, count in enumerate(GRAPH_COUNTS, 1)
    for index in range(count)
)


@lru_cache(None)
def vertex_orders(degrees):
    """
    Generate vertex orders within equal-degree classes

    Args:
        degrees:  Vertex degrees in input order
    Returns:
        result:  Relabelings that sort vertices by degree
    """
    groups = [tuple(i for i, d in enumerate(degrees) if d == k) for k in sorted(set(degrees))]
    return tuple(tuple(i for group in order for i in group)
                 for order in product(*(permutations(group) for group in groups)))


def canonical(edges):
    """
    Choose a unique edge tuple for an undirected multigraph

    Args:
        edges:  Edges with contiguous vertex indices
    Returns:
        result:  Canonical edge tuple
    """
    degrees = [0] * (1 + max(max(edge) for edge in edges))
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    best = None
    for order in vertex_orders(tuple(degrees)):
        labels = dict(zip(order, range(len(order))))
        candidate = tuple(sorted((min(labels[i], labels[j]), max(labels[i], labels[j])) for i, j in edges))
        if best is None or candidate < best:
            best = candidate
    return best


@lru_cache(None)
def graph_catalog():
    """
    Enumerate each graph as a multiset of connected components

    Returns:
        connected:  Canonical edge lists ordered by degree and edge tuple
        graphs:  Component indices for each feature in GRAPH_NAMES
    """
    level = [((0, 1),)]
    connected = list(level)
    for _ in range(2, MAX_EDGES + 1):
        candidates = set()
        for edges in level:
            n = 1 + max(max(edge) for edge in edges)
            for i in range(n):
                for j in range(i + 1, n + 1):
                    candidates.add(canonical((*edges, (i, j))))
        level = sorted(candidates)
        connected.extend(level)

    by_degree = [[()]]
    for degree in range(1, MAX_EDGES + 1):
        terms = []
        for i, edges in enumerate(connected):
            remaining = degree - len(edges)
            if remaining < 0:
                break
            terms.extend((i, *tail) for tail in by_degree[remaining] if not tail or i <= tail[0])
        by_degree.append(sorted(terms))
    return tuple(connected), tuple(term for terms in by_degree[1:] for term in terms)


@lru_cache(None)
def contraction_plan():
    """
    Expand each edge in the four factors of the spherical distance kernel

    Returns:
        result:  Moment powers, products and signed sums for the graph basis
    """
    connected, graphs = graph_catalog()
    radix = MAX_EDGES + 1
    powers = [(a, b, c) for c in range(radix) for b in range(radix - c)
              for a in range(radix - c - b)]
    codes = np.zeros(radix**3, dtype=int)
    for index, (a, b, c) in enumerate(powers):
        codes[a + radix*b + radix**2*c] = index

    products = {}
    indices, factors, starts = [], [], []
    choices = {}
    for edges in connected:
        degree = len(edges)
        if degree not in choices:
            colors = np.array(list(product(range(4), repeat=degree)), dtype=int)
            signs = (-1)**np.count_nonzero(colors, axis=1)
            choices[degree] = np.array([0, 1, radix, radix**2])[colors], signs / 2**degree
        colors, signs = choices[degree]
        exponents = np.zeros((len(colors), max(max(e) for e in edges) + 1), dtype=int)
        for k, (i, j) in enumerate(edges):
            exponents[:, i] += colors[:, k]
            exponents[:, j] += colors[:, k]
        terms, inverse = np.unique(np.sort(codes[exponents], axis=1), axis=0, return_inverse=True)
        coefficients = np.bincount(inverse.ravel(), weights=signs)
        starts.append(len(indices))
        for row, coefficient in zip(terms, coefficients, strict=True):
            term = tuple(row[row != 0])
            indices.append(products.setdefault(term, len(products)))
            factors.append(coefficient)

    moment_indices = np.zeros((len(products), MAX_EDGES + 1), dtype=int)
    for term, index in products.items():
        moment_indices[index, :len(term)] = term
    # Index zero supplies the unit factor for empty component slots
    component_indices = np.zeros((len(graphs), MAX_EDGES), dtype=int)
    for i, graph in enumerate(graphs):
        component_indices[i, :len(graph)] = np.asarray(graph) + 1
    return (np.asarray(powers).T, moment_indices, np.asarray(indices), np.asarray(factors),
            np.asarray(starts), component_indices)


@lru_cache(None)
def product_plan():
    """
    Reuse common moment products in the finite-rank graph contraction

    Returns:
        result:  Degree slices, prefix and terminal factors, graph term indices
    """
    _, products, indices, _, _, _ = contraction_plan()
    terms = [tuple(row[row != 0]) for row in products]
    prefixes = {term[:length] for term in terms for length in range(len(term)+1)}
    ordered = sorted(prefixes, key=lambda key: (len(key), key))
    lookup = {term: i for i, term in enumerate(ordered)}
    levels = []
    for degree in range(1, max(map(len, ordered))+1):
        block = [term for term in ordered if len(term) == degree]
        section = slice(lookup[block[0]], lookup[block[-1]]+1)
        levels.append((section, np.array([lookup[term[:-1]] for term in block]), np.array([term[-1] for term in block])))
    return len(ordered), levels, np.asarray([lookup[term] for term in terms])[indices]


def squared_distance_graphs(points, weights):
    """
    Contract squared-distance edges, double edges, wedges and triangles

    Args:
        points:   Batched point coordinates
        weights:  Normalized weights with zero weight for padded points
    Returns:
        rows:  Four graph coefficients per point cloud
    """
    radius2 = points.square().sum(-1, keepdim=True)
    ones = torch.ones_like(radius2)
    left = torch.cat((ones, radius2, points), -1)
    right = torch.cat((radius2, ones, -2 * points), -1)
    neighbor = (left * (weights[..., None] * right).sum(1, keepdim=True)).sum(-1)
    transfer = right.transpose(1, 2) @ (weights[..., None] * left)
    square = transfer @ transfer
    return torch.stack(((weights * neighbor).sum(-1), square.diagonal(dim1=1, dim2=2).sum(-1),
                        (weights * neighbor.square()).sum(-1),
                        (square * transfer.transpose(1, 2)).sum((1, 2))), -1).clamp_min(0)


def angular_flow_graphs(points, weights, radii, angular=True, *, periodic_azimuth=False):
    """
    Contract weighted Gaussian and angular-distance graphs on a point cloud

    Args:
        points:  Two angular coordinates per point
        weights:  Normalized point weights with zero weight for padding
        radii:  Gaussian edge widths in coordinate units
        angular:  Include power-law angular correlations
        periodic_azimuth:  Wrap the final coordinate over 2 pi radians
    Returns:
        rows:  Angular correlation columns for each cloud
    """
    positive = weights > 0
    multiplicity = positive.sum(1).clamp_min(1)
    dr2 = angular_distance_squared(points[:, :, None], points[:, None], periodic_azimuth=periodic_azimuth)
    fields = []
    # Matrix products sum over assignments to the graph vertices
    for radius in radii:
        kernel = (-dr2 / (2 * radius**2)).exp()
        neighbor = (kernel @ weights.unsqueeze(-1)).squeeze(-1)
        endpoint = weights * neighbor
        path = (endpoint * (kernel @ endpoint.unsqueeze(-1)).squeeze(-1)).sum(1)
        fields.extend((endpoint.sum(1), (weights * (kernel.square() @ weights.unsqueeze(-1)).squeeze(-1)).sum(1),
                       (weights * neighbor.square()).sum(1), (weights * neighbor.pow(3)).sum(1), path,
                       (neighbor * positive).sum(1) / multiplicity))
    # The trace of the cubed transfer matrix closes the three-point cycle
    if angular:
        dr = dr2.sqrt()
        for kernel in (dr.sqrt(), dr):
            transfer = kernel * weights[:, None, :]
            neighbor = transfer.sum(-1)
            edge = (weights * neighbor).sum(1)
            triangle = ((transfer @ transfer) * transfer.transpose(1, 2)).sum((1, 2))
            fields.extend((edge, (weights * neighbor.square()).sum(1), triangle, triangle / edge.pow(3).clamp_min(1e-12)))
    return torch.stack(fields, 1) if fields else weights[:, :0]


def projected_graph_rows(coordinate, weights, scales):
    """
    Contract edges and wedges from sorted transverse projections

    Args:
        coordinate:  One axis for each point in each cloud
        weights:     Normalized point weights
        scales:      Positive edge widths in coordinate units
    Returns:
        rows:  Edge and wedge for each scale
    """
    # Use the native CUDA ordering kernel before the prefix moments
    if coordinate.is_cuda:
        u, order = coordinate.topk(coordinate.shape[1], dim=1, largest=False, sorted=True)
    else:
        u, order = coordinate.sort(dim=1)
    w = weights.gather(1, order)
    u = (u - (w * u).sum(1, keepdim=True)).contiguous()
    zero = torch.zeros_like(u[:, :1])
    charge = torch.cat((zero, w.cumsum(1)), 1)
    moment = torch.cat((zero, (w*u).cumsum(1)), 1)
    low = torch.searchsorted(u, (u[..., None] - scales).flatten(1))
    high = torch.searchsorted(u, (u[..., None] + scales).flatten(1), right=True)
    shape = (u.shape[0], u.shape[1], scales.numel())
    left = charge.gather(1, low).reshape(shape)
    right = charge.gather(1, high).reshape(shape)
    distance = u[..., None] * (2 * charge[:, :-1, None] - left - right)
    distance += (moment.gather(1, high) + moment.gather(1, low)).reshape(shape) - 2 * moment[:, :-1, None]
    neighbor = (right - left - distance / scales).clamp(0, 1)
    return torch.stack(((w[..., None] * neighbor).sum(1), (w[..., None] * neighbor.square()).sum(1)), -1).flatten(1)


class SphericalGraphs(nn.Module):
    def __init__(self, graph_indices=None):
        """
        Load the shared moment products and signed sums for each multigraph

        Args:
            graph_indices:  Optional nonempty sequence of output column indices
        """
        super().__init__()
        powers, products, indices, factors, starts, components = contraction_plan()
        if graph_indices is not None:
            # Keep the connected factors and moment products of the requested graphs
            if not len(graph_indices):
                raise ValueError("Select at least one spherical graph")
            components = components[list(graph_indices)]
            connected = np.unique(components)
            connected = connected[connected > 0]
            ends = np.r_[starts[1:], len(indices)]
            terms = [np.arange(starts[i - 1], ends[i - 1]) for i in connected]
            selected = np.concatenate(terms)
            starts = np.cumsum([0] + [len(t) for t in terms[:-1]])
            needed, indices = np.unique(indices[selected], return_inverse=True)
            products, factors = products[needed], factors[selected]
            components = np.searchsorted(np.r_[0, connected], components)
        self.register_buffer("powers", torch.from_numpy(powers))
        self.register_buffer("products", torch.from_numpy(products))
        self.register_buffer("components", torch.from_numpy(components))
        self.register_buffer("indices", torch.from_numpy(indices))
        self.register_buffer("starts", torch.from_numpy(starts))
        self.register_buffer("factors", torch.from_numpy(factors))

    def forward(self, points, weights):
        """
        Evaluate all 313 multigraphs from moments of the spherical distance kernel

        Args:
            points:   Batched planar coordinates in units of the kernel scale
            weights:  Normalized point weights with zero weight for padded points
        Returns:
            rows:  Coefficients in the shared graph catalogue order
        """
        radius2 = points.square().sum(-1, keepdim=True)
        vector = torch.cat((2 * points, 1 - radius2), -1) / (1 + radius2)
        powers = [torch.ones_like(vector)]
        for _ in range(MAX_EDGES):
            powers.append(powers[-1] * vector)
        values = torch.stack(powers, dim=2)
        a, b, c = self.powers.unbind()
        moments = (weights[..., None] * values[:, :, a, 0] * values[:, :, b, 1] * values[:, :, c, 2]).sum(1)
        unit = torch.ones_like(moments[:, :1])
        moments = torch.cat((unit, moments[:, 1:]), 1)
        # Each row holds one moment product for all jets in the batch
        products = moments.T.contiguous()[self.products].prod(1)
        sums = F.embedding_bag(self.indices, products, self.starts, mode="sum", per_sample_weights=self.factors)
        connected = torch.cat((unit, sums.T.clamp(0, 1)), 1)
        return connected[:, self.components].prod(-1)


def point_tensor(value, device=None):
    """
    Convert point arrays to doubles while preserving the tensor device

    Args:
        value:   Point coordinates or weights
        device:  Optional target device
    Returns:
        tensor:  Float64 values on the requested device
    """
    if isinstance(value, np.ndarray) and any(stride < 0 for stride in value.strides):
        value = np.ascontiguousarray(value)
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def graph_polynomials(points, fraction):
    """
    Evaluate squared-distance graphs on one cloud or a batch of clouds

    Args:
        points:    Coordinates with final axes for vertices and dimensions
        fraction:  Normalized vertex weights
    Returns:
        result:  Four graph coefficients in the input array type
    """
    coords = point_tensor(points)
    weights = point_tensor(fraction, coords.device)
    single = coords.ndim == 2
    if single:
        coords, weights = coords.unsqueeze(0), weights.unsqueeze(0)
    rows = squared_distance_graphs(coords, weights)
    result = rows[0] if single else rows
    return result if torch.is_tensor(points) else result.numpy()


def projected_graphs(coordinate, fraction, scales):
    """
    Evaluate compact edges and wedges on one axis

    Args:
        coordinate:  Projected coordinates with an optional leading batch axis
        fraction:    Normalized vertex weights
        scales:      Positive kernel widths in coordinate units
    Returns:
        result:  Projected coefficients in the input array type
    """
    u = point_tensor(coordinate)
    weights = point_tensor(fraction, u.device)
    widths = point_tensor(scales, u.device)
    if widths.ndim != 1 or not widths.numel() or not torch.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("Projected graph scales must be finite and positive")
    single = u.ndim == 1
    if single:
        u, weights = u.unsqueeze(0), weights.unsqueeze(0)
    if u.shape[1]:
        rows = projected_graph_rows(u, weights, widths)
    else:
        rows = u.new_zeros((u.shape[0], 2 * widths.numel()))
    result = rows[0] if single else rows
    return result if torch.is_tensor(coordinate) else result.numpy()


@lru_cache(maxsize=8)
def spherical_kernel(device):
    """Cache the shared multigraph catalogue on the selected device"""
    return SphericalGraphs().to(device)


def spherical_graphs(points, weights):
    """
    Evaluate the complete spherical graph basis

    Args:
        points:   Planar coordinates with an optional leading batch axis
        weights:  Normalized vertex weights
    Returns:
        result:  Graph sums in the input array type and GRAPH_NAMES order
    """
    coords = point_tensor(points)
    fraction = point_tensor(weights, coords.device)
    if coords.ndim == 1 and not coords.numel():
        coords = coords.reshape(0, 2)
    single = coords.ndim == 2
    if single:
        coords, fraction = coords.unsqueeze(0), fraction.unsqueeze(0)
    if coords.shape[1]:
        rows = spherical_kernel(coords.device)(coords, fraction)
    else:
        rows = coords.new_zeros((coords.shape[0], len(GRAPH_NAMES)))
    result = rows[0] if single else rows
    return result if torch.is_tensor(points) else result.numpy()


def unit_phase(angle):
    """
    Form the complex Fourier phase exp(i angle)

    Args:
        angle:  Real phase angles in radians
    Returns:
        wave:  Unit complex amplitudes
    """
    if angle.device.type == "cpu":
        return torch.complex(angle.cos(), angle.sin())
    return torch.polar(torch.ones_like(angle), angle)


def harmonic_products(moments, fundamental, overtone):
    """
    Contract Fourier edges and phase closure triangles on the last axis

    Args:
        moments:      Complex weighted Fourier moments
        fundamental:  Indices for wave vectors k
        overtone:     Matching indices for wave vectors 2k
    Returns:
        result:  Tensors of squared magnitude and complex three-point phase closure
    """
    if isinstance(moments, np.ndarray) and any(stride < 0 for stride in moments.strides):
        moments = np.ascontiguousarray(moments)
    values = torch.as_tensor(moments)
    edge = values.abs().square()
    triangle = values[..., fundamental].square() * values[..., overtone].conj()
    return edge, triangle
