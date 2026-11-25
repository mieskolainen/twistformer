# Calorimeter energy-flow geometry
#
# mikael.mieskolainen@cern.ch (2026)

import math
import numpy as np
import torch

def validate_proxy_options(enabled, origin):
    """
    Validate the nominal production point for energy-flow proxies

    Args:
        enabled:  Enable null momentum proxies
        origin:  Fixed laboratory production point in mm
    Returns:
        None: Invalid coordinates or inactive steering raise ValueError
    """
    if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
        raise ValueError("proxy_origin must contain three finite coordinates in mm")
    if not enabled and any(v != 0 for v in origin):
        raise ValueError("proxy_origin requires momentum_proxy=true")

def hit_momenta(features, origin=(0.0, 0.0, 0.0)):
    """
    Assign null energy-flow momenta to calorimeter hits

    Args:
        features:  Unstandardized x,y,z,E hits in mm and GeV (N,4)
        origin:  Fixed laboratory production point in mm
    Returns:
        p4: Float64 proxies E,px,py,pz in GeV (N,4)
    """
    validate_proxy_options(True, origin)
    if features.ndim != 2 or features.shape[-1] != 4:
        raise ValueError("Energy-flow proxies require x,y,z,E hit features")
    hits = features.double()
    if not torch.isfinite(hits).all() or (hits[:, 3] <= 0).any():
        raise ValueError("Energy-flow proxies require finite hits and positive energy")
    direction = hits[:, :3] - hits.new_tensor(origin)
    radius = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if (radius <= 0).any():
        raise ValueError("Energy-flow direction is undefined at proxy_origin")
    energy = hits[:, 3:4]
    return torch.cat((energy, energy * (direction / radius)), dim=-1)

def gather_hits(chunks, events, momentum_proxy, origin):
    """
    Gather ragged calorimeter events and compute geometry once per batch

    Args:
        chunks:  Compact chunks in the requested event order
        events:  Event indices within each chunk
        momentum_proxy:  Attach null energy-flow geometry
        origin:  Nominal production point in mm
    Returns:
        events: Feature views, physical momenta and selected energy sums
    """
    arrays = []
    for chunk, event in zip(chunks, events, strict=True):
        start, stop = chunk.offsets[event : event + 2]
        arrays.append(chunk.features[start:stop])
    if not arrays:
        return []
    lengths = [len(array) for array in arrays]
    if not all(lengths):
        raise ValueError("Each calorimeter event must contain at least one hit")
    features = torch.from_numpy(np.concatenate(arrays))
    p4 = hit_momenta(features, origin).split(lengths) if momentum_proxy else [None] * len(arrays)
    offsets = np.cumsum([0, *lengths[:-1]])
    energies = np.add.reduceat(features[:, 3].numpy().astype(np.float64), offsets)
    return list(zip(features.split(lengths), p4, energies, strict=True))
