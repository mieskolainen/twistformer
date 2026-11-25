# Torch graph descriptors for ECAL showers
#
# mikael.mieskolainen@cern.ch (2026)

from functools import partial

import torch

from .axis_graphs import AXIS_NAMES, axis_graph_rows, axis_names
from .energy_scaled_graphs import ENERGY_SCALED_GRAPH_NAMES, SHOWER_ENERGY_SCALED_NAMES, energy_scaled_graph_rows
from .graph_inputs import RADIAL_GRID_STEP, SHOWER_DEPTH_EDGES_MM, SHOWER_DEPTH_REGIONS
from .harmonic_graphs import HARMONIC_NAMES, harmonic_names, harmonic_rows
from .lobe_graphs import LOBE_GRAPH_NAMES, LOBE_TRANSPORT_NAMES, lobe_graph_rows
from .profile_graphs import PROFILE_GRAPH_NAMES, profile_graph_names, profile_graph_rows
from .sampling_rank_graphs import SAMPLING_RANK_NAMES, radial_and_sampling_rank_rows, sampling_rank_rows
from .spectral_depth_graphs import SPECTRAL_DEPTH_NAMES, spectral_depth_rows
from .transport_graphs import SHOWER_TRANSPORT_NAMES, TRANSPORT_NAMES, transport_rows


SUMMARY = ("log_hit_count", "log_hit_energy", "leading_fraction", "sum_fraction_squared", "mean_depth")
BASES = {
    "harmonic_graphs": SUMMARY + HARMONIC_NAMES,
    "energy_scaled_graphs": ENERGY_SCALED_GRAPH_NAMES,
    "projected_profiles": PROFILE_GRAPH_NAMES,
    "axis_graphs": AXIS_NAMES,
    "transport_graphs": TRANSPORT_NAMES,
}
KERNELS = {
    "harmonic_graphs": harmonic_rows,
    "energy_scaled_graphs": energy_scaled_graph_rows,
    "projected_profiles": profile_graph_rows,
    "axis_graphs": axis_graph_rows,
    "transport_graphs": transport_rows,
}
RADIUS_KERNELS = {"energy_scaled_graphs", "transport_graphs"}

# All bases use the same column order for training and prediction
BATCH_BASES = {name: (name,) for name in BASES}
BATCH_BASES["energy_scaled_profile_graphs"] = ("harmonic_graphs", "projected_profiles", "energy_scaled_graphs")
BATCH_BASES["shower_shape_graphs"] = (*BATCH_BASES["energy_scaled_profile_graphs"], "axis_graphs", "transport_graphs")

SHOWER_SUMMARY = ("depth_rms", "log_hit_energy", "theta_rms", "phi_rms", "mean_abs_depth")
N_DEPTH_BINS = len(SHOWER_DEPTH_EDGES_MM) + 1
SHOWER_FRAME_COMPONENTS = {
    "harmonic_graphs": (harmonic_rows, SHOWER_SUMMARY + harmonic_names(N_DEPTH_BINS)),
    "projected_profiles": (profile_graph_rows, profile_graph_names(
        (*((k, k + 1) for k in range(N_DEPTH_BINS)), *SHOWER_DEPTH_REGIONS), N_DEPTH_BINS)),
    "energy_scaled_graphs": (energy_scaled_graph_rows, SHOWER_ENERGY_SCALED_NAMES),
    "axis_graphs": (axis_graph_rows, axis_names(SHOWER_DEPTH_REGIONS, N_DEPTH_BINS, shower_frame=True)),
    "transport_graphs": (transport_rows, SHOWER_TRANSPORT_NAMES),
}
SHOWER_FRAME_KERNELS = {}
for component, (kernel, names) in SHOWER_FRAME_COMPONENTS.items():
    basis = f"shower_frame_{component}"
    SHOWER_FRAME_KERNELS[basis] = partial(kernel, shower_frame=True)
    BASES[basis] = tuple(f"shower_frame_{name.replace('_L', '_D')}" for name in names)
    BATCH_BASES[basis] = (basis,)
BATCH_BASES["shower_frame_profile_graphs"] = (
    "shower_frame_harmonic_graphs", "shower_frame_projected_profiles", "shower_frame_energy_scaled_graphs",
)
BATCH_BASES["shower_frame_shape_graphs"] = (
    *BATCH_BASES["shower_frame_profile_graphs"], "shower_frame_axis_graphs", "shower_frame_transport_graphs",
)
RADIAL_SHOWER_BASIS = "shower_frame_radial_shape_graphs"
RADIAL_ENERGY_BASIS = "shower_radial_energy_scaled_graphs"
RADIAL_COMPONENTS = ("shower_radial_harmonic_graphs", "shower_radial_projected_profiles", RADIAL_ENERGY_BASIS)
for basis in RADIAL_COMPONENTS:
    component = basis.removeprefix("shower_radial_")
    kernel, names = SHOWER_FRAME_COMPONENTS[component]
    if component != "energy_scaled_graphs":
        kernel = partial(kernel, angular_grid_step=RADIAL_GRID_STEP)
    SHOWER_FRAME_KERNELS[basis] = partial(kernel, shower_frame=True)
    BASES[basis] = tuple(f"shower_radial_{name.replace('_L', '_D')}" for name in names)
    BATCH_BASES[basis] = (basis,)
BATCH_BASES[RADIAL_SHOWER_BASIS] = (*BATCH_BASES["shower_frame_shape_graphs"], *RADIAL_COMPONENTS)

SPECTRAL_DEPTH_BASIS = "shower_radial_spectral_depth"
SPECTRAL_SHOWER_BASIS = "shower_frame_spectral_shape_graphs"
SHOWER_FRAME_KERNELS[SPECTRAL_DEPTH_BASIS] = spectral_depth_rows
BASES[SPECTRAL_DEPTH_BASIS] = SPECTRAL_DEPTH_NAMES
BATCH_BASES[SPECTRAL_DEPTH_BASIS] = (SPECTRAL_DEPTH_BASIS,)
BATCH_BASES[SPECTRAL_SHOWER_BASIS] = (*BATCH_BASES[RADIAL_SHOWER_BASIS], SPECTRAL_DEPTH_BASIS)

SAMPLING_RANK_BASIS = "shower_radial_sampling_rank"
LOBE_GRAPH_BASIS = "shower_frame_lobe_graphs"
LOBE_TRANSPORT_BASIS = "shower_frame_lobe_transport"
TWISTGRAPH_BASIS = "twistgraph"
SHOWER_FRAME_KERNELS[SAMPLING_RANK_BASIS] = sampling_rank_rows
BASES[SAMPLING_RANK_BASIS] = SAMPLING_RANK_NAMES
BATCH_BASES[SAMPLING_RANK_BASIS] = (SAMPLING_RANK_BASIS,)
SHOWER_FRAME_KERNELS[LOBE_GRAPH_BASIS] = lobe_graph_rows
BASES[LOBE_GRAPH_BASIS] = LOBE_GRAPH_NAMES
BATCH_BASES[LOBE_GRAPH_BASIS] = (LOBE_GRAPH_BASIS,)
SHOWER_FRAME_KERNELS[LOBE_TRANSPORT_BASIS] = partial(lobe_graph_rows, include_lobes=False, include_transport=True)
BASES[LOBE_TRANSPORT_BASIS] = LOBE_TRANSPORT_NAMES
BATCH_BASES[LOBE_TRANSPORT_BASIS] = (LOBE_TRANSPORT_BASIS,)
BATCH_BASES[TWISTGRAPH_BASIS] = (*BATCH_BASES[SPECTRAL_SHOWER_BASIS], SAMPLING_RANK_BASIS)
RADIAL_SHOWER_BASES = {
    *RADIAL_COMPONENTS, RADIAL_SHOWER_BASIS, SPECTRAL_DEPTH_BASIS, SPECTRAL_SHOWER_BASIS,
    SAMPLING_RANK_BASIS, TWISTGRAPH_BASIS,
}
SHOWER_FRAME_BASES = {
    *SHOWER_FRAME_KERNELS, "shower_frame_profile_graphs", "shower_frame_shape_graphs",
    *RADIAL_SHOWER_BASES,
}

for name, components in BATCH_BASES.items():
    if len(components) > 1:
        BASES[name] = tuple(field for component in components for field in BASES[component])

# Accept the old names for these Torch bases
for old_basis, basis in (
    ("mass_graphs", "energy_scaled_graphs"),
    ("mass_profile_graphs", "energy_scaled_profile_graphs"),
    ("mass_axis_transport_graphs", "shower_shape_graphs"),
):
    BASES[old_basis] = BASES[basis]
    BATCH_BASES[old_basis] = BATCH_BASES[basis]


def shower_graph_components(basis, groups=None):
    """
    Select graph groups in the canonical column order of the basis

    Args:
        basis:  Graph basis or None for raw cell inputs
        groups:  Selected component names or None for the default basis
    Returns:
        components:  Ordered components to compute
    """
    if basis is None:
        if groups is not None:
            raise ValueError("shower_groups requires a shower_basis")
        return ()
    if not isinstance(basis, str) or basis not in BATCH_BASES:
        raise ValueError(f"Unsupported pi0gamma shower_basis: {basis}")
    components = BATCH_BASES[basis]
    if groups is None:
        return components
    if basis == TWISTGRAPH_BASIS:
        components = (*components, LOBE_GRAPH_BASIS, LOBE_TRANSPORT_BASIS)
    if not isinstance(groups, list) or not groups or any(not isinstance(name, str) for name in groups):
        raise ValueError("shower_groups must be a nonempty list of group names or null")
    selected = set(groups)
    if len(selected) != len(groups):
        raise ValueError("shower_groups must contain unique group names")
    if selected.difference(components):
        raise ValueError(f"shower_groups for {basis} must use {', '.join(components)}")
    return tuple(name for name in components if name in selected)


def descriptor_rows(showers, basis, device="cpu", inner_radius_mm=None):
    """
    Reduce a batch of ECAL showers to a named graph expansion

    Args:
        showers:  Hit arrays in source order, with projective and radial views for radial bases
        basis:  Named graph basis or ordered tuple of components
        device:  Device for the tensor contractions
        inner_radius_mm:  ECAL radius for the original detector-frame bases
    Returns:
        result:  Float32 tensor with one row per shower on the selected device
    """
    components = BATCH_BASES[basis] if isinstance(basis, str) else basis
    projective_hits = radial_hits = showers
    paired_frames = showers[0].shape[1] == 8
    if not paired_frames and any(component in RADIAL_SHOWER_BASES for component in components):
        raise ValueError("Radial shower graphs require both coordinate views")
    if paired_frames:
        projective_hits = [hits[:, :4] for hits in showers]
        radial_hits = [hits[:, 4:] for hits in showers]
    shared = {}
    if RADIAL_ENERGY_BASIS in components and SAMPLING_RANK_BASIS in components:
        radial, ranked = radial_and_sampling_rank_rows(radial_hits, device)
        shared = {RADIAL_ENERGY_BASIS: radial, SAMPLING_RANK_BASIS: ranked}
    if LOBE_GRAPH_BASIS in components and LOBE_TRANSPORT_BASIS in components:
        rows = lobe_graph_rows(projective_hits, device, include_transport=True)
        lobes, transport = rows.split((len(LOBE_GRAPH_NAMES), len(LOBE_TRANSPORT_NAMES)), dim=1)
        shared.update({LOBE_GRAPH_BASIS: lobes, LOBE_TRANSPORT_BASIS: transport})
    fields = []
    for component in components:
        if component in shared:
            fields.append(shared[component])
            continue
        shower_kernel = SHOWER_FRAME_KERNELS.get(component)
        if shower_kernel is not None:
            hits = radial_hits if component in RADIAL_SHOWER_BASES else projective_hits
            fields.append(shower_kernel(hits, device=device))
            continue
        kernel = KERNELS[component]
        if component in RADIUS_KERNELS:
            rows = kernel(projective_hits, device=device, inner_radius_mm=inner_radius_mm)
        else:
            rows = kernel(projective_hits, device=device)
        fields.append(rows)
    return fields[0] if len(fields) == 1 else torch.cat(fields, dim=1)
