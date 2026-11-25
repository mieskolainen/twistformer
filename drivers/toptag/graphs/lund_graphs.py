# Lund observables for the hadronic top decay hypothesis
#
# mikael.mieskolainen@cern.ch (2026)

from contextlib import contextmanager

import torch

from twistformer.geometry.lorentz_hypergraphs import CASCADE_NAMES, COMPLEX_NAMES, GRAM_NAMES, REFERENCE_NAMES, SHEAF_NAMES, SIMPLEX_NAMES
from twistformer.geometry.native_lund import native_lund_rows
from twistformer.geometry.lund_graphs import (
    MassHypothesis, flow_names, history_names as jet_history_names, jet_rows,
)

TOP_DECAY = MassHypothesis(parent_mass=172.5, pair_mass=80.4, parent_width=25., pair_width=15.)
TOP_FIELDS = ("top_mass", "w_mass", "w_pt_fraction", "top_pt_fraction", "min_bq_mass")


def history_names(algorithm):
    """
    Name the clustering and top decay observables

    Args:
        algorithm:  ca or kt
    Returns:
        names:  Top-tagging columns in checkpoint order
    """
    return jet_history_names(algorithm, TOP_FIELDS)


LUND_NAMES = history_names("ca") + history_names("kt") + flow_names()


def lund_rows(momenta):
    """
    Reconstruct Lund splittings and hadronic top candidates

    Args:
        momenta:  Batched E, px, py, pz in GeV with zero padding
    Returns:
        rows:  Float32 tensor in LUND_NAMES order
    """
    return jet_rows(momenta, mass_hypothesis=TOP_DECAY)


# Durham jet resolution, subjet spectra and W-pair assignments for each partition
REST_NAMES = tuple(
    f"rest_n{count}_{field}"
    for count in (3, 4, 6, 8)
    for field in (
        "next_kt",
        *(f"axis{rank}_{quantity}" for rank in (1, 2, 3) for quantity in (
            "energy_fraction", "mass", "count", "leading_fraction", "energy_fraction_squared",
            "lab_pt_fraction", "last_kt", "mass_over_energy")),
        "chi2", "chi2_gap", "resonance_sum", "top_mass", "w_mass", "bq_mass_high", "bq_mass_low",
        "helicity", "top_energy_fraction", "w_energy_fraction", "b_mass", "b_count",
        "b_leading_fraction", "b_lab_pt_fraction",
    )
)

PARTITION_NAMES = tuple(
    f"partition_n{count}_{field}"
    for count in (5, 7)
    for field in ("log_w", "log_top", "log_direct_w", "log_direct_top", "log_trim_sum",
                  "log_trim_max", "direct_fraction", "best_w_mass", "mean_w_mass", "mean_b_mass",
                  "w_mass_spread", "effective_assignments", "trim_mass", "trim_pt_fraction")
)

GROOMED_NAMES = tuple(
    f"groomed_{scale}_{field}"
    for scale in ("z05", "z12", "z05_beta1")
    for field in ("mass", "pt_fraction", "count", "z", "kt", "kt1", "kt2", "kt1_mass",
                  "hard_mass", "soft_mass", "hard_kt", "resonance_sum", "chi2", "top_mass", "w_mass",
                  "b_mass", "bq_mass_low", "bq_mass_high", "top_pt_fraction", "w_pt_fraction",
                  "w_count", "b_count", "w_kt", "top_kt", "w_z")
)

PARTICLE_NAMES = tuple(
    f"particle_tree{tree}_leaf{leaf}_{field}"
    for tree in range(16) for leaf in range(16) for field in ("count", "pt_fraction")
)

COLOR_NAMES = tuple(
    f"color_{partition}_{field}"
    for partition in ("exclusive", "core")
    for field in (
        "top_mass", "w_mass", "b_mass", "q1_mass", "q2_mass", "top_pt_fraction", "bq1_mass", "bq2_mass",
        *(f"{axis}_{quantity}" for axis in ("b", "q1", "q2") for quantity in (
            "count", "pt_fraction", "share_squared", "leading_share", "width", "width_squared",
            "pull_next", "pull_previous", "core_count", "core_pt_fraction", "soft_count", "soft_pt_fraction")),
        *(f"{pair}_{quantity}" for pair in ("qq", "q2b", "bq1") for quantity in (
            "gap_count", "gap_fraction", "dipolarity", "gap_asymmetry", "ring_count", "ring_fraction",
            "antenna", "antenna_opposite")),
    )
)

DECAY_NAMES = LUND_NAMES + REST_NAMES + PARTITION_NAMES + GROOMED_NAMES + PARTICLE_NAMES + COLOR_NAMES

LORENTZ_NAMES = DECAY_NAMES + SIMPLEX_NAMES + COMPLEX_NAMES + GRAM_NAMES + CASCADE_NAMES + SHEAF_NAMES
NATIVE_NAMES = LORENTZ_NAMES + REFERENCE_NAMES


@contextmanager
def native_top_rows(features, threads):
    """
    Select compiled Lund and decay columns in model order

    Args:
        features:  Ordered detector-level feature names
        threads:   CPU workers for the jet batch
    Yields:
        project:  Batch projection for energy-first CPU four-vectors
    """
    groups = (("lund_ca_", 1), ("lund_kt_", 2), ("lund_flow_", 4), ("rest_", 8), ("partition_", 16), ("groomed_", 32),
              ("particle_", 64), ("color_", 128), ("simplex_", 256), ("complex_", 512), ("gram_", 1024), ("cascade_", 2048), ("sheaf_", 4096), ("reference_", 8192))
    selected = {group: bit for group, bit in groups if any(name.startswith(group) for name in features)}
    layout = tuple(name for name in NATIVE_NAMES if name.startswith(tuple(selected)))
    indices = torch.tensor([layout.index(name) for name in features]) if tuple(features) != layout else None
    with native_lund_rows(sum(selected.values()), TOP_DECAY, threads) as extract:
        def project(p4):
            """Project a padded jet batch in source order"""
            rows = extract(p4)
            return rows if indices is None else rows.index_select(1, indices)

        yield project
