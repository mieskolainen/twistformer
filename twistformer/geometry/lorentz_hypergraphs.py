# Ordered observables for native Lorentz hypergraphs and simplicial complexes
#
# mikael.mieskolainen@cern.ch (2026)

# Jet rest-frame Gram contractions and transverse radiation transport
VERTEX_FIELDS = (
    "energy", "pt", "mass", "leaves", "jet_mass", "particles", "dy", "radius", "leading", "sum_z2",
    "width", "last_kt",
    *(f"flow{s}_{field}" for s in range(3) for field in ("count", "energy", "mass2", "chord")),
    "flow_volume", "flow_dot", "w_support", "w_energy", "w_neighbor_energy", "w_bivector",
    *(f"{role}_{field}" for role in ("b", "q") for field in
      ("support", "maximum", "energy", "gram3", "trivector", "flow", "leaves", "asymmetry")),
    "top_mass", "w_mass", "w_gram3", "w_plane", "bq_low", "bq_high", "w_submass", "soft_energy",
    "volume4_sum", "volume4_max", "volume4_top", "volume4_angle", "best_pair", "jet_pt", "abs_y",
    "flow_plane", "near_b", "near_q", "w_b", "w_q", "w_volume4", "w_flow_plane",
)
TRANSPORT_NAMES = tuple(
    f"simplex_sheaf_{scale}_{weight}_{field}" for scale in range(3) for weight in ("energy", "w")
    for field in ("long_energy", "normal_energy", "long_product", "normal_product", "pull_difference",
                  "section_area", "long_sum", "max_energy")
) + tuple(f"simplex_holonomy_{weight}_{field}" for weight in ("energy", "count")
          for field in ("sum", "square", "max", "section0", "section1", "section2"))
SIMPLEX_NAMES = TRANSPORT_NAMES + tuple(
    f"simplex_{field}_{projection}" for field in VERTEX_FIELDS for projection in ("mean", "max", "v1", "v2", "v3")
)

GRAM_NAMES = tuple(
    f"gram_{channel}_{low}_{high}_bin{band}" for channel in ("energy", "pt", "local")
    for high in range(5) for low in range(high+1) for band in range(7)
) + tuple(f"gram_wedge_energy{energy}_{low}_{high}" for energy in range(5)
          for high in range(7) for low in range(high+1))
COMPLEX_NAMES = tuple(
    f"complex_{axes}_{metric}_{field}" for axes in ("kt", "rest") for metric in ("angle", "mass")
    for field in ("scale",
        *(f"h{dim}_{stat}" for dim in range(3) for stat in
          ("count", "total", "square", "longest", "second", "birth", "death", "betti10", "betti25", "betti50", "betti75")),
        *(f"d{dim}_r{radius}_{stat}" for dim in range(1, 4) for radius in (25, 50, 75, 100)
          for stat in ("count", "volume", "angular_volume")))
)
CASCADE_NAMES = tuple(
    f"cascade_{field}_o{order}_{i}_{j}"
    for field, ni, nj in (("split", 6, 5), ("adjacent", 6, 6), ("planes", 5, 5))
    for order in range(3) for i in range(ni) for j in range(nj)
)
SHEAF_NAMES = tuple(
    f"sheaf_{scale}_{axis}_{field}" for scale in ("narrow", "wide") for axis in ("v1", "v2", "v3", "v4", "mean")
    for field in ("weight", "occupancy", "soft_count", "core_fraction", "share_squared", "width", "width_squared",
                  "trace", "quadrupole", "alignment", "section_norm2", "return_norm2", "return_product",
                  "laplacian_norm2", "laplacian_product", "edge_residual", "oriented_area")
)

REFERENCE_NAMES = (
    "reference_time_energy", "reference_beam_pt_fraction", "reference_longitudinal_fraction",
    "reference_time_concentration",
    *(f"reference_axis{i}_energy_share" for i in range(1, 5)),
    *(f"reference_{weight}_{field}" for weight in ("count", "energy")
      for field in ("beam_norm2", "beam_area2", "longitudinal2", "beam_area4", "longitudinal4",
                    "beam_longitudinal_product", "beam_alignment")),
    *(f"reference_pair{i}{j}_{field}" for i in range(1, 5) for j in range(i+1, 5)
      for field in ("energy_share", "time_asymmetry", "beam_area2", "longitudinal2",
                    "time_wedge_norm2", "beam_dot", "weighted_beam_area2")),
    *(f"reference_triple{i}{j}{k}_{field}" for i in range(1, 5) for j in range(i+1, 5) for k in range(j+1, 5)
      for field in ("beam_area2", "time_volume2", "weighted_time_volume2")),
)
