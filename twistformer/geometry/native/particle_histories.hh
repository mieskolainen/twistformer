// Multiplicity and transverse momentum flow in learned constituent partitions
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include <utility>
#include "particle_messages.hh"
#include "particle_basis.hh"

namespace geometry {

constexpr int particle_histogram_fields = 16*16*2;

// Compile each trained cut to a fixed coordinate and threshold
template<int Tree, int Node = 0>
inline int particle_leaf(const ParticleMessage &row) {
    constexpr auto cut = particle_basis::trees[Tree][Node];
    if constexpr (cut.feature < 0) return cut.left;
    else return row[cut.feature] < cut.threshold ? particle_leaf<Tree, cut.left>(row)
                                                  : particle_leaf<Tree, cut.right>(row);
}

template<std::size_t... Trees>
inline void fill_particle_leaves(float *output, const ParticleMessage &row, std::index_sequence<Trees...>) {
    auto deposit = [&](int bin) {
        output[bin] += 1.f;
        output[bin+1] += row[1];
    };
    (deposit(32*Trees + 2*particle_leaf<Trees>(row)), ...);
}

inline void particle_histories(const JetList &input, const ClusterTree &tree, const double *fit, float *&output) {
    std::fill_n(output, particle_histogram_fields, 0.f);
    particle_messages(input, tree, fit, [&](int, const ParticleMessage &row) {
        fill_particle_leaves(output, row, std::make_index_sequence<16>{});
    });
    output += particle_histogram_fields;
}
} // namespace geometry
