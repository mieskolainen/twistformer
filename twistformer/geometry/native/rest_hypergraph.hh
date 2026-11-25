// Durham subjets and three-body decay hyperedges in the jet rest frame
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include <array>

namespace geometry {

constexpr int rest_axis_fields = 8, rest_candidate_fields = 14;
constexpr int rest_snapshot_fields = 1 + 3 * rest_axis_fields + rest_candidate_fields;
constexpr int rest_graph_fields = 4 * rest_snapshot_fields;

namespace rest_graph {

struct Axis {
    Momentum p;
    double nx, ny, nz, leading, energy2, lab_pt, split;
    int count;
};

inline Axis axis(Momentum p, double leading, double energy2, double lab_pt, double split, int count) {
    const double length = std::sqrt(p.modp2());
    const double norm = length > 0. ? 1. / length : 0.;
    return {p, p.x*norm, p.y*norm, p.z*norm, leading, energy2, lab_pt, split, count};
}

inline double distance(const Axis &a, const Axis &b) {
    const double energy = std::min(a.p.energy, b.p.energy);
    const double cosine = a.nx*b.nx + a.ny*b.ny + a.nz*b.nz;
    return 2. * energy * energy * std::clamp(1. - cosine, 0., 2.);
}

// Resolve the W pair and bachelor by a two-mass fit to each unordered triplet
inline void snapshot(const std::vector<Axis> &axes, const std::vector<int> &active,
                     double scale, double mass, double total_pt, const double *fit, float *out) {
    std::array<const Axis *, 8> sorted{};
    const int size = active.size();
    for (int i = 0; i < size; ++i) sorted[i] = &axes[active[i]];
    std::sort(sorted.begin(), sorted.begin() + size, [](auto a, auto b) { return a->p.energy > b->p.energy; });
    *out++ = std::sqrt(scale);
    for (int i = 0; i < 3; ++i) {
        const auto &a = *sorted[i];
        const double energy = std::max(a.p.energy, 1.e-12);
        for (double v : {energy/mass, a.p.m(), double(a.count), a.leading/energy, a.energy2/(energy*energy),
                         a.lab_pt/total_pt, std::sqrt(a.split), a.p.m()/energy}) *out++ = v;
    }
    double pairs[8][8]{};
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < i; ++j)
            pairs[i][j] = pairs[j][i] = (sorted[i]->p + sorted[j]->p).m();
    double best = std::numeric_limits<double>::infinity(), second = best, support = 0.;
    std::array<double, rest_candidate_fields> values{};
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < i; ++j) {
            const auto pair = sorted[i]->p + sorted[j]->p;
            const double w = pairs[i][j], dw = (w - fit[1])/fit[3];
            for (int k = 0; k < size; ++k) {
                if (k == i || k == j) continue;
                const auto &b = *sorted[k];
                const auto parent = pair + b.p;
                const double m = parent.m(), dt = (m - fit[0])/fit[2], chi2 = dt*dt + dw*dw;
                support += 1. / ((1. + dt*dt)*(1. + dw*dw));
                if (chi2 < best) {
                    second = best;
                    best = chi2;
                    const double hi = std::max(pairs[i][k], pairs[j][k]), lo = std::min(pairs[i][k], pairs[j][k]);
                    values = {chi2, 0., 0., m, w, hi, lo, (hi*hi-lo*lo)/std::max(1.e-12, m*m-w*w),
                              parent.energy/mass, pair.energy/mass, b.p.m(), double(b.count),
                              b.leading/std::max(1.e-12, b.p.energy), b.lab_pt/total_pt};
                } else {
                    second = std::min(second, chi2);
                }
            }
        }
    }
    values[1] = second - best;
    values[2] = support;
    for (double value : values) *out++ = value;
}

} // namespace rest_graph

// Cache the minimum Durham distance in each lower triangular row
inline void rest_hypergraph(const JetList &input, const double *fit, float *&output) {
    using namespace rest_graph;
    float *rows = output;
    std::fill_n(rows, rest_graph_fields, 0.f);
    output += rest_graph_fields;
    Momentum root;
    double total_pt = 0.;
    for (const auto &p : input) {
        root = root + p;
        total_pt += p.pt();
    }
    const double mass = root.m();
    if (input.size() < 3 || mass < 1.e-4) return;
    const int n = input.size();
    // Retain storage across events on each worker
    thread_local std::vector<Axis> axes;
    thread_local std::vector<double> distances, minimum;
    thread_local std::vector<int> nearest, active;
    axes.clear();
    axes.reserve(n);
    distances.resize(static_cast<std::size_t>(n)*n);
    minimum.assign(n, std::numeric_limits<double>::infinity());
    nearest.assign(n, -1);
    active.resize(n);
    std::iota(active.begin(), active.end(), 0);
    for (int i = 0; i < n; ++i) {
        Momentum p = input[i];
        p.unboost(root);
        axes.push_back(axis(p, p.energy, p.energy*p.energy, input[i].pt(), 0., 1));
        for (int j = 0; j < i; ++j) {
            const double d = distance(axes[i], axes[j]);
            distances[static_cast<std::size_t>(i)*n+j] = d;
            if (d < minimum[i]) { minimum[i] = d; nearest[i] = j; }
        }
    }
    while (active.size() >= 3) {
        int hi = active.front();
        for (int i : active) if (minimum[i] < minimum[hi]) hi = i;
        const int lo = nearest[hi];
        const double d = minimum[hi];
        constexpr std::array<int, 4> sizes{3, 4, 6, 8};
        for (int v = 0; v < 4; ++v) {
            if (static_cast<int>(active.size()) == sizes[v])
                snapshot(axes, active, d, mass, total_pt, fit, rows + v*rest_snapshot_fields);
        }
        if (active.size() == 3) break;
        const auto &a = axes[lo], &b = axes[hi];
        axes[lo] = axis(a.p+b.p, std::max(a.leading,b.leading), a.energy2+b.energy2,
                        a.lab_pt+b.lab_pt, d, a.count+b.count);
        active.erase(std::lower_bound(active.begin(), active.end(), hi));
        for (int i : active) {
            if (i == lo) continue;
            const int row = std::max(i, lo), column = std::min(i, lo);
            distances[static_cast<std::size_t>(row)*n+column] = distance(axes[i], axes[lo]);
        }
        for (int i : active) {
            if (i < lo) continue;
            const auto *row = distances.data() + static_cast<std::size_t>(i)*n;
            if (i == lo || nearest[i] == lo || nearest[i] == hi) {
                minimum[i] = std::numeric_limits<double>::infinity();
                nearest[i] = -1;
                for (int j : active) {
                    if (j >= i) break;
                    if (row[j] < minimum[i]) { minimum[i] = row[j]; nearest[i] = j; }
                }
            } else if (row[lo] < minimum[i]) {
                minimum[i] = row[lo];
                nearest[i] = lo;
            }
        }
    }
}

} // namespace geometry
