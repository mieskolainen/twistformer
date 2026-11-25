// Local radiation around each constituent and the exclusive three-prong decay axes
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include <array>

namespace geometry {

constexpr int particle_fields = 31;
using ParticleMessage = std::array<float, particle_fields>;

inline double minkowski(const Momentum &a, const Momentum &b) {
    return a.energy*b.energy - a.x*b.x - a.y*b.y - a.z*b.z;
}

// Visit each reconstructed particle once after a symmetric pair contraction
template<class Visitor>
void particle_messages(const JetList &input, const ClusterTree &tree, const double *fit, Visitor visit) {
    if (input.empty()) return;
    auto axes = sorted_by_pt(tree.exclusive_jets(std::min(3, int(input.size()))));
    const Momentum root = tree.exclusive_jets(1).front();
    const double mt = std::max(.01, root.m()), pt = std::max(.01,root.pt());
    int bachelor = 0;
    double best = std::numeric_limits<double>::infinity();
    if (axes.size() == 3) {
        for (int i = 0; i < 3; ++i) {
            const double pull = std::abs((axes[(i+1)%3]+axes[(i+2)%3]).m()-fit[1]);
            if (pull < best) { best = pull; bachelor = i; }
        }
    }
    const int n = input.size();
    struct Environment {
        std::array<double, 3> sum{}, count{};
        double near2 = 1., hard2 = 1., kt = 1.e6;
    };
    thread_local std::vector<Environment> env;
    env.assign(n, {});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            const double d = input[i].squared_distance(input[j]);
            const double kt2 = std::min(input[i].pt2(),input[j].pt2()) * d;
            env[i].near2 = std::min(env[i].near2,d);
            env[j].near2 = std::min(env[j].near2,d);
            env[i].kt = std::min(env[i].kt,kt2);
            env[j].kt = std::min(env[j].kt,kt2);
            if (input[j].pt() > input[i].pt()) env[i].hard2 = std::min(env[i].hard2,d);
            if (input[i].pt() > input[j].pt()) env[j].hard2 = std::min(env[j].hard2,d);
            const int bin = d < .0025 ? 0 : d < .0225 ? 1 : d < .09 ? 2 : 3;
            if (bin < 3) {
                env[i].sum[bin] += input[j].pt();
                env[j].sum[bin] += input[i].pt();
                ++env[i].count[bin];
                ++env[j].count[bin];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        const auto &p = input[i];
        const auto &e = env[i];
        std::array<double, 3> distance{1.,1.,1.};
        int nearest = 0;
        for (std::size_t j = 0; j < axes.size(); ++j) {
            distance[j] = p.squared_distance(axes[j]);
            if (distance[j] < distance[nearest]) nearest = j;
        }
        const auto &a = axes[nearest];
        const auto &b = axes[bachelor];
        const double rest_e = std::max(0., minkowski(root,p)/mt);
        const double axis_e = std::max(.001, minkowski(root,a)/mt);
        const double dot = std::max(0.,minkowski(a,p));
        const int q1 = (bachelor+1)%3, q2 = (bachelor+2)%3;
        const double ring = axes.size()==3 ? (distance[q1]+distance[q2])/
            std::max(1.e-8, axes[q1].squared_distance(axes[q2])) : 0.;
        ParticleMessage row{};
        int k = 0;
        for (double x : {p.pt(), p.pt()/pt, rest_e/mt, p.squared_distance(root), std::abs(p.rap()),
                         double(n), mt, double(nearest), double(nearest==bachelor), distance[nearest],
                         p.pt()/std::max(.001,a.pt()), a.m(), double(a.leaf_count), axis_e/mt,
                         dot/std::max(.0001,axis_e*rest_e), dot/(mt*mt), distance[bachelor],
                         ring, std::max(0.,minkowski(b,p))/(mt*mt), e.near2, e.hard2, e.kt}) row[k++] = x;
        for (int j = 0; j < 3; ++j) {
            row[k++] = e.count[j];
            row[k++] = e.sum[j]/pt;
            row[k++] = p.pt()/(p.pt()+e.sum[j]);
        }
        visit(i, row);
    }
}

} // namespace geometry
