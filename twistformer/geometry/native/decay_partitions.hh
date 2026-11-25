// Resonance weights over binary histories of exclusive kT microjets
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include <array>

namespace geometry {

constexpr int partition_fields = 14;

// Weight a two-body decay with a finite mass resolution
inline double resonance_weight(double mass, double pole, double width) {
    const double pull = (mass - pole) / width;
    return 1. / (1. + pull*pull);
}

// Sum over all unordered splits of each subset of up to seven microjets
inline void decay_partition(const JetList &axes, const double *fit, float *&output) {
    const int size = 1 << axes.size(), full = size - 1;
    struct State {
        Momentum p;
        double mass = 0., pt = 0., qcd = 0., w = 0., top = 0., w_direct = 0., top_direct = 0.;
        double best_w = 0., w_mean = 0., b_mean = 0., w_width = 0., effective = 0.;
    };
    std::array<State, 128> subsets{};
    for (std::size_t i = 0; i < axes.size(); ++i) {
        auto &leaf = subsets[1 << i];
        leaf.p = axes[i];
        leaf.pt = axes[i].pt();
        leaf.mass = axes[i].m();
        leaf.qcd = 1.;
    }
    for (int set = 1; set < size; ++set) {
        const int first = set & -set, rest = set ^ first;
        if (!rest) continue;
        auto &s = subsets[set];
        s.p = subsets[first].p + subsets[rest].p;
        s.mass = s.p.m();
        s.pt = s.p.pt();
        const double w_shape = resonance_weight(s.mass, fit[1], fit[3]);
        const double top_shape = resonance_weight(s.mass, fit[0], fit[2]);
        double largest = 0., second_moment = 0., sum_squared = 0.;
        // Keep the lowest bit on the left to count each binary split once
        for (int right = rest; right; right = (right-1) & rest) {
            const auto &a = subsets[set^right], &b = subsets[right];
            const double z = std::min(a.pt,b.pt) / std::max(1.e-12, a.pt+b.pt);
            const double soft = std::max(.01, z*(1.-z));
            const double virtuality = std::max(0., s.mass*s.mass-a.mass*a.mass-b.mass*b.mass);
            const double collinear = 100. / ((virtuality+25.) * soft);
            const double decay = 100. / (s.mass*s.mass + 25.);
            const double daughters = a.qcd*b.qcd;
            s.qcd += collinear*daughters;
            s.w_direct += decay*w_shape*daughters;
            s.w += collinear*.25*(a.w*b.qcd + b.w*a.qcd);
            const double weight_a = decay*top_shape*a.w*b.qcd, weight_b = decay*top_shape*b.w*a.qcd;
            s.top_direct += weight_a + weight_b;
            s.top += collinear*(a.top*b.qcd + b.top*a.qcd);
            if (set == full) {
                if (weight_a > largest) { largest = weight_a; s.best_w = a.mass; }
                if (weight_b > largest) { largest = weight_b; s.best_w = b.mass; }
                s.w_mean += weight_a*a.mass + weight_b*b.mass;
                s.b_mean += weight_a*b.mass + weight_b*a.mass;
                second_moment += weight_a*a.mass*a.mass + weight_b*b.mass*b.mass;
                sum_squared += weight_a*weight_a + weight_b*weight_b;
            }
        }
        s.w += s.w_direct;
        s.top += s.top_direct;
        if (set == full && s.top_direct > 0.) {
            s.w_mean /= s.top_direct;
            s.b_mean /= s.top_direct;
            s.w_width = std::sqrt(std::max(0., second_moment/s.top_direct - s.w_mean*s.w_mean));
            s.effective = s.top_direct*s.top_direct/sum_squared;
        }
    }
    const auto &s = subsets[full];
    double trim_sum = 0., trim_max = 0., trim_mass = 0., trim_pt = 0.;
    for (std::size_t i = 0; i < axes.size(); ++i) {
        const auto &part = subsets[full^(1 << i)];
        const double fraction = axes[i].pt() / std::max(1.e-9, s.pt);
        const double ratio = part.top/std::max(1.e-90, part.qcd) / (1.+16.*fraction*fraction);
        trim_sum += ratio;
        if (ratio > trim_max) {
            trim_max = ratio;
            trim_mass = part.mass;
            trim_pt = fraction;
        }
    }
    for (double value : {s.w/s.qcd, s.top/s.qcd, s.w_direct/s.qcd, s.top_direct/s.qcd, trim_sum, trim_max})
        *output++ = std::log(std::max(1.e-30, value));
    for (double value : {s.top_direct/std::max(1.e-90,s.top), s.best_w, s.w_mean, s.b_mean,
                         s.w_width, s.effective, trim_mass, trim_pt}) *output++ = value;
}

inline void decay_partitions(const JetList &input, const ClusterTree &tree, const double *fit, float *&output) {
    for (int count : {5, 7}) {
        if (int(input.size()) >= count) {
            decay_partition(tree.exclusive_jets(count), fit, output);
        } else {
            std::fill_n(output, partition_fields, 0.f);
            output += partition_fields;
        }
    }
}

} // namespace geometry
