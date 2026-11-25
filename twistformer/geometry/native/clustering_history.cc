// Exclusive clustering histories and optional two-stage decay correlations
//
// mikael.mieskolainen@cern.ch (2026)

#include <algorithm>
#include <array>
#include <cmath>
#include "clustering.hh"
#include <vector>

namespace geometry {

// Return the physical mass for timelike four-momenta
double positive_mass(const Momentum &p) { return std::sqrt(std::max(0., p.m2())); }

// Write the ordered splittings, groomed jets and exclusive subjet correlations
void clustering_history(const JetList &input, const ClusterTree &sequence, const double *mass_fit, float *&out) {
    auto root = sequence.exclusive_jets(1)[0];
    const double pt = std::max(root.pt(), 1.e-8);
    const double mass = std::max(positive_mass(root), 1.e-8);
    using Split = std::array<double, 10>;
    std::vector<Split> splits;
    std::vector<Momentum> stack{root};
    while (!stack.empty()) {
        auto p = stack.back();
        stack.pop_back();
        Momentum a, b;
        if (!p.has_parents(a, b))
            continue;
        if (a.pt2() < b.pt2())
            std::swap(a, b);
        auto angle = a.delta_R(b);
        auto z = b.pt() / std::max(1.e-12, a.pt() + b.pt());
        splits.push_back({b.pt() * angle, angle, z, positive_mass(p), positive_mass(a), positive_mass(b),
                          p.pt() / pt, a.delta_phi_to(b), b.rap() - a.rap(),
                          double(b.leaf_count)});
        stack.push_back(b);
        stack.push_back(a);
    }
    std::sort(splits.begin(), splits.end(), [](const auto &a, const auto &b) { return a[0] > b[0]; });
    for (size_t i = 0; i < 7; i++) {
        for (size_t j = 0; j < 10; j++)
            *out++ = i < splits.size() ? splits[i][j] : 0.;
    }
    constexpr std::array<double, 4> zcuts = {.03, .08, .15, .25};
    std::array<std::array<double, 4>, 4> groomed{};
    auto branch = root;
    Momentum hard, soft;
    while (branch.has_parents(hard, soft)) {
        if (hard.pt2() < soft.pt2()) std::swap(hard, soft);
        const double total_pt = hard.pt() + soft.pt();
        for (size_t i = 0; i < zcuts.size(); ++i) {
            if (soft.pt() <= zcuts[i] * total_pt) break;
            auto &fields = groomed[i];
            if (fields[3] == 0.) {
                fields = {positive_mass(branch), hard.delta_R(soft), branch.pt() / pt, 0.};
            }
            ++fields[3];
        }
        branch = hard;
    }
    for (const auto &fields : groomed)
        for (double value : fields) *out++ = value;
    double last_tau = 1.;
    for (int n : {2, 3, 4, 5, 6, 8}) {
        auto axes = sorted_by_pt(sequence.exclusive_jets(std::min(n, int(input.size()))));
        double tau = 0., tau2 = 0., tauhalf = 0.;
        for (const auto &p : input) {
            double r2 = 1.e9;
            for (const auto &a : axes)
                r2 = std::min(r2, p.squared_distance(a));
            tau += p.pt() * std::sqrt(r2) / pt;
            tau2 += p.pt() * r2 / pt;
            tauhalf += p.pt() * std::sqrt(std::sqrt(r2)) / pt;
        }
        for (double x : {tau, tau2, tauhalf, tau / std::max(1.e-10, last_tau)})
            *out++ = x;
        last_tau = tau;
        if (n == 2 || n == 3 || n == 5) {
            for (int i = 0; i < n; ++i) {
                std::array<double, 8> values{};
                if (i < int(axes.size())) {
                    const auto &p = axes[i];
                    auto constituents = p.constituents();
                    double leading = 0., sq = 0., width = 0.;
                    for (const auto &q : constituents) {
                        auto f = q.pt() / std::max(1.e-12, p.pt());
                        leading = std::max(leading, f);
                        sq += f * f;
                        width += f * q.delta_R(p);
                    }
                    values = {p.pt() / pt, positive_mass(p),     double(constituents.size()), leading, sq,
                              width,       p.rap() - root.rap(), root.delta_phi_to(p)};
                }
                for (auto x : values)
                    *out++ = x;
            }
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j) {
                    std::array<double, 4> values{};
                    if (j < int(axes.size())) {
                        auto pair = axes[i] + axes[j];
                        auto angle = axes[i].delta_R(axes[j]);
                        values = {positive_mass(pair), positive_mass(pair) / mass, angle,
                                  axes[j].pt() * angle};
                    }
                    for (auto x : values)
                        *out++ = x;
                }
        }
        if (mass_fit && n >= 4) {
            const double parent_mass = mass_fit[0], pair_mass = mass_fit[1];
            const double parent_width = mass_fit[2], pair_width = mass_fit[3];
            double best = 1.e20, fitted_parent = 0., fitted_pair = 0., pair_share = 0., fraction = 0., other = 0.;
            for (size_t i = 0; i < axes.size(); i++)
                for (size_t j = i + 1; j < axes.size(); j++) {
                    auto pair = axes[i] + axes[j];
                    const double v = positive_mass(pair);
                    for (size_t k = 0; k < axes.size(); k++) {
                        if (k == i || k == j)
                            continue;
                        auto parent = pair + axes[k];
                        double t = positive_mass(parent);
                        double chi = std::pow((t - parent_mass) / parent_width, 2) + std::pow((v - pair_mass) / pair_width, 2);
                        if (chi < best) {
                            best = chi;
                            fitted_parent = t;
                            fitted_pair = v;
                            pair_share = pair.pt() / std::max(parent.pt(), 1.e-10);
                            fraction = parent.pt() / pt;
                            other =
                                std::min(positive_mass(axes[i] + axes[k]), positive_mass(axes[j] + axes[k]));
                        }
                    }
                }
            for (double x : {fitted_parent, fitted_pair, pair_share, fraction, other})
                *out++ = x;
        }
    }
}

} // namespace geometry
