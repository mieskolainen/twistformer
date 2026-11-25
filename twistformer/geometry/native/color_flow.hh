// Soft radiation and constituent flow for the W dipole and the bachelor subjet
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include <array>

namespace geometry {

constexpr int color_axis_fields = 12, color_pair_fields = 8;
constexpr int color_fields = 3*(color_axis_fields + color_pair_fields) + 8;

// Resolve three axes before and after a single collinear core update
inline void color_flow(const JetList &input, const ClusterTree &tree, const double *fit, float *&out) {
    if (input.size() < 3) {
        std::fill_n(out, 2*color_fields, 0.f);
        out += 2*color_fields;
        return;
    }
    const auto root = tree.exclusive_jets(1).front();
    const double pt = std::max(.001, root.pt());
    auto axes = sorted_by_pt(tree.exclusive_jets(3));
    for (int pass = 0; pass < 2; ++pass) {
        int b = 0;
        double closest = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 3; ++i) {
            const double pull = std::abs((axes[(i+1)%3]+axes[(i+2)%3]).m()-fit[1]);
            if (pull < closest) { closest = pull; b = i; }
        }
        std::array<Momentum,3> a{axes[b],axes[(b+1)%3],axes[(b+2)%3]};
        if (a[1].pt() < a[2].pt()) std::swap(a[1],a[2]);
        std::array<Momentum,3> core{};
        std::array<std::array<double,color_axis_fields>,3> axis{};
        std::array<std::array<double,color_pair_fields>,3> pair{};
        double separation[3]{}, dy[3]{}, dp[3]{};
        for (int i = 0; i < 3; ++i) {
            const int j = (i+1)%3, k = (i+2)%3;
            dy[i] = a[k].rap()-a[j].rap();
            dp[i] = a[j].delta_phi_to(a[k]);
            separation[i] = std::max(1.e-8, dy[i]*dy[i]+dp[i]*dp[i]);
        }
        for (const auto &p : input) {
            double d[3], y[3], phi[3];
            int near = 0;
            for (int i = 0; i < 3; ++i) {
                y[i] = p.rap()-a[i].rap();
                phi[i] = a[i].delta_phi_to(p);
                d[i] = y[i]*y[i]+phi[i]*phi[i];
                if (d[i] < d[near]) near = i;
            }
            const double fraction = p.pt()/pt, r = std::sqrt(d[near]);
            const double z = p.pt()/std::max(.001, a[near].pt());
            auto &f = axis[near];
            f[0]++;
            f[1] += fraction;
            f[2] += z*z;
            f[3] = std::max(f[3],z);
            f[4] += z*r;
            f[5] += z*d[near];
            f[6] += z*r*y[near];
            f[7] += z*r*phi[near];
            const double scale = std::min(separation[(near+1)%3],separation[(near+2)%3]);
            if (d[near] < .16*scale) {
                core[near] = core[near]+p;
                f[8] += 1.;
                f[9] += fraction;
            }
            if (p.pt() < 3.) { f[10]++; f[11] += fraction; }
            if (z > .35) continue;
            for (int i = 0; i < 3; ++i) {
                const int j = (i+1)%3, k = (i+2)%3;
                const double t = (y[j]*dy[i]+phi[j]*dp[i])/separation[i];
                const double cross = y[j]*dp[i]-phi[j]*dy[i];
                const double perpendicular = cross*cross/(separation[i]*separation[i]);
                const double ring = (d[j]+d[k])/separation[i];
                auto &g = pair[i];
                if (t > .1 && t < .9 && perpendicular < .16 && near != i) {
                    g[0]++;
                    g[1] += fraction;
                    g[2] += fraction*perpendicular;
                    g[3] += fraction*std::abs(t-.5);
                }
                if (ring < 1. && near != i) { g[4]++; g[5] += fraction; }
                const double antenna = separation[i]/((.0025+d[j])*(.0025+d[k]));
                g[6] += fraction*std::log1p(antenna);
                g[7] += fraction*std::log1p(antenna*d[i]);
            }
        }
        const auto w = core[1]+core[2], top = w+core[0];
        for (double value : {top.m(),w.m(),core[0].m(),core[1].m(),core[2].m(),
                             top.pt()/pt,(core[0]+core[1]).m(),(core[0]+core[2]).m()}) *out++ = value;
        for (int i = 0; i < 3; ++i) {
            // Project the pull toward the other two decay axes
            const double py = axis[i][6], pp = axis[i][7];
            for (int n = 0; n < 2; ++n) {
                const int j = (i+1+n)%3;
                axis[i][6+n] = (py*(a[j].rap()-a[i].rap()) + pp*a[i].delta_phi_to(a[j])) /
                                 std::sqrt(std::max(1.e-8,a[i].squared_distance(a[j])));
            }
            for (double value : axis[i]) *out++ = value;
        }
        for (const auto &fields : pair) for (double value : fields) *out++ = value;
        if (pass == 0) {
            for (int i = 0; i < 3; ++i) {
                axes[i] = core[i].pt() > 0. ? core[i] : a[i];
                axes[i].set_angles();
            }
        }
    }
}
} // namespace geometry
