// Incidence counts and four-prong plane correlations on the Cambridge–Aachen cascade
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include "tangent_transport.hh"

namespace geometry::cascades {

constexpr int orders = 3, scales = 6, bands = 5;
constexpr int fields = orders*scales*bands + orders*scales*scales + orders*bands*bands;

// Resolve each branch at its daughter transverse momentum scale in GeV
inline int resolution(const Momentum &node) {
    constexpr std::array<double, scales-1> cuts{.35, 1., 3., 9., 27.};
    const double kt = std::min(node.left->pt(), node.right->pt())*node.left->delta_R(*node.right);
    return std::upper_bound(cuts.begin(), cuts.end(), kt) - cuts.begin();
}

// Construct the two decay planes in their common four-prong rest frame
inline transport::Spatial plane(const Momentum &branch, const Momentum &parent) {
    auto a = *branch.left, b = *branch.right;
    a.unboost(parent);
    b.unboost(parent);
    return transport::cross({a.x, a.y, a.z}, {b.x, b.y, b.z});
}

// Count vertices, adjacent resolved splittings and pairs of incident decay faces
inline void project(const ClusterTree &tree, float *output) {
    std::fill_n(output, fields, 0.f);
    const auto jet = tree.exclusive_jets(1).front();
    if (!jet.left) return;
    struct Branch { const Momentum *node; int order, parent_scale; };
    thread_local std::vector<Branch> stack;
    stack.clear();
    stack.push_back({&jet, 0, -1});
    constexpr std::array<double, bands-1> zcuts{.025, .075, .16, .30};
    constexpr std::array<double, bands-1> plane_cuts{.10, .35, .65, .90};
    while (!stack.empty()) {
        const auto branch = stack.back();
        stack.pop_back();
        const auto &node = *branch.node;
        auto hard = node.left, soft = node.right;
        if (hard->pt2() < soft->pt2()) std::swap(hard, soft);
        const double z = soft->pt()/std::max(1.e-12, hard->pt()+soft->pt());
        const int zbin = std::upper_bound(zcuts.begin(), zcuts.end(), z) - zcuts.begin();
        const int scale = resolution(node);
        ++output[(branch.order*scales+scale)*bands+zbin];
        float *adjacent = output + orders*scales*bands;
        if (branch.parent_scale >= 0)
            ++adjacent[(branch.order*scales+branch.parent_scale)*scales+scale];
        if (hard->left && soft->left && node.m() > .01) {
            const auto a = plane(*hard, node), b = plane(*soft, node);
            const double norm2 = transport::inner(a,a)*transport::inner(b,b);
            if (norm2 > 1.e-20) {
                const double cosine = std::abs(transport::inner(a,b))/std::sqrt(norm2);
                const int angle = std::upper_bound(plane_cuts.begin(), plane_cuts.end(), cosine) - plane_cuts.begin();
                ++adjacent[orders*scales*scales + (branch.order*bands+zbin)*bands+angle];
            }
        }
        if (hard->left) stack.push_back({hard, branch.order, scale});
        if (soft->left) stack.push_back({soft, std::min(orders-1, branch.order+1), scale});
    }
}

} // namespace geometry::cascades
