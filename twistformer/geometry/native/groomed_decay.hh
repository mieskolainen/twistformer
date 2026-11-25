// Recursive Soft Drop and nested decay candidates in the Cambridge–Aachen tree
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include <array>

namespace geometry {

constexpr int groomed_fields = 25;

namespace groomed_decay {
struct Node {
    Momentum p;
    int left = -1, right = -1, leaves = 1;
    double kt = 0., z = 0.;
};

// Groom both subtrees after each accepted split
inline int groom(const Momentum &p, double zcut, bool angular, std::vector<Node> &nodes) {
    if (!p.left) {
        nodes.push_back({p});
        return nodes.size()-1;
    }
    const Momentum *a = p.left, *b = p.right;
    if (a->pt2() < b->pt2()) std::swap(a,b);
    const double angle = a->delta_R(*b);
    const double z = b->pt()/std::max(1.e-12, a->pt()+b->pt());
    if (z < zcut*(angular ? angle/.8 : 1.)) return groom(*a, zcut, angular, nodes);
    int i = groom(*a, zcut, angular, nodes), j = groom(*b, zcut, angular, nodes);
    const auto &hard = nodes[i], &soft = nodes[j];
    nodes.push_back({hard.p+soft.p, i, j, hard.leaves+soft.leaves, b->pt()*angle, z});
    return nodes.size()-1;
}

// Assign a W subtree to each ancestor and retain its complementary b system
inline void candidates(const std::vector<Node> &nodes, int root, double pt, const double *fit, float *&out) {
    const Node &jet = nodes[root];
    double best = std::numeric_limits<double>::infinity(), support = 0.;
    std::array<double, 13> fields{};
    for (int i = 0; i <= root; ++i) {
        const Node &top = nodes[i];
        if (top.leaves < 3) continue;
        const double mt = top.p.m(), dt = (mt-fit[0])/fit[2];
        // Postorder indexing places all descendants immediately before their parent
        const int first = i + 1 - (2*top.leaves-1);
        for (int j = first; j < i; ++j) {
            const auto &w = nodes[j];
            if (w.left < 0) continue;
            const double mw = w.p.m(), dw = (mw-fit[1])/fit[3];
            const Momentum b{top.p.x-w.p.x, top.p.y-w.p.y, top.p.z-w.p.z, top.p.energy-w.p.energy};
            const double mb = b.m(), db = mb / fit[2];
            const double score = dt*dt + dw*dw + db*db;
            const double evidence = 1./((1.+dt*dt)*(1.+dw*dw)*(1.+db*db));
            support += evidence;
            if (score >= best) continue;
            best = score;
            const auto &q1 = nodes[w.left].p, &q2 = nodes[w.right].p;
            const double m1 = (b+q1).m(), m2 = (b+q2).m();
            fields = {score, mt, mw, mb, std::min(m1,m2), std::max(m1,m2), top.p.pt()/pt,
                      w.p.pt()/pt, double(w.leaves), double(top.leaves-w.leaves), w.kt, top.kt, w.z};
        }
    }
    int hard = jet.left, soft = jet.right;
    if (hard >= 0 && nodes[hard].p.pt2() < nodes[soft].p.pt2()) std::swap(hard,soft);
    double widest_mass = 0., kt1 = 0., kt2 = 0.;
    for (const auto &node : nodes) {
        if (node.kt > kt1) { kt2 = kt1; kt1 = node.kt; widest_mass = node.p.m(); }
        else kt2 = std::max(kt2, node.kt);
    }
    for (double value : {jet.p.m(), jet.p.pt()/pt, double(jet.leaves), jet.z, jet.kt, kt1, kt2,
                         widest_mass, hard >= 0 ? nodes[hard].p.m() : 0., soft >= 0 ? nodes[soft].p.m() : 0.,
                         hard >= 0 ? nodes[hard].kt : 0., support}) *out++ = value;
    for (double value : fields) *out++ = value;
}
} // namespace groomed_decay

inline void groomed_histories(const ClusterTree &tree, const double *fit, float *&output) {
    const Momentum root = tree.exclusive_jets(1).front();
    thread_local std::vector<groomed_decay::Node> nodes;
    for (int trial = 0; trial < 3; ++trial) {
        nodes.clear();
        nodes.reserve(2*root.leaf_count);
        const int jet = groomed_decay::groom(root, trial == 1 ? .12 : .05, trial == 2, nodes);
        groomed_decay::candidates(nodes, jet, std::max(root.pt(), 1.e-12), fit, output);
    }
}
} // namespace geometry
