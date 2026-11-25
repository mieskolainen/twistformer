// Exclusive Cambridge–Aachen and kT clustering in rapidity and azimuth
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace geometry {

constexpr double pi = 3.14159265358979323846;

// Four-momentum in GeV with E-scheme recombination
struct Momentum {
    double x, y, z, energy;
    double transverse2, transverse, rapidity = 0., phi = 0.;
    const Momentum *left = nullptr, *right = nullptr;
    int leaf_count = 1;

    Momentum(double px = 0., double py = 0., double pz = 0., double e = 0.)
        : x(px), y(py), z(pz), energy(e), transverse2(x*x + y*y), transverse(std::sqrt(transverse2)) {}

    double pt() const { return transverse; }
    double pt2() const { return transverse2; }
    double rap() const { return rapidity; }
    double m2() const { return (energy + z) * (energy - z) - transverse2; }
    double m() const { return std::sqrt(std::max(0., m2())); }
    double modp2() const { return transverse2 + z*z; }

    // Cache angles only for lab-frame particles and recombined axes
    void set_angles() {
        phi = std::atan2(y, x);
        if (phi < 0.) phi += 2.*pi;
        if (transverse2 == 0. && energy == std::abs(z)) {
            rapidity = std::copysign(1.e5 + std::abs(z), z);
        } else {
            const double scale = energy + std::abs(z);
            const double mt2 = std::max(1.e-300, transverse2 + std::max(0., m2()));
            rapidity = mt2 == 1.e-300 ? .5*std::log(mt2) - std::log(scale)
                                      : .5*std::log(mt2 / (scale * scale));
            if (z > 0.) rapidity = -rapidity;
        }
    }

    Momentum operator+(const Momentum &other) const {
        return {x + other.x, y + other.y, z + other.z, energy + other.energy};
    }

    double delta_phi_to(const Momentum &other) const {
        double angle = other.phi - phi;
        if (angle > pi) angle -= 2.*pi;
        if (angle < -pi) angle += 2.*pi;
        return angle;
    }
    double squared_distance(const Momentum &other) const {
        const double dy = rapidity - other.rapidity, dp = delta_phi_to(other);
        return dy*dy + dp*dp;
    }
    double delta_R(const Momentum &other) const { return std::sqrt(squared_distance(other)); }

    bool has_parents(Momentum &a, Momentum &b) const {
        if (!left) return false;
        a = *left;
        b = *right;
        return true;
    }

    std::vector<Momentum> constituents() const {
        std::vector<Momentum> result;
        result.reserve(leaf_count);
        std::vector<const Momentum *> stack{this};
        while (!stack.empty()) {
            const Momentum *node = stack.back();
            stack.pop_back();
            if (node->left) {
                stack.push_back(node->right);
                stack.push_back(node->left);
            } else {
                result.push_back(*node);
            }
        }
        return result;
    }

    // Apply the inverse Lorentz boost from the lab to the jet rest frame
    void unboost(const Momentum &root) {
        if (root.modp2() == 0.) return;
        const double mass = root.m();
        const double rest_energy = (energy * root.energy - (x*root.x + y*root.y + z*root.z)) / mass;
        const double factor = (rest_energy + energy) / (root.energy + mass);
        *this = {x - factor*root.x, y - factor*root.y, z - factor*root.z, rest_energy};
    }
};

using JetList = std::vector<Momentum>;

// Each active axis contributes only its geometrical nearest neighbour to the global minimum
class ClusterTree {
    struct Axis {
        double rapidity, phi, scale;
        int node, neighbour = -1;
        double distance = std::numeric_limits<double>::infinity();
    };
    JetList nodes;
    int particles;

    static double separation(const Axis &a, const Axis &b) {
        const double dy = a.rapidity - b.rapidity;
        double dp = std::abs(a.phi - b.phi);
        if (dp > pi) dp = 2.*pi - dp;
        return dy*dy + dp*dp;
    }
    static void update(Axis &axis, int neighbour, double distance) {
        if (distance < axis.distance) {
            axis.distance = distance;
            axis.neighbour = neighbour;
        }
    }
public:
    ClusterTree(const ClusterTree &) = delete;
    ClusterTree &operator=(const ClusterTree &) = delete;

    ClusterTree(const JetList &input, bool kt) : nodes(input), particles(static_cast<int>(input.size())) {
        nodes.reserve(2 * input.size() - 1);
        std::vector<Axis> axes;
        axes.reserve(input.size());
        auto axis_for = [&](int index) {
            const auto &p = nodes[index];
            return Axis{p.rapidity, p.phi, kt ? p.pt2() : 1., index};
        };
        for (int i = 0; i < particles; ++i) {
            axes.push_back(axis_for(i));
            for (int j = 0; j < i; ++j) {
                const double distance = separation(axes[i], axes[j]);
                update(axes[i], j, distance);
                update(axes[j], i, distance);
            }
        }
        while (axes.size() > 1) {
            double best = std::numeric_limits<double>::infinity();
            int first = 0;
            for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
                const auto &axis = axes[i];
                const double cost = axis.distance * std::min(axis.scale, axes[axis.neighbour].scale);
                if (cost < best) {
                    best = cost;
                    first = i;
                }
            }
            int second = axes[first].neighbour;
            if (first > second) std::swap(first, second);
            const auto &a = nodes[std::min(axes[first].node, axes[second].node)];
            const auto &b = nodes[std::max(axes[first].node, axes[second].node)];
            nodes.push_back(a + b);
            auto &parent = nodes.back();
            parent.set_angles();
            parent.left = &a;
            parent.right = &b;
            parent.leaf_count = a.leaf_count + b.leaf_count;

            const int last = static_cast<int>(axes.size()) - 1;
            axes[second] = axes[last];
            axes.pop_back();
            axes[first] = axis_for(static_cast<int>(nodes.size()) - 1);
            for (int i = 0; i < last; ++i) {
                if (i == first) continue;
                auto &axis = axes[i];
                if (axis.neighbour == first || axis.neighbour == second) {
                    axis.distance = std::numeric_limits<double>::infinity();
                    for (int j = 0; j < last; ++j)
                        if (j != i) update(axis, j, separation(axis, axes[j]));
                } else {
                    if (axis.neighbour == last) axis.neighbour = second;
                    update(axis, first, separation(axis, axes[first]));
                }
                update(axes[first], i, separation(axis, axes[first]));
            }
        }
    }

    // Read the partition after particles minus count E-scheme merges
    JetList exclusive_jets(int count) const {
        if (count == 1) return {nodes.back()};
        JetList result;
        result.reserve(count);
        const int limit = 2 * particles - count;
        const Momentum *cut = nodes.data() + limit;
        for (auto parent = cut; parent != nodes.data() + nodes.size(); ++parent) {
            if (parent->left < cut) result.push_back(*parent->left);
            if (parent->right < cut) result.push_back(*parent->right);
        }
        return result;
    }
};

inline JetList sorted_by_pt(JetList axes) {
    std::stable_sort(axes.begin(), axes.end(), [](const Momentum &a, const Momentum &b) {
        return a.pt2() > b.pt2();
    });
    return axes;
}

void clustering_history(const JetList &input, const ClusterTree &tree, const double *mass_fit, float *&output);
void angular_flow(const JetList &input, const ClusterTree &tree, float *&output);

} // namespace geometry
