// Time and beam-plane contractions of jet momenta
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include "tangent_transport.hh"

namespace geometry::lab_references {

using transport::Spatial;
using transport::inner;
constexpr int axes = 4;
constexpr int moment_fields = 7;
constexpr int pair_fields = 7;
constexpr int triple_fields = 3;
constexpr int fields = 4 + axes + 2*moment_fields + 6*pair_fields + 4*triple_fields;

// Project momentum onto the time reference and the transverse plane
inline Spatial velocity(const Momentum &p) {
    return {p.x/p.energy, p.y/p.energy, p.z/p.energy};
}

inline double beam_area(const Spatial &a, const Spatial &b) {
    return a[0]*b[1] - a[1]*b[0];
}

// Pool constituent contractions and retain individual exclusive kT microjet contractions
inline void project(const JetList &input, const ClusterTree &tree, float *output) {
    std::fill_n(output, fields, 0.f);
    const Momentum jet = tree.exclusive_jets(1).front();
    const double inverse_energy = 1./jet.energy;
    output[0] = jet.energy/20.;
    output[1] = jet.pt()*inverse_energy;
    output[2] = std::abs(jet.z)*inverse_energy;
    const Spatial direction = jet.pt() > 0. ? Spatial{jet.x/jet.pt(), jet.y/jet.pt(), 0.} : Spatial{};
    const double longitudinal = jet.z*inverse_energy;
    std::array<double, moment_fields> count_moments{}, energy_moments{};
    double concentration = 0.;
    for (const auto &p : input) {
        const auto v = velocity(p);
        const double area = beam_area(direction, v);
        const double dz = v[2]-longitudinal;
        const double area2 = area*area, dz2 = dz*dz;
        const double weight = p.energy*inverse_energy;
        const double row[moment_fields]{
            v[0]*v[0]+v[1]*v[1], area2, dz2, area2*area2, dz2*dz2, area2*dz2, inner(direction,v)
        };
        concentration += weight*weight;
        for (int k = 0; k < moment_fields; ++k) {
            count_moments[k] += row[k];
            energy_moments[k] += weight*row[k];
        }
    }
    output[3] = concentration;
    for (int k = 0; k < moment_fields; ++k) {
        output[4+axes+k] = count_moments[k]/input.size();
        output[4+axes+moment_fields+k] = energy_moments[k];
    }

    const int count = std::min<int>(axes,input.size());
    const auto subjets = sorted_by_pt(tree.exclusive_jets(count));
    std::array<Spatial,axes> v{};
    std::array<double,axes> z{};
    for (int i = 0; i < count; ++i) {
        v[i] = velocity(subjets[i]);
        z[i] = subjets[i].energy*inverse_energy;
        output[4+i] = z[i];
    }
    float *pairs = output + 4 + axes + 2*moment_fields;
    float *triples = pairs + 6*pair_fields;
    for (int i = 0; i < axes; ++i) {
        for (int j = i+1; j < axes; ++j, pairs += pair_fields) {
            if (j >= count) continue;
            const double area = beam_area(v[i],v[j]), dz = v[i][2]-v[j][2];
            const double dx = v[i][0]-v[j][0], dy = v[i][1]-v[j][1];
            pairs[0] = z[i]+z[j];
            pairs[1] = std::abs(z[i]-z[j])/(z[i]+z[j]);
            pairs[2] = area*area;
            pairs[3] = dz*dz;
            pairs[4] = dx*dx+dy*dy+dz*dz;
            pairs[5] = v[i][0]*v[j][0]+v[i][1]*v[j][1];
            pairs[6] = z[i]*z[j]*area*area;
        }
    }
    for (int i = 0; i < axes; ++i) {
        for (int j = i+1; j < axes; ++j) {
            for (int k = j+1; k < axes; ++k, triples += triple_fields) {
                if (k >= count) continue;
                const double area = beam_area(v[i],v[j])+beam_area(v[j],v[k])+beam_area(v[k],v[i]);
                const double volume = inner(v[i], transport::cross(v[j],v[k]));
                triples[0] = area*area;
                triples[1] = volume*volume;
                triples[2] = z[i]*z[j]*z[k]*volume*volume;
            }
        }
    }
}
} // namespace geometry::lab_references
