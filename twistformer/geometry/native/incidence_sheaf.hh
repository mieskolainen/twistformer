// Tangent-vector transport on the incidence graph of constituents and microjets
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "lorentz_simplices.hh"

namespace geometry::incidence {

using transport::Spatial;
using transport::inner;
constexpr int axes = 4, channels = 17, fields = 2*(axes+1)*channels;

inline void accumulate(Spatial &sum, const Spatial &v, double weight) {
    for (int k=0; k<3; ++k) sum[k] += weight*v[k];
}

inline Spatial direction(const simplices::Vector &v) {
    Spatial n{v[1],v[2],v[3]};
    const double length = std::sqrt(inner(n,n));
    if (length < 1.e-12) return {};
    for (double &x : n) x /= length;
    return n;
}

// Parallel transport along the short geodesic of the direction sphere
inline Spatial carry(Spatial section, const Spatial &a, const Spatial &b, double cosine) {
    const double denominator = 1.+cosine;
    if (denominator < 1.e-9) return {};
    const double scale = inner(section,b)/denominator;
    for (int k=0; k<3; ++k) section[k] -= scale*(a[k]+b[k]);
    return section;
}

struct Edge { double cosine=0., membership=0.; };
struct Particle { Spatial n{}; double energy=0.; std::array<Edge,axes> edges{}; };
struct Stalk {
    double weight=0., occupancy=0., width=0., width2=0., core=0., soft=0., square=0.;
    Spatial gradient{}, returned{};
    double second[3][3]{}, residual=0.;
};

// Read section moments and the two-step connection operator at four ordered microjets
inline void project(const JetList &input, const ClusterTree &tree, float *out) {
    std::fill_n(out,fields,0.f);
    const auto jet = tree.exclusive_jets(1).front();
    const double mass = jet.m();
    if (mass < .01 || input.size()<2) return;
    const int count = std::min<int>(axes,input.size());
    auto subjets = tree.exclusive_jets(count);
    std::array<simplices::Vector,axes> momenta{};
    for (int i=0; i<count; ++i) momenta[i]=simplices::rest(subjets[i],jet,mass);
    std::stable_sort(momenta.begin(),momenta.begin()+count,[](const auto &a,const auto &b){return a[0]>b[0];});
    std::array<Spatial,axes> directions{};
    for (int i=0; i<count; ++i) directions[i]=direction(momenta[i]);
    thread_local std::vector<Particle> particles;
    particles.resize(input.size());
    for (std::size_t h=0; h<input.size(); ++h) {
        auto &p=particles[h];
        const auto v=simplices::rest(input[h],jet,mass);
        p.energy=std::max(0.,v[0]);
        p.n=direction(v);
        for (int i=0; i<count; ++i) p.edges[i].cosine=std::clamp(inner(p.n,directions[i]),-1.,1.);
    }
    for (const double radius : {.04,.25}) {
        std::array<Stalk,axes> stalks{};
        for (auto &p : particles) {
            double sum=0.;
            for (int i=0; i<count; ++i) {
                auto &e=p.edges[i];
                const double kernel=radius/(radius+1.-e.cosine);
                e.membership=kernel*kernel;
                sum+=e.membership;
            }
            for (int i=0; i<count; ++i) {
                auto &e=p.edges[i];
                e.membership/=sum;
                const double w=p.energy*e.membership, chord=1.-e.cosine;
                auto &s=stalks[i];
                s.weight+=w;
                s.occupancy+=e.membership;
                s.width+=w*chord;
                s.width2+=w*chord*chord;
                s.square+=w*w;
                if (chord<radius) s.core+=w;
                if (p.energy<.02) s.soft+=e.membership;
                Spatial tangent=p.n;
                accumulate(tangent,directions[i],-e.cosine);
                accumulate(s.gradient,tangent,w);
                for (int j=0; j<3; ++j) for (int k=0; k<=j; ++k) s.second[j][k]+=w*tangent[j]*tangent[k];
            }
        }
        for (int i=0; i<count; ++i) {
            auto &s=stalks[i];
            for (double &x:s.gradient) x/=std::max(1.e-14,s.weight);
        }
        for (const auto &p:particles) {
            Spatial mixed{};
            std::array<Spatial,axes> sections{};
            for (int i=0; i<count; ++i) {
                const auto &e=p.edges[i];
                sections[i]=carry(stalks[i].gradient,directions[i],p.n,e.cosine);
                accumulate(mixed,sections[i],e.membership);
            }
            for (int i=0; i<count; ++i) {
                const auto &e=p.edges[i];
                const double w=p.energy*e.membership;
                accumulate(stalks[i].returned,carry(mixed,p.n,directions[i],e.cosine),w);
                accumulate(sections[i],mixed,-1.);
                stalks[i].residual+=w*inner(sections[i],sections[i]);
            }
        }
        for (int i=0; i<count; ++i) {
            auto &s=stalks[i];
            const double norm=std::max(1.e-14,s.weight);
            for (double &x:s.returned) x/=norm;
            double trace=0.,square_trace=0.,alignment=0.;
            for (int j=0; j<3; ++j) for (int k=0; k<=j; ++k) {
                const double t=s.second[j][k]/norm, m=j==k ? 1. : 2.;
                if (j==k) trace+=t;
                square_trace+=m*t*t;
                alignment+=m*t*s.gradient[j]*s.gradient[k];
            }
            const double spin2=std::max(0.,2.*square_trace-trace*trace);
            Spatial laplacian=s.gradient;
            accumulate(laplacian,s.returned,-1.);
            const double row[channels]{s.weight,s.occupancy,s.soft,s.core/norm,s.square/(norm*norm),
                s.width/norm,s.width2/norm,trace,spin2,alignment,inner(s.gradient,s.gradient),
                inner(s.returned,s.returned),inner(s.gradient,s.returned),inner(laplacian,laplacian),
                inner(laplacian,s.gradient),s.residual/norm,inner(directions[i],transport::cross(s.gradient,s.returned))};
            for (int k=0; k<channels; ++k) {
                out[i*channels+k]=row[k];
                out[axes*channels+k]+=s.weight*row[k];
            }
        }
        out+=(axes+1)*channels;
    }
}
} // namespace geometry::incidence
