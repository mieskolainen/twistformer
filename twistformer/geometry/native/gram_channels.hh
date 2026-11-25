// Pair channels and rooted wedges of the constituent Minkowski Gram matrix
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "lorentz_simplices.hh"

namespace geometry::gram_channels {

constexpr int bands=7, types=5, pairs=15, fields=3*pairs*bands+types*28;

template<std::size_t N>
inline int bin(double value,const std::array<double,N> &cuts) {
    return std::upper_bound(cuts.begin(),cuts.end(),value)-cuts.begin();
}

inline int triangle_bin(int first,int second) {
    if (first>second) std::swap(first,second);
    return second*(second+1)/2+first;
}

// Row aggregates distinguish a hard constituent from diffuse radiation in its neighbourhood
inline void project(const JetList &input,const Momentum &jet,float *&out) {
    float *result=out;
    out+=fields;
    std::fill_n(result,fields,0.f);
    const double mass=jet.m();
    if (mass<.01 || input.size()<2) return;
    struct Node {
        simplices::Vector v;
        double local_energy=0.;
        int rest_bin=0,pt_bin=0,local_bin=0;
        std::array<int,bands> neighbors{};
    };
    struct Edge { unsigned char angle, mixed; };
    const int n=input.size();
    thread_local std::vector<Node> nodes;
    thread_local std::vector<Edge> edges;
    nodes.assign(n,{});
    edges.resize(std::size_t(n)*(n-1)/2);
    constexpr std::array<double,4> energies{.006,.018,.045,.11}, transverse{.9,2.5,6.5,18.};
    constexpr std::array<double,4> sharing{.04,.12,.33,.72};
    constexpr std::array<double,6> opening{.008,.035,.13,.42,.95,1.6};
    constexpr std::array<double,6> invariant{.000003,.000025,.00015,.0008,.004,.025};
    std::size_t edge=0;
    for (int i=0;i<n;++i) {
        auto &a=nodes[i];
        a.v=simplices::rest(input[i],jet,mass);
        a.rest_bin=bin(a.v[0],energies);
        a.pt_bin=bin(input[i].pt(),transverse);
        a.local_energy=a.v[0];
        for (int j=0;j<i;++j) {
            auto &b=nodes[j];
            const double gram=std::max(0.,simplices::dot(a.v,b.v));
            const double chord=gram/std::max(1.e-14,a.v[0]*b.v[0]);
            const int band=bin(chord,opening);
            edges[edge++]={static_cast<unsigned char>(band),static_cast<unsigned char>(bin(gram,invariant))};
            const double kernel=.055/(.055+chord);
            a.local_energy+=kernel*kernel*b.v[0];
            b.local_energy+=kernel*kernel*a.v[0];
            ++a.neighbors[band];
            ++b.neighbors[band];
        }
    }
    edge=0;
    for (int i=0;i<n;++i) {
        auto &a=nodes[i];
        a.local_bin=bin(a.v[0]/std::max(1.e-12,a.local_energy),sharing);
        for (int j=0;j<i;++j) {
            const auto &b=nodes[j];
            const auto e=edges[edge++];
            ++result[bands*triangle_bin(a.rest_bin,b.rest_bin)+e.angle];
            ++result[bands*(pairs+triangle_bin(a.pt_bin,b.pt_bin))+e.angle];
            ++result[bands*(2*pairs+triangle_bin(a.local_bin,b.local_bin))+e.mixed];
        }
        float *wedge=result+3*pairs*bands+28*a.rest_bin;
        for (int j=0;j<bands;++j) if (a.neighbors[j]) {
            for (int k=0;k<=j;++k) if (a.neighbors[k]>(k==j ? 1 : 0)) ++wedge[triangle_bin(j,k)];
        }
    }
}
} // namespace geometry::gram_channels
