// Lorentz scalar messages on microjet edges, triangles and tetrahedra
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "clustering.hh"
#include "tangent_transport.hh"
#include <array>

namespace geometry::simplices {

constexpr int vertices = 8, fields = 68;
using Row = std::array<float, fields>;
using Rows = std::array<Row, vertices>;
using Vector = std::array<double, 4>;

inline double dot(const Vector &a, const Vector &b) {
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
}

// Contract the trivector with the unit timelike vector of the jet
inline double volume(const Vector &a, const Vector &b, const Vector &c) {
    return a[1]*(b[2]*c[3]-b[3]*c[2]) - a[2]*(b[1]*c[3]-b[3]*c[1]) + a[3]*(b[1]*c[2]-b[2]*c[1]);
}

inline double chord(const Vector &a, const Vector &b) {
    const double product = std::max(1.e-14, a[0]*b[0]);
    return std::clamp(dot(a,b)/product, 0., 2.);
}

// Project to dimensionless coordinates in the jet rest frame
inline Vector rest(Momentum p, const Momentum &jet, double mass) {
    p.unboost(jet);
    return {p.energy/mass, p.x/mass, p.y/mass, p.z/mass};
}

inline double band(double value, double center, double width) {
    const double pull = (value-center)/width;
    return 1./(1.+pull*pull);
}

// Keep separate incident hyperedge messages at each vertex
inline int messages(const JetList &input, const ClusterTree &tree, const double *fit, Rows &out, float *sheaf = nullptr) {
    const Momentum jet = tree.exclusive_jets(1).front();
    const double mass = jet.m(), pt = std::max(.001,jet.pt());
    if (mass < .01) return 0;
    const int n = std::min(int(input.size()),vertices);
    auto axes = tree.exclusive_jets(n);
    auto rest_energy = [&](const Momentum &p) {
        return jet.energy*p.energy-jet.x*p.x-jet.y*p.y-jet.z*p.z;
    };
    std::stable_sort(axes.begin(),axes.end(),[&](const auto &a,const auto &b) { return rest_energy(a)>rest_energy(b); });
    std::array<Vector,vertices> v{};
    thread_local std::vector<Vector> particles;
    particles.resize(input.size());
    for (std::size_t p=0; p<input.size(); ++p) particles[p]=rest(input[p],jet,mass);
    double gram[vertices][vertices]{}, pair_mass[vertices][vertices]{};
    std::array<std::array<Vector,3>,vertices> flow{};
    for (int i=0; i<n; ++i) {
        auto &r=out[i];
        const auto &a=axes[i];
        v[i]=rest(a,jet,mass);
        r[0]=v[i][0];
        r[1]=a.pt()/pt;
        r[2]=a.m();
        r[3]=a.leaf_count;
        r[4]=mass;
        r[5]=input.size();
        r[6]=std::abs(a.rap()-jet.rap());
        r[7]=a.delta_R(jet);
        double lead=0., sum2=0., angle=0.;
        for (const auto &p : a.constituents()) {
            const double z=p.pt()/std::max(.001,a.pt());
            lead=std::max(lead,z);
            sum2+=z*z;
            angle+=z*p.delta_R(a);
        }
        r[8]=lead;
        r[9]=sum2;
        r[10]=angle;
        r[11]=a.left ? std::min(a.left->pt(),a.right->pt())*a.left->delta_R(*a.right) : 0.;
        constexpr double scales[]{.04,.20,.8};
        for (const auto &p : particles) {
            const double d=chord(v[i],p);
            for (int s=0; s<3; ++s) {
                const double weight=scales[s]/(scales[s]+d);
                const double w=weight*weight;
                for (int k=0; k<4; ++k) flow[i][s][k]+=w*p[k];
                r[12+4*s]+=w;
            }
        }
        for (int s=0; s<3; ++s) {
            const auto &f=flow[i][s];
            r[13+4*s]=f[0];
            r[14+4*s]=std::max(0.,dot(f,f));
            r[15+4*s]=dot(v[i],f)/std::max(1.e-12,v[i][0]*f[0]);
        }
        r[24]=std::abs(volume(v[i],flow[i][0],flow[i][2]));
        r[25]=dot(flow[i][0],flow[i][2]);
        for (int j=0; j<=i; ++j) {
            gram[j][i]=gram[i][j]=dot(v[i],v[j]);
            pair_mass[j][i]=pair_mass[i][j]=(axes[i]+axes[j]).m();
        }
    }
    for (int i=0; i<n; ++i) {
        auto &r=out[i];
        double best=0., strongest_w=0.;
        for (int j=0; j<n; ++j) {
            if (i==j) continue;
            const double w=band(pair_mass[i][j],fit[1],fit[3]);
            strongest_w=std::max(strongest_w,w);
            r[26]+=w;
            r[27]+=w*v[j][0];
            r[28]+=w*flow[j][1][0];
            r[29]+=w*std::max(0.,gram[i][j]*gram[i][j]-gram[i][i]*gram[j][j]);
            for (int k=0; k<j; ++k) {
                if (k==i) continue;
                const double ab=gram[i][j], ac=gram[i][k], bc=gram[j][k];
                const double aa=gram[i][i], bb=gram[j][j], cc=gram[k][k];
                const double norm3=std::max(0.,aa*bb*cc+2*ab*ac*bc-aa*bc*bc-bb*ac*ac-cc*ab*ab);
                const double plane=std::abs(volume(v[i],v[j],v[k]));
                const double mt=mass*std::sqrt(std::max(0.,aa+bb+cc+2*(ab+ac+bc)));
                const double tw=band(mt,fit[0],fit[2]);
                const double as_b=tw*band(pair_mass[j][k],fit[1],fit[3]);
                const double as_q=tw*std::max(w,band(pair_mass[i][k],fit[1],fit[3]));
                for (int mode=0; mode<2; ++mode) {
                    const double weight=mode==0 ? as_b : as_q;
                    const int f=30+mode*8;
                    r[f]+=weight;
                    r[f+1]=std::max<double>(r[f+1],weight);
                    r[f+2]+=weight*(v[j][0]+v[k][0]);
                    r[f+3]+=weight*norm3;
                    r[f+4]+=weight*plane;
                    r[f+5]+=weight*(flow[j][1][0]+flow[k][1][0]);
                    r[f+6]+=weight*(axes[j].leaf_count+axes[k].leaf_count);
                    r[f+7]+=weight*std::abs(ab-ac)/std::max(1.e-12,ab+ac);
                }
                if (as_b>best) {
                    best=as_b;
                    r[46]=mt;
                    r[47]=pair_mass[j][k];
                    r[48]=norm3;
                    r[49]=plane;
                    r[50]=std::min(pair_mass[i][j],pair_mass[i][k]);
                    r[51]=std::max(pair_mass[i][j],pair_mass[i][k]);
                    r[52]=axes[j].m()+axes[k].m();
                    r[53]=std::min(v[j][0],v[k][0]);
                }
                for (int h=0; h<k; ++h) {
                    if (h==i) continue;
                    // Grade-four coefficient of the exterior product of the four momenta
                    const double det=v[i][0]*volume(v[j],v[k],v[h])-v[j][0]*volume(v[i],v[k],v[h])
                                    +v[k][0]*volume(v[i],v[j],v[h])-v[h][0]*volume(v[i],v[j],v[k]);
                    const double absdet=std::abs(det);
                    r[54]+=absdet;
                    r[55]=std::max<double>(r[55],absdet);
                    r[56]+=absdet*tw;
                    r[57]+=absdet/(1.e-12+v[i][0]*v[j][0]*v[k][0]*v[h][0]);
                }
            }
        }
        for (int f : {30,38})
            for (int t=2; t<8; ++t) r[f+t]/=std::max(1.e-12f,r[f]);
        r[58]=strongest_w;
        r[59]=jet.pt();
        r[60]=std::abs(jet.rap());
        r[61]=std::abs(volume(v[i],flow[i][1],flow[i][2]))/std::max(1.e-12,v[i][0]*flow[i][1][0]*flow[i][2][0]);
    }
    // A second edge contraction carries neighboring triangle responses to each root
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            if (j==i) continue;
            const double d=chord(v[i],v[j]), near=.2/(.2+d), w=band(pair_mass[i][j],fit[1],fit[3]);
            out[i][62]+=near*out[j][30];
            out[i][63]+=near*out[j][38];
            out[i][64]+=w*out[j][31];
            out[i][65]+=w*out[j][39];
            out[i][66]+=w*out[j][55];
            out[i][67]+=w*out[j][61];
        }
    }
    if (sheaf) transport::observables(n,v,flow,pair_mass,fit,sheaf);
    return n;
}
} // namespace geometry::simplices
