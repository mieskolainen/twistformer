// Persistent homology and Lorentz blade volumes on filtered microjet complexes
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "lorentz_simplices.hh"
#include <cstdint>

namespace geometry::complexes {

using simplices::Vector;
constexpr int points=8, barcode_fields=11;
constexpr int view_fields=1+3*barcode_fields+3*4*3, output_fields=4*view_fields;

struct Cell { int mask; double level, volume, angular_volume; };

inline double spatial_angle(const Vector &a,const Vector &b) {
    const double aa=a[1]*a[1]+a[2]*a[2]+a[3]*a[3],bb=b[1]*b[1]+b[2]*b[2]+b[3]*b[3];
    if (aa*bb<1.e-28) return .5;
    return .5*(1.-std::clamp((a[1]*b[1]+a[2]*b[2]+a[3]*b[3])/std::sqrt(aa*bb),-1.,1.));
}

// Form an energy-weighted angular partition using only jet rest-frame scalars
inline std::array<Vector,points> angular_partition(const JetList &input,const Momentum &jet,double mass,int n) {
    thread_local std::vector<Vector> v;
    thread_local std::vector<double> distance;
    thread_local std::vector<int> owner;
    thread_local std::vector<unsigned char> used;
    v.resize(input.size());
    distance.assign(input.size(),2.);
    owner.assign(input.size(),0);
    for (std::size_t i=0;i<input.size();++i) v[i]=simplices::rest(input[i],jet,mass);
    used.assign(input.size(),0);
    for (int s=0;s<n;++s) {
        int seed=-1;
        double best=-1.;
        for (int i=0;i<int(input.size());++i) {
            const double priority=v[i][0]*distance[i];
            if (!used[i] && priority>best) { seed=i; best=priority; }
        }
        if (seed<0) break;
        used[seed]=true;
        for (int i=0;i<int(input.size());++i) {
            const double d=spatial_angle(v[i],v[seed]);
            if (d<distance[i] || i==seed) { distance[i]=d; owner[i]=s; }
        }
    }
    std::array<Vector,points> result{};
    for (std::size_t i=0;i<input.size();++i)
        for (int k=0;k<4;++k) result[owner[i]][k]+=v[i][k];
    return result;
}

inline void interval(float *row,double birth,double death) {
    const double length=death-birth;
    if (length<=1.e-10) return;
    row[0]+=1.f;
    row[1]+=length;
    row[2]+=length*length;
    if (length>row[3]) {
        row[4]=row[3];
        row[3]=length;
        row[5]=birth;
        row[6]=death;
    } else row[4]=std::max<double>(row[4],length);
    constexpr double cuts[]{.10,.25,.50,.75};
    for (int i=0;i<4;++i) if (birth<=cuts[i] && cuts[i]<death) row[7+i]+=1.f;
}

inline int bit_index(std::uint64_t x) {
    // Each nonzero word is a boundary on at most 56 triangles
    int index=0;
    if (x>>32) { x>>=32; index+=32; }
    if (x>>16) { x>>=16; index+=16; }
    if (x>>8) { x>>=8; index+=8; }
    if (x>>4) { x>>=4; index+=4; }
    if (x>>2) { x>>=2; index+=2; }
    return index+int(x>>1);
}

// Reduce boundary columns over F2 through the tetrahedra of the flag complex
inline void filtration(const std::array<Vector,points> &v,int n,bool invariant_mass,float *out) {
    double gram[points][points]{}, edge[points][points]{}, scale=0.;
    for (int i=0;i<n;++i) for (int j=0;j<=i;++j) {
        gram[i][j]=gram[j][i]=simplices::dot(v[i],v[j]);
        if (j==i) continue;
        edge[j][i]=invariant_mass ? std::sqrt(std::max(0.,2.*gram[i][j])) : spatial_angle(v[i],v[j]);
        scale=std::max(scale,edge[j][i]);
    }
    out[0]=scale;
    if (scale<1.e-12) return;
    std::array<std::array<Cell,70>,4> cells;
    std::array<int,4> sizes{};
    auto add=[&](int dim,int mask,double max_edge,double volume,double energy) {
        cells[dim][sizes[dim]++]={mask,max_edge/scale,volume,volume/std::max(1.e-20,energy)};
    };
    for (int i=0;i<n;++i) {
        add(0,1<<i,0.,0.,1.);
        for (int j=i+1;j<n;++j) {
            const int pair=(1<<i)|(1<<j);
            const double ab=gram[i][j],aa=gram[i][i],bb=gram[j][j],energy=v[i][0]*v[j][0];
            add(1,pair,edge[i][j],std::sqrt(std::max(0.,ab*ab-aa*bb)),energy);
            for (int k=j+1;k<n;++k) {
                const double ac=gram[i][k],bc=gram[j][k],cc=gram[k][k];
                const double det=aa*bb*cc+2*ab*ac*bc-aa*bc*bc-bb*ac*ac-cc*ab*ab;
                const double level=std::max({edge[i][j],edge[i][k],edge[j][k]});
                const int face=pair|(1<<k);
                add(2,face,level,std::sqrt(std::max(0.,det)),energy*v[k][0]);
                for (int h=k+1;h<n;++h) {
                    const double volume=v[i][0]*simplices::volume(v[j],v[k],v[h])-v[j][0]*simplices::volume(v[i],v[k],v[h])
                                       +v[k][0]*simplices::volume(v[i],v[j],v[h])-v[h][0]*simplices::volume(v[i],v[j],v[k]);
                    add(3,face|(1<<h),std::max({level,edge[i][h],edge[j][h],edge[k][h]}),std::abs(volume),energy*v[k][0]*v[h][0]);
                }
            }
        }
    }
    std::array<int,1<<points> order{};
    for (int dim=0;dim<4;++dim) {
        auto &group=cells[dim];
        std::sort(group.begin(),group.begin()+sizes[dim],[](const Cell &a,const Cell &b) {
            return a.level==b.level ? a.mask<b.mask : a.level<b.level;
        });
        std::array<std::uint64_t,56> pivots{};
        for (int c=0;c<sizes[dim];++c) {
            const auto &cell=group[c];
            order[cell.mask]=c;
            if (!dim) continue;
            std::uint64_t boundary=0;
            for (int bits=cell.mask;bits;bits&=bits-1) {
                const int opposite=cell.mask^(bits&-bits);
                boundary|=std::uint64_t{1}<<order[opposite];
            }
            while (boundary) {
                const int p=bit_index(boundary);
                if (pivots[p]) boundary^=pivots[p];
                else {
                    pivots[p]=boundary;
                    interval(out+1+(dim-1)*barcode_fields,cells[dim-1][p].level,cell.level);
                    break;
                }
            }
            float *moments=out+1+3*barcode_fields+(dim-1)*12;
            for (int b=0;b<4;++b) if (cell.level<=.25*(b+1)) {
                moments[3*b]+=1.f;
                moments[3*b+1]+=cell.volume;
                moments[3*b+2]+=cell.angular_volume;
            }
        }
    }
}

// Evaluate rest-frame and exclusive kT point sets with angle and mass filtrations
inline void project(const JetList &input,const ClusterTree &tree,float *&out) {
    std::fill_n(out,output_fields,0.f);
    const auto jet=tree.exclusive_jets(1).front();
    const double mass=jet.m();
    if (mass<.01 || input.size()<2) { out+=output_fields; return; }
    const int n=std::min(int(input.size()),points);
    std::array<Vector,points> v{};
    const auto axes=tree.exclusive_jets(n);
    for (int i=0;i<n;++i) v[i]=simplices::rest(axes[i],jet,mass);
    for (int view=0;view<2;++view) {
        if (view) v=angular_partition(input,jet,mass,n);
        for (bool invariant_mass:{false,true}) {
            filtration(v,n,invariant_mass,out);
            out+=view_fields;
        }
    }
}
} // namespace geometry::complexes
