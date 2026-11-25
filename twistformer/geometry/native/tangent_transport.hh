// Cellular sheaf of transverse radiation vectors on the rest-frame direction sphere
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace geometry::transport {

using Spatial = std::array<double,3>;
using Four = std::array<double,4>;
constexpr int fields=60;

inline double inner(const Spatial &a,const Spatial &b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

inline Spatial cross(const Spatial &a,const Spatial &b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}

// Compare endpoint sections in a parallel edge frame and pool triangular holonomy
inline void observables(int n,const std::array<Four,8> &v,const std::array<std::array<Four,3>,8> &flow,
                        const double pair_mass[8][8],const double *fit,float *out) {
    std::array<Spatial,8> axis{};
    std::array<std::array<Spatial,8>,3> section{};
    for (int i=0;i<n;++i) {
        axis[i]={v[i][1],v[i][2],v[i][3]};
        const double norm=std::sqrt(inner(axis[i],axis[i]));
        if (norm<1.e-12) continue;
        for (double &x:axis[i]) x/=norm;
        for (int s=0;s<3;++s) {
            const auto &f=flow[i][s];
            const Spatial q{f[1],f[2],f[3]};
            const double radial=inner(q,axis[i]);
            for (int k=0;k<3;++k) section[s][i][k]=(q[k]-radial*axis[i][k])/std::max(1.e-12,f[0]);
        }
    }
    for (int i=0;i<n;++i) {
        for (int j=0;j<i;++j) {
            auto normal=cross(axis[i],axis[j]);
            const double sin_angle=std::sqrt(inner(normal,normal));
            if (sin_angle<1.e-8) continue;
            for (double &x:normal) x/=sin_angle;
            const auto ti=cross(normal,axis[i]), tj=cross(normal,axis[j]);
            const double pull=(pair_mass[i][j]-fit[1])/fit[3];
            const double weights[]{v[i][0]*v[j][0],1./(1.+pull*pull)};
            for (int s=0;s<3;++s) {
                const auto &a=section[s][i], &b=section[s][j];
                const double ai=inner(a,ti),bj=inner(b,tj),an=inner(a,normal),bn=inner(b,normal);
                for (int w=0;w<2;++w) {
                    float *f=out+8*(2*s+w);
                    const double weight=weights[w];
                    f[0]+=weight*(ai-bj)*(ai-bj);
                    f[1]+=weight*(an-bn)*(an-bn);
                    f[2]+=weight*ai*bj;
                    f[3]+=weight*an*bn;
                    f[4]+=weight*(ai-bj);
                    f[5]+=weight*std::abs(ai*bn-an*bj);
                    f[6]+=weight*(ai+ bj)*(ai+bj);
                    f[7]=std::max<double>(f[7],weight*((ai-bj)*(ai-bj)+(an-bn)*(an-bn)));
                }
            }
            for (int k=0;k<j;++k) {
                const double det=sin_angle*inner(normal,axis[k]);
                const double denominator=1.+inner(axis[i],axis[j])+inner(axis[j],axis[k])+inner(axis[k],axis[i]);
                // Squared rotation induced by transport around the spherical triangle
                const double half_sine2=det*det/std::max(1.e-18,det*det+denominator*denominator);
                const double weights3[]{v[i][0]*v[j][0]*v[k][0],1.};
                for (int w=0;w<2;++w) {
                    float *f=out+48+6*w;
                    const double h=weights3[w]*half_sine2;
                    f[0]+=h;
                    f[1]+=h*half_sine2;
                    f[2]=std::max<double>(f[2],h);
                    for (int s=0;s<3;++s)
                        f[3+s]+=h*(inner(section[s][i],section[s][i])+inner(section[s][j],section[s][j])
                                   +inner(section[s][k],section[s][k]));
                }
            }
        }
    }
}
} // namespace geometry::transport
