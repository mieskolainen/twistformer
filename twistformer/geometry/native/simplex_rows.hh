// Algebraic vertex readout for Lorentz simplices
//
// mikael.mieskolainen@cern.ch (2026)

#pragma once

#include "lorentz_simplices.hh"


namespace geometry::simplices {

constexpr int output_fields=transport::fields+5*fields;

inline void project(const JetList &input,const ClusterTree &tree,const double *fit,float *&out) {
    std::fill_n(out,output_fields,0.f);
    Rows rows{};
    const int n=messages(input,tree,fit,rows,out);
    out+=transport::fields;
    for (int f=0;f<fields;++f) {
        double sum=0.,maximum=0.;
        for (int i=0;i<n;++i) {
            sum+=rows[i][0]*rows[i][f];
            maximum=i ? std::max<double>(maximum,rows[i][f]) : rows[i][f];
        }
        *out++=sum;
        *out++=maximum;
        for (int i=0;i<3;++i) *out++=rows[i][f];
    }
}
} // namespace geometry::simplices
