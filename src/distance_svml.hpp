#pragma once
#include <cmath>

// Copyright (C) 2021 Intel Corporation. SPDX-License-Identifier: BSD-3-Clause
// Polynomial and evaluation order adapted from numpy/SVML, revision
// 1c5260a61e7dce6be48073dfa96291edb0a11d79,
// linux/avx512/svml_z0_atan2_d_la.s. See third_party/svml-LICENSE.txt.
namespace skimagexx::detail {
template<class M> typename M::R svml_atan_polynomial(typename M::R r) {
    const auto r2=M::mul(r,r),r4=M::mul(r2,r2),r8=M::mul(r4,r4);
    auto p1=M::fma(r8,M::c(-0x1.4fdb537abc7a3p-16),M::c(-0x1.2b2ec618e4badp-7));
    auto p2=M::fma(r8,M::c(0x1.ced0a36665209p-13),M::c(0x1.14f4c661116a5p-6));
    auto p3=M::fma(r8,M::c(-0x1.2e67c93954c23p-10),M::c(-0x1.a5e83b081f69cp-6));
    auto p4=M::fma(r8,M::c(0x1.f5a1dae82afb3p-9),M::c(0x1.169980cb6ad4fp-5));
    p1=M::fma(p1,r8,M::c(-0x1.4efa2e563c1bcp-5));p2=M::fma(p2,r8,M::c(0x1.7ec0fbc50683bp-5));
    p3=M::fma(p3,r8,M::c(-0x1.ad261eaa09954p-5));p4=M::fma(p4,r8,M::c(0x1.e1749bd612dcfp-5));
    p1=M::fma(p1,r8,M::c(-0x1.11084009435e0p-4));p2=M::fma(p2,r8,M::c(0x1.3b12a49295651p-4));
    p3=M::fma(p3,r8,M::c(-0x1.745d009bada94p-4));p4=M::fma(p4,r8,M::c(0x1.c71c707f7d5b5p-4));
    p4=M::mul(r8,p4);p1=M::fma(p1,r8,M::c(-0x1.2492491ee55c7p-3));
    p2=M::fma(p2,r8,M::c(0x1.999999997ee34p-3));
    p1=M::fma(p1,r2,p2);p3=M::fma(p3,r8,M::c(-0x1.55555555553c5p-2));
    p3=M::fma(p3,r2,p4);p1=M::fma(p1,r4,p3);
    return M::fma(p1,r,r);
}
struct ScalarSvmlMath {
    using R=double;
    static R c(double x){return x;}
    static R mul(R x,R y){return x*y;}
    static R fma(R x,R y,R z){return std::fma(x,y,z);}
};
inline double svml_atan2(double y,double x) {
    const auto ax=std::abs(x),ay=std::abs(y);
    if(ax<0x1p-500 || ay<0x1p-500 || ax>0x1p500 || ay>0x1p500 || !std::isfinite(ax+ay))
        return std::atan2(y,x);
    const bool swap=ay>ax;
    const auto r=swap ? -ax/ay : ay/ax;
    auto z=svml_atan_polynomial<ScalarSvmlMath>(r)+(swap ? 0x1.921fb54442d18p+0 : 0.);
    z=std::copysign(z,x);
    if(x<=0)z+=0x1.921fb54442d18p+1;
    return std::copysign(z,y);
}
}
