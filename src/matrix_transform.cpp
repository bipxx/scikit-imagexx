#include "skimagexx/matrix_transform.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#endif

namespace skimagexx {
namespace {
struct Span { std::uintptr_t lo, hi; };
bool span(const MatrixReadView& v, Span& out) {
    constexpr auto max = std::numeric_limits<std::ptrdiff_t>::max();
    if (!v.data || !v.allocation || !v.rows || !v.columns ||
        v.row_stride == -max-1 || v.column_stride == -max-1) return false;
    const auto r = v.row_stride < 0 ? -v.row_stride : v.row_stride;
    const auto c = v.column_stride < 0 ? -v.column_stride : v.column_stride;
    if ((r && v.rows-1 > std::size_t(max/r)) ||
        (c && v.columns-1 > std::size_t(max/c))) return false;
    const auto dr=std::ptrdiff_t(v.rows-1)*r, dc=std::ptrdiff_t(v.columns-1)*c;
    if (dr > max-dc-8) return false;
    const auto below=(v.row_stride < 0 ? dr : 0)+(v.column_stride < 0 ? dc : 0);
    const auto above=(v.row_stride > 0 ? dr : 0)+(v.column_stride > 0 ? dc : 0)+8;
    const auto p=reinterpret_cast<std::uintptr_t>(v.data);
    const auto base=reinterpret_cast<std::uintptr_t>(v.allocation);
    if (p < std::uintptr_t(below) || p > UINTPTR_MAX-std::uintptr_t(above) ||
        base > UINTPTR_MAX-v.allocation_bytes) return false;
    out={p-std::uintptr_t(below),p+std::uintptr_t(above)};
    return out.lo >= base && out.hi <= base+v.allocation_bytes;
}
bool intersects(Span a,Span b) { return a.lo < b.hi && b.lo < a.hi; }
double load(const MatrixReadView& v,std::size_t i,std::size_t j) {
    double value;
    std::memcpy(&value,static_cast<const unsigned char*>(v.data)+
        std::ptrdiff_t(i)*v.row_stride+std::ptrdiff_t(j)*v.column_stride,8);
    return value;
}
bool admitted(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,8);bits &= UINT64_C(0x7fffffffffffffff);
    // Absolute range [2^-200,2^20], plus either sign of zero. This makes all
    // products, cancellations and guarded divisions normal and finite.
    return bits==0 || (bits>=UINT64_C(0x3370000000000000) && bits<=UINT64_C(0x4130000000000000));
}
bool gradual_nearest() {
    if (std::fegetround()!=FE_TONEAREST) return false;
#if defined(__x86_64__) || defined(_M_X64)
    if (_mm_getcsr() & ((1u<<15)|(1u<<6))) return false;
#endif
    return true;
}
bool fma_available() {
#if defined(SKIMAGEXX_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    if (!has_avx2()) return false; // includes OS AVX context + user SIMD policy
#ifdef _MSC_VER
    int registers[4];__cpuid(registers,1);return (registers[2]&(1<<12))!=0;
#else
    return __builtin_cpu_supports("fma");
#endif
#else
    return false;
#endif
}
double dot(const double* x,const double* m,std::size_t d,MatrixArithmeticProfile profile,std::size_t component) {
    if (profile==MatrixArithmeticProfile::cooperlake0321_gemv_nontransposed) {
        // Exact pinned 0.3.21 ELF: K3 scalar tail and K5 lda5 denominator
        // use +0-seeded FMA. The K4/K5 packed rows instead seed product1.
        if (d==2 || (d==4 && component==4)) {
            double value=0.;
            for(std::size_t k=0;k<d;++k)value=std::fma(m[k],x[k],value);
            value=std::fma(m[d],1.,value);
            return std::fma(1.,value,0.);
        }
        double value=m[1]*x[1];
        value=std::fma(m[0],x[0],value);
        value=std::fma(m[2],x[2],value);
        value=std::fma(m[3],d==3 ? 1.:x[3],value);
        value=std::fma(1.,value,0.);
        if (d==4) { const double tail=m[4]*1.;value=value+tail; }
        return value;
    }
    if (profile==MatrixArithmeticProfile::haswell_gemv_transposed) {
        // Small K=3 scalar remainder: actual audited GCC contraction starts
        // from product1, fuses product0, then product2. Larger K uses independent
        // lane products and (lane0+lane2)+(lane1+lane3), followed by tail1.
        if (d==2) return std::fma(1.,m[2],std::fma(x[0],m[0],x[1]*m[1]))+0.;
        const double p0=x[0]*m[0]+0.,p1=x[1]*m[1]+0.,p2=x[2]*m[2]+0.;
        const double p3=(d==3 ? 1.:x[3])*m[3]+0.;
        double value=((p0+p2)+(p1+p3))+0.;
        if (d==4)value=value+1.*m[4];
        return value;
    }
    if (profile==MatrixArithmeticProfile::haswell_gemv_nontransposed && d>=3) {
        // Packed rows of the nontransposed microkernel: two independent FMA
        // pairs, then add. The K=5 final output row instead uses scalar C's
        // left-associated separate four-product expression.
        double value;
        if (component<4) {
            value=(std::fma(x[2],m[2],x[0]*m[0])+
                std::fma(d==3 ? 1.:x[3],m[3],x[1]*m[1]))+0.;
        }else {
            value=(((x[0]*m[0]+x[1]*m[1])+x[2]*m[2])+x[3]*m[3])+0.;
        }
        if (d==4)value=value+1.*m[4];
        return value+0.;
    }
    const bool fused=profile==MatrixArithmeticProfile::haswell_gemm_fma;
    double value=0.;
    for (std::size_t j=0;j<d;++j)
        value=fused ? std::fma(x[j],m[j],value) : value+x[j]*m[j];
    // The homogeneous input coordinate is literally +1, including in FMA.
    value=fused ? std::fma(1.,m[d],value) : value+1.*m[d];
    // OpenBLAS beta=0 initializes C to +0 and accumulates alpha*dot.
    return profile==MatrixArithmeticProfile::numpy124_noblas ||
        profile==MatrixArithmeticProfile::numpy254_special_noblas ? value : value+0.;
}
struct Environment {
    std::fenv_t saved;
    Environment() { std::feholdexcept(&saved); }
    ~Environment() { std::fesetenv(&saved); }
};
}

MatrixMapStatus matrix_transform_into(const MatrixReadView& p,const MatrixReadView& m,
    const MatrixWriteView& output,MatrixArithmeticProfile profile) {
    if (profile==MatrixArithmeticProfile::cooperlake0321_gemv_nontransposed)
        return MatrixMapStatus::unsupported_profile;
    return matrix_transform_into_v2(p,m,output,profile);
}

MatrixMapStatus matrix_transform_into_v2(const MatrixReadView& p,const MatrixReadView& m,
    const MatrixWriteView& output,MatrixArithmeticProfile profile) {
    const bool fused=profile==MatrixArithmeticProfile::haswell_gemm_fma;
    const bool cooper=profile==MatrixArithmeticProfile::cooperlake0321_gemv_nontransposed;
    const bool gemv=profile==MatrixArithmeticProfile::haswell_gemv_transposed ||
        profile==MatrixArithmeticProfile::haswell_gemv_nontransposed || cooper;
    if (!fused && !gemv && profile!=MatrixArithmeticProfile::numpy124_noblas &&
        profile!=MatrixArithmeticProfile::numpy254_special_noblas)
        return MatrixMapStatus::unsupported_profile;
    const auto d=p.columns,n=p.rows;
    if (d<2 || d>4 || !n || (fused && n==1) || (gemv && n!=1) ||
        (profile==MatrixArithmeticProfile::numpy254_special_noblas && n!=1) || m.rows!=d+1 || m.columns!=d+1 ||
        output.rows!=n || output.columns!=d) return MatrixMapStatus::unsupported_shape;
    if (n>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(d*8))
        return MatrixMapStatus::invalid_view;
    const auto bytes=n*d*8;
    Span ps,ms;
    const auto op=reinterpret_cast<std::uintptr_t>(output.data);
    if (!span(p,ps) || !span(m,ms) || !output.data || op%alignof(double) ||
        output.allocation_bytes<bytes || op>UINTPTR_MAX-bytes)
        return MatrixMapStatus::invalid_view;
    const Span os{op,op+bytes};
    if (intersects(os,ps) || intersects(os,ms)) return MatrixMapStatus::overlap;
    if (!gradual_nearest()) return MatrixMapStatus::unsafe_fp_environment;
    double coefficients[25];
    for (std::size_t i=0;i<=d;++i)for (std::size_t j=0;j<=d;++j) {
        const auto v=load(m,i,j);
        if (!admitted(v)) return MatrixMapStatus::nonfinite_or_range;
        coefficients[i*(d+1)+j]=v;
    }
    // Entire value/domain scan is before any store, including on late errors.
    for (std::size_t i=0;i<n;++i)for (std::size_t j=0;j<d;++j)
        if (!admitted(load(p,i,j))) return MatrixMapStatus::nonfinite_or_range;
    Environment held;
    for (std::size_t i=0;i<n;++i) {
        double x[4];for(std::size_t j=0;j<d;++j)x[j]=load(p,i,j);
        const double den=dot(x,coefficients+d*(d+1),d,profile,d);
        if (den!=0. && std::abs(den)<0x1p-400) return MatrixMapStatus::unsafe_divisor;
    }
    std::fenv_t caller;std::fegetenv(&caller);
    const bool vectorized=fused && fma_available();
    const bool vectorized_cooper=cooper && fma_available();
    detail::partition(n,detail::active_workers(n*d),[&](auto begin,auto end,auto) {
        Environment thread_held;std::fesetenv(&caller);
        std::size_t first=begin;
#ifdef SKIMAGEXX_AVX2
        if (vectorized_cooper && begin<end) {
            simd::matrix_transform_cooperlake_gemv_n(p,coefficients,
                static_cast<double*>(output.data),d);
            return;
        }
        if (vectorized) first+=simd::matrix_transform_fma(p,coefficients,
            static_cast<double*>(output.data),begin,end,d);
#else
        (void)vectorized;
        (void)vectorized_cooper;
#endif
        for (std::size_t i=first;i<end;++i) {
            double x[4];for(std::size_t j=0;j<d;++j)x[j]=load(p,i,j);
            double den=dot(x,coefficients+d*(d+1),d,profile,d);
            if (den==0.)den=std::numeric_limits<double>::epsilon();
            for (std::size_t j=0;j<d;++j)
                static_cast<double*>(output.data)[i*d+j]=dot(x,coefficients+j*(d+1),d,profile,j)/den;
        }
    });
    return MatrixMapStatus::ok;
}
}
