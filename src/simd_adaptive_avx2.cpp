#include "simd.hpp"
#include <immintrin.h>
#include <type_traits>
namespace skimagexx::simd {
void adaptive_mean4(const double* input,double* output,std::size_t length,std::size_t window) {
    auto total=_mm256_setzero_pd();const auto divisor=_mm256_set1_pd(double(window));
    for(std::size_t k=0;k<window;++k)total=_mm256_add_pd(total,_mm256_loadu_pd(input+4*k));
    _mm256_storeu_pd(output,_mm256_div_pd(total,divisor));
    for(std::size_t x=1;x<length;++x){const auto difference=_mm256_sub_pd(_mm256_loadu_pd(input+4*(x+window-1)),_mm256_loadu_pd(input+4*(x-1)));total=_mm256_add_pd(total,difference);_mm256_storeu_pd(output+4*x,_mm256_div_pd(total,divisor));}
}
namespace {
template<class T> std::size_t mean_columns(const T* input,T* output,std::size_t length,std::size_t pitch,
    std::size_t window,const std::ptrdiff_t* mapping,std::size_t begin,std::size_t end,double cval) {
    const auto divisor=_mm256_set1_pd(double(window)),constant=_mm256_set1_pd(cval);
    auto load=[&](std::ptrdiff_t row,std::size_t column){
        if(row<0)return constant;
        const auto* p=input+std::size_t(row)*pitch+column;
        if constexpr(std::is_same_v<T,float>)return _mm256_cvtps_pd(_mm_loadu_ps(p));
        else return _mm256_loadu_pd(p);
    };
    auto save=[&](std::size_t row,std::size_t column,__m256d value){
        auto* p=output+row*pitch+column;
        if constexpr(std::is_same_v<T,float>)_mm_storeu_ps(p,_mm256_cvtpd_ps(value));
        else _mm256_storeu_pd(p,value);
    };
    auto column=begin;
    for(;column+4<=end;column+=4) {
        auto total=_mm256_setzero_pd();
        for(std::size_t k=0;k<window;++k)total=_mm256_add_pd(total,load(mapping[k],column));
        save(0,column,_mm256_div_pd(total,divisor));
        for(std::size_t x=1;x<length;++x){const auto difference=_mm256_sub_pd(load(mapping[x+window-1],column),load(mapping[x-1],column));total=_mm256_add_pd(total,difference);save(x,column,_mm256_div_pd(total,divisor));}
    }
    return column;
}
template<class T> struct V;
template<> struct V<float> {
    using R = __m256;
    static constexpr std::size_t lanes = 8;
    static R load(const float* p) { return _mm256_loadu_ps(p); }
    static void store(float* p, R x) { _mm256_storeu_ps(p, x); }
    static R low(R a, R b) { return _mm256_min_ps(a, b); }
    static R high(R a, R b) { return _mm256_max_ps(a, b); }
    static R set(float value) { return _mm256_set1_ps(value); }
    static R add(R a, R b) { return _mm256_add_ps(a, b); }
    static R sub(R a, R b) { return _mm256_sub_ps(a, b); }
    static R mul(R a, R b) { return _mm256_mul_ps(a, b); }
    static R div(R a, R b) { return _mm256_div_ps(a, b); }
    static R mean(const double* p, __m256d divisor) {
        const auto lo = _mm256_cvtpd_ps(_mm256_div_pd(_mm256_loadu_pd(p), divisor));
        const auto hi = _mm256_cvtpd_ps(_mm256_div_pd(_mm256_loadu_pd(p + 4), divisor));
        return _mm256_set_m128(hi, lo);
    }
};
template<> struct V<double> {
    using R = __m256d;
    static constexpr std::size_t lanes = 4;
    static R load(const double* p) { return _mm256_loadu_pd(p); }
    static void store(double* p, R x) { _mm256_storeu_pd(p, x); }
    static R low(R a, R b) { return _mm256_min_pd(a, b); }
    static R high(R a, R b) { return _mm256_max_pd(a, b); }
    static R set(double value) { return _mm256_set1_pd(value); }
    static R add(R a, R b) { return _mm256_add_pd(a, b); }
    static R sub(R a, R b) { return _mm256_sub_pd(a, b); }
    static R mul(R a, R b) { return _mm256_mul_pd(a, b); }
    static R div(R a, R b) { return _mm256_div_pd(a, b); }
    static R mean(const double* p, __m256d divisor) { return div(load(p), divisor); }
};
template<int Phase,int K,class M> void network(typename M::R (&values)[9]) {
    if constexpr(Phase<9) {
        if constexpr(K<8){const auto a=values[K],b=values[K+1];values[K]=M::low(a,b);values[K+1]=M::high(a,b);network<Phase,K+2,M>(values);}
        else network<Phase+1,(Phase+1)%2,M>(values);
    }
}
template<int K,class T> void load9(const T* input,std::size_t pitch,typename V<T>::R (&values)[9]) {
    if constexpr(K<9){values[K]=V<T>::load(input+std::size_t(K/3)*pitch+std::size_t(K%3));load9<K+1>(input,pitch,values);}
}
template<class T> std::size_t median9(const T* input,T* output,std::size_t count,std::size_t pitch) {
    using M=V<T>;std::size_t x=0;
    for(;x+M::lanes<=count;x+=M::lanes){typename M::R values[9];load9<0>(input+x,pitch,values);
        // Fixed odd/even sorting network: independent pixels occupy SIMD lanes.
        network<0,0,M>(values);
        M::store(output+x,values[4]);
    }return x;
}

template<class T> struct Vertical4;
template<> struct Vertical4<float> {
    using R = __m128;
    static R load(const float* p) { return _mm_loadu_ps(p); }
    static void store(float* p, R x) { _mm_storeu_ps(p, x); }
    static R mul(R a, R b) { return _mm_mul_ps(a, b); }
    static __m256d widen(R a) { return _mm256_cvtps_pd(a); }
    static R narrow(__m256d a) { return _mm256_cvtpd_ps(a); }
};
template<> struct Vertical4<double> : V<double> {
    static R widen(R a) { return a; }
    static R narrow(R a) { return a; }
};

template<bool Rolling, bool Emit, class T>
std::size_t ssim_vertical(const T* a, const T* b, const T* old_a, const T* old_b,
                          T* products, const T* old_products, double* sums, T* means,
                          std::size_t cols, double window) {
    using M = Vertical4<T>;
    const auto divisor = _mm256_set1_pd(window);
    std::size_t x = 0;
    for (; x + 4 <= cols; x += 4) {
        const auto ax = M::load(a + x), bx = M::load(b + x);
        const auto aa = M::mul(ax, ax), bb = M::mul(bx, bx), ab = M::mul(ax, bx);
        const typename M::R next[] = {ax, bx, aa, bb, ab};
        const auto update = [&](std::size_t m, typename M::R value, const T* previous) {
            auto delta = M::widen(value);
            if constexpr (Rolling) delta = _mm256_sub_pd(delta, M::widen(M::load(previous)));
            auto* sum = sums + m * cols + x;
            const auto total = _mm256_add_pd(_mm256_loadu_pd(sum), delta);
            _mm256_storeu_pd(sum, total);
            if constexpr (Emit) M::store(means + m * cols + x, M::narrow(_mm256_div_pd(total, divisor)));
        };
        if constexpr (Rolling) {
            update(0, ax, old_a + x);
            update(1, bx, old_b + x);
            update(2, aa, old_products + x);
            update(3, bb, old_products + cols + x);
            update(4, ab, old_products + 2 * cols + x);
        } else {
            for (std::size_t m = 0; m < 5; ++m) update(m, next[m], nullptr);
        }
        M::store(products + x, aa);
        M::store(products + cols + x, bb);
        M::store(products + 2 * cols + x, ab);
    }
    return x;
}
}
std::size_t adaptive_mean_columns(const void* input,void* output,std::size_t length,std::size_t pitch,
    std::size_t window,const std::ptrdiff_t* mapping,std::size_t begin,std::size_t end,double cval,bool single) {
    return single ? mean_columns(static_cast<const float*>(input),static_cast<float*>(output),length,pitch,window,mapping,begin,end,cval)
                  : mean_columns(static_cast<const double*>(input),static_cast<double*>(output),length,pitch,window,mapping,begin,end,cval);
}
std::size_t adaptive_median9(const void* input,void* output,std::size_t count,std::size_t pitch,bool single) {
    return single ? median9(static_cast<const float*>(input),static_cast<float*>(output),count,pitch) : median9(static_cast<const double*>(input),static_cast<double*>(output),count,pitch);
}
std::size_t adaptive_offset(const void* input,void* output,std::size_t count,bool source_single,bool target_single,double value) {
    std::size_t i=0;
    if(target_single && source_single){auto* dst=static_cast<float*>(output);const auto* src=static_cast<const float*>(input);const auto scalar=_mm256_set1_ps(float(value));for(;i+8<=count;i+=8)_mm256_storeu_ps(dst+i,_mm256_sub_ps(_mm256_loadu_ps(src+i),scalar));}
    else if(!target_single){auto* dst=static_cast<double*>(output);const auto scalar=_mm256_set1_pd(value);for(;i+4<=count;i+=4){const auto x=source_single ? _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(input)+i)) : _mm256_loadu_pd(static_cast<const double*>(input)+i);_mm256_storeu_pd(dst+i,_mm256_sub_pd(x,scalar));}}
    return i;
}

template<class T>
std::size_t ssim_vertical_row(const T* a, const T* b, const T* old_a, const T* old_b,
                              T* products, const T* old_products, double* sums, T* means,
                              std::size_t cols, double window) {
    if (means)
        return ssim_vertical<true, true>(a, b, old_a, old_b, products, old_products, sums, means, cols, window);
    if (old_a)
        return ssim_vertical<true, false>(a, b, old_a, old_b, products, old_products, sums, means, cols, window);
    return ssim_vertical<false, false>(a, b, old_a, old_b, products, old_products, sums, means, cols, window);
}

template<class T>
std::size_t ssim_score_row(const double* sums, T* values, std::size_t width, double window,
                           T norm, T c1, T c2) {
    using M = V<T>;
    const auto divisor = _mm256_set1_pd(window);
    const auto norm_v = M::set(norm), c1_v = M::set(c1), c2_v = M::set(c2), two = M::set(T(2));
    std::size_t x = 0;
    for (; x + M::lanes <= width; x += M::lanes) {
        const auto a = M::mean(sums + x, divisor), b = M::mean(sums + width + x, divisor);
        const auto aa = M::mul(a, a), bb = M::mul(b, b), ab = M::mul(a, b);
        const auto vx = M::mul(norm_v, M::sub(M::mean(sums + 2 * width + x, divisor), aa));
        const auto vy = M::mul(norm_v, M::sub(M::mean(sums + 3 * width + x, divisor), bb));
        const auto vxy = M::mul(norm_v, M::sub(M::mean(sums + 4 * width + x, divisor), ab));
        const auto a1 = M::add(M::mul(M::mul(two, a), b), c1_v);
        const auto a2 = M::add(M::mul(two, vxy), c2_v);
        const auto b1 = M::add(M::add(aa, bb), c1_v);
        const auto b2 = M::add(M::add(vx, vy), c2_v);
        M::store(values + x, M::div(M::mul(a1, a2), M::mul(b1, b2)));
    }
    return x;
}

template std::size_t ssim_vertical_row<float>(const float*, const float*, const float*, const float*,
    float*, const float*, double*, float*, std::size_t, double);
template std::size_t ssim_vertical_row<double>(const double*, const double*, const double*, const double*,
    double*, const double*, double*, double*, std::size_t, double);
template std::size_t ssim_score_row<float>(const double*, float*, std::size_t, double, float, float, float);
template std::size_t ssim_score_row<double>(const double*, double*, std::size_t, double, double, double, double);
}
