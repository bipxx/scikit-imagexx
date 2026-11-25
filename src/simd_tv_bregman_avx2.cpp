#include <immintrin.h>
#include <cstddef>
namespace skimagexx::simd {
void tv_bregman_copy4(const float* input,float* dx,float* dy,float* bx,float* by,std::size_t n) {
    std::size_t i=0;for(;i+8<=n;i+=8){const auto v=_mm256_loadu_ps(input+i);_mm256_storeu_ps(dx+i,v);_mm256_storeu_ps(dy+i,v);_mm256_storeu_ps(bx+i,v);_mm256_storeu_ps(by+i,v);}
    for(;i<n;++i){const float v=input[i];dx[i]=v;dy[i]=v;bx[i]=v;by[i]=v;}
}
void tv_bregman_copy4(const double* input,double* dx,double* dy,double* bx,double* by,std::size_t n) {
    std::size_t i=0;for(;i+4<=n;i+=4){const auto v=_mm256_loadu_pd(input+i);_mm256_storeu_pd(dx+i,v);_mm256_storeu_pd(dy+i,v);_mm256_storeu_pd(bx+i,v);_mm256_storeu_pd(by+i,v);}
    for(;i<n;++i){const double v=input[i];dx[i]=v;dy[i]=v;bx[i]=v;by[i]=v;}
}

namespace {
__m256d shrink(__m256d value, __m256d threshold) {
    const auto negative = _mm256_xor_pd(threshold, _mm256_set1_pd(-0.0));
    const auto high = _mm256_cmp_pd(value, threshold, _CMP_GT_OQ);
    const auto low = _mm256_cmp_pd(value, negative, _CMP_LT_OQ);
    const auto active = _mm256_or_pd(high,low);
    const auto offset = _mm256_blendv_pd(threshold,negative,high);
    return _mm256_add_pd(_mm256_and_pd(value,active),_mm256_and_pd(offset,active));
}

template<class T> struct Wave;

template<> struct Wave<double> {
    using R = __m256d;
    using Index = __m128i;
    static constexpr int lanes = 4;
    template<int> static Index index(int step) { return _mm_setr_epi32(0,step,2*step,3*step); }
    static R load(const double* p, Index i) { return _mm256_i32gather_pd(p,i,8); }
    static R splat(double x) { return _mm256_set1_pd(x); }
    static R add(R a,R b) { return _mm256_add_pd(a,b); }
    static R sub(R a,R b) { return _mm256_sub_pd(a,b); }
    static R mul(R a,R b) { return _mm256_mul_pd(a,b); }
    static R div(R a,R b) { return _mm256_div_pd(a,b); }
    static R root(R x) { return _mm256_sqrt_pd(x); }
    static R quotient(R num,R factor) { return div(num,add(factor,splat(1.0))); }
    static R threshold(R x,R limit) { return shrink(x,limit); }
    template<int> static R north(R x,double top) {
        return _mm256_blend_pd(_mm256_permute4x64_pd(x,_MM_SHUFFLE(2,1,0,0)),splat(top),1);
    }
    template<int> static R south(R x,double bottom) {
        return _mm256_blend_pd(_mm256_permute4x64_pd(x,_MM_SHUFFLE(3,3,2,1)),splat(bottom),8);
    }
    static void store(double* p,R x) { _mm256_storeu_pd(p,x); }
    template<int> static void scatter(double* p,std::ptrdiff_t step,R x) {
        const auto low = _mm256_castpd256_pd128(x), high = _mm256_extractf128_pd(x,1);
        _mm_store_sd(p,low);
        _mm_storeh_pd(p+step,low);
        _mm_store_sd(p+2*step,high);
        _mm_storeh_pd(p+3*step,high);
    }
};

template<> struct Wave<float> {
    using R = __m256;
    using Index = __m256i;
    static constexpr int lanes = 8;
    template<int Height> static Index index(int step) {
        if constexpr(Height==4) return _mm256_setr_epi32(0,step,2*step,3*step,0,step,2*step,3*step);
        else return _mm256_setr_epi32(0,step,2*step,3*step,4*step,5*step,6*step,7*step);
    }
    static R load(const float* p, Index i) { return _mm256_i32gather_ps(p,i,4); }
    static R splat(float x) { return _mm256_set1_ps(x); }
    static R add(R a,R b) { return _mm256_add_ps(a,b); }
    static R sub(R a,R b) { return _mm256_sub_ps(a,b); }
    static R mul(R a,R b) { return _mm256_mul_ps(a,b); }
    static R div(R a,R b) { return _mm256_div_ps(a,b); }
    static __m256d lo(R x) { return _mm256_cvtps_pd(_mm256_castps256_ps128(x)); }
    static __m256d hi(R x) { return _mm256_cvtps_pd(_mm256_extractf128_ps(x,1)); }
    static R pack(__m256d a,__m256d b) {
        return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(a)),_mm256_cvtpd_ps(b),1);
    }
    static R root(R x) { return pack(_mm256_sqrt_pd(lo(x)),_mm256_sqrt_pd(hi(x))); }
    static R quotient(R num,R factor) {
        const auto one = _mm256_set1_pd(1.0);
        return pack(_mm256_div_pd(lo(num),_mm256_add_pd(lo(factor),one)),
                    _mm256_div_pd(hi(num),_mm256_add_pd(hi(factor),one)));
    }
    static R threshold(R x,__m256d limit) { return pack(shrink(lo(x),limit),shrink(hi(x),limit)); }
    template<int Height> static R north(R x,float top) {
        if constexpr(Height==4) {
            return _mm256_blend_ps(_mm256_permute_ps(x,_MM_SHUFFLE(2,1,0,0)),splat(top),0x11);
        }else {
            const auto shift = _mm256_setr_epi32(0,0,1,2,3,4,5,6);
            return _mm256_blend_ps(_mm256_permutevar8x32_ps(x,shift),splat(top),1);
        }
    }
    template<int Height> static R south(R x,float bottom) {
        if constexpr(Height==4) {
            return _mm256_blend_ps(_mm256_permute_ps(x,_MM_SHUFFLE(3,3,2,1)),splat(bottom),0x88);
        }else {
            const auto shift = _mm256_setr_epi32(1,2,3,4,5,6,7,7);
            return _mm256_blend_ps(_mm256_permutevar8x32_ps(x,shift),splat(bottom),128);
        }
    }
    static void store(float* p,R x) { _mm256_storeu_ps(p,x); }
    static void scatter4(float* p,std::ptrdiff_t step,__m128 x) {
        _mm_store_ss(p,x);
        _mm_store_ss(p+step,_mm_shuffle_ps(x,x,_MM_SHUFFLE(1,1,1,1)));
        _mm_store_ss(p+2*step,_mm_movehl_ps(x,x));
        _mm_store_ss(p+3*step,_mm_shuffle_ps(x,x,_MM_SHUFFLE(3,3,3,3)));
    }
    template<int Height> static void scatter(float* p,std::ptrdiff_t step,R x) {
        scatter4(p,step,_mm256_castps256_ps128(x));
        if constexpr(Height==8) scatter4(p+4*step,step,_mm256_extractf128_ps(x,1));
    }
};

template<class T,int Height,bool Isotropic>
void wavefront(const T* input,T* out,T* dx,T* dy,T* bx,T* by,T* squared,
               std::size_t cols,T weight,T lam,T norm) {
    using V = Wave<T>;
    constexpr auto lanes = V::lanes;
    const std::ptrdiff_t stride = std::ptrdiff_t(cols)+2;
    const auto step = stride-1;
    const auto offset = V::template index<Height>(int(step)), source_offset = V::template index<Height>(int(cols)-1);
    const auto weight_v = V::splat(weight), lam_v = V::splat(lam), norm_v = V::splat(norm);
    const auto threshold = _mm256_set1_pd(Isotropic ? 0.0 : 1.0/double(lam));

    // Lane k is at (k, diagonal-k). West is the previous diagonal's same
    // lane, north its previous lane. East and south still hold old values.
    const auto first = Height-1;
    auto previous = V::load(out+first,offset);
    auto west = V::load(out+first-1,offset);
    auto west_dx = V::load(dx+first-1,offset);
    auto west_bx = V::load(bx+first-1,offset);
    auto north_dy = V::load(dy+first-1,offset);
    auto north_by = V::load(by+first-1,offset);
    for(std::ptrdiff_t diagonal=first;diagonal<std::ptrdiff_t(cols);++diagonal) {
        const auto east = V::load(out+diagonal+1,offset);
        const auto south = V::template south<Height>(east,out[diagonal+Height*step+1]);
        const auto dxx = V::load(dx+diagonal,offset);
        const auto dyy = V::load(dy+diagonal,offset);
        const auto bxx = V::load(bx+diagonal,offset);
        const auto byy = V::load(by+diagonal,offset);
        const auto ux = V::sub(east,previous), uy = V::sub(south,previous);

        auto total = V::add(south,V::template north<Height>(west,out[diagonal-stride]));
        total = V::add(total,east);
        total = V::add(total,west);
        total = V::add(total,west_dx);
        total = V::sub(total,dxx);
        total = V::add(total,V::template north<Height>(north_dy,dy[diagonal-stride]));
        total = V::sub(total,dyy);
        total = V::sub(total,west_bx);
        total = V::add(total,bxx);
        total = V::sub(total,V::template north<Height>(north_by,by[diagonal-stride]));
        total = V::add(total,byy);
        const auto source = V::load(input+diagonal,source_offset);
        const auto unew = V::div(V::add(V::mul(lam_v,total),V::mul(weight_v,source)),norm_v);
        const auto change = V::sub(unew,previous);
        V::store(squared+diagonal*lanes,V::mul(change,change));

        const auto tx = V::add(ux,bxx), ty = V::add(uy,byy);
        if constexpr(Isotropic) {
            const auto length = V::root(V::add(V::mul(tx,tx),V::mul(ty,ty)));
            const auto factor = V::mul(length,lam_v);
            west_dx = V::quotient(V::mul(factor,tx),factor);
            north_dy = V::quotient(V::mul(factor,ty),factor);
        }else {
            west_dx = V::threshold(tx,threshold);
            north_dy = V::threshold(ty,threshold);
        }
        west_bx = V::add(bxx,V::sub(ux,west_dx));
        north_by = V::add(byy,V::sub(uy,north_dy));
        west = unew;
        previous = east;
        V::template scatter<Height>(out+diagonal,step,west);
        V::template scatter<Height>(dx+diagonal,step,west_dx);
        V::template scatter<Height>(dy+diagonal,step,north_dy);
        V::template scatter<Height>(bx+diagonal,step,west_bx);
        V::template scatter<Height>(by+diagonal,step,north_by);
    }
}

template<int Height,class T>
void wavefront_dispatch(const T* input,T* out,T* dx,T* dy,T* bx,T* by,T* squared,
                        std::size_t cols,T weight,T lam,T norm,bool isotropic) {
    if(isotropic) wavefront<T,Height,true>(input,out,dx,dy,bx,by,squared,cols,weight,lam,norm);
    else wavefront<T,Height,false>(input,out,dx,dy,bx,by,squared,cols,weight,lam,norm);
}
} // namespace

void tv_bregman_wavefront(const float* input,float* out,float* dx,float* dy,float* bx,float* by,
                          float* squared,std::size_t cols,std::size_t height,float weight,float lam,float norm,bool isotropic) {
    if(height==8) wavefront_dispatch<8>(input,out,dx,dy,bx,by,squared,cols,weight,lam,norm,isotropic);
    else wavefront_dispatch<4>(input,out,dx,dy,bx,by,squared,cols,weight,lam,norm,isotropic);
}
void tv_bregman_wavefront(const double* input,double* out,double* dx,double* dy,double* bx,double* by,
                          double* squared,std::size_t cols,std::size_t,double weight,double lam,double norm,bool isotropic) {
    wavefront_dispatch<4>(input,out,dx,dy,bx,by,squared,cols,weight,lam,norm,isotropic);
}

} // namespace skimagexx::simd
