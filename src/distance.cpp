#include "skimagexx/distance.hpp"
#include "skimagexx/image.hpp"
#include "distance_detail.hpp"
#include "distance_metrics.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <limits>

namespace skimagexx {
namespace {
void validate_distance(std::size_t n,PixelType type,std::ptrdiff_t pixels,std::ptrdiff_t channels) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/std::max(pixel_size(type),sizeof(double));
    const auto magnitude=[](std::ptrdiff_t v){return v<0 ? std::size_t(-(v+1))+1 : std::size_t(v);};
    const auto p=magnitude(pixels),c=magnitude(channels);
    if(n>limit || c>limit/2 || (n>1 && p>(limit-2*c)/(n-1)))
        throw std::overflow_error("distance strides overflow");
}
template<class T> void execute_distance(const void* a,const void* b,T* out,std::size_t n,PixelType at,PixelType bt,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    const auto ar=detail::distance_reader<T>(at),br=detail::distance_reader<T>(bt);
    const auto ab=std::ptrdiff_t(pixel_size(at)),bb=std::ptrdiff_t(pixel_size(bt));
    const bool vectorized=has_avx2();
    detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto) {
        const auto* first=static_cast<const unsigned char*>(a)+std::ptrdiff_t(begin)*ap*ab;
        const auto* second=static_cast<const unsigned char*>(b)+std::ptrdiff_t(begin)*bp*bb;
        std::size_t consumed=0;
#ifdef SKIMAGEXX_AVX2
        if(vectorized) consumed=simd::delta_e76(first,second,out+begin,end-begin,at,bt,ap,ac,bp,bc);
#else
        (void)vectorized;
#endif
        for(auto j=consumed;j<end-begin;++j)
            out[begin+j]=detail::distance_pixel(first+std::ptrdiff_t(j)*ap*ab,second+std::ptrdiff_t(j)*bp*bb,ar,br,ac,bc);
    });
}
}
namespace {
template<class T,class Output> void execute_metric(const void* a,const void* b,Output* out,std::size_t n,
    PixelType at,PixelType bt,DistanceMetric metric,const DistanceParameters& parameters,bool wide_trig,bool scalar_cmc,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    const auto ar=detail::distance_reader<T>(at),br=detail::distance_reader<T>(bt);
    const auto ab=std::ptrdiff_t(pixel_size(at)),bb=std::ptrdiff_t(pixel_size(bt));
    const bool vectorized=has_avx2();
    detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto) {
        const auto* first=static_cast<const unsigned char*>(a)+std::ptrdiff_t(begin)*ap*ab;
        const auto* second=static_cast<const unsigned char*>(b)+std::ptrdiff_t(begin)*bp*bb;
        std::size_t consumed=0;
#ifdef SKIMAGEXX_AVX2
        if(vectorized) consumed=simd::metric_distance(first,second,out+begin,end-begin,at,bt,metric,parameters,
            wide_trig,scalar_cmc,ap,ac,bp,bc);
#else
        (void)vectorized;
#endif
        for(auto j=consumed;j<end-begin;++j) {
            const auto* pa=first+std::ptrdiff_t(j)*ap*ab;
            const auto* pb=second+std::ptrdiff_t(j)*bp*bb;
            double av[]={ar(pa,0),ar(pa,ac),ar(pa,2*ac)},bv[]={br(pb,0),br(pb,bc),br(pb,2*bc)};
            out[begin+j]=Output(detail::distance_metric<detail::ScalarDistanceMath<T>>(av,bv,metric,parameters,wide_trig,scalar_cmc));
        }
    });
}
}
void delta_e_into(const void* a,const void* b,void* out,std::size_t n,PixelType at,PixelType bt,DistanceMetric metric,
    DistanceParameters p,bool wide_trig,bool scalar_cmc,std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    if(int(metric)<0 || int(metric)>2) throw std::invalid_argument("unknown color distance metric");
    for(auto v:{p.kL,p.kC,p.kH,p.k1,p.k2}) if(!std::isfinite(v))
        throw std::invalid_argument("distance parameters must be finite");
    const bool single=at==PixelType::float32 && bt==PixelType::float32;
    if(scalar_cmc && (!single || metric!=DistanceMetric::cmc))
        throw std::invalid_argument("CMC scalar promotion requires float32 CMC inputs");
    validate_distance(n,at,ap,ac); validate_distance(n,bt,bp,bc);
    (void)detail::distance_reader<double>(at); (void)detail::distance_reader<double>(bt);
    if(!n) return;
    if(!a || !b || !out) throw std::invalid_argument("distance buffers must be nonnull");
    if(single && !scalar_cmc) execute_metric<float>(a,b,static_cast<float*>(out),n,at,bt,metric,p,wide_trig,scalar_cmc,ap,ac,bp,bc);
    else if(single) execute_metric<float>(a,b,static_cast<double*>(out),n,at,bt,metric,p,wide_trig,scalar_cmc,ap,ac,bp,bc);
    else execute_metric<double>(a,b,static_cast<double*>(out),n,at,bt,metric,p,wide_trig,scalar_cmc,ap,ac,bp,bc);
}
void delta_e76_into(const void* a,const void* b,void* out,std::size_t n,PixelType at,PixelType bt,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    validate_distance(n,at,ap,ac); validate_distance(n,bt,bp,bc);
    // Validate even empty arrays, without dereferencing their buffers.
    (void)detail::distance_reader<double>(at); (void)detail::distance_reader<double>(bt);
    if(!n) return;
    if(!a || !b || !out) throw std::invalid_argument("distance buffers must be nonnull");
    if(at==PixelType::float32 && bt==PixelType::float32)
        execute_distance(a,b,static_cast<float*>(out),n,at,bt,ap,ac,bp,bc);
    else execute_distance(a,b,static_cast<double*>(out),n,at,bt,ap,ac,bp,bc);
}
namespace {
template<class T> void execute_distance_rows(const void* a,const void* b,T* out,std::size_t count,std::size_t columns,
    PixelType at,PixelType bt,const std::ptrdiff_t* rows_a,const std::ptrdiff_t* rows_b,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    const auto ar=detail::distance_reader<T>(at),br=detail::distance_reader<T>(bt);
    const auto ab=std::ptrdiff_t(pixel_size(at)),bb=std::ptrdiff_t(pixel_size(bt));
    const bool vectorized=has_avx2();
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
        auto position=begin;
        while(position<end) {
            const auto row=position/columns,col=position%columns;
            const auto length=std::min(end-position,columns-col);
            const auto* first=static_cast<const unsigned char*>(a)+(rows_a[row]+std::ptrdiff_t(col)*ap)*ab;
            const auto* second=static_cast<const unsigned char*>(b)+(rows_b[row]+std::ptrdiff_t(col)*bp)*bb;
            std::size_t consumed=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized) consumed=simd::delta_e76(first,second,out+position,length,at,bt,ap,ac,bp,bc);
#else
            (void)vectorized;
#endif
            for(auto j=consumed;j<length;++j)
                out[position+j]=detail::distance_pixel(first+std::ptrdiff_t(j)*ap*ab,second+std::ptrdiff_t(j)*bp*bb,ar,br,ac,bc);
            position+=length;
        }
    });
}
void validate_rows(std::size_t rows,std::size_t columns,PixelType type,const std::ptrdiff_t* offsets,
    std::ptrdiff_t pixels,std::ptrdiff_t channels) {
    validate_distance(columns,type,pixels,channels);
    const auto magnitude=[](std::ptrdiff_t v){return v<0 ? std::size_t(-(v+1))+1 : std::size_t(v);};
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/std::max(pixel_size(type),sizeof(double));
    const auto span=(columns-1)*magnitude(pixels)+2*magnitude(channels);
    for(std::size_t row=0;row<rows;++row) if(magnitude(offsets[row])>limit-span)
        throw std::overflow_error("distance row offsets overflow");
}
}
void delta_e76_rows_into(const void* a,const void* b,void* out,std::size_t rows,std::size_t columns,
    PixelType at,PixelType bt,const std::ptrdiff_t* rows_a,const std::ptrdiff_t* rows_b,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    (void)detail::distance_reader<double>(at);(void)detail::distance_reader<double>(bt);
    if(!rows || !columns) return;
    if(!a || !b || !out || !rows_a || !rows_b) throw std::invalid_argument("distance row buffers must be nonnull");
    const bool single=at==PixelType::float32 && bt==PixelType::float32;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(single?sizeof(float):sizeof(double));
    if(rows>limit/columns) throw std::overflow_error("distance output size overflows");
    validate_rows(rows,columns,at,rows_a,ap,ac);validate_rows(rows,columns,bt,rows_b,bp,bc);
    if(single) execute_distance_rows(a,b,static_cast<float*>(out),rows*columns,columns,at,bt,rows_a,rows_b,ap,ac,bp,bc);
    else execute_distance_rows(a,b,static_cast<double*>(out),rows*columns,columns,at,bt,rows_a,rows_b,ap,ac,bp,bc);
}
}
