#include "skimagexx/morphology.hpp"
#include "skimagexx/image.hpp"
#include "morphology_detail.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <memory>
#include <stdexcept>
#include <numeric>

namespace skimagexx {
namespace {
using Mode = MorphologyMode;
using Op = MorphologyOp;
struct Plan {
    std::vector<std::ptrdiff_t> offsets, low, high;
    std::size_t points = 0;
    std::int64_t iterations = 1;
    bool rectangle = false, separable = false;
};
bool map_index(std::ptrdiff_t x, std::ptrdiff_t offset, std::size_t length, Mode mode, std::size_t& result) {
    const auto n = std::ptrdiff_t(length);
    if (offset >= -x && offset < n-x) { result = std::size_t(x+offset); return true; }
    if (mode == Mode::constant) return false;
    if (mode == Mode::nearest) { result = offset < -x ? 0 : length-1; return true; }
    if (mode == Mode::mirror && n == 1) { result = 0; return true; }
    const auto period = mode == Mode::wrap ? n : mode == Mode::mirror ? 2*(n-1) : 2*n;
    auto y = (x+offset%period)%period;
    if (y < 0) y += period;
    if (mode == Mode::reflect && y >= n) y = 2*n-1-y;
    if (mode == Mode::mirror && y >= n) y = period-y;
    result = std::size_t(y); return true;
}
std::vector<std::size_t> strides_for(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size(),1);
    for (std::size_t j=shape.size()-1;j>0;--j) strides[j-1]=strides[j]*shape[j];
    return strides;
}
std::vector<Plan> make_plans(const std::vector<MorphologyFootprint>& footprints, std::size_t ndim) {
    std::vector<Plan> result;
    for (const auto& fp : footprints) {
        if (fp.offsets.size()%ndim) throw std::invalid_argument("footprint rank mismatch");
        Plan p; p.offsets=fp.offsets; p.points=p.offsets.size()/ndim;
        p.iterations=fp.iterations; p.separable=fp.scipy_separable;
        p.low.assign(ndim,0); p.high.assign(ndim,0);
        if (p.points) {
            for (std::size_t d=0;d<ndim;++d) {
                p.low[d]=p.high[d]=p.offsets[d];
                for (std::size_t k=0;k<p.points;++k) {
                    const auto v=p.offsets[k*ndim+d];
                    const auto limit=std::numeric_limits<std::ptrdiff_t>::max()/8;
                    if (v < -limit || v > limit) throw std::overflow_error("footprint offset is too large");
                    p.low[d]=std::min(p.low[d],v); p.high[d]=std::max(p.high[d],v);
                }
            }
            std::size_t volume=1;
            for (std::size_t d=0;d<ndim;++d) {
                const auto extent=std::size_t(p.high[d]-p.low[d]+1);
                if (volume > p.points/extent) { volume=0; break; }
                volume*=extent;
            }
            p.rectangle=volume==p.points;
        }
        result.push_back(std::move(p));
    }
    return result;
}
Plan mirrored(Plan p, std::size_t ndim) {
    for (auto& v : p.offsets) v=-v;
    // SciPy iterates the reflected footprint in lexicographic order.
    for (std::size_t k=0;k<p.points/2;++k)
        for (std::size_t d=0;d<ndim;++d) std::swap(p.offsets[k*ndim+d],p.offsets[(p.points-1-k)*ndim+d]);
    for (std::size_t d=0;d<ndim;++d) { const auto low=p.low[d]; p.low[d]=-p.high[d]; p.high[d]=-low; }
    return p;
}
template<class T> void resolve_mode(Mode& mode, double& cval, bool erode, bool binary, bool boolean) {
    if (mode == Mode::ignore) mode=erode ? Mode::maximum : Mode::minimum;
    if (mode == Mode::maximum || mode == Mode::minimum) {
        const bool maximum=mode==Mode::maximum;
        cval=binary || boolean ? double(maximum) : std::is_floating_point_v<T> ?
            (maximum ? INFINITY : -INFINITY) : maximum ? double(std::numeric_limits<T>::max()) : double(std::numeric_limits<T>::lowest());
        mode=Mode::constant;
    }
}
template<class T> bool has_nan(const T* p, std::size_t n, PixelType type) {
    if constexpr (std::is_floating_point_v<T>) {
#ifdef SKIMAGEXX_AVX2
        if (has_avx2()) return simd::morphology_has_nan(p,n,type);
#else
        (void)type;
#endif
        for (std::size_t i=0;i<n;++i) if (std::isnan(p[i])) return true;
    } else { (void)p; (void)n; (void)type; }
    return false;
}
template<class T> void stencil(const T* source, T* output, const std::vector<std::size_t>& shape,
    const Plan& fp, bool erode, Mode mode, double cval, PixelType type, int workers, bool binary=false) {
    const auto ndim=shape.size(), width=shape.back();
    const auto strides=strides_for(shape);
    const auto n=shape[0]*strides[0], rows=n/width;
    const bool vectorized=has_avx2();
    detail::partition(rows,workers,[&](auto begin,auto end,auto) {
        std::vector<const void*> pointers(fp.points);
        std::vector<std::size_t> coordinates(ndim,0);
        for (auto row=begin;row<end;++row) {
            auto index=row*width;
            for (std::size_t d=0;d+1<ndim;++d) { coordinates[d]=index/strides[d]; index%=strides[d]; }
            bool forced=false;
            std::size_t usable=0;
            double minimum=double(std::numeric_limits<T>::lowest()), maximum=double(std::numeric_limits<T>::max());
            if constexpr (std::is_floating_point_v<T>) { minimum=-INFINITY; maximum=INFINITY; }
            const bool below=cval<minimum;
            const bool above=std::is_integral_v<T> && sizeof(T)==8 ? cval>=std::ldexp(1.0,int(sizeof(T)*8)-(std::is_signed_v<T>?1:0)) : cval>maximum;
            auto* out=output+row*width;
            const auto left=std::min(width,std::size_t(std::max(std::ptrdiff_t(0),-fp.low.back())));
            const auto right=fp.high.back() > std::ptrdiff_t(width) ? std::size_t(0) :
                std::min(width,std::size_t(std::max(std::ptrdiff_t(0),std::ptrdiff_t(width)-fp.high.back())));
            const auto border=[&](std::size_t x) {
                double value=erode ? INFINITY : -INFINITY;
                for (std::size_t k=0;k<fp.points;++k) {
                    std::size_t base=0; bool valid=true;
                    for (std::size_t d=0;d<ndim;++d) {
                        std::size_t mapped;
                        if (!map_index(std::ptrdiff_t(d+1==ndim ? x : coordinates[d]),fp.offsets[k*ndim+d],shape[d],mode,mapped)) { valid=false; break; }
                        base+=mapped*strides[d];
                    }
                    value=detail::extrema(value,valid ? double(source[base]) : cval,erode);
                }
                out[x]=detail::morphology_cast<T>(value);
            };
            if (!fp.points) { std::fill(out,out+width,T(erode)); continue; }
            if (left>=right) { for (std::size_t x=0;x<width;++x) border(x); continue; }
            // Build interior pointers once per row, retaining footprint order.
            for (std::size_t k=0;k<fp.points;++k) {
                std::size_t base=0; bool valid=true;
                for (std::size_t d=0;d+1<ndim;++d) {
                    std::size_t mapped;
                    if (!map_index(std::ptrdiff_t(coordinates[d]),fp.offsets[k*ndim+d],shape[d],mode,mapped)) { valid=false; break; }
                    base+=mapped*strides[d];
                }
                if (!valid && ((erode && below) || (!erode && above))) forced=true;
                if (!valid && ((erode && above) || (!erode && below))) continue;
                pointers[usable++]=valid ? source+base+std::ptrdiff_t(left)+fp.offsets[k*ndim+ndim-1] : nullptr;
            }
            if (forced) { std::fill(out,out+width,detail::morphology_cast<T>(cval)); continue; }
            for (std::size_t x=0;x<left;++x) border(x);
            std::size_t consumed=0;
            const auto fill=detail::morphology_cast<T>(cval);
            if (!usable) { std::fill(out+left,out+right,fill); consumed=right-left; }
#ifdef SKIMAGEXX_AVX2
            else if (vectorized) consumed=simd::morphology_line(pointers.data(),usable,out+left,right-left,type,erode,double(fill),binary);
#else
            (void)vectorized; (void)type;
#endif
            for (auto x=left+consumed;x<right;++x) {
                T value=pointers[0] ? static_cast<const T*>(pointers[0])[x-left] : fill;
                for (std::size_t k=1;k<usable;++k) {
                    if (binary && value==T(!erode)) break;
                    value=detail::extrema(value,pointers[k] ? static_cast<const T*>(pointers[k])[x-left] : fill,erode);
                }
                out[x]=detail::morphology_round(value);
            }
            for (auto x=right;x<width;++x) border(x);
        }
    });
}
// van Herk/Gil-Werman: linear work per line, independent of window length.
template<class Input,class Output> void rectangular_axis(const Input* source, Output* output,
    const std::vector<std::size_t>& shape,std::size_t axis,std::ptrdiff_t low,std::ptrdiff_t high,
    bool erode,Mode mode,double cval,int workers) {
    const auto strides=strides_for(shape);
    const auto length=shape[axis], step=strides[axis];
    const auto n=shape[0]*strides[0], lines=n/length, window=std::size_t(high-low+1), extended=length+window-1;
    detail::partition(lines,workers,[&](auto begin,auto end,auto) {
        std::unique_ptr<double[]> values(new double[extended]), prefix(new double[extended]), suffix(new double[extended]);
        for (auto line=begin;line<end;++line) {
            auto remaining=line; std::size_t base=0;
            for (std::size_t d=shape.size();d-->0;) if (d!=axis) { base+=(remaining%shape[d])*strides[d]; remaining/=shape[d]; }
            for (std::size_t j=0;j<extended;++j) {
                const auto position=std::ptrdiff_t(j)+low;
                if (position>=0 && std::size_t(position)<length) values[j]=double(source[base+std::size_t(position)*step]);
                else {
                    std::size_t mapped;
                    values[j]=map_index(std::ptrdiff_t(j),low,length,mode,mapped) ? double(source[base+mapped*step]) : cval;
                }
            }
            for (std::size_t block=0;block<extended;block+=window) {
                const auto stop=std::min(extended,block+window);
                prefix[block]=values[block];
                for (auto j=block+1;j<stop;++j) prefix[j]=detail::extrema(prefix[j-1],values[j],erode);
                suffix[stop-1]=values[stop-1];
                for (auto j=stop-1;j>block;--j) suffix[j-1]=detail::extrema(suffix[j],values[j-1],erode);
            }
            for (std::size_t j=0;j<length;++j) output[base+j*step]=detail::morphology_cast<Output>(detail::extrema(suffix[j],prefix[j+window-1],erode));
        }
    });
}
template<class T> void apply_footprint(const T* source,T* output,const std::vector<std::size_t>& shape,
    const Plan& fp,bool erode,Mode mode,double cval,PixelType type,int workers,bool binary) {
    const auto n=std::accumulate(shape.begin(),shape.end(),std::size_t(1),std::multiplies<std::size_t>());
    // SciPy's generic ND path casts the border value to the input storage type;
    // its separable min/max filter keeps the original double border value.
    if (!fp.separable) {
        if constexpr (std::is_same_v<T,std::uint64_t>) {
            if (cval<0 && cval>=-9223372036854775808.0) cval=double(std::uint64_t(std::int64_t(cval)));
            else cval=double(detail::morphology_cast<T>(cval));
        } else cval=double(detail::morphology_cast<T>(cval));
    }
    if (!fp.rectangle) { stencil(source,output,shape,fp,erode,mode,cval,type,workers,binary); return; }
    // Small odd rectangles use SIMD separable stencils; large windows use vHGW.
    if (fp.separable) {
        std::vector<std::size_t> axes;
        for (std::size_t d=0;d<shape.size();++d) if (fp.low[d] || fp.high[d]) axes.push_back(d);
        if (axes.empty()) { std::memcpy(output,source,n*sizeof(T)); return; }
        std::unique_ptr<T[]> a,b;
        const T* current=source;
        for (std::size_t j=0;j<axes.size();++j) {
            const auto axis=axes[j];
            T* destination;
            if (j+1==axes.size()) destination=output;
            else if (current==a.get()) { if (!b) b.reset(new T[n]); destination=b.get(); }
            else { if (!a) a.reset(new T[n]); destination=a.get(); }
            const auto width=fp.high[axis]-fp.low[axis]+1;
            if (width<=15) {
                Plan line; line.points=std::size_t(width); line.low.assign(shape.size(),0); line.high=line.low;
                line.low[axis]=fp.low[axis]; line.high[axis]=fp.high[axis]; line.offsets.assign(line.points*shape.size(),0);
                for (std::size_t k=0;k<line.points;++k) line.offsets[k*shape.size()+axis]=fp.low[axis]+std::ptrdiff_t(k);
                stencil(current,destination,shape,line,erode,mode,cval,type,workers,binary);
            } else rectangular_axis(current,destination,shape,axis,fp.low[axis],fp.high[axis],erode,mode,cval,workers);
            current=destination;
        }
    } else {
        std::unique_ptr<double[]> a(new double[n]),b(new double[n]);
        for (std::size_t i=0;i<n;++i) a[i]=double(source[i]);
        auto* current=a.get(); auto* destination=b.get();
        for (std::size_t d=0;d<shape.size();++d) if (fp.low[d] || fp.high[d]) {
            rectangular_axis(current,destination,shape,d,fp.low[d],fp.high[d],erode,mode,cval,workers);
            std::swap(current,destination);
        }
        for (std::size_t i=0;i<n;++i) output[i]=detail::morphology_cast<T>(current[i]);
    }
}
template<class T> MorphologyInfo run_gray(const T* input,T* output,const std::vector<std::size_t>& shape,
    const std::vector<Plan>& plans,Op op,Mode mode,double cval,PixelType type,bool binary=false) {
    MorphologyInfo info;
    const auto n=std::accumulate(shape.begin(),shape.end(),std::size_t(1),std::multiplies<std::size_t>());
    if (!n) return info;
    if (has_nan(input,n,type) || (mode==Mode::constant && std::isnan(cval))) { info.unsupported_nan=true; return info; }
    const auto workers=detail::active_workers(n);
    const bool boolean=type==PixelType::boolean;
    const auto source_address=reinterpret_cast<std::uintptr_t>(input),target_address=reinterpret_cast<std::uintptr_t>(output);
    std::unique_ptr<T[]> original;
    if ((source_address<=target_address ? target_address-source_address : source_address-target_address)<n*sizeof(T)) {
        original.reset(new T[n]); std::memcpy(original.get(),input,n*sizeof(T)); input=original.get();
    }
    const bool top=op==Op::white_tophat || op==Op::black_tophat;
    const bool two=op==Op::opening || op==Op::closing || top || op==Op::binary_opening || op==Op::binary_closing;
    const bool first_erode=op==Op::erosion || op==Op::opening || op==Op::white_tophat || op==Op::binary_erosion || op==Op::binary_opening;
    std::unique_ptr<T[]> a,b;
    const T* current=input;
    for (int phase=0;phase<(two?2:1);++phase) {
        const bool erode=phase==0 ? first_erode : !first_erode;
        const bool mirror=binary ? !erode : phase==1;
        Mode actual=mode; double fill=cval; resolve_mode<T>(actual,fill,erode,binary,boolean);
        for (std::size_t f=0;f<plans.size();++f) {
            const auto fp=mirror ? mirrored(plans[f],shape.size()) : plans[f];
            const bool stable=binary && fp.iterations<1;
            const auto repeats=binary ? fp.iterations : f==0 ? std::max(std::int64_t(1),fp.iterations) : std::max(std::int64_t(0),fp.iterations);
            for (std::int64_t iteration=0;stable || iteration<repeats;++iteration) {
                const bool last=!stable && !top && phase==(two?1:0) && f+1==plans.size() && iteration+1==repeats;
                T* destination;
                if (last) destination=output;
                else if (current==a.get()) { if (!b) b.reset(new T[n]); destination=b.get(); }
                else { if (!a) a.reset(new T[n]); destination=a.get(); }
                apply_footprint(current,destination,shape,fp,erode,actual,fill,type,workers,binary);
                const bool unchanged=binary && (stable || repeats>1) && std::memcmp(current,destination,n*sizeof(T))==0;
                current=destination;
                if (unchanged) break;
            }
        }
    }
    if (top) {
        // Per-worker exception flags are merged after independent pixel ranges.
        std::vector<unsigned char> invalids(std::size_t(workers),0),overflows(std::size_t(workers),0);
        detail::partition(n,workers,[&](auto begin,auto end,auto id) {
            bool invalid=false,overflow=false;
            auto first=begin;
#ifdef SKIMAGEXX_AVX2
            if (has_avx2()) first+=simd::morphology_difference(op==Op::white_tophat ? input+begin : current+begin,
                op==Op::white_tophat ? current+begin : input+begin,output+begin,end-begin,type,invalid,overflow);
#endif
            for (auto i=first;i<end;++i) {
                const T left=op==Op::white_tophat ? input[i] : current[i], right=op==Op::white_tophat ? current[i] : input[i];
                if (boolean) output[i]=T((left!=0)!=(right!=0));
                else if constexpr (std::is_integral_v<T>) {
                    using U=std::make_unsigned_t<T>; const U bits=U(U(left)-U(right)); std::memcpy(output+i,&bits,sizeof(T));
                } else {
                    output[i]=T(left-right);
                    invalid=invalid || (std::isnan(output[i]) && !std::isnan(left) && !std::isnan(right));
                    overflow=overflow || (std::isinf(output[i]) && std::isfinite(left) && std::isfinite(right));
                }
            }
            invalids[id]=static_cast<unsigned char>(invalid); overflows[id]=static_cast<unsigned char>(overflow);
        });
        info.invalid_subtract=std::any_of(invalids.begin(),invalids.end(),[](auto v){return v!=0;});
        info.overflow_subtract=std::any_of(overflows.begin(),overflows.end(),[](auto v){return v!=0;});
    } else if (current!=output) std::memcpy(output,current,n*sizeof(T));
    return info;
}
template<class T> void truth_into(const void* input,std::uint8_t* output,std::size_t n) {
    const auto* p=static_cast<const T*>(input);
    for (std::size_t i=0;i<n;++i) output[i]=std::uint8_t(p[i]!=0);
}
template<class T> void binary_output(const std::uint8_t* input,void* output,std::size_t n) {
    auto* out=static_cast<T*>(output); for (std::size_t i=0;i<n;++i) out[i]=T(input[i]);
}
}
MorphologyInfo morphology_into(const void* input,void* output,PixelType in_type,PixelType out_type,
    const std::vector<std::size_t>& shape,const std::vector<MorphologyFootprint>& footprints,Op op,Mode mode,double cval) {
    if (shape.empty() || footprints.empty()) throw std::invalid_argument("shape and footprint sequence must be nonempty");
    (void)pixel_size(in_type); (void)pixel_size(out_type);
    if (int(op)<0 || int(op)>9 || int(mode)<0 || int(mode)>7) throw std::invalid_argument("invalid morphology operation/mode");
    const bool binary=int(op)>=6;
    if (!binary && in_type!=out_type) throw std::invalid_argument("grayscale output dtype must match input");
    if (binary && mode!=Mode::maximum && mode!=Mode::minimum && mode!=Mode::ignore) throw std::invalid_argument("unsupported binary morphology mode");
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/8;
    std::size_t n=1;
    for (auto size:shape) { if (size>limit || (size && n>limit/size)) throw std::overflow_error("image shape overflows"); n*=size; }
    const auto plans=make_plans(footprints,shape.size());
    for (const auto& p:plans) if (!binary && !p.points) throw std::invalid_argument("All-zero footprint is not supported.");
    if (n && (!input || !output)) throw std::invalid_argument("input and output must be nonnull");
    if (!n) return {};
    if (binary) {
        std::unique_ptr<std::uint8_t[]> normalized(new std::uint8_t[n]),result(new std::uint8_t[n]);
#define MORPH_TRUTH(KIND,TYPE) case PixelType::KIND: truth_into<TYPE>(input,normalized.get(),n); break
        switch(in_type) {
            MORPH_TRUTH(boolean,std::uint8_t); MORPH_TRUTH(uint8,std::uint8_t); MORPH_TRUTH(int8,std::int8_t);
            MORPH_TRUTH(uint16,std::uint16_t); MORPH_TRUTH(int16,std::int16_t); MORPH_TRUTH(uint32,std::uint32_t);
            MORPH_TRUTH(int32,std::int32_t); MORPH_TRUTH(uint64,std::uint64_t); MORPH_TRUTH(int64,std::int64_t);
            MORPH_TRUTH(float32,float); MORPH_TRUTH(float64,double);
        }
#undef MORPH_TRUTH
        auto info=run_gray(normalized.get(),result.get(),shape,plans,op,mode,cval,PixelType::boolean,true);
#define MORPH_OUTPUT(KIND,TYPE) case PixelType::KIND: binary_output<TYPE>(result.get(),output,n); break
        switch(out_type) {
            MORPH_OUTPUT(boolean,std::uint8_t); MORPH_OUTPUT(uint8,std::uint8_t); MORPH_OUTPUT(int8,std::int8_t);
            MORPH_OUTPUT(uint16,std::uint16_t); MORPH_OUTPUT(int16,std::int16_t); MORPH_OUTPUT(uint32,std::uint32_t);
            MORPH_OUTPUT(int32,std::int32_t); MORPH_OUTPUT(uint64,std::uint64_t); MORPH_OUTPUT(int64,std::int64_t);
            MORPH_OUTPUT(float32,float); MORPH_OUTPUT(float64,double);
        }
#undef MORPH_OUTPUT
        return info;
    }
#define MORPH_GRAY(KIND,TYPE) case PixelType::KIND: return run_gray(static_cast<const TYPE*>(input),static_cast<TYPE*>(output),shape,plans,op,mode,cval,in_type)
    switch(in_type) {
        MORPH_GRAY(boolean,std::uint8_t); MORPH_GRAY(uint8,std::uint8_t); MORPH_GRAY(int8,std::int8_t);
        MORPH_GRAY(uint16,std::uint16_t); MORPH_GRAY(int16,std::int16_t); MORPH_GRAY(uint32,std::uint32_t);
        MORPH_GRAY(int32,std::int32_t); MORPH_GRAY(uint64,std::uint64_t); MORPH_GRAY(int64,std::int64_t);
        MORPH_GRAY(float32,float); MORPH_GRAY(float64,double);
    }
#undef MORPH_GRAY
    throw std::invalid_argument("unsupported morphology dtype");
}
} // namespace skimagexx
