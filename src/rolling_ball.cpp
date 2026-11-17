#include <skimagexx/rolling_ball.hpp>
#include <skimagexx/rolling_ball_kernels.hpp>
#include <skimagexx/image.hpp>
#include "execution.hpp"
#include "half.hpp"
#include <cfenv>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#pragma STDC FENV_ACCESS ON
namespace skimagexx { namespace {
constexpr std::size_t rank_limit=32;
std::size_t product(const std::vector<std::ptrdiff_t>& shape,bool positive=false) {
    std::size_t n=1;
    for(auto size:shape) {
        if(size<0||(positive&&size==0))throw std::invalid_argument("rolling_ball requires positive kernel/nonnegative image extents");
        if(size&&n>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/std::size_t(size))
            throw std::overflow_error("rolling_ball shape product overflow");
        n*=std::size_t(size);
    }
    return n;
}
std::size_t offset(std::size_t i,const std::vector<std::ptrdiff_t>& shape,
                   const std::array<std::size_t,rank_limit>& stride) {
    std::size_t result=0;
    for(auto a=shape.size();a-->0;){result+=(i%std::size_t(shape[a]))*stride[a];i/=std::size_t(shape[a]);}
    return result;
}
int workers_snapshot(int requested) {
    const int n=requested==0?get_num_threads():requested;
    if(n<1||n>1024)throw std::invalid_argument("rolling_ball workers must be in [1,1024] or0");
    return n;
}
}
namespace detail {
template<class T> std::vector<T> rolling_ball_from_difference(
    const T* input,const RollingBallGeometry& g,const T* difference,bool safe,int requested) {
    static_assert(std::is_same_v<T,float>||std::is_same_v<T,double>);
    const auto ndim=g.shape.size();
    if(ndim==0||ndim>rank_limit||g.strides.size()!=ndim||g.kernel_shape.size()!=ndim)
        throw std::invalid_argument("rolling_ball requires same-rank image/kernel");
    const auto n=product(g.shape),k=product(g.kernel_shape,true);
    const int workers=workers_snapshot(requested);
    if((n&&!input)||!difference)throw std::invalid_argument("rolling_ball null storage");
    std::vector<T> out(n);if(!n)return out;
    std::vector<std::ptrdiff_t> padded_shape(ndim);
    for(std::size_t a=0;a<ndim;++a) {
        const auto padding=g.kernel_shape[a]/2;
        if(padding>(std::numeric_limits<std::ptrdiff_t>::max()-g.shape[a])/2)
            throw std::overflow_error("rolling_ball padded extent overflow");
        padded_shape[a]=g.shape[a]+2*padding;
    }
    const auto padded_count=product(padded_shape,true);
    std::vector<T> padded(padded_count,std::numeric_limits<T>::infinity());
    std::array<std::size_t,rank_limit> strides{};
    std::size_t step=1;
    for(auto a=ndim;a-->0;){strides[a]=step;step*=std::size_t(padded_shape[a]);}
    std::size_t origin=0;
    for(std::size_t a=0;a<ndim;++a)origin+=std::size_t(g.kernel_shape[a]/2)*strides[a];
    // One rolling coordinate vector; no allocation/division per copied pixel.
    std::array<std::ptrdiff_t,rank_limit> coordinate{};
    std::ptrdiff_t source=0;std::size_t target=origin;
    for(std::size_t i=0;i<n;++i) {
        T value;std::memcpy(&value,reinterpret_cast<const char*>(input)+source,sizeof(T));
        padded[target]=value;
        for(auto a=ndim;a-->0;) {
            if(++coordinate[a]<g.shape[a]){source+=g.strides[a];target+=strides[a];break;}
            source-=coordinate[a]*g.strides[a]-g.strides[a];
            target-=std::size_t(coordinate[a]-1)*strides[a];coordinate[a]=0;
        }
    }
    const auto last=std::size_t(g.kernel_shape.back()),rows=k/last;
    std::vector<std::size_t> row_offset(rows);
    for(std::size_t r=0;r<rows;++r)row_offset[r]=offset(r*last,g.kernel_shape,strides);
    // Source order within each output is never reassociated. Independent
    // output pixels may be partitioned; small images remain default serial.
    detail::partition(n,n>=detail::parallel_min?workers:1,[&](auto begin,auto end,auto){
        for(auto i=begin;i<end;++i) {
            const auto base=offset(i,g.shape,strides);T minimum=std::numeric_limits<T>::infinity();
            for(std::size_t r=0;r<rows;++r) {
                const auto start=base+row_offset[r],kernel=r*last;
                for(std::size_t j=0;j<last;++j) {
                    const T value=padded[start+j]+difference[kernel+j];
                    if(minimum>value||(safe&&std::isnan(value)))minimum=value;
                }
                // NaN-safe source finishes the whole contiguous kernel row.
                if(safe&&std::isnan(minimum))break;
            }
            out[i]=minimum;
        }
    });
    return out;
}
template std::vector<float> rolling_ball_from_difference(const float*,const RollingBallGeometry&,const float*,bool,int);
template std::vector<double> rolling_ball_from_difference(const double*,const RollingBallGeometry&,const double*,bool,int);
}
template<class T> std::vector<T> rolling_ball(const T* input,const RollingBallGeometry& supplied,
    const T* custom,double radius,bool safe,int requested) {
    RollingBallGeometry g=supplied;const auto ndim=g.shape.size();
    if(ndim==0||ndim>rank_limit)throw std::invalid_argument("rolling_ball rank outside[1,32]");
    if(g.strides.size()!=ndim)throw std::invalid_argument("rolling_ball stride rank mismatch");
    const auto input_count=product(g.shape);
    if(input_count&&!input)throw std::invalid_argument("rolling_ball null input storage");
    const int workers=workers_snapshot(requested);std::vector<T> generated;
    if(!custom) {
        if(!std::isfinite(radius))throw std::invalid_argument("rolling_ball finite radius required");
        const double ceiling=std::ceil(radius);
        if(ceiling<0||ceiling>double(std::numeric_limits<std::ptrdiff_t>::max()/2-1))
            throw std::invalid_argument("rolling_ball nonempty finite ball required");
        const auto side=std::ptrdiff_t(2*ceiling+1);g.kernel_shape.assign(ndim,side);
        const auto count=product(g.kernel_shape,true);
        std::vector<double> axis(std::size_t(side),0.),kernel(count);
        for(std::ptrdiff_t i=0;i<side;++i)axis[std::size_t(i)]=double(i)-ceiling;
        std::vector<detail::RollingAxis> axes(ndim,{axis.data(),axis.size(),1.});
        if(!detail::ball_kernel_into({axes.data(),ndim,count},radius,radius*radius,kernel.data(),workers))
            throw std::domain_error("rolling_ball ball primitive refused arithmetic environment");
        generated.assign(kernel.begin(),kernel.end());custom=generated.data();
    }
    if(g.kernel_shape.size()!=ndim)throw std::invalid_argument("rolling_ball kernel rank mismatch");
    const auto k=product(g.kernel_shape,true);std::size_t center=0;
    for(auto size:g.kernel_shape)center=center*std::size_t(size)+std::size_t(size/2);
    std::vector<T> difference(k);const T height=custom[center];
    for(std::size_t i=0;i<k;++i) {
        const T v=height-custom[i];
        difference[i]=custom[i]==std::numeric_limits<T>::infinity()?std::numeric_limits<T>::infinity():v;
    }
    return detail::rolling_ball_from_difference(input,g,difference.data(),safe,workers);
}
template std::vector<float> rolling_ball(const float*,const RollingBallGeometry&,const float*,double,bool,int);
template std::vector<double> rolling_ball(const double*,const RollingBallGeometry&,const double*,double,bool,int);
namespace {
std::size_t storage_width(RollingBallStorage view) {
    if(view.half&&view.type!=PixelType::uint16)throw std::invalid_argument("rolling_ball half requires uint16 storage");
    switch(view.type) {
    case PixelType::boolean:case PixelType::uint8:case PixelType::int8:return 1;
    case PixelType::uint16:case PixelType::int16:return 2;
    case PixelType::uint32:case PixelType::int32:case PixelType::float32:return 4;
    case PixelType::uint64:case PixelType::int64:case PixelType::float64:return 8;
    }
    throw std::invalid_argument("rolling_ball invalid storage dtype");
}
template<class T>T storage_read(const unsigned char* p,bool swap) {
    T value;
    if(swap&&sizeof(T)>1) {
        std::array<unsigned char,sizeof(T)> bytes;
        for(std::size_t i=0;i<sizeof(T);++i)bytes[i]=p[sizeof(T)-1-i];
        std::memcpy(&value,bytes.data(),sizeof(T));
    }else std::memcpy(&value,p,sizeof(T));
    return value;
}
template<class T>void storage_write(unsigned char* p,T value,bool swap) {
    std::array<unsigned char,sizeof(T)> bytes;std::memcpy(bytes.data(),&value,sizeof(T));
    for(std::size_t i=0;i<sizeof(T);++i)p[i]=bytes[swap?sizeof(T)-1-i:i];
}
template<class Work>Work read_number(const unsigned char* p,RollingBallStorage v) {
    if(v.half)return Work(detail::decode_half(storage_read<std::uint16_t>(p,v.swap)));
    switch(v.type) {
    case PixelType::boolean:return Work(storage_read<std::uint8_t>(p,false)!=0);
    case PixelType::uint8:return Work(storage_read<std::uint8_t>(p,false));
    case PixelType::int8:return Work(storage_read<std::int8_t>(p,false));
    case PixelType::uint16:return Work(storage_read<std::uint16_t>(p,v.swap));
    case PixelType::int16:return Work(storage_read<std::int16_t>(p,v.swap));
    case PixelType::uint32:return Work(storage_read<std::uint32_t>(p,v.swap));
    case PixelType::int32:return Work(storage_read<std::int32_t>(p,v.swap));
    case PixelType::uint64:return Work(storage_read<std::uint64_t>(p,v.swap));
    case PixelType::int64:return Work(storage_read<std::int64_t>(p,v.swap));
    case PixelType::float32:return Work(storage_read<float>(p,v.swap));
    case PixelType::float64:return Work(storage_read<double>(p,v.swap));
    }
    throw std::invalid_argument("rolling_ball invalid storage dtype");
}
void storage_span(const RollingBallGeometry& g,std::size_t width) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    std::size_t negative=0,positive=width-1;
    for(std::size_t a=0;a<g.shape.size();++a) {
        if(g.shape[a]<=1)continue;
        const auto stride=g.strides[a];
        if(stride==std::numeric_limits<std::ptrdiff_t>::min())throw std::overflow_error("rolling_ball stride span overflow");
        const auto magnitude=std::size_t(stride<0?-stride:stride),steps=std::size_t(g.shape[a]-1);
        auto& span=stride<0?negative:positive;
        if(magnitude>(limit-span)/steps)throw std::overflow_error("rolling_ball stride span overflow");
        span+=magnitude*steps;
    }
}
template<class Work>std::vector<Work> storage_image(RollingBallStorage v,const RollingBallGeometry& g,std::size_t n) {
    const auto width=storage_width(v);storage_span(g,width);
    if(n&&!v.data)throw std::invalid_argument("rolling_ball null image storage");
    std::vector<Work> values(n);std::array<std::ptrdiff_t,rank_limit> coordinate{};std::ptrdiff_t source=0;
    for(std::size_t i=0;i<n;++i) {
        values[i]=read_number<Work>(static_cast<const unsigned char*>(v.data)+source,v);
        if(i+1==n)break;
        for(auto a=g.shape.size();a-->0;) {
            if(++coordinate[a]<g.shape[a]){source+=g.strides[a];break;}
            source-=(coordinate[a]-1)*g.strides[a];coordinate[a]=0;
        }
    }
    return values;
}
template<class Integer,class Work>void integer_result(unsigned char* p,Work value,bool swap) {
    // Avoid undefined C++ float->integer casts. Do not invent NumPy's separate
    // architecture-dependent invalid/overflow word or warning policy.
    const double number=double(value),truncated=std::trunc(number);
    constexpr int digits=std::numeric_limits<Integer>::digits;
    const double upper=std::ldexp(1.,digits),lower=std::is_signed_v<Integer>?-upper:0.;
    if(!std::isfinite(number)||truncated<lower||truncated>=upper)
        throw std::domain_error("rolling_ball integer result outside finite representable cast domain");
    storage_write(p,static_cast<Integer>(truncated),swap);
}
template<class Work>RollingBallResult storage_operation(RollingBallStorage v,const RollingBallGeometry& supplied,
    RollingBallStorage custom,double radius,bool safe,int workers) {
    RollingBallGeometry g=supplied;const auto n=product(g.shape),width=storage_width(v);
    auto values=storage_image<Work>(v,g,n);std::vector<Work> kernel;
    if(custom.data) {
        if(g.kernel_shape.size()!=g.shape.size())throw std::invalid_argument("rolling_ball kernel rank mismatch");
        const auto k=product(g.kernel_shape,true),size=storage_width(custom);
        if(k>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/size)throw std::overflow_error("rolling_ball kernel bytes overflow");
        kernel.resize(k);for(std::size_t i=0;i<k;++i)kernel[i]=read_number<Work>(static_cast<const unsigned char*>(custom.data)+i*size,custom);
    }
    std::size_t step=sizeof(Work);
    for(auto a=g.shape.size();a-->0;) {
        if(step>std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))throw std::overflow_error("rolling_ball converted stride overflow");
        g.strides[a]=std::ptrdiff_t(step);
        if(g.shape[a]&&step>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/std::size_t(g.shape[a]))throw std::overflow_error("rolling_ball converted bytes overflow");
        step*=std::size_t(g.shape[a]);
    }
    auto background=rolling_ball(values.data(),g,kernel.empty()?nullptr:kernel.data(),radius,safe,workers);
    if(n>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/width)throw std::overflow_error("rolling_ball result bytes overflow");
    RollingBallResult result;result.shape=g.shape;result.type=v.type;result.half=v.half;result.swap=v.swap;result.captured_workers=workers;result.bytes.resize(n*width);
    if(v.half&&std::fegetround()!=FE_TONEAREST)throw std::domain_error("rolling_ball half result requires nearest rounding");
    for(std::size_t i=0;i<n;++i) {
        auto* p=result.bytes.data()+i*width;const auto value=background[i];
        if(v.half){storage_write(p,detail::encode_half(float(value)),v.swap);continue;}
        switch(v.type) {
        case PixelType::boolean:storage_write(p,std::uint8_t(value!=Work(0)),false);break;
        case PixelType::uint8:integer_result<std::uint8_t>(p,value,false);break;
        case PixelType::int8:integer_result<std::int8_t>(p,value,false);break;
        case PixelType::uint16:integer_result<std::uint16_t>(p,value,v.swap);break;
        case PixelType::int16:integer_result<std::int16_t>(p,value,v.swap);break;
        case PixelType::uint32:integer_result<std::uint32_t>(p,value,v.swap);break;
        case PixelType::int32:integer_result<std::int32_t>(p,value,v.swap);break;
        case PixelType::uint64:integer_result<std::uint64_t>(p,value,v.swap);break;
        case PixelType::int64:integer_result<std::int64_t>(p,value,v.swap);break;
        case PixelType::float32:storage_write(p,float(value),v.swap);break;
        case PixelType::float64:storage_write(p,double(value),v.swap);break;
        }
    }
    return result;
}
} // namespace
RollingBallResult rolling_ball(RollingBallStorage input,const RollingBallGeometry& g,
    RollingBallStorage custom,double radius,bool safe,int requested) {
    if(g.shape.empty()||g.shape.size()>rank_limit||g.strides.size()!=g.shape.size())throw std::invalid_argument("rolling_ball rank/stride mismatch");
    storage_width(input);storage_width(custom);product(g.shape);
    const int workers=workers_snapshot(requested);
    if(input.half||input.type==PixelType::float32)return storage_operation<float>(input,g,custom,radius,safe,workers);
    return storage_operation<double>(input,g,custom,radius,safe,workers);
}

}
