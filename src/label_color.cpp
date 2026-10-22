#include "skimagexx/label_color.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx {
namespace {
template<class T> T load(const unsigned char* source,bool swap) {
    T result;
    if(!swap || sizeof(T)==1)std::memcpy(&result,source,sizeof(T));
    else {
        std::array<unsigned char,sizeof(T)> bytes;
        std::reverse_copy(source,source+sizeof(T),bytes.begin());
        std::memcpy(&result,bytes.data(),sizeof(T));
    }
    return result;
}
template<class T> bool finite(T value) {
    if constexpr(std::is_floating_point_v<T>) {
        using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        U bits;std::memcpy(&bits,&value,sizeof(T));
        const U exponent=sizeof(T)==4 ? U(0x7f800000u) : U(UINT64_C(0x7ff0000000000000));
        return (bits&exponent)!=exponent;
    } else return true;
}
template<class T> std::uint64_t ordered_key(T value) {
    if constexpr(std::is_floating_point_v<T>) {
        using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        U bits;std::memcpy(&bits,&value,sizeof(T));
        const U sign=U(1)<<(sizeof(T)*8-1);
        if(value==T(0))bits=0; // NumPy unique merges positive and negative zero.
        return std::uint64_t(bits&sign ? U(~bits) : U(bits^sign));
    } else if constexpr(std::is_signed_v<T>) {
        using U=std::make_unsigned_t<T>;
        return std::uint64_t(U(value)^(U(1)<<(sizeof(T)*8-1)));
    } else return std::uint64_t(value);
}
template<class T,bool Boolean=false> bool map(const void* source,std::size_t count,bool swap,
    std::uint64_t background,bool overlay,std::uint64_t* output,LabelSummary& summary) {
    const auto* bytes=static_cast<const unsigned char*>(source);
    const int workers=detail::active_workers(count);
    std::vector<std::uint64_t> values(count);
    std::uint64_t low=std::numeric_limits<std::uint64_t>::max(),high=0,bg_key=background;
    bool can_match=true;
    for(std::size_t i=0;i<count;++i) {
        T value=load<T>(bytes+i*sizeof(T),swap);
        if constexpr(Boolean)value=T(value!=0);
        if(!finite(value))return false;
        if(overlay) {
            if constexpr(std::is_floating_point_v<T>) {
                if(background || value<T(0) || double(value)>=18446744073709551616.0)return false;
            } else if constexpr(std::is_signed_v<T>)if(value<0)return false;
            values[i]=std::uint64_t(value);
            low=std::min(low,values[i]);high=std::max(high,values[i]);
        } else values[i]=ordered_key(value);
    }
    if(!overlay) {
        if constexpr(std::is_floating_point_v<T>) {
            // Keep Python-int comparison promotion exact in both NumPy1/2.
            const auto limit=sizeof(T)==4 ? UINT64_C(16777216) : UINT64_C(9007199254740992);
            if(background>limit)return false;
            bg_key=ordered_key(T(background));
        } else {
            const auto maximum=Boolean ? std::uint64_t(1) : std::uint64_t(std::numeric_limits<T>::max());
            can_match=background<=maximum;
            if(can_match)bg_key=ordered_key(T(background));
        }
    }
    std::uint64_t mask=std::numeric_limits<std::uint64_t>::max();
    if(overlay) {
        const auto offset=std::min(low,background);
        if(offset) {
            if constexpr(Boolean || std::is_floating_point_v<T>)return false;
            else {
                if(background>std::uint64_t(std::numeric_limits<T>::max()) || background>UINT64_C(9007199254740992))return false;
                for(auto& value:values)value-=offset;
                high-=offset;background-=offset;bg_key=background;
            }
        }
        if(high<=UINT8_MAX)mask=UINT8_MAX;
        else if(high<=UINT16_MAX)mask=UINT16_MAX;
        else if(high<=UINT32_MAX)mask=UINT32_MAX;
    }
    auto unique=values;
    std::sort(unique.begin(),unique.end());
    unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
    const auto bg=std::lower_bound(unique.begin(),unique.end(),bg_key);
    const bool present=can_match && bg!=unique.end() && *bg==bg_key;
    const auto bg_rank=std::size_t(bg-unique.begin());
    if(unique.size()==std::numeric_limits<std::size_t>::max())throw std::overflow_error("too many label regions");
    summary={unique.size()+(present ? 0 : 1),background,present};
    detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto i=begin;i<end;++i) {
            const auto position=std::size_t(std::lower_bound(unique.begin(),unique.end(),values[i])-unique.begin());
            const auto rank=present && position==bg_rank ? 0 : position+(!present || position<bg_rank ? 1 : 0);
            output[i]=std::uint64_t(rank)&mask;
        }
    });
    return true;
}
}
bool label_ranks(const void* input,std::size_t count,PixelType type,bool swap,
    std::uint64_t background,bool overlay,std::uint64_t* ranks,LabelSummary& summary) {
    if(count && (!input || !ranks))throw std::invalid_argument("nonnull label storage required");
    if(!count) {summary={1,background,false};return true;}
    if(count>std::numeric_limits<std::size_t>::max()/sizeof(std::uint64_t))throw std::overflow_error("label storage too large");
    switch(type) {
    case PixelType::boolean:return map<std::uint8_t,true>(input,count,swap,background,overlay,ranks,summary);
#define LABEL_CASE(kind,T) case PixelType::kind:return map<T>(input,count,swap,background,overlay,ranks,summary)
    LABEL_CASE(uint8,std::uint8_t);LABEL_CASE(int8,std::int8_t);
    LABEL_CASE(uint16,std::uint16_t);LABEL_CASE(int16,std::int16_t);
    LABEL_CASE(uint32,std::uint32_t);LABEL_CASE(int32,std::int32_t);
    LABEL_CASE(uint64,std::uint64_t);LABEL_CASE(int64,std::int64_t);
    LABEL_CASE(float32,float);LABEL_CASE(float64,double);
#undef LABEL_CASE
    }
    throw std::invalid_argument("unsupported label storage");
}

namespace {
template<class T> bool image_valid(const void* source,std::size_t n,bool overlay) {
    const auto* p=static_cast<const unsigned char*>(source);
    for(std::size_t i=0;i<n;++i) {
        const auto v=load<T>(p+i*sizeof(T),false);
        if(!finite(v))return false;
        if constexpr(std::is_floating_point_v<T>) {
            if(overlay ? (v<T(0) || v>T(1)) : std::abs(v)>T(1e6))return false;
        }
    }
    return true;
}
template<class T> void saturation(void* source,std::size_t n,double scale) {
    auto* p=static_cast<unsigned char*>(source);const int workers=detail::active_workers(n);
    detail::partition(n,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto i=begin;i<end;++i) {
            auto* item=p+(3*i+1)*sizeof(T);
            const T value=T(load<T>(item,false)*T(scale));std::memcpy(item,&value,sizeof(T));
        }
    });
}
template<class Input,class Output> void overlay(const std::uint64_t* ranks,std::size_t n,
    const void* image,bool rgb,const double* palette,std::size_t colors,
    double alpha,double image_bg,double beta,bool remove,std::uint64_t background,void* output) {
    using F=std::conditional_t<std::is_same_v<Input,float>,float,double>;
    const auto* pixels=static_cast<const unsigned char*>(image);auto* out=static_cast<Output*>(output);
    const int workers=detail::active_workers(n);
    detail::partition(n,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto i=begin;i<end;++i) {
            const auto rank=ranks[i];
            const auto slot=rank==0 ? 0 : 1+(rank-1)%(colors-1);
            for(std::size_t channel=0;channel<3;++channel) {
                F value=0;
                if(pixels) {
                    const auto raw=load<Input>(pixels+(rgb ? 3*i+channel : i)*sizeof(Input),false);
                    if constexpr(std::is_same_v<Input,std::uint8_t>)value=double(raw)*(1.0/255.0);
                    else value=raw;
                    value=F(F(value*F(alpha))+F(image_bg));
                }
                if(remove && rank==background)out[3*i+channel]=Output(value);
                else out[3*i+channel]=Output(Output(palette[3*slot+channel])+Output(F(value*F(beta))));
            }
        }
    });
}
template<class F,class Reader> F indexed_pairwise(std::size_t begin,std::size_t n,const Reader& read) {
    if(n<8) {F value=-F(0);for(std::size_t i=0;i<n;++i)value+=read(begin+i);return value;}
    if(n<=128) {
        F r[8];for(int j=0;j<8;++j)r[j]=read(begin+std::size_t(j));
        std::size_t i=8;for(;i+8<=n;i+=8)for(int j=0;j<8;++j)r[j]+=read(begin+i+std::size_t(j));
        F value=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for(;i<n;++i)value+=read(begin+i);return value;
    }
    auto half=n/2;half-=half%8;
    return indexed_pairwise<F>(begin,half,read)+indexed_pairwise<F>(begin+half,n-half,read);
}
template<class Input> void average(const std::uint64_t* ranks,std::size_t n,std::size_t regions,
    const void* image,bool rgb,const void* background,bool has_background,void* output) {
    using Acc=std::conditional_t<std::is_same_v<Input,float>,float,double>;
    const int workers=detail::active_workers(n);
    const auto* pixels=static_cast<const unsigned char*>(image);
    const auto* bg=static_cast<const unsigned char*>(background);
    auto* out=static_cast<Input*>(output);
    std::vector<std::size_t> counts(regions,0),offsets(regions+1,0),indices(n);
    for(std::size_t i=0;i<n;++i) {
        if(ranks[i]>=regions)throw std::invalid_argument("label rank outside region table");
        ++counts[std::size_t(ranks[i])];
    }
    for(std::size_t j=0;j<regions;++j)offsets[j+1]=offsets[j]+counts[j];
    auto cursor=offsets;
    for(std::size_t i=0;i<n;++i)indices[cursor[std::size_t(ranks[i])]++]=i;
    std::vector<Input> means(3*regions,Input(0));
    detail::partition(regions,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto region=begin;region<end;++region) {
            if(region==0 && has_background) {
                for(int c=0;c<3;++c)means[c]=load<Input>(bg+std::size_t(c)*sizeof(Input),false);
                continue;
            }
            if(!counts[region])continue;
            Acc sum[3]={0,0,0};
            if(rgb) {
                for(auto j=offsets[region];j<offsets[region+1];++j)
                    for(std::size_t c=0;c<3;++c)sum[c]+=Acc(load<Input>(pixels+(3*indices[j]+c)*sizeof(Input),false));
            } else {
                auto read=[&](std::size_t j){return Acc(load<Input>(pixels+indices[j]*sizeof(Input),false));};
                sum[0]=Acc(Acc(0)+indexed_pairwise<Acc>(offsets[region],counts[region],read));
                sum[1]=sum[2]=sum[0];
            }
            for(std::size_t c=0;c<3;++c)means[3*region+c]=Input(sum[c]/Acc(counts[region]));
        }
    });
    detail::partition(n,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto i=begin;i<end;++i)for(std::size_t c=0;c<3;++c)out[3*i+c]=means[3*std::size_t(ranks[i])+c];
    });
}
}
bool label_image_valid(const void* input,std::size_t n,PixelType type,bool overlay) {
    if(n && !input)throw std::invalid_argument("nonnull underlay required");
    if(type==PixelType::uint8)return image_valid<std::uint8_t>(input,n,overlay);
    if(type==PixelType::float32)return image_valid<float>(input,n,overlay);
    if(type==PixelType::float64)return image_valid<double>(input,n,overlay);
    return false;
}
void label_saturation(void* hsv,std::size_t n,PixelType type,double scale) {
    if(n && !hsv)throw std::invalid_argument("nonnull HSV storage required");
    if(n>std::numeric_limits<std::size_t>::max()/24)throw std::overflow_error("HSV storage too large");
    if(type==PixelType::float32)saturation<float>(hsv,n,scale);
    else if(type==PixelType::float64)saturation<double>(hsv,n,scale);
    else throw std::invalid_argument("floating HSV required");
}
void label_overlay(const std::uint64_t* ranks,std::size_t n,const void* image,PixelType type,bool rgb,
    const double* palette,std::size_t colors,double alpha,double image_bg,double beta,
    bool remove,std::uint64_t background,void* output,bool single) {
    if((n && (!ranks || !output)) || !palette || colors<2)throw std::invalid_argument("label overlay storage required");
    if(n>std::numeric_limits<std::size_t>::max()/24 || colors>std::numeric_limits<std::size_t>::max()/3)throw std::overflow_error("label output too large");
#define OVERLAY_CASE(kind,T) case PixelType::kind: if(single)overlay<T,float>(ranks,n,image,rgb,palette,colors,alpha,image_bg,beta,remove,background,output);else overlay<T,double>(ranks,n,image,rgb,palette,colors,alpha,image_bg,beta,remove,background,output);return
    switch(type) {
    OVERLAY_CASE(uint8,std::uint8_t);OVERLAY_CASE(float32,float);OVERLAY_CASE(float64,double);
    default:throw std::invalid_argument("unsupported overlay underlay");
    }
#undef OVERLAY_CASE
}
void label_average(const std::uint64_t* ranks,std::size_t n,std::size_t regions,const void* image,
    PixelType type,bool rgb,const void* background,bool has_background,void* output) {
    if((n && (!ranks || !image || !output)) || !background || !regions)throw std::invalid_argument("label average storage required");
    if(n>std::numeric_limits<std::size_t>::max()/24 || regions>std::numeric_limits<std::size_t>::max()/3-1)throw std::overflow_error("label average too large");
    if(type==PixelType::uint8)average<std::uint8_t>(ranks,n,regions,image,rgb,background,has_background,output);
    else if(type==PixelType::float32)average<float>(ranks,n,regions,image,rgb,background,has_background,output);
    else if(type==PixelType::float64)average<double>(ranks,n,regions,image,rgb,background,has_background,output);
    else throw std::invalid_argument("unsupported average underlay");
}
}
