#include "skimagexx/small_edge.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include "half.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <cstring>
#include <memory>
namespace skimagexx {
namespace {
template<class T> double value(T x) {
    if constexpr(std::is_same_v<T,std::uint8_t>)return double(x)*(1.0/255.0);
    else return double(x);
}
template<class Input,class Output> void column_run(const Input* image,Output* output,std::size_t height,SmallEdge operation) {
    detail::partition(height,detail::active_workers(height),[&](auto begin,auto end,auto) {
        auto scalar=[&](std::size_t y) {
            const double current=value(image[y]),next=value(image[std::min(y+1,height-1)]);
            if(operation==SmallEdge::laplace) {
                double sum=0;sum-=value(image[y ? y-1 : 0]);sum-=current;
                sum+=4*current;sum-=current;sum-=next;
                output[y]=sum==0 ? Output(0) : Output(sum);
            } else {
                double difference=0;difference+=next;difference-=current;
                const Output gradient=difference==0 ? Output(0) : Output(difference);
                if(operation!=SmallEdge::roberts)output[y]=gradient;
                else {
                    const Output squared=Output(gradient*gradient),sum=Output(squared+squared);
                    output[y]=Output(double(Output(std::sqrt(sum)))/std::sqrt(2.0));
                }
            }
        };
        if(operation==SmallEdge::laplace && begin==0 && begin<end)scalar(begin++);
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin=simd::small_edge_column(image,output,height,begin,end,int(operation));
#endif
        for(auto y=begin;y<end;++y)scalar(y);
    });
}
template<class Input,class Output> void run(const Input* image,Output* output,std::size_t h,std::size_t w,SmallEdge operation) {
    if(!h || !w || !image || !output)throw std::invalid_argument("nonempty image and output required");
    if(h>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/w)throw std::overflow_error("image dimensions overflow");
    if(int(operation)<0 || int(operation)>3)throw std::invalid_argument("unsupported small edge operation");
    if(w==1) {column_run(image,output,h,operation);return;}
    const auto count=h*w;
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
        while(begin<end) {
            const auto y=begin/w,start=begin%w,stop=std::min(w,start+end-begin);
            const auto* mid=image+y*w;
            const auto* top=image+(y ? y-1 : 0)*w;
            const auto* bottom=image+std::min(y+1,h-1)*w;
            auto* target=output+y*w;
            auto scalar=[&](std::size_t x) {
                const auto left=x ? x-1 : 0,right=std::min(x+1,w-1);
                if(operation==SmallEdge::laplace) {
                    double sum=0;sum-=value(top[x]);sum-=value(mid[left]);sum+=4*value(mid[x]);
                    sum-=value(mid[right]);sum-=value(bottom[x]);target[x]=Output(sum);
                } else {
                    double positive=0;positive+=value(bottom[right]);positive-=value(mid[x]);
                    double negative=0;negative+=value(bottom[x]);negative-=value(mid[right]);
                    const Output a=positive==0 ? Output(0) : Output(positive),b=negative==0 ? Output(0) : Output(negative);
                    if(operation==SmallEdge::roberts_positive)target[x]=a;
                    else if(operation==SmallEdge::roberts_negative)target[x]=b;
                    else {
                        const Output sum=Output(Output(a*a)+Output(b*b));
                        target[x]=Output(double(Output(std::sqrt(sum)))/std::sqrt(2.0));
                    }
                }
            };
            auto x=start;
            if(x==0 && x<stop)scalar(x++);
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())x=simd::small_edge(top,mid,bottom,target,w,x,stop,int(operation));
#endif
            for(;x<stop;++x)scalar(x);
            begin+=stop-start;
        }
    });
}
}
namespace {
template<class Input,class Output> void laplace_run(const Input* image,Output* output,const std::vector<std::size_t>& shape) {
    if(shape.empty())throw std::invalid_argument("at least one image dimension required");
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return;
    std::size_t count=1;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(Output);
    for(auto length:shape) {if(length>limit/count)throw std::overflow_error("Laplacian dimensions overflow");count*=length;}
    if(!image || !output)throw std::invalid_argument("nonnull Laplacian storage required");
    if(shape.size()==2) {run(image,output,shape[0],shape[1],SmallEdge::laplace);return;}
    std::vector<std::size_t> strides(shape.size(),1);
    for(std::size_t axis=shape.size()-1;axis>0;--axis)strides[axis-1]=strides[axis]*shape[axis];
    auto axis=shape.size()-1;
    while(axis>0 && shape[axis]==1)--axis;
    const auto width=shape[axis],singletons=shape.size()-1-axis;
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
        std::vector<const Input*> negative(axis),positive(axis);
        while(begin<end) {
            const auto start=begin%width,base=begin-start,stop=std::min(width,start+end-begin);
            const auto* mid=image+base;auto* target=output+base;
            for(std::size_t outer=0;outer<axis;++outer) {
                const auto coordinate=(base/strides[outer])%shape[outer];
                negative[outer]=image+base-(coordinate ? strides[outer] : 0);
                positive[outer]=image+base+(coordinate+1<shape[outer] ? strides[outer] : 0);
            }
            auto scalar=[&](std::size_t x) {
                double sum=0;
                for(auto source:negative)sum-=value(source[x]);
                sum-=value(mid[x ? x-1 : 0]);
                for(std::size_t k=0;k<singletons;++k)sum-=value(mid[x]);
                sum+=double(2*shape.size())*value(mid[x]);
                for(std::size_t k=0;k<singletons;++k)sum-=value(mid[x]);
                sum-=value(mid[std::min(x+1,width-1)]);
                for(std::size_t outer=axis;outer>0;--outer)sum-=value(positive[outer-1][x]);
                target[x]=sum==0 ? Output(0) : Output(sum);
            };
            auto x=start;if(x==0 && x<stop)scalar(x++);
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())x=simd::laplace_nd(negative.data(),positive.data(),mid,target,width,x,stop,axis,singletons);
#endif
            for(;x<stop;++x)scalar(x);
            begin+=stop-start;
        }
    });
}
template<class F> void mask_run(F* output,const unsigned char* mask,std::size_t rows,std::size_t columns,
    const std::ptrdiff_t* offsets,std::ptrdiff_t stride) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(F);
    if(columns && rows>limit/columns)throw std::overflow_error("mask dimensions overflow");
    const auto count=rows*columns;if(!count)return;
    if(!output || !mask || !offsets)throw std::invalid_argument("nonnull mask storage required");
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
        while(begin<end) {
            const auto row=begin/columns,column=begin%columns,span=std::min(end-begin,columns-column);
            const auto* source=mask+offsets[row]+std::ptrdiff_t(column)*stride;
            std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())x=simd::edge_mask(output+begin,source,span,stride);
#endif
            for(;x<span;++x)output[begin+x]*=F(source[std::ptrdiff_t(x)*stride]!=0);
            begin+=span;
        }
    });
}
}
template<class T> void laplace_nd_into(const T* image,T* output,const std::vector<std::size_t>& shape) {laplace_run(image,output,shape);}
void laplace_nd_into(const std::uint8_t* image,double* output,const std::vector<std::size_t>& shape) {laplace_run(image,output,shape);}
template void laplace_nd_into(const float*,float*,const std::vector<std::size_t>&);
template void laplace_nd_into(const double*,double*,const std::vector<std::size_t>&);
void edge_half_to_float(const std::uint16_t* input,float* output,std::size_t count) {
    if(!count)return;
    if(!input || !output)throw std::invalid_argument("nonnull half storage required");
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::edge_half(input+begin,output+begin,end-begin);
#endif
        for(auto j=begin;j<end;++j) {
            output[j]=detail::decode_half(input[j]);
        }
    });
}
void edge_mask_rows_into(void* output,bool single,const unsigned char* mask,std::size_t rows,std::size_t columns,
    const std::ptrdiff_t* offsets,std::ptrdiff_t stride) {
    if(single)mask_run(static_cast<float*>(output),mask,rows,columns,offsets,stride);
    else mask_run(static_cast<double*>(output),mask,rows,columns,offsets,stride);
}
void edge_swap_into(const void* input,void* output,std::size_t count,std::size_t bytes) {
    if(bytes!=2 && bytes!=4 && bytes!=8)throw std::invalid_argument("two, four or eight byte storage required");
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes)throw std::overflow_error("byte swap size overflow");
    if(!count)return;
    if(!input || !output)throw std::invalid_argument("nonnull byte swap storage required");
    const auto* source=static_cast<const unsigned char*>(input);auto* target=static_cast<unsigned char*>(output);
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::edge_swap(source+begin*bytes,target+begin*bytes,end-begin,bytes);
#endif
        for(auto i=begin;i<end;++i)for(std::size_t j=0;j<bytes;++j)target[i*bytes+j]=source[i*bytes+bytes-1-j];
    });
}
namespace {
std::size_t numeric_size(const std::vector<std::size_t>& shape,SmallEdge operation,std::size_t bytes) {
    if(shape.empty() || (operation!=SmallEdge::laplace && shape.size()!=2))throw std::invalid_argument("invalid edge image dimensions");
    if(int(operation)<0 || int(operation)>3)throw std::invalid_argument("unsupported small edge operation");
    if(std::find(shape.begin(),shape.end(),0)!=shape.end()) {
        if(operation!=SmallEdge::laplace)throw std::invalid_argument("Roberts image must be nonempty");
        return 0;
    }
    std::size_t count=1;const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes;
    for(auto length:shape) {if(length>limit/count)throw std::overflow_error("edge dimensions overflow");count*=length;}
    return count;
}
template<class Input,class Output> void prepared_edge(const Input* input,Output* output,const std::vector<std::size_t>& shape,SmallEdge operation) {
    if(operation==SmallEdge::laplace)laplace_run(input,output,shape);
    else run(input,output,shape[0],shape[1],operation);
}
}
void small_edge_numeric_into(const void* input,void* output,const std::vector<std::size_t>& shape,PixelType type,SmallEdge operation) {
    const auto count=numeric_size(shape,operation,type==PixelType::float32 ? sizeof(float) : sizeof(double));
    if(!count)return;
    if(!input || !output)throw std::invalid_argument("nonnull edge storage required");
    if(type==PixelType::float32)prepared_edge(static_cast<const float*>(input),static_cast<float*>(output),shape,operation);
    else if(type==PixelType::float64)prepared_edge(static_cast<const double*>(input),static_cast<double*>(output),shape,operation);
    else if(type==PixelType::uint8)prepared_edge(static_cast<const std::uint8_t*>(input),static_cast<double*>(output),shape,operation);
    else {
        std::unique_ptr<double[]> prepared(new double[count]);
        convert_into(input,prepared.get(),count,type,PixelType::float64);
        prepared_edge(prepared.get(),static_cast<double*>(output),shape,operation);
    }
}
void small_edge_half_into(const std::uint16_t* input,float* output,const std::vector<std::size_t>& shape,SmallEdge operation) {
    const auto count=numeric_size(shape,operation,sizeof(float));if(!count)return;
    if(!input || !output)throw std::invalid_argument("nonnull half edge storage required");
    std::unique_ptr<float[]> prepared(new float[count]);edge_half_to_float(input,prepared.get(),count);
    prepared_edge(prepared.get(),output,shape,operation);
}
template<class T> void small_edge_into(const T* image,T* output,std::size_t h,std::size_t w,SmallEdge operation) {run(image,output,h,w,operation);}
void small_edge_into(const std::uint8_t* image,double* output,std::size_t h,std::size_t w,SmallEdge operation) {run(image,output,h,w,operation);}
template void small_edge_into(const float*,float*,std::size_t,std::size_t,SmallEdge);
template void small_edge_into(const double*,double*,std::size_t,std::size_t,SmallEdge);
}
