#include "skimagexx/farid.hpp"
#include "skimagexx/small_edge.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
namespace skimagexx {
namespace {
constexpr double smooth[]={.0376593171958126,.249153396177344,.426374573253687,.249153396177344,.0376593171958126};
constexpr double edge[]={.109603762960254,.276690988455557,0,-.276690988455557,-.109603762960254};
struct Kernel {std::vector<std::ptrdiff_t> coordinates,shifts;std::vector<double> weights;};
Kernel kernel(std::size_t rank,std::ptrdiff_t axis,std::size_t pivot) {
    if(axis < -std::ptrdiff_t(rank) || axis>=std::ptrdiff_t(rank))throw std::invalid_argument("Farid axis out of range");
    const auto actual=std::size_t(axis<0 ? axis+std::ptrdiff_t(rank) : axis);
    std::size_t count=1;
    for(std::size_t j=0;j<rank;++j) {
        if(count>1000000/5)throw std::length_error("Farid kernel exceeds native plan capacity");
        count*=5;
    }
    Kernel result;result.coordinates.reserve(count*rank);result.shifts.reserve(count);result.weights.reserve(count);
    std::vector<std::size_t> indices(rank);
    for(std::size_t j=0;j<count;++j) {
        auto value=j;
        for(std::size_t d=rank;d>0;--d) {indices[d-1]=value%5;value/=5;}
        double weight=edge[4-indices[actual]];
        // Upstream excludes the literal edge_dim from its smoothing set. A
        // negative edge_dim therefore also smooths its actual derivative axis.
        for(std::size_t d=0;d<rank;++d)if(std::ptrdiff_t(d)!=axis)weight*=smooth[4-indices[d]];
        if(weight==0)continue;
        result.weights.push_back(weight);result.shifts.push_back(std::ptrdiff_t(indices[pivot])-2);
        for(auto index:indices)result.coordinates.push_back(std::ptrdiff_t(index)-2);
    }
    return result;
}
std::ptrdiff_t mapped(std::ptrdiff_t index,std::size_t length,FaridBorder mode) {
    const auto n=std::ptrdiff_t(length);
    if(index>=0 && index<n)return index;
    if(mode==FaridBorder::constant)return -1;
    if(mode==FaridBorder::nearest)return std::clamp(index,std::ptrdiff_t(0),n-1);
    if(mode==FaridBorder::wrap) {index%=n;return index<0 ? index+n : index;}
    if(mode==FaridBorder::mirror && n==1)return 0;
    const auto period=mode==FaridBorder::reflect ? 2*n : 2*(n-1);
    index%=period;if(index<0)index+=period;
    if(index>=n)index=mode==FaridBorder::reflect ? 2*n-1-index : period-index;
    return index;
}
std::size_t checked_count(const std::vector<std::size_t>& shape,std::size_t bytes) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t count=1;const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes;
    for(auto length:shape) {if(length>limit/count)throw std::overflow_error("Farid dimensions overflow");count*=length;}
    return count;
}
template<class T> double normalized(T value) {
    if constexpr(std::is_same_v<T,std::uint8_t>)return double(value)*(1.0/255.0);
    else return double(value);
}
template<class Input,class Output> void run(const Input* input,Output* output,const std::vector<std::size_t>& shape,
    const std::vector<std::ptrdiff_t>& axes,FaridBorder border,double cval,bool directional) {
    if(int(border)<0 || int(border)>4)throw std::invalid_argument("unsupported Farid border mode");
    if(directional && axes.size()!=1)throw std::invalid_argument("one directional Farid axis required");
    for(auto axis:axes)if(axis < -std::ptrdiff_t(shape.size()) || axis>=std::ptrdiff_t(shape.size()))throw std::invalid_argument("Farid axis out of range");
    const auto count=checked_count(shape,sizeof(Output));if(!count)return;
    if(!output || (!input && !axes.empty()))throw std::invalid_argument("nonnull Farid storage required");
    const auto workers=detail::active_workers(count);
    detail::partition(count,workers,[&](auto begin,auto end,auto){std::fill(output+begin,output+end,Output(0));});
    if(axes.empty())return;
    std::vector<std::size_t> strides(shape.size(),1);
    for(std::size_t d=shape.size()-1;d>0;--d)strides[d-1]=strides[d]*shape[d];
    auto pivot=shape.size()-1;while(pivot>0 && shape[pivot]==1)--pivot;
    const auto width=shape[pivot];const bool square=axes.size()>1;
#ifdef SKIMAGEXX_AVX2
    const bool vectorized=has_avx2();
#endif
    for(auto axis:axes) {
        const auto plan=kernel(shape.size(),axis,pivot);const auto taps=plan.weights.size();
        detail::partition(count,workers,[&](auto begin,auto end,auto) {
            std::vector<const void*> rows(taps);
            std::vector<std::size_t> coordinates(shape.size());
            while(begin<end) {
                const auto start=begin%width,base=begin-start,stop=std::min(width,start+end-begin);
                for(std::size_t d=0;d<shape.size();++d)coordinates[d]=(base/strides[d])%shape[d];
                for(std::size_t tap=0;tap<taps;++tap) {
                    bool constant=false;std::size_t offset=0;
                    for(std::size_t d=0;d<shape.size();++d)if(d!=pivot) {
                        const auto index=mapped(std::ptrdiff_t(coordinates[d])+plan.coordinates[tap*shape.size()+d],shape[d],border);
                        if(index<0) {constant=true;break;}
                        offset+=std::size_t(index)*strides[d];
                    }
                    rows[tap]=constant ? nullptr : static_cast<const void*>(input+offset);
                }
                auto scalar=[&](std::size_t x) {
                    double sum=0;
                    for(std::size_t tap=0;tap<taps;++tap) {
                        const auto column=mapped(std::ptrdiff_t(x)+plan.shifts[tap],width,border);
                        const auto* row=static_cast<const Input*>(rows[tap]);
                        const double value=row && column>=0 ? normalized(row[column]) : cval;
                        sum+=value*plan.weights[tap];
                    }
                    Output value=Output(sum);if(square)value=Output(value*value);
                    output[base+x]=directional ? value : Output(output[base+x]+value);
                };
                auto x=start;for(;x<stop && x<2;++x)scalar(x);
#ifdef SKIMAGEXX_AVX2
                if(vectorized)x=simd::farid_row(rows.data(),plan.shifts.data(),plan.weights.data(),taps,output+base,
                    std::is_same_v<Input,float> ? PixelType::float32 : std::is_same_v<Input,double> ? PixelType::float64 : PixelType::uint8,
                    width,x,stop,cval,directional,square);
#endif
                for(;x<stop;++x)scalar(x);
                begin+=stop-start;
            }
        });
    }
    if(square) {
        const Output divisor=std::sqrt(Output(shape.size()));
        detail::partition(count,workers,[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
            if(vectorized)begin+=simd::farid_finish(output+begin,end-begin,double(divisor));
#endif
            for(auto i=begin;i<end;++i)output[i]=Output(std::sqrt(output[i])/divisor);
        });
    }
}
}
void farid_into(const void* input,void* output,PixelType type,const std::vector<std::size_t>& shape,
    const std::vector<std::ptrdiff_t>& axes,FaridBorder border,double cval,bool directional) {
    const auto count=checked_count(shape,type==PixelType::float32 ? sizeof(float) : sizeof(double));
    if(type==PixelType::float32)run(static_cast<const float*>(input),static_cast<float*>(output),shape,axes,border,cval,directional);
    else if(type==PixelType::float64)run(static_cast<const double*>(input),static_cast<double*>(output),shape,axes,border,cval,directional);
    else if(type==PixelType::uint8)run(static_cast<const std::uint8_t*>(input),static_cast<double*>(output),shape,axes,border,cval,directional);
    else {
        std::unique_ptr<double[]> prepared(new double[count]);
        convert_into(input,prepared.get(),count,type,PixelType::float64);
        run(prepared.get(),static_cast<double*>(output),shape,axes,border,cval,directional);
    }
}
void farid_half_into(const std::uint16_t* input,float* output,const std::vector<std::size_t>& shape,
    const std::vector<std::ptrdiff_t>& axes,FaridBorder border,double cval,bool directional) {
    const auto count=checked_count(shape,sizeof(float));std::unique_ptr<float[]> prepared(new float[count]);
    edge_half_to_float(input,prepared.get(),count);run(prepared.get(),output,shape,axes,border,cval,directional);
}
}
