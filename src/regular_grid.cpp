#include "skimagexx/regular_grid.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace skimagexx { namespace {
struct Type {unsigned bytes;bool sign=false,floating=false,boolean=false;};
Type type(PixelType t) {
    switch(t){
        case PixelType::boolean:return {1,false,false,true};
        case PixelType::int8:return {1,true};case PixelType::uint8:return {1};
        case PixelType::int16:return {2,true};case PixelType::uint16:return {2};
        case PixelType::int32:return {4,true};case PixelType::uint32:return {4};
        case PixelType::int64:return {8,true};case PixelType::uint64:return {8};
        case PixelType::float32:return {4,false,true};case PixelType::float64:return {8,false,true};
        default:throw RegularGridError(RegularGridErrorKind::unsupported,"regular_grid shape dtype unsupported");
    }
}
std::uint64_t mask(unsigned bytes){return bytes==8 ? ~std::uint64_t(0) : (std::uint64_t(1)<<(8*bytes))-1;}
std::uint64_t extend(std::uint64_t word,unsigned bytes,bool sign){word&=mask(bytes);if(sign && bytes<8 && (word&(std::uint64_t(1)<<(bytes*8-1))))word|=~mask(bytes);return word;}
struct Value {std::uint64_t word;double number;std::size_t index;};
double product(const std::vector<Value>& values,std::size_t first,Type t,unsigned integer_bytes) {
    if(t.floating){
        if(t.bytes==4){float p=1;for(auto i=first;i<values.size();++i)p*=float(values[i].number);return p;}
        double p=1;for(auto i=first;i<values.size();++i)p*=values[i].number;return p;
    }
    const auto width=std::max(t.bytes,integer_bytes);std::uint64_t p=1;
    for(auto i=first;i<values.size();++i)p*=values[i].word;
    p=extend(p,width,t.sign || t.boolean);
    return t.sign || t.boolean ? double(std::int64_t(p)) : double(p);
}
double power(double space,double points,std::size_t rank) {
    if(points==0)throw RegularGridError(RegularGridErrorKind::division_zero,"float division by zero");
    const double quotient=space/points;
    if(rank==0)throw RegularGridError(RegularGridErrorKind::division_zero,"float division by zero");
    const double exponent=1.0/double(rank);
    if(quotient<0 && std::floor(exponent)!=exponent)throw RegularGridError(RegularGridErrorKind::complex_power,"float() argument must be a string or a real number, not 'complex'");
    return std::pow(quotient,exponent);
}
double even_round(double x){const auto low=std::floor(x),part=x-low;return part<.5 ? low : part>.5 ? low+1 : std::fmod(low,2)==0 ? low : low+1;}
std::int64_t integer(double x){if(!std::isfinite(x) || x < -0x1p63 || x>=0x1p63)throw RegularGridError(RegularGridErrorKind::unsupported,"regular_grid nonfinite/out-of-range integer conversion unsupported");return std::int64_t(x);}
} // namespace
RegularGridPrepared regular_grid_prepare(const ComponentsView& view,unsigned integer_bytes) {
    if(view.shape.size()!=1 || view.strides.size()!=1 || view.half || (integer_bytes!=4 && integer_bytes!=8))throw RegularGridError(RegularGridErrorKind::unsupported,"regular_grid requires a1D typed shape and explicit32/64-bit default integer");
    const auto t=type(view.type);const auto n=view.shape[0];
    if(n && !view.data)throw std::invalid_argument("nonnull regular_grid source required");
    std::vector<Value> values;values.reserve(n);
    for(std::size_t i=0;i<n;++i){
        std::uint64_t word=0;std::memcpy(&word,static_cast<const char*>(view.data)+std::ptrdiff_t(i)*view.strides[0],t.bytes);
        if(view.swap){std::uint64_t swapped=0;for(unsigned j=0;j<t.bytes;++j)swapped=(swapped<<8)|((word>>(8*j))&255);word=swapped;}
        word=t.boolean ? std::uint64_t(word!=0) : extend(word,t.bytes,t.sign);double number;
        if(t.floating){if(t.bytes==4){const auto bits=std::uint32_t(word);float x;std::memcpy(&x,&bits,4);number=x;}else std::memcpy(&number,&word,8);}
        else number=t.sign ? double(std::int64_t(word)) : double(word);
        if(!std::isfinite(number))throw RegularGridError(RegularGridErrorKind::unsupported,"regular_grid nonfinite dimension unsupported");
        values.push_back({word,number,i});
    }
    RegularGridPrepared out;out.space_size=product(values,0,t,integer_bytes);
    std::stable_sort(values.begin(),values.end(),[&](const auto& a,const auto& b){return t.floating ? a.number<b.number : t.sign ? std::int64_t(a.word)<std::int64_t(b.word) : a.word<b.word;});
    out.inverse_order.resize(n);out.source_type=view.type;out.default_integer_bytes=integer_bytes;
    for(std::size_t i=0;i<n;++i){out.sorted_dimensions.push_back(values[i].number);out.sorted_words.push_back(values[i].word);out.inverse_order[values[i].index]=i;}
    return out;
}
double regular_grid_suffix_product(const RegularGridPrepared& p,std::size_t first) {
    const auto n=p.sorted_dimensions.size();if(first>n || p.sorted_words.size()!=n)throw std::invalid_argument("invalid regular_grid suffix");
    std::vector<Value> values;values.reserve(n);for(std::size_t i=0;i<n;++i)values.push_back({p.sorted_words[i],p.sorted_dimensions[i],i});
    return product(values,first,type(p.source_type),p.default_integer_bytes);
}
std::vector<RegularGridSlice> regular_grid_finish(const RegularGridPrepared& p,double points) {
    const auto n=p.sorted_dimensions.size();if(p.inverse_order.size()!=n || p.sorted_words.size()!=n)throw std::invalid_argument("invalid regular_grid preparation");
    std::vector<RegularGridSlice> out(n);
    std::vector<double> steps(n,power(p.space_size,points,n));
    bool small=false;for(std::size_t i=0;i<n;++i)small=small || p.sorted_dimensions[i]<steps[i];
    if(small)for(std::size_t i=0;i<n;++i){
        steps[i]=p.sorted_dimensions[i];const auto value=power(regular_grid_suffix_product(p,i+1),points,n-i-1);
        std::fill(steps.begin()+i+1,steps.end(),value);
        bool fitted=true;for(std::size_t j=0;j<n;++j)fitted=fitted && p.sorted_dimensions[j]>=steps[j];if(fitted)break;
    }
    for(std::size_t i=0;i<n;++i){const auto j=p.inverse_order[i];out[i]={RegularGridSliceKind::strided,integer(std::floor(steps[j]/2)),integer(even_round(steps[j]))};}
    return out;
}
std::vector<RegularGridSlice> regular_grid(const ComponentsView& shape,double points,unsigned integer_bytes){
    const auto prepared=regular_grid_prepare(shape,integer_bytes);
    if(prepared.space_size<=points)return std::vector<RegularGridSlice>(prepared.sorted_dimensions.size());
    return regular_grid_finish(prepared,points);
}
} // namespace skimagexx
namespace skimagexx::detail {
bool regular_grid_slices_i64(const std::int64_t* shape,std::size_t ndim,std::uint64_t points,RegularGridSlice* output) {
    if((ndim && (!shape || !output)) || !points || points>(std::uint64_t(1)<<53))return false;
    std::uint64_t product=1;for(std::size_t i=0;i<ndim;++i){if(shape[i]<=0 || std::uint64_t(shape[i])>(std::uint64_t(1)<<53)/product)return false;product*=std::uint64_t(shape[i]);}
    auto result=skimagexx::regular_grid({shape,PixelType::int64,false,false,{ndim},{8}},double(points));std::copy(result.begin(),result.end(),output);return true;
}
} // namespace skimagexx::detail
