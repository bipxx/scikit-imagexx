// scikit-image0.26 radon_transform.py source recurrence, BSD-3-Clause.
#include <skimagexx/order_angles_golden_ratio.hpp>
#include <skimagexx/image.hpp>
#include "half.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace skimagexx { namespace {
// Exact already-admitted IGG scalar/ufunc boundary primitive. NumPy's retained
// npy_clear_floatstatus_barrier clears0x1d, NOT inexact or denormal, before real
// scalar arithmetic. Never restore/mask the outgoing target status to gold.
void source_clear(){constexpr int mask=FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW;if(std::fetestexcept(mask))std::feclearexcept(mask);}
std::size_t bytes(const ComponentsView& v) {
    if(v.half)return 2;
    switch(v.type){case PixelType::boolean:case PixelType::uint8:case PixelType::int8:return 1;case PixelType::uint16:case PixelType::int16:return 2;case PixelType::uint32:case PixelType::int32:case PixelType::float32:return 4;default:return 8;}
}
void validate(const ComponentsView& v) {
    if(v.shape.size()!=1||v.strides.size()!=1)throw std::invalid_argument("golden angles require a1D view");
    const auto n=v.shape[0];if(n&&!v.data)throw std::invalid_argument("golden angles missing storage");
    if(n>std::size_t(PTRDIFF_MAX)/sizeof(std::size_t))throw std::overflow_error("golden angles size overflow");
    const auto stride=v.strides[0];const auto magnitude=stride<0?std::uintmax_t(-(stride+1))+1:std::uintmax_t(stride);
    if(n>1&&magnitude>std::uintmax_t(PTRDIFF_MAX)/(n-1))throw std::overflow_error("golden angles stride overflow");
}
template<class T>T word(const ComponentsView& v,std::size_t i) {
    unsigned char data[sizeof(T)];std::memcpy(data,static_cast<const unsigned char*>(v.data)+std::ptrdiff_t(i)*v.strides[0],sizeof(T));
    if(v.swap)std::reverse(data,data+sizeof(T));T result;std::memcpy(&result,data,sizeof(T));return result;
}
double value(const ComponentsView& v,std::size_t i) {
    if(v.half)return detail::decode_half(word<std::uint16_t>(v,i));
    double out=0;
    switch(v.type){
    case PixelType::boolean:case PixelType::uint8:out=word<std::uint8_t>(v,i);break;
    case PixelType::int8:out=word<std::int8_t>(v,i);break;
    case PixelType::uint16:out=word<std::uint16_t>(v,i);break;
    case PixelType::int16:out=word<std::int16_t>(v,i);break;
    case PixelType::uint32:out=word<std::uint32_t>(v,i);break;
    case PixelType::int32:out=word<std::int32_t>(v,i);break;
    case PixelType::uint64:{auto x=word<std::uint64_t>(v,i);if(x>UINT64_C(9007199254740992))throw std::invalid_argument("golden uint64 exact-conversion frontier");out=double(x);break;}
    case PixelType::int64:{auto x=word<std::int64_t>(v,i);if(x>INT64_C(9007199254740992)||x< -INT64_C(9007199254740992))throw std::invalid_argument("golden int64 exact-conversion frontier");out=double(x);break;}
    case PixelType::float32:out=word<float>(v,i);break;
    case PixelType::float64:out=word<double>(v,i);break;
    }
    if(!std::isfinite(out))throw std::invalid_argument("golden nonfinite frontier");return out;
}
double quantize(double v,unsigned bits) {
    if(bits==16)return detail::decode_half(detail::encode_half(float(v)));
    if(bits==32)return float(v);return v;
}
template<class T>T remainder_source(T a,T b) {
    T mod=std::fmod(a,b);T div=(a-mod)/b;
    if(mod!=T(0)){if((b<T(0))!=(mod<T(0))){mod+=b;div-=T(1);}}
    else mod=std::copysign(T(0),b);
    T floor_div;
    if(div!=T(0)){floor_div=std::floor(div);if(div-floor_div>T(.5))floor_div+=T(1);}
    else floor_div=std::copysign(T(0),a/b);
    // NumPy's out-of-line divmod computes the quotient even when remainder is
    // requested. Retain those source arithmetic effects, not just fmod's value.
    volatile T retained=floor_div;(void)retained;return mod;
}
double add(double a,double b,unsigned p){source_clear();a=quantize(a,p);b=quantize(b,p);return quantize(p<64?double(float(a)+float(b)):a+b,p);}
double difference(double a,double b,unsigned p){source_clear();a=quantize(a,p);b=quantize(b,p);return quantize(p<64?double(float(a)-float(b)):a-b,p);}
double absolute(double a,unsigned p){source_clear();return quantize(std::fabs(a),p);}
double rem(double a,double b,unsigned p){source_clear();a=quantize(a,p);b=quantize(b,p);return quantize(p<64?double(remainder_source(float(a),float(b))):remainder_source(a,b),p);}
}

GoldenAngleOrder::GoldenAngleOrder(const ComponentsView& view,GoldenScalarPolicy policy,int workers):view_(view),policy_(policy),requested_(workers){validate(view);}
std::int64_t GoldenAngleOrder::begin() {
    if(phase_)throw std::logic_error("golden begin repeated");
    workers_=requested_?requested_:get_num_threads();if(workers_<1||workers_>1024)throw std::invalid_argument("golden worker budget");
    const auto n=view_.shape[0];std::vector<double> values(n);for(std::size_t i=0;i<n;++i)values[i]=value(view_,i);
    remaining_.resize(n);std::iota(remaining_.begin(),remaining_.end(),std::size_t(0));
    // Stable insertion ordering matches NumPy quicksort's small-array leaf;
    // larger duplicate tie order is explicitly unadmitted, never speed credit.
    std::stable_sort(remaining_.begin(),remaining_.end(),[&](auto a,auto b){return values[a]<values[b];});
    if(!n)throw std::out_of_range("list index out of range");
    const auto first=remaining_.front();angle_=values[first];remaining_.erase(remaining_.begin());phase_=1;
    if(policy_==GoldenScalarPolicy::weak_input){if(view_.half)precision_=16;else if(view_.type==PixelType::float32)precision_=32;}
    return std::int64_t(first);
}
bool GoldenAngleOrder::resume() {
    if(!phase_)throw std::logic_error("golden resume before begin");
    if(phase_==1){const double golden=0x1.9e3779b97f4a8p+0;increment_=180./std::pow(golden,2.);phase_=2;}
    return !remaining_.empty();
}
std::int64_t GoldenAngleOrder::next(const ComponentsView& live) {
    if(phase_!=2||remaining_.empty())throw std::logic_error("golden next outside resumed state");
    validate(live);
    if(live.data!=view_.data||live.type!=view_.type||live.half!=view_.half||live.swap!=view_.swap||live.shape!=view_.shape||live.strides!=view_.strides)throw std::invalid_argument("golden live descriptor changed");
    std::vector<double> gathered;gathered.reserve(remaining_.size());for(auto i:remaining_)gathered.push_back(value(live,i));
    angle_=rem(add(angle_,increment_,precision_),180.,precision_);
    std::size_t low=0,high=gathered.size();while(low<high){const auto mid=low+(high-low)/2;if(gathered[mid]<angle_)low=mid+1;else high=mid;}
    const auto below=low?low-1:gathered.size()-1,above=low%gathered.size();
    const auto diff_below=absolute(difference(angle_,gathered[below],precision_),precision_);
    const auto below_positive=rem(diff_below,180.,precision_),below_negative=rem(diff_below,-180.,precision_);
    const auto distance_below=std::min(below_positive,below_negative);
    const auto diff_above=absolute(difference(angle_,gathered[above],precision_),precision_);
    const auto above_positive=rem(diff_above,180.,precision_),above_negative=rem(diff_above,-180.,precision_);
    const auto distance_above=std::min(above_positive,above_negative);
    const auto selected=distance_below<distance_above?below:above;
    const auto result=remaining_[selected];remaining_.erase(remaining_.begin()+std::ptrdiff_t(selected));return std::int64_t(result);
}
std::vector<std::int64_t> order_angles_golden_ratio(const ComponentsView& v,GoldenScalarPolicy p,int workers) {
    GoldenAngleOrder state(v,p,workers);std::vector<std::int64_t> out;out.reserve(v.shape.at(0));out.push_back(state.begin());while(state.resume())out.push_back(state.next(v));return out;
}
}
