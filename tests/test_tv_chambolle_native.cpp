#include "skimagexx/tv_chambolle.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace skimagexx;
namespace {
std::size_t checks=0;
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);++checks;}
template<class T> T rounded(T value){volatile T result=value;return result;}
template<class T> constexpr TvScalarType type(){return std::is_same_v<T,float> ? TvScalarType::float32:TvScalarType::float64;}
std::size_t size(const std::vector<std::size_t>& shape){std::size_t n=1;for(auto v:shape)n*=v;return n;}
std::vector<std::size_t> steps(const std::vector<std::size_t>& shape){std::vector<std::size_t> r(shape.size());std::size_t n=1;for(auto a=shape.size();a-->0;){r[a]=n;n*=shape[a];}return r;}
template<class T> T read(const void* p,std::ptrdiff_t byte){T value;std::memcpy(&value,static_cast<const unsigned char*>(p)+byte,sizeof(T));return value;}
template<class T> void write(void* p,std::ptrdiff_t byte,T value){std::memcpy(static_cast<unsigned char*>(p)+byte,&value,sizeof(T));}
template<class T> struct Array {
    std::vector<unsigned char> bytes;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    std::size_t origin=0;
    Array(std::vector<std::size_t> dimensions,int layout):shape(std::move(dimensions)),strides(shape.size()) {
        std::ptrdiff_t pitch=sizeof(T);
        for(std::size_t k=0;k<shape.size();++k) {
            const auto a=layout==1 ? k:shape.size()-1-k;strides[a]=pitch;
            pitch*=std::ptrdiff_t(std::max(std::size_t{1},shape[a]));
            if(layout==2)pitch+=3;
        }
        if(layout==3)strides.back()=-strides.back();
        if(layout==4)strides.front()=0; // Read-only broadcast test.
        std::ptrdiff_t low=0,high=0;
        for(std::size_t a=0;a<shape.size();++a)if(shape[a]){const auto distance=std::ptrdiff_t(shape[a]-1)*strides[a];if(distance<0)low+=distance;else high+=distance;}
        origin=std::size_t(-low)+17;bytes.assign(std::size_t(high-low)+sizeof(T)+34,0xa5);
    }
    void* data(){return bytes.data()+origin;}
    const void* data()const{return bytes.data()+origin;}
    std::ptrdiff_t offset(std::size_t i)const {std::ptrdiff_t result=0;for(std::size_t a=shape.size();a-->0;){result+=std::ptrdiff_t(i%shape[a])*strides[a];i/=shape[a];}return result;}
    TvView view(){return {data(),type<T>(),shape,strides};}
    TvConstView const_view()const{return {data(),type<T>(),shape,strides};}
};
template<class T> T sample(std::size_t i) {
    if(i%19==0)return -T(0);
    const auto sign=i%2 ? T(-1):T(1);
    return rounded(sign*rounded(T((i*7919+17)%997)/T(37)));
}
template<class T> void stage_case(const std::vector<std::size_t>& shape,int input_layout,int output_layout,int workers) {
    const auto n=size(shape),rank=shape.size();const auto st=steps(shape);
    Array<T> input(shape,input_layout),d(shape,output_layout),norm(shape,output_layout),denominator(shape,input_layout);
    for(std::size_t i=0;i<n;++i){write<T>(input.data(),input.offset(i),sample<T>(i));write<T>(denominator.data(),denominator.offset(i),T(1)+T(i%11)/T(4));}
    const auto original_input=input.bytes,original_denominator=denominator.bytes;
    std::vector<unsigned char> p(rank*n*sizeof(T)+2,0xa5),g(p.size(),0xa5);
    void* pp=p.data()+1;void* gg=g.data()+1;
    for(std::size_t i=0;i<rank*n;++i){write<T>(pp,std::ptrdiff_t(i*sizeof(T)),sample<T>(i+13));write<T>(gg,std::ptrdiff_t(i*sizeof(T)),T(99));}
    const auto p_before=p;
    auto d_expected=d.bytes,norm_expected=norm.bytes,g_expected=g;
    for(std::size_t i=0;i<n;++i) {
        T sum=T(0);
        for(std::size_t a=0;a<rank;++a)sum=rounded<T>(sum+read<T>(pp,std::ptrdiff_t((a*n+i)*sizeof(T))));
        T divergence=rounded<T>(-sum);
        for(std::size_t a=0;a<rank;++a)if((i/st[a])%shape[a])divergence=rounded<T>(divergence+read<T>(pp,std::ptrdiff_t((a*n+i-st[a])*sizeof(T))));
        write<T>(d_expected.data()+d.origin,d.offset(i),divergence);
        T energy=T(0);
        for(std::size_t a=0;a<rank;++a) {
            T gradient=T(0);
            if((i/st[a])%shape[a]+1<shape[a])gradient=rounded<T>(read<T>(input.data(),input.offset(i+st[a]))-read<T>(input.data(),input.offset(i)));
            write<T>(g_expected.data()+1,std::ptrdiff_t((a*n+i)*sizeof(T)),gradient);
            const T square=rounded<T>(gradient*gradient);energy=rounded<T>(energy+square);
        }
        write<T>(norm_expected.data()+norm.origin,norm.offset(i),rounded<T>(std::sqrt(energy)));
    }
    set_num_threads(workers);
    tv_chambolle_divergence_into({pp,type<T>(),shape},d.view());
    require(d.bytes==d_expected,"divergence or output guard differs from typed reference");require(p==p_before,"divergence changed p");
    tv_chambolle_gradient_into(input.const_view(),{gg,type<T>(),shape});
    require(g==g_expected,"gradient/terminal planes/guard differs from typed reference");
    tv_chambolle_norm_into({gg,type<T>(),shape},norm.view());
    require(norm.bytes==norm_expected,"norm typed axis fold/sqrt differs");
    auto p_expected=p;
    const T tau=T(1./(2.*double(rank)));
    for(std::size_t i=0;i<n;++i)for(std::size_t a=0;a<rank;++a) {
        const auto byte=std::ptrdiff_t((a*n+i)*sizeof(T));const T product=rounded<T>(tau*read<T>(gg,byte));
        const T delta=rounded<T>(read<T>(pp,byte)-product);const T updated=rounded<T>(delta/read<T>(denominator.data(),denominator.offset(i)));
        write<T>(p_expected.data()+1,byte,updated);
    }
    tv_chambolle_dual_into({pp,type<T>(),shape},{gg,type<T>(),shape},denominator.const_view(),double(tau));
    require(p==p_expected,"dual multiply/subtract/divide typed stages differ");
    require(input.bytes==original_input && denominator.bytes==original_denominator,"stage modified an input");
    set_num_threads(1);
}
template<class T> void zero_and_overflow_controls() {
    for(std::size_t rank=1;rank<=4;++rank) {
        std::vector<std::size_t> shape(rank,1);shape.back()=17;const auto n=size(shape);std::vector<T> p(n*rank,-T(0)),out(n,T(99)),g(n*rank,-T(0));
        const auto st=steps(shape);std::vector<std::ptrdiff_t> strides;for(auto step:st)strides.push_back(std::ptrdiff_t(step*sizeof(T)));
        tv_chambolle_divergence_into({p.data(),type<T>(),shape},{out.data(),type<T>(),shape,strides});
        require(out[0]==T(0) && std::signbit(out[0]),"divergence sum must start at +0 before negation");
        tv_chambolle_norm_into({g.data(),type<T>(),shape},{out.data(),type<T>(),shape,strides});
        require(std::all_of(out.begin(),out.end(),[](T x){return x==T(0) && !std::signbit(x);}),"norm zero sign differs");
        g[0]=std::numeric_limits<T>::max();tv_chambolle_norm_into({g.data(),type<T>(),shape},{out.data(),type<T>(),shape,strides});require(std::isinf(out[0]) && !std::signbit(out[0]),"finite norm square overflow must remain IEEE infinity");
        g[0]=std::numeric_limits<T>::quiet_NaN();tv_chambolle_norm_into({g.data(),type<T>(),shape},{out.data(),type<T>(),shape,strides});require(std::isnan(out[0]),"norm NaN classification differs");
    }
}
template<class T> void independent_first_update(std::vector<std::size_t> shape,T weight) {
    // Literal two-iteration Chambolle definition: g=forwarddiff(image),
    // p=-tau*g/(1+tau*norm/weight), out=image+negative-divergence(p).
    const auto n=size(shape),rank=shape.size();Array<T> image(shape,0),d(shape,0),norm(shape,0);const auto st=steps(shape);
    std::vector<T> p(n*rank,T(0)),g(n*rank,T(0));for(std::size_t i=0;i<n;++i)write<T>(image.data(),image.offset(i),T((i*7)%11)/T(8));
    tv_chambolle_gradient_into(image.const_view(),{g.data(),type<T>(),shape});tv_chambolle_norm_into({g.data(),type<T>(),shape},norm.view());
    const T tau=T(1./(2.*double(rank)));
    for(std::size_t i=0;i<n;++i)write<T>(norm.data(),norm.offset(i),rounded<T>(rounded<T>(read<T>(norm.data(),norm.offset(i))*rounded<T>(tau/weight))+T(1)));
    tv_chambolle_dual_into({p.data(),type<T>(),shape},{g.data(),type<T>(),shape},norm.const_view(),double(tau));tv_chambolle_divergence_into({p.data(),type<T>(),shape},d.view());
    for(std::size_t i=0;i<n;++i) {
        T expected=T(0);
        for(std::size_t a=0;a<rank;++a)expected=rounded<T>(expected+p[a*n+i]);expected=rounded<T>(-expected);
        for(std::size_t a=0;a<rank;++a)if((i/st[a])%shape[a])expected=rounded<T>(expected+p[a*n+i-st[a]]);
        require(std::memcmp(&expected,static_cast<unsigned char*>(d.data())+d.offset(i),sizeof(T))==0,"independent first dual stencil differs");
    }
}
void prewrite_guards() {
    std::vector<double> p(24,1),g(24,2),output(12,99);const auto unchanged=output;
    auto reject=[&](auto fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}require(caught && output==unchanged,"invalid stage must reject before writing");};
    const TvConstPlanes planes{p.data(),TvScalarType::float64,{3,4}};
    reject([&]{tv_chambolle_divergence_into(planes,{output.data(),TvScalarType::float32,{3,4},{16,4}});});
    reject([&]{tv_chambolle_divergence_into(planes,{output.data(),TvScalarType::float64,{3,5},{40,8}});});
    reject([&]{tv_chambolle_divergence_into(planes,{output.data(),TvScalarType::float64,{3,4},{8,8}});});
    reject([&]{tv_chambolle_divergence_into(planes,{output.data(),TvScalarType::float64,{3,4},{}});});
    reject([&]{tv_chambolle_divergence_into({nullptr,TvScalarType::float64,{3,4}},{output.data(),TvScalarType::float64,{3,4},{32,8}});});
    reject([&]{tv_chambolle_divergence_into({p.data(),TvScalarType::float64,{}},{output.data(),TvScalarType::float64,{},{}});});
    reject([&]{tv_chambolle_divergence_into({p.data(),TvScalarType::float64,{1,1,1,1,1}},{output.data(),TvScalarType::float64,{1,1,1,1,1},{8,8,8,8,8}});});
    const auto p_before=p,g_before=g;
    bool caught=false;try{tv_chambolle_gradient_into({p.data(),TvScalarType::float64,{3,4},{32,8}},{p.data(),TvScalarType::float64,{3,4}});}catch(const std::invalid_argument&){caught=true;}require(caught && p==p_before,"gradient overlap must reject prewrite");
    caught=false;try{tv_chambolle_dual_into({p.data(),TvScalarType::float64,{3,4}},{g.data(),TvScalarType::float64,{3,4}},{p.data(),TvScalarType::float64,{3,4},{32,8}},.25);}catch(const std::invalid_argument&){caught=true;}require(caught && p==p_before && g==g_before,"dual overlap must reject prewrite");
    tv_chambolle_divergence_into({nullptr,TvScalarType::float64,{0,3}},{nullptr,TvScalarType::float64,{0,3},{24,8}});
    caught=false;try{tv_chambolle_gradient_into({nullptr,TvScalarType::float32,{2,0},{}},{nullptr,TvScalarType::float32,{2,0}});}catch(const std::invalid_argument&){caught=true;}require(caught,"invalid stride metadata must reject even when empty");
}
}
int main() {
    try {
        require(get_num_threads()==1,"global workers must default to one");require(tv_chambolle_capabilities()==0x3f,"private stage capabilities differ");
        const std::vector<std::vector<std::size_t>> shapes={{1},{2},{3},{4},{7},{8},{9},{15},{16},{17},{31},{32},{33},{1,1},{1,17},{3,17},{4,5,7},{2,1,3,9},{0,3}};
        for(const auto& shape:shapes)for(int input_layout=0;input_layout<5;++input_layout)for(int output_layout=0;output_layout<4;++output_layout){stage_case<float>(shape,input_layout,output_layout,1);stage_case<double>(shape,input_layout,output_layout,1);}
        zero_and_overflow_controls<float>();zero_and_overflow_controls<double>();
        // At least the24tiny stencil/energy-shape controls, independently
        // supplemented above by all stages/layouts/type widths/tails.
        for(const auto& shape:std::vector<std::vector<std::size_t>>{{2},{3},{2,3},{1,3,1},{2,2,3},{1,1,1,2}})for(double w:{.1,.5}){independent_first_update<float>(shape,float(w));independent_first_update<double>(shape,w);}
        for(int worker:{1,4}) {
#ifndef _OPENMP
            if(worker==4){bool rejected=false;try{set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}require(rejected && get_num_threads()==1,"noOMP must reject worker4");continue;}
#endif
            stage_case<float>({513,513},0,0,worker);stage_case<double>({65,65,65},0,0,worker);stage_case<float>({1,1,513,513},2,3,worker);
        }
        // Empty valid metadata is supported; malformed descriptors still fail.
        tv_chambolle_gradient_into({nullptr,TvScalarType::float32,{2,0},{0,4}},{nullptr,TvScalarType::float32,{2,0}});
        prewrite_guards();
        require(get_num_threads()==1,"global worker state not restored");std::cout<<"TV private typed stages "<<checks<<" checks passed; SIMD="<<has_avx2()<<"\n";
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
