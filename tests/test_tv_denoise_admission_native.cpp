#include "../src/tv_denoise_internal.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

// Private harness compiles EXACT5e source, renaming ONLY these two functions.
// It shares the same unchanged stage/reduction/image objects and worker state.
namespace skimagexx {
const char* tv_denoise_baseline_ineligibility(const TvConstView&,const TvDenoiseWorkspace&,
    const TvDenoiseParameters&,TvDenoiseTrace*,std::size_t) noexcept;
TvDenoiseResult tv_denoise_baseline_finite_into(const TvConstView&,const TvDenoiseWorkspace&,
    const TvDenoiseParameters&,TvDenoiseTrace*,std::size_t);
}

namespace {
std::size_t checks=0,accepted=0,rejected=0,early_stops=0;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);++checks;}
template<class T> bool exact(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(T));}
template<class T> struct Buffers {
    std::vector<T> p,g,d,out,squared,norm;
    skimagexx::TvDenoiseWorkspace w;
    explicit Buffers(const std::vector<std::size_t>& shape){
        using namespace skimagexx;std::size_t n=1;for(auto s:shape)n*=s;
        std::vector<std::ptrdiff_t> strides(shape.size());std::size_t step=sizeof(T);
        for(std::size_t a=shape.size();a-->0;){strides[a]=std::ptrdiff_t(step);step*=shape[a];}
        const auto type=sizeof(T)==4?TvScalarType::float32:TvScalarType::float64;
        p.assign(n*shape.size(),T(0));g=p;d.assign(n,T(0));out.assign(n,T(17));squared.assign(n,T(18));norm.assign(n,T(19));
        w={{p.data(),type,shape},{g.data(),type,shape},{d.data(),type,shape,strides},
           {out.data(),type,shape,strides},{squared.data(),type,shape,strides},{norm.data(),type,shape,strides},
           {p.data(),type,shape},{g.data(),type,shape},{out.data(),type,shape,strides},{norm.data(),type,shape,strides}};
    }
    auto bytes()const{return std::array<std::vector<T>,6>{p,g,d,out,squared,norm};}
};
template<class T> struct Input {
    std::vector<unsigned char> storage;
    skimagexx::TvConstView view;
    Input(const std::vector<std::size_t>& shape,int layout,bool constant=false){
        std::size_t n=1;for(auto v:shape)n*=v;
        std::vector<std::ptrdiff_t> strides(shape.size());std::size_t step=sizeof(T);
        if(layout==1){for(std::size_t a=0;a<shape.size();++a){strides[a]=std::ptrdiff_t(step);step*=shape[a];}}
        else {for(std::size_t a=shape.size();a-->0;){strides[a]=std::ptrdiff_t(step);step*=shape[a]*(layout==3?2:1);}}
        std::size_t start=layout==4?1:0;
        if(layout==2){start+=(shape[0]-1)*std::size_t(strides[0]);strides[0]=-strides[0];}
        if(layout==5)strides[0]=0;
        storage.assign(step+sizeof(T)+1,0x5a);
        for(std::size_t i=0;i<n;++i){std::size_t rest=i;std::ptrdiff_t offset=0;
            for(std::size_t a=shape.size();a-->0;){offset+=std::ptrdiff_t(rest%shape[a])*strides[a];rest/=shape[a];}
            const T value=constant?T(.25):T((i*7+i/5)%31)/T(16);
            std::memcpy(storage.data()+start+offset,&value,sizeof(T));
        }
        view={storage.data()+start,sizeof(T)==4?skimagexx::TvScalarType::float32:skimagexx::TvScalarType::float64,shape,strides};
    }
};
bool same_trace(const skimagexx::TvDenoiseTrace& a,const skimagexx::TvDenoiseTrace& b){
    return a.square_sum_bits==b.square_sum_bits&&a.norm_sum_bits==b.norm_sum_bits&&a.energy_bits==b.energy_bits&&
        a.difference_bits==b.difference_bits&&a.threshold_bits==b.threshold_bits&&a.energy_is_float32==b.energy_is_float32&&a.stopped==b.stopped;
}
template<class T> void success(const std::vector<std::size_t>& shape,int layout,
    skimagexx::TvReductionProfile profile,std::size_t iterations,double eps,bool trace,bool constant=false,double weight=.5){
    using namespace skimagexx;Input<T> input(shape,layout,constant);const auto before=input.storage;
    Buffers<T> old(shape),now(shape),sdk(shape);
    TvDenoiseParameters parameters{weight,eps,iterations,{profile,layout%2?17u:8192u}};
    std::vector<TvDenoiseTrace> a(iterations),b(iterations),c(iterations);
    for(auto& row:a)row={0x5a5a5a5a5a5a5a5aull,0x5a5a5a5a5a5a5a5aull,0x5a5a5a5a5a5a5a5aull,0x5a5a5a5a5a5a5a5aull,0x5a5a5a5a5a5a5a5aull,false,false};b=a;c=a;
    require(!tv_denoise_baseline_ineligibility(input.view,old.w,parameters,trace?a.data():nullptr,trace?a.size():0),"baseline fixture rejected");
    std::feclearexcept(FE_ALL_EXCEPT);const auto first=tv_denoise_baseline_finite_into(input.view,old.w,parameters,trace?a.data():nullptr,trace?a.size():0);const int af=std::fetestexcept(FE_ALL_EXCEPT);
    TvDenoiseResult second{999,true};std::feclearexcept(FE_ALL_EXCEPT);
    require(!detail::tv_denoise_try_finite_into(input.view,now.w,parameters,second,trace?b.data():nullptr,trace?b.size():0),"checked valid fixture rejected");const int bf=std::fetestexcept(FE_ALL_EXCEPT);
    std::feclearexcept(FE_ALL_EXCEPT);const auto third=tv_denoise_finite_into(input.view,sdk.w,parameters,trace?c.data():nullptr,trace?c.size():0);const int cf=std::fetestexcept(FE_ALL_EXCEPT);
    require(first.iterations==second.iterations&&first.input_alias==second.input_alias&&first.iterations==third.iterations&&first.input_alias==third.input_alias,"iteration/alias dispatch differs");
    require(af==bf&&af==cf,"accepted FP flags differ");require(before==input.storage,"input storage changed");
    const auto aa=old.bytes(),bb=now.bytes(),cc=sdk.bytes();for(std::size_t k=0;k<aa.size();++k){require(exact(aa[k],bb[k]),"checked raw workspace differs");require(exact(aa[k],cc[k]),"public SDK raw workspace differs");}
    if(trace)for(std::size_t i=0;i<iterations;++i){
        if(i<first.iterations){require(same_trace(a[i],b[i])&&same_trace(a[i],c[i]),"raw energy/stopping trace differs");}
        else {require(same_trace(a[i],b[i])&&same_trace(a[i],c[i])&&a[i].square_sum_bits==0x5a5a5a5a5a5a5a5aull&&!a[i].stopped,"unused trace fields changed");}
    }
    if(first.iterations<iterations)++early_stops;
    if(iterations==1){require(first.input_alias&&first.iterations==1,"one-body alias semantics changed");for(T v:now.out)require(v==T(17),"one-body uninitialized output exposed");}
    if(constant)require(first.iterations==iterations,"zero-energy strict stopping changed");
    ++accepted;
}

template<class T> void rejection_controls(){
    using namespace skimagexx;const std::vector<std::size_t> shape{3,5};Input<T> input(shape,0);Buffers<T> buffers(shape);
    TvDenoiseParameters parameters{.5,0.,3,{TvReductionProfile::numpy_254_aligned_contiguous,8192}};
    std::vector<TvDenoiseTrace> trace(3);for(auto& row:trace)row={1,2,3,4,5,false,false};
    auto reject=[&](TvConstView image,TvDenoiseWorkspace w,TvDenoiseParameters p,TvDenoiseTrace* rows,std::size_t capacity){
        const auto data=input.storage;const auto bb=buffers.bytes();
        std::vector<unsigned char> tt(trace.size()*sizeof(TvDenoiseTrace));std::memcpy(tt.data(),trace.data(),tt.size());
        TvDenoiseResult result;std::memset(&result,0xa5,sizeof(result));result.iterations=999;result.input_alias=true;
        std::array<unsigned char,sizeof(TvDenoiseResult)> before{};std::memcpy(before.data(),&result,sizeof(result));
        const char* old=tv_denoise_baseline_ineligibility(image,w,p,rows,capacity);require(old!=nullptr,"bad control unexpectedly eligible");
        const int flags=std::fetestexcept(FE_ALL_EXCEPT);const int rounding=std::fegetround();
        const char* reason=detail::tv_denoise_try_finite_into(image,w,p,result,rows,capacity);
        require(reason&&std::strcmp(reason,old)==0,"checked rejection reason differs");
        require(!std::memcmp(&result,before.data(),sizeof(result)),"rejection changed result object");
        bool thrown=false;try{tv_denoise_finite_into(image,w,p,rows,capacity);}catch(const std::invalid_argument& e){thrown=std::strcmp(e.what(),old)==0;}
        require(thrown,"public SDK lost exact invalid_argument reason");
        require(flags==std::fetestexcept(FE_ALL_EXCEPT)&&rounding==std::fegetround(),"rejection changed FP state");
        const auto after=buffers.bytes();for(std::size_t k=0;k<bb.size();++k)require(exact(bb[k],after[k]),"rejection stored workspace bytes");
        require(data==input.storage&&!std::memcmp(trace.data(),tt.data(),tt.size()),"rejection stored input/trace bytes");++rejected;
    };
    auto test=[&](TvConstView image,TvDenoiseWorkspace w,TvDenoiseParameters p){reject(image,w,p,trace.data(),trace.size());};
    auto p=parameters;
    for(double weight:{0.,-0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),std::ldexp(1.,-21),std::ldexp(1.,21)}){p=parameters;p.weight=weight;test(input.view,buffers.w,p);}
    for(double eps:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),std::ldexp(1.,21)}){p=parameters;p.eps=eps;test(input.view,buffers.w,p);}
    p=parameters;p.max_iterations=0;test(input.view,buffers.w,p);p.max_iterations=20001;test(input.view,buffers.w,p);
    p=parameters;p.reduction.buffer_elements=0;test(input.view,buffers.w,p);p=parameters;p.reduction.profile=static_cast<TvReductionProfile>(99);test(input.view,buffers.w,p);
    auto image=input.view;image.shape.clear();image.strides.clear();test(image,buffers.w,parameters);
    image=input.view;image.shape={1,1,1,1,15};image.strides={0,0,0,0,std::ptrdiff_t(sizeof(T))};test(image,buffers.w,parameters);
    image=input.view;image.shape[0]=0;test(image,buffers.w,parameters);image=input.view;image.data=nullptr;test(image,buffers.w,parameters);
    image=input.view;image.shape[0]=std::size_t{1}<<30;test(image,buffers.w,parameters);
    image=input.view;image.strides[0]=std::numeric_limits<std::ptrdiff_t>::min();test(image,buffers.w,parameters);
    image=input.view;image.data=buffers.d.data();test(image,buffers.w,parameters);
    auto w=buffers.w;w.out.data=buffers.d.data();w.out_read.data=buffers.d.data();test(input.view,w,parameters);
    w=buffers.w;w.norm_read.data=buffers.squared.data();test(input.view,w,parameters);
    w=buffers.w;w.out.strides[0]=sizeof(T);test(input.view,w,parameters);
    w=buffers.w;w.p_read.shape[0]=1;test(input.view,w,parameters);
    w=buffers.w;w.g.type=sizeof(T)==4?TvScalarType::float64:TvScalarType::float32;test(input.view,w,parameters);
    w=buffers.w;w.p.data=reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max()-1);w.p_read.data=w.p.data;test(input.view,w,parameters);
    for(auto* v:{&buffers.p,&buffers.g,&buffers.d}){(*v)[0]=-T(0);test(input.view,buffers.w,parameters);(*v)[0]=T(1);test(input.view,buffers.w,parameters);(*v)[0]=T(0);}
    const auto original=input.storage;
    for(T value:{std::numeric_limits<T>::infinity(),std::numeric_limits<T>::quiet_NaN(),T(2097152)}){std::memcpy(input.storage.data(),&value,sizeof(T));test(input.view,buffers.w,parameters);std::memcpy(input.storage.data(),original.data(),original.size());}
    reject(input.view,buffers.w,parameters,trace.data(),2);
    reject(input.view,buffers.w,parameters,reinterpret_cast<TvDenoiseTrace*>(buffers.out.data()),3);
    const int rounding=std::fegetround();std::fesetround(FE_DOWNWARD);test(input.view,buffers.w,parameters);std::fesetround(rounding);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto csr=_mm_getcsr();_mm_setcsr(csr|(1u<<15));test(input.view,buffers.w,parameters);_mm_setcsr(csr);
    _mm_setcsr(csr|(1u<<6));test(input.view,buffers.w,parameters);_mm_setcsr(csr);
#endif
}

template<class T> void accepted_controls(){
    using namespace skimagexx;
    for(auto profile:{TvReductionProfile::numpy_1244_buffered,TvReductionProfile::numpy_254_aligned_contiguous}){
        for(const auto& shape:std::vector<std::vector<std::size_t>>{{1},{17},{3,5},{2,3,5},{2,3,4,5}})
            for(int layout=0;layout<6;++layout)
                for(std::size_t iterations:{std::size_t(1),std::size_t(2),std::size_t(7)})
                    for(bool trace:{false,true})success<T>(shape,layout,profile,iterations,iterations==7?1048576.:0.,trace);
        success<T>({3,5},0,profile,5,.1,true,true);
        success<T>({3,5},0,profile,3,-0.,true,false,std::ldexp(1.,-20));
        success<T>({3,5},1,profile,3,1048576.,true,false,std::ldexp(1.,20));
        success<T>({513,519},0,profile,3,0.,true);
        success<T>({513,519},2,profile,3,0.,false);
    }
}
} // namespace

int main(){
    try{
        require(skimagexx::get_num_threads()==1,"default worker limit changed");
        for(int workers:{1,4}){
#ifndef _OPENMP
            if(workers==4){bool denied=false;try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){denied=true;}require(denied&&skimagexx::get_num_threads()==1,"noOMP4 must reject, not serial-pass");continue;}
#endif
            skimagexx::set_num_threads(workers);rejection_controls<float>();rejection_controls<double>();accepted_controls<float>();accepted_controls<double>();
        }
        require(early_stops>0,"early stopping controls did not execute");skimagexx::set_num_threads(1);
        std::cout<<"TV checked admission "<<checks<<" assertions; accepted="<<accepted<<" rejected="<<rejected<<" early_stops="<<early_stops<<"\n";
    }catch(const std::exception& e){std::cerr<<"TV checked admission FAILED after "<<checks<<": "<<e.what()<<'\n';return 1;}
}
