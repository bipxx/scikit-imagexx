#include "skimagexx/tv_denoise.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::size_t checks=0;
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);++checks;}
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size() && (!a.size()||!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)));}
template<class T> struct Work {
    std::vector<T> p,g,d,out,squared,norm;
    skimagexx::TvDenoiseWorkspace view;
    explicit Work(const std::vector<std::size_t>& shape) {
        using namespace skimagexx;
        std::size_t n=1;for(auto s:shape)n*=s;
        std::vector<std::ptrdiff_t> strides(shape.size());std::size_t step=sizeof(T);
        for(std::size_t a=shape.size();a-->0;){strides[a]=std::ptrdiff_t(step);step*=shape[a];}
        auto type=sizeof(T)==4?TvScalarType::float32:TvScalarType::float64;
        p.resize(n*shape.size());g.resize(p.size());d.resize(n);out.assign(n,T(99));squared.assign(n,T(98));norm.assign(n,T(97));
        view={{p.data(),type,shape},{g.data(),type,shape},{d.data(),type,shape,strides},{out.data(),type,shape,strides},{squared.data(),type,shape,strides},{norm.data(),type,shape,strides},{p.data(),type,shape},{g.data(),type,shape},{out.data(),type,shape,strides},{norm.data(),type,shape,strides}};
    }
};
// Independent recursive coordinate traversal: no division/modulo cursor and
// no production helper are reused by the byte-address oracle.
void addresses(const std::vector<std::size_t>& shape,const std::vector<std::ptrdiff_t>& stride,std::size_t axis,std::ptrdiff_t byte,std::vector<std::ptrdiff_t>& out){
    if(axis==shape.size()){out.push_back(byte);return;}
    for(std::size_t c=0;c<shape[axis];++c)addresses(shape,stride,axis+1,byte+std::ptrdiff_t(c)*stride[axis],out);
}
template<class T> void one(const std::vector<std::size_t>& shape,unsigned kind,int worker){
    using namespace skimagexx;
    std::size_t n=1;for(auto s:shape)n*=s;
    std::vector<std::ptrdiff_t> stride(shape.size());std::ptrdiff_t step=sizeof(T);
    if(kind==1){for(std::size_t a=0;a<shape.size();++a){stride[a]=step;step*=std::ptrdiff_t(shape[a]);}}
    else {for(std::size_t a=shape.size();a-->0;){stride[a]=step;step*=std::ptrdiff_t(shape[a]);}}
    if(kind==2)for(auto& s:stride)s*=2;
    if(kind==3)for(std::size_t a=0;a<shape.size();++a)if(a%2==0)stride[a]=-stride[a];
    if(kind==4)stride.back()=0;
    if(kind==6){for(auto& s:stride)s*=3;stride.back()=-stride.back();}
    std::vector<std::ptrdiff_t> offsets;addresses(shape,stride,0,0,offsets);require(offsets.size()==n,"independent address cardinality");
    const auto bounds=std::minmax_element(offsets.begin(),offsets.end());const auto low=*bounds.first,high=*bounds.second;
    const auto start=std::size_t(-low)+(kind==5?1:0);std::vector<unsigned char> storage(std::size_t(high-low)+sizeof(T)+(kind==5?1:0),0xa5);
    for(std::size_t i=0;i<n;++i){T value=T(int((i*13+i/7)%29)-14)/T(16);if(i%31==0)value=-T(0);std::memcpy(storage.data()+start+offsets[i],&value,sizeof(T));}
    std::vector<T> packed(n);for(std::size_t i=0;i<n;++i)std::memcpy(&packed[i],storage.data()+start+offsets[i],sizeof(T));
    const auto before=storage;const auto type=sizeof(T)==4?TvScalarType::float32:TvScalarType::float64;TvConstView input{storage.data()+start,type,shape,stride};
    std::vector<std::ptrdiff_t> cs(shape.size());std::size_t count=sizeof(T);for(std::size_t a=shape.size();a-->0;){cs[a]=std::ptrdiff_t(count);count*=shape[a];}
    TvConstView canonical{packed.data(),type,shape,cs};
    // Test the changed generic gradient against direct neighbor coordinates,
    // including negative/zero/unaligned byte strides and terminal boundaries.
    std::vector<T> gradient(n*shape.size(),T(123));tv_chambolle_gradient_into(input,{gradient.data(),type,shape});
    std::size_t logical_step=n;
    for(std::size_t a=0;a<shape.size();++a){logical_step/=shape[a];for(std::size_t i=0;i<n;++i){T expected=T(0);if((i/logical_step)%shape[a]+1<shape[a])expected=T(packed[i+logical_step]-packed[i]);require(!std::memcmp(&expected,&gradient[a*n+i],sizeof(T)),"gradient byte/address oracle differs");}}
    for(auto profile:{TvReductionProfile::numpy_1244_buffered,TvReductionProfile::numpy_254_aligned_contiguous})for(auto iterations:{std::size_t(1),std::size_t(2),std::size_t(8)}){
        Work<T> actual(shape),expected(shape);std::vector<TvDenoiseTrace> ta(iterations),te(iterations);TvDenoiseParameters params{0.5,0.,iterations,{profile,32}};
        require(!tv_denoise_ineligibility(input,actual.view,params,ta.data(),ta.size()),"certified input ineligible");
        std::feclearexcept(FE_ALL_EXCEPT);auto a=tv_denoise_finite_into(input,actual.view,params,ta.data(),ta.size());const int fa=std::fetestexcept(FE_ALL_EXCEPT);
        std::feclearexcept(FE_ALL_EXCEPT);auto e=tv_denoise_finite_into(canonical,expected.view,params,te.data(),te.size());const int fe=std::fetestexcept(FE_ALL_EXCEPT);
        require(a.iterations==e.iterations && a.input_alias==e.input_alias && fa==fe,"iteration/alias/fenv differs");
        require(same(actual.p,expected.p)&&same(actual.g,expected.g)&&same(actual.d,expected.d)&&same(actual.out,expected.out)&&same(actual.squared,expected.squared)&&same(actual.norm,expected.norm),"workspace raw bytes differ from unchanged contiguous path");
        for(std::size_t i=0;i<iterations;++i){const auto& x=ta[i];const auto& y=te[i];require(x.square_sum_bits==y.square_sum_bits&&x.norm_sum_bits==y.norm_sum_bits&&x.energy_bits==y.energy_bits&&x.difference_bits==y.difference_bits&&x.threshold_bits==y.threshold_bits&&x.energy_is_float32==y.energy_is_float32&&x.stopped==y.stopped,"per-iteration trace differs");}
    }
    require(storage==before,"read-only input changed");
    // Last logical coordinate must still be scanned before any workspace write.
    if(kind!=4){T bad=std::numeric_limits<T>::infinity();std::memcpy(storage.data()+start+offsets.back(),&bad,sizeof(T));Work<T> w(shape);const auto original=w.out;TvDenoiseParameters params{0.5,0.,2,{TvReductionProfile::numpy_254_aligned_contiguous,8192}};require(tv_denoise_ineligibility(input,w.view,params)!=nullptr,"last-coordinate nonfinite not rejected");bool rejected=false;try{tv_denoise_finite_into(input,w.view,params);}catch(const std::invalid_argument&){rejected=true;}require(rejected&&same(w.out,original),"prewrite rejection changed");}
    (void)worker;
}
}
int main(){
    try {
        using namespace skimagexx;require(get_num_threads()==1,"default workers not1");
        for(int worker:{1,4}){
#ifndef _OPENMP
            if(worker==4){bool rejected=false;try{set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}require(rejected&&get_num_threads()==1,"portable worker4 must reject");continue;}
#endif
            set_num_threads(worker);
            for(const auto& shape:std::vector<std::vector<std::size_t>>{{1},{2},{7},{8},{9},{33},{1,17},{3,17},{4,1,7},{2,1,3,9}})for(unsigned kind=0;kind<7;++kind){one<float>(shape,kind,worker);one<double>(shape,kind,worker);}
            // Force real optional partitioning with non-row-aligned starts.
            for(unsigned kind:{1u,2u,3u,4u,5u,6u})one<float>({513,515},kind,worker);
            set_num_threads(1);
        }
        std::cout<<"TV cursor independent-address/byte-trace "<<checks<<" assertions; SIMD="<<has_avx2()<<"\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
