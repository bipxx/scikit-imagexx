#include "skimagexx/tv_denoise.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::size_t checks=0;
void require(bool okay,const char* message) {if(!okay)throw std::runtime_error(message);++checks;}
template<class T> std::uint64_t raw(T value) {std::uint64_t result=0;std::memcpy(&result,&value,sizeof(T));return result;}
template<class T> struct Buffers {
    std::vector<T> p,g,d,out,squared,norm;
    skimagexx::TvDenoiseWorkspace workspace;
    Buffers(const std::vector<std::size_t>& shape) {
        using namespace skimagexx;
        std::size_t n=1;for(auto v:shape)n*=v;
        std::vector<std::ptrdiff_t> strides(shape.size());std::size_t step=sizeof(T);
        for(std::size_t a=shape.size();a-->0;){strides[a]=std::ptrdiff_t(step);step*=shape[a];}
        const auto type=sizeof(T)==4?TvScalarType::float32:TvScalarType::float64;
        p.resize(n*shape.size());g.resize(p.size());d.resize(n);
        out.assign(n,T(99));squared.assign(n,T(98));norm.assign(n,T(97));
        workspace={{p.data(),type,shape},{g.data(),type,shape},
            {d.data(),type,shape,strides},{out.data(),type,shape,strides},
            {squared.data(),type,shape,strides},{norm.data(),type,shape,strides},
            {p.data(),type,shape},{g.data(),type,shape},
            {out.data(),type,shape,strides},{norm.data(),type,shape,strides}};
    }
};
template<class T> void controls() {
    using namespace skimagexx;
    std::vector<std::size_t> shape{3,5};Buffers<T> arrays(shape);
    std::vector<T> input(15);for(std::size_t i=0;i<input.size();++i)input[i]=T(i)/T(8);
    const auto type=sizeof(T)==4?TvScalarType::float32:TvScalarType::float64;
    TvConstView image{input.data(),type,shape,{std::ptrdiff_t(5*sizeof(T)),std::ptrdiff_t(sizeof(T))}};
    TvDenoiseParameters parameters{0.5,0.,1,{TvReductionProfile::numpy_254_aligned_contiguous,8192}};
    require(!tv_denoise_ineligibility(image,arrays.workspace,parameters),"valid input rejected");
    auto reject=[&](TvConstView view,TvDenoiseWorkspace workspace,TvDenoiseParameters options){
        const auto p=arrays.p,g=arrays.g,d=arrays.d,out=arrays.out,squared=arrays.squared,norm=arrays.norm;
        require(tv_denoise_ineligibility(view,workspace,options)!=nullptr,"bad guard accepted");
        bool rejected=false;
        try{tv_denoise_finite_into(view,workspace,options);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"bad entry accepted");
        auto unchanged=[](const std::vector<T>& before,const std::vector<T>& after){
            return before.size()==after.size() && std::memcmp(before.data(),after.data(),before.size()*sizeof(T))==0;
        };
        require(unchanged(p,arrays.p)&&unchanged(g,arrays.g)&&unchanged(d,arrays.d)&&unchanged(out,arrays.out)&&unchanged(squared,arrays.squared)&&unchanged(norm,arrays.norm),"rejection wrote workspace");
    };
    auto bad=parameters;bad.weight=0;reject(image,arrays.workspace,bad);
    bad=parameters;bad.weight=std::numeric_limits<double>::infinity();reject(image,arrays.workspace,bad);
    bad=parameters;bad.eps=-1;reject(image,arrays.workspace,bad);
    bad=parameters;bad.eps=std::numeric_limits<double>::quiet_NaN();reject(image,arrays.workspace,bad);
    bad=parameters;bad.max_iterations=0;reject(image,arrays.workspace,bad);
    bad=parameters;bad.max_iterations=20001;reject(image,arrays.workspace,bad);
    bad=parameters;bad.reduction.buffer_elements=0;reject(image,arrays.workspace,bad);
    bad=parameters;bad.reduction.profile=static_cast<TvReductionProfile>(999);reject(image,arrays.workspace,bad);
    auto wrong_input=image;wrong_input.shape.clear();wrong_input.strides.clear();reject(wrong_input,arrays.workspace,parameters);
    wrong_input=image;wrong_input.shape[0]=0;reject(wrong_input,arrays.workspace,parameters);
    wrong_input=image;wrong_input.strides.pop_back();reject(wrong_input,arrays.workspace,parameters);
    wrong_input=image;wrong_input.data=arrays.out.data();reject(wrong_input,arrays.workspace,parameters);
    auto workspace=arrays.workspace;workspace.out.data=arrays.d.data();workspace.out_read.data=arrays.d.data();reject(image,workspace,parameters);
    workspace=arrays.workspace;workspace.norm_read.data=arrays.squared.data();reject(image,workspace,parameters);
    workspace=arrays.workspace;workspace.norm.type=sizeof(T)==4?TvScalarType::float64:TvScalarType::float32;reject(image,workspace,parameters);
    workspace=arrays.workspace;workspace.out.strides[0]=sizeof(T);reject(image,workspace,parameters);
    workspace=arrays.workspace;workspace.p_read.shape[0]=1;reject(image,workspace,parameters);
    arrays.p[0]=T(1);reject(image,arrays.workspace,parameters);arrays.p[0]=0;
    arrays.p[0]=-T(0);reject(image,arrays.workspace,parameters);arrays.p[0]=0;
    arrays.g[0]=T(1);reject(image,arrays.workspace,parameters);arrays.g[0]=0;
    arrays.d[0]=T(1);reject(image,arrays.workspace,parameters);arrays.d[0]=0;
    input[0]=std::numeric_limits<T>::quiet_NaN();reject(image,arrays.workspace,parameters);input[0]=0;
    input[0]=T(2)*T(1048576);reject(image,arrays.workspace,parameters);input[0]=0;
    const int rounding=std::fegetround();std::fesetround(FE_DOWNWARD);reject(image,arrays.workspace,parameters);std::fesetround(rounding);
    TvDenoiseTrace trace{};
    require(tv_denoise_ineligibility(image,arrays.workspace,parameters,&trace,0)!=nullptr,"short trace accepted");
    require(tv_denoise_ineligibility(image,arrays.workspace,parameters,reinterpret_cast<TvDenoiseTrace*>(arrays.out.data()),1)!=nullptr,"overlapping trace accepted");
    const auto result=tv_denoise_finite_into(image,arrays.workspace,parameters,&trace,1);
    require(result.iterations==1&&result.input_alias,"one-body return semantics");
    for(T v:arrays.out)require(v==T(99),"one-body writes output rather than returning input");
    require(arrays.p[0]!=T(0),"one-body omitted dual update");
    Buffers<T> constant(shape);std::fill(input.begin(),input.end(),T(0.25));parameters.max_iterations=5;parameters.eps=0.1;
    std::vector<TvDenoiseTrace> traces(5);
    const auto full=tv_denoise_finite_into(image,constant.workspace,parameters,traces.data(),traces.size());
    require(full.iterations==5&&!full.input_alias,"zero-energy strict stopping must not stop");
    for(const auto& t:traces)require(!t.stopped&&t.energy_bits==0,"zero-energy trace");
}
std::uint64_t read64(std::istream& stream) {
    std::uint64_t result=0;stream.read(reinterpret_cast<char*>(&result),8);
    if(!stream)throw std::runtime_error("truncated finite-loop oracle");return result;
}
double read_double(std::istream& stream){auto bits=read64(stream);double result;std::memcpy(&result,&bits,8);return result;}
template<class T> void exact(const std::vector<T>& a,const std::vector<T>& b,const char* reason) {
    require(a.size()==b.size() && (!a.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0),reason);
}
template<class T> void row(std::istream& stream,std::uint64_t index) {
    using namespace skimagexx;
    const auto profile=read64(stream),rank=read64(stream),n=read64(stream);
    const auto weight=read_double(stream),eps=read_double(stream);
    const auto maxiter=read64(stream),buffer=read64(stream),extent=read64(stream),start=read64(stream);
    const auto iterations=read64(stream),alias=read64(stream);
    std::vector<std::size_t> shape(rank);std::vector<std::ptrdiff_t> strides(rank);
    for(auto& v:shape)v=std::size_t(read64(stream));
    for(auto& v:strides){auto word=read64(stream);std::int64_t signed_word;std::memcpy(&signed_word,&word,8);v=std::ptrdiff_t(signed_word);}
    std::vector<unsigned char> storage(extent);stream.read(reinterpret_cast<char*>(storage.data()),std::streamsize(extent));
    std::vector<T> expected(n),p(n*rank),g(n*rank),d(n),norm(n);
    for(auto* values:{&expected,&p,&g,&d,&norm})stream.read(reinterpret_cast<char*>(values->data()),std::streamsize(values->size()*sizeof(T)));
    std::vector<TvDenoiseTrace> expected_trace(iterations);
    for(auto& t:expected_trace){t.square_sum_bits=read64(stream);t.norm_sum_bits=read64(stream);t.energy_bits=read64(stream);t.difference_bits=read64(stream);t.threshold_bits=read64(stream);t.energy_is_float32=read64(stream)!=0;t.stopped=read64(stream)!=0;}
    if(!stream)throw std::runtime_error("truncated finite-loop arrays");
    Buffers<T> arrays(shape);std::vector<TvDenoiseTrace> trace(maxiter);
    const auto type=sizeof(T)==4?TvScalarType::float32:TvScalarType::float64;
    TvConstView input{storage.data()+start,type,shape,strides};const auto unchanged=storage;
    TvDenoiseParameters parameters{weight,eps,std::size_t(maxiter),{profile==0?TvReductionProfile::numpy_1244_buffered:TvReductionProfile::numpy_254_aligned_contiguous,std::size_t(buffer)}};
    try {
        require(!tv_denoise_ineligibility(input,arrays.workspace,parameters,trace.data(),trace.size()),"reference fixture ineligible");
        const auto result=tv_denoise_finite_into(input,arrays.workspace,parameters,trace.data(),trace.size());
        require(result.iterations==iterations&&result.input_alias==bool(alias),"iteration/alias mismatch");
        if(!result.input_alias)exact(arrays.out,expected,"output nonidentical");
        exact(arrays.p,p,"final p mismatch");exact(arrays.g,g,"final g mismatch");exact(arrays.d,d,"final d mismatch");exact(arrays.norm,norm,"final denominator mismatch");
        for(std::size_t i=0;i<iterations;++i) {
            const auto& a=trace[i];const auto& b=expected_trace[i];
            if(a.square_sum_bits!=b.square_sum_bits||a.norm_sum_bits!=b.norm_sum_bits||a.energy_bits!=b.energy_bits||a.difference_bits!=b.difference_bits||a.threshold_bits!=b.threshold_bits||a.energy_is_float32!=b.energy_is_float32||a.stopped!=b.stopped) {
                std::cerr<<"trace row="<<index<<" iteration="<<i<<" sum "<<a.square_sum_bits<<'/'<<b.square_sum_bits<<" norm "<<a.norm_sum_bits<<'/'<<b.norm_sum_bits<<" E "<<a.energy_bits<<'/'<<b.energy_bits<<" diff "<<a.difference_bits<<'/'<<b.difference_bits<<" eps "<<a.threshold_bits<<'/'<<b.threshold_bits<<'\n';
                throw std::runtime_error("raw energy trace mismatch");
            }
            ++checks;
        }
        require(storage==unchanged,"input storage mutated");
    }catch(const std::exception&){std::cerr<<"finite reference row="<<index<<" dtype="<<sizeof(T)<<" profile="<<profile<<" rank="<<rank<<" n="<<n<<'\n';throw;}
}
void oracle(const char* filename) {
    std::ifstream stream(filename,std::ios::binary);if(!stream)throw std::runtime_error("missing oracle file");
    require(read64(stream)==0x31564f4c5654ull,"finite oracle magic");const auto count=read64(stream);
    for(std::uint64_t i=0;i<count;++i){auto width=read64(stream);if(width==4)row<float>(stream,i);else if(width==8)row<double>(stream,i);else throw std::runtime_error("bad scalar width");}
    require(stream.peek()==std::char_traits<char>::eof(),"finite oracle trailing data");
}
} // namespace
int main(int argc,char** argv) {
    try {
        require(skimagexx::get_num_threads()==1,"default global worker count");
        for(int workers:{1,4}) {
#ifndef _OPENMP
            if(workers==4){bool rejected=false;try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"portable worker4 must reject");continue;}
#endif
            skimagexx::set_num_threads(workers);controls<float>();controls<double>();if(argc==2)oracle(argv[1]);
        }
        skimagexx::set_num_threads(1);std::cout<<"TV finite loop "<<checks<<" checks passed\n";
    }catch(const std::exception& error){std::cerr<<"TV finite FAILED after "<<checks<<": "<<error.what()<<'\n';return 1;}
}
