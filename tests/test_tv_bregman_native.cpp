#include "skimagexx/tv_bregman.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
namespace {
using namespace skimagexx;
std::size_t checks=0,oracle_rows=0;
void require(bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);}
template<class T> TvBregmanType kind(){return sizeof(T)==4?TvBregmanType::float32:TvBregmanType::float64;}
template<class T> bool exact(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;}
template<class T> struct State {
    std::size_t r,c,d,n,padded;std::vector<T> input,out,dx,dy,bx,by;
    State(std::size_t rows,std::size_t cols,std::size_t channels):r(rows),c(cols),d(channels),n(r*c*d),padded((r+2)*(c+2)*d),input(n),out(padded),dx(padded),dy(padded),bx(padded),by(padded){}
    TvBregmanImage image(){return {input.data(),kind<T>(),r,c,d,n};}
    TvBregmanWorkspace workspace(){return {{out.data(),padded},{dx.data(),padded},{dy.data(),padded},{bx.data(),padded},{by.data(),padded}};}
};
template<class T> void corners(State<T>& a){
    const auto stride=(a.c+2)*a.d;
    for(std::size_t k=0;k<a.d;++k){a.out[k]=T(-0.0);a.out[(a.c+1)*a.d+k]=T(0.375);a.out[(a.r+1)*stride+k]=T(-0.25);a.out[(a.r+1)*stride+(a.c+1)*a.d+k]=T(0.5);}
}
template<class T> void initialized(const State<T>& a,const std::vector<T>& initial){
    const auto stride=(a.c+2)*a.d;
    require(exact(a.dx,initial)&&exact(a.dy,initial)&&exact(a.bx,initial)&&exact(a.by,initial),"aux must copy CURRENT out before reset");
    for(std::size_t y=0;y<a.r+2;++y)for(std::size_t x=0;x<a.c+2;++x)for(std::size_t k=0;k<a.d;++k){
        const auto i=y*stride+x*a.d+k;T expected;
        if((y==0 || y==a.r+1)&&(x==0 || x==a.c+1))expected=initial[i];
        else {const auto yy=y==0?1:y==a.r+1?a.r-1:y-1;const auto xx=x==0?1:x==a.c+1?a.c-1:x-1;expected=a.input[(yy*a.c+xx)*a.d+k];}
        require(std::memcmp(&a.out[i],&expected,sizeof(T))==0,"source asymmetric halo/interior/corner initialization");
    }
}
template<class T> void guards(){
    State<T> a(4,5,1);for(std::size_t i=0;i<a.n;++i)a.input[i]=T(i%7)/T(8);for(std::size_t i=0;i<a.padded;++i)a.out[i]=T(i%11)/T(8);
    auto image=a.image();auto workspace=a.workspace();TvBregmanParameters p{0.5,2,1e-6,true};std::vector<TvBregmanTrace> trace(2);
    auto reject=[&](TvBregmanImage im,TvBregmanWorkspace ws,TvBregmanParameters parameters,TvBregmanTrace* tr=nullptr,std::size_t cap=0){
        auto input=a.input,out=a.out,dx=a.dx,dy=a.dy,bx=a.bx,by=a.by;
        require(tv_bregman_ineligibility(im,ws,parameters,tr,cap)!=nullptr,"bad metadata must fail guard");
        bool caught=false;try{tv_bregman_into(im,ws,parameters,tr,cap);}catch(const std::invalid_argument&){caught=true;}require(caught,"bad metadata must throw");
        require(exact(a.input,input)&&exact(a.out,out)&&exact(a.dx,dx)&&exact(a.dy,dy)&&exact(a.bx,bx)&&exact(a.by,by),"guard failure wrote buffers");
    };
    auto im=image;im.rows=1;reject(im,workspace,p);im=image;im.cols=1;reject(im,workspace,p);im=image;im.channels=0;reject(im,workspace,p);
    im=image;im.rows=std::numeric_limits<std::size_t>::max();reject(im,workspace,p);im=image;im.elements--;reject(im,workspace,p);im=image;im.type=static_cast<TvBregmanType>(99);reject(im,workspace,p);
    auto ws=workspace;ws.dx.elements--;reject(image,ws,p);ws=workspace;ws.out.data=a.input.data();reject(image,ws,p);ws=workspace;ws.bx=ws.dx;reject(image,ws,p);
    im=image;im.data=reinterpret_cast<const char*>(a.input.data())+1;reject(im,workspace,p);
    ws=workspace;ws.dy.data=reinterpret_cast<char*>(a.dy.data())+1;reject(image,ws,p);
    reject(image,workspace,p,trace.data(),1);reject(image,workspace,p,nullptr,2);reject(image,workspace,p,reinterpret_cast<TvBregmanTrace*>(a.out.data()),2);
    reject(image,workspace,p,trace.data(),std::numeric_limits<std::size_t>::max());
    auto bad=p;bad.weight=0;reject(image,workspace,bad);bad=p;bad.weight=-1;reject(image,workspace,bad);bad=p;bad.weight=std::numeric_limits<double>::infinity();reject(image,workspace,bad);bad=p;bad.eps=-1;reject(image,workspace,bad);bad=p;bad.eps=std::numeric_limits<double>::quiet_NaN();reject(image,workspace,bad);
    a.input[3]=std::numeric_limits<T>::quiet_NaN();reject(image,workspace,p);a.input[3]=T(0);
    a.out[3]=std::numeric_limits<T>::infinity();reject(image,workspace,p);a.out[3]=T(0);
    std::fenv_t original_environment;std::fegetenv(&original_environment);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    const int initial_flags=std::fetestexcept(FE_ALL_EXCEPT);
    require(tv_bregman_ineligibility(image,workspace,p)==nullptr,"valid guard accepted with prior FP flags");require(std::fetestexcept(FE_ALL_EXCEPT)==initial_flags,"successful guard changed FP flags");
    bad=p;bad.weight=std::numeric_limits<double>::max();require(tv_bregman_ineligibility(image,workspace,bad)!=nullptr,"overflow coefficient guard");require(std::fetestexcept(FE_ALL_EXCEPT)==initial_flags,"rejected guard changed FP flags");std::fesetenv(&original_environment);
    const int rounding=std::fegetround();std::fesetround(FE_DOWNWARD);reject(image,workspace,p);std::fesetround(rounding);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto csr=_mm_getcsr();_mm_setcsr(csr|(1u<<15));reject(image,workspace,p);_mm_setcsr(csr);
#endif
}
template<class T> void controls(){
    for(auto channels:{std::size_t{1},std::size_t{3},std::size_t{9}})for(int maximum:{0,-1,std::numeric_limits<int>::min()}){
        State<T> a(2,3,channels);for(std::size_t i=0;i<a.n;++i)a.input[i]=T(int(i%9)-4)/T(8);for(std::size_t i=0;i<a.padded;++i)a.out[i]=T(int(i%11)-5)/T(8);corners(a);auto initial=a.out;
        std::fill(a.dx.begin(),a.dx.end(),std::numeric_limits<T>::quiet_NaN());std::fill(a.dy.begin(),a.dy.end(),T(999));std::fill(a.bx.begin(),a.bx.end(),T(-999));std::fill(a.by.begin(),a.by.end(),T(77));
        auto result=tv_bregman_into(a.image(),a.workspace(),{0.5,maximum,1e-6,true});require(result.iterations==0 && result.rmse==std::numeric_limits<double>::max(),"no-loop result");initialized(a,initial);
    }
    for(auto shape:std::vector<std::array<std::size_t,3>>{{2,2,1},{3,5,3},{4,7,9},{15,17,1},{16,18,1},{17,19,3},{513,519,1}}){
        State<T> a(shape[0],shape[1],shape[2]);std::fill(a.input.begin(),a.input.end(),T(.25));corners(a);auto initial=a.out;std::vector<TvBregmanTrace> trace(3);
        auto result=tv_bregman_into(a.image(),a.workspace(),{.5,3,1e-6,true},trace.data(),trace.size());require(result.iterations==1 && result.rmse==0,"constant first-sweep fixed point");require(trace[0].rmse_bits==0 && trace[0].stopped,"constant RMSE trace");
        for(std::size_t y=0;y<a.r;++y)for(std::size_t x=0;x<a.c;++x)for(std::size_t k=0;k<a.d;++k)require(a.out[((y+1)*(a.c+2)+x+1)*a.d+k]==T(.25),"constant pixels");
        for(auto i:{std::size_t{0},(a.c+1)*a.d,(a.r+1)*(a.c+2)*a.d,((a.r+2)*(a.c+2)-1)*a.d})for(std::size_t k=0;k<a.d;++k)require(std::memcmp(&a.out[i+k],&initial[i+k],sizeof(T))==0,"corners unchanged");
        require(result.initialization_workers>=1 && result.initialization_workers<=get_num_threads(),"actual copy team reported");
#ifdef _OPENMP
        if(a.padded>=256*1024 && get_num_threads()==4)require(result.initialization_workers==4,"large initialization actual parallel copy team");
#endif
    }
    State<T> reused(4,5,1),fresh(4,5,1);for(std::size_t i=0;i<reused.n;++i)reused.input[i]=T(int(i%13)-6)/T(8);
    tv_bregman_into(reused.image(),reused.workspace(),{.5,2,1e-6,true});auto first_channel=reused.out;
    for(std::size_t i=0;i<reused.n;++i)reused.input[i]=fresh.input[i]=T(int((i*3)%11)-5)/T(8);
    tv_bregman_into(reused.image(),reused.workspace(),{.5,0,1e-6,true});initialized(reused,first_channel);reused.out=first_channel;
    tv_bregman_into(reused.image(),reused.workspace(),{.5,2,1e-6,true});tv_bregman_into(fresh.image(),fresh.workspace(),{.5,2,1e-6,true});require(!exact(reused.out,fresh.out),"channel_out reuse must not silently reset aux to zero");
    require(!tv_bregman_parallel_sweep,"no false parallel sweep capability");guards<T>();
}
std::uint64_t read64(std::istream& stream){std::uint64_t v;stream.read(reinterpret_cast<char*>(&v),8);if(!stream)throw std::runtime_error("truncated oracle");return v;}
double read_double(std::istream& stream){auto bits=read64(stream);double v;std::memcpy(&v,&bits,8);return v;}
template<class T> void read_array(std::istream& stream,std::vector<T>& values){stream.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()*sizeof(T)));if(!stream)throw std::runtime_error("truncated oracle array");}
template<class T> void oracle_row(std::istream& stream,const std::string& name){
    const auto r=read64(stream),c=read64(stream),d=read64(stream);require(r>=4 && r<=64 && c>=4 && c<=64 && d>=1 && d<=3,"reference shape safety");
    const auto maximum=read64(stream);const double weight=read_double(stream),eps=read_double(stream);const auto isotropic=read64(stream);require(maximum>=1 && maximum<=8 && isotropic<=1,"reference parameters");
    State<T> a(r,c,d);read_array(stream,a.input);read_array(stream,a.out);std::vector<T> expected(a.padded);read_array(stream,expected);auto input=a.input;std::vector<TvBregmanTrace> trace(maximum);
    const auto result=tv_bregman_into(a.image(),a.workspace(),{weight,int(maximum),eps,bool(isotropic)},trace.data(),trace.size());
    if(!exact(a.out,expected)){std::cerr<<"original word mismatch: "<<name<<" width="<<sizeof(T)<<" workers="<<get_num_threads()<<'\n';throw std::runtime_error("original padded output differs");}
    require(result.iterations>=1 && result.iterations<=maximum,"oracle iteration count");require(exact(a.input,input),"oracle input mutation");require(trace[result.iterations-1].stopped==(result.rmse<=eps),"oracle stop trace");++oracle_rows;
}
void oracle(const char* filename){
    std::ifstream stream(filename,std::ios::binary);require(bool(stream),"oracle missing");require(read64(stream)==0x315247425654ull,"oracle magic");const auto n=read64(stream);require(n>=112 && n<1024,"eligible original oracle rows only");
    for(std::uint64_t i=0;i<n;++i){const auto length=read64(stream);require(length<2048,"oracle name bound");std::string name(length,' ');stream.read(&name[0],std::streamsize(length));const auto width=read64(stream);if(width==4)oracle_row<float>(stream,name);else if(width==8)oracle_row<double>(stream,name);else throw std::runtime_error("oracle width");}
    require(stream.peek()==std::char_traits<char>::eof(),"oracle trailing bytes");
}
} // namespace
int main(int argc,char** argv){
    try{
        require(skimagexx::get_num_threads()==1,"default shared global workers");
        for(int workers:{1,4}){
#ifndef _OPENMP
            if(workers==4){bool rejected=false;try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}require(rejected && skimagexx::get_num_threads()==1,"portable worker4 rejects");continue;}
#endif
            skimagexx::set_num_threads(workers);controls<float>();controls<double>();if(argc==2)oracle(argv[1]);
        }
        skimagexx::set_num_threads(1);std::cout<<"TV Bregman "<<checks<<" checks, original rows="<<oracle_rows<<" parallel_sweep=false SIMD="<<skimagexx::has_avx2()<<" passed\n";
    }catch(const std::exception& error){std::cerr<<"TV Bregman FAILED after "<<checks<<": "<<error.what()<<'\n';return 1;}
}
