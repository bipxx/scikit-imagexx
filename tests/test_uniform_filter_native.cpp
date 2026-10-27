#include "uniform_filter.hpp"
#include "skimagexx/adaptive.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#endif
using namespace skimagexx;
namespace {
std::size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("uniform control "+std::to_string(checks));}
template<class T>void raw(std::ofstream& journal,const T* p,std::size_t n){if(journal.is_open())journal.write(reinterpret_cast<const char*>(p),std::streamsize(n*sizeof(T)));}
std::size_t count(const std::vector<std::size_t>& shape){std::size_t n=1;for(auto v:shape)n*=v;return n;}
template<class T>constexpr PixelType type(){return std::is_same_v<T,float>?PixelType::float32:PixelType::float64;}
void old_controls(std::ofstream& journal){
    // The four original adaptive blocks from tests/test_native.cpp are retained
    // literally below; additional raw streams compare the unchanged SDK leaf.
    {
        const std::vector<std::size_t> shape{5,4},window{3,1};
        const std::vector<GaussianBorder> modes(2,GaussianBorder::reflect);
        double input[20],output[20];for(int row=0;row<5;++row)for(int col=0;col<4;++col)input[row*4+col]=row;
        adaptive_filter_into(input,output,PixelType::float64,shape,window,modes,AdaptiveMethod::mean);
        const double expected[]={1./3.,1.,2.,3.,11./3.};
        for(int row=0;row<5;++row)for(int col=0;col<4;++col)require(output[row*4+col]==expected[row]);
        adaptive_offset_into(output,output,20,PixelType::float64,PixelType::float64,.25);
        for(int row=0;row<5;++row)for(int col=0;col<4;++col)require(output[row*4+col]==expected[row]-.25);
        // Each tile contains one of all 512 binary 3x3 neighborhoods. SIMD
        // spans multiple tiles; compare every tile center with its population.
        std::vector<float> binary(3*1536),ranked(binary.size());
        for(unsigned pattern=0;pattern<512;++pattern)for(unsigned k=0;k<9;++k)binary[(k/3)*1536+pattern*3+k%3]=float((pattern>>k)&1);
        adaptive_filter_into(binary.data(),ranked.data(),PixelType::float32,{3,1536},{3,3},modes,AdaptiveMethod::median);
        for(unsigned pattern=0;pattern<512;++pattern){unsigned ones=0;for(unsigned k=0;k<9;++k)ones+=(pattern>>k)&1;require(ranked[1536+pattern*3+1]==float(ones>=5));}
        double constant[20];std::fill_n(constant,20,7.);
        adaptive_filter_into(constant,output,PixelType::float64,{4,5},{3,3},modes,AdaptiveMethod::median,0.,true);
        for(int row=0;row<4;++row)for(int col=0;col<5;++col)require(output[row*5+col]==(row==3 || col==4 ? 7. : 0.));
        raw(journal,output,20);raw(journal,ranked.data(),ranked.size());
    }
    for(auto shape:std::vector<std::vector<std::size_t>>{{17},{7,11},{3,5,7}}){
        const auto n=count(shape);std::vector<float> a(n),b(n);std::vector<double> c(n),d(n);
        for(std::size_t i=0;i<n;++i)a[i]=float(int(i*37%251)-125)/256,c[i]=double(a[i]);
        for(int mode=0;mode<5;++mode)for(std::size_t window:{1u,2u,5u,11u}){
            std::vector<std::size_t> windows(shape.size(),window);std::vector<GaussianBorder> modes(shape.size(),GaussianBorder(mode));
            adaptive_filter_into(a.data(),b.data(),PixelType::float32,shape,windows,modes,AdaptiveMethod::mean,.125);
            adaptive_filter_into(c.data(),d.data(),PixelType::float64,shape,windows,modes,AdaptiveMethod::mean,.125);
            raw(journal,b.data(),n);raw(journal,d.data(),n);
        }
        for(int mode=0;mode<5;++mode){
            std::vector<std::size_t> windows(shape.size(),3);std::vector<GaussianBorder> modes(shape.size(),GaussianBorder(mode));
            adaptive_filter_into(a.data(),b.data(),PixelType::float32,shape,windows,modes,AdaptiveMethod::median,.125);
            adaptive_filter_into(c.data(),d.data(),PixelType::float64,shape,windows,modes,AdaptiveMethod::median,.125);
            raw(journal,b.data(),n);raw(journal,d.data(),n);
        }
        adaptive_offset_into(a.data(),b.data(),n,PixelType::float32,PixelType::float32,.1);raw(journal,b.data(),n);
        adaptive_offset_into(c.data(),d.data(),n,PixelType::float64,PixelType::float64,.1);raw(journal,d.data(),n);
        adaptive_offset_into(a.data(),d.data(),n,PixelType::float32,PixelType::float64,.1);raw(journal,d.data(),n);
        adaptive_offset_into(c.data(),b.data(),n,PixelType::float64,PixelType::float32,.1);raw(journal,b.data(),n);
    }
}
#ifndef SKIMAGEXX_UNIFORM_BASELINE
// Source-derived controls: explicit reflected padded line, then SciPy's
// increasing initial taps and entering-minus-leaving recurrence. No np.sum.
std::size_t reflect(std::ptrdiff_t i,std::size_t n){while(i<0||i>=std::ptrdiff_t(n))i=i<0?-i-1:2*std::ptrdiff_t(n)-i-1;return std::size_t(i);}
template<class T>std::vector<T> literal(const std::vector<T>& input,const std::vector<std::size_t>& shape,std::size_t axis,std::size_t window){
    const auto n=shape[axis],total=input.size();std::size_t step=1;for(auto d=axis+1;d<shape.size();++d)step*=shape[d];
    std::vector<T> output(total);std::vector<double> line(n+window-1);
    for(std::size_t j=0;j<total/n;++j){const auto base=(j/step)*n*step+j%step;
        for(std::size_t k=0;k<line.size();++k)line[k]=double(input[base+reflect(std::ptrdiff_t(k)-std::ptrdiff_t(window/2),n)*step]);
        double sum=0.;for(std::size_t k=0;k<window;++k)sum+=line[k];output[base]=T(sum/double(window));
        for(std::size_t k=1;k<n;++k){sum+=line[k+window-1]-line[k-1];output[base+k*step]=T(sum/double(window));}
    }return output;
}
template<class T>void run_case(const std::vector<T>& input,const std::vector<std::size_t>& shape,std::size_t axis,std::size_t window,
                               int workers,std::ofstream& journal,const std::vector<T>* genuine=nullptr){
    const auto expected=genuine?*genuine:literal(input,shape,axis,window),saved=input;
    std::vector<T> storage(input.size()+2,T(123.5));auto* output=storage.data()+1;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    require(detail::uniform_filter1d_into(input.data(),output,type<T>(),shape,axis,window,workers));
    require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);require(std::fegetround()==FE_TONEAREST);std::feclearexcept(FE_ALL_EXCEPT);
    require(storage.front()==T(123.5)&&storage.back()==T(123.5));require(std::memcmp(input.data(),saved.data(),input.size()*sizeof(T))==0);
    for(std::size_t i=0;i<input.size();++i)require(std::memcmp(output+i,expected.data()+i,sizeof(T))==0);
    raw(journal,output,input.size());
}
template<class T>void matrix(int workers,std::ofstream& journal){
    for(auto shape:std::vector<std::vector<std::size_t>>{{1},{2},{7},{8},{9},{31},{32},{33},{7,3},{5,4},{7,5},{3,5,7}}){
        std::vector<T> input(count(shape));for(std::size_t i=0;i<input.size();++i)input[i]=T(int(i*37%251)-125)/T(256);
        for(std::size_t axis=0;axis<shape.size();++axis)for(std::size_t window:{1u,2u,5u,11u,31u})run_case(input,shape,axis,window,workers,journal);
    }
    // Size1 recurrence may cancel a small entering value; identity-copy is wrong.
    std::vector<T> cancel{T(0x1p100),T(1),T(-0x1p100),T(-0.),T(0.)};run_case(cancel,{5},0,1,workers,journal);
    for(T value:{T(0),T(-0.),T(.5),T(-.5)}){std::vector<T> input(35,value);for(auto axis:{0u,1u})run_case(input,{5,7},axis,11,workers,journal);}
}
void refusals(int workers){
    std::vector<double> input(7,.5),output(7,123.5);
    for(double bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::max()/4}){
        input[6]=bad;std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        require(!detail::uniform_filter1d_into(input.data(),output.data(),PixelType::float64,{7},0,11,workers));
        require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);for(auto v:output)require(v==123.5);
    }
    input.assign(7,.5);std::fesetround(FE_UPWARD);require(!detail::uniform_filter1d_into(input.data(),output.data(),PixelType::float64,{7},0,1,workers));require(std::fegetround()==FE_UPWARD);std::fesetround(FE_TONEAREST);
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
    const auto csr=_mm_getcsr();_mm_setcsr(csr|0x8040);require(!detail::uniform_filter1d_into(input.data(),output.data(),PixelType::float64,{7},0,1,workers));require((_mm_getcsr()&0x8040)==0x8040);_mm_setcsr(csr);
#endif
    // Normal finite input can still overflow the DP initial sum: refuse and restore.
    input.assign(7,std::numeric_limits<double>::max()/8);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    require(!detail::uniform_filter1d_into(input.data(),output.data(),PixelType::float64,{7},0,31,workers));require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    std::vector<float> tiny(35,0),small(35);tiny[17]=std::numeric_limits<float>::min();std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    require(!detail::uniform_filter1d_into(tiny.data(),small.data(),PixelType::float32,{5,7},1,31,workers));require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);std::feclearexcept(FE_ALL_EXCEPT);
    std::vector<double> parallel(2*131073,0),discarded(parallel.size());parallel[100000]=parallel[231073]=std::numeric_limits<double>::max()/8;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    require(!detail::uniform_filter1d_into(parallel.data(),discarded.data(),PixelType::float64,{2,131073},0,31,workers));
    require(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);std::feclearexcept(FE_ALL_EXCEPT);
    bool invalid=false;try{detail::uniform_filter1d_into(input.data(),output.data(),PixelType::float64,{0},0,1,workers);}catch(const std::invalid_argument&){invalid=true;}require(invalid);
    require(!detail::uniform_filter1d_into(input.data(),output.data(),PixelType::float64,{7},0,100001,workers));
}
template<class T>void read(std::ifstream& f,T& value){f.read(reinterpret_cast<char*>(&value),sizeof(value));require(bool(f));}
void saved_oracles(const char* path,int workers,std::ofstream& journal){
    // Optional retained genuine arrays; normal CTest has no ignored-path dependency.
    std::ifstream f(path,std::ios::binary);require(bool(f));std::uint32_t records;read(f,records);
    for(std::uint32_t row=0;row<records;++row){std::uint32_t bytes,rank,axis,window;std::uint64_t n;
        read(f,bytes);read(f,rank);read(f,axis);read(f,window);read(f,n);std::vector<std::size_t> shape(rank);
        for(auto& d:shape){std::uint64_t x;read(f,x);d=std::size_t(x);}require(count(shape)==n);
        if(bytes==4){std::vector<float>a(n),b(n);f.read(reinterpret_cast<char*>(a.data()),std::streamsize(n*4));f.read(reinterpret_cast<char*>(b.data()),std::streamsize(n*4));require(bool(f));run_case(a,shape,axis,window,workers,journal,&b);}
        else{require(bytes==8);std::vector<double>a(n),b(n);f.read(reinterpret_cast<char*>(a.data()),std::streamsize(n*8));f.read(reinterpret_cast<char*>(b.data()),std::streamsize(n*8));require(bool(f));run_case(a,shape,axis,window,workers,journal,&b);}
    }require(f.peek()==std::char_traits<char>::eof());
}
#endif
}
int main(int argc,char** argv){
    std::ofstream journal,old;if(argc>1){journal.open(argv[1],std::ios::binary);old.open(std::string(argv[1])+".old",std::ios::binary);require(bool(journal)&&bool(old));}
    require(get_num_threads()==1);std::vector<int> budgets{1};
#ifdef _OPENMP
    budgets.push_back(4);
#else
    bool refused=false;try{set_num_threads(4);}catch(const std::invalid_argument&){refused=true;}require(refused&&get_num_threads()==1);
#endif
    for(int workers:budgets){set_num_threads(workers);old_controls(old);
#ifndef SKIMAGEXX_UNIFORM_BASELINE
        matrix<float>(workers,journal);matrix<double>(workers,journal);
        std::vector<float> large(513*513);for(std::size_t i=0;i<large.size();++i)large[i]=float(i*37%251)/256;
        run_case(large,{513,513},0,11,workers,journal);run_case(large,{513,513},1,11,workers,journal);refusals(workers);
        if(argc>2)saved_oracles(argv[2],workers,journal);
#endif
    }set_num_threads(1);require(get_num_threads()==1);
    std::cout<<checks<<" Uniform controls PASS avx2="<<has_avx2()<<" workers="<<get_num_threads()<<"\n";
}
