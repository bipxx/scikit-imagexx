#include "coords_internal.hpp"
#include "skimagexx/image.hpp"
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef NDEBUG
#error Native controls require assertions
#endif
namespace {
unsigned checks=0,calls=0;std::ofstream journal;
void require(bool value){++checks;if(!value)throw std::runtime_error("coordinate moments native control failed");}
template<class T>unsigned call(const std::vector<T>&powers,std::size_t N,std::size_t D,std::size_t K,int workers,std::vector<T>&out){
    require(skimagexx::get_num_threads()==1);skimagexx::set_num_threads(workers);
    std::size_t count=1;for(std::size_t i=0;i<D;++i)count*=K;out.assign(count+2,T(123));
    std::fenv_t saved;require(std::feholdexcept(&saved)==0);require(std::feraiseexcept(FE_DIVBYZERO)==0);
    const auto prior=std::fetestexcept(FE_ALL_EXCEPT);
    const auto flags=skimagexx::detail::moments_coords_reduce(N?powers.data():nullptr,N,D,K,out.data()+1,workers);++calls;
    require(std::fetestexcept(FE_ALL_EXCEPT)==prior);require(std::fesetenv(&saved)==0);
    require(out.front()==T(123)&&out.back()==T(123));skimagexx::set_num_threads(1);
    if(journal){for(auto word:{std::size_t(sizeof(T)),N,D,K,std::size_t(workers),std::size_t(flags)})journal.write(reinterpret_cast<const char*>(&word),sizeof(word));journal.write(reinterpret_cast<const char*>(out.data()),out.size()*sizeof(T));}
    return flags;
}
template<class T>void run(int workers){
    std::vector<T>out;
    require(call<T>({1,2,1,3,1,-1,1,4},2,2,2,workers,out)==0);
    const T expected2[]={2,7,1,2};for(unsigned i=0;i<4;++i)require(out[i+1]==expected2[i]);
    require(call<T>({1,2,1,3,1,4,1,-1,1,4,1,T(.5)},2,3,2,workers,out)==0);
    const T expected3[]={2,T(4.5),7,14,1,T(7.5),2,22};for(unsigned i=0;i<8;++i)require(out[i+1]==expected3[i]);
    require(call<T>({1,1,1,1,1,2,4,8,1,-1,1,-1},3,1,4,workers,out)==0);
    const T line[]={3,2,6,8};for(unsigned i=0;i<4;++i)require(out[i+1]==line[i]);
    // Last point must survive cancellation; no horizontal reassociation.
    require(call<T>({1,T(1e20),1,T(-1e20),1,3},3,1,2,workers,out)==0);require(out[2]==3);
    for(std::size_t K:{3,5}){
        std::vector<T>p(2*K);for(std::size_t k=0;k<K;++k){p[k]=std::ldexp(T(1),-int(k));p[K+k]=std::ldexp(T(1),int(k));}
        require(call(p,1,2,K,workers,out)==0);
        for(std::size_t i=0;i<K;++i)for(std::size_t j=0;j<K;++j)require(out[1+i*K+j]==std::ldexp(T(1),int(j)-int(i)));
    }
    constexpr std::size_t N=8193;std::vector<T>large(N*3*4);
    for(std::size_t point=0;point<N;++point)for(std::size_t k=0;k<4;++k){large[(point*3)*4+k]=(point%2&&k%2)?T(-1):T(1);large[(point*3+1)*4+k]=std::ldexp(T(1),int(k));large[(point*3+2)*4+k]=std::ldexp(T(1),-int(k));}
    require(call(large,N,3,4,workers,out)==0);
    for(std::size_t a=0;a<4;++a)for(std::size_t b=0;b<4;++b)for(std::size_t c=0;c<4;++c)require(out[1+(a*4+b)*4+c]==T(a%2?1:N)*std::ldexp(T(1),int(b)-int(c)));
    // Above the worker threshold, K5 gives unequal worker blocks and tails.
    constexpr std::size_t M=1025,K=5;std::vector<T>awkward(M*3*K);
    for(std::size_t point=0;point<M;++point)for(std::size_t k=0;k<K;++k){awkward[(point*3)*K+k]=(point%2&&k%2)?T(-1):T(1);awkward[(point*3+1)*K+k]=std::ldexp(T(1),int(k));awkward[(point*3+2)*K+k]=std::ldexp(T(1),-int(k));}
    require(call(awkward,M,3,K,workers,out)==0);
    for(std::size_t a=0;a<K;++a)for(std::size_t b=0;b<K;++b)for(std::size_t c=0;c<K;++c)require(out[1+(a*K+b)*K+c]==T(a%2?1:M)*std::ldexp(T(1),int(b)-int(c)));
    require(call<T>({},0,3,3,workers,out)==0);for(std::size_t i=1;i+1<out.size();++i)require(out[i]==0&&!std::signbit(out[i]));
    const auto max=std::numeric_limits<T>::max();auto flags=call<T>({1,max,1,max},2,1,2,workers,out);require((flags&2)&&(flags&16)&&std::isinf(out[2]));
    flags=call<T>({1,std::numeric_limits<T>::denorm_min(),1,T(.5)},1,2,2,workers,out);require((flags&4)&&out[4]==0);
    flags=call<T>({1,std::numeric_limits<T>::infinity(),1,0},1,2,2,workers,out);require((flags&8)&&(flags&16)&&std::isnan(out[4]));
    flags=call<T>({1,std::numeric_limits<T>::quiet_NaN()},1,1,2,workers,out);require((flags&16)&&std::isnan(out[2]));
}
}
int main(int argc,char**argv){
    const bool vectorized=skimagexx::has_avx2();
    require(std::getenv("SKIMAGEXX_DISABLE_SIMD")==nullptr || !vectorized);
    require(argc==2);journal.open(argv[1],std::ios::binary);require(bool(journal));
    std::vector<int> budgets{1};int maximum=1;
#ifdef _OPENMP
    maximum=omp_get_max_threads();if(maximum>=4)budgets.push_back(4);
#endif
    for(int workers:budgets){run<float>(workers);run<double>(workers);}
#ifndef _OPENMP
    bool refused=false;
    try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){refused=true;}
    require(refused && skimagexx::get_num_threads()==1);
    float powers[2]={1,1},output[2];refused=false;
    try{skimagexx::detail::moments_coords_reduce(powers,1,1,2,output,4);}
    catch(const std::invalid_argument&){refused=true;}
    require(refused);
#endif
    journal.close();require(skimagexx::get_num_threads()==1);
    std::cout<<checks<<" controls / "<<calls<<" calls PASS; actual_avx2="<<vectorized<<" omp_max_threads="<<maximum<<" worker4_exercised="<<(budgets.size()==2)<<"\n";
}
