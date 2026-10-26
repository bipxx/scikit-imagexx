#include "moments_internal.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef NDEBUG
#error Native controls require assertions
#endif
namespace {
std::size_t controls=0;
void check(bool value){++controls;assert(value);}
using skimagexx::detail::moments_project;
template<class T> void literal() {
    const T a[]={T(1),T(2),T(3),T(4),T(5),T(6)};
    const T p[]={T(1),T(0),T(1),T(1),T(1),T(2)};
    T out[8];std::fill(out,out+8,T(-123));
    check(moments_project(a,{2,3},p,2,1,out+1,1)==0);
    const T expected[]={T(6),T(8),T(15),T(17)};
    for(int i=0;i<4;++i)check(out[i+1]==expected[i]);
    check(out[0]==T(-123)&&out[5]==T(-123));
    check(moments_project(a,{2,3},p,2,0,out+1,1)==0);
    const T axis0[]={T(5),T(7),T(9),T(4),T(5),T(6)};
    for(int i=0;i<6;++i)check(out[i+1]==axis0[i]);
    check(out[0]==T(-123)&&out[7]==T(-123));
    const T cube[]={T(1),T(2),T(3),T(4),T(5),T(6),T(7),T(8)};
    T result[8];check(moments_project(cube,{2,2,2},p,2,1,result,1)==0);
    const T expected3[]={T(4),T(6),T(3),T(4),T(12),T(14),T(7),T(8)};
    for(int i=0;i<8;++i)check(result[i]==expected3[i]);
    // Independent tail lengths cross both vector widths, with literal sums.
    for(std::size_t width=1;width<=17;++width) {
        std::vector<T> data(2*width,T(2)),buffer(width+2,T(-91));
        const T weights[]={T(1),T(3)};
        check(moments_project(data.data(),{2,width},weights,1,0,buffer.data()+1,1)==0);
        for(std::size_t i=0;i<width;++i)check(buffer[i+1]==T(8));
        check(buffer.front()==T(-91)&&buffer.back()==T(-91));
    }
    T zero[5]={T(-4),T(-4),T(-4),T(-4),T(-4)};
    check(moments_project(static_cast<const T*>(nullptr),{0,3},static_cast<const T*>(nullptr),1,0,zero+1,1)==0);
    for(int i=1;i<=3;++i)check(zero[i]==T(0));
    check(zero[0]==T(-4)&&zero[4]==T(-4));
    check(moments_project(static_cast<const T*>(nullptr),{2,3},static_cast<const T*>(nullptr),0,0,static_cast<T*>(nullptr),1)==0);
}
template<class R> void complex_literal() {
    using T=std::complex<R>;const T a[]={T(1,2),T(3,-1)},p[]={T(2,1),T(1,-2)};T out;
    check(moments_project(a,{2},p,1,0,&out,1)==0);check(out==T(1,-2));
    const T b[]={T(1,2),T(3,-1),T(1,2),T(3,-1)};T result[2];
    check(moments_project(b,{2,2},p,1,1,result,1)==0);
    check(result[0]==T(1,-2)&&result[1]==T(1,-2));
    // Nine non-real outputs exercise full vector batches and a scalar tail.
    std::vector<T> packed(18);for(std::size_t i=0;i<9;++i){packed[2*i]=a[0];packed[2*i+1]=a[1];}
    T vector_result[11];std::fill(vector_result,vector_result+11,T(-91));
    check(moments_project(packed.data(),{9,2},p,1,1,vector_result+1,1)==0);
    for(std::size_t i=1;i<=9;++i)check(vector_result[i]==T(1,-2));
    check(vector_result[0]==T(-91)&&vector_result[10]==T(-91));
}
template<class T> void floating() {
    T out;const T a[]={std::numeric_limits<T>::max(),std::numeric_limits<T>::max()},p[]={T(2),T(2)};
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto saved=std::fetestexcept(FE_ALL_EXCEPT);
    const auto flags=moments_project(a,{2},p,1,0,&out,1);
    check(std::isinf(out)&&flags==2);check(std::fetestexcept(FE_ALL_EXCEPT)==saved);std::feclearexcept(FE_ALL_EXCEPT);
    const T inf[]={std::numeric_limits<T>::infinity()},zero[]={T(0)};
    check((moments_project(inf,{1},zero,1,0,&out,1)&8)!=0);check(std::isnan(out));
    const T small[]={std::numeric_limits<T>::min()},half[]={T(.1)};
    check((moments_project(small,{1},half,1,0,&out,1)&4)!=0);check(out>0&&out<std::numeric_limits<T>::min());
    std::feclearexcept(FE_ALL_EXCEPT);
}
template<class T> void contiguous_fast_paths() {
    // Contiguous powers, K17 crosses a vector boundary between outer rows.
    const T a[]={T(1),T(2),T(3),T(4),T(5),T(6)};
    T powers[51],out[36];std::fill(out,out+36,T(-71));
    for(std::size_t j=0;j<3;++j)for(std::size_t k=0;k<17;++k)powers[j*17+k]=T((j+1)*(k+1));
    check(moments_project(a,{2,3},powers,17,1,out+1,1)==0);
    for(std::size_t k=0;k<17;++k){check(out[k+1]==T(14*(k+1)));check(out[k+18]==T(32*(k+1)));}
    check(out[0]==T(-71)&&out[35]==T(-71));
    // The same two vector layouts must retain exact FP status and caller flags.
    for(bool contiguous_powers:{false,true}) {
        const std::vector<std::size_t> shape=contiguous_powers?std::vector<std::size_t>{2}:std::vector<std::size_t>{2,17};
        std::vector<T> values(contiguous_powers?2:34,std::numeric_limits<T>::max());
        std::vector<T> weights(contiguous_powers?34:2,T(2)),answer(19,T(-71));
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto saved=std::fetestexcept(FE_ALL_EXCEPT);
        check(moments_project(values.data(),shape,weights.data(),contiguous_powers?17:1,0,answer.data()+1,1)==2);
        check(std::fetestexcept(FE_ALL_EXCEPT)==saved);
        for(std::size_t k=1;k<=17;++k)check(std::isinf(answer[k]));
        check(answer.front()==T(-71)&&answer.back()==T(-71));
        std::feclearexcept(FE_ALL_EXCEPT);
        std::fill(values.begin(),values.end(),std::numeric_limits<T>::infinity());std::fill(weights.begin(),weights.end(),T(0));
        check(moments_project(values.data(),shape,weights.data(),contiguous_powers?17:1,0,answer.data()+1,1)==8);
        for(std::size_t k=1;k<=17;++k)check(std::isnan(answer[k]));
        check(answer.front()==T(-71)&&answer.back()==T(-71));
        check(moments_project(static_cast<const T*>(nullptr),contiguous_powers?std::vector<std::size_t>{0}:std::vector<std::size_t>{0,17},static_cast<const T*>(nullptr),contiguous_powers?17:1,0,answer.data()+1,1)==0);
        for(std::size_t k=1;k<=17;++k)check(answer[k]==T(0)&&!std::signbit(answer[k]));
    }
}
template<class T> void fused_powers() {
    constexpr std::size_t inner=17;T image[6*inner];
    for(std::size_t outer=0;outer<2;++outer)for(std::size_t j=0;j<3;++j)
        for(std::size_t x=0;x<inner;++x)image[(outer*3+j)*inner+x]=T(outer*3+j+1);
    for(std::size_t K:{std::size_t(2),std::size_t(4)}) {
        const T p4[]={1,0,0,0,1,1,1,1,1,2,4,8};const T p2[]={1,0,1,1,1,2};
        const T* powers=K==2?p2:p4;std::vector<T> answer(2*K*inner+2,T(-73));
        check(moments_project(image,{2,3,inner},powers,K,1,answer.data()+1,1)==0);
        const T expected[2][4]={{6,8,14,26},{15,17,29,53}};
        for(std::size_t outer=0;outer<2;++outer)for(std::size_t k=0;k<K;++k)
            for(std::size_t x=0;x<inner;++x)check(answer[1+(outer*K+k)*inner+x]==expected[outer][k]);
        check(answer.front()==T(-73)&&answer.back()==T(-73));
        std::fill(answer.begin(),answer.end(),T(-73));
        check(moments_project(static_cast<const T*>(nullptr),{2,0,inner},static_cast<const T*>(nullptr),K,1,answer.data()+1,1)==0);
        for(std::size_t i=1;i+1<answer.size();++i)check(answer[i]==T(0)&&!std::signbit(answer[i]));
        check(answer.front()==T(-73)&&answer.back()==T(-73));
        std::vector<T> values(2*inner,std::numeric_limits<T>::max()),weights(2*K,T(2));
        check(moments_project(values.data(),{2,inner},weights.data(),K,0,answer.data()+1,1)==2);
        for(std::size_t i=1;i<=K*inner;++i)check(std::isinf(answer[i]));
        std::fill(values.begin(),values.end(),std::numeric_limits<T>::infinity());std::fill(weights.begin(),weights.end(),T(0));
        check(moments_project(values.data(),{2,inner},weights.data(),K,0,answer.data()+1,1)==8);
        for(std::size_t i=1;i<=K*inner;++i)check(std::isnan(answer[i]));
        check(answer.front()==T(-73)&&answer.back()==T(-73));
    }
}
void fused_worker_blocks(int workers,bool complete) {
    const std::size_t outer=complete?8:1,inner=complete?4097:65537;
    std::vector<double> image(outer*3*inner),answer(outer*4*inner+2,-81);
    for(std::size_t o=0;o<outer;++o)for(std::size_t j=0;j<3;++j)
        std::fill(image.begin()+(o*3+j)*inner,image.begin()+(o*3+j+1)*inner,double(j+1));
    const double powers[]={1,0,0,0,1,1,1,1,1,2,4,8},expected[]={6,8,14,26};
    skimagexx::set_num_threads(workers);
    check(moments_project(image.data(),{outer,3,inner},powers,4,1,answer.data()+1,skimagexx::get_num_threads())==0);
    for(std::size_t o=0;o<outer;++o)for(std::size_t k=0;k<4;++k)for(std::size_t x=0;x<inner;++x)
        check(answer[1+(o*4+k)*inner+x]==expected[k]);
    check(answer.front()==-81&&answer.back()==-81);skimagexx::set_num_threads(1);
}
void guards() {
    double a=1,p=1,out=-7;
    const auto rejects=[&](const std::vector<std::size_t>&s,std::size_t K,std::size_t axis,int workers) {
        bool rejected=false;try{moments_project(&a,s,&p,K,axis,&out,workers);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected&&out==-7);
    };
    rejects({},1,0,1);rejects({1},1,1,1);rejects({1},1,0,0);rejects({1},1,0,1025);
    rejects({std::numeric_limits<std::size_t>::max()},1,0,1);rejects({1},std::numeric_limits<std::size_t>::max(),0,1);
    bool rejected=false;try{moments_project(static_cast<const double*>(nullptr),{1},&p,1,0,&out,1);}catch(const std::invalid_argument&){rejected=true;}check(rejected&&out==-7);
#ifndef _OPENMP
    rejects({1},1,0,4);
#endif
}
void large(int workers) {
    constexpr std::size_t width=131073;
    std::vector<float> a(3*width),out(2*width+2,-99);
    std::fill(a.begin(),a.begin()+width,1);std::fill(a.begin()+width,a.begin()+2*width,2);std::fill(a.begin()+2*width,a.end(),3);
    const float p[]={1,0,1,1,1,2};
    skimagexx::set_num_threads(workers);check(skimagexx::get_num_threads()==workers);
    check(moments_project(a.data(),{3,width},p,2,0,out.data()+1,skimagexx::get_num_threads())==0);
    for(std::size_t i=0;i<width;++i){check(out[i+1]==6);check(out[width+i+1]==8);}
    check(out.front()==-99&&out.back()==-99);skimagexx::set_num_threads(1);
}
}
int main() {
    check(skimagexx::get_num_threads()==1);
    literal<float>();literal<double>();literal<std::complex<float>>();literal<std::complex<double>>();
    complex_literal<float>();complex_literal<double>();floating<float>();floating<double>();contiguous_fast_paths<float>();contiguous_fast_paths<double>();fused_powers<float>();fused_powers<double>();guards();large(1);fused_worker_blocks(1,true);fused_worker_blocks(1,false);
#ifdef _OPENMP
    large(4);fused_worker_blocks(4,true);fused_worker_blocks(4,false);
#else
    bool rejected=false;try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
#endif
    check(skimagexx::get_num_threads()==1);
    std::cout<<"PASS moments controls="<<controls<<" avx2="<<skimagexx::has_avx2()<<'\n';
}
