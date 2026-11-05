#pragma once
#include "skimagexx/matrix_transform.hpp"
#include "skimagexx/image.hpp"
#include "fixtures/matrix_prescott_native.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
namespace prescott_controls {
struct Counts {std::size_t checks=0,raw_cases=0,raw_words=0,mapping_values=0;};
inline Counts run() {
    using namespace skimagexx;
    Counts counts;
    const auto check=[&](bool ok,const std::string& name){++counts.checks;if(!ok)throw std::runtime_error("Prescott "+name+" check"+std::to_string(counts.checks));};
    const auto from=[](std::uint64_t word){double x;std::memcpy(&x,&word,8);return x;};
    const auto bits=[](double x){std::uint64_t word;std::memcpy(&word,&x,8);return word;};
    struct Restore {std::fenv_t fp;int workers;Restore():workers(get_num_threads()){std::fegetenv(&fp);}~Restore(){set_num_threads(workers);std::fesetenv(&fp);}} restore;
    check(get_num_threads()==1,"defaultworker");
    for(const auto& row:prescott_fixture::cases) {
        const auto profile=row.kind==0 ? MatrixArithmeticProfile::prescott0321_gemm : row.kind==1 ? MatrixArithmeticProfile::prescott0321_gemv_nontransposed : row.residue ? MatrixArithmeticProfile::prescott0321_gemv_transposed_peeled : MatrixArithmeticProfile::prescott0321_gemv_transposed_aligned;
        const int outputs=row.kind==0 ? row.m*row.n : row.k;
        for(int index=0;index<outputs;++index) {
            double x[5],m[5];int component=index;
            if(row.kind==0) {
                const int i=index%row.m,j=index/row.m;component=j;
                for(int k=0;k<row.k;++k){x[k]=from(row.a[i+k*row.m]);m[k]=from(row.b[k+j*row.k]);}
            }else for(int k=0;k<row.k;++k) {
                x[k]=from(row.x[k]);m[k]=from(row.a[row.kind==1 ? index+k*row.lda : k+index*row.lda]);
            }
            const double actual=detail::matrix_prescott_dot_f64(x,m,row.k,profile,component);
            check(bits(actual)==row.expected[index],std::string(row.id)+" word"+std::to_string(index)+" actual"+std::to_string(bits(actual))+" expected"+std::to_string(row.expected[index]));
            ++counts.raw_words;
        }
        ++counts.raw_cases;
    }
    check(counts.raw_cases==48 && counts.raw_words==432,"rawscope");
    for(const int workers:{1
#ifdef _OPENMP
        ,4
#endif
    }) {
        set_num_threads(workers);
        for(int k=3;k<=5;++k)for(int policy=7;policy<=10;++policy) {
            const auto mode=MatrixArithmeticProfile(policy);const std::size_t d=k-1,n=policy==7?5:1;
            std::vector<double> p(n*d),m(k*k),o(n*d,-123.);
            for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<d;++j)p[i*d+j]=double(int((i+j)%5)-2);
            for(std::size_t j=0;j<d;++j){m[j*k+j]=1.;m[j*k+d]=double(j+1)*.5;}m[d*k+d]=1.;
            const MatrixReadView pv{p.data(),n,d,std::ptrdiff_t(d*8),8,p.data(),p.size()*8},mv{m.data(),std::size_t(k),std::size_t(k),k*8,8,m.data(),m.size()*8};
            const MatrixWriteView ov{o.data(),n,d,o.size()*8};
            std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
            check(matrix_transform_into_v3(pv,mv,ov,mode)==MatrixMapStatus::ok,"mappingstatus");
            check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO && get_num_threads()==workers,"mappingstate");
            for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<d;++j){check(bits(o[i*d+j])==bits(p[i*d+j]+double(j+1)*.5),"mappingwords");++counts.mapping_values;}
            std::fill(o.begin(),o.end(),-123.);
            check(matrix_transform_into(pv,mv,ov,mode)==MatrixMapStatus::unsupported_profile && matrix_transform_into_v2(pv,mv,ov,mode)==MatrixMapStatus::unsupported_profile,"oldversionrefusal");
            check(std::all_of(o.begin(),o.end(),[](double v){return v==-123.;}),"oldversionnowrite");
            const int rounding=std::fegetround();std::fesetround(FE_DOWNWARD);
            check(matrix_transform_into_v3(pv,mv,ov,mode)==MatrixMapStatus::unsafe_fp_environment,"x87roundrefusal");std::fesetround(rounding);
#if defined(__x86_64__) || defined(_M_X64)
            const auto csr=_mm_getcsr();_mm_setcsr((csr&~(3u<<13))|(1u<<13));
            check(matrix_transform_into_v3(pv,mv,ov,mode)==MatrixMapStatus::unsafe_fp_environment,"SSEroundrefusal");_mm_setcsr(csr);
#endif
            auto bad=pv;bad.allocation_bytes=1;check(matrix_transform_into_v3(bad,mv,ov,mode)==MatrixMapStatus::invalid_view,"boundsrefusal");
            p.back()=std::numeric_limits<double>::quiet_NaN();check(matrix_transform_into_v3(pv,mv,ov,mode)==MatrixMapStatus::nonfinite_or_range,"nonfinitebeforewrite");
            check(std::all_of(o.begin(),o.end(),[](double v){return v==-123.;}),"refusalnowrite");
        }
    }
    // Actual global budget threshold, independently exact dyadic identity+translation.
    const std::size_t n=131073,d=2;std::vector<double> p(n*d),o(n*d),serial;
    for(std::size_t i=0;i<p.size();++i)p[i]=double(int(i%17)-8);
    std::vector<double> m={1,0,.5,0,1,-.5,0,0,1};
    const MatrixReadView pv{p.data(),n,d,16,8,p.data(),p.size()*8},mv{m.data(),3,3,24,8,m.data(),m.size()*8};
    const MatrixWriteView ov{o.data(),n,d,o.size()*8};
    for(int workers:{1
#ifdef _OPENMP
        ,4
#endif
    }) {set_num_threads(workers);check(matrix_transform_into_v3(pv,mv,ov,MatrixArithmeticProfile::prescott0321_gemm)==MatrixMapStatus::ok,"largecurrentworkers");
        for(std::size_t i=0;i<n;++i){check(bits(o[2*i])==bits(p[2*i]+.5)&&bits(o[2*i+1])==bits(p[2*i+1]-.5),"largewords");counts.mapping_values+=2;}
        if(workers==1)serial=o;else check(std::memcmp(o.data(),serial.data(),o.size()*8)==0,"budgetsamewords");}
    set_num_threads(1);
    std::cout<<"Prescott rawcases="<<counts.raw_cases<<" rawwords="<<counts.raw_words<<" checks="<<counts.checks<<" mappingvalues="<<counts.mapping_values<<" PASS\n";
    return counts;
}
}
