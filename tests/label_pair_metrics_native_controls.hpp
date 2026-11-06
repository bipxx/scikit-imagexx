#pragma once
#include <skimagexx/label_pair_metrics.hpp>
#include <skimagexx/image.hpp>
#include "fixtures/label_pair_metrics_originals_v1.hpp"
#include <array>
#include <cfenv>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace label_pair_controls {
using namespace skimagexx;
using namespace skimagexx::detail;
inline std::uint64_t bits(double value){std::uint64_t result;std::memcpy(&result,&value,8);return result;}
inline std::size_t run() {
    std::size_t checks=0;
    const auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("label-pair native/SDK control failed");};
    const int old=get_num_threads();check(old==1);
    std::fenv_t original_env;std::fegetenv(&original_env);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto original_csr=_mm_getcsr();
#endif
    for(int workers:{1,4}) {
        set_num_threads(workers);
        for(const auto& f:label_pair_fixture::originals) {
            const LabelPairInput input{f.truth.data(),f.test.data(),f.truth.size(),f.ignored.data(),f.ignored.size(),f.normalize};
            const auto truth=f.truth,test=f.test;
            std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
            const int rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            const auto csr=_mm_getcsr();
#endif
            LabelPairCsr actual;check(label_pair_counts(input,actual));
            check(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            check(_mm_getcsr()==csr);
#endif
            check(actual.rows==f.rows&&actual.columns==f.columns&&actual.total_samples==f.truth.size());
            check(actual.indptr==f.indptr&&actual.indices==f.indices&&actual.data.size()==f.data_bits.size());
            for(std::size_t i=0;i<actual.data.size();++i)check(bits(actual.data[i])==f.data_bits[i]);
            check(actual.normalization_fp_status==f.status&&f.truth==truth&&f.test==test);
            check(actual.indices.data()!=f.truth.data()&&actual.indptr.data()!=f.truth.data());
            check(get_num_threads()==workers);
        }
        // Independent dense analytic histogram; no original call/new golden.
        constexpr std::size_t n=262151;
        std::vector<std::int64_t> truth(n),test(n);
        std::size_t kept=0;
        for(std::size_t i=0;i<n;++i){truth[i]=static_cast<std::int64_t>(i%5);test[i]=static_cast<std::int64_t>((i/5)%7);kept+=truth[i]!=0;}
        const std::int64_t ignored[]{0,0,-1};
        for(bool normalize:{false,true}) {
            std::array<double,35> expected{};
            const double weight=normalize?1./static_cast<double>(kept):1.;
            for(std::size_t i=0;i<n;++i)expected[std::size_t(truth[i])*7+std::size_t(test[i])]+=truth[i]?weight:0.;
            std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
            const int rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
            LabelPairCsr actual;check(label_pair_counts({truth.data(),test.data(),n,ignored,3,normalize},actual,workers));
            check(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
            check(actual.rows==5&&actual.columns==7&&actual.indices.size()==35&&actual.kept_samples==kept&&actual.normalization_fp_status==0);
            for(std::size_t row=0;row<5;++row){check(actual.indptr[row]==std::int64_t(row*7));for(std::size_t col=0;col<7;++col){check(actual.indices[row*7+col]==std::int64_t(col));check(bits(actual.data[row*7+col])==bits(expected[row*7+col]));}}
            check(actual.indptr.back()==35&&get_num_threads()==workers);
        }
    }
    const std::int64_t truth[]{0,1},test[]{1,0},ignored[]{0};
    const LabelPairInput valid{truth,test,2,ignored,1,true};
    LabelPairCsr unchanged;unchanged.rows=99;unchanged.columns=98;unchanged.indptr={7};unchanged.indices={6};unchanged.data={5.};
    const auto reject=[&](LabelPairInput input,int workers=1){
        check(!label_pair_counts(input,unchanged,workers));
        check(unchanged.rows==99&&unchanged.columns==98&&unchanged.indptr==std::vector<std::int64_t>{7}&&unchanged.indices==std::vector<std::int64_t>{6}&&unchanged.data==std::vector<double>{5.});
    };
    auto input=valid;input.count=0;reject(input);input=valid;input.count=label_pair_max_samples+1;reject(input);
    input=valid;input.truth=nullptr;reject(input);input=valid;input.test=nullptr;reject(input);
    input=valid;input.ignored_truth_labels=nullptr;reject(input);input=valid;input.ignored_count=65;reject(input);
    reject(valid,-1);reject(valid,1025);
    const std::int64_t negative[]{-1,0},excessive[]{label_pair_max_label+1,0};
    input=valid;input.truth=negative;reject(input);input=valid;input.test=negative;reject(input);
    input=valid;input.truth=excessive;reject(input);input=valid;input.test=excessive;reject(input);
    std::fesetround(FE_DOWNWARD);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    reject(valid);check(std::fegetround()==FE_DOWNWARD&&std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    input=valid;input.normalize=false;LabelPairCsr integer;check(label_pair_counts(input,integer,1));
    check(integer.data==std::vector<double>({0.,1.})&&std::fegetround()==FE_DOWNWARD&&std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    std::fesetround(FE_TONEAREST);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto csr=_mm_getcsr();
    for(unsigned mask:{(1u<<13),(1u<<15),(1u<<6)}) {
        _mm_setcsr((csr&~(3u<<13))|mask);const auto before=_mm_getcsr();
        reject(valid);check(_mm_getcsr()==before);
    }
    _mm_setcsr(csr);
#endif
    const std::int64_t maximum[]{label_pair_max_label};LabelPairCsr sparse;
    check(label_pair_counts({maximum,maximum,1,nullptr,0,false},sparse,1));
    check(sparse.rows==std::size_t(label_pair_max_label)+1&&sparse.columns==sparse.rows);
    check(sparse.indptr.size()==sparse.rows+1&&sparse.indptr.front()==0&&sparse.indptr[sparse.rows-1]==0&&sparse.indptr.back()==1);
    check(sparse.indices==std::vector<std::int64_t>{label_pair_max_label}&&sparse.data==std::vector<double>{1.});
    set_num_threads(old);check(get_num_threads()==1);std::fesetenv(&original_env);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    _mm_setcsr(original_csr);
#endif
    return checks;
}
// Dense one-pair streams have an independent complete-storage expectation:
// one logical repeated sum, followed by the unchanged sorted source words.
// Non-dyadic weights distinguish ordered addition from count*weight.
inline std::size_t density() {
    std::size_t checks=0;const auto check=[&](bool v){++checks;if(!v)throw std::runtime_error("label-pair dense storage control failed");};
    const auto initial=get_num_threads();check(initial==1);std::fenv_t caller;check(std::fegetenv(&caller)==0);
    for(auto n:{std::size_t{4},std::size_t{31},std::size_t{262151}}){
        std::vector<std::int64_t> truth(n,1),test(n,2);
        for(int workers:{1,4}){set_num_threads(workers);
            for(bool normalize:{false,true}){
                const double weight=normalize?1./double(n):1.;double expected=weight;
                for(std::size_t i=1;i<n;++i)expected+=weight;
                const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();LabelPairCsr result;
                check(label_pair_counts({truth.data(),test.data(),n,nullptr,0,normalize},result));
                check(result.rows==2&&result.columns==3&&result.indptr==std::vector<std::int64_t>({0,0,1}));
                check(result.indices==std::vector<std::int64_t>{2}&&result.data.size()==1);
                check(bits(result.data[0])==bits(expected)&&result.kept_samples==n&&result.normalization_fp_status==0);
                check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding);
            }
            for(std::uint64_t word:{UINT64_C(0),bits(1./17.),UINT64_C(0x7ff8123456789abc),UINT64_C(0xfff8fedcba987654)}){
                double value;std::memcpy(&value,&word,8);std::vector<double> weights(n,value);double expected=value;
                for(std::size_t i=1;i<n;++i)expected+=value;
                LabelPairCsrStorage result;const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
                check(label_pair_counts_preweighted({truth.data(),test.data(),weights.data(),n},result));
                check(result.rows==2&&result.columns==3&&result.indptr==std::vector<std::int64_t>({0,0,1}));
                check(result.indices.size()==n&&result.data.size()==n&&bits(result.data[0])==bits(expected));
                for(std::size_t i=0;i<n;++i){check(result.indices[i]==2);if(i)check(bits(result.data[i])==word);check(bits(weights[i])==word);}
                check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding);
            }
            check(get_num_threads()==workers);
        }
    }
    // The unnormalized dense integer path accepts all caller rounding modes.
    const std::int64_t truth[]{0,0,0,0,0,0,0,0},test[]{0,0,0,0,0,0,0,0};
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        check(std::fesetround(rounding)==0);LabelPairCsr result;const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        check(label_pair_counts({truth,test,8,nullptr,0,false},result,1));check(result.data==std::vector<double>{8.});
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding);
    }
    set_num_threads(initial);check(std::fesetenv(&caller)==0);check(get_num_threads()==1);return checks;
}
}
