#pragma once
#include <skimagexx/label_pair_metrics.hpp>
#include <skimagexx/image.hpp>
#include "fixtures/label_pair_metrics_originals_v1.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace contingency_preweighted_controls {
inline std::uint64_t bits(double value){std::uint64_t result;std::memcpy(&result,&value,8);return result;}
inline double word(std::uint64_t value){double result;std::memcpy(&result,&value,8);return result;}
inline std::size_t run() {
    using namespace skimagexx;using namespace skimagexx::detail;
    std::size_t checks=0;
    const auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("Preweighted control "+std::to_string(checks));};
    const int old=get_num_threads();check(old==1);
    std::fenv_t saved;check(std::fegetenv(&saved)==0);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto original_csr=_mm_getcsr();
#endif
    for(int workers:{1,4}) {
        set_num_threads(workers);
        for(const auto& f:label_pair_fixture::originals) {
            std::size_t kept=0;
            for(auto value:f.truth)kept+=std::find(f.ignored.begin(),f.ignored.end(),value)==f.ignored.end();
            const double weight=f.normalize?(kept?1./double(kept):word(UINT64_C(0xfff8000000000000))):1.;
            std::vector<double> weights;
            for(auto value:f.truth)weights.push_back(f.normalize&&!kept?weight:
                (std::find(f.ignored.begin(),f.ignored.end(),value)==f.ignored.end()?weight:0.));
            const auto before=weights;
            std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
            const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            const auto csr=_mm_getcsr();
#endif
            LabelPairCsrStorage actual;
            check(label_pair_counts_preweighted({f.truth.data(),f.test.data(),weights.data(),weights.size()},actual));
            check(actual.rows==f.rows&&actual.columns==f.columns&&actual.indptr==f.indptr);
            check(actual.indices.size()==f.truth.size()&&actual.data.size()==f.truth.size());
            check(std::size_t(actual.indptr.back())==f.indices.size());
            for(std::size_t i=0;i<f.indices.size();++i){check(actual.indices[i]==f.indices[i]);check(bits(actual.data[i])==f.data_bits[i]);}
            check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            check(_mm_getcsr()==csr);
#endif
            for(std::size_t i=0;i<weights.size();++i)check(bits(weights[i])==bits(before[i]));
            check(get_num_threads()==workers);
        }
        // Weights precede a callback-style relabel. Old ignored zero joins a
        // current kept pair; re-applying ignore labels would give wrong data.
        const std::int64_t truth[]{1,1,1,2},test[]{0,0,0,0};
        const double weights[]{0.,.5,.5,0.};LabelPairCsrStorage changed;
        check(label_pair_counts_preweighted({truth,test,weights,4},changed,workers));
        check(changed.rows==3&&changed.columns==1&&changed.indptr==std::vector<std::int64_t>({0,0,1,2}));
        check(changed.indices[0]==0&&changed.indices[1]==0&&bits(changed.data[0])==bits(1.)&&bits(changed.data[1])==0);
        constexpr std::size_t n=262151;
        std::vector<std::int64_t> a(n),b(n);std::vector<double> values(n);std::array<double,35> expected{};
        for(std::size_t i=0;i<n;++i){a[i]=std::int64_t(i%5);b[i]=std::int64_t((i/5)%7);values[i]=(i%3)?.25:0.;expected[std::size_t(a[i])*7+std::size_t(b[i])]+=values[i];}
        LabelPairCsrStorage large;check(label_pair_counts_preweighted({a.data(),b.data(),values.data(),n},large));
        check(large.rows==5&&large.columns==7&&large.indptr.back()==35&&large.data.size()==n);
        for(std::size_t r=0;r<5;++r){check(large.indptr[r]==std::int64_t(r*7));for(std::size_t c=0;c<7;++c){check(large.indices[r*7+c]==std::int64_t(c));check(bits(large.data[r*7+c])==bits(expected[r*7+c]));}}
    }
    const std::int64_t a[]{0,1},b[]{0,1};const double weights[]{1.,1.};
    LabelPairCsrStorage unchanged;unchanged.rows=9;unchanged.columns=8;unchanged.indptr={7};unchanged.indices={6};unchanged.data={5.};
    const auto reject=[&](LabelPairWeightedInput input,int workers=1){
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        check(!label_pair_counts_preweighted(input,unchanged,workers));
        check(unchanged.rows==9&&unchanged.columns==8&&unchanged.indptr==std::vector<std::int64_t>{7}&&unchanged.indices==std::vector<std::int64_t>{6}&&unchanged.data==std::vector<double>{5.});
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
    };
    const LabelPairWeightedInput valid{a,b,weights,2};
    auto input=valid;input.count=0;reject(input);input=valid;input.count=label_pair_max_samples+1;reject(input);
    input=valid;input.truth=nullptr;reject(input);input=valid;input.test=nullptr;reject(input);input=valid;input.weights=nullptr;reject(input);
    reject(valid,-1);reject(valid,1025);
    const std::int64_t negative[]{-1,0},excess[]{label_pair_max_label+1,0};
    input=valid;input.truth=negative;reject(input);input=valid;input.test=excess;reject(input);
    const std::array<std::array<double,2>,5> bad_weights{{{.25,.5},
        {word(UINT64_C(0x7ff0000000000001)),1.},{-0.,1.},
        {word(UINT64_C(0xfff8000000000000)),0.},{2.,2.}}};
    for(const auto& bad:bad_weights){input=valid;input.weights=bad.data();reject(input);}
    check(std::fesetround(FE_DOWNWARD)==0);reject(valid);check(std::fegetround()==FE_DOWNWARD);
    std::fesetenv(&saved);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    for(unsigned mask:{1u<<13,1u<<15,1u<<6}){_mm_setcsr(original_csr|mask);const auto csr=_mm_getcsr();reject(valid);check(_mm_getcsr()==csr);}
    _mm_setcsr(original_csr);
#endif
    set_num_threads(old);check(get_num_threads()==1);std::fesetenv(&saved);
    return checks;
}
}
