#include <skimagexx/label_pair_metrics.hpp>
#include <skimagexx/image.hpp>
#include "fixtures/label_pair_rand_originals_v1.hpp"
#include <cfenv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

int main() {
    using namespace skimagexx;using namespace skimagexx::detail;
    std::size_t checks=0;const auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("Rand control "+std::to_string(checks));};
    const int initial=get_num_threads();check(initial==1);
    std::fenv_t original;std::fegetenv(&original);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto initial_csr=_mm_getcsr();
#endif
    for(int workers:{1,4}) {
        set_num_threads(workers);
        for(const auto& row:label_pair_rand_originals::rows) {
            const auto before_true=row.truth,before_test=row.test;
            LabelPairRandTerms terms;
            check(label_pair_rand_terms({row.truth.data(),row.test.data(),row.truth.size(),row.ignored.data(),row.ignored.size(),false},terms));
            check(terms.joint==row.terms[0]&&terms.row==row.terms[1]&&terms.column==row.terms[2]);
            LabelPairScalar64 precision,recall,fscore,error;
            std::feraiseexcept(FE_UNDERFLOW);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            const auto csr=_mm_getcsr();
#endif
            check(label_pair_rand_divide(terms.joint,terms.row,precision));
            check(label_pair_rand_divide(terms.joint,terms.column,recall));
            check(label_pair_rand_fscore(terms,row.alpha,fscore));
            check(label_pair_rand_error(fscore.bits,error));
            check(error.bits==row.words[0]&&precision.bits==row.words[1]&&recall.bits==row.words[2]);
            check(precision.fp_status==(terms.row==0?1u:0u));
            check(recall.fp_status==(terms.column==0?1u:0u));
            check(fscore.fp_status==((row.alpha*terms.row+(1-row.alpha)*terms.column)==0?1u:0u)&&error.fp_status==0);
            check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            check(_mm_getcsr()==csr);
#endif
            check(before_true==row.truth&&before_test==row.test&&get_num_threads()==workers);
        }
    }
    constexpr std::size_t n=262151;std::vector<std::int64_t> labels(n,1);LabelPairRandTerms large;
    check(label_pair_rand_terms({labels.data(),labels.data(),n,nullptr,0,false},large));
    const auto pairs=double(std::uint64_t(n)*(n-1));check(large.joint==pairs&&large.row==pairs&&large.column==pairs);
    LabelPairScalar64 value{123,45};const auto unchanged=[&]{return value.bits==123&&value.fp_status==45;};
    check(!label_pair_rand_divide(-1,2,value)&&unchanged());
    check(!label_pair_rand_divide(3,2,value)&&unchanged());
    check(!label_pair_rand_divide(.5,2,value)&&unchanged());
    check(!label_pair_rand_fscore({0,0,0},2,value)&&unchanged());
    const std::uint64_t snan_bits=UINT64_C(0x7ff0000000000001);double snan;std::memcpy(&snan,&snan_bits,8);
    const auto before_flags=std::fetestexcept(FE_ALL_EXCEPT);
    check(!label_pair_rand_divide(snan,2,value)&&unchanged());
    check(!label_pair_rand_error(snan_bits,value)&&unchanged());
    check(std::fetestexcept(FE_ALL_EXCEPT)==before_flags);
    check(std::fesetround(FE_DOWNWARD)==0);
    check(!label_pair_rand_divide(0,0,value)&&unchanged());
    check(!label_pair_rand_fscore({0,0,0},.5,value)&&unchanged());
    check(!label_pair_rand_error(0,value)&&unchanged());
    check(std::fegetround()==FE_DOWNWARD);std::fesetenv(&original);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    _mm_setcsr(initial_csr|(1u<<13));const auto changed=_mm_getcsr();
    check(!label_pair_rand_divide(0,0,value)&&unchanged());
    check(_mm_getcsr()==changed);_mm_setcsr(initial_csr);
#endif
    set_num_threads(initial);check(get_num_threads()==1);std::fesetenv(&original);
    std::cout<<"Rand controls="<<checks<<"; originals=5; workers=1,4; restored=1\n";
}
