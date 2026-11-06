#include "skimagexx/label_pair_metrics.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <utility>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx::detail {
namespace {
class Environment {
    std::fenv_t saved_{};
    bool active_,valid_=true;
public:
    explicit Environment(bool active):active_(active) {
        if(active_)valid_=std::feholdexcept(&saved_)==0;
    }
    ~Environment(){if(active_&&valid_)std::fesetenv(&saved_);}
    bool valid()const noexcept{return valid_;}
};
unsigned status() noexcept {
    const int flags=std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    return (flags&FE_INVALID?1u:0u)|(flags&FE_DIVBYZERO?2u:0u)
         | (flags&FE_OVERFLOW?4u:0u)|(flags&FE_UNDERFLOW?8u:0u);
}
}

bool label_pair_counts(const LabelPairInput& input,LabelPairCsr& output,int workers) {
    const int snapshot=workers==0?skimagexx::get_num_threads():workers;
    if(snapshot<1||snapshot>1024||!input.count||input.count>label_pair_max_samples
       ||!input.truth||!input.test||input.ignored_count>64
       ||(input.ignored_count&&!input.ignored_truth_labels))return false;
    if(input.normalize) {
        if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        if(_mm_getcsr()&((3u<<13)|(1u<<15)|(1u<<6)))return false;
#endif
    }
    std::array<std::int64_t,64> ignored;
    if(input.ignored_count)std::copy_n(input.ignored_truth_labels,input.ignored_count,ignored.begin());
    const auto ignored_end=ignored.begin()+static_cast<std::ptrdiff_t>(input.ignored_count);
    std::sort(ignored.begin(),ignored_end);
    const auto is_ignored=[&](std::int64_t row) {
        return input.ignored_count&&std::binary_search(ignored.begin(),ignored_end,row);
    };
    std::int64_t max_row=0,max_column=0;
    std::size_t kept=0;
    for(std::size_t i=0;i<input.count;++i) {
        const auto row=input.truth[i],column=input.test[i];
        if(row<0||row>label_pair_max_label||column<0||column>label_pair_max_label)return false;
        max_row=std::max(max_row,row);max_column=std::max(max_column,column);
        kept+=!is_ignored(row);
    }
    Environment environment(input.normalize);
    if(!environment.valid())return false;
    std::vector<std::uint64_t> keys(input.count);
    auto* packed=keys.data();const auto* truth=input.truth;const auto* test=input.test;
    const int team=input.count>=parallel_min?snapshot:1;
    partition(input.count,team,[&](std::size_t first,std::size_t last,std::size_t) {
#if defined(_OPENMP) && !defined(_MSC_VER)
        #pragma omp simd
#endif
        for(std::size_t i=first;i<last;++i)
            packed[i]=(static_cast<std::uint64_t>(truth[i])<<20)|static_cast<std::uint64_t>(test[i]);
    });
    std::sort(keys.begin(),keys.end());
    LabelPairCsr result;
    result.rows=static_cast<std::size_t>(max_row)+1;
    result.columns=static_cast<std::size_t>(max_column)+1;
    result.total_samples=input.count;result.kept_samples=kept;
    result.indptr.assign(result.rows+1,0);
    result.indices.reserve(input.count);result.data.reserve(input.count);
    double inverse=0.;
    if(input.normalize) {
        if(kept)inverse=1.0/static_cast<double>(kept);
        else {volatile double zero=0.;inverse=zero/zero;}
    }
    for(std::size_t first=0;first<keys.size();) {
        std::size_t last=first+1;
        while(last<keys.size()&&keys[last]==keys[first])++last;
        const auto row=static_cast<std::size_t>(keys[first]>>20);
        const auto column=static_cast<std::int64_t>(keys[first]&((UINT64_C(1)<<20)-1));
        const bool ignored_row=is_ignored(static_cast<std::int64_t>(row));
        double value;
        if(!input.normalize)value=ignored_row?0.:static_cast<double>(last-first);
        else {
            const double weight=kept?(ignored_row?0.:inverse):inverse;
            value=weight;
            // Match per-pixel division followed by ordered duplicate addition.
            // count/N or count*inverse would have different rounding.
            for(std::size_t i=first+1;i<last;++i)value+=weight;
        }
        ++result.indptr[row+1];result.indices.push_back(column);result.data.push_back(value);
        first=last;
    }
    for(std::size_t row=0;row<result.rows;++row)result.indptr[row+1]+=result.indptr[row];
    result.normalization_fp_status=input.normalize?status():0;
    output=std::move(result);
    return true;
}
}
