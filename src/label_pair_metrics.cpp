#include "skimagexx/label_pair_metrics.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
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
void pack_pairs(const std::int64_t* truth,const std::int64_t* test,
                std::size_t count,std::uint64_t* packed,int snapshot) {
    const int team=count>=parallel_min?snapshot:1;
    partition(count,team,[&](std::size_t first,std::size_t last,std::size_t) {
#if defined(_OPENMP) && !defined(_MSC_VER)
        #pragma omp simd
#endif
        for(std::size_t i=first;i<last;++i)
            packed[i]=(static_cast<std::uint64_t>(truth[i])<<20)|static_cast<std::uint64_t>(test[i]);
    });
}
// Bounded low-cardinality aggregation avoids sorting only when the complete
// grid is small relative to the sample stream. Wider/sparser shapes retain
// the original packed-key path. Counts fit uint32 because N<=2^20.
bool dense_shape(std::size_t count,std::size_t rows,std::size_t columns,std::size_t& cells) {
    constexpr std::size_t maximum_cells=65536; // at most256KiB of counters
    if(rows>maximum_cells/columns)return false;
    cells=rows*columns;return cells<=count/4;
}
std::vector<std::uint32_t> dense_histogram(const std::int64_t* truth,const std::int64_t* test,
    std::size_t count,std::size_t columns,std::size_t cells) {
    std::vector<std::uint32_t> histogram(cells,0);
    for(std::size_t i=0;i<count;++i)++histogram[std::size_t(truth[i])*columns+std::size_t(test[i])];
    return histogram;
}
bool uniform_row_weights(const LabelPairWeightedInput& input,std::size_t rows,
    std::vector<std::uint64_t>& words) {
    words.assign(rows,0);std::vector<unsigned char> seen(rows,0);
    for(std::size_t i=0;i<input.count;++i){
        const auto row=std::size_t(input.truth[i]);std::uint64_t bits;std::memcpy(&bits,input.weights+i,8);
        if(seen[row] && words[row]!=bits)return false;
        seen[row]=1;words[row]=bits;
    }
    return true;
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
    std::size_t cells=0;
    const auto rows=static_cast<std::size_t>(max_row)+1,columns=static_cast<std::size_t>(max_column)+1;
    const bool dense=dense_shape(input.count,rows,columns,cells);
    std::vector<std::uint64_t> keys;std::vector<std::uint32_t> histogram;
    if(dense)histogram=dense_histogram(input.truth,input.test,input.count,columns,cells);
    else {
        keys.resize(input.count);
        pack_pairs(input.truth,input.test,input.count,keys.data(),snapshot);
        std::sort(keys.begin(),keys.end());
    }
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
    auto append=[&](std::size_t row,std::int64_t column,std::size_t multiplicity) {
        const bool ignored_row=is_ignored(static_cast<std::int64_t>(row));
        double value;
        if(!input.normalize)value=ignored_row?0.:static_cast<double>(multiplicity);
        else {
            const double weight=kept?(ignored_row?0.:inverse):inverse;
            value=weight;
            // Match per-pixel division followed by ordered duplicate addition.
            // count/N or count*inverse would have different rounding.
            for(std::size_t i=1;i<multiplicity;++i)value+=weight;
        }
        ++result.indptr[row+1];result.indices.push_back(column);result.data.push_back(value);
    };
    if(dense) {
        for(std::size_t row=0;row<rows;++row)for(std::size_t column=0;column<columns;++column)
            if(const auto n=histogram[row*columns+column])append(row,static_cast<std::int64_t>(column),n);
    } else for(std::size_t first=0;first<keys.size();) {
        std::size_t last=first+1;
        while(last<keys.size()&&keys[last]==keys[first])++last;
        append(static_cast<std::size_t>(keys[first]>>20),
            static_cast<std::int64_t>(keys[first]&((UINT64_C(1)<<20)-1)),last-first);
        first=last;
    }
    for(std::size_t row=0;row<result.rows;++row)result.indptr[row+1]+=result.indptr[row];
    result.normalization_fp_status=input.normalize?status():0;
    output=std::move(result);
    return true;
}

bool label_pair_rand_terms(const LabelPairInput& input,LabelPairRandTerms& output,int workers) {
    if(input.normalize)return false;
    LabelPairCsr counts;
    if(!label_pair_counts(input,counts,workers))return false;
    std::vector<std::uint64_t> columns(counts.columns,0);
    std::uint64_t joint=0,rows=0;
    for(std::size_t r=0;r<counts.rows;++r) {
        std::uint64_t total=0;
        for(auto i=counts.indptr[r];i<counts.indptr[r+1];++i) {
            const auto count=static_cast<std::uint64_t>(counts.data[std::size_t(i)]);
            joint+=count*(count?count-1:0);total+=count;
            columns[std::size_t(counts.indices[std::size_t(i)])]+=count;
        }
        rows+=total*(total?total-1:0);
    }
    std::uint64_t cols=0;
    for(const auto count:columns)cols+=count*(count?count-1:0);
    output={static_cast<double>(joint),static_cast<double>(rows),static_cast<double>(cols)};
    return true;
}

namespace {
bool finite_word(double value) noexcept {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}
bool integer_pair_total(double value) noexcept {
    return finite_word(value)&&value>=0.&&value<=0x1p40
        &&static_cast<double>(static_cast<std::uint64_t>(value))==value;
}
bool rand_environment() noexcept {
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr()&((3u<<13)|(1u<<15)|(1u<<6)))return false;
#endif
    return true;
}
void scalar_result(double value,LabelPairScalar64& output) noexcept {
    std::memcpy(&output.bits,&value,sizeof(value));output.fp_status=status();
}
}

bool label_pair_rand_divide(double numerator,double denominator,LabelPairScalar64& output) {
    if(!integer_pair_total(numerator)||!integer_pair_total(denominator)
       ||numerator>denominator||!rand_environment())return false;
    Environment environment(true);if(!environment.valid())return false;
    volatile double first=numerator,second=denominator;
    scalar_result(first/second,output);return true;
}

bool label_pair_rand_fscore(const LabelPairRandTerms& terms,double alpha,LabelPairScalar64& output) {
    if(!integer_pair_total(terms.joint)||!integer_pair_total(terms.row)
       ||!integer_pair_total(terms.column)||terms.joint>terms.row||terms.joint>terms.column
       ||!finite_word(alpha)||alpha<0.||alpha>1.||!rand_environment())return false;
    Environment environment(true);if(!environment.valid())return false;
    // Literal source scalar association; contraction/reassociation is disabled.
    volatile double weight=alpha,row=terms.row,column=terms.column,joint=terms.joint;
    volatile double first=weight*row;
    volatile double other=1.-weight;
    volatile double second=other*column;
    volatile double denominator=first+second;
    scalar_result(joint/denominator,output);return true;
}

bool label_pair_rand_error(std::uint64_t bits,LabelPairScalar64& output) {
    const bool quiet_nan=(bits&UINT64_C(0x7ff8000000000000))==UINT64_C(0x7ff8000000000000);
    double fscore;std::memcpy(&fscore,&bits,sizeof(fscore));
    // A rounded convex denominator can place fscore just above1. This bounded
    // stage accepts [0,2] plus quiet NaNs, never signaling NaNs/infinities.
    if((!quiet_nan&&(!finite_word(fscore)||fscore<0.||fscore>2.))||!rand_environment())return false;
    Environment environment(true);if(!environment.valid())return false;
    volatile double value=fscore;
    scalar_result(1.-value,output);return true;
}

bool label_pair_counts_preweighted(const LabelPairWeightedInput& input,
                                  LabelPairCsrStorage& output,int workers) {
    // SciPy sparse source conventions, BSD-3-Clause notice retained in
    // third_party/scipy-sparse-LICENSE.txt.
    const int snapshot=workers==0?skimagexx::get_num_threads():workers;
    if(snapshot<1||snapshot>1024||!input.count||input.count>label_pair_max_samples
       ||!input.truth||!input.test||!input.weights||!rand_environment())return false;
    std::int64_t max_row=0,max_column=0;
    std::uint64_t common=0;bool nan_weights=false;
    for(std::size_t i=0;i<input.count;++i) {
        const auto row=input.truth[i],column=input.test[i];
        if(row<0||row>label_pair_max_label||column<0||column>label_pair_max_label)return false;
        max_row=std::max(max_row,row);max_column=std::max(max_column,column);
        std::uint64_t bits;std::memcpy(&bits,input.weights+i,sizeof(bits));
        const bool quiet_nan=(bits&UINT64_C(0x7ff8000000000000))==UINT64_C(0x7ff8000000000000);
        if(quiet_nan) {
            if(i&&(!nan_weights||bits!=common))return false;
            nan_weights=true;common=bits;
        } else {
            if(nan_weights)return false;
            if(bits==0)continue;
            if(bits<UINT64_C(0x3eb0000000000000)||bits>UINT64_C(0x3ff0000000000000)
               ||(common&&common!=bits))return false;
            common=bits;
        }
    }
    Environment environment(true);if(!environment.valid())return false;
    LabelPairCsrStorage result;
    result.rows=static_cast<std::size_t>(max_row)+1;
    result.columns=static_cast<std::size_t>(max_column)+1;
    result.indptr.assign(result.rows+1,0);
    result.indices.resize(input.count);result.data.resize(input.count);
    std::size_t cells=0;std::vector<std::uint64_t> row_words;
    if(dense_shape(input.count,result.rows,result.columns,cells) && uniform_row_weights(input,result.rows,row_words)) {
        const auto histogram=dense_histogram(input.truth,input.test,input.count,result.columns,cells);
        std::size_t cursor=0;
        for(std::size_t row=0;row<result.rows;++row){
            result.indptr[row]=static_cast<std::int64_t>(cursor);
            double weight;std::memcpy(&weight,&row_words[row],8);
            for(std::size_t column=0;column<result.columns;++column){
                const auto n=histogram[row*result.columns+column];
                std::fill_n(result.indices.data()+cursor,n,static_cast<std::int64_t>(column));
                std::fill_n(result.data.data()+cursor,n,weight);cursor+=n;
            }
        }
        result.indptr[result.rows]=static_cast<std::int64_t>(cursor);
        // Uniform row words make the complete source sorted buffers (including
        // backing tails) deterministic. Keep the same ordered coalescer below.
        // Mixed-weight rows retain the original unstable-sort path verbatim.
    } else {
    std::vector<std::uint64_t> keys(input.count);
    pack_pairs(input.truth,input.test,input.count,keys.data(),snapshot);
    // Same row grouping as source coo_tocsr: stable input-pixel traversal.
    for(auto key:keys)++result.indptr[std::size_t(key>>20)];
    std::int64_t total=0;
    for(std::size_t row=0;row<result.rows;++row) {
        const auto count=result.indptr[row];result.indptr[row]=total;total+=count;
    }
    result.indptr[result.rows]=total;
    for(std::size_t i=0;i<input.count;++i) {
        const auto row=std::size_t(keys[i]>>20);
        const auto dest=std::size_t(result.indptr[row]++);
        result.indices[dest]=static_cast<std::int64_t>(keys[i]&((UINT64_C(1)<<20)-1));
        std::memcpy(result.data.data()+dest,input.weights+i,sizeof(double));
    }
    std::int64_t previous=0;
    for(auto& pointer:result.indptr){const auto next=pointer;pointer=previous;previous=next;}
    // Source column-sort comparator does not impose a stable tie order.
    // Accepted weights have only one nonzero word (or all identical qNaNs).
    std::vector<std::pair<std::int64_t,double>> row_values;
    for(std::size_t row=0;row<result.rows;++row) {
        const auto first=std::size_t(result.indptr[row]),last=std::size_t(result.indptr[row+1]);
        row_values.resize(last-first);
        for(std::size_t i=first;i<last;++i)row_values[i-first]={result.indices[i],result.data[i]};
        std::sort(row_values.begin(),row_values.end(),[](const auto& a,const auto& b){return a.first<b.first;});
        for(std::size_t i=first;i<last;++i){result.indices[i]=row_values[i-first].first;result.data[i]=row_values[i-first].second;}
    }
    }
    std::int64_t nnz=0,row_end=0;
    for(std::size_t row=0;row<result.rows;++row) {
        auto i=row_end;row_end=result.indptr[row+1];
        while(i<row_end) {
            const auto column=result.indices[std::size_t(i)];
            double value=result.data[std::size_t(i++)];
            while(i<row_end&&result.indices[std::size_t(i)]==column)value+=result.data[std::size_t(i++)];
            result.indices[std::size_t(nnz)]=column;result.data[std::size_t(nnz++)]=value;
        }
        result.indptr[row+1]=nnz;
    }
    output=std::move(result);return true;
}
}
