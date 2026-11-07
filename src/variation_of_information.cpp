// Source order: scikit-image0.26 metrics/_variation_of_information.py
// BSD-3-Clause; see third_party/scikit-image-LICENSE.txt.
#include <skimagexx/variation_of_information.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <vector>
#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace skimagexx {
namespace {
bool environment() {
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64)
    const auto control=_mm_getcsr();
    if(control&((3u<<13)|(1u<<15)|(1u<<6))||(control&0x1f80u)!=0x1f80u)return false;
#endif
#if defined(__GNUC__) && defined(__x86_64__)
    unsigned short control87;
    __asm__ volatile("fnstcw %0" : "=m"(control87));
    if((control87&0x3fu)!=0x3fu)return false;
#endif
    return true;
}
bool validate(VariationTableView a) {
    if(!a.rows||!a.columns||a.rows>detail::label_pair_max_samples
       ||a.columns>detail::label_pair_max_samples||a.nnz>detail::label_pair_max_samples
       ||!a.indptr||(a.nnz&&(!a.data||!a.indices))||a.indptr[0]!=0
       ||a.indptr[a.rows]!=static_cast<std::int64_t>(a.nnz))return false;
    for(std::size_t r=0;r<a.rows;++r)
        if(a.indptr[r]<0||a.indptr[r]>a.indptr[r+1])return false;
    for(std::size_t j=0;j<a.nnz;++j) {
        if(a.indices[j]<0||std::size_t(a.indices[j])>=a.columns)return false;
        // Classify before arithmetic: signaling NaNs must refuse without
        // raising invalid, while both signs of quiet source NaNs are retained.
        std::uint64_t bits;std::memcpy(&bits,a.data+j,8);
        const auto magnitude=bits&UINT64_C(0x7fffffffffffffff);
        if((magnitude&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000)) {
            if(!(magnitude&UINT64_C(0x000fffffffffffff))||!(magnitude&UINT64_C(0x0008000000000000)))return false;
        }else if((bits>>63)&&magnitude)return false;
    }
    return true;
}
// The diagonal-times-CSR and CSR-times-diagonal products insert each newly
// encountered column at the sparse product's head, then emit in reverse first
// encounter order. Accumulate duplicates after each coefficient multiplication.
// This is specific VI row scaling, not a generic sparse algebra engine.
std::vector<std::size_t> scaled_row(VariationTableView a,std::size_t r,
    const std::vector<double>& inverse,bool left,std::vector<double>& coefficients,
    std::vector<unsigned char>& seen) {
    std::vector<std::size_t> order;
    for(auto j=a.indptr[r];j<a.indptr[r+1];++j) {
        const auto c=std::size_t(a.indices[j]);
        const double factor=left?inverse[r]:inverse[c];
        if(factor==0.)continue;
        if(!seen[c]){seen[c]=1;coefficients[c]=0.;order.push_back(c);}
        // Explicit source binary operation; FMA contraction is forbidden.
        coefficients[c]+=left?factor*a.data[j]:a.data[j]*factor;
    }
    std::reverse(order.begin(),order.end());return order;
}
double xlogx(double value){return value==0.?0.:value*std::log2(value);}
double source_dot_product(double left,double right) {
#if defined(__GNUC__) && defined(__x86_64__)
    // Retained NumPy/OpenBLAS small contiguous dot loads right, then multiplies
    // left into that destination. This operand order is observable when both
    // operands are quiet NaNs of opposite sign. C++ multiplication alone lets
    // the compiler commute them even without fast-math. Use the source backend
    // instruction for every coefficient, without any NaN/case/result branch.
    __asm__ volatile("mulsd %1,%0" : "+x"(right) : "x"(left));
    return right;
#else
    // Finite arithmetic proposal only; exact nonfinite backend/platform parity
    // is not established by the retained Linux reference records.
    return right*left;
#endif
}
}

bool variation_of_information(VariationTableView a,VariationInformation& output,int workers) {
    const int snapshot=workers==0?get_num_threads():workers;
    if(snapshot<1||snapshot>1024||!environment()||!validate(a))return false;
    std::vector<double> px(a.rows,0.),py(a.columns,0.);
    for(std::size_t r=0;r<a.rows;++r) {
        // Source minor reduction starts with the first stored coefficient;
        // column reduction is a scatter addition into positive-zero storage.
        const auto first=a.indptr[r],last=a.indptr[r+1];
        if(first<last){px[r]=a.data[first];for(auto j=first+1;j<last;++j)px[r]+=a.data[j];}
        for(auto j=first;j<last;++j)py[std::size_t(a.indices[j])]+=a.data[j];
    }
    std::vector<double> ix(px),iy(py);
    for(auto& v:ix)if(v!=0.)v=1./v;
    for(auto& v:iy)if(v!=0.)v=1./v;
    std::vector<double> row_entropy(a.rows,0.),column_entropy(a.columns,0.),coefficients(a.columns);
    std::vector<unsigned char> seen(a.columns,0);
    for(std::size_t r=0;r<a.rows;++r) {
        auto order=scaled_row(a,r,ix,true,coefficients,seen);
        bool first=true;
        for(auto c:order){
            if(coefficients[c]!=0.){const double term=xlogx(coefficients[c]);
                if(first){row_entropy[r]=term;first=false;}else row_entropy[r]+=term;}
            seen[c]=0;
        }
        order=scaled_row(a,r,iy,false,coefficients,seen);
        for(auto c:order){if(coefficients[c]!=0.)column_entropy[c]+=xlogx(coefficients[c]);seen[c]=0;}
    }
    VariationInformation result;result.captured_workers=snapshot;
    for(std::size_t r=0;r<a.rows;++r)result.values[0]+=source_dot_product(-px[r],row_entropy[r]);
    for(std::size_t c=0;c<a.columns;++c)result.values[1]+=source_dot_product(-column_entropy[c],py[c]);
    output=result;return true;
}

bool variation_of_information(const detail::LabelPairInput& input,VariationInformation& output,int workers) {
    const int snapshot=workers==0?get_num_threads():workers;
    auto normalized=input;normalized.normalize=true;detail::LabelPairCsr table;
    if(!detail::label_pair_counts(normalized,table,snapshot))return false;
    VariationInformation result;
    if(!variation_of_information({table.data.data(),table.indices.data(),table.indptr.data(),table.rows,table.columns,table.data.size()},result,snapshot))return false;
    result.normalization_fp_status=table.normalization_fp_status;output=result;return true;
}
} // namespace skimagexx
