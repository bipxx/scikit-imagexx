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
template<class T,class F> T pairwise(std::size_t n,const F& get) {
    if(n<8){T sum=-T(0);for(std::size_t i=0;i<n;++i)sum=T(sum+T(get(i)));return sum;}
    T lanes[8];for(std::size_t i=0;i<8;++i)lanes[i]=T(get(i));std::size_t i=8;
    for(;i+8<=n;i+=8)for(std::size_t j=0;j<8;++j)lanes[j]=T(lanes[j]+T(get(i+j)));
    T sum=T(T(T(lanes[0]+lanes[1])+T(lanes[2]+lanes[3]))+T(T(lanes[4]+lanes[5])+T(lanes[6]+lanes[7])));
    for(;i<n;++i)sum=T(sum+T(get(i)));
    return sum;
}
// Same 128-element tree boundary as metrics.cpp; local to the VI operation.
template<class F> double reduction_tree(std::size_t n,const F& get,std::size_t begin=0) {
    if(n<=128)return pairwise<double>(n,[&](auto i){return get(begin+i);});
    auto half=n/2;half-=half%8;
    return reduction_tree(half,get,begin)+reduction_tree(n-half,get,begin+half);
}
// ufunc.reduceat seeds from the first stored item, then reduces the tail.
template<class F> double minor_sum(std::size_t n,const F& get) {
    if(!n)return 0.;
    if(n==1)return get(0);
    return get(0)+reduction_tree(n-1,[&](auto i){return get(i+1);});
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
// Contiguous retained DDOT: four four-lane accumulators with fused products,
// high+low halves, grouped horizontal sum, then the scalar remainder. Keeping
// the arithmetic graph in scalar C++ also preserves it with SIMD disabled.
template<class L,class R> double source_dot(std::size_t n,const L& left,const R& right) {
    const auto packet=n-n%16;double result=0.;
    if(packet) {
        double lanes[16]={};
        for(std::size_t i=0;i<packet;i+=16)
            for(std::size_t j=0;j<16;++j)
                lanes[j]=std::fma(left(i+j),right(i+j),lanes[j]);
        double halves[8];
        for(std::size_t g=0;g<4;++g)
            for(std::size_t j=0;j<2;++j)halves[2*g+j]=lanes[4*g+2+j]+lanes[4*g+j];
        const double low=(halves[6]+halves[4])+(halves[2]+halves[0]);
        const double high=(halves[7]+halves[5])+(halves[3]+halves[1]);
        result=low+high;
    }
    // Retain the established small/tail multiplication operand semantics,
    // including opposite-signed quiet NaNs from all-ignored inputs.
    for(std::size_t i=packet;i<n;++i)result+=source_dot_product(left(i),right(i));
    return result;
}
// Keep each sparse multiplication and its nonzero xlogx batch in the source
// direction/order. Reuse scaled_row's multiply-before-coalescing graph and the
// admitted reduction/scatter graphs; the host supplies only elementary log2.
bool batch_entropy(VariationTableView a,const std::vector<double>& inverse,bool left,
    VariationLog2Batch provider,std::vector<double>& entropy) {
    std::vector<double> values;values.reserve(a.nnz);
    std::vector<std::size_t> columns;columns.reserve(a.nnz);
    std::vector<std::size_t> offsets(a.rows+1);
    std::vector<double> coefficients(a.columns);
    std::vector<unsigned char> seen(a.columns,0);
    for(std::size_t r=0;r<a.rows;++r) {
        auto order=scaled_row(a,r,inverse,left,coefficients,seen);
        for(auto c:order) {
            if(coefficients[c]!=0.){values.push_back(coefficients[c]);columns.push_back(c);}
            seen[c]=0;
        }
        offsets[r+1]=values.size();
    }
    std::vector<double> logarithms(values.size());
    if(!provider.function(values.data(),logarithms.data(),values.size(),provider.context))return false;
    // Original z[nz] *= log2(z[nz]): products precede either sparse reduction.
    for(std::size_t i=0;i<values.size();++i)values[i]*=logarithms[i];
    if(left) {
        for(std::size_t r=0;r<a.rows;++r)
            entropy[r]=minor_sum(offsets[r+1]-offsets[r],[&](auto i){return values[offsets[r]+i];});
    }else {
        for(std::size_t i=0;i<values.size();++i)entropy[columns[i]]+=values[i];
    }
    return true;
}
} // namespace

bool variation_of_information(VariationTableView a,VariationInformation& output,int workers) {
    const int snapshot=workers==0?get_num_threads():workers;
    if(snapshot<1||snapshot>1024||!environment()||!validate(a))return false;
    std::vector<double> px(a.rows,0.),py(a.columns,0.);
    for(std::size_t r=0;r<a.rows;++r) {
        // Source minor reduction starts with the first stored coefficient;
        // column reduction is a scatter addition into positive-zero storage.
        const auto first=a.indptr[r],last=a.indptr[r+1];
        px[r]=minor_sum(std::size_t(last-first),[&](auto i){return a.data[first+i];});
        for(auto j=first;j<last;++j)py[std::size_t(a.indices[j])]+=a.data[j];
    }
    std::vector<double> ix(px),iy(py);
    for(auto& v:ix)if(v!=0.)v=1./v;
    for(auto& v:iy)if(v!=0.)v=1./v;
    std::vector<double> row_entropy(a.rows,0.),column_entropy(a.columns,0.),coefficients(a.columns);
    std::vector<unsigned char> seen(a.columns,0);
    for(std::size_t r=0;r<a.rows;++r) {
        auto order=scaled_row(a,r,ix,true,coefficients,seen);
        std::vector<double> terms;terms.reserve(order.size());
        for(auto c:order){
            if(coefficients[c]!=0.){const double term=xlogx(coefficients[c]);
                terms.push_back(term);}
            seen[c]=0;
        }
        row_entropy[r]=minor_sum(terms.size(),[&](auto i){return terms[i];});
        order=scaled_row(a,r,iy,false,coefficients,seen);
        for(auto c:order){if(coefficients[c]!=0.)column_entropy[c]+=xlogx(coefficients[c]);seen[c]=0;}
    }
    VariationInformation result;result.captured_workers=snapshot;
    result.values[0]=source_dot(a.rows,[&](auto r){return -px[r];},[&](auto r){return row_entropy[r];});
    result.values[1]=source_dot(a.columns,[&](auto c){return -column_entropy[c];},[&](auto c){return py[c];});
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

bool variation_of_information_compatible(VariationTableView a,VariationInformation& output,
    VariationLog2Batch provider,int workers) {
    const int snapshot=workers==0?get_num_threads():workers;
    if(!provider.function||snapshot<1||snapshot>1024||!environment()||!validate(a))return false;
    std::vector<double> px(a.rows,0.),py(a.columns,0.);
    for(std::size_t r=0;r<a.rows;++r) {
        const auto first=a.indptr[r],last=a.indptr[r+1];
        px[r]=minor_sum(std::size_t(last-first),[&](auto i){return a.data[first+i];});
        for(auto j=first;j<last;++j)py[std::size_t(a.indices[j])]+=a.data[j];
    }
    std::vector<double> ix(px),iy(py);
    for(auto& v:ix)if(v!=0.)v=1./v;
    for(auto& v:iy)if(v!=0.)v=1./v;
    VariationInformation result;result.captured_workers=snapshot;
    std::vector<double> row_entropy(a.rows,0.);
    if(!batch_entropy(a,ix,true,provider,row_entropy))return false;
    result.values[0]=source_dot(a.rows,[&](auto r){return -px[r];},[&](auto r){return row_entropy[r];});
    // The first complete entropy/dot precedes the second source xlogx batch.
    std::vector<double> column_entropy(a.columns,0.);
    if(!batch_entropy(a,iy,false,provider,column_entropy))return false;
    result.values[1]=source_dot(a.columns,[&](auto c){return -column_entropy[c];},[&](auto c){return py[c];});
    output=result;return true;
}

bool variation_of_information_compatible(const detail::LabelPairInput& input,
    VariationInformation& output,VariationLog2Batch provider,int workers) {
    if(!provider.function)return false;
    const int snapshot=workers==0?get_num_threads():workers;
    auto normalized=input;normalized.normalize=true;detail::LabelPairCsr table;
    if(!detail::label_pair_counts(normalized,table,snapshot))return false;
    VariationInformation result;
    if(!variation_of_information_compatible({table.data.data(),table.indices.data(),table.indptr.data(),table.rows,table.columns,table.data.size()},result,provider,snapshot))return false;
    result.normalization_fp_status=table.normalization_fp_status;output=result;return true;
}

bool variation_of_information_compatible(VariationTableView a,VariationInformation& output,
    VariationLog2Batch logarithm,VariationInnerProduct inner_product,int workers) {
    const int snapshot=workers==0?get_num_threads():workers;
    if(!logarithm.function||!inner_product.function||snapshot<1||snapshot>1024
       ||!environment()||!validate(a))return false;
    std::vector<double> px(a.rows,0.),py(a.columns,0.);
    for(std::size_t r=0;r<a.rows;++r) {
        const auto first=a.indptr[r],last=a.indptr[r+1];
        px[r]=minor_sum(std::size_t(last-first),[&](auto i){return a.data[first+i];});
        for(auto j=first;j<last;++j)py[std::size_t(a.indices[j])]+=a.data[j];
    }
    std::vector<double> ix(px),iy(py);
    for(auto& v:ix)if(v!=0.)v=1./v;
    for(auto& v:iy)if(v!=0.)v=1./v;
    VariationInformation result;result.captured_workers=snapshot;
    std::vector<double> row_entropy(a.rows,0.);
    if(!batch_entropy(a,ix,true,logarithm,row_entropy))return false;
    for(auto& v:px)v=-v;
    if(!inner_product.function(px.data(),row_entropy.data(),a.rows,
        &result.values[0],inner_product.context))return false;
    // Preserve the original two operator directions and their source order.
    std::vector<double> column_entropy(a.columns,0.);
    if(!batch_entropy(a,iy,false,logarithm,column_entropy))return false;
    for(auto& v:column_entropy)v=-v;
    if(!inner_product.function(column_entropy.data(),py.data(),a.columns,
        &result.values[1],inner_product.context))return false;
    output=result;return true;
}

bool variation_of_information_compatible(const detail::LabelPairInput& input,
    VariationInformation& output,VariationLog2Batch logarithm,
    VariationInnerProduct inner_product,int workers) {
    if(!logarithm.function||!inner_product.function)return false;
    const int snapshot=workers==0?get_num_threads():workers;
    auto normalized=input;normalized.normalize=true;detail::LabelPairCsr table;
    if(!detail::label_pair_counts(normalized,table,snapshot))return false;
    VariationInformation result;
    if(!variation_of_information_compatible({table.data.data(),table.indices.data(),table.indptr.data(),
        table.rows,table.columns,table.data.size()},result,logarithm,inner_product,snapshot))return false;
    result.normalization_fp_status=table.normalization_fp_status;output=result;return true;
}
} // namespace skimagexx
