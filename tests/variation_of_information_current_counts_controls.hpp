#pragma once
#include <skimagexx/variation_of_information.hpp>
#include <skimagexx/image.hpp>
#include <cstring>
#include <iostream>
#include <vector>

// Candidate equivalence only: no new original reference or broad accuracy claim.
inline int vi_current_counts_controls() {
    using namespace skimagexx;
    using namespace skimagexx::detail;
    if(get_num_threads()!=1)return 1;
    unsigned checked=0;
    auto equal=[](double a,double b){return std::memcmp(&a,&b,8)==0;};
    for(const std::size_t n:{4096u,4099u})for(int ignored_mode:{0,1,2}) {
        std::vector<std::int64_t> a(n),b(n),wide_a(n),wide_b(n);
        for(std::size_t i=0;i<n;++i){a[i]=(i/7)%4;b[i]=(i/3)%8;wide_a[i]=a[i]+64;wide_b[i]=b[i]+64;}
        // 4x8<=N/4 enters density gate; 68x72>N/4 forces packed-key fallback.
        if(!(32<=n/4 && 68*72>n/4))return 2;
        const std::int64_t labels[]={0,1,2,3},wide_labels[]={64,65,66,67};
        const auto ignored=ignored_mode==2?4u:ignored_mode==1?1u:0u;
        for(int workers:{1,4,0}) {
            set_num_threads(workers==0?4:workers);
            LabelPairInput in{a.data(),b.data(),n,labels,ignored,true};
            LabelPairInput wide{wide_a.data(),wide_b.data(),n,wide_labels,ignored,true};
            LabelPairCsr dense,sparse;
            if(!label_pair_counts(in,dense,workers)||!label_pair_counts(wide,sparse,workers))return 3;
            if(dense.rows!=4||dense.columns!=8||sparse.rows!=68||sparse.columns!=72
               ||dense.data.size()!=32||sparse.data.size()!=32||dense.indices.size()!=32)return 4;
            for(std::size_t r=0;r<dense.rows;++r) {
                if(dense.indptr[r]!=sparse.indptr[r+64])return 5;
                for(auto j=dense.indptr[r];j<dense.indptr[r+1];++j)
                    if(dense.indices[j]+64!=sparse.indices[j]||!equal(dense.data[j],sparse.data[j]))return 6;
            }
            if(dense.normalization_fp_status!=sparse.normalization_fp_status)return 7;
            VariationInformation x,y,z;
            if(!variation_of_information(in,x,workers)||!variation_of_information(wide,y,workers)
               ||!variation_of_information({dense.data.data(),dense.indices.data(),dense.indptr.data(),4,8,32},z,workers))return 8;
            const int captured=workers==0?4:workers;
            if(x.captured_workers!=captured||y.captured_workers!=captured||z.captured_workers!=captured)return 9;
            for(unsigned j=0;j<2;++j)if(!equal(x.values[j],y.values[j])||!equal(x.values[j],z.values[j]))return 10;
            // Positive explicit budget must override the global value, not resnapshot it.
            set_num_threads(4);VariationInformation explicit_one;
            if(!variation_of_information(in,explicit_one,1)||explicit_one.captured_workers!=1||get_num_threads()!=4)return 11;
            std::size_t kept=0;for(auto v:a)if(ignored_mode==0||(ignored_mode==1&&v!=0))++kept;
            std::vector<double> weights(n);
            // Test setup uses the count primitive's already-owned row coefficient,
            // not another original/NumPy arithmetic implementation or expected gold.
            for(std::size_t i=0;i<n;++i) {
                if(ignored_mode==2)weights[i]=dense.data[0];
                else weights[i]=(ignored_mode==1&&a[i]==0)?0.:1./static_cast<double>(kept);
            }
            LabelPairCsrStorage storage;
            if(!label_pair_counts_preweighted({a.data(),b.data(),weights.data(),n},storage,workers))return 12;
            const auto logical=static_cast<std::size_t>(storage.indptr.back());
            if(storage.data.size()!=n||storage.indices.size()!=n||logical!=32||logical>=storage.data.size())return 13;
            VariationInformation logical_result;
            if(!variation_of_information({storage.data.data(),storage.indices.data(),storage.indptr.data(),4,8,logical},logical_result,workers))return 14;
            for(unsigned j=0;j<2;++j)if(!equal(logical_result.values[j],x.values[j]))return 15;
            VariationInformation untouched;untouched.values={123.,456.};
            if(variation_of_information({storage.data.data(),storage.indices.data(),storage.indptr.data(),4,8,n},untouched,workers)
               ||untouched.values[0]!=123.||untouched.values[1]!=456.)return 16;
            ++checked;
        }
    }
    set_num_threads(1);
    std::cout<<"VI current counts candidate-equivalence controls="<<checked
             <<" logicalNNZ32 physicalN4096or4099 workers0,1,4 restored="<<get_num_threads()<<" originals=0 timing=0\n";
    return 0;
}
