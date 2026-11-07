#include "../variation_of_information_current_counts_controls.hpp"
#include "../variation_of_information_large_controls.hpp"
#include <skimagexx/variation_of_information.hpp>
#include <skimagexx/image.hpp>
#include "fixtures/variation_of_information_saved_v1.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>

int main() {
    if(int status=vi_current_counts_controls())return status;
    if(skimagexx::get_num_threads()!=1)return 2;
    unsigned checked=0;
    for(int budget:{1,4}) {
        skimagexx::set_num_threads(budget);
        // End-to-end image counts+entropy, all-ignored status, and supplied CSR.
        for(auto index:{3u,4u,6u}) {
            const auto& test=vi_saved[index];skimagexx::VariationInformation result;
            bool accepted;
            if(test.supplied) {
                const std::int64_t columns[]={1,0,1,2},rows[]={0,2,4};
                accepted=skimagexx::variation_of_information({test.data.data(),columns,rows,2,3,4},result);
            }else {
                skimagexx::detail::LabelPairInput input;
                input.truth=test.truth.data();input.test=test.test.data();input.count=test.truth.size();
                input.ignored_truth_labels=test.ignored.data();input.ignored_count=test.ignored.size();
                accepted=skimagexx::variation_of_information(input,result);
            }
            if(!accepted||result.captured_workers!=budget)return 3;
            for(unsigned i=0;i<2;++i){std::uint64_t word;std::memcpy(&word,&result.values[i],8);if(word!=test.words[i])return 4;}
            if(result.normalization_fp_status!=(index==4?1u:0u))return 5;
            ++checked;
        }
    }
    skimagexx::set_num_threads(1);
    if(int status=vi_large_saved_controls())return status;
    std::cout<<"SDK VI complete controls="<<checked<<" workers=1,4 restored="<<skimagexx::get_num_threads()
        <<" available="<<skimagexx::has_avx2()<<" originals=0 timing=0\n";
}
