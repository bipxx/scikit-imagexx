#include "../variation_of_information_current_counts_controls.hpp"
#include "../variation_of_information_large_controls.hpp"
#include <skimagexx/variation_of_information.hpp>
#include <skimagexx/image.hpp>
#include "fixtures/variation_of_information_saved_v1.hpp"
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
// VI_SDK_EXTENSION_INCLUDE
#include "../variation_of_information_provider_controls.hpp"

// Explicit host callback plumbing. This provider is scalar std::log2 and is
// not evidence of NumPy/SVML semantics; public Python provider evidence is separate.
struct ViLog2Audit {unsigned calls=0;unsigned empty=0;unsigned fail_on=0;};
bool sdk_log2(const double* values,double* logs,std::size_t count,void* data) {
    auto& audit=*static_cast<ViLog2Audit*>(data);++audit.calls;
    if(audit.calls==audit.fail_on)return false;
    if(!count)++audit.empty;
    for(std::size_t i=0;i<count;++i)logs[i]=std::log2(values[i]);
    return true;
}
int provider_controls() {
    const std::int64_t columns[]={1,1,0,1},rows[]={0,3,4};
    const double values[]={0.,.25,.25,.5};
    const skimagexx::VariationTableView table{values,columns,rows,2,2,4};
    skimagexx::VariationInformation baseline,result;
    ViLog2Audit audit;
    if(!skimagexx::variation_of_information(table,baseline,1))return 20;
    if(!skimagexx::variation_of_information_compatible(table,result,{sdk_log2,&audit},1))return 21;
    if(audit.calls!=2||std::memcmp(baseline.values.data(),result.values.data(),16))return 22;
    result.values={123.,456.};audit={0,0,2};
    if(skimagexx::variation_of_information_compatible(table,result,{sdk_log2,&audit},1))return 23;
    if(audit.calls!=2||result.values[0]!=123.||result.values[1]!=456.)return 24;
    const std::int64_t empty_rows[]={0,0,0};audit={};
    if(!skimagexx::variation_of_information_compatible({nullptr,nullptr,empty_rows,2,2,0},result,{sdk_log2,&audit},0))return 25;
    if(audit.calls!=2||audit.empty!=2||result.captured_workers!=1)return 26;
    return 0;
}

int vi_sdk_owning_main() {
    // VI_SDK_EXTENSION_ENTRY_BEGIN
    if(!vi_sdk_controls::layout())return 90;
    vi_sdk_controls::Caller32 process("process");if(!process.good())return 91;
    // VI_SDK_EXTENSION_ENTRY_END
    if(int status=vi_current_counts_controls())return status;
    if(int status=provider_controls())return status;
    unsigned provider_checked=0;
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
            ViLog2Audit audit;skimagexx::VariationInformation compatible;
            if(test.supplied) {
                const std::int64_t columns[]={1,0,1,2},rows[]={0,2,4};
                accepted=skimagexx::variation_of_information_compatible({test.data.data(),columns,rows,2,3,4},compatible,{sdk_log2,&audit});
            }else {
                skimagexx::detail::LabelPairInput input;
                input.truth=test.truth.data();input.test=test.test.data();input.count=test.truth.size();
                input.ignored_truth_labels=test.ignored.data();input.ignored_count=test.ignored.size();
                accepted=skimagexx::variation_of_information_compatible(input,compatible,{sdk_log2,&audit});
            }
            if(!accepted||audit.calls!=2||compatible.captured_workers!=budget)return 27;
            if(std::memcmp(result.values.data(),compatible.values.data(),16)
               ||compatible.normalization_fp_status!=result.normalization_fp_status)return 28;
            ++provider_checked;
            for(unsigned i=0;i<2;++i){std::uint64_t word;std::memcpy(&word,&result.values[i],8);if(word!=test.words[i])return 4;}
            if(result.normalization_fp_status!=(index==4?1u:0u))return 5;
            ++checked;
        }
    }
    skimagexx::set_num_threads(1);
    if(int status=vi_large_saved_controls())return status;
    std::cout<<"SDK VI complete controls="<<checked<<" workers=1,4 restored="<<skimagexx::get_num_threads()
        <<" host_scalar_log2_compatible="<<provider_checked<<" provider_plumbing_controls=3"
        <<" available="<<skimagexx::has_avx2()<<" originals=0 timing=0\n";
    // VI_SDK_EXTENSION_TAIL_BEGIN
    if(int status=vi_sdk_controls::run())return status;
    if(!process.finish())return 92;
    std::cout<<"VI SDK extended compatiblecalls="<<vi_sdk_controls::counters.owning
        <<" default_FENV_refusals="<<vi_sdk_controls::counters.default_refusals
        <<" callbacks="<<vi_sdk_controls::counters.callbacks
        <<" caller32checks="<<vi_sdk_controls::counters.checked32
        <<" restores="<<vi_sdk_controls::counters.restores32<<" reads="<<vi_sdk_controls::counters.reads32
        <<" defaultworkers="<<skimagexx::get_num_threads()<<" originals=0 timing=0\n";
    return 0;
    // VI_SDK_EXTENSION_TAIL_END
}

// VI_SDK_EXTENSION_MAIN_BEGIN
int main() {
    try{return vi_sdk_owning_main();}
    catch(const std::exception& error){std::cerr<<"VI SDK primary: "<<error.what()<<'\n';return 1;}
    catch(...){std::cerr<<"VI SDK primary: nonstandard exception\n";return 1;}
}
// VI_SDK_EXTENSION_MAIN_END
