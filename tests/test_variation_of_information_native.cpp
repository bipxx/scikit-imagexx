#include "fixtures/variation_of_information_saved_v1.hpp"
#include <skimagexx/image.hpp>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

int main() {
    // One smallest mathematical viability barrier BEFORE binding/SDK campaigns.
    // Strict raw words, including signs of zero/NaN; never an ULP tolerance.
    if(skimagexx::get_num_threads()!=1||std::fegetround()!=FE_TONEAREST)return 2;
    std::size_t failures=0,records=0,controls=0;
    for(int workers:{0,1,4}) {
    for(const auto& test:vi_saved) {
        skimagexx::VariationInformation actual;bool accepted;
        if(test.supplied) {
            const std::int64_t indices[]={1,0,1,2},indptr[]={0,2,4};
            accepted=skimagexx::variation_of_information({test.data.data(),indices,indptr,2,3,4},actual,workers);
        } else {
            skimagexx::detail::LabelPairInput input;
            input.truth=test.truth.data();input.test=test.test.data();input.count=test.truth.size();
            input.ignored_truth_labels=test.ignored.data();input.ignored_count=test.ignored.size();
            accepted=skimagexx::variation_of_information(input,actual,workers);
        }
        std::cout<<test.id<<" accepted="<<accepted;
        bool equal=accepted&&actual.captured_workers==(workers?workers:1);
        for(std::size_t i=0;i<2;++i) {
            std::uint64_t word;std::memcpy(&word,&actual.values[i],8);
            std::cout<<" actual"<<i<<"="<<std::hex<<std::setw(16)<<std::setfill('0')<<word
                <<" expected"<<i<<"="<<std::setw(16)<<test.words[i]<<std::dec;
            equal=equal&&word==test.words[i];
        }
        failures+=!equal;++records;std::cout<<" strict_equal="<<equal<<'\n';
    }
    }
    const std::int64_t indices[]={0,1,0,1},indptr[]={0,2,4};
    const double coefficients[]={1.,1.,1.,1.};
    skimagexx::VariationTableView view{coefficients,indices,indptr,2,2,4};
    skimagexx::VariationInformation out;
    skimagexx::set_num_threads(4);
    const bool parallel=skimagexx::variation_of_information(view,out);
    skimagexx::set_num_threads(1);
    failures+=!(parallel&&out.captured_workers==4&&out.values[0]==4.&&out.values[1]==4.);++controls;
    for(int fault=0;fault<7;++fault) {
        auto bad=view;double value;std::uint64_t word=0;
        if(fault==0)bad.rows=0;
        else if(fault==1)bad.columns=0;
        else if(fault==2)bad.nnz=3;
        else if(fault==3)bad.data=nullptr;
        else {
            word=fault==4?UINT64_C(0xbff0000000000000):fault==5?UINT64_C(0x7ff0000000000000):UINT64_C(0x7ff0000000000001);
            std::memcpy(&value,&word,8);bad.data=&value;bad.nnz=1;
            static const std::int64_t oneptr[]={0,1};bad.rows=bad.columns=1;bad.indptr=oneptr;
        }
        skimagexx::VariationInformation sentinel;sentinel.values={123.,456.};
        const int flags=std::fetestexcept(FE_ALL_EXCEPT);
        const bool refused=!skimagexx::variation_of_information(bad,sentinel,1);
        failures+=!(refused&&sentinel.values[0]==123.&&sentinel.values[1]==456.&&std::fetestexcept(FE_ALL_EXCEPT)==flags);++controls;
    }
    const auto rounding=std::fegetround();std::fesetround(FE_DOWNWARD);
    const bool rounded=skimagexx::variation_of_information(view,out,1);
    std::fesetround(rounding);failures+=rounded;++controls;
    std::cout<<"saved="<<records<<" controls="<<controls<<" failures="<<failures<<" workers=0,1,4 restored="<<skimagexx::get_num_threads()<<" originals=0 timed=0\n";
    return failures?1:0;
}
