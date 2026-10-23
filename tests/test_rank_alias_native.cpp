#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::uint64_t hash(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t result=14695981039346656037ull;
    for(auto byte:bytes){result^=byte;result*=1099511628211ull;}
    return result;
}
// Sealed0.26 safe population-drift anchors, independently replayed in the
// accompanying Python contract before the original was invoked.
const std::uint64_t expected[4][2]={
    {1039442944648922692ull, 15203337413541299381ull},
    {14705214769023751416ull, 14473765702329981551ull},
    {13092704029666740513ull, 17034498599818728572ull},
    {3170940303037936604ull, 7285866218590421460ull}
};
std::vector<std::uint8_t> drift(int operation,bool volume,int workers) {
    const std::size_t planes=volume ? 2:1,count=planes*49,item=operation==8 ? 8:1;
    std::vector<std::uint8_t> owner(item*(count+1)+32,0x5a),image(count),footprint((volume ? 3:1)*9,1);
    for(std::size_t i=0;i<count;++i){owner[i]=0;image[i]=std::uint8_t((i*7+3)%23);}
    owner[0]=owner[1]=owner[7]=1;
    skimagexx::RankView view;view.image=image.data();view.mask=owner.data();view.output=owner.data()+item;view.footprint=footprint.data();
    view.shape={planes,7,7};view.footprint_shape={volume ? 3u:1u,3,3};view.center={volume ? 1:0,1,1};
    view.output_strides={std::ptrdiff_t(49*item),std::ptrdiff_t(7*item),std::ptrdiff_t(item)};
    view.output_type=operation==8 ? skimagexx::PixelType::float64:skimagexx::PixelType::uint8;
    const auto aliases=skimagexx::rank_aliases(view);assert(!aliases.image_output && aliases.mask_output);
    skimagexx::set_num_threads(workers);
    skimagexx::rank_filter_into(view,skimagexx::RankOperation(operation),aliases);
    assert(hash(owner)==expected[operation-5][volume ? 1:0]);
    return owner;
}
void admission() {
    std::vector<std::uint8_t> owner(128,0x5a);std::uint8_t footprint=1;
    skimagexx::RankView view;view.image=owner.data()+4;view.output=owner.data();view.footprint=&footprint;
    view.shape={1,1,2};view.footprint_shape={1,1,1};view.output_strides={16,16,8};
    owner[4]=3;owner[5]=7;
    auto alias=skimagexx::rank_aliases(view);assert(!alias.image_output && !alias.mask_output);
    skimagexx::rank_filter_into(view,skimagexx::RankOperation::median,alias);
    assert(owner[0]==3 && owner[8]==7);
    bool rejected=false;try{skimagexx::rank_filter_into(view,skimagexx::RankOperation::median,{true,false});}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
    view.output=owner.data()+4;alias=skimagexx::rank_aliases(view);assert(alias.image_output);
    const auto before=owner;
    rejected=false;try{skimagexx::rank_filter_into(view,skimagexx::RankOperation::entropy,alias);}catch(const std::invalid_argument&){rejected=true;}assert(rejected && owner==before);
    view.serial_alias=true;
    rejected=false;try{skimagexx::rank_filter_into(view,skimagexx::RankOperation::median);}catch(const std::invalid_argument&){rejected=true;}assert(rejected && owner==before);
}
}
int main() {
    assert(skimagexx::rank_histogram_capabilities()==8191);
    assert(skimagexx::rank_alias_capabilities()==0x1e0);
    for(int op=5;op<=8;++op)for(bool volume:{false,true}) {
        const auto one=drift(op,volume,1);
#ifdef _OPENMP
        const auto four=drift(op,volume,4);assert(one==four);
#endif
    }
    admission();
    std::cout<<"rank alias native: signed feedback, no-store, exact descriptor and bounds admission passed\n";
}
