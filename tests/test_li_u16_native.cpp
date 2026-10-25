#include "skimagexx/li.hpp"
#include "skimagexx/image.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {
bool same(double a,double b) {return std::memcmp(&a,&b,sizeof(a))==0;}
void compare(const std::vector<std::uint8_t>& bytes) {
    const std::vector<std::uint16_t> wide(bytes.begin(),bytes.end());
    std::vector<unsigned char> storage(wide.size()*sizeof(std::uint16_t)+1);
    std::memcpy(storage.data()+1,wide.data(),wide.size()*sizeof(std::uint16_t));
    const skimagexx::LiImage old(bytes.data(),bytes.size(),skimagexx::PixelType::uint8);
    const skimagexx::LiImage added(wide.data(),wide.size(),skimagexx::PixelType::uint16);
    const skimagexx::LiImage unaligned(storage.data()+1,wide.size(),skimagexx::PixelType::uint16);
    assert(added.kind()==old.kind());
    assert(unaligned.kind()==old.kind());
    assert(same(added.value(),old.value()));
    assert(same(unaligned.value(),old.value()));
    assert(same(added.minimum(),old.minimum()));
    assert(same(unaligned.minimum(),old.minimum()));
    assert(same(added.maximum(),old.maximum()));
    assert(same(unaligned.maximum(),old.maximum()));
    if(added.kind()==skimagexx::LiKind::ready) {
        assert(added.spacing()==1. && old.spacing()==1. && unaligned.spacing()==1.);
        for(const double tolerance:{.5,.125,std::numeric_limits<double>::infinity()}) {
            const auto a=added.run(tolerance),b=old.run(tolerance),c=unaligned.run(tolerance);
            assert(same(a.value,b.value) && a.updated==b.updated && a.status==b.status);
            assert(same(c.value,b.value) && c.updated==b.updated && c.status==b.status);
        }
    }
}
}

int main() {
    using namespace skimagexx;
    assert(get_num_threads()==1);
    // The size guard must reject before dereferencing even one element.
    const std::uint16_t singleton=7;
    const LiImage too_large(&singleton,std::size_t(2147483648ULL),PixelType::uint16);
    assert(too_large.kind()==LiKind::unsupported);
    const LiImage empty(nullptr,0,PixelType::uint16);
    assert(empty.kind()==LiKind::empty);
    for(const auto value:{256,65535}) {
        const std::uint16_t constant=std::uint16_t(value);
        const LiImage outside(&constant,1,PixelType::uint16);
        assert(outside.kind()==LiKind::unsupported);
        const std::uint16_t mixed[]={0,1,std::uint16_t(value),255};
        assert(LiImage(mixed,4,PixelType::uint16).kind()==LiKind::unsupported);
    }
    for(const int workers:{1,4}) {
        set_num_threads(workers);
        for(unsigned value=0;value<256;++value)compare(std::vector<std::uint8_t>(9,std::uint8_t(value)));
        compare({0,0,1,2,3,4,4,5,7,9,11,12});
        compare({0,1,2,254,255});
        compare({10,10,11,12,13,14,14,15,17,19,21,22});
        // Exceeds the existing histogram's real global-worker threshold.
        std::vector<std::uint8_t> large(262151);
        for(std::size_t i=0;i<large.size();++i)large[i]=std::uint8_t(i*37);
        compare(large);
    }
    set_num_threads(1);
    std::cout<<"Li uint16 bounded range, size-before-read, constants, unaligned storage, exact byte-solver reuse and workers1/4 passed; AVX2="<<has_avx2()<<'\n';
}
