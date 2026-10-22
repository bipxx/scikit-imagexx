#include <skimagexx/clahe.hpp>
#include <skimagexx/image.hpp>
#include <skimagexx/threshold.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

int main() {
    if(skimagexx::get_num_threads()!=1)throw std::runtime_error("default worker limit changed");
    std::vector<std::uint16_t> input(35);
    std::vector<double> output(input.size());
    for(std::size_t i=0;i<input.size();++i)input[i]=std::uint16_t(i*1307);
    skimagexx::clahe_into(input.data(),output.data(),false,{5,7},{3,5},1,7);
    std::uint64_t hash=1469598103934665603ULL;
    for(const auto value:output) {
        if(value!=std::clamp(value,0.,1.))throw std::runtime_error("invalid normalized output");
        std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
        for(unsigned b=0;b<8;++b){hash^=(bits>>(8*b))&255;hash*=1099511628211ULL;}
    }
    if(hash!=0x3bf05b9bc1d6a616ULL)throw std::runtime_error("CLAHE parity mismatch");
    const float counts[]={1,9,2,4,8,3,1,7,4};
    if(skimagexx::threshold_multiotsu(counts,9,3)!=std::vector<std::size_t>{2,5})
        throw std::runtime_error("multi-Otsu parity mismatch");
}
