#include <skimagexx/order_angles_golden_ratio.hpp>
#include <skimagexx/image.hpp>
#include "order_angles_golden_ratio_fenv.hpp"
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
template<class T>T read(std::istream& in){T out{};in.read(reinterpret_cast<char*>(&out),sizeof(out));if(!in)throw std::runtime_error("truncated golden SDK fixture");return out;}
struct Record {
    std::string id;unsigned type,half,swap,policy,partial,error,n,item;
    std::ptrdiff_t stride,offset;int mutation;unsigned index;
    std::vector<unsigned char> data,mutword;std::vector<std::int64_t> expected;
};
Record record(std::istream& in) {
    Record r;const auto length=read<std::uint32_t>(in);assert(length<128);r.id.resize(length);in.read(r.id.data(),length);
    r.type=read<std::uint8_t>(in);r.half=read<std::uint8_t>(in);r.swap=read<std::uint8_t>(in);r.policy=read<std::uint8_t>(in);r.partial=read<std::uint8_t>(in);r.error=read<std::uint8_t>(in);
    r.n=read<std::uint32_t>(in);r.item=read<std::uint32_t>(in);r.stride=read<std::int64_t>(in);r.offset=read<std::int64_t>(in);r.mutation=read<std::int32_t>(in);r.index=read<std::uint32_t>(in);
    assert(r.type<=10&&r.n<=16&&r.item<=8&&r.item>0&&r.policy<=1&&r.half<=1&&r.swap<=1);
    r.mutword.resize(r.item);in.read(reinterpret_cast<char*>(r.mutword.data()),r.item);
    const auto bytes=read<std::uint32_t>(in);assert(bytes<=256);r.data.resize(bytes);in.read(reinterpret_cast<char*>(r.data.data()),bytes);
    const auto count=read<std::uint32_t>(in);assert(count<=16);r.expected.resize(count);in.read(reinterpret_cast<char*>(r.expected.data()),count*8);assert(in);return r;
}
ComponentsView view(const Record& r,const std::vector<unsigned char>& bytes) {
    return {bytes.empty()?nullptr:bytes.data()+r.offset,PixelType(r.type),bool(r.half),bool(r.swap),{r.n},{r.stride}};
}
void mutate(const Record& r,std::vector<unsigned char>& bytes) {
    assert(r.index<r.n);const auto offset=r.offset+std::ptrdiff_t(r.index)*r.stride;
    assert(offset>=0&&std::size_t(offset)+r.item<=bytes.size());std::memcpy(bytes.data()+offset,r.mutword.data(),r.item);
}
int main(int argc,char** argv) {
    assert(argc==3&&sizeof(std::int64_t)==8);GoldenFPEnvelope environment(argv[2]);std::ifstream in(argv[1],std::ios::binary);char magic[8];in.read(magic,8);assert(std::memcmp(magic,"GOLDSDK1",8)==0);
    const auto count=read<std::uint32_t>(in);assert(count==36);std::vector<Record> rows;for(unsigned i=0;i<count;++i)rows.push_back(record(in));
    assert(get_num_threads()==1);std::size_t cases=0,words=0,complete_operations=0,errors=0;
    for(int budget:{0,1,4})for(const auto& r:rows) {
        auto data=r.data,expected_data=r.data;std::vector<std::int64_t> actual;
        if(r.mutation==0){mutate(r,data);mutate(r,expected_data);}
        const auto policy=r.policy?GoldenScalarPolicy::weak_input:GoldenScalarPolicy::wide;
        try {
            GoldenAngleOrder state(view(r,data),policy,budget);assert(state.worker_budget()==0);
            actual.push_back(state.begin());assert(state.worker_budget()==(budget?budget:1));
            if(!r.partial)while(true) {
                if(r.mutation==int(actual.size())){mutate(r,data);mutate(r,expected_data);}
                if(!state.resume())break;
                actual.push_back(state.next(view(r,data)));
            }
            assert(!r.error&&actual==r.expected);
        }catch(const std::out_of_range& e){assert(r.error&&std::string(e.what())=="list index out of range");++errors;}
        assert(data==expected_data);words+=actual.size();++cases;
        if(r.mutation<0&&!r.partial) {
            try{const auto result=order_angles_golden_ratio(view(r,r.data),policy,budget);assert(!r.error&&result==r.expected);}
            catch(const std::out_of_range& e){assert(r.error&&std::string(e.what())=="list index out of range");}
            ++complete_operations;
        }
    }
    assert(get_num_threads()==1);std::cout<<"SDK cases="<<cases<<" words="<<words<<" complete_operations="<<complete_operations<<" errors="<<errors<<" workers=0,1,4 default=1\n";environment.finish();
}
