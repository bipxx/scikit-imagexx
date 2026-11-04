#include <skimagexx/regionprops_state.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0,records=0;
void require(bool ok){++checks;if(!ok)throw std::runtime_error("RegionProperties native control failed");}
template<class T>T read(std::ifstream& stream){T v{};stream.read(reinterpret_cast<char*>(&v),sizeof(v));require(bool(stream));return v;}
std::vector<std::uint8_t> bytes(std::ifstream& stream){const auto n=read<std::uint64_t>(stream);std::vector<std::uint8_t> v(n);stream.read(reinterpret_cast<char*>(v.data()),std::streamsize(n));require(bool(stream));return v;}
std::vector<std::intptr_t> coords(std::ifstream& stream){const auto n=read<std::uint64_t>(stream);std::vector<std::intptr_t> v(n);for(auto& x:v)x=std::intptr_t(read<std::int64_t>(stream));return v;}
std::array<std::size_t,3> position(std::size_t i,const std::vector<std::size_t>& shape){std::array<std::size_t,3> p{};for(auto a=shape.size();a-->0;){p[a]=i%shape[a];i/=shape[a];}return p;}
std::ptrdiff_t offset(const ComponentsView& v,const std::array<std::size_t,3>& p){std::ptrdiff_t n=0;for(std::size_t a=0;a<v.shape.size();++a)n+=std::ptrdiff_t(p[a])*v.strides[a];return n;}
void check_coords(const RegionPropsState& state,std::size_t r,const std::vector<std::uint8_t>& mask,const std::vector<std::intptr_t>& expected) {
    const auto n=regionprops_mask_population(mask.data(),mask.size());
    require(expected.size()==n*state.labels.shape.size());
    constexpr std::intptr_t guard=918273;
    std::vector<std::intptr_t> out(expected.size()+2,guard);
    regionprops_coords_into(state,r,mask.data(),out.data()+1);
    require(out.front()==guard && out.back()==guard);
    require(std::equal(expected.begin(),expected.end(),out.begin()+1));
}
std::vector<std::uint8_t> check_mask(const RegionPropsState& state,std::size_t r,const std::vector<std::uint8_t>& expected) {
    std::vector<std::uint8_t> guarded(expected.size()+6,91);
    const auto count=regionprops_mask_into(state,r,guarded.data()+3);
    require(std::all_of(guarded.begin(),guarded.begin()+3,[](auto v){return v==91;}));
    require(std::all_of(guarded.end()-3,guarded.end(),[](auto v){return v==91;}));
    require(std::equal(expected.begin(),expected.end(),guarded.begin()+3));
    require(count==regionprops_mask_population(expected.data(),expected.size()));
    return {guarded.begin()+3,guarded.end()-3};
}
void frozen(const char* path,int budget) {
    std::ifstream stream(path,std::ios::binary);require(bool(stream));char magic[8];stream.read(magic,8);require(std::string(magic,8)=="SXXRPS01");
    const auto number=read<std::uint64_t>(stream);
    for(std::uint64_t k=0;k<number;++k) {
        const auto ndim=read<std::uint64_t>(stream),width=read<std::uint64_t>(stream);
        ComponentsView view;view.type=PixelType(read<std::uint64_t>(stream));view.shape.resize(ndim);view.strides.resize(ndim);
        for(auto& n:view.shape)n=std::size_t(read<std::uint64_t>(stream));
        for(auto& n:view.strides)n=std::ptrdiff_t(read<std::int64_t>(stream));
        const auto logical=bytes(stream);std::size_t count=1;std::ptrdiff_t low=0,high=0;
        for(std::size_t a=0;a<ndim;++a){count*=view.shape[a];const auto span=std::ptrdiff_t(view.shape[a]-1)*view.strides[a];if(span<0)low+=span;else high+=span;}
        require(logical.size()==count*width);
        std::vector<std::uint8_t> storage(std::size_t(high-low)+width+34,137);view.data=storage.data()+17-low;
        for(std::size_t i=0;i<count;++i)std::memcpy(const_cast<char*>(static_cast<const char*>(view.data))+offset(view,position(i,view.shape)),logical.data()+i*width,width);
        auto expected_storage=storage;
        set_num_threads(budget);RegionPropsState state;require(prepare_regionprops_state(view,state));require(state.workers==budget && state.labels.data==view.data);
        // Later global changes must not replace this call's captured policy.
        set_num_threads(budget==1?4:1);
        const auto nregions=read<std::uint64_t>(stream);require(state.regions.size()==nregions);
        std::uint64_t previous=0;
        for(std::size_t r=0;r<nregions;++r) {
            const auto label=read<std::uint64_t>(stream);require(label>previous);previous=label;
            std::array<std::size_t,3> start{},stop{};
            for(std::size_t a=0;a<ndim;++a)start[a]=std::size_t(read<std::uint64_t>(stream));
            for(std::size_t a=0;a<ndim;++a)stop[a]=std::size_t(read<std::uint64_t>(stream));
            const auto population=read<std::uint64_t>(stream);const auto image=bytes(stream);const auto expected=coords(stream);
            const auto& bounds=state.regions[r];require(bounds.label==label && bounds.start==start && bounds.stop==stop && bounds.initial_population==population);
            auto mask=check_mask(state,r,image);check_coords(state,r,mask,expected);
            const auto mutate=read<std::uint64_t>(stream);
            if(mutate) {
                std::array<std::size_t,3> point{};for(std::size_t a=0;a<ndim;++a)point[a]=std::size_t(read<std::uint64_t>(stream));
                std::memset(const_cast<char*>(static_cast<const char*>(view.data))+offset(view,point),0,width);
                std::memset(expected_storage.data()+17-low+offset(view,point),0,width);
                // A retained/cached mask remains independently useful after input mutation.
                check_coords(state,r,mask,expected);
                const auto fresh=bytes(stream);const auto fresh_coords=coords(stream);
                check_coords(state,r,check_mask(state,r,fresh),fresh_coords);
                require(state.regions[r].start==start && state.regions[r].stop==stop && state.regions[r].initial_population==population);
                require(regionprops_mask_population(fresh.data(),fresh.size())+1==population);
            }
            // Publicly modified cached masks must allocate using their CURRENT population.
            const auto first=std::find(mask.begin(),mask.end(),std::uint8_t(1));
            if(first!=mask.end()) {
                *first=0;auto fewer=expected;fewer.erase(fewer.begin(),fewer.begin()+std::ptrdiff_t(ndim));check_coords(state,r,mask,fewer);
            }
        }
        require(storage==expected_storage);
        require(std::all_of(storage.begin(),storage.begin()+17,[](auto v){return v==137;}));
        require(std::all_of(storage.end()-17,storage.end(),[](auto v){return v==137;}));
        ++records;
    }
    require(stream.peek()==std::char_traits<char>::eof());
}
void controls() {
    std::int16_t labels[]={0,1,3,1};ComponentsView input{labels,PixelType::int16,false,false,{2,2},{4,2}};
    RegionPropsState state;require(prepare_regionprops_state(input,state));const auto original=state.labels.data;
    for(auto type:{PixelType::boolean,PixelType::float32,PixelType::float64}){auto bad=input;bad.type=type;require(!prepare_regionprops_state(bad,state));require(state.labels.data==original);}
    for(int kind=0;kind<7;++kind){auto bad=input;if(kind==0)bad.swap=true;if(kind==1)bad.half=true;if(kind==2)bad.data=nullptr;if(kind==3)bad.shape={0,2};if(kind==4)bad.strides={2};if(kind==5)bad.strides[0]=std::numeric_limits<std::ptrdiff_t>::min();if(kind==6)bad.shape={std::size_t(PTRDIFF_MAX),2};require(!prepare_regionprops_state(bad,state));require(state.labels.data==original);}
    for(int budget:{1,4}) {
        std::vector<std::int16_t> large(4*65537,1);ComponentsView v{large.data(),PixelType::int16,false,false,{4,65537},{65537*2,2}};
        set_num_threads(budget);RegionPropsState prepared;require(prepare_regionprops_state(v,prepared));require(prepared.workers==budget);
        std::vector<std::uint8_t> mask(large.size());require(regionprops_mask_into(prepared,0,mask.data())==large.size());
        std::vector<std::intptr_t> output(2*large.size());regionprops_coords_into(prepared,0,mask.data(),output.data());
        require(output[0]==0 && output[1]==0 && output[output.size()-2]==3 && output.back()==65536);
        require(std::all_of(mask.begin(),mask.end(),[](auto value){return value==1;}));
    }
}
}
int main(int argc,char** argv) {
    require(argc==2);require(sizeof(std::intptr_t)==8);
    for(int budget:{1,4})frozen(argv[1],budget);
    controls();set_num_threads(1);require(get_num_threads()==1);
    std::cout<<"regionprops saved constructor records="<<records<<"; controls="<<checks<<"; workers=1,4/default1; fresh and cached masks; ordered coords\n";
}
