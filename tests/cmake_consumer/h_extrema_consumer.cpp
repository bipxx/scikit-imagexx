// Real installed H stages + reconstruction symbols, with mandatory source masks.
#include <skimagexx/h_extrema.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
namespace {
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Reader {
    std::ifstream f;
    explicit Reader(const char* name):f(name,std::ios::binary){check(bool(f),"H oracle unavailable");}
    std::uint64_t integer(unsigned width=8){std::uint64_t result=0;for(unsigned i=0;i<width;++i){char c;check(bool(f.get(c)),"truncated H oracle");result|=std::uint64_t(static_cast<unsigned char>(c))<<(8*i);}return result;}
    std::vector<unsigned char> bytes(){const auto n=integer();check(n<1024*1024,"oversized H oracle field");std::vector<unsigned char>b(static_cast<std::size_t>(n));if(n)check(bool(f.read(reinterpret_cast<char*>(b.data()),std::streamsize(n))),"truncated H oracle bytes");return b;}
};
struct Case {std::string id;ReconstructionType type;ReconstructionMethod method;std::size_t width;std::uint64_t height;std::vector<std::size_t> shape;std::vector<std::ptrdiff_t> strides;std::vector<unsigned char> image,expected;};
Case read(Reader& r){
    Case c;auto name=r.bytes();c.id.assign(name.begin(),name.end());const auto tag=r.integer(4);check(tag>=1&&tag<=8,"H integer dtype invalid");c.type=static_cast<ReconstructionType>(tag);c.width=std::size_t(1)<<((tag-1)/2);
    const auto op=r.integer(4);check(op<2,"H method invalid");c.method=static_cast<ReconstructionMethod>(op);const auto rank=r.integer(4);check(rank==2,"SDK source case rank changed");c.shape.resize(rank);for(auto& d:c.shape){d=std::size_t(r.integer());check(d>=4&&d<64,"H source dimensions unsafe");}
    c.strides.resize(rank);std::size_t step=c.width;for(auto d=rank;d-->0;){c.strides[d]=std::ptrdiff_t(step);step*=c.shape[d];}c.height=r.integer();c.image=r.bytes();c.expected=r.bytes();check(c.image.size()==step&&c.expected.size()==step/c.width,"H source size mismatch");return c;
}
ReconstructionView view(const std::vector<unsigned char>& b,const Case& c){return {b.data(),b.data(),b.size(),{c.shape.data(),c.shape.size()},{c.strides.data(),c.strides.size()},c.type,false};}
void run(const Case& c){
    const auto image=view(c.image,c);const auto n=h_extrema_validate_view(image);check(n==c.expected.size(),"H metadata size mismatch");
    const auto range=h_extrema_integer_range(image);check(range.bits==12&&range.samples==n,"H source range mismatch");
    std::vector<unsigned char> shifted(c.image.size());const auto prep=h_extrema_integer_seed_into(image,c.height,c.method,{shifted.data(),shifted.size()});check(prep.workers>=1&&prep.workers<=get_num_threads(),"H prep worker oversubscription");
    const auto seed=view(shifted,c);std::vector<std::size_t> fp_shape(c.shape.size(),3),offset(c.shape.size(),1);std::vector<std::ptrdiff_t> fp_strides(c.shape.size());std::size_t count=1;for(auto d=c.shape.size();d-->0;){fp_strides[d]=std::ptrdiff_t(count);count*=3;}
    std::vector<unsigned char> fp(count,1);ReconstructionFootprint footprint{{fp.data(),fp.data(),fp.size(),{fp_shape.data(),fp_shape.size()},{fp_strides.data(),fp_strides.size()},ReconstructionType::boolean,false},{offset.data(),offset.size()}};
    const auto requested=reconstruction_requirements(seed,image,footprint);check(requested.output_type==ReconstructionType::float64,"H reconstruction precision changed");
    std::vector<double> out(requested.padded_samples),mask(requested.samples);
    std::vector<std::size_t> to(requested.samples),from(requested.padded_samples),queue(requested.samples),position(requested.samples),heads(requested.bucket_heads);
    std::vector<std::ptrdiff_t> neighbors(requested.neighbor_capacity);
    ReconstructionBuffers buffers{{reinterpret_cast<unsigned char*>(out.data()),out.size()*8},{reinterpret_cast<unsigned char*>(mask.data()),mask.size()*8},{to.data(),to.size()},{from.data(),from.size()},{queue.data(),queue.size()},{position.data(),position.size()},{heads.data(),heads.size()},{neighbors.data(),neighbors.size()}};
    const auto info=reconstruction_into(seed,image,footprint,c.method,buffers);check(!info.parallel_flood,"dependent flood falsely parallel");
    // The actual padded output owner is retained; residue reads its interior.
    std::vector<std::ptrdiff_t> strides(c.shape.size());std::size_t stride=8,interior=0;for(auto d=c.shape.size();d-->0;){strides[d]=std::ptrdiff_t(stride);interior+=stride;stride*=c.shape[d]+2;}
    const auto* data=reinterpret_cast<const unsigned char*>(out.data())+interior;
    ReconstructionView reconstructed{data,out.data(),out.size()*8,{c.shape.data(),c.shape.size()},{strides.data(),strides.size()},ReconstructionType::float64,false};
    std::vector<unsigned char> actual(n,255);const auto compared=h_extrema_integer_mask_into(image,reconstructed,double(c.height),c.method,{actual.data(),actual.size()});check(compared.workers==1,"H residue ordering changed");
    if(actual!=c.expected)throw std::runtime_error("source H mask mismatch: "+c.id);
}
}
int main(int argc,char** argv){try{
    check(argc==2,"exact H source oracle argument required");check(get_num_threads()==1,"SDK default worker is not1");Reader r(argv[1]);char magic[8];check(bool(r.f.read(magic,8))&&std::memcmp(magic,"HXSDK001",8)==0,"wrong H oracle magic");const auto count=r.integer(4);check(count==16,"H source oracle cases incomplete");std::vector<Case> cases;for(std::uint64_t i=0;i<count;++i)cases.push_back(read(r));check(r.f.peek()==std::char_traits<char>::eof(),"trailing H oracle bytes");
    for(int workers:{1,4}){try{set_num_threads(workers);}catch(const std::exception& e){check(workers==4&&get_num_threads()==1&&std::string(e.what()).find("no OpenMP support")!=std::string::npos,"unexpected SDK worker rejection");continue;}check(get_num_threads()==workers,"SDK shared worker mismatch");for(const auto& c:cases)run(c);set_num_threads(1);}
    check(get_num_threads()==1,"SDK worker not restored");std::cout<<"H integer SDK source masks16, workers1/4 when OpenMP enabled; fullAPI remains partial\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
