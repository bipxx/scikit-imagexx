#include "skimagexx/reconstruction.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
namespace {
std::size_t assertions=0;
void check(bool condition,const char* message){++assertions;if(!condition)throw std::runtime_error(message);}
struct Reader {
    std::ifstream f;
    explicit Reader(const char* path):f(path,std::ios::binary){if(!f)throw std::runtime_error("golden file unavailable");}
    std::uint64_t integer(unsigned n){std::uint64_t x=0;for(unsigned i=0;i<n;++i){char c;if(!f.get(c))throw std::runtime_error("truncated golden file");x|=std::uint64_t(static_cast<unsigned char>(c))<<(8*i);}return x;}
    std::vector<unsigned char> bytes(){auto n=integer(8);check(n<(1u<<24),"oversized golden field");std::vector<unsigned char>b(static_cast<std::size_t>(n));if(n&&!f.read(reinterpret_cast<char*>(b.data()),std::streamsize(n)))throw std::runtime_error("truncated golden bytes");return b;}
};
std::size_t itemsize(ReconstructionType t){switch(t){case ReconstructionType::boolean:case ReconstructionType::uint8:case ReconstructionType::int8:return 1;case ReconstructionType::uint16:case ReconstructionType::int16:case ReconstructionType::float16:return 2;case ReconstructionType::uint32:case ReconstructionType::int32:case ReconstructionType::float32:return 4;default:return 8;}}
struct Array {
    ReconstructionType type=ReconstructionType::float64;bool swapped=false;
    std::vector<std::size_t> shape;std::vector<std::ptrdiff_t> strides;std::vector<unsigned char> storage;std::size_t origin=0;
    ReconstructionView view()const{return {storage.data()+origin,storage.data(),storage.size(),{shape.data(),shape.size()},{strides.data(),strides.size()},type,swapped};}
    std::size_t count()const{std::size_t n=1;for(auto d:shape)n*=d;return n;}
    void allocate(const std::vector<unsigned char>& logical,bool unaligned) {
        std::size_t lo=0,hi=0;for(std::size_t d=0;d<shape.size();++d){auto delta=strides[d]*std::ptrdiff_t(shape[d]-1);if(delta<0)lo+=std::size_t(-delta);else hi+=std::size_t(delta);}
        origin=lo+(unaligned ? 1 : 16);storage.assign(origin+hi+itemsize(type)+16,0x5a);
        check(logical.size()==count()*itemsize(type),"golden logical bytes mismatch");
        for(std::size_t i=0;i<count();++i){auto index=i;std::ptrdiff_t off=0;for(auto d=shape.size();d-->0;){auto c=index%shape[d];index/=shape[d];off+=std::ptrdiff_t(c)*strides[d];}std::memcpy(storage.data()+origin+off,logical.data()+i*itemsize(type),itemsize(type));}
    }
};
Array array(Reader& r) {
    Array a;a.type=static_cast<ReconstructionType>(r.integer(4));auto rank=r.integer(4);check(rank<=32,"invalid golden rank");a.swapped=r.integer(4)!=0;bool unaligned=r.integer(4)!=0;
    a.shape.resize(static_cast<std::size_t>(rank));a.strides.resize(static_cast<std::size_t>(rank));for(auto& x:a.shape)x=std::size_t(r.integer(8));for(auto& x:a.strides){auto u=r.integer(8);std::int64_t s;std::memcpy(&s,&u,8);x=std::ptrdiff_t(s);}a.allocate(r.bytes(),unaligned);return a;
}
struct Case {std::string id;Array seed,mask,fp;std::vector<std::size_t> offset;ReconstructionMethod method;ReconstructionZero zero;std::vector<unsigned char> expected;};
Case read_case(Reader& r) {
    Case c;auto id=r.bytes();c.id.assign(id.begin(),id.end());c.seed=array(r);c.mask=array(r);auto rank=std::size_t(r.integer(4));check(rank==c.seed.shape.size(),"footprint rank mismatch");c.fp.type=ReconstructionType::boolean;c.fp.shape.resize(rank);c.offset.resize(rank);for(auto& x:c.fp.shape)x=std::size_t(r.integer(8));for(auto& x:c.offset)x=std::size_t(r.integer(8));c.fp.strides.resize(rank);std::size_t step=1;for(auto d=rank;d-->0;){c.fp.strides[d]=std::ptrdiff_t(step);step*=c.fp.shape[d];}c.fp.allocate(r.bytes(),false);c.method=static_cast<ReconstructionMethod>(r.integer(4));c.zero.supplied=r.integer(4)!=0;c.zero.negative=r.integer(4)!=0;c.expected=r.bytes();return c;
}
template<class T> struct Buffers {
    std::vector<T> out,mask;std::vector<std::size_t> to,from,queue,pos,heads;std::vector<std::ptrdiff_t> neighbors;
    static constexpr std::size_t marker=0x12345678;
    explicit Buffers(const ReconstructionRequirements& r):out(r.padded_samples+2,T(123.25)),mask(r.samples+2,T(123.25)),to(r.samples+2,marker),from(r.padded_samples+2,marker),queue(r.samples+2,marker),pos(r.samples+2,marker),heads(r.bucket_heads+2,marker),neighbors(r.neighbor_capacity+2,std::ptrdiff_t(marker)){}
    ReconstructionBuffers get(){return {{reinterpret_cast<unsigned char*>(out.data()+1),(out.size()-2)*sizeof(T)},{reinterpret_cast<unsigned char*>(mask.data()+1),(mask.size()-2)*sizeof(T)},{to.data()+1,to.size()-2},{from.data()+1,from.size()-2},{queue.data()+1,queue.size()-2},{pos.data()+1,pos.size()-2},{heads.data()+1,heads.size()-2},{neighbors.data()+1,neighbors.size()-2}};}
    void canaries()const{check(out.front()==T(123.25)&&out.back()==T(123.25)&&mask.front()==T(123.25)&&mask.back()==T(123.25),"floating canary changed");for(const auto* a:{&to,&from,&queue,&pos,&heads})check(a->front()==marker&&a->back()==marker,"index canary changed");check(neighbors.front()==std::ptrdiff_t(marker)&&neighbors.back()==std::ptrdiff_t(marker),"neighbor canary changed");}
    bool untouched()const{auto same=[](const auto& v,const auto& x){return std::all_of(v.begin(),v.end(),[&](auto y){return y==x;});};return same(out,T(123.25))&&same(mask,T(123.25))&&same(to,marker)&&same(from,marker)&&same(queue,marker)&&same(pos,marker)&&same(heads,marker)&&same(neighbors,std::ptrdiff_t(marker));}
};
template<class T> void run_case(const Case& c,ReconstructionQueue queue) {
    auto f=ReconstructionFootprint{c.fp.view(),{c.offset.data(),c.offset.size()}};auto r=reconstruction_requirements(c.seed.view(),c.mask.view(),f,queue);Buffers<T>b(r);auto seed=c.seed.storage,mask=c.mask.storage,fp=c.fp.storage;
    auto info=reconstruction_into(c.seed.view(),c.mask.view(),f,c.method,b.get(),c.zero,queue);
    check(info.queue==r.queue,"queue dispatch mismatch");check(!info.parallel_flood,"dependent flood falsely parallel");check(c.expected.size()==r.output_bytes,"expected owner bytes mismatch");
    if(std::memcmp(b.out.data()+1,c.expected.data(),c.expected.size())!=0)throw std::runtime_error("source output mismatch: "+c.id);
    ++assertions;b.canaries();check(c.seed.storage==seed&&c.mask.storage==mask&&c.fp.storage==fp,"input mutated");
    if(queue==ReconstructionQueue::automatic && c.zero.supplied) {
        Buffers<T> rejected(r);bool caught=false;
        try{reconstruction_into(c.seed.view(),c.mask.view(),f,c.method,rejected.get(),{},queue);}catch(const std::domain_error&){caught=true;}
        check(caught&&rejected.untouched(),"missing zero profile wrote storage");
    }
    if(queue==ReconstructionQueue::automatic) {
        Buffers<T> bad(r);auto spans=bad.get();spans.mask.data=spans.output.data;bool rejected=false;
        try{reconstruction_into(c.seed.view(),c.mask.view(),f,c.method,spans,c.zero,queue);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected&&bad.untouched(),"alias rejection wrote storage");
    }
}
Array typed_double(std::size_t h,std::size_t w,double fill) {
    Array a;a.shape={h,w};a.strides={std::ptrdiff_t(w*8),8};a.type=ReconstructionType::float64;std::vector<double>x(h*w,fill);std::vector<unsigned char>b(x.size()*8);std::memcpy(b.data(),x.data(),b.size());a.allocate(b,false);return a;
}
template<class T> Array integer_image(std::size_t h,std::size_t w,T fill,ReconstructionType type) {
    Array a;a.type=type;a.shape={h,w};a.strides={std::ptrdiff_t(w*sizeof(T)),std::ptrdiff_t(sizeof(T))};std::vector<T> x(h*w,fill);std::vector<unsigned char> b(x.size()*sizeof(T));std::memcpy(b.data(),x.data(),b.size());a.allocate(b,false);return a;
}
void integer_fronts() {
    for(bool wide:{false,true})for(bool erosion:{false,true}) {
        auto type=wide ? ReconstructionType::uint16 : ReconstructionType::uint8;
        Array seed=wide ? integer_image<std::uint16_t>(257,259,erosion ? 65535 : 0,type) : integer_image<std::uint8_t>(257,259,erosion ? 255 : 0,type);
        Array mask=wide ? integer_image<std::uint16_t>(257,259,erosion ? 0 : 65535,type) : integer_image<std::uint8_t>(257,259,erosion ? 0 : 255,type);
        if(wide){std::uint16_t marker=erosion ? 37 : 60000;std::memcpy(seed.storage.data()+seed.origin,&marker,2);}else seed.storage[seed.origin]=erosion ? 7 : 231;
        Array fp;fp.type=ReconstructionType::boolean;fp.shape={3,3};fp.strides={3,1};fp.allocate(std::vector<unsigned char>(9,1),false);std::vector<std::size_t> offset{1,1};auto footprint=ReconstructionFootprint{fp.view(),{offset.data(),2}};
        for(auto queue:{ReconstructionQueue::automatic,ReconstructionQueue::indexed_heap,ReconstructionQueue::radix64}) {
            auto r=reconstruction_requirements(seed.view(),mask.view(),footprint,queue);Buffers<double>b(r);auto result=reconstruction_into(seed.view(),mask.view(),footprint,erosion ? ReconstructionMethod::erosion : ReconstructionMethod::dilation,b.get(),{},queue);
            const double expected=wide ? (erosion ? 37 : 60000) : (erosion ? 7 : 231);
            for(std::size_t i=0;i<seed.count();++i)check(b.out[b.to[i+1]+1]==expected,"integer front failed");
            if(queue==ReconstructionQueue::automatic)check(result.queue==(wide ? ReconstructionQueue::bucket16 : ReconstructionQueue::bucket8),"integer bucket not used");
            b.canaries();
        }
    }
}
void large_and_guards(int workers) {
    Case c;c.id="large-directed-front";c.seed=typed_double(257,259,0);c.mask=typed_double(257,259,10);double nine=9;std::memcpy(c.seed.storage.data()+c.seed.origin,&nine,8);c.fp.type=ReconstructionType::boolean;c.fp.shape={3,3};c.fp.strides={3,1};c.fp.allocate(std::vector<unsigned char>(9,1),false);c.offset={1,1};c.method=ReconstructionMethod::dilation;
    auto fp=ReconstructionFootprint{c.fp.view(),{c.offset.data(),2}};auto r=reconstruction_requirements(c.seed.view(),c.mask.view(),fp);Buffers<double>b(r);auto info=reconstruction_into(c.seed.view(),c.mask.view(),fp,c.method,b.get());
    for(std::size_t i=0;i<c.seed.count();++i)check(b.out[b.to[i+1]+1]==9,"large front failed");
#ifdef _OPENMP
    check(info.preprocessing_workers==workers,"actual preparation team mismatch");
#else
    (void)workers;check(info.preprocessing_workers==1,"portable preprocessing team mismatch");
#endif
    b.canaries();
    Buffers<double> rejected(r);auto original=c.seed.storage;double nan=std::numeric_limits<double>::quiet_NaN();std::memcpy(c.seed.storage.data()+c.seed.origin,&nan,8);bool caught=false;
    try{reconstruction_into(c.seed.view(),c.mask.view(),fp,c.method,rejected.get());}catch(const std::domain_error&){caught=true;}
    check(caught&&rejected.untouched(),"nonfinite guard wrote storage");c.seed.storage=original;
    auto round=std::fegetround();std::fesetround(FE_DOWNWARD);caught=false;try{reconstruction_into(c.seed.view(),c.mask.view(),fp,c.method,rejected.get());}catch(const std::domain_error&){caught=true;}
    check(caught&&rejected.untouched()&&std::fegetround()==FE_DOWNWARD,"rounding guard failed restoration");std::fesetround(round);
}
}
int main(int argc,char** argv) {
    try {
        check(get_num_threads()==1,"global default is not1");check(argc>=2,"golden files required");std::size_t source_cases=0;
        for(int workers:{1,4}) {
#ifndef _OPENMP
            if(workers==4){bool rejected=false;try{set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}check(rejected&&get_num_threads()==1,"portable worker4 not rejected");continue;}
#endif
            set_num_threads(workers);
            for(int arg=1;arg<argc;++arg){Reader input(argv[arg]);char magic[8];input.f.read(magic,8);check(std::memcmp(magic,"RECON01\0",8)==0,"invalid golden schema");auto count=input.integer(4);check(count==259,"incomplete numerical corpus");for(std::size_t i=0;i<count;++i){auto c=read_case(input);auto type=c.mask.type;bool single=type==ReconstructionType::float16||type==ReconstructionType::float32;if(single){run_case<float>(c,ReconstructionQueue::automatic);run_case<float>(c,ReconstructionQueue::indexed_heap);}else{run_case<double>(c,ReconstructionQueue::automatic);run_case<double>(c,ReconstructionQueue::indexed_heap);}++source_cases;}check(input.f.peek()==std::char_traits<char>::eof(),"trailing golden data");}
            large_and_guards(workers);integer_fronts();
        }
        set_num_threads(1);std::cout<<"PASS assertions="<<assertions<<" source_projections="<<source_cases<<" avx2="<<has_avx2()<<"\n";
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" assertions="<<assertions<<"\n";return 1;}
}
