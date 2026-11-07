// SOURCE PREPARATION ONLY: link an admitted relocated SDK + existing FEnv32
// test helper after root GO. No mask/primitive-composed MARK substitute.
#include "../fixtures/mark_wide_outer_sdk_v1.hpp"
#include <skimagexx/image.hpp>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>
#if defined(__GNUC__) && defined(__x86_64__) && defined(__linux__)
extern "C" int reference_fenv_restore(const void*,std::size_t);
#ifdef MARK_WRAP_GETTER
std::size_t snapshots=0;
extern "C" int __real__ZN9skimagexx15get_num_threadsEv();
extern "C" int __wrap__ZN9skimagexx15get_num_threadsEv(){++snapshots;return __real__ZN9skimagexx15get_num_threadsEv();}
#endif
namespace {
using namespace skimagexx;
using namespace skimagexx_test;
void check(bool value,const std::string& why){if(!value)throw std::runtime_error(why);}
// Same admitted rolling SDK FP transport: only semantic words in a CURRENT
// trusted empty-stack template; no historical addresses or after-call masks.
WideFP state(const std::fenv_t& f){WideFP s;const auto* p=reinterpret_cast<const unsigned char*>(&f);std::memcpy(&s.control,p,2);std::memcpy(&s.status,p+4,2);std::memcpy(&s.mxcsr,p+28,4);return s;}
bool equal(WideFP a,WideFP b){return a.control==b.control&&a.status==b.status&&a.mxcsr==b.mxcsr;}
void set(std::fenv_t& f,WideFP s){auto* p=reinterpret_cast<unsigned char*>(&f);std::memcpy(p,&s.control,2);std::memcpy(p+4,&s.status,2);std::memcpy(p+28,&s.mxcsr,4);}
bool restore_exact(const std::fenv_t& caller){std::fenv_t restored;return reference_fenv_restore(&caller,32)==0&&std::fegetenv(&restored)==0&&std::memcmp(&caller,&restored,32)==0;}

template<class Operation>void target(const WideSaved& row,Operation operation){
    std::fenv_t caller,incoming,entry,after;check(std::fegetenv(&caller)==0,"read caller");incoming=caller;set(incoming,row.before);
    std::exception_ptr primary;
    try{
        check(reference_fenv_restore(&incoming,32)==0,"restore trusted incoming");check(std::fegetenv(&entry)==0,"read entry");
        operation();check(std::fegetenv(&after)==0,"read immediate target after");
    }catch(...){primary=std::current_exception();}
    const bool restored=restore_exact(caller);
    if(primary){if(!restored)std::cerr<<"secondary caller32 restore failure\n";std::rethrow_exception(primary);}
    check(restored,"caller32 differs");
    if(!equal(state(entry),row.before)||!equal(state(after),row.after)){
        const auto actual=state(after),expected=row.after;
        std::cerr<<"FIRST_STRICT_FP_FAILURE "<<row.data.id<<" actual="<<actual.control<<','<<actual.status<<','<<actual.mxcsr<<" expected="<<expected.control<<','<<expected.status<<','<<expected.mxcsr<<'\n';
        throw std::runtime_error("target semantic FP differs; do not normalize");
    }
}

void run_case(const WideSaved& row,int workers){
    const auto& c=row.data;
    if(row.mark){
        MarkImage result;
        {MarkStorage image(c.image,c.image_bytes),labels(c.labels,c.label_bytes);const auto a=image.words,b=labels.words;
#ifdef MARK_WRAP_GETTER
        snapshots=0;
#endif
        target(row,[&]{result=skimagexx::mark_boundaries(image.view,labels.view,c.parameters,workers);});
#ifdef MARK_WRAP_GETTER
        check(snapshots==std::size_t(workers==0),"owning MARK worker snapshot count");
#endif
        check(image.words==a&&labels.words==b,"owning MARK mutated input");
        check(result.data()!=image.view.data&&result.data()!=labels.view.data,"owning MARK aliases input");}
        // Owners are now destroyed: validate the complete returned owning value.
        check(result.type==row.result_type,"owning MARK output dtype");
        const auto n=c.expected.size()/pixel_size(result.type);
        if(result.type==PixelType::float32)check(result.f32.size()==n&&result.f64.empty()&&result.f32.capacity()>=n,"owning f32 size/capacity");
        else check(result.f64.size()==n&&result.f32.empty()&&result.f64.capacity()>=n,"owning f64 size/capacity");
        mark_compare(result,c);auto copied=result;check(copied.data()!=result.data(),"owning copy aliases");auto moved=std::move(copied);mark_compare(moved,c);
    }else{
        MarkStorage labels(c.labels,c.label_bytes);const auto before=labels.words;
        std::array<std::uint8_t,28> output;output.fill(0xad);bool accepted=false;
#ifdef MARK_WRAP_GETTER
        snapshots=0;
#endif
        target(row,[&]{accepted=detail::find_boundaries_into(labels.view,c.parameters.boundary,output.data()+8,c.result_strides,workers);});
#ifdef MARK_WRAP_GETTER
        check(snapshots==std::size_t(workers==0),"find worker snapshot count");
#endif
        check(accepted&&c.expected.size()==12,"complete find refused or size differs");
        check(labels.words==before,"find mutated labels");
        check(std::all_of(output.begin(),output.begin()+8,[](auto x){return x==0xad;})&&std::all_of(output.begin()+20,output.end(),[](auto x){return x==0xad;}),"find output capacity canary");
        check(std::equal(c.expected.begin(),c.expected.end(),output.begin()+8),"find exact saved bool words");
        check(std::all_of(output.begin()+8,output.begin()+20,[](auto x){return x<=1;}),"find noncanonical bool");
    }
}

void refusal(const WideSaved& row,bool nonnearest){
    const auto& c=row.data;std::fenv_t caller;check(std::fegetenv(&caller)==0,"read refusal caller");std::exception_ptr primary;
    try{
        if(nonnearest)check(std::fesetround(FE_DOWNWARD)==0,"set refusal rounding");
        auto parameters=c.parameters;if(!nonnearest)parameters.boundary.background_representable=false;
        MarkStorage labels(c.labels,c.label_bytes);const auto b=labels.words;
        if(row.mark){MarkStorage image(c.image,c.image_bytes);const auto a=image.words;bool rejected=false;
            try{auto result=skimagexx::mark_boundaries(image.view,labels.view,parameters);(void)result;}catch(const std::invalid_argument&){rejected=true;}
            check(rejected&&image.words==a,"MARK unsupported frontier/input");
        }else{std::array<std::uint8_t,28> output;output.fill(0xad);
            check(!detail::find_boundaries_into(labels.view,parameters.boundary,output.data()+8,c.result_strides),"find accepted unsupported frontier");
            check(std::all_of(output.begin(),output.end(),[](auto x){return x==0xad;}),"find refusal wrote output");}
        check(labels.words==b,"refusal changed labels");
    }catch(...){primary=std::current_exception();}
    const bool restored=restore_exact(caller);
    if(primary){if(!restored)std::cerr<<"secondary refusal caller32 restore failure\n";std::rethrow_exception(primary);}
    check(restored,"refusal caller32 differs");
}
}

int main(){
    static_assert(sizeof(std::fenv_t)==32,"Retained GNUx64 glibc32 FEnv ABI required");
    std::fenv_t process;if(std::fegetenv(&process)!=0)return 2;std::exception_ptr primary;std::size_t checked=0;
    try{
        std::uint16_t endian=1;check(*reinterpret_cast<unsigned char*>(&endian)==1,"little endian literal words required");
        check(skimagexx::get_num_threads()==1,"initial SDK defaultworkers1");const auto rows=skimagexx_test::mark_wide_saved_cases();check(rows.size()==8,"eight saved rows");
        const int all_budgets[3][2]={{1,0},{4,0},{1,4}};
        for(const auto& budgets:all_budgets){
            skimagexx::set_num_threads(budgets[0]);
            for(const auto& row:rows){run_case(row,budgets[1]);check(skimagexx::get_num_threads()==budgets[0],"SDK globalworker changed");++checked;}
        }
        skimagexx::set_num_threads(1);for(bool nonnearest:{false,true}){refusal(rows.at(0),nonnearest);refusal(rows.at(2),nonnearest);}
    }catch(...){primary=std::current_exception();}
    skimagexx::set_num_threads(1);const bool restored=restore_exact(process);
    if(primary){try{std::rethrow_exception(primary);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';}if(!restored)std::cerr<<"secondary process32 restore failure\n";return 1;}
    if(!restored||checked!=24)return 1;
    std::cout<<"PASS new wide saved SDK records=8 completecalls=24 frontier_refusals=4 profiles=2 default1/global4/explicit4 caller32exact; no originals or timing\n";return 0;
}
#else
int main(){std::cerr<<"New wide OUTER SDK domain requires GNUx64 Linux/glibc32; Windows/other platforms remain unsupported, not a parity PASS\n";return 2;}
#endif
