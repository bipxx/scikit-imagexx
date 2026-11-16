#pragma once
// Owning SDK controls only. std::log2 is a disclosed scalar host provider;
// these checks do not establish NumPy/SVML or original-public FP parity.
#include <skimagexx/variation_of_information.hpp>
#include <skimagexx/image.hpp>
#include "fixtures/variation_of_information_saved_v1.hpp"
#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(SKIMAGEXX_VI_FENV32)
#include <xmmintrin.h>
#if !defined(__linux__) || !defined(__x86_64__) || !defined(__GLIBC__)
#error "VI full32 controls require the admitted Linux x86-64 glibc ABI"
#endif
extern "C" int reference_fenv_restore(const void*,std::size_t);
#endif

namespace vi_sdk_controls {
using skimagexx::VariationInformation;
using skimagexx::VariationTableView;
using skimagexx::VariationLog2Batch;
using skimagexx::detail::LabelPairInput;
struct Counters {unsigned owning=0,default_refusals=0,callbacks=0,reads32=0,restores32=0,checked32=0;};
inline Counters counters;

inline bool same(const VariationInformation& a,const VariationInformation& b) {
    return std::memcmp(a.values.data(),b.values.data(),16)==0
        &&a.normalization_fp_status==b.normalization_fp_status
        &&a.captured_workers==b.captured_workers;
}
inline VariationInformation sentinel() {
    VariationInformation out;out.values={123.,456.};
    out.normalization_fp_status=0x1234u;out.captured_workers=91;return out;
}

#if defined(SKIMAGEXX_VI_FENV32)
inline bool read32(std::fenv_t& out){++counters.reads32;return std::fegetenv(&out)==0;}
inline void words32(const std::fenv_t& value) {
    constexpr char hex[]="0123456789abcdef";
    const auto* bytes=reinterpret_cast<const unsigned char*>(&value);
    for(std::size_t i=0;i<32;++i)std::cout<<hex[bytes[i]>>4]<<hex[bytes[i]&15];
}
#endif
inline bool layout() {
#if defined(SKIMAGEXX_VI_FENV32)
    static_assert(sizeof(std::fenv_t)==32,"Admitted GNUx64 glibc FEnv32 required");
    static_assert(offsetof(std::fenv_t,__control_word)==0&&offsetof(std::fenv_t,__status_word)==4
        &&offsetof(std::fenv_t,__tags)==8&&offsetof(std::fenv_t,__eip)==12
        &&offsetof(std::fenv_t,__data_offset)==20&&offsetof(std::fenv_t,__mxcsr)==28
        &&sizeof(void*)==8,"Admitted FEnv field offsets required");
    const std::uint16_t endian=1;return *reinterpret_cast<const unsigned char*>(&endian)==1;
#else
    return true;
#endif
}
// Capture CURRENT caller bytes, never historical instruction/data addresses.
// Restore after a test group (including exception paths), never inside math.
class Caller32 {
    int workers_=skimagexx::get_num_threads();
    bool valid_=true,finished_=false,restored_=false;
    const char* name_;
#if defined(SKIMAGEXX_VI_FENV32)
    std::fenv_t before_{};
#endif
public:
    Caller32(const Caller32&)=delete;
    Caller32& operator=(const Caller32&)=delete;
    explicit Caller32(const char* name):name_(name) {
#if defined(SKIMAGEXX_VI_FENV32)
        valid_=read32(before_);
        std::uint16_t tags=0;std::memcpy(&tags,reinterpret_cast<const unsigned char*>(&before_)+8,2);
        valid_=valid_&&tags==0xffffu&&(control()&0x3fu)==0x3fu
            &&(mxcsr()&0x1f80u)==0x1f80u&&!(mxcsr()&0xffff0000u);
#endif
    }
    bool good()const{return valid_;}
#if defined(SKIMAGEXX_VI_FENV32)
    std::uint16_t control()const{std::uint16_t v;std::memcpy(&v,&before_,2);return v;}
    std::uint16_t status()const{std::uint16_t v;std::memcpy(&v,reinterpret_cast<const unsigned char*>(&before_)+4,2);return v;}
    std::uint32_t mxcsr()const{std::uint32_t v;std::memcpy(&v,reinterpret_cast<const unsigned char*>(&before_)+28,4);return v;}
#endif
    bool finish()noexcept {
        if(finished_)return restored_;
        finished_=true;skimagexx::set_num_threads(workers_);
        restored_=skimagexx::get_num_threads()==workers_;
#if defined(SKIMAGEXX_VI_FENV32)
        std::fenv_t after{};bool exact=false;
        if(valid_){++counters.restores32;exact=reference_fenv_restore(&before_,32)==0&&read32(after)
            &&std::memcmp(&before_,&after,32)==0;}
        restored_=restored_&&exact;++counters.checked32;
        std::cout<<"VI_SDK_CALLER32 group="<<name_<<" before=";words32(before_);
        std::cout<<" restored=";words32(after);std::cout<<" equal="<<exact<<'\n';
#endif
        if(!restored_)std::cerr<<"secondary VI SDK caller restoration failure group="<<name_<<'\n';
        return restored_;
    }
    ~Caller32(){if(!finished_)finish();}
};

struct ProviderFailure:std::runtime_error {
    const void* identity;
    explicit ProviderFailure(const void* value):std::runtime_error("VI SDK provider primary"),identity(value){}
};
struct Audit {
    unsigned calls=0,empty=0,fail_on=0,throw_on=0;
    std::thread::id caller=std::this_thread::get_id();
    bool wrong_thread=false,wrong_arguments=false;
    const double* expected[2]={nullptr,nullptr};std::size_t expected_count=0;
};
inline bool log2_batch(const double* values,double* logs,std::size_t count,void* context) {
    auto& a=*static_cast<Audit*>(context);++a.calls;++counters.callbacks;
    if(std::this_thread::get_id()!=a.caller){a.wrong_thread=true;return false;}
    if(a.expected_count&&(a.calls>2||count!=a.expected_count
       ||std::memcmp(values,a.expected[a.calls-1],count*sizeof(double)))){a.wrong_arguments=true;return false;}
    if(a.calls==a.throw_on) {
#if defined(SKIMAGEXX_VI_FENV32)
        if(std::fesetround(FE_DOWNWARD)!=0)throw std::runtime_error("VI SDK throw-context setup failed");
#endif
        throw ProviderFailure(&a);
    }
    if(a.calls==a.fail_on)return false;
    if(!count)++a.empty;
    for(std::size_t i=0;i<count;++i)logs[i]=std::log2(values[i]);
    return true;
}
template<class Input>bool invoke(Input input,VariationInformation& out,VariationLog2Batch provider,int workers) {
    ++counters.owning;return skimagexx::variation_of_information_compatible(input,out,provider,workers);
}
template<class Input>bool success(Input input,VariationInformation& out,Audit& audit,int workers) {
    const auto before=audit.calls;
    return invoke(input,out,{log2_batch,&audit},workers)&&audit.calls==before+2
        &&!audit.wrong_thread&&!audit.wrong_arguments;
}

inline int compatible_counts() {
    Caller32 caller("compatible-counts");if(!caller.good())return 100;
    unsigned checked=0;
    // Same immutable fixture recipes as current_counts_controls.hpp. This
    // group calls the compatible operation; it does not rerun default VI.
    for(const std::size_t n:{4096u,4099u})for(int ignored_mode:{0,1,2}) {
        std::vector<std::int64_t> a(n),b(n),wide_a(n),wide_b(n);
        for(std::size_t i=0;i<n;++i){a[i]=(i/7)%4;b[i]=(i/3)%8;wide_a[i]=a[i]+64;wide_b[i]=b[i]+64;}
        const auto before_a=a,before_b=b,before_wide_a=wide_a,before_wide_b=wide_b;
        const std::int64_t labels[]={0,1,2,3},wide_labels[]={64,65,66,67};
        const auto ignored=ignored_mode==2?4u:ignored_mode==1?1u:0u;
        LabelPairInput in{a.data(),b.data(),n,labels,ignored,true};
        LabelPairInput wide{wide_a.data(),wide_b.data(),n,wide_labels,ignored,true};
        skimagexx::detail::LabelPairCsr dense;
        if(!skimagexx::detail::label_pair_counts(in,dense,1)||dense.data.size()!=32)return 101;
        std::size_t kept=0;for(auto v:a)if(ignored_mode==0||(ignored_mode==1&&v!=0))++kept;
        std::vector<double> weights(n);
        for(std::size_t i=0;i<n;++i) {
            if(ignored_mode==2)weights[i]=dense.data[0];
            else weights[i]=(ignored_mode==1&&a[i]==0)?0.:1./static_cast<double>(kept);
        }
        skimagexx::detail::LabelPairCsrStorage storage;
        if(!skimagexx::detail::label_pair_counts_preweighted({a.data(),b.data(),weights.data(),n},storage,1)
           ||storage.indptr.back()!=32||storage.data.size()!=n)return 102;
        for(int workers:{1,4,0}) {
            skimagexx::set_num_threads(workers==0?4:workers);const int captured=workers==0?4:workers;
            Audit audit;VariationInformation x,y,z,logical,explicit_one;
            if(!success(in,x,audit,workers)||!success(wide,y,audit,workers)
               ||!success(VariationTableView{dense.data.data(),dense.indices.data(),dense.indptr.data(),4,8,32},z,audit,workers)
               ||!success(VariationTableView{storage.data.data(),storage.indices.data(),storage.indptr.data(),4,8,32},logical,audit,workers))return 103;
            if(!same(x,y)||!same(z,logical)||std::memcmp(x.values.data(),z.values.data(),16)
               ||x.normalization_fp_status!=dense.normalization_fp_status||z.normalization_fp_status!=0
               ||x.captured_workers!=captured||z.captured_workers!=captured)return 104;
            skimagexx::set_num_threads(4);
            if(!success(in,explicit_one,audit,1)||explicit_one.captured_workers!=1
               ||std::memcmp(x.values.data(),explicit_one.values.data(),16)
               ||x.normalization_fp_status!=explicit_one.normalization_fp_status||skimagexx::get_num_threads()!=4)return 105;
            auto untouched=sentinel();const auto expected=untouched;const auto calls=audit.calls;
            if(invoke(VariationTableView{storage.data.data(),storage.indices.data(),storage.indptr.data(),4,8,n},untouched,{log2_batch,&audit},workers)
               ||!same(untouched,expected)||audit.calls!=calls)return 106;
            if(a!=before_a||b!=before_b||wide_a!=before_wide_a||wide_b!=before_wide_b)return 107;
            ++checked;
        }
    }
    skimagexx::set_num_threads(1);
    if(!caller.finish()||checked!=18)return 108;
    std::cout<<"VI SDK compatible counts="<<checked<<" logicalNNZ32 physicalN4096or4099 workers0,1,4 explicit1overglobal4\n";
    return 0;
}

inline int compatible_large() {
    Caller32 caller("compatible-large");if(!caller.good())return 110;
    // Verbatim admitted original words from unchanged large_controls.hpp.
    const std::uint64_t expected[4][2]={
        {UINT64_C(0x4007cbe4873c419a),UINT64_C(0x40064266ad26a8cd)},
        {UINT64_C(0x401190faa8ca5c4c),UINT64_C(0x40117a1cdc29005c)},
        {UINT64_C(0x4007cbbfce8962e6),UINT64_C(0x4006423b60ce96dd)},
        {UINT64_C(0x401191ad2ded50af),UINT64_C(0x40117acb63349d16)},
    };
    unsigned checked=0;skimagexx::set_num_threads(1);
    for(unsigned c=0;c<4;++c) {
        const std::size_t n=c<2?16384:65536,labels=c%2?64:8;
        std::vector<std::int64_t> truth(n),test(n);const std::int64_t ignore=0;
        for(std::size_t i=0;i<n;++i){truth[i]=i%labels;test[i]=(i/3+i%7)%labels;}
        const auto before_truth=truth,before_test=test;
        for(int workers:{0,1,4}) {
            Audit audit;VariationInformation result;
            LabelPairInput in{truth.data(),test.data(),n,&ignore,1,true};
            if(!success(in,result,audit,workers)||result.captured_workers!=(workers?workers:1)
               ||result.normalization_fp_status!=0||std::memcmp(result.values.data(),expected[c],16)
               ||truth!=before_truth||test!=before_test)return 111;
            ++checked;
        }
    }
    if(!caller.finish()||checked!=12)return 112;
    std::cout<<"VI SDK compatible large_saved_records="<<checked<<" strict_words="<<checked*2<<'\n';return 0;
}

template<class Input>int failure_cases(Input input) {
    {
        Caller32 caller("null-provider");if(!caller.good())return 120;
        auto out=sentinel();const auto before=out;
        if(invoke(input,out,{},1)||!same(out,before))return 121;
        if(!caller.finish())return 122;
    }
    for(bool throwing:{false,true})for(unsigned phase:{1u,2u}) {
        Caller32 caller("provider-failure");if(!caller.good())return 123;
        Audit audit;if(throwing)audit.throw_on=phase;else audit.fail_on=phase;
        auto out=sentinel();const auto before=out;bool caught=false,accepted=false;
        try{accepted=invoke(input,out,{log2_batch,&audit},1);}
        catch(const ProviderFailure& error){if(!throwing||error.identity!=&audit)throw;caught=true;}
        if(accepted||caught!=throwing||audit.calls!=phase||audit.wrong_thread
           ||audit.wrong_arguments||!same(out,before))return 124;
#if defined(SKIMAGEXX_VI_FENV32)
        if(throwing&&std::fegetround()!=FE_DOWNWARD)return 125; // callback effects survive unwind
#endif
        if(!caller.finish())return 126;
    }
    return 0;
}

inline int array_boundaries() {
    const std::int64_t columns[]={1,1,0,1},rows[]={0,3,4};
    {
        Caller32 caller("stored-zero");if(!caller.good())return 130;
        const double values[]={0.,-0.,0.,-0.};Audit audit;VariationInformation out;
        if(!success(VariationTableView{values,columns,rows,2,2,4},out,audit,0)
           ||audit.empty!=2||out.captured_workers!=1||out.normalization_fp_status!=0)return 131;
        const std::uint64_t zeros[2]={};if(std::memcmp(out.values.data(),zeros,16))return 132;
        if(!caller.finish())return 133;
    }
    {
        Caller32 caller("duplicate-order");if(!caller.good())return 134;
        const std::int64_t index[]={1,0,1,0,1,0,1},offset[]={0,4,7};
        const double values[]={.25,.125,.5,.125,.5,.75,.75};
        const double left[]={.25,.75,.375,.625},right[]={.25,.375,.75,.625};
        Audit audit;audit.expected[0]=left;audit.expected[1]=right;audit.expected_count=4;
        VariationInformation out;
        if(!success(VariationTableView{values,index,offset,2,2,7},out,audit,1)
           ||out.captured_workers!=1||out.normalization_fp_status!=0)return 135;
        if(!caller.finish())return 136;
    }
    return 0;
}

inline int fenv_refusals(VariationTableView table,LabelPairInput image) {
#if defined(SKIMAGEXX_VI_FENV32)
    for(unsigned mode=0;mode<5;++mode) {
        Caller32 caller("FENV-refusal");if(!caller.good())return 140;
        if(mode==0){if(std::fesetround(FE_DOWNWARD)!=0)return 141;}
        if(mode==1)_mm_setcsr(caller.mxcsr()|(1u<<15));
        if(mode==2)_mm_setcsr(caller.mxcsr()|(1u<<6));
        // Unmask only a currently clear divide-by-zero flag. Never clear/seed
        // caller sticky bits to make a numerical observation pass.
        if(mode==3){if(caller.mxcsr()&(1u<<2))return 142;_mm_setcsr(caller.mxcsr()&~(1u<<9));}
        if(mode==4){if(caller.status()&FE_DIVBYZERO)return 143;
            const auto cw=std::uint16_t(caller.control()&~std::uint16_t(FE_DIVBYZERO));
            __asm__ volatile("fldcw %0"::"m"(cw):"memory");}
        std::fenv_t entered{};if(!read32(entered))return 144;
        for(bool compatible:{false,true})for(bool use_image:{false,true}) {
            Audit audit;auto out=sentinel();const auto before=out;
            bool accepted;
            if(compatible)accepted=use_image?invoke(image,out,{log2_batch,&audit},1):invoke(table,out,{log2_batch,&audit},1);
            else {++counters.default_refusals;accepted=use_image?skimagexx::variation_of_information(image,out,1):skimagexx::variation_of_information(table,out,1);}
            std::fenv_t after{};if(!read32(after))return 145;
            std::cout<<"VI_SDK_FENV_REFUSAL mode="<<mode<<" compatible="<<compatible<<" image="<<use_image<<" entered=";
            words32(entered);std::cout<<" after=";words32(after);std::cout<<'\n';
            if(accepted||audit.calls||!same(out,before)||std::memcmp(&entered,&after,32))return 146;
        }
        if(!caller.finish())return 147;
    }
    std::cout<<"VI SDK FENV refusals=20 default/compatible rounding/FTZ/DAZ/SSE-unmasked/x87-unmasked caller32exact\n";
#else
    (void)table;(void)image;
    std::cout<<"VI SDK FENV full32/nondefault controls=UNASSESSED_PLATFORM\n";
#endif
    return 0;
}

inline int run() {
    if(int code=compatible_counts())return code;
    if(int code=compatible_large())return code;
    const std::int64_t columns[]={1,1,0,1},rows[]={0,3,4};const double data[]={0.,.25,.25,.5};
    const VariationTableView table{data,columns,rows,2,2,4};
    const auto& saved=vi_saved[3];
    const LabelPairInput image{saved.truth.data(),saved.test.data(),saved.truth.size(),saved.ignored.data(),saved.ignored.size(),true};
    if(int code=failure_cases(table))return code;
    if(int code=failure_cases(image))return code;
    if(int code=array_boundaries())return code;
    if(int code=fenv_refusals(table,image))return code;
    return 0;
}
} // namespace vi_sdk_controls
