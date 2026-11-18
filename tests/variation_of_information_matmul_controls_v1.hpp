#pragma once
// Owning SDK callback plumbing, not NumPy/BLAS or original-public parity.
// Reuse the frozen F1 layout/Caller32/sentinel machinery without editing it.
#include "variation_of_information_provider_controls.hpp"
#include <string>
#include <type_traits>
#include <utility>

namespace vi_matmul_sdk_controls {
using skimagexx::VariationInformation;
using skimagexx::VariationTableView;
using skimagexx::VariationLog2Batch;
using skimagexx::VariationInnerProduct;
using skimagexx::detail::LabelPairInput;
using vi_sdk_controls::Caller32;
using vi_sdk_controls::same;
using vi_sdk_controls::sentinel;
using vi_sdk_controls::ProviderFailure;
static_assert(std::is_same_v<VariationInnerProduct::Function,
    bool(*)(const double*,const double*,std::size_t,double*,void*)>);
struct Totals {unsigned calls=0,callbacks=0,logs=0,dots=0;};
inline Totals totals;
struct Record {
    char kind;std::vector<double> left,right,output;bool completed=false;
};
struct Audit;
struct Context {Audit* owner;char kind;};
struct Audit {
    Context logarithm{this,'L'},inner{this,'D'};
    std::thread::id caller=std::this_thread::get_id();
    std::vector<Record> records;
    unsigned fail_on=0,throw_on=0;
    bool wrong=false,change_global=false;
#if defined(SKIMAGEXX_VI_FENV32)
    std::fenv_t callback_effect{};bool effect_captured=false;
#endif
    Audit()=default;Audit(const Audit&)=delete;Audit& operator=(const Audit&)=delete;
};
inline void raw(const std::vector<double>& values) {
    constexpr char hex[]="0123456789abcdef";
    for(const double& v:values){std::uint64_t w;std::memcpy(&w,&v,8);
        for(int shift=60;shift>=0;shift-=4)std::cout<<hex[(w>>shift)&15];std::cout<<',';}
}
inline void emit(const Audit& audit,const char* name) {
    unsigned step=0;
    for(const auto& r:audit.records) {
        std::cout<<"VI_MATMUL_CALLBACK case="<<name<<" step="<<++step<<" kind="<<r.kind
            <<" count="<<r.left.size()<<" completed="<<r.completed<<" left=";raw(r.left);
        std::cout<<" right=";raw(r.right);std::cout<<" output=";raw(r.output);std::cout<<'\n';
    }
}
inline bool equal(const std::vector<double>& a,const std::vector<double>& b) {
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0);
}
inline bool enter(Context& context,char kind,const double* a,const double* b,std::size_t n) {
    auto& audit=*context.owner;++totals.callbacks;
    if(kind=='L')++totals.logs;else ++totals.dots;
    const auto next=audit.records.size()+1;
    if(std::this_thread::get_id()!=audit.caller||context.kind!=kind
       ||&context!=(kind=='L'?&audit.logarithm:&audit.inner)
       ||next>4||kind!=(next%2?'L':'D')||(n&&(!a||(kind=='D'&&!b)))){audit.wrong=true;return false;}
    Record record;record.kind=kind;
    if(n){record.left.assign(a,a+n);if(kind=='D')record.right.assign(b,b+n);}
    audit.records.push_back(std::move(record));
#ifdef _OPENMP
    if(next==1&&audit.change_global)skimagexx::set_num_threads(2);
#endif
    if(next==audit.fail_on||next==audit.throw_on) {
#if defined(SKIMAGEXX_VI_FENV32)
        if(std::fesetround(FE_DOWNWARD)!=0)throw std::runtime_error("matmul callback effect setup failed");
        audit.effect_captured=vi_sdk_controls::read32(audit.callback_effect);
        if(!audit.effect_captured)throw std::runtime_error("matmul callback effect capture failed");
#endif
        if(next==audit.throw_on)throw ProviderFailure(&audit);
        return false;
    }
    return true;
}
inline bool logarithm(const double* input,double* output,std::size_t n,void* data) {
    auto& context=*static_cast<Context*>(data);
    if(!enter(context,'L',input,nullptr,n))return false;
    auto& record=context.owner->records.back();
    if(n&&!output){context.owner->wrong=true;return false;}
    for(std::size_t i=0;i<n;++i)output[i]=std::log2(input[i]);
    if(n)record.output.assign(output,output+n);record.completed=true;return true;
}
inline bool inner(const double* left,const double* right,std::size_t n,double* output,void* data) {
    auto& context=*static_cast<Context*>(data);
    if(!enter(context,'D',left,right,n))return false;
    if(!output){context.owner->wrong=true;return false;}
    double result=0.;for(std::size_t i=0;i<n;++i)result+=left[i]*right[i];
    *output=result;auto& record=context.owner->records.back();record.output={result};record.completed=true;return true;
}
template<class Input>bool invoke(Input input,VariationInformation& out,Audit& audit,int workers,
    bool null_log=false,bool null_dot=false) {
    ++totals.calls;
    return skimagexx::variation_of_information_compatible(input,out,
        VariationLog2Batch{null_log?nullptr:logarithm,&audit.logarithm},
        VariationInnerProduct{null_dot?nullptr:inner,&audit.inner},workers);
}
inline bool ordered(const Audit& audit) {
    return !audit.wrong&&audit.records.size()==4&&audit.records[0].kind=='L'
        &&audit.records[1].kind=='D'&&audit.records[2].kind=='L'&&audit.records[3].kind=='D';
}
inline bool published(const Audit& audit,const VariationInformation& out) {
    return ordered(audit)&&audit.records[1].completed&&audit.records[3].completed
        &&std::memcmp(&out.values[0],audit.records[1].output.data(),8)==0
        &&std::memcmp(&out.values[1],audit.records[3].output.data(),8)==0;
}
inline bool uniform_vectors(const Audit& a,std::size_t extent,std::size_t offset) {
    std::vector<double> negative(extent,-0.),entropy(extent,0.),positive(extent,0.),negative_entropy(extent,-0.);
    for(auto i:{offset,offset+1}){negative[i]=-.5;entropy[i]=-1.;positive[i]=.5;negative_entropy[i]=1.;}
    return ordered(a)&&equal(a.records[0].left,{.5,.5,.5,.5})&&equal(a.records[2].left,{.5,.5,.5,.5})
        &&equal(a.records[1].left,negative)&&equal(a.records[1].right,entropy)
        &&equal(a.records[3].left,negative_entropy)&&equal(a.records[3].right,positive);
}
inline int successes(VariationTableView table,LabelPairInput image,LabelPairInput wide) {
    Caller32 caller("matmul-successes");if(!caller.good())return 201;
#ifdef _OPENMP
    constexpr int extra_budget=4;
#else
    constexpr int extra_budget=1;
#endif
    for(int workers:{1,4,0}) {
        const int global=workers==0?extra_budget:1,captured=workers?workers:global;
        skimagexx::set_num_threads(global);
        for(unsigned which=0;which<3;++which) {
            Audit a;VariationInformation out;
            const bool ok=which==0?invoke(table,out,a,workers):which==1?invoke(image,out,a,workers):invoke(wide,out,a,workers);
            emit(a,"finite");
            if(!ok||!published(a,out)||!uniform_vectors(a,which==2?66:2,which==2?64:0)
               ||out.values[0]!=1.||out.values[1]!=1.||out.normalization_fp_status!=0
               ||out.captured_workers!=captured||skimagexx::get_num_threads()!=global)return 202;
        }
    }
    skimagexx::set_num_threads(extra_budget);Audit a;VariationInformation out;
#ifdef _OPENMP
    a.change_global=true;
    constexpr int after_callback=2;
    constexpr const char* trace="explicit1-global4-callback2";
#else
    constexpr int after_callback=1;
    constexpr const char* trace="explicit1-global1-callback";
#endif
    const bool ok=invoke(image,out,a,1);emit(a,trace);
    if(!ok||!published(a,out)||!uniform_vectors(a,2,0)||out.captured_workers!=1
       ||skimagexx::get_num_threads()!=after_callback)return 203;
    return caller.finish()?0:204;
}
inline int boundaries() {
    Caller32 caller("matmul-boundaries");if(!caller.good())return 210;
    skimagexx::set_num_threads(1);
    const std::int64_t indices[]={1,0,1,0,1,0,1},offsets[]={0,4,7};
    const double values[]={.25,.125,.5,.125,.5,.75,.75};
    Audit duplicate;VariationInformation out;
    if(!invoke(VariationTableView{values,indices,offsets,2,2,7},out,duplicate,1))return 211;
    emit(duplicate,"duplicate-order");
    if(!published(duplicate,out)||!equal(duplicate.records[0].left,{.25,.75,.375,.625})
       ||!equal(duplicate.records[2].left,{.25,.375,.75,.625})
       ||!equal(duplicate.records[1].left,{-1.,-2.})||!equal(duplicate.records[3].right,{1.,2.}))return 212;
    const std::int64_t columns[]={1,1,0,1},rows[]={0,3,4},empty_rows[]={0,0,0};
    const double zeros[]={0.,-0.,0.,-0.};
    for(bool stored:{false,true}) {
        Audit a;VariationInformation result;
        const VariationTableView input=stored?VariationTableView{zeros,columns,rows,2,2,4}
            :VariationTableView{nullptr,nullptr,empty_rows,2,2,0};
        if(!invoke(input,result,a,0))return 213;emit(a,stored?"stored-zero":"empty-batches");
        const std::uint64_t words[2]={};
        if(!published(a,result)||!a.records[0].left.empty()||!a.records[2].left.empty()
           ||a.records[1].left.size()!=2||a.records[3].left.size()!=2||std::memcmp(result.values.data(),words,16)
           ||result.captured_workers!=1||result.normalization_fp_status!=0)return 214;
    }
    return caller.finish()?0:215;
}
template<class Input>int failures(Input input) {
    for(bool null_log:{true,false}) {
        Caller32 caller("matmul-null-provider");if(!caller.good())return 220;
        Audit a;auto out=sentinel();const auto before=out;
        if(invoke(input,out,a,1,null_log,!null_log)||!a.records.empty()||!same(out,before))return 221;
        if(!caller.finish())return 222;
    }
    for(bool throwing:{false,true})for(unsigned phase:{1u,2u,3u,4u}) {
        Caller32 caller("matmul-provider-refusal");if(!caller.good())return 223;
        Audit a;if(throwing)a.throw_on=phase;else a.fail_on=phase;
        auto out=sentinel();const auto before=out;bool accepted=false,caught=false;
        try{accepted=invoke(input,out,a,1);}
        catch(const ProviderFailure& error){if(!throwing||error.identity!=&a)throw;caught=true;}
        emit(a,throwing?"throw":"false");
        if(accepted||caught!=throwing||a.wrong||a.records.size()!=phase||!same(out,before))return 224;
#if defined(SKIMAGEXX_VI_FENV32)
        std::fenv_t after{};if(!vi_sdk_controls::read32(after)||!a.effect_captured
           ||std::memcmp(&a.callback_effect,&after,32))return 225;
        std::cout<<"VI_MATMUL_CALLBACK_EFFECT phase="<<phase<<" throwing="<<throwing<<" entered=";
        vi_sdk_controls::words32(a.callback_effect);std::cout<<" after=";vi_sdk_controls::words32(after);std::cout<<'\n';
        if(std::fegetround()!=FE_DOWNWARD)return 226;
#endif
        if(!caller.finish())return 227;
    }
    return 0;
}
inline int invalid(VariationTableView table,LabelPairInput image) {
    Caller32 caller("matmul-invalid");if(!caller.good())return 230;
    for(int workers:{-1,1025})for(bool labels:{false,true}) {
        Audit a;auto out=sentinel();const auto before=out;
        if((labels?invoke(image,out,a,workers):invoke(table,out,a,workers))||!a.records.empty()||!same(out,before))return 231;
    }
    auto malformed=table;malformed.nnz=3;
    Audit a;auto out=sentinel();const auto before=out;
    if(invoke(malformed,out,a,1)||!a.records.empty()||!same(out,before))return 232;
    const std::int64_t bad_labels[]={-1,0};const LabelPairInput bad{bad_labels,bad_labels,2,nullptr,0,true};
    Audit b;if(invoke(bad,out,b,1)||!b.records.empty()||!same(out,before))return 233;
    return caller.finish()?0:234;
}
inline int fenv_refusals(VariationTableView table,LabelPairInput image) {
#if defined(SKIMAGEXX_VI_FENV32)
    for(unsigned mode=0;mode<5;++mode) {
        Caller32 caller("matmul-FENV-refusal");if(!caller.good())return 240;
        if(mode==0){if(std::fesetround(FE_DOWNWARD)!=0)return 241;}
        if(mode==1)_mm_setcsr(caller.mxcsr()|(1u<<15));
        if(mode==2)_mm_setcsr(caller.mxcsr()|(1u<<6));
        if(mode==3){if(caller.mxcsr()&(1u<<2))return 242;_mm_setcsr(caller.mxcsr()&~(1u<<9));}
        if(mode==4){if(caller.status()&FE_DIVBYZERO)return 243;const auto cw=std::uint16_t(caller.control()&~std::uint16_t(FE_DIVBYZERO));__asm__ volatile("fldcw %0"::"m"(cw):"memory");}
        std::fenv_t entered{};if(!vi_sdk_controls::read32(entered))return 244;
        for(bool labels:{false,true}) {
            Audit a;auto out=sentinel();const auto before=out;
            const bool accepted=labels?invoke(image,out,a,1):invoke(table,out,a,1);
            std::fenv_t after{};if(!vi_sdk_controls::read32(after))return 245;
            std::cout<<"VI_MATMUL_FENV_REFUSAL mode="<<mode<<" image="<<labels<<" entered=";
            vi_sdk_controls::words32(entered);std::cout<<" after=";vi_sdk_controls::words32(after);std::cout<<'\n';
            if(accepted||!a.records.empty()||!same(out,before)||std::memcmp(&entered,&after,32))return 246;
        }
        if(!caller.finish())return 247;
    }
#else
    (void)table;(void)image;std::cout<<"VI matmul SDK full32/nondefault controls=UNASSESSED_PLATFORM\n";
#endif
    return 0;
}
inline int run() {
    if(!vi_sdk_controls::layout()||skimagexx::get_num_threads()!=1)return 250;
    const auto before_counters=vi_sdk_controls::counters;
    Caller32 process("matmul-process");if(!process.good())return 251;
    const std::int64_t columns[]={0,1,0,1},rows[]={0,2,4};const double data[]={.25,.25,.25,.25};
    const VariationTableView table{data,columns,rows,2,2,4};
    const std::array<std::int64_t,4> original_columns={0,1,0,1};const std::array<std::int64_t,3> original_rows={0,2,4};
    const std::array<double,4> original_data={.25,.25,.25,.25};
    std::array<std::int64_t,16> truth{},test{},wide_truth{},wide_test{};
    for(std::size_t i=0;i<16;++i){truth[i]=(i/2)%2;test[i]=i%2;wide_truth[i]=truth[i]+64;wide_test[i]=test[i]+64;}
    const auto original_truth=truth,original_test=test,original_wide_truth=wide_truth,original_wide_test=wide_test;
    const LabelPairInput image{truth.data(),test.data(),16,nullptr,0,true};
    const LabelPairInput wide{wide_truth.data(),wide_test.data(),16,nullptr,0,true};
    if(int code=successes(table,image,wide))return code;
    if(int code=boundaries())return code;
    if(int code=failures(table))return code;
    if(int code=failures(image))return code;
    if(int code=invalid(table,image))return code;
    if(int code=fenv_refusals(table,image))return code;
    if(truth!=original_truth||test!=original_test||wide_truth!=original_wide_truth||wide_test!=original_wide_test
       ||std::memcmp(columns,original_columns.data(),sizeof(columns))||std::memcmp(rows,original_rows.data(),sizeof(rows))
       ||std::memcmp(data,original_data.data(),sizeof(data)))return 252;
    if(!process.finish()||skimagexx::get_num_threads()!=1)return 253;
    const auto& after=vi_sdk_controls::counters;
    unsigned expected_calls=39;
#if defined(SKIMAGEXX_VI_FENV32)
    expected_calls+=10;
    if(after.checked32-before_counters.checked32!=29||after.restores32-before_counters.restores32!=29
       ||after.reads32-before_counters.reads32!=105)return 254;
#endif
    if(totals.calls!=expected_calls||totals.callbacks!=92||totals.logs!=50||totals.dots!=42)return 255;
    std::cout<<"VI matmul SDK calls="<<totals.calls<<" callbacks="<<totals.callbacks<<" log_callbacks="<<totals.logs
        <<" dot_callbacks="<<totals.dots<<" caller32="<<after.checked32-before_counters.checked32
        <<" restores="<<after.restores32-before_counters.restores32<<" reads="<<after.reads32-before_counters.reads32
        <<" defaultworkers=1 originals=0 timing=0 provider=scalar-libm-linear-dot\n";
    return 0;
}
} // namespace vi_matmul_sdk_controls
