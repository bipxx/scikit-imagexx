// Focused private target: this includes the production finite TU exactly once
// to exercise its internal helper without adding an exported test hook. Link
// the unchanged stage/reduction/image foundation and the renamed af7 control
// into THIS executable; do not also link a candidate tv_denoise object/library. Other native tests
// and the ordinary private module compile/link the candidate TU once instead.
#include "../src/tv_denoise.cpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// Exact frozen af7 source is compiled in a separate control TU with only its
// three finite entry names mechanically renamed. No control owns worker state.
namespace skimagexx {
const char* tv_denoise_zero_baseline_ineligibility(const TvConstView&,const TvDenoiseWorkspace&,
    const TvDenoiseParameters&,TvDenoiseTrace*,std::size_t) noexcept;
TvDenoiseResult tv_denoise_zero_baseline_finite_into(const TvConstView&,const TvDenoiseWorkspace&,
    const TvDenoiseParameters&,TvDenoiseTrace*,std::size_t);
}

namespace {
std::uint64_t checks=0,helper_cases=0,page_cases=0,guard_rejections=0,accepted=0;

void require(bool value,const char* why) {
    if(!value)throw std::runtime_error(why);
    ++checks;
}

struct SavedEnvironment {
    std::fenv_t value{};
    SavedEnvironment(){require(std::fegetenv(&value)==0,"cannot capture FP environment");}
    ~SavedEnvironment(){std::fesetenv(&value);}
};

template<class T> std::vector<unsigned char> bytes_of(const std::vector<T>& values) {
    std::vector<unsigned char> result(values.size()*sizeof(T));
    if(!result.empty())std::memcpy(result.data(),values.data(),result.size());
    return result;
}

// The expected result follows a controlled one-bit mutation, not an alternate
// implementation of the candidate's word/OR algorithm. Nonzero redzones make
// accidental reads outside the declared byte count observable as well.
void helper_controls() {
    alignas(64) std::array<unsigned char,512> storage{};
    require(skimagexx::zero_bytes(nullptr,0),"empty scan must not dereference");
    for(std::size_t length=0;length<=257;++length) {
        for(std::size_t alignment=0;alignment<64;++alignment) {
            storage.fill(0xa5);
            auto* data=storage.data()+64+alignment;
            std::fill(data,data+length,0);
            const auto pristine=storage;
            const int flags=std::fetestexcept(FE_ALL_EXCEPT);
            const int rounding=std::fegetround();
            require(skimagexx::zero_bytes(data,length),"all-zero byte range rejected");
            ++helper_cases;
            for(std::size_t position=0;position<length;++position) {
                for(unsigned bit=0;bit<8;++bit) {
                    data[position]=static_cast<unsigned char>(1u<<bit);
                    require(!skimagexx::zero_bytes(data,length),"nonzero byte was accepted");
                    require(data[position]==static_cast<unsigned char>(1u<<bit),"scan modified dirty byte");
                    data[position]=0;
                    ++helper_cases;
                }
            }
            require(storage==pristine,"scan modified active bytes or redzones");
            require(flags==std::fetestexcept(FE_ALL_EXCEPT)&&rounding==std::fegetround(),"integer scan changed FP state");
        }
    }
    // Nonzero bit patterns include signed zero and signaling-NaN words. They
    // are never interpreted as floating values by this test or the helper.
    for(std::uint64_t pattern:std::array<std::uint64_t,5>{0x80000000ull,0x7f800001ull,
            0x8000000000000000ull,0x7ff0000000000001ull,0xffffffffffffffffull}) {
        storage.fill(0);
        std::memcpy(storage.data()+3,&pattern,sizeof(pattern));
        const auto before=storage;
        std::feclearexcept(FE_ALL_EXCEPT);
        require(!skimagexx::zero_bytes(storage.data()+1,31),"nonzero special word accepted");
        require(std::fetestexcept(FE_ALL_EXCEPT)==0&&storage==before,"special-word scan changed flags or bytes");
    }
    storage.fill(0);
    for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        require(std::fesetround(mode)==0,"cannot select helper rounding control");
        std::feclearexcept(FE_ALL_EXCEPT);
        require(std::feraiseexcept(FE_INEXACT)==0,"cannot set prior FP flag");
        const int flags=std::fetestexcept(FE_ALL_EXCEPT);
        require(skimagexx::zero_bytes(storage.data()+1,255),"rounding changed zero result");
        require(std::fegetround()==mode&&std::fetestexcept(FE_ALL_EXCEPT)==flags,"scan did not preserve prior FP state");
    }
    require(std::fesetround(FE_TONEAREST)==0,"cannot restore nearest rounding");
    std::feclearexcept(FE_ALL_EXCEPT);
}

// Valid byte regions terminate immediately beside inaccessible pages. The
// caller never supplies an out-of-bounds region or requests an invalid read.
class GuardedPage {
    void* allocation_=nullptr;
    std::size_t allocation_bytes_=0;
public:
    unsigned char* data=nullptr;
    std::size_t size=0;
    GuardedPage() {
#if defined(_WIN32)
        SYSTEM_INFO info{};GetSystemInfo(&info);size=info.dwPageSize;
#else
        const long page=sysconf(_SC_PAGESIZE);
        require(page>0,"cannot determine system page size");size=static_cast<std::size_t>(page);
#endif
        require(size>=257&&size<=std::numeric_limits<std::size_t>::max()/3,"invalid page size");
        allocation_bytes_=3*size;
#if defined(_WIN32)
        allocation_=VirtualAlloc(nullptr,allocation_bytes_,MEM_RESERVE|MEM_COMMIT,PAGE_NOACCESS);
        require(allocation_!=nullptr,"guard-page allocation failed");
        data=static_cast<unsigned char*>(allocation_)+size;
        DWORD previous=0;
        if(!VirtualProtect(data,size,PAGE_READWRITE,&previous)) {
            VirtualFree(allocation_,0,MEM_RELEASE);allocation_=nullptr;
            throw std::runtime_error("guard-page protection failed");
        }
#else
        allocation_=mmap(nullptr,allocation_bytes_,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(allocation_==MAP_FAILED){allocation_=nullptr;throw std::runtime_error("guard-page allocation failed");}
        data=static_cast<unsigned char*>(allocation_)+size;
        if(mprotect(data,size,PROT_READ|PROT_WRITE)!=0) {
            munmap(allocation_,allocation_bytes_);allocation_=nullptr;
            throw std::runtime_error("guard-page protection failed");
        }
#endif
        std::memset(data,0,size);
    }
    GuardedPage(const GuardedPage&)=delete;
    GuardedPage& operator=(const GuardedPage&)=delete;
    ~GuardedPage() {
        if(!allocation_)return;
#if defined(_WIN32)
        VirtualFree(allocation_,0,MEM_RELEASE);
#else
        munmap(allocation_,allocation_bytes_);
#endif
    }
};

void page_controls() {
    GuardedPage page;
    std::vector<std::size_t> lengths;
    for(std::size_t n=0;n<=257;++n)lengths.push_back(n);
    for(std::size_t n:{std::size_t{511},std::size_t{512},std::size_t{513},std::size_t{1023},
            std::size_t{1024},std::size_t{1025},page.size-1,page.size})if(n<=page.size)lengths.push_back(n);
    std::sort(lengths.begin(),lengths.end());lengths.erase(std::unique(lengths.begin(),lengths.end()),lengths.end());
    for(auto length:lengths)for(bool end:{false,true}) {
        auto* data=end?page.data+page.size-length:page.data;
        const int flags=std::fetestexcept(FE_ALL_EXCEPT);
        require(skimagexx::zero_bytes(data,length),"valid page-edge zero range rejected");
        ++page_cases;
        if(length)for(std::size_t position:{std::size_t{0},length-1}) {
            data[position]=0x80;
            require(!skimagexx::zero_bytes(data,length),"page-edge dirty byte missed");
            require(data[position]==0x80,"page-edge scan wrote input");
            data[position]=0;++page_cases;
        }
        require(std::all_of(page.data,page.data+page.size,[](unsigned char x){return x==0;}),"page-edge scan wrote other bytes");
        require(std::fetestexcept(FE_ALL_EXCEPT)==flags,"page-edge scan changed FP flags");
    }
}

template<class T> struct Fixture {
    std::vector<T> image,p,g,d,out,squared,norm;
    std::vector<skimagexx::TvDenoiseTrace> trace;
    skimagexx::TvConstView input;
    skimagexx::TvDenoiseWorkspace work;
    explicit Fixture(const std::vector<std::size_t>& shape) {
        using namespace skimagexx;
        std::size_t n=1;for(auto size:shape)n*=size;
        std::vector<std::ptrdiff_t> strides(shape.size());std::size_t step=sizeof(T);
        for(std::size_t axis=shape.size();axis-->0;){strides[axis]=static_cast<std::ptrdiff_t>(step);step*=shape[axis];}
        const auto type=sizeof(T)==4?TvScalarType::float32:TvScalarType::float64;
        image.assign(n,T(0));p.assign(n*shape.size(),T(0));g=p;d=image;
        out.assign(n,T(17));squared.assign(n,T(18));norm.assign(n,T(19));
        trace.assign(2,TvDenoiseTrace{1,2,3,4,5,false,false});
        input={image.data(),type,shape,strides};
        work={{p.data(),type,shape},{g.data(),type,shape},{d.data(),type,shape,strides},
            {out.data(),type,shape,strides},{squared.data(),type,shape,strides},{norm.data(),type,shape,strides},
            {p.data(),type,shape},{g.data(),type,shape},{out.data(),type,shape,strides},{norm.data(),type,shape,strides}};
    }
    auto snapshot()const {
        return std::array<std::vector<unsigned char>,8>{bytes_of(image),bytes_of(p),bytes_of(g),bytes_of(d),
            bytes_of(out),bytes_of(squared),bytes_of(norm),bytes_of(trace)};
    }
};

template<class T> void guard_controls(skimagexx::TvReductionProfile profile) {
    using namespace skimagexx;
    for(const auto& shape:std::vector<std::vector<std::size_t>>{{1},{17},{3,11}}) {
        Fixture<T> f(shape);
        const TvDenoiseParameters parameters{.5,0.,2,{profile,8192}};
        require(tv_denoise_ineligibility(f.input,f.work,parameters,f.trace.data(),f.trace.size())==nullptr,"valid guard fixture rejected");
        for(auto* buffer:{&f.p,&f.g,&f.d}) {
            auto* raw=reinterpret_cast<unsigned char*>(buffer->data());
            const std::size_t size=buffer->size()*sizeof(T);
            for(std::size_t position=0;position<size;++position)for(unsigned bit=0;bit<8;++bit) {
                raw[position]=static_cast<unsigned char>(1u<<bit);
                const auto before=f.snapshot();
                TvDenoiseResult result{1234,true};
                std::array<unsigned char,sizeof(result)> result_before{};
                std::memcpy(result_before.data(),&result,sizeof(result));
                const int flags=std::fetestexcept(FE_ALL_EXCEPT);
                const int rounding=std::fegetround();
                const char* baseline=tv_denoise_zero_baseline_ineligibility(f.input,f.work,parameters,f.trace.data(),f.trace.size());
                const char* reason=detail::tv_denoise_try_finite_into(f.input,f.work,parameters,result,f.trace.data(),f.trace.size());
                require(reason&&std::strcmp(reason,"TV finite p/g/d must start positive zero")==0,"dirty workspace reason changed");
                require(baseline&&std::strcmp(reason,baseline)==0,"af7 guard reason differs");
                require(!std::memcmp(result_before.data(),&result,sizeof(result)),"rejected guard wrote result descriptor");
                bool rejected=false;
                try{tv_denoise_finite_into(f.input,f.work,parameters,f.trace.data(),f.trace.size());}
                catch(const std::invalid_argument& error){rejected=std::strcmp(error.what(),reason)==0;}
                require(rejected,"public finite entry lost exact dirty-workspace refusal");
                require(f.snapshot()==before,"rejected finite entry wrote image/workspaces/trace");
                require(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding,"rejected finite entry changed FP state");
                raw[position]=0;++guard_rejections;
            }
        }
        // The all-zero fixture must still enter and execute the real finite
        // algorithm; one iteration aliases only after all updates have run.
        for(std::size_t iterations:{std::size_t{1},std::size_t{2}}) {
            Fixture<T> good(shape),baseline(shape);
            const TvDenoiseParameters p{.5,0.,iterations,{profile,8192}};
            const auto original_image=bytes_of(good.image);
            std::feclearexcept(FE_ALL_EXCEPT);
            const auto control=tv_denoise_zero_baseline_finite_into(baseline.input,baseline.work,p,baseline.trace.data(),baseline.trace.size());
            const int baseline_flags=std::fetestexcept(FE_ALL_EXCEPT);
            std::feclearexcept(FE_ALL_EXCEPT);
            const auto result=tv_denoise_finite_into(good.input,good.work,p,good.trace.data(),good.trace.size());
            require(std::fetestexcept(FE_ALL_EXCEPT)==baseline_flags,"accepted af7 FP flags differ");
            require(result.iterations==control.iterations&&result.input_alias==control.input_alias,"accepted af7 result metadata differs");
            const auto actual_bytes=good.snapshot(),control_bytes=baseline.snapshot();
            for(std::size_t k=0;k<7;++k)require(actual_bytes[k]==control_bytes[k],"accepted af7 image/workspace bytes differ");
            require(result.iterations==iterations&&result.input_alias==(iterations==1),"positive fixture iteration/alias changed");
            require(bytes_of(good.image)==original_image,"accepted finite call wrote input");
            for(T x:good.out)require(x==(iterations==1?T(17):T(0)),"one-body output exposure or constant result changed");
            for(T x:good.norm)require(x==T(1),"accepted call did not perform denominator update");
            for(std::size_t i=0;i<iterations;++i) {
                const auto& row=good.trace[i];
                const auto& prior=baseline.trace[i];
                require(row.square_sum_bits==prior.square_sum_bits&&row.norm_sum_bits==prior.norm_sum_bits&&row.energy_bits==prior.energy_bits&&row.difference_bits==prior.difference_bits&&row.threshold_bits==prior.threshold_bits&&row.energy_is_float32==prior.energy_is_float32&&row.stopped==prior.stopped,"accepted af7 trace fields differ");
                require(row.square_sum_bits==0&&row.norm_sum_bits==0&&row.energy_bits==0&&row.difference_bits==0&&row.threshold_bits==0&&!row.stopped,"constant finite trace changed");
                require(row.energy_is_float32==(sizeof(T)==4&&profile==TvReductionProfile::numpy_254_aligned_contiguous),"finite energy profile changed");
            }
            ++accepted;
        }
    }
}
} // namespace

int main() {
    try {
        SavedEnvironment environment;
        require(skimagexx::get_num_threads()==1,"default worker owner is not1");
        helper_controls();page_controls();
        for(int workers:{1,4}) {
#ifndef _OPENMP
            if(workers==4) {
                bool rejected=false;
                try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}
                require(rejected&&skimagexx::get_num_threads()==1,"portable worker4 must explicitly reject");
                continue;
            }
#endif
            skimagexx::set_num_threads(workers);
            require(skimagexx::get_num_threads()==workers,"selected worker state differs");
            for(auto profile:{skimagexx::TvReductionProfile::numpy_1244_buffered,
                    skimagexx::TvReductionProfile::numpy_254_aligned_contiguous}) {
                guard_controls<float>(profile);guard_controls<double>(profile);
            }
        }
        skimagexx::set_num_threads(1);
        require(helper_cases>0&&page_cases>0&&guard_rejections>0&&accepted>0,"empty focused test");
        std::cout<<"TV positive-zero "<<checks<<" checks; helper="<<helper_cases
            <<" page="<<page_cases<<" rejected="<<guard_rejections<<" accepted="<<accepted<<"\n";
    } catch(const std::exception& error) {
        skimagexx::set_num_threads(1);
        std::cerr<<"TV positive-zero FAILED after "<<checks<<": "<<error.what()<<'\n';
        return 1;
    }
}
