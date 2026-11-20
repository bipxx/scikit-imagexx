#pragma once
// Test-only thin use of the existing admitted97ff/0c40 restoration helper.
// Caller authenticates that fixed library before this process is launched.
#include <cassert>
#include <cfenv>
#include <cstdint>
#include <cstring>
#if defined(__linux__) && defined(__x86_64__)
#include <dlfcn.h>
struct GoldenFPEnvelope {
    void* library;std::fenv_t saved;
    using Restore=int(*)(const void*,std::size_t);Restore restore;
    explicit GoldenFPEnvelope(const char* path) {
        static_assert(sizeof(std::fenv_t)==32);
        library=dlopen(path,RTLD_NOW|RTLD_LOCAL);assert(library);
        auto layout=reinterpret_cast<int(*)(std::uint32_t*,std::size_t)>(dlsym(library,"reference_fenv_layout"));
        restore=reinterpret_cast<Restore>(dlsym(library,"reference_fenv_restore"));assert(layout&&restore);
        std::uint32_t actual[8],expected[]={32,0,4,8,12,20,28,8};assert(!layout(actual,8)&&!std::memcmp(actual,expected,sizeof(actual)));
        assert(!std::fegetenv(&saved));
    }
    void finish(){assert(!restore(&saved,32));std::fenv_t after;assert(!std::fegetenv(&after)&&!std::memcmp(&saved,&after,32));}
    ~GoldenFPEnvelope(){dlclose(library);}
};
#else
// Windows proof is compilation/value scope only; no LinuxFEnv32 claim.
struct GoldenFPEnvelope {explicit GoldenFPEnvelope(const char*){}void finish(){}};
#endif
