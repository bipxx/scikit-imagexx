#pragma once
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#ifdef __linux__
#include <cfenv>
#endif
#endif

namespace skimagexx::detail {
constexpr std::size_t parallel_min = 256*1024;
inline int active_workers(std::size_t count) { return count >= parallel_min ? get_num_threads() : 1; }
template<class F> void partition(std::size_t length, int threads, F body) {
#ifdef _OPENMP
    if (threads > 1) {
#ifdef __linux__
        // The dynamic scheduler can raise FE_INEXACT before entering a team.
        // Mask traps during scheduling and preserve only the primary body state.
        std::fenv_t caller_before, caller_after;
        const bool isolate = omp_get_dynamic() && std::feholdexcept(&caller_before) == 0;
        bool caller_after_saved = false;
#endif
        #pragma omp parallel num_threads(threads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto team = static_cast<std::size_t>(omp_get_num_threads());
            const auto block = length/team, extra = length%team;
            const auto begin = block*id+std::min(id,extra);
#ifdef __linux__
            std::fenv_t worker_before;
            const bool worker_saved = isolate && id != 0 && std::fegetenv(&worker_before) == 0;
            if (isolate && (id == 0 || worker_saved)) {
                std::fesetenv(&caller_before);
            }
#endif
            body(begin,begin+block+(id < extra),id);
#ifdef __linux__
            if (isolate && id == 0) {
                caller_after_saved = std::feholdexcept(&caller_after) == 0;
            } else if (worker_saved) {
                std::fesetenv(&worker_before);
            }
#endif
        }
#ifdef __linux__
        if (isolate) std::fesetenv(caller_after_saved ? &caller_after : &caller_before);
#endif
        return;
    }
#else
    (void)threads;
#endif
    body(std::size_t{0},length,std::size_t{0});
}
} // namespace skimagexx::detail
