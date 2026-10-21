#pragma once
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace skimagexx::detail {
constexpr std::size_t parallel_min = 256*1024;
inline int active_workers(std::size_t count) { return count >= parallel_min ? get_num_threads() : 1; }
template<class F> void partition(std::size_t length, int threads, F body) {
#ifdef _OPENMP
    if (threads > 1) {
        #pragma omp parallel num_threads(threads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto team = static_cast<std::size_t>(omp_get_num_threads());
            const auto block = length/team, extra = length%team;
            const auto begin = block*id+std::min(id,extra);
            body(begin,begin+block+(id < extra),id);
        }
        return;
    }
#else
    (void)threads;
#endif
    body(std::size_t{0},length,std::size_t{0});
}
} // namespace skimagexx::detail
