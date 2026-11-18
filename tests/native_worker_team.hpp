#pragma once

#ifdef _OPENMP
#include <algorithm>
#include <omp.h>
#endif

namespace native_test {

inline bool matches_worker_team(int actual, int requested) {
#ifdef _OPENMP
    int limit = requested;
    // OpenMP 2 runtimes expose dynamic teams without a thread-limit query.
#if _OPENMP >= 200805
    limit = std::min(limit, omp_get_thread_limit());
#endif
    if (omp_get_dynamic()) {
        return actual >= 1 && actual <= limit;
    }
    return actual == limit;
#else
    return requested == 1 && actual == 1;
#endif
}

} // namespace native_test
