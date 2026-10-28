#pragma once
#include <cstddef>
namespace skimagexx::detail {
struct RollingRowPlan {
    double prefix=0, left=0, pair45=0, lane6=0;
    bool spine=false, add_lane7=false;
};
}
namespace skimagexx::simd {
std::size_t rolling_kernel_row(const double*,double*,std::size_t,
    const detail::RollingRowPlan&,bool,double,double,double);
void rolling_kernel_four_rows(const double*,double*,std::size_t,
    const detail::RollingRowPlan*,bool,double,double,double);
}
