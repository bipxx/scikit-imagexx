// Ordered circle AA recurrence from scikit-image0.26.0 _draw.pyx.
// Copyright scikit-image contributors, BSD-3-Clause; attribution/license:
// third_party/scikit-image-LICENSE.txt. Preserve source order and FP association.
#include <skimagexx/circle_perimeter_aa.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace skimagexx::detail {
namespace {
using Int = std::intptr_t;
// Fixed-size independent stores let baseline compilers vectorize contiguous
// output. The recurrence and clipped compaction retain literal serial order.
template<bool Fill, std::size_t N>
void block(const CirclePerimeterAARequest& q, const Int (&rr)[N], const Int (&cc)[N],
           const double (&ww)[N], DrawCoordinatePair output, double* weights,
           std::ptrdiff_t weight_step, std::size_t& count) noexcept {
    if (!q.clipped) {
        if constexpr (Fill) {
            for (std::size_t i=0; i<N; ++i)
                output.rows[std::ptrdiff_t(count+i)*output.row_step] = rr[i]+q.row;
            for (std::size_t i=0; i<N; ++i)
                output.columns[std::ptrdiff_t(count+i)*output.column_step] = cc[i]+q.column;
            for (std::size_t i=0; i<N; ++i)
                weights[std::ptrdiff_t(count+i)*weight_step] = ww[i];
        }
        count += N;
    } else {
        for (std::size_t i=0; i<N; ++i) {
            const auto row=rr[i]+q.row, column=cc[i]+q.column;
            if (row>=0 && row<q.height && column>=0 && column<q.width) {
                if constexpr (Fill) {
                    output.rows[std::ptrdiff_t(count)*output.row_step] = row;
                    output.columns[std::ptrdiff_t(count)*output.column_step] = column;
                    weights[std::ptrdiff_t(count)*weight_step] = ww[i];
                }
                ++count;
            }
        }
    }
}

template<bool Fill>
std::size_t visit(const CirclePerimeterAA& prepared, DrawCoordinatePair output,
                  double* weights, std::ptrdiff_t weight_step) noexcept {
    const auto& q=prepared.request;
    auto r=q.radius;
    Int c=0;
    double previous=0;
    std::size_t count=0;
    const Int initial_rows[]{r,c,r,c,-r,-c,-r,-c};
    const Int initial_columns[]{c,r,-c,-r,c,r,-c,-r};
    const double initial_weights[]{1,1,1,1,1,1,1,1};
    block<Fill>(q,initial_rows,initial_columns,initial_weights,output,weights,weight_step,count);
    while (r>c+1) {
        ++c;
        double distance=std::sqrt(double(prepared.squared_radius-c*c));
        distance=std::ceil(distance)-distance;
        if (distance<previous) --r;
        const Int rr[]{r,r-1,c,c,r,r-1,c,c,-r,1-r,-c,-c,-r,1-r,-c,-c};
        const Int cc[]{c,c,r,r-1,-c,-c,-r,1-r,c,c,r,r-1,-c,-c,-r,1-r};
        const double complement=1-distance;
        const double ww[]{complement,distance,complement,distance,complement,distance,complement,distance,
                          complement,distance,complement,distance,complement,distance,complement,distance};
        block<Fill>(q,rr,cc,ww,output,weights,weight_step,count);
        previous=distance;
    }
    return count;
}
}

bool prepare_circle_perimeter_aa(const CirclePerimeterAARequest& q, CirclePerimeterAA& prepared) noexcept {
    constexpr auto high=std::numeric_limits<Int>::max(), low=std::numeric_limits<Int>::min();
    if (q.radius==low || (q.radius>0 && q.radius>high/q.radius)) return false;
    const auto extent=q.radius<0 ? -q.radius : q.radius;
    if (q.row<low+extent || q.row>high-extent || q.column<low+extent || q.column>high-extent) return false;
    constexpr auto maximum_count=std::min(
        std::min(std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(Int),
                 std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)),
        std::numeric_limits<std::size_t>::max()/(2*sizeof(Int)+sizeof(double)));
    const auto iterations=q.radius>1 ? std::size_t(q.radius-1) : std::size_t(0);
    if (iterations>(maximum_count-8)/16) return false;
    prepared={q,8+16*iterations,q.radius>0 ? q.radius*q.radius : Int(0)};
    return true;
}

std::size_t circle_perimeter_aa_count(const CirclePerimeterAA& prepared, int workers) noexcept {
    (void)workers; // Caller snapshot; recurrence-dependent, no private worker pool.
    return visit<false>(prepared,{},nullptr,0);
}

void circle_perimeter_aa_fill(const CirclePerimeterAA& prepared, DrawCoordinatePair output,
                              double* weights, std::ptrdiff_t weight_step, int workers) noexcept {
    (void)workers;
    visit<true>(prepared,output,weights,weight_step);
}
}
