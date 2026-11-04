// Ordered circle recurrence ported from scikit-image0.26.0 _draw.pyx.
// Copyright scikit-image contributors. BSD-3-Clause attribution/license:
// third_party/scikit-image-LICENSE.txt. No changes to algorithm tie/order rules.
#include <skimagexx/circle_perimeter.hpp>
#include <limits>

namespace skimagexx::detail {
namespace {
using Int=std::intptr_t;
// Shared by exact count and fill, so ordering/clipping cannot diverge. The
// preparation boundary proves all integer recurrence and translation sums.
template<class Emit> void visit_circle(const CirclePerimeter& prepared,Emit emit) noexcept {
    const auto& q=prepared.request;
    auto r=q.radius;
    Int c=0;
    if(r<0)return;
    auto d=q.method==CirclePerimeterMethod::bresenham?Int(3)-2*r:r-1;
    while(r>=c) {
        const Int rr[]{r,-r,r,-r,c,-c,c,-c};
        const Int cc[]{c,c,-c,-c,r,r,-r,-r};
        for(unsigned i=0;i<8;++i) {
            const auto row=rr[i]+q.row,column=cc[i]+q.column;
            if(!q.clipped||(row>=0&&row<q.height&&column>=0&&column<q.width))emit(row,column);
        }
        if(q.method==CirclePerimeterMethod::bresenham) {
            if(d<0)d+=4*c+6;
            else {d+=4*(c-r)+10;--r;}
            ++c;
        } else {
            if(d>=2*(c-1)) {d-=2*c;++c;}
            else if(d<=2*(q.radius-r)) {d+=2*r-1;--r;}
            else {d+=2*(r-c-1);--r;++c;}
        }
    }
}
}

bool prepare_circle_perimeter(const CirclePerimeterRequest& q,CirclePerimeter& prepared) noexcept {
    if(q.method!=CirclePerimeterMethod::bresenham&&q.method!=CirclePerimeterMethod::andres)return false;
    constexpr auto maximum=std::numeric_limits<Int>::max();
    constexpr auto minimum=std::numeric_limits<Int>::min();
    // Conservative arithmetic partition: eight output points per step, exact
    // intp-sized arrays addressable by signed offsets, and ample decision headroom.
    constexpr Int radius_limit=(maximum/Int(sizeof(Int))-8)/8;
    if(q.radius < -radius_limit||q.radius>radius_limit)return false;
    const auto extent=q.radius>0?q.radius:Int(0);
    if(q.row<minimum+extent||q.row>maximum-extent||q.column<minimum+extent||q.column>maximum-extent)return false;
    prepared={q,q.radius<0?std::size_t(0):std::size_t(8)*(std::size_t(q.radius)+1)};
    return true;
}

std::size_t circle_perimeter_count(const CirclePerimeter& prepared,int workers) noexcept {
    (void)workers;
    std::size_t count=0;
    visit_circle(prepared,[&](Int,Int){++count;});
    return count;
}

void circle_perimeter_fill(const CirclePerimeter& prepared,DrawCoordinatePair output,int workers) noexcept {
    (void)workers;
    std::size_t index=0;
    visit_circle(prepared,[&](Int row,Int column){
        output.rows[std::ptrdiff_t(index)*output.row_step]=row;
        output.columns[std::ptrdiff_t(index)*output.column_step]=column;
        ++index;
    });
}
}
