#pragma once
#include "approximate_polygon.hpp"
#include <cmath>

namespace skimagexx::detail {
struct ApproximatePolygonSegment {
    double r0,c0,r1,c1,dr,dc,sine,cosine,offset;
};
inline double approximate_polygon_distance(const ApproximatePolygonInput& input,
    std::size_t index,const ApproximatePolygonSegment& segment) {
    const auto* point=input.data+std::ptrdiff_t(index)*input.row_stride;
    const auto row=point[0],column=point[input.column_stride];
    const auto dr0=row-segment.r0,dc0=column-segment.c0;
    const auto dr1=row-segment.r1,dc1=column-segment.c1;
    const auto projected0=dr0*segment.dr+dc0*segment.dc;
    const auto projected1=-dr1*segment.dr-dc1*segment.dc;
    if(projected0>0 && projected1>0)
        return std::abs(row*segment.cosine+column*segment.sine-segment.offset);
    const auto first=std::sqrt(dc0*dc0+dr0*dr0);
    const auto second=std::sqrt(dc1*dc1+dr1*dr1);
    return first<second?first:second;
}
}
namespace skimagexx::simd {
std::size_t approximate_polygon_distances(const detail::ApproximatePolygonInput&,
    std::size_t begin,std::size_t count,double* distances,
    const detail::ApproximatePolygonSegment&);
}
