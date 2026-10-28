#pragma once
#include "subdivide_polygon.hpp"

namespace skimagexx::detail {
struct SubdividePolygonWeights { unsigned count;double even[5],odd[5]; };
inline const SubdividePolygonWeights* subdivide_polygon_weights(unsigned degree) {
    static constexpr SubdividePolygonWeights values[]={
        {2,{1./2,1./2},{1./2,1./2}},
        {2,{3./4,1./4},{1./4,3./4}},
        {3,{1./8,6./8,1./8},{0,4./8,4./8}},
        {3,{5./16,10./16,1./16},{1./16,10./16,5./16}},
        {4,{1./32,15./32,15./32,1./32},{0,6./32,20./32,6./32}},
        {4,{7./64,35./64,21./64,1./64},{1./64,21./64,35./64,7./64}},
        {5,{1./128,28./128,70./128,28./128,1./128},{0,8./128,56./128,56./128,8./128}}
    };
    switch(degree){case 1:case 2:case 3:case 4:case 5:case 6:case 7:return values+degree-1;default:return nullptr;}
}
inline std::size_t subdivide_polygon_index(std::size_t count,std::size_t point,
    unsigned tap,unsigned length,bool circular) {
    auto index=std::ptrdiff_t(point)+std::ptrdiff_t(circular?(length-1)/2:length-1)-std::ptrdiff_t(tap);
    if(circular&&(index<0||std::size_t(index)>=count)) {
        index%=std::ptrdiff_t(count);if(index<0)index+=std::ptrdiff_t(count);
    }
    return std::size_t(index);
}
inline void subdivide_polygon_point(const SubdividePolygonInput& input,
    std::size_t point,const SubdividePolygonWeights& weights,bool circular,double* output) {
    double even0=0.,even1=0.,odd0=0.,odd1=0.;
    for(unsigned k=0;k<weights.count;++k) {
        const auto index=subdivide_polygon_index(input.count,point,k,weights.count,circular);
        const auto* value=input.data+std::ptrdiff_t(index)*input.row_stride;
        even0+=weights.even[k]*value[0];even1+=weights.even[k]*value[input.column_stride];
        odd0+=weights.odd[k]*value[0];odd1+=weights.odd[k]*value[input.column_stride];
    }
    output[point*4]=odd0;output[point*4+1]=odd1;
    output[point*4+2]=even0;output[point*4+3]=even1;
}
}
namespace skimagexx::simd {
std::size_t subdivide_polygon_points(const detail::SubdividePolygonInput&,
    std::size_t begin,std::size_t count,const detail::SubdividePolygonWeights&,
    bool circular,double* output);
}
