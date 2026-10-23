#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
namespace skimagexx::detail {
struct DetBox {std::ptrdiff_t row,column,rows,columns;};
inline std::ptrdiff_t floor_div(std::ptrdiff_t x,std::ptrdiff_t y){return x/y-(x%y<0);}
inline void det_boxes(std::ptrdiff_t size,DetBox* boxes){
 const auto s2=floor_div(size-1,2),s3=floor_div(size,3),half=floor_div(s3,2);
 boxes[0]={-s3,-s3,s3,s3};boxes[1]={1,1,s3,s3};
 boxes[2]={-s3,1,s3,s3};boxes[3]={1,-s3,s3,s3};
 boxes[4]={-s3+1,-s2,2*s3-1,size};boxes[5]={-s3+1,-half,2*s3-1,s3};
 boxes[6]={-s2,-s3+1,size,2*s3-1};boxes[7]={-half,-s3+1,s3,2*s3-1};
}
inline double det_box_value(const double* integral,std::ptrdiff_t height,std::ptrdiff_t width,
 std::ptrdiff_t r,std::ptrdiff_t c,const DetBox& b){
 r=std::clamp(r+b.row,std::ptrdiff_t(0),height-1);c=std::clamp(c+b.column,std::ptrdiff_t(0),width-1);
 const auto r2=std::clamp(r+b.rows,std::ptrdiff_t(0),height-1),c2=std::clamp(c+b.columns,std::ptrdiff_t(0),width-1);
 return std::max(0.,((integral[r*width+c]+integral[r2*width+c2])-integral[r*width+c2])-integral[r2*width+c]);
}
inline double det_box_response(const double* b,double weight){
 const double xy=-(((b[2]+b[3])-b[0])-b[1])*weight;
 const double xx=-(b[4]-3*b[5])*weight,yy=-(b[6]-3*b[7])*weight;
 return xx*yy-.81*(xy*xy);
}
}
namespace skimagexx::simd {
std::size_t hessian_det_boxes(const double* const*,const double* const*,double*,std::size_t,std::size_t,const detail::DetBox*,double);
std::size_t hessian_det_fields(const void* const*,void*,std::size_t,std::size_t,std::size_t,bool,bool&);
}
