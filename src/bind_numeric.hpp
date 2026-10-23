#pragma once
#include "skimagexx/convert.hpp"
#include <pybind11/numpy.h>
namespace skimagexx::bindings {
inline PixelType real_pixel_type(pybind11::dtype dtype){
    const auto bytes=dtype.itemsize();
    if(dtype.kind()=='b')return PixelType::boolean;
    if(dtype.kind()=='f' && bytes==4)return PixelType::float32;
    if(dtype.kind()=='f' && bytes==8)return PixelType::float64;
    if(dtype.kind()=='i'){if(bytes==1)return PixelType::int8;if(bytes==2)return PixelType::int16;if(bytes==4)return PixelType::int32;if(bytes==8)return PixelType::int64;}
    if(dtype.kind()=='u'){if(bytes==1)return PixelType::uint8;if(bytes==2)return PixelType::uint16;if(bytes==4)return PixelType::uint32;if(bytes==8)return PixelType::uint64;}
    throw pybind11::type_error("real image input required");
}
}
