#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#include "bind_numeric.hpp"
#include "half.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstring>
namespace py=pybind11;
namespace {
void buffer(const py::array& a,int rank) {
    if(a.ndim()!=rank)throw py::value_error("Buffer has wrong number of dimensions (expected "+std::to_string(rank)+", got "+std::to_string(a.ndim())+")");
    if(!(a.flags()&py::array::c_style))throw py::value_error("ndarray is not C-contiguous");
    if(!a.writeable())throw py::value_error("buffer source array is read-only");
    if(!a.dtype().attr("isnative").cast<bool>())throw py::value_error("Big-endian buffer not supported on little-endian compiler");
}
int shift(py::handle value) {
        if(!PyNumber_Check(value.ptr()))throw py::type_error("an integer is required");
        const auto integer=py::reinterpret_steal<py::object>(PyNumber_Long(value.ptr()));
        if(!integer)throw py::error_already_set();
        const auto number=PyLong_AsLongLong(integer.ptr());
        if(PyErr_Occurred()){PyErr_Clear();PyErr_SetString(PyExc_OverflowError,"Python int too large to convert to C long");throw py::error_already_set();}
        if(number<-128 || number>127){PyErr_SetString(PyExc_OverflowError,"value too large to convert to signed char");throw py::error_already_set();}
        return int(number);
}
template<class T> bool signaling(T value) {
    using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;U word;std::memcpy(&word,&value,sizeof(T));
    if constexpr(sizeof(T)==4)return (word&0x7f800000)==0x7f800000 && (word&0x007fffff) && !(word&0x00400000);
    else return (word&0x7ff0000000000000ULL)==0x7ff0000000000000ULL && (word&0x000fffffffffffffULL) && !(word&0x0008000000000000ULL);
}
template<class T> void positive(const void* source,std::uint8_t* target,std::size_t begin,std::size_t count) {
    const auto* data=static_cast<const T*>(source);for(auto i=begin;i<count;++i)target[i]=data[i]>T(0) ? 255:0;
}
}
void bind_rank(py::module_& module) {
    const auto numpy=py::module_::import("numpy");
    module.def("rank_convert",[numpy](py::array image) {
        const bool half=image.dtype().kind()=='f' && image.itemsize()==2;
        const auto dtype=py::dtype::from_args(image.dtype().attr("type"));
        auto input=numpy.attr("require")(image,py::arg("dtype")=dtype,py::arg("requirements")=py::make_tuple("C","A")).cast<py::array>();
        py::array_t<std::uint8_t> result(std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        skimagexx::ConversionInfo info;skimagexx::RankHalfConversion half_info;
        const auto source_type=half ? skimagexx::PixelType::uint16:skimagexx::bindings::real_pixel_type(dtype);
        {
            py::gil_scoped_release release;
            if(half)half_info=skimagexx::rank_half_u8(static_cast<const std::uint16_t*>(input.data()),result.mutable_data(),std::size_t(input.size()));
            else info=skimagexx::convert_into(input.data(),result.mutable_data(),std::size_t(input.size()),source_type,skimagexx::PixelType::uint8);
        }
        if(info.nan_to_integer && input.dtype().kind()=='f') {
            if(input.itemsize()==4){const auto* p=static_cast<const float*>(input.data());for(py::ssize_t i=0;i<input.size();++i)half_info.invalid_multiply|=signaling(p[i]);}
            else if(input.itemsize()==8){const auto* p=static_cast<const double*>(input.data());for(py::ssize_t i=0;i<input.size();++i)half_info.invalid_multiply|=signaling(p[i]);}
        }
        py::object maximum=input.dtype().kind()=='u' ? py::object(py::int_(info.maximum_unsigned)):py::object(py::int_(info.maximum_signed));
        return py::make_tuple(result,info.downcast_without_scaling,maximum,info.nan_to_integer || half_info.nan,info.floating_overflow || half_info.overflow,info.floating_underflow,info.invalid_float_cast,half_info.invalid_multiply);
    });
    module.def("rank_positive",[numpy](py::array image) {
        const auto dtype=py::dtype::from_args(image.dtype().attr("newbyteorder")("="));
        auto input=numpy.attr("require")(image,py::arg("dtype")=dtype,py::arg("requirements")=py::make_tuple("C","A")).cast<py::array>();
        py::array_t<std::uint8_t> result(std::vector<py::ssize_t>(image.shape(),image.shape()+image.ndim()));
        const auto n=std::size_t(input.size());const auto kind=dtype.kind();const auto width=dtype.itemsize();
        // Dtype dispatch copies a Python dtype reference: finish it before
        // releasing the GIL, even though the subsequent comparison is native.
        const auto type=(kind=='f' || kind=='c' || kind=='m' || kind=='M')
            ? skimagexx::PixelType::int64:skimagexx::bindings::real_pixel_type(dtype);
        bool invalid=false;
        {
            py::gil_scoped_release release;auto* out=result.mutable_data();std::size_t consumed=0;
            if(kind=='b' || (kind=='u' && width==1)) {
#ifdef SKIMAGEXX_AVX2
                if(skimagexx::has_avx2())consumed=skimagexx::simd::rank_positive(input.data(),out,n,skimagexx::PixelType::uint8);
#endif
            }
            if(kind=='c') {
                for(std::size_t i=0;i<n;++i) {
                    if(width==8){const auto* p=static_cast<const float*>(input.data())+2*i;invalid|=std::isnan(p[0]) || std::isnan(p[1]);out[i]=(!std::isnan(p[0]) && !std::isnan(p[1]) && (p[0]>0 || (p[0]==0 && p[1]>0))) ? 255:0;}
                    else if(width>16){const auto* p=static_cast<const long double*>(input.data())+2*i;invalid|=std::isnan(p[0]) || std::isnan(p[1]);out[i]=(!std::isnan(p[0]) && !std::isnan(p[1]) && (p[0]>0 || (p[0]==0 && p[1]>0))) ? 255:0;}
                    else{const auto* p=static_cast<const double*>(input.data())+2*i;invalid|=std::isnan(p[0]) || std::isnan(p[1]);out[i]=(!std::isnan(p[0]) && !std::isnan(p[1]) && (p[0]>0 || (p[0]==0 && p[1]>0))) ? 255:0;}
                }
            } else if(kind=='f') {
                for(std::size_t i=0;i<n;++i) {
                    if(width>8){const auto value=static_cast<const long double*>(input.data())[i];out[i]=!std::isnan(value) && value>0 ? 255:0;continue;}
                    double value;
                    if(width==2){const auto word=static_cast<const std::uint16_t*>(input.data())[i];value=skimagexx::detail::decode_half(word);invalid|=(word&0x7c00)==0x7c00 && (word&0x03ff) && !(word&0x0200);}
                    else if(width==4){const float f=static_cast<const float*>(input.data())[i];std::uint32_t word;std::memcpy(&word,&f,4);invalid|=(word&0x7f800000)==0x7f800000 && (word&0x007fffff) && !(word&0x00400000);value=f;}
                    else{value=static_cast<const double*>(input.data())[i];std::uint64_t word;std::memcpy(&word,&value,8);invalid|=(word&0x7ff0000000000000ULL)==0x7ff0000000000000ULL && (word&0x000fffffffffffffULL) && !(word&0x0008000000000000ULL);}
                    out[i]=!std::isnan(value) && value>0 ? 255:0;
                }
            } else {
                switch(type) {
                    #define POS(tag,T) case skimagexx::PixelType::tag:positive<T>(input.data(),out,consumed,n);break;
                    POS(boolean,std::uint8_t) POS(uint8,std::uint8_t) POS(int8,std::int8_t)
                    POS(uint16,std::uint16_t) POS(int16,std::int16_t) POS(uint32,std::uint32_t) POS(int32,std::int32_t) POS(uint64,std::uint64_t) POS(int64,std::int64_t)
                    #undef POS
                    default:break;
                }
            }
        }
        return py::make_tuple(result,invalid);
    });
    module.def("rank_bins",[](py::array image) {
        std::uint16_t maximum=0;
        {py::gil_scoped_release release;maximum=skimagexx::rank_max_u16(static_cast<const std::uint16_t*>(image.data()),std::size_t(image.size()));}
        return std::size_t(std::max(std::uint16_t{3},maximum))+1;
    });
    module.def("rank_shift",[](py::object value){return shift(value);});
    module.def("rank_run",[numpy](py::array image,py::array footprint,py::object mask_object,py::array output,std::vector<py::object> requested_shifts,std::size_t bins,int operation) {
        const int rank=int(requested_shifts.size());
        if((rank!=2 && rank!=3) || image.ndim()!=rank)throw py::type_error("Function call with ambiguous argument types");
        const auto dtype=output.dtype();const auto kind=dtype.kind();const auto width=dtype.itemsize();
        if(output.ndim()!=rank+1 || !((kind=='u' && (width==1 || width==2)) || kind=='b' || (kind=='f' && (width==4 || width==8))))throw py::type_error("Function call with ambiguous argument types");
        buffer(image,rank);buffer(footprint,rank);
        py::array mask;if(!mask_object.is_none()){mask=mask_object.cast<py::array>();buffer(mask,rank);}
        buffer(output,rank+1);
        std::vector<int> shifts;for(const auto& value:requested_shifts)shifts.push_back(shift(value));
        skimagexx::RankView view;view.uint16=image.itemsize()==2;view.image=image.data();view.footprint=static_cast<const std::uint8_t*>(footprint.data());view.output=output.mutable_data();view.bins=bins;
        view.output_type=skimagexx::bindings::real_pixel_type(dtype);
        const int skip=rank==2 ? 1:0;view.shape[0]=view.footprint_shape[0]=1;
        for(int axis=0;axis<rank;++axis) {
            view.shape[axis+skip]=std::size_t(image.shape(axis));view.footprint_shape[axis+skip]=std::size_t(footprint.shape(axis));view.output_strides[axis+skip]=output.strides(axis);
            const auto shift=rank==2 ? shifts[1-axis]:shifts[axis];
            view.center[axis+skip]=footprint.shape(axis)/2+shift;
        }
        if(rank==2) {
            const auto cy=view.center[1],cx=view.center[2];const auto rows=view.footprint_shape[1],cols=view.footprint_shape[2];
            std::string error;if(cy<0)error="centre_r "+std::to_string(cy)+" < 0";else if(cx<0)error="centre_c "+std::to_string(cx)+" < 0";else if(std::size_t(cy)>=rows)error="centre_r "+std::to_string(cy)+" >= srows "+std::to_string(rows);else if(std::size_t(cx)>=cols)error="centre_c "+std::to_string(cx)+" >= scols "+std::to_string(cols);
            if(!error.empty()){PyErr_SetString(PyExc_AssertionError,error.c_str());throw py::error_already_set();}
        } else for(int axis=0;axis<3;++axis)if(view.center[axis]<0 || std::size_t(view.center[axis])>=view.footprint_shape[axis])throw py::value_error(std::string("half footprint + shift_")+char('x'+axis)+" must be between 0 and footprint");
        // These invalid buffers are undefined upstream: reject rather than
        // reproduce unchecked Cython reads/writes outside an allocation.
        if(output.shape(rank)<1)throw py::value_error("rank output pixel depth must be positive");
        for(int axis=0;axis<rank;++axis)if(output.shape(axis)<image.shape(axis))throw py::value_error("rank output is smaller than image");
        if(!mask_object.is_none()) {
            if(mask.size()<image.size())throw py::value_error("rank mask storage is smaller than image");
            view.mask=static_cast<const std::uint8_t*>(mask.data());
        }
        view.serial_alias=numpy.attr("shares_memory")(image,output).cast<bool>() || (!mask_object.is_none() && numpy.attr("shares_memory")(mask,output).cast<bool>());
        {py::gil_scoped_release release;skimagexx::rank_filter_into(view,static_cast<skimagexx::RankOperation>(operation));}
    });
}
