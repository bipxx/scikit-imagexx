#include <skimagexx/draw_coordinates.hpp>
#include "bind_array_storage.hpp"
#include "bind_helpers.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace py=pybind11;
namespace {
using namespace skimagexx::bindings;
using Int=std::intptr_t;
using UInt=std::uintptr_t;
constexpr auto limit=std::numeric_limits<py::ssize_t>::max();

void worker_snapshot(int workers) {
    if(workers<1||workers>1024)throw py::value_error("drawing worker snapshot outside [1,1024]");
}

py::array_t<Int> coordinates(const py::ssize_t* shape,int rank) {
    // NumPy empty grids have zero strides on every axis, not partial C strides.
    return empty_c_array<Int>(shape,rank,true);
}

bool integer_vector(const py::array& value,std::vector<Int>& output) {
    using Array=py::array_t<Int,py::array::c_style>;
    if(!Array::check_(value)||value.ndim()!=1
        ||!is_native_byteorder(value)
        ||reinterpret_cast<UInt>(value.data())%alignof(Int))return false;
    skimagexx::bindings::check_array_storage(value);
    output.clear();
    if(value.size()) {
        const auto* begin=static_cast<const Int*>(value.data());
        output.assign(begin,begin+value.size());
    }
    return true;
}

template<class T> bool real_pair_values(const py::array& value,std::array<double,2>& output) {
    const auto* bytes=static_cast<const unsigned char*>(value.data());
    for(std::size_t i=0;i<2;++i) {
        T item;std::memcpy(&item,bytes+i*sizeof(T),sizeof(T));
        if constexpr(std::is_integral_v<T>) {
            if constexpr(sizeof(T)>=8) {
                if constexpr(std::is_signed_v<T>) {
                    if(item<-T(9007199254740992LL)||item>T(9007199254740992LL))return false;
                } else if(item>T(9007199254740992ULL))return false;
            }
        } else {
            // Reject exceptional/subnormal conversion before any FP comparison.
            // This leaves literal-provider refusals free of new FP status flags.
            if constexpr(sizeof(T)==4) {
                std::uint32_t word;std::memcpy(&word,&item,4);
                const auto exponent=word&0x7f800000u,magnitude=word&0x7fffffffu;
                if(exponent==0x7f800000u||(!exponent&&magnitude))return false;
            } else {
                std::uint64_t word;std::memcpy(&word,&item,8);
                const auto exponent=word&0x7ff0000000000000ULL,magnitude=word&0x7fffffffffffffffULL;
                if(exponent==0x7ff0000000000000ULL||(!exponent&&magnitude))return false;
            }
        }
        output[i]=static_cast<double>(item);
    }
    return true;
}

bool real_pair(const py::array& value,std::array<double,2>& output) {
    if(value.ndim()!=1||value.size()!=2||!(value.flags()&py::array::c_style)
        ||!is_native_byteorder(value))return false;
    const auto kind=value.dtype().kind();const auto bytes=value.itemsize();
    if((kind!='i'&&kind!='u'&&kind!='f')||bytes<1||bytes>8)return false;
    skimagexx::bindings::check_array_storage(value);
    if(kind=='f'&&bytes==4)return real_pair_values<float>(value,output);
    if(kind=='f'&&bytes==8)return real_pair_values<double>(value,output);
    if(kind=='i') {
        if(bytes==1)return real_pair_values<std::int8_t>(value,output);
        if(bytes==2)return real_pair_values<std::int16_t>(value,output);
        if(bytes==4)return real_pair_values<std::int32_t>(value,output);
        if(bytes==8)return real_pair_values<std::int64_t>(value,output);
    }
    if(kind=='u') {
        if(bytes==1)return real_pair_values<std::uint8_t>(value,output);
        if(bytes==2)return real_pair_values<std::uint16_t>(value,output);
        if(bytes==4)return real_pair_values<std::uint32_t>(value,output);
        if(bytes==8)return real_pair_values<std::uint64_t>(value,output);
    }
    return false;
}

bool finite_geometry(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,8);
    return (bits&0x7ff0000000000000ULL)!=0x7ff0000000000000ULL;
}
}

void bind_draw_coordinates(py::module_& module) {
    module.def("draw_line",[](py::ssize_t r0,py::ssize_t c0,py::ssize_t r1,
                               py::ssize_t c1,int workers)->py::object {
        worker_snapshot(workers);
        const UInt dr=r1>=r0?UInt(r1)-UInt(r0):UInt(r0)-UInt(r1);
        const UInt dc=c1>=c0?UInt(c1)-UInt(c0):UInt(c0)-UInt(c1);
        if(dr>UInt(limit)/2||dc>UInt(limit)/2)return py::none();
        const UInt count=std::max(dr,dc)+1;
        if(count>UInt(limit)/sizeof(Int))return py::none();
        const py::ssize_t shape[]{py::ssize_t(count)};
        auto rows=coordinates(shape,1),columns=coordinates(shape,1);
        const skimagexx::detail::DrawCoordinatePair output{
            rows.mutable_data(),columns.mutable_data(),1,1};
        {
            optional_gil_release release(count>=4096);
            skimagexx::detail::draw_line_into(r0,c0,r1,c1,output,workers);
        }
        return py::make_tuple(rows,columns);
    },py::arg("r0"),py::arg("c0"),py::arg("r1"),py::arg("c1"),py::arg("workers"));

    module.def("draw_rectangle",[](const py::array& top_left,const py::array& bottom_right,
                                    int workers)->py::object {
        worker_snapshot(workers);std::vector<Int> starts,stops;
        if(!integer_vector(top_left,starts)||!integer_vector(bottom_right,stops)
            ||starts.size()!=stops.size())return py::none();
        const auto rank=starts.size();py::list result;if(!rank)return result;
        std::vector<std::size_t> lengths(rank);
        std::array<py::ssize_t,64> shape;
        std::size_t count=1;
        for(std::size_t i=0;i<rank;++i) {
            const UInt length=stops[i]<=starts[i]?0:UInt(stops[i])-UInt(starts[i]);
            if(length>UInt(limit))return py::none();
            lengths[i]=std::size_t(length);
            if(i<shape.size())shape[i]=py::ssize_t(length);
            if(length&&count>std::size_t(limit)/length)return py::none();
            count*=length;
        }
        if(count>std::size_t(limit)/sizeof(Int))return py::none();
        if(rank>=2)std::swap(shape[0],shape[1]);
        std::vector<py::array> owners;owners.reserve(rank);std::vector<Int*> outputs;outputs.reserve(rank);
        for(std::size_t i=0;i<rank;++i) {
            // NumPy rejects ranks above 64 before it reads the dimensions.
            auto output=coordinates(shape.data(),static_cast<int>(rank));
            outputs.push_back(output.mutable_data());owners.push_back(std::move(output));
        }
        {
            optional_gil_release release(count>=4096);
            skimagexx::detail::draw_rectangle_into(starts.data(),lengths.data(),rank,count,outputs.data(),workers);
        }
        for(const auto& output:owners)result.append(output);
        return result;
    },py::arg("top_left").noconvert(),py::arg("bottom_right").noconvert(),py::arg("workers"));

    module.def("draw_ellipse",[](const py::array& bounding_shape,const py::array& shifted_center,
        const py::array& radii,double sin_alpha,double cos_alpha,const py::array& upper_left,
        int workers)->py::object {
        worker_snapshot(workers);std::vector<Int> dimensions,origin;std::array<double,2> center,radius;
        if(!integer_vector(bounding_shape,dimensions)||dimensions.size()!=2
            ||!integer_vector(upper_left,origin)||origin.size()!=2
            ||!real_pair(shifted_center,center)||!real_pair(radii,radius)
            ||!finite_geometry(sin_alpha)||!finite_geometry(cos_alpha)||radius[0]<=0||radius[1]<=0)return py::none();
        const auto rows=std::size_t(std::max<Int>(dimensions[0],0));
        const auto columns=std::size_t(std::max<Int>(dimensions[1],0));
        if(rows>9007199254740992ULL||columns>9007199254740992ULL
            ||(columns&&rows>std::size_t(limit)/columns)
            ||rows>=std::size_t(limit)/sizeof(std::size_t))return py::none();
        if((rows&&origin[0]>limit-Int(rows-1))||(columns&&origin[1]>limit-Int(columns-1)))return py::none();
        const skimagexx::detail::DrawEllipseGrid grid{rows,columns,center[0],center[1],radius[0],radius[1],
            sin_alpha,cos_alpha,origin[0],origin[1]};
        std::vector<std::uint8_t> mask(rows*columns);std::vector<std::size_t> prefix(rows+1);
        bool accepted;
        {
            optional_gil_release release(mask.size()>=4096);
            accepted=skimagexx::detail::draw_ellipse_mask_into(grid,mask.data(),prefix.data(),workers);
        }
        if(!accepted)return py::none();
        const auto count=prefix[rows];if(count>std::size_t(limit)/(2*sizeof(Int)))return py::none();
        const py::ssize_t shape[]{py::ssize_t(count),2};
        auto output=coordinates(shape,2);
        auto* values=output.mutable_data();
        // Empty columns use the same pointer: there are no addresses to emit.
        const skimagexx::detail::DrawCoordinatePair pair{values,count?values+1:values,2,2};
        {
            optional_gil_release release(mask.size()>=4096);
            skimagexx::detail::draw_ellipse_emit_into(grid,mask.data(),prefix.data(),pair,workers);
        }
        return output;
    },py::arg("bounding_shape").noconvert(),py::arg("shifted_center").noconvert(),
      py::arg("radii").noconvert(),py::arg("sin_alpha"),py::arg("cos_alpha"),
      py::arg("upper_left").noconvert(),py::arg("workers"));
}
