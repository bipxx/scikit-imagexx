#include "skimagexx/footprint_dense.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
namespace py=pybind11;

void bind_footprint_dense(py::module_& module) {
    module.def("footprint_rectangle_numeric_into",[](py::array output) {
        if(!(output.flags()&py::array::c_style) || !output.writeable())
            throw py::value_error("rectangle output must be writable C-contiguous");
        const auto dtype=output.dtype();const auto kind=dtype.kind();
        const auto itemsize=std::size_t(dtype.itemsize());
        std::array<unsigned char,16> one{};
        auto component=itemsize;
        if(kind=='b' && itemsize==1)one[0]=1;
        else if((kind=='i' || kind=='u') && (itemsize==1 || itemsize==2 || itemsize==4 || itemsize==8)) {
            switch(itemsize) {
                case 1:{const std::uint8_t value=1;std::memcpy(one.data(),&value,1);break;}
                case 2:{const std::uint16_t value=1;std::memcpy(one.data(),&value,2);break;}
                case 4:{const std::uint32_t value=1;std::memcpy(one.data(),&value,4);break;}
                default:{const std::uint64_t value=1;std::memcpy(one.data(),&value,8);break;}
            }
        } else if((kind=='f' && (itemsize==2 || itemsize==4 || itemsize==8)) ||
                  (kind=='c' && (itemsize==8 || itemsize==16))) {
            component=kind=='c' ? itemsize/2:itemsize;
            if(component==2){const std::uint16_t value=0x3c00;std::memcpy(one.data(),&value,2);}
            else if(component==4){const float value=1;std::memcpy(one.data(),&value,4);}
            else{const double value=1;std::memcpy(one.data(),&value,8);}
        } else throw py::type_error("ordinary numeric rectangle dtype required");
        if(!dtype.attr("isnative").cast<bool>()) {
            for(std::size_t offset=0;offset<itemsize;offset+=component)
                std::reverse(one.begin()+offset,one.begin()+offset+component);
        }
        // Atoms live on this call's stack, disjoint from the owned output.
        // No Python scalar conversions, dtype caches, or geometry changes.
        auto* data=output.mutable_data();const auto count=std::size_t(output.size());
        {py::gil_scoped_release release;
         skimagexx::footprint_dense_into(data,count,itemsize,one.data(),one.data(),
             nullptr,0,skimagexx::DenseFootprint::rectangle,0.);}
    });
    module.def("footprint_dense_into",[](py::array output,
        py::array_t<double,py::array::c_style> axis,int kind,double limit,
        py::array zero,py::array one) {
        if(kind<0 || kind>3)throw py::value_error("invalid dense footprint kind");
        if(!(output.flags()&py::array::c_style) || !output.writeable())
            throw py::value_error("dense footprint output must be writable C-contiguous");
        if(output.dtype().attr("hasobject").cast<bool>())
            throw py::type_error("dense footprint atoms cannot contain Python references");
        if(axis.ndim()!=1)throw py::value_error("dense footprint axis must be one-dimensional");
        for(const auto& atom:{zero,one})
            if(atom.size()!=1 || !atom.dtype().attr("__eq__")(output.dtype()).cast<bool>() || !(atom.flags()&py::array::c_style))
                throw py::value_error("dense footprint atoms must match output dtype");
        if(kind!=0) {
            const auto dimensions=kind==3 ? 3:2;
            if(output.ndim()!=dimensions)throw py::value_error("dense footprint output rank mismatch");
            for(py::ssize_t d=0;d<output.ndim();++d)
                if(output.shape(d)!=axis.size())throw py::value_error("dense footprint output shape mismatch");
        }
        // Reject overlapping buffers before releasing the GIL. No dtype may
        // expose Python references to the byte-writing C++ kernel.
        const auto np=py::module_::import("numpy");
        for(const auto& input:{py::array(axis),zero,one})
            if(np.attr("shares_memory")(output,input).cast<bool>())
                throw py::value_error("disjoint dense footprint buffers required");
        const auto itemsize=std::size_t(output.itemsize());
        py::gil_scoped_release release;
        skimagexx::footprint_dense_into(output.mutable_data(),std::size_t(output.size()),
            itemsize,zero.data(),one.data(),axis.data(),
            std::size_t(axis.size()),static_cast<skimagexx::DenseFootprint>(kind),limit);
    });
}
