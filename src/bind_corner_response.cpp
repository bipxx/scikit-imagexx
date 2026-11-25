#include "skimagexx/corner_response.hpp"
#include "bind_helpers.hpp"
#include <array>
#include <algorithm>
#include <utility>
namespace py=pybind11;
void bind_corner_response(py::module_& module){
    module.def("corner_response",[](py::list fields,int method,double parameter,bool wide)->py::object{
        if(fields.size()!=3 || method<0 || method>3)throw py::value_error("invalid corner response plan");
        auto first=py::array::ensure(fields[0]);
        if(!first || first.dtype().kind()!='f' || (first.itemsize()!=4 && first.itemsize()!=8))return py::none();
        const bool single=first.itemsize()==4;
        if(!single)wide=false;
        std::array<py::array,3> prepared;std::array<const void*,3> pointers;
        for(std::size_t f=0;f<3;++f){
            prepared[f]=py::array::ensure(fields[py::ssize_t(f)],py::array::c_style);
            if(!prepared[f] || prepared[f].ndim()!=first.ndim() || !prepared[f].dtype().equal(first.dtype()) || !prepared[f].dtype().attr("isnative").cast<bool>())return py::none();
            if(!std::equal(first.shape(),first.shape()+first.ndim(),prepared[f].shape()))return py::none();
            if(!prepared[f].attr("flags").attr("aligned").cast<bool>())prepared[f]=prepared[f].attr("copy")().cast<py::array>();
            pointers[f]=prepared[f].data();
        }
        const auto dtype=single && !wide ? py::dtype::of<float>() : py::dtype::of<double>();
        const auto rank = static_cast<int>(first.ndim());
        auto output = skimagexx::bindings::empty_c_array(dtype, first.shape(), rank);
        py::object second;
        void* extra = nullptr;
        if (method == 3) {
            auto array = skimagexx::bindings::empty_c_array(dtype, output.shape(), rank);
            extra = array.mutable_data();
            second = std::move(array);
        }
        bool valid;
        {py::gil_scoped_release release;valid=skimagexx::corner_response_into(pointers.data(),output.mutable_data(),extra,std::size_t(first.size()),single,static_cast<skimagexx::CornerResponse>(method),parameter,wide);}
        if(!valid)return py::none();
        if(method==3)return py::make_tuple(output,second);
        return output;
    });
}
