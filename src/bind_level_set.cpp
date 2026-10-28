#include "level_set.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {
struct Grid {
    std::vector<py::ssize_t> shape;
    std::vector<std::size_t> dimensions;
    std::size_t count=1;
    explicit Grid(const py::tuple& supplied) {
        for(const auto value:supplied) {
            const auto size=py::cast<py::ssize_t>(value);
            if(size<0)throw py::value_error("negative dimensions are not allowed");
            shape.push_back(size);dimensions.push_back(std::size_t(size));
        }
        for(const auto size:dimensions)if(!size){count=0;break;}
        if(count)for(const auto size:dimensions) {
            if(size>std::size_t(std::numeric_limits<py::ssize_t>::max())/count)
                throw std::overflow_error("level-set shape exceeds addressable storage");
            count*=size;
        }
    }
    skimagexx::detail::LevelSetShape view()const{return {dimensions.data(),dimensions.size(),count};}
    py::array_t<std::int8_t> allocate()const {
        if(!count)return py::array_t<std::int8_t>(shape,std::vector<py::ssize_t>(shape.size(),0));
        return py::array_t<std::int8_t>(shape);
    }
};
py::object result(py::array_t<std::int8_t>& output,bool scalar) {
    if(scalar)return output.attr("__getitem__")(py::tuple());
    return output;
}
void workers_valid(int workers) {
    if(workers<1 || workers>1024)throw py::value_error("level-set worker snapshot outside [1,1024]");
}
}

void bind_level_set(py::module_& module) {
    module.def("level_set_checkerboard",[](const py::tuple& shape,std::int64_t square_size,int workers)->py::object {
        workers_valid(workers);Grid grid(shape);
        if(!square_size)throw py::value_error("native checkerboard requires nonzero square size");
        auto output=grid.allocate();auto* destination=output.mutable_data();
        {
            py::gil_scoped_release release;
            skimagexx::detail::checkerboard_level_set_into(grid.view(),square_size,destination,workers);
        }
        return result(output,grid.shape.empty());
    },py::arg("shape").noconvert(),py::arg("square_size"),py::arg("workers"));
    module.def("level_set_disk",[](const py::tuple& shape,const py::tuple& center,double radius,int workers)->py::object {
        workers_valid(workers);Grid grid(shape);
        if(center.size()!=1 && center.size()!=grid.dimensions.size())
            throw py::value_error("native disk requires one center coordinate per axis or one broadcast coordinate");
        std::vector<std::int64_t> coordinates(grid.dimensions.size());
        for(std::size_t axis=0;axis<coordinates.size();++axis)
            coordinates[axis]=py::cast<std::int64_t>(center[center.size()==1?0:axis]);
        if(!skimagexx::detail::disk_level_set_geometry_admissible(grid.view(),coordinates.data(),radius))return py::none();
        auto output=grid.allocate();auto* destination=output.mutable_data();bool accepted;
        {
            py::gil_scoped_release release;
            accepted=skimagexx::detail::disk_level_set_into(grid.view(),coordinates.data(),radius,destination,workers);
        }
        if(!accepted)return py::none();
        return result(output,grid.shape.empty());
    },py::arg("shape").noconvert(),py::arg("center").noconvert(),py::arg("radius"),py::arg("workers"));
}
