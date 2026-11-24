#include <skimagexx/gray_texture.hpp>
#include <pybind11/pybind11.h>
namespace py=pybind11;
void bind_gray_texture(py::module_&);
PYBIND11_MODULE(_core,m){m.def("get_num_threads",&skimagexx::get_num_threads);m.def("set_num_threads",&skimagexx::set_num_threads);m.def("has_avx2",&skimagexx::has_avx2);bind_gray_texture(m);}
