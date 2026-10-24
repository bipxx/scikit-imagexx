#include "skimagexx/matrix_transform.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <atomic>
namespace py=pybind11;
namespace {
std::atomic<std::uint64_t> completed{0};
bool owner(const py::array& a,skimagexx::MatrixReadView& v) {
    py::array root=a;
    for(int depth=0;depth<64;++depth) {
        if(root.owndata()) {
            if(!(root.flags()&(py::array::c_style|py::array::f_style)))return false;
            v={a.data(),std::size_t(a.shape(0)),std::size_t(a.shape(1)),
                a.strides(0),a.strides(1),root.data(),std::size_t(root.nbytes())};
            return true;
        }
        const auto base=root.base();
        if(py::isinstance<py::array>(base)) {
            root=py::reinterpret_borrow<py::array>(base);
        }else if(PyMemoryView_Check(base.ptr())) {
            // C-level memoryview owner only. Do not call .obj or arbitrary
            // buffer providers; recurse only through an actual ndarray base.
            auto* object=PyMemoryView_GET_BASE(base.ptr());
            if(!object || !py::isinstance<py::array>(py::handle(object)))return false;
            root=py::reinterpret_borrow<py::array>(object);
        }else return false;
    }
    return false;
}
py::tuple run(py::array points,py::array matrix,unsigned profile) {
    for(const auto& a:{points,matrix}) {
        if(a.dtype().kind()!='f' || a.itemsize()!=8 || !a.dtype().attr("isnative").cast<bool>())
            throw py::type_error("matrix transform requires native float64 arrays");
        if(a.ndim()!=2)throw py::value_error("matrix transform requires two-dimensional arrays");
    }
    skimagexx::MatrixReadView p{},m{};
    if(!owner(points,p) || !owner(matrix,m))
        return py::make_tuple(py::none(),unsigned(skimagexx::MatrixMapStatus::invalid_view));
    // Allocate only after all Python metadata is acquired. No NumPy APIs run
    // under released GIL, and no convenience conversion changes profile layout.
    py::array_t<double> output({points.shape(0),points.shape(1)});
    const skimagexx::MatrixWriteView o{output.mutable_data(),p.rows,p.columns,std::size_t(output.nbytes())};
    skimagexx::MatrixMapStatus status;
    {py::gil_scoped_release release;
        status=skimagexx::matrix_transform_into(p,m,o,skimagexx::MatrixArithmeticProfile(profile));}
    if(status!=skimagexx::MatrixMapStatus::ok)return py::make_tuple(py::none(),unsigned(status));
    completed.fetch_add(1,std::memory_order_relaxed);
    return py::make_tuple(output,0u);
}
}
void bind_matrix_transform(py::module_& m) {
    m.attr("matrix_transform_interface")="matrix-transform-math-v1";
    m.attr("matrix_transform_capabilities")=31;
    m.def("matrix_transform_run",&run,py::arg("points").noconvert(),py::arg("matrix").noconvert(),py::arg("profile"));
    m.def("matrix_transform_completed",[]{return completed.load(std::memory_order_relaxed);});
}
